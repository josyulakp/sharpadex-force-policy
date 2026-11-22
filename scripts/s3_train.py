"""Stage 3: train bounded actor-critic corrections around the frozen four-expert policy in the simulator.

    python scripts/s3_train.py [name=rl] [learner.key=value ...]      -> runs/stage3/<name>/{ckpt.pt, log.jsonl, config.yaml}

Needs scripts/s3_prepare.py and scripts/s3_scenes.py first. Simulator processes run whole episodes with the latest actor
and send the transitions back; the learner follows Facet-0's online adaptation with the robot replaced by the simulator:
  seed     the frozen policy plus exploration noise fills the replay; fixed normalisation is taken from it
  critic   twin critics train on applied corrections; the actor is held until the critic passes its gate on monitor episodes
  actor    while the gate is open: -Q (scaled) + beta * imitation of the corrections of successful episodes (TD3+BC)
`learner.actor_q_weight=0` is the matched control: the same schedule and imitation term without the Q gradient.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sharpa.stage3 import Actor, Critic, limit  # noqa: E402


def load_cfg(args: list[str]) -> tuple[dict, dict, str]:
    s3 = yaml.safe_load((REPO / "configs" / "stage3.yaml").read_text())
    sim = yaml.safe_load((REPO / "configs" / "sim.yaml").read_text())
    name = "rl"
    for a in args:
        k, v = a.split("=", 1)
        if k == "name":
            name = v
            continue
        node = s3
        *path, last = k.split(".")
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(v)
    return s3, sim, name


def scenes(s3: dict) -> dict:
    """Built scenes by split, each {uid, objects, stream, frames}."""
    rows = json.loads((Path(s3["data_dir"]) / "scenes.json").read_text())
    out = {"train": [], "heldout": []}
    for r in rows:
        if "error" not in r and r["frames"] <= s3["env"]["max_frames"]:
            out[r["split"]].append({"uid": r["uid"], "objects": r["objects"], "frames": r["frames"],
                                    "stream": str(Path(s3["data_dir"]) / "streams" / (r["uid"].replace("/", "__") + ".npz"))})
    n = s3["learner"].get("validation_scenes", 0)                         # training scenes set aside: never trained in, used to choose
    pick = set(np.random.default_rng(0).permutation(len(out["train"]))[:n].tolist())
    out["validation"] = [s for i, s in enumerate(out["train"]) if i in pick]
    out["train"] = [s for i, s in enumerate(out["train"]) if i not in pick]
    return out


def token_dim(s3: dict) -> int:
    e = s3["env"]
    return 256 if ("token" in e["actor_blocks"] if "actor_blocks" in e else e.get("actor_token", True)) else 0


def make_actor(s3: dict, dims: tuple) -> Actor:
    return Actor(dims[0], dims[2], s3["learner"]["token_proj"], s3["learner"]["hidden"], token_dim(s3))


def synergies(s3: dict, train: list[dict]) -> dict:
    from sim.env import fit_synergies
    hand = np.r_[7:29, 36:58]
    cmds = np.concatenate([np.load(s["stream"])["targets__recorded"][::3, hand] for s in train])
    return fit_synergies(cmds.astype(np.float64), s3["env"]["synergies"])


def run_episode(env, actor, task: dict) -> dict:
    """One episode. task: scene, seed, noise (std, corr), perturb (bool), zero (bool: no actor, correction = noise only)."""
    rng = np.random.default_rng(task["seed"])
    oa, oc = env.reset(task["scene"]["stream"], task["scene"]["objects"], rng if task["perturb"] else None, task.get("stream", "base"))
    std, corr = task["noise"]
    corr = float(rng.choice(corr)) if isinstance(corr, list) else corr     # one correlation per episode: jitter, drift or a near-constant offset
    n = rng.standard_normal(env.act_dim)
    rows = {k: [] for k in ("obs", "act", "rew", "done")}
    while True:
        with torch.no_grad():
            a = np.zeros(env.act_dim) if task["zero"] else actor(torch.from_numpy(oa)[None]).numpy()[0].astype(np.float64)
        if std > 0:
            n = corr * n + np.sqrt(1 - corr ** 2) * rng.standard_normal(env.act_dim)
            a = a + std * n
        rows["obs"].append(oc)
        oa, oc, r, done, info = env.step(a)
        rows["act"].append(info["applied"]); rows["rew"].append(r); rows["done"].append(done)
        if done:
            break
    rows["obs"].append(oc)                                                # the observation after the last step
    return {"obs": np.stack(rows["obs"]).astype(np.float32), "act": np.stack(rows["act"]).astype(np.float32),
            "rew": np.asarray(rows["rew"], np.float32), "outcome": info["outcome"], "frames": info["frames"],
            "uid": task["scene"]["uid"], "kind": task["kind"], "index": task["index"], "seed": task["seed"]}


def worker(tasks, results, s3: dict, sim: dict, synergy: dict, dims: tuple):
    torch.set_num_threads(1)
    from sim.env import ResidualEnv
    env = ResidualEnv(sim, s3, synergy)
    actor = make_actor(s3, dims)
    while True:
        task = tasks.get()
        if task is None:
            break
        try:
            if task["actor"] is not None:
                actor.load_state_dict(task["actor"])
            results.put(run_episode(env, actor, task))
        except Exception as e:                                            # a scene that fails to build or a diverged simulation
            results.put({"error": f"{type(e).__name__}: {e}", "uid": task["scene"]["uid"], "kind": task["kind"], "index": task["index"]})


class Replay:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        z = lambda *s: torch.zeros(*s, device="cuda")  # noqa: E731
        self.obs, self.obs2, self.act = z(capacity, obs_dim), z(capacity, obs_dim), z(capacity, act_dim)
        self.rew, self.done, self.ret, self.ok = z(capacity), z(capacity), z(capacity), torch.zeros(capacity, dtype=torch.bool, device="cuda")
        self.n, self.ptr, self.capacity = 0, 0, capacity

    def add(self, ep: dict, gamma: float):
        T = len(ep["act"])
        ret, acc = np.zeros(T, np.float32), 0.0
        for t in range(T - 1, -1, -1):
            acc = ep["rew"][t] + gamma * acc
            ret[t] = acc
        idx = (self.ptr + torch.arange(T, device="cuda")) % self.capacity
        obs = torch.from_numpy(ep["obs"]).cuda()
        self.obs[idx], self.obs2[idx], self.act[idx] = obs[:-1], obs[1:], torch.from_numpy(ep["act"]).cuda()
        self.rew[idx], self.ret[idx] = torch.from_numpy(ep["rew"]).cuda(), torch.from_numpy(ret).cuda()
        done = torch.zeros(T, device="cuda")
        done[-1] = 1.0
        self.done[idx], self.ok[idx] = done, bool(ep["outcome"]["success"])
        self.ptr, self.n = (self.ptr + T) % self.capacity, min(self.n + T, self.capacity)

    def sample(self, batch: int, only_ok: bool = False):
        if only_ok:
            pool = torch.nonzero(self.ok[:self.n]).squeeze(1)
            i = pool[torch.randint(len(pool), (batch,), device="cuda")]
        else:
            i = torch.randint(self.n, (batch,), device="cuda")
        return self.obs[i], self.act[i], self.rew[i], self.obs2[i], self.done[i], self.ret[i]


def main():
    s3, sim, name = load_cfg(sys.argv[1:])
    L = s3["learner"]
    out = REPO / "runs" / "stage3" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"stage3": s3, "sim": sim}, sort_keys=False))
    sc = scenes(s3)
    syn = synergies(s3, sc["train"])
    np.savez(out / "synergy.npz", **syn)
    from sim.env import ResidualEnv
    env = ResidualEnv(sim, s3, syn)
    oa, oc = env.reset(sc["train"][0]["stream"], sc["train"][0]["objects"])
    Da, Dc, A = len(oa), len(oc), env.act_dim
    rate = torch.from_numpy(env.rate).float().cuda()
    prev = slice(Da - 1 - A, Da - 1)                                      # the last executed correction inside the observation
    K = token_dim(s3)
    print(f"{name}: {len(sc['train'])} training scenes, {len(sc['validation'])} validation, {len(sc['heldout'])} held-out; "
          f"actor obs {Da}, critic obs {Dc}, action {A}", flush=True)

    torch.manual_seed(L.get("seed", 0))
    actor, critic = make_actor(s3, (Da, Dc, A)).cuda(), Critic(Dc, A, L["token_proj"], L["hidden"], K).cuda()
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    opt_a, opt_c = torch.optim.Adam(actor.parameters(), L["actor_lr"]), torch.optim.Adam(critic.parameters(), L["critic_lr"])
    replay, monitor = Replay(L["capacity"], Dc, A), Replay(60000, Dc, A)
    cpu_actor = lambda: {k: v.detach().cpu() for k, v in actor.state_dict().items()}  # noqa: E731

    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(tasks, results, s3, sim, syn, (Da, Dc, A)), daemon=True) for _ in range(L["workers"])]
    [p.start() for p in procs]
    rng = np.random.default_rng(L.get("seed", 0))                         # scenes, perturbations and exploration noise of the run
    st = {"sent": 0, "updates": 0, "open_updates": 0, "streak": 0, "gate": False, "normalised": False, "actor_updates": 0}

    def submit():
        k = st["sent"]
        mon = k % L["monitor_every"] == L["monitor_every"] - 1
        val = mon and len(sc["validation"]) > 0 and (k // L["monitor_every"]) % 2 == 1      # every other monitor episode: a scene never trained in
        seed_phase = not st["normalised"]
        pool = sc["validation"] if val else sc["train"]
        tasks.put({"scene": pool[rng.integers(len(pool))], "seed": int(rng.integers(2 ** 31)), "perturb": True,
                   "noise": (0.0, 0.0) if mon else (L["noise_std"], L["noise_corr"]), "zero": seed_phase,
                   "actor": None if seed_phase else cpu_actor(),
                   "kind": "validation" if val else "monitor" if mon else ("seed" if seed_phase else "train"), "index": k})
        st["sent"] += 1

    def critic_update():
        o, a, r, o2, d, G = replay.sample(L["batch"])
        with torch.no_grad():
            a2 = actor_t(o2[:, :Da]) + (torch.randn_like(a) * L["target_noise"]).clamp(-L["target_noise_clip"], L["target_noise_clip"])
            a2 = limit(a2.clamp(-1, 1), a, rate)                          # the next correction is rate-limited around this one
            y = r + L["gamma"] * (1 - d) * torch.minimum(*critic_t(o2, a2))
            y = G if L["critic_target"] == "mc" else torch.maximum(y, G) if L["critic_target"] == "td_mc_max" else y
        q1, q2 = critic(o, a)
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_c.zero_grad(set_to_none=True); loss.backward(); opt_c.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_t.parameters()):
                pt.lerp_(p, L["tau"])
        return loss.item(), q1.mean().item()

    def actor_update():
        o = replay.sample(L["batch"])[0]
        beta = L["bc_beta"] + (L["bc_beta_end"] - L["bc_beta"]) * min(st["open_updates"] / L["bc_anneal_updates"], 1.0)
        loss, info = 0.0, {"beta": beta}
        if L["actor_q_weight"] > 0:
            a = actor(o[:, :Da])
            q = critic(o, limit(a, o[:, prev], rate))[0]
            loss = loss - L["actor_q_weight"] * L["q_scale"] * q.mean() / q.abs().mean().detach().clamp(min=1e-3) + L["action_penalty"] * a.pow(2).mean()
        if replay.ok[:replay.n].any():                                    # imitate the corrections of successful episodes
            od, ad = replay.sample(L["batch"], only_ok=True)[:2]
            bc = F.mse_loss(limit(actor(od[:, :Da]), od[:, prev], rate), ad)
            loss, info["bc"] = loss + beta * bc, bc.item()
        if torch.is_tensor(loss):
            opt_a.zero_grad(set_to_none=True); loss.backward(); opt_a.step()
            with torch.no_grad():
                for p, pt in zip(actor.parameters(), actor_t.parameters()):
                    pt.lerp_(p, L["tau"])
            st["actor_updates"] += 1
        return info

    def gate_check() -> dict:
        g = L["gate"]
        with torch.no_grad():
            n = monitor.n
            q1, q2 = critic(monitor.obs[:n], monitor.act[:n])
            G = monitor.ret[:n]
            r2 = 1 - ((torch.minimum(q1, q2) - G) ** 2).mean().item() / G.var().item()   # share of the variance of the returns it explains
            dis = (q1 - q2).abs().mean().item()
        ok = st["updates"] >= g["min_updates"] and r2 >= g["min_explained"] and dis <= g["max_disagreement"]
        st["streak"] = st["streak"] + 1 if ok else 0
        st["gate"] = st["streak"] >= g["passes"]
        return {"gate_explained": r2, "gate_disagreement": dis, "gate_open": st["gate"]}

    log = (out / "log.jsonl").open("a")
    for _ in range(L["workers"]):
        submit()
    done, t0, recent, recent_val, owed, last = 0, time.time(), [], [], 0.0, {}
    while done < L["episodes"]:
        ep = results.get()
        done += 1
        if st["sent"] < L["episodes"]:
            submit()
        if "error" in ep:
            log.write(json.dumps(ep) + "\n")
            continue
        row = {"index": ep["index"], "kind": ep["kind"], "uid": ep["uid"], "return": float(ep["rew"].sum()), "frames": ep["frames"],
               "mean_abs_correction": float(np.abs(ep["act"]).mean()), **ep["outcome"]}
        if ep["kind"] == "validation":                                    # only watched: not in the replay, not in the gate
            recent_val = (recent_val + [(row["success"], row["return"])])[-300:]
        else:
            (monitor if ep["kind"] == "monitor" else replay).add(ep, L["gamma"])
        if ep["kind"] == "monitor":
            recent = (recent + [(row["success"], row["return"], row["mean_abs_correction"])])[-300:]
        if not st["normalised"] and done >= L["seed_episodes"]:           # fixed normalisation from the seed data
            x = replay.obs[:replay.n, K:]
            mean, std = x.mean(0), x.std(0)
            for f, d in ((actor.f, Da), (actor_t.f, Da), (critic.f1, Dc), (critic.f2, Dc), (critic_t.f1, Dc), (critic_t.f2, Dc)):
                f.set_stats(mean[:d - K], std[:d - K])
            st["normalised"] = True
        if st["normalised"] and ep["kind"] not in ("monitor", "validation"):
            owed += L["updates_per_transition"] * len(ep["act"])
            while owed >= 1:
                owed -= 1
                closs, qmean = critic_update()
                st["updates"] += 1
                last.update(critic_loss=closs, q_mean=qmean)
                if st["updates"] % L["gate"]["eval_every"] == 0 and monitor.n > 0:
                    last.update(gate_check())
                if st["gate"]:
                    st["open_updates"] += 1
                    if st["updates"] % L["policy_delay"] == 0:
                        last.update(actor_update())
        log.write(json.dumps(row) + "\n")
        if done % 50 == 0:
            log.flush()
            sr = np.mean([r[0] for r in recent]) if recent else float("nan")
            rt = np.mean([r[1] for r in recent]) if recent else float("nan")
            corr = np.mean([r[2] for r in recent[-60:]]) if recent else float("nan")
            vs = f"validation success {np.mean([r[0] for r in recent_val]):.3f} return {np.mean([r[1] for r in recent_val]):.3f} (last {len(recent_val)})  " if recent_val else ""
            print(f"{done:6d} ep  {time.time() - t0:6.0f}s  {vs}monitor success {sr:.3f} return {rt:.3f} (last {len(recent)}) |correction| {corr:.2f}  updates {st['updates']} "
                  f"actor {st['actor_updates']}  replay {replay.n}  " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in last.items()), flush=True)
        if done % L["checkpoint_every"] == 0 or done == L["episodes"]:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "dims": (Da, Dc, A), "state": st, "episodes": done},
                       out / ("ckpt.pt" if done == L["episodes"] else f"ckpt_{done}.pt"))
    [tasks.put(None) for _ in procs]
    print("finished", name, flush=True)


if __name__ == "__main__":
    main()
