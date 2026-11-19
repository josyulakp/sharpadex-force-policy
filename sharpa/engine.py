"""Training loop, checkpoint I/O and held-out evaluation."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import REPO_ROOT, load_config
from .data.dataset import FrameData, joint_dims, skill_vocab
from .data.split import build_splits
from .data.mpc import MPCData
from .model import ForcePolicy, loss_fn, mpc_loss_fn


def run_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg["train"]["out_root"] / cfg["train"]["name"] / f"seed{cfg['train']['seed']}"


def build_model(cfg: dict, stats: dict, vocab: list[str], vis_shape) -> ForcePolicy:
    return ForcePolicy(cfg, stats, joint_dims(cfg), len(vocab), len(cfg["data"]["tasks"]),
                       vis_tokens=vis_shape[-2], vis_dim=vis_shape[-1])


def train(cfg: dict) -> Path:
    t, out = cfg["train"], run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    vocab, tasks = skill_vocab(cfg), cfg["data"]["tasks"]
    splits = build_splits(cfg)
    train_refs = [r for task in tasks for r in splits[task]["train"]]
    data = FrameData(cfg, train_refs, vocab, tasks)
    stats = data.stats()
    mpc = MPCData(cfg, "train") if cfg.get("mpc", {}).get("enabled") else None
    if mpc:
        stats |= mpc.stats()
    model = build_model(cfg, stats, vocab, data.feats.shape).cuda()
    if t.get("init_from"):                                               # upcycling: start from a trained run
        src = torch.load(REPO_ROOT / t["init_from"].format(seed=t["seed"]) / "ckpt.pt", map_location="cuda", weights_only=False)["model"]
        own = model.state_dict()
        keep = {k: v for k, v in src.items() if k in own and v.shape == own[k].shape and not (t.get("init_exclude", "experts.1") and k.startswith(t.get("init_exclude", "experts.1")))}
        model.load_state_dict(keep, strict=False)
        print(f"initialised {len(keep)}/{len(own)} tensors from {t['init_from'].format(seed=t['seed'])}")
    if model.ht:                                                         # HT-Policy: sigma starts at the untrained model's residual size
        with torch.no_grad():
            model.eval()
            g = torch.Generator().manual_seed(t["seed"])
            b = data.batch(torch.randint(len(data), (t["ht"]["calibration_frames"],), generator=g))
            o = model(b)
            av = b["action_valid"][..., None].float()
            rms = ((((o["delta_norm"].float() - model.targets(b, o)["delta_norm"]) ** 2) * av).sum() / (av.sum() * model.J)).sqrt().item()
            x = rms - model.ht_min_scale
            model.ht_scale_bias.fill_(x + math.log(-math.expm1(-x)))     # inverse softplus
            print(f"HT loss: starting residual RMS {rms:.4f} (normalised), scale bias {model.ht_scale_bias.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    hours = len(data) / cfg["data"]["fps"] / 3600
    print(f"{out.name}: {len(train_refs)} train episodes, {len(data)} frames ({hours:.2f} h), {n_params / 1e6:.2f} M params")

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / t["warmup_steps"]) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / t["steps"]))))
    gen = torch.Generator().manual_seed(t["seed"])
    log, t0 = [], time.time()
    model.train()
    for step in range(t["steps"]):
        b = data.batch(torch.randint(len(data), (t["batch_size"],), generator=gen))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t["amp"]):
            o = model(b)
        loss, logs = loss_fn(model, o, b, cfg)
        if mpc:                                                          # one MPC batch next to every robot batch
            bm = mpc.batch(torch.randint(len(mpc), (cfg["mpc"]["batch_size"],), generator=gen))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t["amp"]):
                om = model.forward_mpc(bm)
            l_mpc, mlogs = mpc_loss_fn(model, om, bm, cfg)
            loss, logs = loss + cfg["mpc"]["weight"] * l_mpc, logs | mlogs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        sched.step()
        if step + 1 in (t.get("eval_at") or []):                            # intermediate checkpoint, evaluated after training
            torch.save({"model": model.state_dict(), "cfg": cfg, "stats": stats, "vocab": vocab,
                        "vis_shape": tuple(data.feats.shape), "train_hours": len(data) / cfg["data"]["fps"] / 3600,
                        "train_episodes": len(train_refs)}, out / f"ckpt_step{step + 1}.pt")
        if step % t["log_every"] == 0 or step == t["steps"] - 1:
            logs |= {"step": step, "loss": loss.item(), "gate": o["gate"].float().mean(0).tolist()}
            log.append(logs)
            print(f"  step {step:5d} loss {loss.item():.4f} " + " ".join(f"{k} {v:.4f}" for k, v in logs.items()
                  if k in ("action", "force", "decision", "mpc_u", "mpc_force")) + f"  {time.time() - t0:.0f}s", flush=True)

    torch.save({"model": model.state_dict(), "cfg": cfg, "stats": stats, "vocab": vocab,
                "vis_shape": tuple(data.feats.shape), "train_hours": hours,
                "train_episodes": len(train_refs)}, out / "ckpt.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "train_log.json").write_text(json.dumps(log))
    del data
    return out


def load_checkpoint(run: Path, name: str = "ckpt.pt"):
    run = Path(run).resolve()
    if not (run / name).exists():                       # released checkpoints are on the Hugging Face Hub at the same relative path
        from huggingface_hub import hf_hub_download
        hf_hub_download(load_config()["release"]["hf_repo"], (run / name).relative_to(REPO_ROOT).as_posix(), local_dir=REPO_ROOT)
    ck = torch.load(run / name, map_location="cuda", weights_only=False)
    model = build_model(ck["cfg"], ck["stats"], ck["vocab"], ck["vis_shape"]).cuda()
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def evaluate(run: Path, eval_overrides: dict | None = None, tag: str = "heldout", ckpt: str = "ckpt.pt") -> dict:
    """Per-frame errors on every held-out frame of the run's tasks -> <run>/eval_<tag>.npz + .json."""
    model, ck = load_checkpoint(run, ckpt)
    cfg = ck["cfg"]
    ev = cfg["eval"] | (eval_overrides or {})
    model.flow_noise = float(ev.get("flow_noise_scale", 1.0))
    model.guidance = float(ev.get("guidance", 1.0))
    if model.energy:                                                     # test-time descent settings (default: as trained)
        model.er_steps, model.er_samples = int(ev.get("energy_steps", model.er_steps)), int(ev.get("energy_samples", model.er_samples))
        model.er_noise = float(ev.get("energy_noise", model.er_noise))
    tasks = cfg["data"]["tasks"]
    splits = build_splits(cfg)
    refs = [r for task in tasks for r in splits[task]["heldout"]]
    # force set to zero: the tactile expert gets the latents of the fingertip images frozen at a no-contact frame (tf_features.py rest=true)
    data = FrameData(cfg, refs, ck["vocab"], tasks, "latents_rest" if ev["zero_force"] in (True, "tactile") else "latents")
    arm = torch.tensor([i for i in range(model.J) if i < 7 or 29 <= i < 36], device="cuda")
    hand = torch.tensor([i for i in range(model.J) if 7 <= i < 29 or 36 <= i < 58], device="cuda")
    rows = {k: [] for k in ("act", "act_arm", "act_hand", "act_first", "act_hold", "force", "force_persist",
                            "contact", "ep", "task", "contact_prob", "contact_label", "phase_ok", "level_abs", "gate0")}
    for s in range(0, len(data), ev["batch_size"]):
        b = data.batch(torch.arange(s, min(s + ev["batch_size"], len(data))))
        if ev["swap_task"] is not None:
            b["task_id"] = torch.full_like(b["task_id"], tasks.index(ev["swap_task"]))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            o = model(b, zero_force=ev["zero_force"])
        p = model.to_physical(o, b)
        av, fv = b["action_valid"].float(), b["force_valid"].float()
        n_a, n_f = av.sum(1), fv.sum(1).clamp(min=1)

        def act_err(pred, dims=None):                                    # mean |error| in rad over valid steps, dims
            e = (pred - b["action"]).abs()
            e = e if dims is None else e[..., dims]
            return (e.mean(-1) * av).sum(1) / n_a

        rows["act"].append(act_err(p["action"]))
        rows.setdefault("act_sq", []).append((((p["action"] - b["action"]) ** 2).mean(-1) * av).sum(1) / n_a)   # rad^2, for the RMSE
        if "sigma" in o:                                                 # HT loss: predicted residual scale of the chunk (normalised units)
            rows.setdefault("sigma", []).append(o["sigma"].float())
        rows["act_arm"].append(act_err(p["action"], arm))
        rows["act_hand"].append(act_err(p["action"], hand))
        rows["act_first"].append((p["action"][:, 0] - b["action"][:, 0]).abs().mean(-1))
        rows["act_hold"].append(act_err(b["state"][:, -1:, :].expand_as(b["action"])))     # "stay where you are"
        rows["force"].append(((p["force"] - b["force"]).abs().mean(-1) * fv).sum(1) / n_f)   # N
        now = b["tactile"][:, -1].view(-1, 10, 6)[..., :3].flatten(1)[:, None]
        rows["force_persist"].append(((now - b["force"]).abs().mean(-1) * fv).sum(1) / n_f)  # "force stays as it is"
        rows["contact"].append(b["contact_now"])
        rows["ep"].append(b["ep_id"])
        rows["task"].append(b["task_id"])
        rows["contact_prob"].append(torch.sigmoid(o["contact_logit"].float()))
        rows["contact_label"].append(b["contact_soon"])
        rows["phase_ok"].append((o["phase_logits"].argmax(-1) == b["skill"]).float())
        level_pred = (o["level_logits"].float() > 0).sum(1)
        rows["level_abs"].append((level_pred - b["level"]).abs().float())
        rows["gate0"].append(o["gate"][:, 0].float())
        step = lambda a: ((a[:, 1:] - a[:, :-1]).abs().mean(-1) * av[:, 1:]).sum(1) / av[:, 1:].sum(1).clamp(min=1)  # noqa: E731
        rows.setdefault("speed", []).append(step(p["action"]))                 # commanded joint motion per frame (rad)
        rows.setdefault("speed_recorded", []).append(step(b["action"]))
        rows.setdefault("force_peak", []).append((p["force"].view(len(av), -1, 10, 3).norm(dim=-1).amax(-1) * fv).amax(1))  # N
        if "adv_label" in b:
            rows.setdefault("adv_label", []).append(b["adv_label"])
            rows.setdefault("adv_regime", []).append(b["adv_intensity"] >= 1.0)
        if model.wrench_expert and not model.flow:
            rows.setdefault("force_wrench_expert", []).append(((p["wrench_force"] - b["force"]).abs().mean(-1) * fv).sum(1) / n_f)
            rows.setdefault("gate_wrench", []).append(o["gate"][:, model.n_experts].float())
            pw = o["wrench_action_norm"].float() * model.delta_std + b["state"][:, -1:, :]
            rows.setdefault("act_wrench_expert", []).append(act_err(pw))
        if "energy" in o:                                                # EBR: final energy, and the same model before descent
            rows.setdefault("energy", []).append(o["energy"].float())
            if "start" in o:
                ps = model.to_physical(o["start"], b)
                rows.setdefault("act_start", []).append(act_err(ps["action"]))
                rows.setdefault("force_start", []).append(((ps["force"] - b["force"]).abs().mean(-1) * fv).sum(1) / n_f)
        if model.tactile_expert:
            rows.setdefault("gate_tactile", []).append(o["gate"][:, -1].float())
            pt = o["tactile_action_norm"].float() * model.delta_std + b["state"][:, -1:, :]
            rows.setdefault("act_tactile_expert", []).append(act_err(pt))
        if model.world_model:                                            # force forecast under other action inputs
            for mode in ("recorded", "zero", "shuffled"):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
                    om = model(b, zero_force=ev["zero_force"], wm_actions=mode)
                pm = model.to_physical(om, b)
                rows.setdefault(f"force_{mode}", []).append(((pm["force"] - b["force"]).abs().mean(-1) * fv).sum(1) / n_f)
    arrays = {k: torch.cat(v).float().cpu().numpy() for k, v in rows.items()}
    arrays["ep_uid"] = np.array([r.uid for r in refs])
    np.savez_compressed(run / f"eval_{tag}.npz", **arrays)

    c = arrays["contact"] > 0.5
    f = cfg["force"]
    summary = {"frames": int(len(c)), "contact_frames": int(c.sum()), "episodes": len(refs),
               "train_hours": ck["train_hours"], "force_output": f["force_output"],
               "decision_supervision": f["decision_supervision"]}
    for k in ("act", "act_arm", "act_hand", "act_first", "act_hold", "force", "force_persist",
              *(k for k in arrays if (k.startswith("force_") and k != "force_persist") or k in ("act_wrench_expert", "act_tactile_expert", "act_start"))):
        summary[k] = {"all": float(arrays[k].mean()), "contact": float(arrays[k][c].mean()),
                      "free": float(arrays[k][~c].mean())}
    summary["act_rmse"] = {"all": float(np.sqrt(arrays["act_sq"].mean())), "contact": float(np.sqrt(arrays["act_sq"][c].mean())),
                           "free": float(np.sqrt(arrays["act_sq"][~c].mean()))}
    summary["contact_soon_acc"] = float(((arrays["contact_prob"] > 0.5) == (arrays["contact_label"] > 0.5)).mean())
    summary["phase_acc"] = float(arrays["phase_ok"].mean())
    summary["level_mae"] = float(arrays["level_abs"].mean())
    for k in ("gate_wrench", "gate_tactile"):
        if k in arrays:
            summary[k + "_contact_vs_free"] = [float(arrays[k][c].mean()), float(arrays[k][~c].mean())]
    summary["gate0_contact_vs_free"] = [float(arrays["gate0"][c].mean()), float(arrays["gate0"][~c].mean())]
    (run / f"eval_{tag}.json").write_text(json.dumps(summary, indent=1))
    return summary


@torch.no_grad()
def evaluate_mpc(run: Path) -> dict:
    """MPC expert vs its teacher: plan error on held-out teacher steps, and closed-loop rollouts in the
    finger-ball simulation from identical starts and targets. -> <run>/eval_mpc.json + .npz"""
    import sys
    model, ck = load_checkpoint(run)
    cfg = ck["cfg"]
    mc = cfg["mpc"]
    data = MPCData(cfg, "heldout")
    rows = {k: [] for k in ("u_first", "u_plan", "force", "contact_ok", "contact", "ep")}
    for s in range(0, len(data), 4096):
        b = data.batch(torch.arange(s, min(s + 4096, len(data))))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            o = model.forward_mpc(b)
        n = b["u"].shape[1]
        eu = (o["u_norm"][:, :n].float() * model.mpc_u_std - b["u"]).abs()
        ef = (o["force_norm"][:, :n].float() * model.mpc_force_std - b["force"]).abs()
        rows["u_first"].append(eu[:, 0].mean(-1)); rows["u_plan"].append(eu.mean((1, 2))); rows["force"].append(ef.mean((1, 2)))
        rows["contact_ok"].append(((o["contact_logit"] > 0) == (b["contact_soon"] > 0.5)).float())
        rows["contact"].append(b["contact_now"].float()); rows["ep"].append(b["ep_id"].float())
    R = {k: torch.cat(v).cpu().numpy() for k, v in rows.items()}
    c = R["contact"] > 0.5
    split = lambda x: {"all": float(x.mean()), "contact": float(x[c].mean()), "free": float(x[~c].mean())}  # noqa: E731
    res = {"heldout_steps": len(c), "heldout_episodes": data.episodes, "contact_steps": int(c.sum()),
           "u_first_n": split(R["u_first"]), "u_plan_n": split(R["u_plan"]), "force_n": split(R["force"]),
           "u_zero_baseline_n": float(data.u.abs().mean()), "contact_soon_acc": float(R["contact_ok"].mean())}

    sys.path.insert(0, mc["sim_path"])
    from finger_ball import FingerBallParams  # noqa: E402
    from rollout import make_controller, rollout, sample_episode  # noqa: E402
    p = FingerBallParams()
    lcs, c3 = make_controller(p, data.horizon)
    hist = data.hist.numpy()

    def student(obs_log, x, xd):
        idx = np.maximum(len(obs_log) - 1 + hist, 0)
        ob = torch.from_numpy(obs_log[idx].astype(np.float32))[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            o = model.forward_mpc({"obs": ob})
        return (o["u_norm"][0, 0].float() * model.mpc_u_std).cpu().numpy().astype(np.float64)

    policies = {"teacher": lambda o, x, xd: c3.solve(x, np.tile(xd, (data.horizon + 1, 1)))["u0"],
                "student": student, "no_control": lambda o, x, xd: np.zeros(2)}
    rng = np.random.default_rng(mc["rollout_seed"])
    starts = [sample_episode(rng, p, mc["rollout_steps"]) for _ in range(mc["rollout_episodes"])]
    roll = {}
    for name, pol in policies.items():
        logs = [rollout(pol, lcs, p, *s) for s in starts]
        err = np.array([np.abs(l["x"][:, 2] - l["target"]).mean() * 1000 for l in logs])
        roll[name] = {"ball_err_mm": err, "ms": float(np.mean([l["ms"][5:].mean() for l in logs])),
                      "normal_force_n": float(np.mean([l["lam"][:, 3].mean() for l in logs])),
                      "example": {k: logs[0][k] for k in ("x", "u", "lam", "target")}}
    res["closed_loop"] = {k: {"ball_err_mm": float(v["ball_err_mm"].mean()), "ms_per_step": v["ms"],
                              "mean_normal_force_n": v["normal_force_n"]} for k, v in roll.items()}
    res["closed_loop"]["episodes"] = len(starts)
    np.savez_compressed(run / "eval_mpc.npz", **{f"{k}_ball_err_mm": v["ball_err_mm"] for k, v in roll.items()},
                        **{f"{k}_example_{f}": a for k, v in roll.items() for f, a in v["example"].items()},
                        **{f"heldout_{k}": v for k, v in R.items()})
    (run / "eval_mpc.json").write_text(json.dumps(res, indent=1))
    return res
