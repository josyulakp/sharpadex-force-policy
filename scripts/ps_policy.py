"""PrimeSim world model, step 7: a policy acting inside the video model on held-out episodes.

    cd <primesim src> && . ./env.sh && .venv2/bin/python <this repo>/scripts/ps_policy.py run=runs/three_tasks/no_force/seed0 [episodes=4 skip=96 seconds=16 denoise_steps=30 force=recorded|zero tag=]
    ... ps_policy.py run=... task=<task> model=<world-model run name> step=<N> kinematics=kinematics_3_tasks.pt      # other tasks

Loop, every 1.5 s: the policy sees the three latest generated camera frames. It is called three times, 0.5 s apart, with those
pictures and a joint state predicted from its own commands (each call returns a 16-frame chunk, 15 frames are executed). The 1.5 s of
commands are turned into hand poses (ps_kinematics.py) and given to the video model (1.5 s kept, later commands held), whose last
kept frame is the next observation. Shorter video steps were tried and degrade the video (scripts/ps_long.py).
Not simulated: fingertip forces and joint torques, and the torso, which the policy does not command (the recorded torso motion is
replayed). Force inputs (`force=`): `recorded` (default for a policy with force inputs) replays the tactile and torque readings of the
held-out recording at the same time, so they match the imagined scene only while the rollout stays close to the recording; `zero`
gives zeros (use with a no-force policy). The wrench history of the wrench expert and the fingertip-image latents of the tactile
expert (scripts/tf_features.py) are replayed from the recording in the same way. Only the frame at `skip`, the two joint states
before it and the replayed signals are real.
Output: reports/world_model_primesim/policy_<name><tag>/{episode_<k>.mp4 (recording | policy in the video model), episode_<k>.npz, summary.md}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ps_kinematics as K  # noqa: E402
import ps_long as L  # noqa: E402
import ps_rollout as R  # noqa: E402
from sharpa.config import load_config  # noqa: E402
from sharpa.data.episodes import load_numeric  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402
from sharpa.engine import load_checkpoint  # noqa: E402

EXECUTE, CALLS = 15, 3                                  # raw frames executed per policy call; policy calls per generated window
KEEP = CALLS * EXECUTE // L.RAW                         # video frames kept per window (1.5 s); shorter steps degrade the video
L_HAND, R_HAND = list(range(7, 29)), list(range(36, 58))


class Kinematics:
    def __init__(self):
        saved = torch.load(Path(R.P["data_dir"]).parent / R.ARGS.get("kinematics", "kinematics.pt"))
        self.nets = []
        for weights in saved["fk"]:
            net = K.fk_net()
            net.load_state_dict(weights)
            self.nets.append(net.eval())
        self.W = saved["response"].numpy()

    @torch.no_grad()
    def wide(self, joints, torso):
        """joints [T, 58] and torso [T, 7] (numpy) -> PrimeSim's 75-D layout [T, 75] (tensor), as in ps_prepare.py."""
        j, t = torch.from_numpy(joints).float(), torch.from_numpy(torso).float()
        out = torch.zeros(len(j), 75)
        for k, (o, arm, hand) in enumerate(((0, list(range(0, 7)), L_HAND), (8, list(range(29, 36)), R_HAND))):
            out[:, o:o + 7] = K.to_pose7(self.nets[k](K.features(torch.cat([j[:, arm], t], 1))))
            out[:, o + 7] = j[:, hand].mean(1)
        out[:, 31:53], out[:, 53:75] = j[:, L_HAND], j[:, R_HAND]
        return out

    def step(self, state, command, t):
        """state[t + 1] from state[t], state[t - 1], command[t], command[t - 1]."""
        W = self.W
        return W[:, 0] * state[t] + W[:, 1] * state[t - 1] + W[:, 2] * command[t] + W[:, 3] * command[t - 1] + W[:, 4]


class Vision:
    """The policy's frozen image encoder, applied to generated frames exactly as scripts/cache_features.py does to recorded ones."""

    def __init__(self, cfg):
        import timm
        v = cfg["vision"]
        self.size, self.grid = v["image_size"], v["pool_grid"]
        self.model = timm.create_model(v["encoder"], pretrained=True, img_size=self.size, num_classes=0).cuda().eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, frames):                         # [V, C, H, W] in [-1, 1] -> [1, V, 1 + grid^2, D]
        x = F.interpolate((frames.cuda().float() + 1) / 2, size=self.size, mode="bicubic", antialias=True).clamp(0, 1)
        tok = self.model.forward_features(((x - self.mean) / self.std).half())
        cls, patch = tok[:, :1], tok[:, self.model.num_prefix_tokens:]
        side = int(patch.shape[1] ** 0.5)
        patch = F.adaptive_avg_pool2d(patch.transpose(1, 2).reshape(len(x), -1, side, side).float(), self.grid).flatten(2).transpose(1, 2)
        return torch.cat([cls.float(), patch], 1)[None]


@torch.no_grad()
def act(model, cfg, feats, state, t, tactile, torque, task_id=0, latents=None):
    """Chunk of absolute joint commands [16, 58] and predicted fingertip forces [16, 30] at frame t, from image tokens, the
    joint-state history up to t and the force inputs [T, 60], [T, 58] (zeros or the recording's). `latents` [T, 16, 128]:
    the recording's fingertip-image latents, for a model with the tactile expert."""
    m = cfg["model"]
    hist = np.maximum(t - np.arange(m["obs_history"] - 1, -1, -1) * m["obs_stride"], 0)
    b = {"feats": feats, "state": torch.from_numpy(state[hist]).float().cuda()[None], "task_id": torch.full((1,), task_id, dtype=torch.long, device="cuda"),
         "tactile": torch.from_numpy(tactile[hist]).float().cuda()[None], "torque": torch.from_numpy(torque[hist]).float().cuda()[None]}
    if model.wrench_expert:                                 # consecutive frames, as in FrameData.batch
        w = np.maximum(t - np.arange(m["wrench_expert"]["window"] - 1, -1, -1), 0)
        b["wrench_hist"] = torch.from_numpy(tactile[w]).float().cuda()[None]
    if model.tactile_expert:
        b["tac_latent"] = torch.from_numpy(latents[min(t, len(latents) - 1)]).cuda()[None]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
        out = model(b)
    p = model.to_physical(out, b)
    return p["action"][0].float().cpu().numpy(), p["force"][0].float().cpu().numpy()


def main():
    import imageio.v2 as imageio

    args = R.ARGS
    run = R.REPO / args.get("run", "runs/three_tasks/no_force/seed0")
    step, denoise, skip = int(args.get("step", 6000)), int(args.get("denoise_steps", 30)), int(args.get("skip", 96))
    seconds, size = float(args.get("seconds", 16)), R.P["image_size"]
    out_dir = R.REPO / "reports" / "world_model_primesim" / ("policy_" + "_".join(run.parts[-3:]) + R.SUFFIX + args.get("tag", ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg0 = load_config()
    refs = {r.uid: r for r in build_splits(cfg0)[R.P["task"]]["heldout"]}
    uids = json.loads((Path(R.P["data_dir"]) / f"{R.P['task']}__heldout" / "meta" / "sharpa_episodes.json").read_text())
    data, _ = R.heldout_windows(1, 1)
    episodes = L.episode_spans(data, int(args.get("episodes", 4)))
    model, ck = load_checkpoint(run)
    cfg = ck["cfg"]
    task_id = cfg["data"]["tasks"].index(R.P["task"])
    force_in = args.get("force", "recorded" if model.use_tactile or model.use_torque else "zero")
    assert force_in == "recorded" or not (model.wrench_expert or model.tactile_expert), "the added experts need the recorded signals"
    vision, kin, pipe = Vision(cfg), Kinematics(), R.build_pipeline(step)

    rows = []
    for k, (episode, first_index, length) in enumerate(episodes):
        rec_state, rec_command = K.load([refs[uids[episode]]])[0][:2]                        # [T, 65] measured / commanded joints
        assert len(rec_state) == length
        numeric = load_numeric(refs[uids[episode]])
        rec_tactile, rec_torque = numeric["observation.tactile"].astype(np.float32), numeric["observation.state.joint_torque"][:, :58].astype(np.float32)
        tactile, torque = (rec_tactile, rec_torque) if force_in == "recorded" else (np.zeros_like(rec_tactile), np.zeros_like(rec_torque))
        ref = refs[uids[episode]]
        latents = np.load(Path(cfg["tactile_world_model"]["data_dir"]) / "latents" / ref.task / ref.season / f"episode_{ref.episode_index:06d}.npy") if model.tactile_expert else None
        n_windows = int(min(seconds * 30, length - 1 - skip) // (CALLS * EXECUTE))
        n_calls = n_windows * CALLS
        T = skip + n_calls * EXECUTE + 1
        state, command, force = np.zeros((T + 1, 58)), np.zeros((T, 58)), np.zeros((T, 30))
        state[:skip + 1], command[:skip] = rec_state[:skip + 1, :58], rec_command[:skip, :58]
        torso_s, torso_c = rec_state[:, 58:], rec_command[:, 58:]
        frame = data[first_index + skip]["videos"][:, :, 0]                                  # the only real picture
        generated, real = [frame[:, :, None]], [frame[:, :, None]]
        for w in range(n_windows):
            t0 = skip + w * CALLS * EXECUTE
            feats = vision(frame)
            for t in range(t0, t0 + CALLS * EXECUTE, EXECUTE):                               # same picture, updated joint state
                chunk, f = act(model, cfg, feats, state, t, tactile, torque, task_id, latents)
                command[t:t + EXECUTE], force[t + 1:t + 1 + EXECUTE] = chunk[:EXECUTE], f[:EXECUTE]   # force step h is frame t + 1 + h
                for u in range(t, t + EXECUTE):
                    state[u + 1] = kin.step(state, command, u)
            idx = t0 + L.RAW * np.minimum(np.arange(L.FRAMES - 1), KEEP - 1)                 # 15 commands, then held
            action = L.relative(kin.wide(command[idx], torso_c[idx]), kin.wide(state[t0:t0 + 1], torso_s[t0:t0 + 1])[0])
            out = L.generate(pipe, frame, action, size, denoise)[:, :, 1:KEEP + 1]
            frame = out[:, :, -1]
            generated.append(out)
            real.append(data[first_index + t0]["videos"][:, :, 1:KEEP + 1])
        g, r = L.stacked(torch.cat(generated, 2)), L.stacked(torch.cat(real, 2))
        imageio.mimsave(out_dir / f"episode_{k}.mp4", list(np.concatenate([r, g], axis=2)), fps=10, codec="libx264", macro_block_size=None)
        np.savez_compressed(out_dir / f"episode_{k}.npz", command=command[:T], state=state[:T], recorded_command=rec_command[:T, :58],
                            recorded_state=rec_state[:T, :58], skip=skip, head=g[:, :size],
                            force=force, recorded_force=rec_tactile[:T].reshape(T, 10, 6)[:, :, :3].reshape(T, 30))
        gap = np.abs(command[skip:T - 1] - rec_command[skip:T - 1, :58])
        arm = [*range(7), *range(29, 36)]
        per_block = [float(1000 * gap[b:b + 96].mean()) for b in range(0, len(gap) - 95, 96)]
        rec_force = rec_tactile[:T].reshape(T, 10, 6)[:, :, :3].reshape(T, 30)
        f_gap = np.abs(force[skip + 1:] - rec_force[skip + 1:]).mean(1)
        rows.append({"force_gap_n": [float(f_gap[b:b + 96].mean()) for b in range(0, len(f_gap) - 95, 96)], "episode": episode, "uid": uids[episode], "seconds": n_calls * EXECUTE / 30, "command_gap_mrad": per_block,
                     "arm_gap_mrad": [float(1000 * gap[b:b + 96][:, arm].mean()) for b in range(0, len(gap) - 95, 96)]})
        print(k, episode, "policy vs operator commands (mrad per 3.2 s):", [round(x) for x in per_block], flush=True)
    n = min(len(r["command_gap_mrad"]) for r in rows)
    lines = [f"# Policy `{'/'.join(run.parts[-3:])}` inside PrimeSim (step {step})", "",
             f"{len(rows)} held-out `{R.P['task']}` episodes, world model `{R.MODEL}`, started {skip / 30:.1f} s into the recording, {denoise} denoising steps.", "",
             "Distance between the policy's commands and the operator's at the same time in the recording (mrad, all joints / arms). "
             "It is a measure of divergence, not of error: after a few seconds the two need not be doing the same thing.", "",
             "| | " + " | ".join(f"{3.2 * b:.1f}-{3.2 * (b + 1):.1f} s" for b in range(n)) + " |", "|---|" + "---|" * n,
             "| all joints | " + " | ".join(f"{np.mean([r['command_gap_mrad'][b] for r in rows]):.0f}" for b in range(n)) + " |",
             "| arms | " + " | ".join(f"{np.mean([r['arm_gap_mrad'][b] for r in rows]):.0f}" for b in range(n)) + " |"]
    if model.predict_force:
        lines += ["| predicted vs recorded fingertip force (N) | " + " | ".join(f"{np.mean([r['force_gap_n'][b] for r in rows]):.2f}" for b in range(n)) + " |",
                  "", f"Force inputs: {force_in}. The force row compares the policy's force forecast (up to 0.5 s ahead) with the force in the recording."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
