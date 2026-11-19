"""Tactile world model, step 3: read fingertip forces and joint torques out of TacForeSight's latent, and test its predictions.

    python scripts/tf_readout.py [ckpt=best_model.pt]     -> <data_dir>/runs/<run_name>/readout.pt + reports/results/tactile_world_model.md

TacForeSight predicts the tactile latent of the next 8 frames (0.27 s) from the latent of the last 8 frames, the measured joints, and
the joint commands of the frames being predicted. Its latent has no decoder, so a small network (the read-out) is fitted on the real
latents of the training episodes: latent -> 60-D fingertip force/torque + 58 joint torques. On held-out episodes:
  read-out         forces read from the real latent of the same frame (the best the read-out can do)
  predicted        forces read from the predicted latent, one step (0.27 s) and chained for up to 2.7 s without seeing a real frame again
  force stays      baseline: the last measured force is kept
  other commands   the prediction when the commands come from another window (does the model use the commands?)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
CFG = load_config()
T = CFG["tactile_world_model"]
sys.path[:0] = [T["src"], T["packages"]]
RUN = Path(T["data_dir"]) / "runs" / T["run_name"]
H = T["horizon"]
FORCE = [6 * k + a for k in range(10) for a in range(3)]            # fingertip force components inside the 60-D vector


class Split:
    def __init__(self, part):
        import zarr
        d = zarr.open_group(str(Path(T["data_dir"]) / part / "replay_buffer.zarr"), "r")
        g = d["data"]
        self.left, self.right = g["tactile_left"][:], g["tactile_right"][:]
        self.command, self.state = g["command_next_4x"][:], g["state_4x"][:]
        self.target = np.concatenate([g["wrench"][:], g["torque"][:]], 1)
        ends = d["meta"]["episode_ends"][:]
        self.spans = list(zip(np.concatenate([[0], ends[:-1]]), ends))

    def windows(self, length, stride):
        return np.array([s for a, b in self.spans for s in range(a, b - length + 1, stride)])

    def batch(self, starts, length, norm):
        """Normalised model inputs for windows [start, start + length), exactly as TacForeSight's dataset builds them."""
        idx = starts[:, None] + np.arange(length)
        pack = lambda x: torch.from_numpy(x[idx].reshape(len(starts), length, 35, 20, 3)).float()  # noqa: E731
        tac = torch.cat([pack(self.left), pack(self.right)], -1)
        tac = torch.cat([tac, tac[:, :, -1:]], 2)                                            # 35 -> 36 rows
        i4 = (4 * starts[:, None] + np.arange(4 * length))
        b = {"tactile": tac.cuda(), "force_4x": torch.from_numpy(self.command[i4]).cuda(), "state_4x": torch.from_numpy(self.state[i4]).cuda()}
        return {k: norm[k].normalize(v) for k, v in b.items()}, torch.from_numpy(self.target[idx]).cuda()


def load_model():
    from trainers.dynamics_trainer import build_model, load_dynamics_checkpoint
    from utils.normalizer import MultiFieldNormalizer
    ck = load_dynamics_checkpoint(str(RUN / "checkpoints" / ARGS.get("ckpt", "best_model.pt")), "cuda")
    model = build_model(yaml.safe_load((RUN / "resolved_config.yaml").read_text())).cuda().eval()
    model.load_state_dict(ck["model"])
    norm = MultiFieldNormalizer()
    norm.load_state_dict(ck["normalizer"])
    norm.to("cuda")
    return model, norm


@torch.no_grad()
def latents(model, norm, split, stride=16):
    zs, ys = [], []
    starts = split.windows(16, stride)
    for s in range(0, len(starts), 256):
        b, y = split.batch(starts[s:s + 256], 16, norm)
        zs.append(model.tokenizer(b["tactile"])["global"].flatten(0, 1)); ys.append(y.flatten(0, 1))
    return torch.cat(zs), torch.cat(ys)


def fit_readout(z, y):
    mean, std = y.mean(0), y.std(0).clamp(min=1e-3)
    net = torch.nn.Sequential(torch.nn.Linear(z.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, y.shape[1])).cuda()
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    steps = 6000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(z), (2048,), device="cuda")
        loss = torch.nn.functional.smooth_l1_loss(net(z[i]), (y[i] - mean) / std)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval(), mean, std


def main():
    model, norm = load_model()
    train, val = Split("train"), Split("val")
    net, mean, std = fit_readout(*latents(model, norm, train))
    torch.save({"net": net.state_dict(), "mean": mean.cpu(), "std": std.cpu()}, RUN / "readout.pt")
    read = lambda z: net(z) * std + mean  # noqa: E731
    thr = CFG["data"]["contact_threshold_n"]

    def force_err(pred, y):                                                                  # [.., 118] -> per-frame mean |error| of the 30 force components (N)
        return (pred[..., FORCE] - y[..., FORCE]).abs().mean(-1)

    chunks = 10                                                                              # 10 x 0.27 s chained
    length = H * (chunks + 1)
    starts = val.windows(length + H, 8)
    acc = {k: [] for k in ("readout", "predicted", "stays", "other", "torque_pred", "torque_stays", "contact", "latent", "latent_stays")}
    roll = {k: [[] for _ in range(chunks)] for k in ("predicted", "stays", "readout")}
    with torch.no_grad():
        for s in range(0, len(starts), 128):
            st = starts[s:s + 128]
            z_prev = None
            for r in range(chunks):
                b, y = val.batch(st + r * H, 2 * H, norm)                                    # frames of chunk r and r + 1
                enc = model.encode(b)
                z, cond = enc["z"], enc["cond"]
                if r == 0:
                    z_prev, y0 = z[:, :H], y[:, H - 1:H]                                     # the only real latents the rollout sees
                    one = model.predict(z[:, :H], cond[:, :H])
                    perm = torch.randperm(len(st), device="cuda")
                    other = model.predict(z[:, :H], model.condition_encoder(force_4x=b["force_4x"][perm], state_4x=b["state_4x"])[:, :H])
                    tgt = y[:, H:]
                    acc["readout"].append(force_err(read(z[:, H:]), tgt)); acc["predicted"].append(force_err(read(one), tgt))
                    acc["other"].append(force_err(read(other), tgt)); acc["stays"].append(force_err(y0.expand_as(tgt), tgt))
                    acc["torque_pred"].append((read(one)[..., 60:] - tgt[..., 60:]).abs().mean(-1))
                    acc["torque_stays"].append((y0.expand_as(tgt)[..., 60:] - tgt[..., 60:]).abs().mean(-1))
                    acc["contact"].append(tgt[..., FORCE].view(*tgt.shape[:2], 10, 3).norm(dim=-1).amax(-1) > thr)
                    acc["latent"].append((one - z[:, H:]).pow(2).mean(-1)); acc["latent_stays"].append((z[:, H - 1:H] - z[:, H:]).pow(2).mean(-1))
                z_prev = model.predict(z_prev, cond[:, :H])                                  # chained: fed its own prediction
                tgt = y[:, H:]
                roll["predicted"][r].append(force_err(read(z_prev), tgt).mean(1)); roll["stays"][r].append(force_err(y0.expand_as(tgt), tgt).mean(1))
                roll["readout"][r].append(force_err(read(z[:, H:]), tgt).mean(1))
    a = {k: torch.cat(v).flatten().float().cpu().numpy() for k, v in acc.items()}
    c = a["contact"] > 0.5
    rows = [("force read from the real latent (read-out limit)", "readout"), ("force read from the predicted latent", "predicted"),
            ("the same with another window's commands", "other"), ("force stays as it is", "stays")]
    lines = ["# TacForeSight tactile world model on SharpaDex (clean_plate_with_eraser)", "",
             f"Held-out episodes, {len(starts)} windows. Prediction is {H} frames ({H / 30:.2f} s) at once. Force error = mean absolute error of the "
             "30 fingertip force components (N).", "", "## One step ahead (0.27 s)", "| | all | in contact | free |", "|---|---|---|---|"]
    lines += [f"| {n} | {a[k].mean():.3f} | {a[k][c].mean():.3f} | {a[k][~c].mean():.3f} |" for n, k in rows]
    lines += ["", f"Joint torque (mean absolute error, raw units): predicted {a['torque_pred'].mean():.3f}, torque stays {a['torque_stays'].mean():.3f}.",
              f"Latent error (mean squared): predicted {a['latent'].mean():.4f}, latent stays {a['latent_stays'].mean():.4f}.", "",
              "## Chained without seeing a real frame again", "| Seconds ahead | " + " | ".join(f"{(r + 1) * H / 30:.2f}" for r in range(chunks)) + " |",
              "|---|" + "---|" * chunks]
    for n, k in (("predicted", "predicted"), ("force stays as it is", "stays"), ("read-out limit", "readout")):
        lines.append(f"| {n} | " + " | ".join(f"{torch.cat(roll[k][r]).mean().item():.3f}" for r in range(chunks)) + " |")
    out = REPO_ROOT / "reports" / "results" / "tactile_world_model.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
