"""Learned forward kinematics of the recorded robot: arm joints (+ torso) -> hand-base pose in the root frame.

    python sim/fit_fk.py [task]      -> sim/fk_<task>.pt   (run with the training environment: needs torch)

Why: the public D01 arm model does not reproduce the recorded hand poses from the recorded arm joints
(3-5 cm at best, see sim/calibrate_arms.py), so the simulator drives the hand bases by pose. To execute a
*predicted* joint target, this map turns its arm part into a hand-base pose. It is fitted on training
episodes only and checked on the held-out episodes and on the recorded commanded poses.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import load_config  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402

ARM = {"left": slice(0, 7), "right": slice(29, 36)}
TCP = {"left": slice(0, 6), "right": slice(6, 12)}


def rotvec_to_matrix(r: torch.Tensor) -> torch.Tensor:
    th = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / th
    K = torch.zeros(*r.shape[:-1], 3, 3, device=r.device, dtype=r.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, device=r.device, dtype=r.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def six_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(x[..., :3], dim=-1)
    b = nn.functional.normalize(x[..., 3:] - (a * x[..., 3:]).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


class FK(nn.Module):
    """[7 arm joints, 5 torso joints] -> position [3] (m), rotation matrix [3, 3]."""

    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 9))

    def forward(self, q):
        y = self.net(torch.cat([q.sin(), q.cos()], -1))
        return y[..., :3], six_to_matrix(y[..., 3:])


def load(refs):
    cols = ["observation.state", "action", "observation.state.tcp_pose", "action.tcp_pose"]
    out = [[], [], [], []]
    for r in refs:
        df = pd.read_parquet(r.path, columns=cols)
        for o, c in zip(out, cols):
            o.append(np.stack(df[c].values))
    return [torch.from_numpy(np.concatenate(o)).float().cuda() for o in out]


def errors(model, q, tcp):
    with torch.no_grad():
        p, Rm = model(q)
        Rt = rotvec_to_matrix(tcp[:, 3:])
        cos = ((Rm.transpose(1, 2) @ Rt).diagonal(dim1=1, dim2=2).sum(1) - 1) / 2
        return (p - tcp[:, :3]).norm(dim=1).mean().item() * 1000, torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item()


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "clean_plate_with_eraser"
    cfg = load_config(overrides=[f"data.tasks=[{task}]"])
    sp = build_splits(cfg)[task]
    st, ac, tcp, atcp = load(sp["train"])
    hst, hac, htcp, hatcp = load(sp["heldout"])
    ckpt = {}
    for side in ("left", "right"):
        qin = lambda s: torch.cat([s[:, ARM[side]], s[:, 58:63]], 1)  # noqa: E731
        torch.manual_seed(0)
        model = FK().cuda()
        opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0)
        steps = 20000
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        q, y = qin(st), tcp[:, TCP[side]]
        Rt = rotvec_to_matrix(y[:, 3:])
        for _ in range(steps):
            i = torch.randint(len(q), (4096,), device="cuda")
            p, Rm = model(q[i])
            loss = (p - y[i, :3]).abs().mean() + 0.2 * (Rm - Rt[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tr, ho = errors(model, q, y), errors(model, qin(hst), htcp[:, TCP[side]])
        # commanded joints of the held-out episodes -> commanded hand pose (torso taken from the state)
        qa = torch.cat([hac[:, ARM[side]], hst[:, 58:63]], 1)
        cmd = errors(model, qa, hatcp[:, TCP[side]])
        print(f"{side}: train {tr[0]:.2f} mm / {tr[1]:.2f} deg | held-out episodes {ho[0]:.2f} mm / {ho[1]:.2f} deg | "
              f"held-out commanded joints vs commanded pose {cmd[0]:.2f} mm / {cmd[1]:.2f} deg", flush=True)
        ckpt[side] = {"model": model.state_dict(), "heldout_mm_deg": ho, "commanded_mm_deg": cmd}
    torch.save(ckpt, Path(__file__).with_name(f"fk_{task}.pt"))


if __name__ == "__main__":
    main()
