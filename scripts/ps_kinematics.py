"""PrimeSim world model, step 6: the two small models a policy needs to act inside the video model.

    python scripts/ps_kinematics.py        -> <primesim_world_model.data_dir>/../kinematics.pt + printed held-out errors

1. Hand pose from arm and torso joints (per hand: 7 + 7 joint angles -> position + rotation), fitted on measured and commanded poses of the
   training episodes. The video model is driven by hand poses; the policy commands joints.
2. Joint response: measured joints one frame later as a linear function of the last two measured joints and commands, per joint.
   The video model returns pictures only, so the joint state fed back to the policy is predicted from its own commands.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import load_config  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402

TORSO = list(range(58, 65))                             # the torso moves the hands by about 2 cm; the policy does not command it
ARMS = [list(range(0, 7)) + TORSO, list(range(29, 36)) + TORSO]
COLS = ["observation.state", "action", "observation.state.tcp_pose", "action.tcp_pose"]


def load(refs):
    out = []
    for r in refs:
        t = pq.read_table(r.path, columns=COLS)
        out.append([np.stack(t.column(c).to_numpy(zero_copy_only=False)).astype(np.float32) for c in COLS])
    return out


def pose9(p6):                                          # xyz + rotation vector -> xyz + first two rotation-matrix columns
    m = Rotation.from_rotvec(p6[:, 3:]).as_matrix()
    return np.concatenate([p6[:, :3], m[:, :, 0], m[:, :, 1]], 1).astype(np.float32)


def to_pose7(p9: torch.Tensor) -> torch.Tensor:         # xyz + two columns -> xyz + quaternion xyzw (Gram-Schmidt)
    a = torch.nn.functional.normalize(p9[:, 3:6], dim=1)
    b = torch.nn.functional.normalize(p9[:, 6:9] - (a * p9[:, 6:9]).sum(1, keepdim=True) * a, dim=1)
    m = torch.stack([a, b, torch.cross(a, b, dim=1)], 2).cpu().numpy()
    return torch.cat([p9[:, :3].cpu(), torch.from_numpy(Rotation.from_matrix(m).as_quat()).float()], 1)


def fk_net():
    return torch.nn.Sequential(torch.nn.Linear(28, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256), torch.nn.SiLU(),
                               torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, 9))


def features(q):                                        # joint angles -> sin, cos
    return torch.cat([q.sin(), q.cos()], 1)


def respond(first_two, command, W):
    """Joint state [T, 58] rolled forward from its first two frames and the commands [T, 58]."""
    p = np.zeros((len(command), 58))
    p[:2] = first_two
    for t in range(2, len(command)):
        p[t] = W[:, 0] * p[t - 1] + W[:, 1] * p[t - 2] + W[:, 2] * command[t - 1] + W[:, 3] * command[t - 2] + W[:, 4]
    return p


def main():
    cfg = load_config()
    P = cfg["primesim_world_model"]
    tasks = sys.argv[1].split(",") if len(sys.argv) > 1 else [P["task"]]          # optional: task,task,... -> kinematics_<n>_tasks.pt
    splits = build_splits(cfg)
    train, held = (load([r for t in tasks for r in splits[t][part]]) for part in ("train", "heldout"))
    result = {"fk": []}
    for k, arm in enumerate(ARMS):
        def pairs(eps):
            q = np.concatenate([np.concatenate([e[0][:, arm], e[1][:, arm]]) for e in eps])
            p = np.concatenate([np.concatenate([e[2][:, 6 * k:6 * k + 6], e[3][:, 6 * k:6 * k + 6]]) for e in eps])
            return torch.from_numpy(q).cuda(), torch.from_numpy(pose9(p)).cuda()
        (q, p), (qh, ph) = pairs(train), pairs(held)
        net = fk_net().cuda()
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.0)
        steps = 20000
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            i = torch.randint(0, len(q), (4096,), device="cuda")
            loss = (net(features(q[i])) - p[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            e = net(features(qh)) - ph
        n = len(qh) // 2                                 # first half measured, second half commanded
        for name, sl in (("measured", slice(0, n)), ("commanded", slice(n, None))):
            pos = e[sl, :3].norm(dim=1).mean().item() * 1000
            rot = np.degrees(np.sqrt(2) * e[sl, 3:].norm(dim=1).mean().item() / np.sqrt(2))
            print(f"hand {k} {name}: position error {pos:.2f} mm, rotation error about {rot:.2f} deg (held-out episodes)")
        result["fk"].append({n_: v.cpu() for n_, v in net.state_dict().items()})

    # joint response: state[t+1] from state[t], state[t-1], command[t], command[t-1] and a constant; least squares per joint
    def design(e):
        s_, a_ = e[0][:, :58], e[1][:, :58]
        return np.stack([s_[1:-1], s_[:-2], a_[1:-1], a_[:-2], np.ones_like(s_[1:-1])], 2), s_[2:]
    xs, ys = zip(*[design(e) for e in train])
    X, Y = np.concatenate(xs), np.concatenate(ys)
    W = np.stack([np.linalg.lstsq(X[:, j], Y[:, j], rcond=None)[0] for j in range(58)]).astype(np.float64)       # [58, 5]
    result["response"] = torch.from_numpy(W)
    errs = []
    for e in held:                                       # whole episode predicted from the commands alone
        errs.append(np.abs(respond(e[0][:2, :58], e[1][:, :58], W) - e[0][:, :58]))
    arm = [*range(7), *range(29, 36)]
    print(f"joint response, whole held-out episodes from commands alone: error {1000 * np.mean([x.mean() for x in errs]):.1f} mrad "
          f"(arms {1000 * np.mean([x[:, arm].mean() for x in errs]):.1f}); assuming joints = command: "
          f"{1000 * np.mean([np.abs(e[0][:, :58] - e[1][:, :58]).mean() for e in held]):.1f} mrad")
    out = Path(P["data_dir"]).parent / ("kinematics.pt" if len(sys.argv) < 2 else f"kinematics_{len(tasks)}_tasks.pt")
    torch.save(result, out)
    print("saved", out)


if __name__ == "__main__":
    main()
