"""Arm kinematic calibration: dataset arm joints -> model arm joints, checked against recorded hand-base poses.

    python sim/calibrate_arms.py [task]     -> sim/calibration_<task>.json

Model:  tcp_recorded = B * FK(sign * q + offset) * X
  sign, offset : per joint, shared by all episodes (the convention difference)
  X            : fixed transform hand mount -> hand_base_link (TCP frame)
  B            : dataset root_link -> model arm base; one per recording season in mode "season" (torso pose
                 is absorbed), or one in total in mode "torso" (torso joints taken from dataset dims 58-62).
Signs come from an exhaustive search (128 combinations per arm) done once; they are fixed here.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.robot import ARM_DIMS, SCENE_XML, TORSO, body_pose, qpos_adr  # noqa: E402

DATA = Path("/opt/dlami/nvme/sharpadex")
SIGNS = {"left": [1, 1, -1, -1, -1, 1, -1], "right": [-1, -1, -1, 1, -1, 1, -1]}
W_ROT = 0.2                                   # 1 rad of rotation error counts like 0.2 m


def load(task, per_season=6, frames=12, offset=0):
    rows = []
    for si, season in enumerate(sorted((DATA / task).glob("season_*"))):
        files = sorted((season / "lerobot_v2.1/data/chunk-000").glob("*.parquet"))
        for f in files[offset::max(1, len(files) // per_season)][:per_season]:
            df = pd.read_parquet(f, columns=["observation.state", "observation.state.tcp_pose"])
            i = np.linspace(0, len(df) - 1, frames).astype(int)
            st, tcp = np.stack(df.iloc[i, 0].values), np.stack(df.iloc[i, 1].values)
            rows += [(si, s, t) for s, t in zip(st, tcp)]
    return rows


class Fit:
    def __init__(self, side, mode):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.d = mujoco.MjData(self.m)
        self.side, self.mode = side, mode
        self.adr = qpos_adr(self.m, [f"{side}_arm_joint_{i}" for i in range(1, 8)])
        self.tor = qpos_adr(self.m, TORSO)
        self.sign = np.array(SIGNS[side], float)
        self.so = 0 if side == "left" else 6
        self.root = "north" if mode == "torso" else "lower_body_link_5"

    def residual(self, p, rows, n_base):
        off, X = p[:7], p[7:13]
        Rx = R.from_rotvec(X[3:]).as_matrix()
        res = []
        for si, st, tcp in rows:
            b = p[13 + 6 * (si if n_base > 1 else 0):][:6]
            Rb = R.from_rotvec(b[3:]).as_matrix()
            self.d.qpos[:] = self.m.qpos0
            self.d.qpos[self.adr] = self.sign * st[ARM_DIMS[self.side]] + off
            if self.mode == "torso":
                self.d.qpos[self.tor] = st[58:63]
            mujoco.mj_kinematics(self.m, self.d)
            pf, Rf = body_pose(self.m, self.d, self.root, f"{self.side}_hand_mount")
            t = tcp[self.so:self.so + 6]
            pos = Rb @ (Rf @ X[:3] + pf) + b[:3]
            rot = (Rb @ Rf @ Rx).T @ R.from_rotvec(t[3:]).as_matrix()
            res += list(pos - t[:3]) + list(W_ROT * R.from_matrix(rot).as_rotvec())
        return np.array(res)

    def errors(self, p, rows, n_base):
        e = self.residual(p, rows, n_base).reshape(-1, 6)
        return float(np.sqrt((e[:, :3] ** 2).sum(1).mean()) * 1000), float(np.degrees(np.sqrt(((e[:, 3:] / W_ROT) ** 2).sum(1).mean())))


def one_fit(job):
    mode, side, seed, rows, n_base = job
    f = Fit(side, mode)
    rng = np.random.default_rng(seed)
    p0 = np.concatenate([rng.uniform(-np.pi, np.pi, 7), np.zeros(3), R.random(random_state=seed).as_rotvec()] +
                        [np.r_[np.zeros(3), R.random(random_state=seed + 100).as_rotvec()]] * n_base)
    r = least_squares(f.residual, p0, args=(rows, n_base), max_nfev=80)
    return mode, side, r.cost, r.x


def main():
    from multiprocessing import Pool
    task = sys.argv[1] if len(sys.argv) > 1 else "clean_plate_with_eraser"
    fit_rows, test_rows = load(task), load(task, offset=3)
    n_season = max(r[0] for r in fit_rows) + 1
    n_bases = {"season": n_season, "torso": 1}
    jobs = [(mode, side, seed, fit_rows, n_bases[mode]) for mode in n_bases for side in ("left", "right") for seed in range(7)]
    with Pool(28) as pool:
        fits = pool.map(one_fit, jobs, chunksize=1)
    out = {"task": task, "signs": SIGNS, "seasons": [s.name for s in sorted((DATA / task).glob("season_*"))]}
    for mode, n_base in n_bases.items():
        for side in ("left", "right"):
            x = min((f for f in fits if f[0] == mode and f[1] == side), key=lambda f: f[2])[3]
            f = Fit(side, mode)
            tr, te = f.errors(x, fit_rows, n_base), f.errors(x, test_rows, n_base)
            print(f"{mode:6s} {side:5s}: fit {tr[0]:.1f} mm / {tr[1]:.2f} deg | other episodes {te[0]:.1f} mm / {te[1]:.2f} deg | "
                  f"offsets deg {np.round(np.degrees((x[:7] + np.pi) % (2 * np.pi) - np.pi), 1).tolist()}", flush=True)
            out[f"{mode}_{side}"] = {"offset": x[:7].tolist(), "tool": x[7:13].tolist(),
                                     "base": x[13:].reshape(-1, 6).tolist(), "fit_mm_deg": tr, "test_mm_deg": te}
    Path(__file__).with_name(f"calibration_{task}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
