"""One recorded episode for the simulator: joint streams, hand-base poses, and where the plate is.

The plate is located from the first stereo head-camera frame (white disc on the dark table), triangulated
with the published calibration, and expressed in the dataset root frame through the recorded hand-base
pose (known both in the camera frame and in the root frame at every frame).
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation as R

DATA = Path("/opt/dlami/nvme/sharpadex")
CALIB = Path("/opt/dlami/nvme/sharpadex_calib")


def pose_matrix(p6) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.from_rotvec(p6[3:]).as_matrix(), p6[:3]
    return T


class Episode:
    def __init__(self, uid: str):
        """uid: <task>/<season>/<episode index, 6 digits>."""
        self.uid = uid
        self.task, self.season, ep = uid.split("/")
        self.index = int(ep)
        self.root = DATA / self.task / self.season / "lerobot_v2.1"
        df = pd.read_parquet(self.root / f"data/chunk-000/episode_{self.index:06d}.parquet")
        col = lambda c: np.stack(df[c].values).astype(np.float64)  # noqa: E731
        self.state, self.action = col("observation.state"), col("action")
        self.tcp, self.tactile = col("observation.state.tcp_pose"), col("observation.tactile").reshape(-1, 10, 6)
        self.subtask = df["subtask_index"].values
        self.force = np.linalg.norm(self.tactile[:, :, :3], axis=2)          # [T, 10] N
        cal_dir = CALIB / self.task / self.season
        groups = json.loads((cal_dir / "lerobot_v2.1/meta/camera_calibration.json").read_text())["calibrations"]
        self.calib = next(g for g in groups if g["episode_start"] <= self.index <= g["episode_end"])
        self.cam_tcp = dict(np.load(cal_dir / f"camera_tcp/episode_{self.index:06d}.npz"))

    def __len__(self):
        return len(self.state)

    def video_frame(self, camera: str, t: int) -> np.ndarray:
        return iio.imread(self.root / f"videos/chunk-000/observation.images.{camera}/episode_{self.index:06d}.mp4", index=t)

    def root_from_head_left(self, t: int = 0) -> np.ndarray:
        """4x4 transform head-left camera -> dataset root frame at frame t (average of both hands)."""
        Ts = [pose_matrix(self.tcp[t, s:s + 6]) @ np.linalg.inv(pose_matrix(self.cam_tcp[f"{side}_arm_tcp_in_head_left_camera"][t]))
              for side, s in (("left", 0), ("right", 6))]
        T = Ts[0].copy()
        T[:3, 3] = (Ts[0][:3, 3] + Ts[1][:3, 3]) / 2
        self.camera_disagreement_mm = float(np.linalg.norm(Ts[0][:3, 3] - Ts[1][:3, 3]) * 1000)
        return T

    def locate_plate(self, t: int = 0) -> dict:
        """Plate centre (root frame, m) and radius (m) from the stereo pair at frame t."""
        K = {c: np.array(self.calib["cameras"][c]["camera_matrix"]) for c in ("head_left", "head_right")}
        cands = {c: _discs(self.video_frame(c, t)) for c in K}
        # the plate is the pair of round blobs on the same image row with a plausible disparity
        T_rl = np.array(self.calib["stereo_extrinsics"]["T_head_right_camera_head_left_camera"])
        P1, P2 = K["head_left"] @ np.eye(4)[:3], K["head_right"] @ T_rl[:3]
        best = None
        for a in cands["head_left"]:
            for b in cands["head_right"]:
                A = np.array([a[0] * P1[2] - P1[0], a[1] * P1[2] - P1[1], b[0] * P2[2] - P2[0], b[1] * P2[2] - P2[1]])
                X = np.linalg.svd(A)[2][-1]
                X = X[:3] / X[3]                                          # head-left camera frame
                radius = a[2] / 2 * X[2] / K["head_left"][0, 0]           # image width of the disc -> metres
                cost = abs(a[1] - b[1]) + 5 * abs(a[3] - b[3])            # same image row, same size
                if 0.06 < radius < 0.15 and 0.3 < X[2] < 1.2 and (best is None or cost < best[0]):
                    best = (cost, a, b, X, radius)
        if best is None:
            raise RuntimeError(f"{self.uid}: no plate found in frame {t}")
        _, a, b, X, radius = best
        centre = self.root_from_head_left(t) @ np.r_[X, 1]
        return {"centre_root": centre[:3], "radius": float(radius), "depth": float(X[2]),
                "pixel_left": a[:2], "pixel_right": b[:2]}


def _discs(img: np.ndarray) -> list[tuple]:
    """Bright, colourless, round blobs that do not touch the image border: (cx, cy, width, height)."""
    g = img.astype(float)
    bright = (g.min(2) > 150) & ((g.max(2) - g.min(2)) < 40)
    lab, _ = ndi.label(ndi.binary_opening(bright, iterations=2))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i
        h, w = m.shape
        touches = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == img.shape[0] or sl[1].stop == img.shape[1]
        if m.sum() < 1200 or touches or m.sum() / (np.pi * h * w / 4) < 0.9:
            continue
        cy, cx = ndi.center_of_mass(lab == i)
        out.append((float(cx), float(cy), float(w), float(h)))
    return out
