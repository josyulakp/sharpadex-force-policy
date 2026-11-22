"""Task outcome of clean_plate_with_eraser in the simulator, from the per-frame log of sim.replay.Sim.

Used by the scene survey, the stage-3 reward and the evaluation, so all three share one definition (configs/stage3.yaml,
section `outcome`). The plate counts as held when its origin is `lift_m` above where it started; wiping is eraser travel
relative to the plate while the eraser presses on the held plate.

Two choices are set in the config, each with the first definition as the default so that earlier runs keep theirs:
  path    raw     frame-to-frame travel of the eraser relative to the plate. Contact jitter adds to it (about half as much again).
          smooth  the same on the eraser's position in the plate's own frame, averaged over the last `smooth_frames` frames.
  placed  upright the plate's origin within `placed_lift_m` of where it started and tilted less than `placed_tilt_deg`.
          resting the plate's lowest point within `placed_gap_m` of the table and tilted less than `placed_tilt_deg`: lying
                  on the table, upright or leaning, but not on its edge, not upside down and not still in the air.
"""
from __future__ import annotations

import numpy as np


def tilt_deg(quat: np.ndarray) -> np.ndarray:
    """Angle between the body's z axis and the world z axis, from quaternions [..., 4] (w, x, y, z)."""
    return np.degrees(np.arccos(np.clip(1 - 2 * (quat[..., 1] ** 2 + quat[..., 2] ** 2), -1, 1)))


def rotation(quat: np.ndarray) -> np.ndarray:
    """Quaternions [T, 4] (w, x, y, z) -> rotation matrices [T, 3, 3] (body axes in world coordinates as columns)."""
    w, x, y, z = (quat / np.linalg.norm(quat, axis=-1, keepdims=True)).T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def progress(log: dict, c: dict) -> dict:
    """Per-frame progress signals. log: plate [T, 7], eraser [T, 7], wipe_force [T]; the first row is the resting pose."""
    plate, eraser = log["plate"], log["eraser"]
    lift = {"plate": plate[:, 2] - plate[0, 2], "eraser": eraser[:, 2] - eraser[0, 2]}
    wiping = (log["wipe_force"] > c["wipe_force_n"]) & (lift["plate"] > c["lift_m"])
    rel = eraser[:, :3] - plate[:, :3]
    step = np.r_[0.0, np.linalg.norm(np.diff(rel, axis=0), axis=1)] * wiping
    local = np.einsum("tji,tj->ti", rotation(plate[:, 3:]), rel)           # the eraser in the plate's own frame
    n = int(c.get("smooth_frames", 9))
    pad = np.pad(local, ((n - 1, 0), (0, 0)), mode="edge")                  # mean of the last n frames
    smooth = np.stack([np.convolve(pad[:, k], np.ones(n) / n, mode="valid") for k in range(3)], 1)
    slow = np.r_[0.0, np.linalg.norm(np.diff(smooth, axis=0), axis=1)] * wiping
    extent = float(np.linalg.norm(np.ptp(smooth[wiping][:, :2], axis=0))) if wiping.any() else 0.0
    return {"plate_lift": lift["plate"], "eraser_lift": lift["eraser"], "wiping": wiping, "wipe_path": np.cumsum(step),
            "wipe_path_smooth": np.cumsum(slow), "wipe_extent": extent, "plate_tilt": tilt_deg(plate[:, 3:])}


def outcome(log: dict, c: dict) -> dict:
    p = progress(log, c)
    tilt, lift = float(p["plate_tilt"][-1]), float(p["plate_lift"][-1])
    if c.get("placed", "upright") == "resting":
        gap = float(log["plate_low"][-1] - log["plate_low"][0])           # lowest point of the plate above the table
        placed = bool(abs(gap) < c["placed_gap_m"] and tilt < c["placed_tilt_deg"])
    else:
        gap, placed = float("nan"), bool(abs(lift) < c["placed_lift_m"] and tilt < c["placed_tilt_deg"])
    raw, smooth = float(p["wipe_path"][-1]), float(p["wipe_path_smooth"][-1])
    path = smooth if c.get("path", "raw") == "smooth" else raw
    return {"plate_lifted": bool(p["plate_lift"].max() > c["lift_m"]), "eraser_lifted": bool(p["eraser_lift"].max() > c["lift_m"]),
            "wipe_path_cm": raw * 100, "wipe_s": float(p["wiping"].sum() / 30), "plate_placed": placed,
            "plate_dropped": bool(lift < -0.05), "plate_max_lift_cm": float(p["plate_lift"].max() * 100),
            "final_plate_lift_cm": lift * 100, "final_plate_tilt_deg": tilt, "final_plate_gap_cm": gap * 100,
            "success": bool(path >= c["wipe_path_m"] and placed),
            # read-outs that contact jitter cannot inflate
            "wipe_path_smooth_cm": smooth * 100, "wipe_extent_cm": p["wipe_extent"] * 100,
            "success_strict": bool(smooth >= c["wipe_path_m"] and placed)}
