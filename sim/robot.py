"""Sharpa D01 MuJoCo model (public assets) and the mapping from SharpaDex joint vectors to it.

Hand joints map one-to-one (documented by the dataset). Arm joints differ by a fixed sign and
zero offset per joint, found by sim/calibrate_arms.py from the recorded hand-base poses.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path("/opt/dlami/nvme/external/sharpa_north_mujoco/mjcf/sharpa-assets/d01/d01-w/ha1.2/mjcf")
ROBOT_XML = ASSETS / "NTH-HA1-precise-collision.xml"
SCENE_XML = ASSETS / "NTH-HA1-scene-precise-collision.xml"
HAND = ['thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA', 'thumb_IP',
        'index_MCP_FE', 'index_MCP_AA', 'index_PIP', 'index_DIP', 'middle_MCP_FE', 'middle_MCP_AA', 'middle_PIP',
        'middle_DIP', 'ring_MCP_FE', 'ring_MCP_AA', 'ring_PIP', 'ring_DIP', 'pinky_CMC', 'pinky_MCP_FE',
        'pinky_MCP_AA', 'pinky_PIP', 'pinky_DIP']
ARM_DIMS = {"left": slice(0, 7), "right": slice(29, 36)}
HAND_DIMS = {"left": slice(7, 29), "right": slice(36, 58)}
TORSO = [f"lower_body_joint_{i}" for i in range(1, 6)]


def joint_names() -> list[str]:
    """Model joint for each of the 58 dataset dims (left arm, left hand, right arm, right hand)."""
    out = []
    for side in ("left", "right"):
        out += [f"{side}_arm_joint_{i}" for i in range(1, 8)] + [f"{side}_{s}" for s in HAND]
    return out


def qpos_adr(m: mujoco.MjModel, names: list[str]) -> np.ndarray:
    return np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])


def body_pose(m, d, root: str, body: str):
    """Pose of `body` in the frame of `root`: (pos [3], rot [3, 3])."""
    r, b = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in (root, body))
    Rr = d.xmat[r].reshape(3, 3)
    return Rr.T @ (d.xpos[b] - d.xpos[r]), Rr.T @ d.xmat[b].reshape(3, 3)
