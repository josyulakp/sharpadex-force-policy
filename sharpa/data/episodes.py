"""Episode index and numeric loading for the SharpaDex LeRobot-v2.1 export.

The release is organised as <task>/<season>/lerobot_v2.1/{data,meta,videos}; episode_index is
only unique within a season, so an episode is identified by (task, season, episode_index).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# 65-D joint layout (state, action, joint_torque): see dataset README.
L_ARM, L_HAND = slice(0, 7), slice(7, 29)
R_ARM, R_HAND = slice(29, 36), slice(36, 58)
TORSO = slice(58, 65)
FINGERS = ["thumb", "index", "middle", "ring", "little"]
FINGERTIP_NAMES = [f"{h}_{f}" for h in ("left", "right") for f in FINGERS]

NUMERIC_COLUMNS = [
    "observation.state", "action", "observation.state.joint_torque",
    "observation.tactile", "subtask_index", "frame_index", "timestamp",
]


@dataclass(frozen=True)
class EpisodeRef:
    task: str
    season: str
    episode_index: int
    num_frames: int
    path: str

    @property
    def uid(self) -> str:
        return f"{self.task}/{self.season}/{self.episode_index:06d}"


def index_task(root: str | Path, task: str, fmt: str = "lerobot_v2.1") -> list[EpisodeRef]:
    """All episodes of one task, in a deterministic order (season, episode_index)."""
    refs = []
    for season_dir in sorted((Path(root) / task).iterdir()):
        meta = season_dir / fmt / "meta"
        if not meta.is_dir():
            continue
        info = json.loads((meta / "info.json").read_text())
        for line in (meta / "episodes.jsonl").read_text().splitlines():
            ep = json.loads(line)
            i = ep["episode_index"]
            path = season_dir / fmt / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
            refs.append(EpisodeRef(task, season_dir.name, i, ep["length"], str(path)))
    return sorted(refs, key=lambda r: (r.season, r.episode_index))


def load_numeric(ref: EpisodeRef) -> dict[str, np.ndarray]:
    """Per-frame numeric streams of one episode (videos are not decoded here)."""
    table = pq.read_table(ref.path, columns=NUMERIC_COLUMNS)
    out = {}
    for c in NUMERIC_COLUMNS:
        col = table.column(c).to_numpy(zero_copy_only=False)
        out[c] = np.stack(col).astype(np.float32) if col.dtype == object else np.asarray(col)
    return out


def fingertip_force_magnitude(tactile: np.ndarray) -> np.ndarray:
    """(T, 60) tactile -> (T, 10) |F| per fingertip in newtons (torques excluded)."""
    t = tactile.reshape(len(tactile), 10, 6)
    return np.linalg.norm(t[:, :, :3], axis=2)


def contact_mask(tactile: np.ndarray, threshold_n: float) -> np.ndarray:
    """Frame-level contact label used for the contact / free-motion evaluation split."""
    return fingertip_force_magnitude(tactile).max(axis=1) > threshold_n


def to_jsonable(refs: list[EpisodeRef]) -> list[dict]:
    return [asdict(r) | {"uid": r.uid} for r in refs]
