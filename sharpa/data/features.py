"""Location of the cached vision features (written by scripts/cache_features.py)."""
from __future__ import annotations

from pathlib import Path

from .episodes import EpisodeRef


def feature_dir(cfg: dict) -> Path:
    v = cfg["vision"]
    cams = "-".join(v["cameras"])
    return Path(v["cache_dir"]) / f"{v['encoder']}_s{v['image_size']}_g{v['pool_grid']}_{cams}"


def feature_path(cfg: dict, ref: EpisodeRef) -> Path:
    return feature_dir(cfg) / ref.task / ref.season / f"episode_{ref.episode_index:06d}.npy"
