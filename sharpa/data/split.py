"""Held-out split (10 % of episodes per task, fixed by seed) and nested training-share subsets.

* Held-out: for each task, round(holdout_fraction * N) episodes, stratified by season and drawn
  with a generator seeded by (split_seed, task). Seeding per task means adding/removing a task
  never changes another task's held-out set.
* Training shares: a second, independent permutation of each task's training episodes
  (seeded by subset_seed, task). The share-p subset is its first ceil(p * N_train) episodes, so
  10 % ⊂ 25 % ⊂ 50 % ⊂ 100 % and the data-scaling curve only ever adds data.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .episodes import EpisodeRef, index_task


def _rng(seed: int, task: str, purpose: str) -> np.random.Generator:
    # stable across Python runs (unlike hash()), independent per task and purpose
    h = int.from_bytes(hashlib.sha256(f"{purpose}:{task}".encode()).digest()[:8], "little")
    return np.random.default_rng([seed, h])


def split_task(refs: list[EpisodeRef], task: str, holdout_fraction: float, split_seed: int):
    """Hold out round(fraction * N) episodes, stratified by collection season.

    Each season gets its proportional share (largest-remainder rounding so the total is exact),
    then episodes are drawn at random within the season. This keeps every season (operator /
    day / scene setup) represented in the held-out set instead of leaving it to chance.
    """
    rng = _rng(split_seed, task, "holdout")
    seasons = sorted({r.season for r in refs})
    by_season = {s: [r for r in refs if r.season == s] for s in seasons}
    n_hold = round(holdout_fraction * len(refs))
    quota = {s: holdout_fraction * len(by_season[s]) for s in seasons}
    alloc = {s: math.floor(q) for s, q in quota.items()}
    for s in sorted(seasons, key=lambda s: quota[s] - alloc[s], reverse=True)[: n_hold - sum(alloc.values())]:
        alloc[s] += 1
    held_ids = set()
    for s in seasons:
        eps = by_season[s]
        held_ids.update(eps[i].uid for i in rng.permutation(len(eps))[: alloc[s]])
    key = lambda r: (r.season, r.episode_index)  # noqa: E731
    held = sorted((r for r in refs if r.uid in held_ids), key=key)
    train = sorted((r for r in refs if r.uid not in held_ids), key=key)
    return train, held


def train_subset(train: list[EpisodeRef], task: str, share: float, subset_seed: int) -> list[EpisodeRef]:
    if not 0 < share <= 1:
        raise ValueError(f"train_episode_share must be in (0, 1], got {share}")
    order = _rng(subset_seed, task, "subset").permutation(len(train))
    k = math.ceil(share * len(train))
    return sorted((train[i] for i in order[:k]), key=lambda r: (r.season, r.episode_index))


def build_splits(cfg: dict) -> dict[str, dict[str, list[EpisodeRef]]]:
    """{task: {"train": [...], "heldout": [...]}} with train already reduced to the configured share."""
    d = cfg["data"]
    out = {}
    for task in d["tasks"]:
        refs = index_task(d["root"], task, d["format"])
        train, held = split_task(refs, task, d["holdout_fraction"], d["split_seed"])
        out[task] = {"train": train_subset(train, task, d["train_episode_share"], d["subset_seed"]),
                     "train_full": train, "heldout": held}
    return out
