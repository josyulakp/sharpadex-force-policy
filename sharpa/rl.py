"""Offline refinement after Facet-0 (stage 2): judge columns, interaction reward, contact regime, credit labels.

The judge columns (subgoal_index, subgoal_complete) come from the dataset's own subtask annotation: every episode runs
through its subgoals in order and ends with an unlabelled tail. Reward, contact intensity and credit follow
facet0/refinement/{reward,credit}.py and facet0/contact.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data.episodes import EpisodeRef


def judge_columns(ref: EpisodeRef, subtask_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """-> subgoal_index [T] (0..S-1 inside a subgoal, S once all are done), subgoal_complete [T] (last frame of each), S."""
    rows = [json.loads(x) for x in (Path(ref.path).parents[2] / "meta" / "subtasks.jsonl").read_text().splitlines()]
    order = {r["subtask_index"]: k for k, r in enumerate(r for r in rows if not r["subtask"].startswith("Task:"))}
    pos = np.array([order.get(int(x), -1) for x in subtask_index])
    sub = np.where(pos >= 0, pos, np.maximum.accumulate(pos) + 1)        # outside a segment: the next subgoal
    complete = np.zeros(len(sub), bool)
    complete[:-1] = (np.diff(sub) > 0) & (pos[:-1] >= 0)
    complete[-1] = pos[-1] >= 0
    return sub, complete, len(order)


def elapsed_in_subgoal(sub: np.ndarray) -> np.ndarray:
    starts = np.r_[0, np.flatnonzero(np.diff(sub)) + 1]
    return np.arange(len(sub)) - np.repeat(starts, np.diff(np.r_[starts, len(sub)])) + 1


def interaction_reward(complete, elapsed, wrench, envelope, time_budget, alpha: float, beta: float) -> np.ndarray:
    """Facet-0 Eq. 4: subgoal completion, scaled up when it comes early, minus the wrench outside its envelope.
    wrench [T, 10, 6], envelope [T, 6], time_budget [T] (frames)."""
    r = np.asarray(complete, np.float64) * (1.0 + beta * np.maximum(1.0 - elapsed / time_budget, 0.0))
    over = (np.abs(wrench) / envelope[:, None, :]).max((1, 2))
    return (r - alpha * np.maximum(over - 1.0, 0.0)).astype(np.float32)


def discounted_returns(r: np.ndarray, gamma: float) -> np.ndarray:
    g = np.zeros(len(r), np.float32)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        g[t] = acc
    return g


def wrench_change(wrench: np.ndarray, window: int = 5, lag: int = 8) -> np.ndarray:
    """|filtered wrench now - filtered wrench `lag` frames ago| over all 60 channels."""
    c = np.cumsum(np.concatenate([np.zeros_like(wrench[:1]), wrench]), 0, dtype=np.float64)
    lo = np.maximum(np.arange(len(wrench)) + 1 - window, 0)
    f = (c[1:] - c[lo]) / (np.arange(len(wrench)) + 1 - lo)[:, None]
    return np.linalg.norm(f - f[np.maximum(np.arange(len(f)) - lag, 0)], axis=-1)


def contact_intensity(changes: list[np.ndarray], floor: float, episode_quantile: float = 0.8) -> list[np.ndarray]:
    """Wrench change scaled by its own episode's 0.8 quantile (never below the corpus floor), clipped to [0, 1].
    The contact regime is intensity == 1."""
    out = []
    for d in changes:
        kappa = max(float(np.quantile(d, episode_quantile)), floor)
        out.append(np.clip(d / kappa, 0.0, 1.0) if kappa > 1e-6 else np.zeros_like(d))
    return out


def episode_credit(values: np.ndarray, rewards: np.ndarray, horizon: int) -> np.ndarray:
    """Facet-0 Eq. 6: reward over the next `horizon` frames + value there - value now (0 after the episode ends)."""
    T = len(values)
    csum = np.r_[0.0, np.cumsum(rewards, dtype=np.float64)]
    t = np.arange(T)
    v_next = np.where(t + horizon < T, values[np.minimum(t + horizon, T - 1)], 0.0)
    return csum[np.minimum(t + horizon, T)] - csum[t] + v_next - values


class Table:
    """Per-frame arrays for many episodes in one .npz: `uid` [E], `length` [E] and flat [sum(length), ...] arrays."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.arrays = {k: z[k] for k in z.files if k not in ("uid", "length", "split")}
        ends = np.cumsum(z["length"])
        self.rows = {str(u): slice(int(e - n), int(e)) for u, e, n in zip(z["uid"], ends, z["length"])}
        self.split = dict(zip(map(str, z["uid"]), map(str, z["split"])))

    def get(self, uid: str, key: str) -> np.ndarray:
        return self.arrays[key][self.rows[uid]]
