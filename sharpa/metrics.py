"""Held-out metrics with intervals over episodes and seeds.

Each run stores one error value per held-out frame. A metric for a frame group (all / contact /
free) is the frame-weighted mean over the group, averaged over seeds. The interval is a two-level
bootstrap: held-out episodes are resampled with replacement (the same draw for every seed and
variant, so comparisons are paired on episodes) and training seeds are resampled with replacement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GROUPS = ("all", "contact", "free")


def load_variant(variant_dir: Path, tag: str = "heldout") -> list[dict]:
    runs = sorted(variant_dir.glob(f"seed*/eval_{tag}.npz"))
    return [dict(np.load(r, allow_pickle=False)) for r in runs]


def episode_tables(seeds: list[dict], key: str):
    """sums[group][seed, episode], counts[group][episode] for one metric."""
    ep = seeds[0]["ep"].astype(int)
    n_ep = ep.max() + 1
    contact = seeds[0]["contact"] > 0.5
    masks = {"all": np.ones_like(contact), "contact": contact, "free": ~contact}
    sums = {g: np.stack([np.bincount(ep[m], s[key][m], n_ep) for s in seeds]) for g, m in masks.items()}
    counts = {g: np.bincount(ep[m], minlength=n_ep).astype(float) for g, m in masks.items()}
    return sums, counts


def _stat(sums, counts, ep_idx, seed_idx):
    return (sums[np.ix_(seed_idx, ep_idx)].sum(1) / counts[ep_idx].sum()).mean()


def bootstrap(variants: dict[str, list[dict]], key: str, reference: str | None, n: int = 2000,
              ci: float = 0.95, seed: int = 0) -> dict:
    """{variant: {group: {mean, lo, hi, [diff, diff_lo, diff_hi vs reference]}}}"""
    rng = np.random.default_rng(seed)
    tabs = {v: episode_tables(s, key) for v, s in variants.items()}
    n_ep = len(next(iter(tabs.values()))[1]["all"])
    ep_draws = rng.integers(0, n_ep, (n, n_ep))
    seed_draws = {v: rng.integers(0, len(s), (n, len(s))) for v, s in variants.items()}
    q = [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100]
    out = {}
    for v, (sums, counts) in tabs.items():
        out[v] = {}
        for g in GROUPS:
            every = np.arange(len(variants[v]))
            point = _stat(sums[g], counts[g], np.arange(n_ep), every)
            boot = np.array([_stat(sums[g], counts[g], ep_draws[i], seed_draws[v][i]) for i in range(n)])
            res = {"mean": float(point), "lo": float(np.percentile(boot, q[0])), "hi": float(np.percentile(boot, q[1])),
                   "per_seed": [float(sums[g][s].sum() / counts[g].sum()) for s in every]}
            if reference and v != reference:
                rs, rc = tabs[reference]
                ref_boot = np.array([_stat(rs[g], rc[g], ep_draws[i], seed_draws[reference][i]) for i in range(n)])
                ref_point = _stat(rs[g], rc[g], np.arange(n_ep), np.arange(len(variants[reference])))
                d = boot - ref_boot
                res |= {"diff": float(point - ref_point), "diff_lo": float(np.percentile(d, q[0])),
                        "diff_hi": float(np.percentile(d, q[1]))}
            out[v][g] = res
    return out
