"""Four experts trained for 25k and 50k steps: L1 action loss against the heavy-tailed residual loss of HT-Policy.

    python scripts/run_experiments.py ht_loss     # trains runs/ht_loss/{l1_25k,ht_25k,ht_50k}/seed*; L1 at 50k is runs/headline/full
    python scripts/ht_report.py                   -> reports/results/ht_loss.md

Held-out joint error as mean absolute error (what L1 minimises) and as root-mean-square error (closer to what the heavy-tailed
loss minimises), at 8k, 25k and 50k steps, with a paired bootstrap over held-out episodes for the difference. For the heavy-tailed
model also: does the scale it predicts for a chunk tell how large the error of that chunk is?
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402

RUNS = REPO_ROOT / "runs"
ROWS = [("two experts, L1, 25k", "flow/regression_25k", "heldout"), ("four experts, L1, 8k (earlier run)", "tactile_expert/four_experts", "heldout"),
        ("four experts, L1, 8k", "ht_loss/l1_25k", "heldout__step8000"), ("four experts, heavy-tailed, 8k", "ht_loss/ht_25k", "heldout__step8000"),
        ("four experts, L1, 25k", "ht_loss/l1_25k", "heldout"), ("four experts, heavy-tailed, 25k", "ht_loss/ht_25k", "heldout"),
        ("four experts, L1, 50k", "headline/full", "heldout"), ("four experts, heavy-tailed, 50k", "ht_loss/ht_50k", "heldout"),
        ("four experts, L1, 50k run at step 25k", "headline/full", "heldout__step25000"),
        ("four experts, heavy-tailed, 50k run at step 25k", "ht_loss/ht_50k", "heldout__step25000")]


def load(path, tag):
    runs = sorted((RUNS / path).glob("seed*"))
    arrs = [np.load(r / f"eval_{tag}.npz") for r in runs if (r / f"eval_{tag}.npz").exists()]
    keys = [k for k in arrs[0].files if k != "ep_uid" and all(k in a.files for a in arrs)]
    return {k: np.stack([a[k] for a in arrs]) for k in keys}          # [seeds, frames]


def paired(a, b, key, ep, scale, rmse=False, n=4000):
    """Mean and 95 % interval of metric(b) - metric(a): seeds averaged per frame, held-out episodes resampled."""
    x, y = a[key].mean(0), b[key].mean(0)
    eps = np.unique(ep)
    sx, sy, cnt = (np.array([v[ep == e].sum() for e in eps]) for v in (x, y, np.ones_like(x)))
    f = (lambda s, c: np.sqrt(s / c)) if rmse else (lambda s, c: s / c)
    pick = np.random.default_rng(0).integers(0, len(eps), (n, len(eps)))
    d = scale * (f(sy[pick].sum(1), cnt[pick].sum(1)) - f(sx[pick].sum(1), cnt[pick].sum(1)))
    return scale * (f(sy.sum(), cnt.sum()) - f(sx.sum(), cnt.sum())), np.quantile(d, [0.025, 0.975])


def main():
    data = {name: load(path, tag) for name, path, tag in ROWS if any((RUNS / path).glob(f"seed*/eval_{tag}.npz"))}
    c = next(iter(data.values()))["contact"][0] > 0.5
    lines = ["# Heavy-tailed residual loss (HT-Policy) on the four-expert model, held-out episodes, mean of seeds", "",
             "Joint error in mrad: MAE = mean absolute error, RMSE = root-mean-square error. Force error in N.", "",
             "| Model | MAE | in contact | free | RMSE | in contact | free | force | per seed (MAE) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, d in data.items():
        mae = lambda m: 1000 * d["act"][:, m].mean()  # noqa: E731
        rm = (lambda m: f"{1000 * np.sqrt(d['act_sq'][:, m].mean()):.2f}") if "act_sq" in d else (lambda m: "")  # noqa: E731
        every = np.ones_like(c)
        lines.append(f"| {name} | {mae(every):.2f} | {mae(c):.2f} | {mae(~c):.2f} | {rm(every)} | {rm(c)} | {rm(~c)} | {d['force'].mean():.4f} | "
                     + ", ".join(f"{1000 * s.mean():.2f}" for s in d["act"]) + " |")
    lines += ["", "## Heavy-tailed minus L1 (paired over held-out episodes, 95 % interval)", "", "| Steps | MAE (mrad) | RMSE (mrad) | force (N) |", "|---|---|---|---|"]
    for steps in ("8k", "25k", "50k"):
        a, b = data.get(f"four experts, L1, {steps}"), data.get(f"four experts, heavy-tailed, {steps}")
        if a is None or b is None:
            continue
        ep = a["ep"][0].astype(int)
        cells = []
        for key, scale, rmse in (("act", 1000, False), ("act_sq", 1000, True), ("force", 1, False)):
            m, (lo, hi) = paired(a, b, key, ep, scale, rmse)
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" if scale == 1000 else f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append(f"| {steps} | " + " | ".join(cells) + " |")
    for steps in ("25k", "50k"):
        d = data.get(f"four experts, heavy-tailed, {steps}")
        if d is None or "sigma" not in d:
            continue
        sig, err = d["sigma"].mean(0), d["act"].mean(0)
        rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
        q = np.quantile(sig, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(sig, q)
        lines += ["", f"## Predicted scale of the heavy-tailed model ({steps})", "",
                  f"Rank correlation between the predicted scale of a chunk and its joint error: {np.corrcoef(rank(sig), rank(err))[0, 1]:+.3f}. "
                  f"Mean scale {sig.mean():.3f} (normalised action units; {sig[c].mean():.3f} in contact, {sig[~c].mean():.3f} free).", "",
                  "| Frames by predicted scale | lowest fifth | 2nd | 3rd | 4th | highest fifth |", "|---|---|---|---|---|---|",
                  "| joint MAE (mrad) | " + " | ".join(f"{1000 * err[bins == k].mean():.2f}" for k in range(5)) + " |",
                  "| share in contact | " + " | ".join(f"{100 * c[bins == k].mean():.0f}%" for k in range(5)) + " |"]
    out = REPO_ROOT / "reports" / "results" / "ht_loss.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
