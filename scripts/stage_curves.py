"""Training curves of stages 1 and 2, from the logs every run keeps (`train_log.json`: one row per 200 steps).

    python scripts/stage_curves.py      -> reports/figures/stage1_curves.png, reports/figures/stage2_curves.png

Stage 1 is the supervised training of the released four-expert model (runs/headline/full, three seeds): its training loss
and the held-out joint error of the two checkpoints every run keeps (25k and 50k steps), against the same model without
force. Stage 2 is Facet-0's offline refinement, which was run on the flow-matching policy (runs/refine): the critic
(runs/refine/critic_log.json, written by scripts/rl_critic.py as <rl.out_dir>/critic_log.json), the training loss of the
refined policy against plain continued training, and the held-out joint error of both. Stage 3: scripts/s3_report.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.results_figures import COLORS, GRID, INK, MUTED, style  # noqa: E402
from sharpa.config import REPO_ROOT  # noqa: E402

RUNS, OUT = REPO_ROOT / "runs", REPO_ROOT / "reports" / "figures"
SEEDS = (0, 1, 2)


def log(run: str, key: str):
    """Steps and the logged value of `key`, [seeds, rows]."""
    rows = [json.loads((RUNS / run / f"seed{s}" / "train_log.json").read_text()) for s in SEEDS]
    return np.array([r["step"] for r in rows[0]]), np.array([[r[key] for r in seed] for seed in rows])


def mrad(run: str, tag: str = "") -> np.ndarray:
    return np.array([1000 * json.loads((RUNS / run / f"seed{s}" / f"eval_heldout{tag}.json").read_text())["act"]["all"] for s in SEEDS])


def curve(ax, x, y, color, label, width=2.0):
    """Mean over seeds as the line, each seed faint behind it."""
    for seed in y:
        ax.plot(x, seed, color=color, linewidth=0.7, alpha=0.3)
    ax.plot(x, y.mean(0), color=color, linewidth=width, label=label)


def finish(ax, title, xlabel, ylabel, legend="upper right"):
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if legend:
        ax.legend(loc=legend, frameon=False, fontsize=8.5, labelcolor=INK)


def points(ax, x, runs: list[tuple], labels: bool = True):
    """runs: (label, colour, [values per seed at each x])."""
    for label, color, vals in runs:
        m = [v.mean() for v in vals]
        ax.plot(x, m, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
        for xi, v in zip(x, vals):
            ax.scatter([xi] * len(v), v, s=12, color=color, alpha=0.35, linewidths=0, zorder=2)
        if labels:
            for xi, mi in zip(x, m):
                ax.annotate(f"{mi:.2f}", (xi, mi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK)


def stage1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.1), dpi=160, gridspec_kw={"width_ratios": [1.55, 1]})
    for key, label, color in (("loss", "total loss (all terms, weighted)", COLORS["force"]), ("action", "joint targets (L1, normalised units)", COLORS["hero"]),
                              ("force", "fingertip-force forecast", COLORS["no_force"])):
        x, y = log("headline/full", key)
        curve(a, x[1:] / 1000, y[:, 1:], color, label)                     # the row of step 0 is before any update
    a.set_yscale("log")
    a.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0], ["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    a.minorticks_off()
    finish(a, "Stage 1: training loss of the released four-expert model", "training steps, thousands", "training loss, one batch every 200 steps (log scale)")
    points(b, [25, 50], [("without force", COLORS["no_force"], [mrad("headline/no_force", "__step25000"), mrad("headline/no_force")]),
                         ("four experts, with force", COLORS["hero"], [mrad("headline/full", "__step25000"), mrad("headline/full")])])
    b.set_xticks([25, 50], ["25k", "50k"])
    b.set_xlim(15, 60)
    b.margins(y=0.2)
    finish(b, "Held-out joint error of the same runs", "training steps (the two checkpoints every run keeps)", "held-out joint error, mrad (small dots = seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "stage1_curves.png", facecolor="white")
    plt.close(fig)


def stage2():
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.4), dpi=160)
    critic = json.loads((RUNS / "refine" / "critic_log.json").read_text())
    x = np.array([r["step"] for r in critic]) / 1000
    ax[0, 0].plot(x[1:], [r["loss"] for r in critic[1:]], color=COLORS["force"], linewidth=2)
    finish(ax[0, 0], "Stage 2: the critic's training loss", "critic steps, thousands", "cross-entropy to the projected target\n(mean of the last 2,000 steps)", legend=None)
    ax[0, 1].plot(x, [r["heldout_abs_error"] for r in critic], color=COLORS["hero"], linewidth=2, label="critic")
    ax[0, 1].axhline(critic[0]["heldout_abs_error_constant"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="constant guess (mean training return)")
    ax[0, 1].set_ylim(bottom=0)
    finish(ax[0, 1], "The critic on held-out episodes", "critic steps, thousands", "|value − return|", legend="center right")
    for run, key, label, color in (("refine/continued", "action", "continued training (control)", COLORS["no_force"]),
                                   ("refine/refined", "action", "refined: plain branch", COLORS["force"]),
                                   ("refine/refined", "action_tagged", "refined: branch tagged as positive credit", COLORS["hero"])):
        s, y = log(run, key)
        curve(ax[1, 0], s / 1000, y, color, label)
    finish(ax[1, 0], "Stage 2: training loss of the flow policy, 8k more steps", "training steps, thousands", "flow-matching loss on joint targets\n(one batch every 200 steps)")
    res = [("continued\ntraining", mrad("refine/continued"), COLORS["no_force"]), ("refined", mrad("refine/refined"), COLORS["hero"])]
    for k, (label, v, color) in enumerate(res):
        ax[1, 1].scatter([k] * len(v), v, s=14, color=color, alpha=0.4, linewidths=0, zorder=2)
        ax[1, 1].scatter([k], [v.mean()], s=70, color=color, edgecolors="white", linewidths=1.5, zorder=3)
        ax[1, 1].annotate(f"{v.mean():.2f}", (k, v.mean()), textcoords="offset points", xytext=(12, 0), va="center", fontsize=9, color=INK)
    ax[1, 1].set_xticks(range(len(res)), [r[0] for r in res], fontsize=9)
    ax[1, 1].set_xlim(-0.6, len(res) - 0.4)
    ax[1, 1].margins(y=0.6)
    finish(ax[1, 1], "Held-out joint error after stage 2", "", "held-out joint error, mrad (small dots = seeds)", legend=None)
    fig.tight_layout()
    fig.savefig(OUT / "stage2_curves.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    stage1()
    if (RUNS / "refine" / "critic_log.json").exists():
        stage2()
    else:
        print("no runs/refine/critic_log.json: run scripts/rl_critic.py and copy <rl.out_dir>/critic_log.json there")
