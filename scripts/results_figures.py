"""Figures of RESULTS.md, drawn from the evaluation files of the finished runs.

    python scripts/results_figures.py      -> reports/figures/{paths.png, budget.png, primesim_<task>.png}

paths.png      held-out joint error of every path that was tried on the three tasks, in the order it was tried
budget.png     held-out joint error against training steps for the main model lines
primesim_*     head-camera frames of one held-out episode: the recording above, the policy acting inside the video world model below
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402

RUNS, OUT = REPO_ROOT / "runs", REPO_ROOT / "reports" / "figures"
INK, MUTED, GRID = "#1f2933", "#5f6b76", "#e3e7eb"
COLORS = {"hero": "#1f6fd1", "force": "#3d4852", "no_force": "#c8631f", "other": "#9aa5b1"}
LABELS = {"hero": "four experts, 50k steps (released model)", "force": "regression model with force", "no_force": "same model without force",
          "other": "other heads, losses and refinements"}

# (label, run, evaluation tag, kind), in the order the paths were tried
PATHS = [("Without force · two experts · 8k", "three_tasks/no_force", "heldout", "no_force"),
         ("With force · two experts · 8k", "three_tasks/full", "heldout", "force"),
         ("Flow-matching head · 25k", "flow/full", "heldout", "other"),
         ("Flow + offline refinement · 25k + 8k", "refine/refined", "heldout", "other"),
         ("Flow, twice the width (27 M weights) · 25k", "flow_scaled/full_wide", "heldout", "other"),
         ("Energy-based refinement of the latents · 8k", "energy/descent", "heldout", "other"),
         ("Three experts: + wrench expert · 8k", "three_tasks/full_wrench_expert", "heldout", "force"),
         ("Four experts: + tactile world-model expert · 8k", "tactile_expert/four_experts", "heldout", "force"),
         ("Two experts · 25k", "flow/regression_25k", "heldout", "force"),
         ("Four experts · 25k", "ht_loss/l1_25k", "heldout", "force"),
         ("Four experts, heavy-tailed loss · 25k", "ht_loss/ht_25k", "heldout", "other"),
         ("Four experts, heavy-tailed loss · 50k", "ht_loss/ht_50k", "heldout", "other"),
         ("Two experts · 50k", "headline/two_experts", "heldout", "force"),
         ("Without force · four plain experts · 50k", "headline/no_force", "heldout", "no_force"),
         ("Four experts · 50k", "headline/full", "heldout", "hero")]

# (line label, kind, [(steps, run, tag)]): only end-of-schedule points, so every point had its full learning-rate decay
LINES = [("without force (two plain experts)", "no_force", [(8000, "three_tasks/no_force", "heldout"), (50000, "headline/no_force_two_experts", "heldout")]),
         ("two experts", "force", [(8000, "three_tasks/full", "heldout"), (25000, "flow/regression_25k", "heldout"), (50000, "headline/two_experts", "heldout")]),
         ("four experts", "hero", [(8000, "tactile_expert/four_experts", "heldout"), (25000, "ht_loss/l1_25k", "heldout"), (50000, "headline/full", "heldout")]),
         ("four experts, heavy-tailed loss", "other", [(25000, "ht_loss/ht_25k", "heldout"), (50000, "ht_loss/ht_50k", "heldout")])]


def mrad(run: str, tag: str) -> tuple[float, list[float]]:
    v = [1000 * json.loads(p.read_text())["act"]["all"] for p in sorted((RUNS / run).glob(f"seed*/eval_{tag}.json"))]
    return float(np.mean(v)), v


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def paths():
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=160)
    y = np.arange(len(PATHS))[::-1]
    for (label, run, tag, kind), yi in zip(PATHS, y):
        m, seeds = mrad(run, tag)
        ax.plot([13.4, m], [yi, yi], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter(seeds, [yi] * len(seeds), s=14, color=COLORS[kind], alpha=0.35, zorder=2, linewidths=0)
        ax.scatter([m], [yi], s=70 if kind == "hero" else 46, color=COLORS[kind], zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(m + 0.12, yi, f"{m:.2f}", va="center", ha="left", fontsize=9, color=INK, fontweight="bold" if kind == "hero" else "normal")
    ax.set_yticks(y, [p[0] for p in PATHS])
    ax.set_xlim(13.4, 19.2)
    ax.set_xlabel("held-out joint error, mrad (lower is better; small dots = seeds)", color=MUTED, fontsize=9)
    style(ax)
    for t, p in zip(ax.get_yticklabels(), PATHS):
        t.set_color(INK)
        t.set_fontweight("bold" if p[3] == "hero" else "normal")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[k], markersize=7, label=LABELS[k]) for k in ("hero", "force", "no_force", "other")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title("Every path tried on the three tasks, in the order it was tried", loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "paths.png", facecolor="white")
    plt.close(fig)


def budget():
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for label, kind, pts in LINES:
        x = [p[0] for p in pts]
        v = [mrad(p[1], p[2])[0] for p in pts]
        ax.plot(x, v, color=COLORS[kind], linewidth=2, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, zorder=3 if kind == "hero" else 2)
        dy = {"other": 0.07, "no_force": -0.07, "force": 0.05}.get(kind, 0.0)
        ax.text(x[-1] * 1.07, v[-1] + dy, f"{label}  {v[-1]:.2f}", va="center", ha="left", fontsize=8.5, color=INK, fontweight="bold" if kind == "hero" else "normal")
    ax.set_xscale("log")
    ax.set_xticks([8000, 25000, 50000], ["8k", "25k", "50k"])
    ax.minorticks_off()
    ax.set_xlim(6800, 190000)
    ax.set_xlabel("training steps (each point is a full run with its own schedule)", color=MUTED, fontsize=9)
    ax.set_ylabel("held-out joint error, mrad", color=MUTED, fontsize=9)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_title("Held-out joint error against training steps", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "budget.png", facecolor="white")
    plt.close(fig)


def primesim():
    import imageio.v3 as iio
    for task in ("clean_plate_with_eraser", "insert_plug_into_socket", "remove_block_from_box"):
        f = REPO_ROOT / "reports" / "world_model_primesim" / f"policy_headline_full_seed0_sharpa_three_tasks_{task}" / "episode_0.mp4"
        if not f.exists():
            continue
        v = iio.imread(f)                                                 # [T, 768, 512, 3] at 10 fps: recording | policy in the video model; head camera on top
        ts = np.linspace(0, len(v) - 1, 5).round().astype(int)
        fig, axes = plt.subplots(2, 5, figsize=(10, 4.25), dpi=130)
        for col, t in enumerate(ts):
            for row, x0 in enumerate((0, 256)):
                ax = axes[row, col]
                ax.imshow(v[t, :256, x0:x0 + 256])
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if row == 0:
                    ax.set_title(f"{t / 10:.1f} s", fontsize=9, color=MUTED)
            axes[0, 0].set_ylabel("recording", fontsize=9, color=INK)
            axes[1, 0].set_ylabel("policy in the\nvideo world model", fontsize=9, color=INK)
        fig.suptitle(f"{task}: held-out episode, head camera", x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.3, h_pad=0.3)
        fig.savefig(OUT / f"primesim_{task}.png", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    paths(), budget(), primesim()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
