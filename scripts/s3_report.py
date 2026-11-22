"""Stage 3: learning curve, held-out result and the result table, from the training logs and evaluation files of runs/stage3/.

    python scripts/s3_report.py [run=rl] [second=rl_seed1] [control=control] [split=heldout]
        -> reports/figures/stage3_curve.png, reports/results/stage3.md

Left panel: share of successful episodes among the episodes run without exploration noise (rolling window), separately for
scenes the actor trains in and for the validation scenes it never trains in; dashed lines are the frozen model in the same
scenes (the noise-free episodes before the actor's first update). Right panel: the evaluation on the held-out scenes
(scripts/s3_eval.py), perturbed scenes, with 95 % intervals over scenes.
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

RUNS = REPO_ROOT / "runs" / "stage3"
TITLES = {"rl": "Stage 3 (critic gradient and imitation of successes)", "rl_seed1": "Stage 3, second training seed", "control": "Matched control (imitation of successes only, no critic gradient)",
          "v1_recorded_token": "First design: recorded token, no object pose"}
METRICS = [("success", "Task success", 100, "%"), ("plate_lifted", "Plate picked up", 100, "%"), ("plate_placed", "Plate back on the table at the end", 100, "%"),
           ("plate_dropped", "Plate dropped off the table", 100, "%"), ("wipe_path_smooth_cm", "Wiped path, cm", 1, "")]


def rolling(rows: list[dict], window: int):
    x = np.array([r["index"] for r in rows])
    y = np.array([float(r["success"]) for r in rows])
    c = np.cumsum(np.r_[0.0, y])
    k = np.arange(window, len(y) + 1)
    return x[k - 1], (c[k] - c[k - window]) / window


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (RUNS / name / "log.jsonl").open() if '"error"' not in l]


def evaluation(name: str, split: str) -> dict | None:
    f = RUNS / name / f"eval_{split}.json"
    return json.loads(f.read_text()) if f.exists() else None


def figure(run: str, second: str, control: str, split: str, window: int = 300):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.1), dpi=160, gridspec_kw={"width_ratios": [1.55, 1]})
    lines = [(run, "monitor", "training scenes", COLORS["force"], 2), (run, "validation", "validation scenes (never trained in)", COLORS["hero"], 2)]
    if (RUNS / second / "log.jsonl").exists():
        lines.append((second, "validation", "second seed, validation scenes", COLORS["hero"], 0.9))
    if (RUNS / control / "log.jsonl").exists():
        lines.append((control, "validation", "matched control, validation scenes", COLORS["no_force"], 1.5))
    ends = []
    for name, kind, label, color, width in lines:
        sel = [r for r in load(name) if r["kind"] == kind]
        if len(sel) < window:
            continue
        x, y = rolling(sel, window)
        ax.plot(x / 1000, 100 * y, color=color, linewidth=width, label=label)
        base = [float(r["success"]) for r in sel if r["mean_abs_correction"] == 0]         # before the actor's first update
        if base and name == run:
            ax.axhline(100 * np.mean(base), color=color, linewidth=1, linestyle=(0, (4, 3)))
        ends.append([x[-1] / 1000, 100 * y[-1]])
    order = np.argsort([e[1] for e in ends])                                               # end labels, pushed apart where they would collide
    at = [ends[i][1] for i in order]
    for j in range(1, len(at)):
        at[j] = max(at[j], at[j - 1] + 3.2)
    for i, yy in zip(order, at):
        ax.text(ends[i][0] + 0.25, yy, f"{ends[i][1]:.0f}%", color=INK, fontsize=9, ha="left", va="center")
    ax.set_xlim(0, max(e[0] for e in ends) + 1.8)
    ax.set_ylim(bottom=ax.get_ylim()[0] - 4 * len(lines))                                  # room for the legend under the dashed lines
    ax.set_title("Learning in the simulator", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlabel("training episodes, thousands", color=MUTED, fontsize=9)
    ax.set_ylabel(f"success without exploration noise, %\n(last {window} such episodes; dashed = frozen model)", color=MUTED, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    bars = []
    ev = evaluation(run, split)
    if ev:
        s = ev["summary"]["perturbed"]["success"]
        bars = [("operator's\ncommands", s["operator"], COLORS["other"]), ("frozen\nmodel", s["frozen"], COLORS["force"])]
        cv = evaluation(control, split)
        if cv:
            bars.append(("matched\ncontrol", cv["summary"]["perturbed"]["success"]["stage3"], COLORS["no_force"]))
        bars.append(("with\nstage 3", s["stage3"], COLORS["hero"]))
        sv = evaluation(second, split)
        if sv:
            bars.append(("stage 3,\nsecond seed", sv["summary"]["perturbed"]["success"]["stage3"], COLORS["hero"]))
        for k, (label, v, color) in enumerate(bars):
            bx.bar(k, 100 * v[0], width=0.5, color=color)
            bx.plot([k, k], [100 * v[1], 100 * v[2]], color=INK, linewidth=1.2)
            bx.text(k, 100 * v[2] + 1.5, f"{100 * v[0]:.0f}%", color=INK, fontsize=9.5, ha="center", va="bottom")
        bx.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=9, color=INK)
        bx.set_xlim(-0.6, len(bars) - 0.4)
        bx.set_ylim(0, 100 * max(b[1][2] for b in bars) + 9)
        bx.set_title(f"{ev['scenes']} held-out scenes, {ev['seeds']} perturbations each", loc="left", fontsize=10.5, color=INK, fontweight="bold")
        bx.set_ylabel("task success, % (95 % interval over scenes)", color=MUTED, fontsize=9)
    for a in (ax, bx):
        style(a)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    out = REPO_ROOT / "reports" / "figures" / "stage3_curve.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def table(name: str, split: str) -> str:
    ev = evaluation(name, split)
    def f(v, s, u, diff=False):
        u = " pts" if diff and u == "%" else u
        g = "+" if diff else ""
        return f"{s * v[0]:{g}.1f}{u} [{s * v[1]:{g}.1f}, {s * v[2]:{g}.1f}]"
    out = [f"## {TITLES.get(name, name)}: {ev['scenes']} {'held-out' if split == 'heldout' else split} scenes\n"]
    for cond, s in ev["summary"].items():
        n = "one run per scene" if cond == "as reconstructed" else f"{ev['seeds']} perturbations per scene"
        out += [f"**Scenes {cond}** ({n}; 95 % interval over scenes)\n", "| | Operator's commands | Frozen model | With corrections | Corrections minus frozen (paired) |", "|---|---|---|---|---|"]
        out += [f"| {label} | {f(s[key]['operator'], sc, u)} | {f(s[key]['frozen'], sc, u)} | {f(s[key]['stage3'], sc, u)} | {f(s[key]['stage3 - frozen'], sc, u, True)} |"
                for key, label, sc, u in METRICS if key in s]
        out.append("")
    return "\n".join(out)


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    run, second, control, split = args.get("run", "rl"), args.get("second", "rl_seed1"), args.get("control", "control"), args.get("split", "heldout")
    print("wrote", figure(run, second, control, split))
    names = [n for n in (run, second, control) if evaluation(n, split)]
    if names:
        md = "# Stage 3 in the simulator: the frozen four-expert model with and without learned corrections\n\n" + "\n".join(table(n, split) for n in names)
        (REPO_ROOT / "reports" / "results" / "stage3.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
