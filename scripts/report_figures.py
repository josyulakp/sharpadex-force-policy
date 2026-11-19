"""Numbers and figures for the two best approaches: the four-expert model (released) and the two-expert model.

    python scripts/report_figures.py      -> reports/report/{numbers.json, model.png, results.png}

numbers.json   every table and plot number, computed from the per-frame evaluation files of the finished runs
               (runs/<group>/<variant>/seed*/eval_*.npz; they come with `hf download`, see README.md)
model.png      the model diagram
results.png    (a) what force adds, (b) held-out error against hours of training data, (c) input checks
Needs the groups headline, three_tasks, scaling, scaling_four_experts, tactile_expert, the input checks of headline/full and
headline/two_experts (scripts/evaluate.py <run> zero_force=... / swap_task=...) and scripts/ps_report.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402
from sharpa.metrics import bootstrap, load_variant  # noqa: E402

RUNS, OUT = REPO_ROOT / "runs", REPO_ROOT / "reports" / "report"
INK, MUTED, GRID, TINT = "#16212c", "#5b6772", "#dfe4e9", "#eef4fb"
LINE = "#6b7682"
FOUR, TWO, NOFORCE = "#1f6fd1", "#3d4852", "#c8631f"          # four experts / two experts / without force
SHARES = ("0.10", "0.25", "0.50", "1.00")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": GRID,
                     "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": MUTED})


# ---------------------------------------------------------------------------------------------- numbers
def numbers() -> dict:
    cfg = load_config()
    ev, tasks = cfg["eval"], cfg["data"]["tasks"]
    boot = lambda v, key, ref=None: bootstrap(v, key, ref, ev["bootstrap_samples"], ev["ci"])   # noqa: E731
    N = {}
    H = {k: load_variant(RUNS / "headline" / k) for k in ("full", "no_force", "two_experts", "no_force_two_experts")}
    assert all(len(v) == 3 for v in H.values()), {k: len(v) for k, v in H.items()}
    N["act"], N["force"] = boot(H, "act", "full"), boot({k: H[k] for k in ("full", "two_experts")}, "force", "full")
    N["gain_two"] = boot({"two_experts": H["two_experts"], "no_force_two_experts": H["no_force_two_experts"]}, "act", "two_experts")["no_force_two_experts"]
    N["gain_four"] = N["act"]["no_force"]
    N["arm"] = {"four": boot({k: H[k] for k in ("full", "no_force")}, "act_arm", "full")["no_force"],
                "two": boot({k: H[k] for k in ("two_experts", "no_force_two_experts")}, "act_arm", "two_experts")["no_force_two_experts"]}
    one = H["full"][0]
    N["hold"] = {g: float(one["act_hold"][m].mean()) for g, m in (("all", slice(None)), ("contact", one["contact"] > 0.5), ("free", one["contact"] < 0.5))}
    N["persist"] = {g: float(one["force_persist"][m].mean()) for g, m in (("all", slice(None)), ("contact", one["contact"] > 0.5), ("free", one["contact"] < 0.5))}
    N["per_seed"] = {k: [float(s["act"].mean()) for s in v] for k, v in H.items()}
    from sharpa.engine import load_checkpoint
    N["weights"] = {k: sum(p.numel() for p in load_checkpoint(RUNS / "headline" / k / "seed0")[0].parameters() if p.requires_grad) for k in H}

    # data scaling: (model, with force?) -> run folder per share
    summary = json.loads((REPO_ROOT / cfg["report"]["out_dir"] / "summary.json").read_text())
    N["hours"] = [sum(summary["tasks"][t]["split"]["train_share_sizes"][p]["hours"] for t in tasks) for p in SHARES]
    src = {"four": lambda p: RUNS / "tactile_expert" / "four_experts" if p == "1.00" else RUNS / "scaling_four_experts" / f"full_{int(float(p) * 100)}",
           "four_no_force": lambda p: RUNS / "scaling_four_experts" / f"no_force_{int(float(p) * 100)}",
           "two": lambda p: RUNS / "three_tasks" / "full" if p == "1.00" else RUNS / "scaling" / f"full_{int(float(p) * 100)}",
           "two_no_force": lambda p: RUNS / "three_tasks" / "no_force" if p == "1.00" else RUNS / "scaling" / f"no_force_{int(float(p) * 100)}"}
    N["scaling"], N["scaling_gain"], N["slope"] = {}, {}, {}
    for m in ("four", "two"):
        per = [{"full": load_variant(src[m](p)), "no_force": load_variant(src[m + "_no_force"](p))} for p in SHARES]
        assert all(len(v) == 3 for d in per for v in d.values()), m
        res = [boot(d, "act", "full") for d in per]
        N["scaling"][m] = [r["full"] for r in res]
        N["scaling"][m + "_no_force"] = [r["no_force"] for r in res]
        N["scaling_gain"][m] = [r["no_force"] for r in res]
        for k in (m, m + "_no_force"):
            N["slope"][k] = float(np.polyfit(np.log2(N["hours"]), np.log2([r["all"]["mean"] for r in N["scaling"][k]]), 1)[0])

    # input checks (no retraining): change against the normal evaluation of the same checkpoints
    N["checks"] = {}
    for m, v in (("four", "full"), ("two", "two_experts")):
        base = H[v]
        group = {"normal": base, "zero_force": load_variant(RUNS / "headline" / v, "heldout__zero_force-True"),
                 "zero_torque": load_variant(RUNS / "headline" / v, "heldout__zero_force-torque"),
                 "zero_fingertip": load_variant(RUNS / "headline" / v, "heldout__zero_force-tactile")}
        swaps = [load_variant(RUNS / "headline" / v, f"heldout__swap_task-{t}") for t in tasks]
        assert all(len(g) == 3 for g in [*group.values(), *swaps]), m
        swapped = []
        for k, s in enumerate(base):                    # every frame under the instructions of the two other tasks, averaged
            task = s["task"].astype(int)
            a = np.nanmean(np.stack([np.where(task == i, np.nan, swaps[i][k]["act"]) for i in range(len(tasks))]), 0)
            f = np.nanmean(np.stack([np.where(task == i, np.nan, swaps[i][k]["force"]) for i in range(len(tasks))]), 0)
            swapped.append(s | {"act": a.astype(np.float32), "force": f.astype(np.float32)})
        group["swapped_instruction"] = swapped
        N["checks"][m] = {"act": boot(group, "act", "normal"), "force": boot(group, "force", "normal")}
    wm = json.loads((REPO_ROOT / "reports" / "results" / "world_model_policy.json").read_text())["headline/full"]     # scripts/ps_report.py
    N["world_model"] = {t: [round(v) for v in wm[t]["command_gap_mrad"]["mean"][:2]] for t in tasks}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1))
    return N


# ---------------------------------------------------------------------------------------------- figures
def model_figure(path: Path):
    """The one diagram: both approaches share everything except the two added experts."""
    fig = plt.figure(figsize=(182 / 25.4, 52 / 25.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 182), ax.set_ylim(-1, 51), ax.axis("off")

    def box(x, y, w, h, title, sub="", kind="plain", size=6.6):
        face, edge, ls = {"plain": ("#ffffff", "#9aa5b1", "-"), "force": (TINT, FOUR, "-"), "added": (TINT, FOUR, (0, (3, 1.6))),
                          "frozen": ("#f3f4f6", "#9aa5b1", "-")}[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1", fc=face, ec=edge, lw=0.8, ls=ls))
        if sub:
            n = sub.count("\n") + 1                    # title above, sub lines below, the block centred in the box
            step = 0.36 * (size - 1.1) * 1.25           # mm per sub line
            top = y + h / 2 + (n * step + 2.6) / 2
            ax.text(x + w / 2, top - 1.1, title, ha="center", va="center", fontsize=size, fontweight="bold", color=INK)
            ax.text(x + w / 2, top - 2.6 - n * step / 2, sub, ha="center", va="center", fontsize=size - 1.1, color=MUTED, linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size, fontweight="bold", color=INK)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=LINE, shrinkA=0, shrinkB=0))

    ins = [("3 cameras", "frozen DINOv2 ViT-S/14, 51 tokens", "frozen"), ("Joint positions", "6 frames, 58 joints", "plain"),
           ("Fingertip force and torque", "6 frames, 10 fingertips x 6", "force"), ("Joint torque", "6 frames, 58 joints", "force"),
           ("Task", "1 token", "plain")]
    for k, (t, s, kind) in enumerate(ins):
        y = 41.5 - k * 8.6
        box(1, y, 40, 7.2, t, s, kind)
        arrow(41, y + 3.6, 47, 25)
    box(47, 14, 22, 22, "Encoder", "4 layers\nwidth 256", size=7)
    arrow(69, 25, 75, 25)
    box(75, 14, 25, 22, "Decoder", "3 layers\n16 chunk slots\n3 decision slots", size=7)
    ex = [("Plain expert 1", "MLP on each chunk slot", "plain"), ("Plain expert 2", "MLP on each chunk slot", "plain"),
          ("Wrench expert", "+ 10 frames of fingertip wrench", "added"), ("Tactile expert", "+ 16 tactile world-model latents", "added")]
    for k, (t, s, kind) in enumerate(ex):
        y = 41.5 - k * 8.6
        box(108, y, 37, 7.2, t, s, kind)
        arrow(100, 27, 108, y + 3.6)
        ax.plot([145, 147.6], [y + 3.6, y + 3.6], color=LINE, lw=0.6)
    box(108, 7.1, 37, 7.2, "Gate", "one weight per expert", "force")
    arrow(100, 21, 108, 10.7)
    ax.plot([145, 147.6, 147.6], [10.7, 10.7, 45.1], color=LINE, lw=0.6)               # the gate mixes the experts' proposals
    arrow(147.6, 36, 151, 36)
    ax.text(149.4, 38.0, "mix", ha="center", va="center", fontsize=5.4, color=MUTED)
    box(151, 31, 30, 10, "Action", "16 joint targets\n58 joints, 0.53 s", size=7)
    box(151, 16.5, 30, 10, "Force forecast", "30 fingertip components\nchange from force now", "force", size=7)
    box(151, 2, 30, 10, "Decisions", "contact soon, phase,\nforce level", "force", size=7)
    ax.plot([88, 88, 149.4], [14, 4, 4], color=LINE, lw=0.6)                            # decoder slots -> force forecast and decisions
    arrow(149.4, 4, 151, 4)
    ax.plot([149.4, 149.4], [4, 18.5], color=LINE, lw=0.6)
    arrow(149.4, 18.5, 151, 18.5)
    # legend
    for y, kind, label in ((46.2, "force", "a way force is used"), (41.8, "added", "only in the four-expert model")):
        face, edge, ls = (TINT, FOUR, "-") if kind == "force" else (TINT, FOUR, (0, (3, 1.6)))
        ax.add_patch(FancyBboxPatch((49, y), 4, 2.6, boxstyle="round,pad=0,rounding_size=0.6", fc=face, ec=edge, lw=0.8, ls=ls))
        ax.text(54.4, y + 1.3, label, va="center", fontsize=6, color=MUTED)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelsize=6.5)
    ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def results_figure(N: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(182 / 25.4, 56 / 25.4), dpi=300, gridspec_kw={"width_ratios": [0.9, 1.45, 1.05]})
    # (a) what force adds: without force minus with force, paired
    ax = axes[0]
    rows = [("all frames", "all"), ("in contact", "contact"), ("free motion", "free")]
    for k, (label, g) in enumerate(rows):
        for dy, key, col, mk in ((0.16, "gain_four", FOUR, "o"), (-0.16, "gain_two", TWO, "s")):
            r = N[key][g]
            y, d = len(rows) - 1 - k + dy, 1000 * r["diff"]
            ax.plot([1000 * r["diff_lo"], 1000 * r["diff_hi"]], [y, y], color=col, lw=1.3, solid_capstyle="round")
            ax.plot([d], [y], mk, color=col, ms=4.6, mec="white", mew=0.7)
            ax.text(1000 * r["diff_hi"] + 0.06, y, f"{d:.2f}", va="center", fontsize=6.3, color=INK)
    ax.axvline(0, color="#9aa5b1", lw=0.7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) + 0.25)
    ends = [1000 * N[k][g][e] for k in ("gain_four", "gain_two") for g in ("all", "contact", "free") for e in ("diff_lo", "diff_hi")]
    ax.set_xlim(min(-0.15, min(ends) - 0.1), max(ends) + 0.5)
    ax.set_xlabel("joint error removed by force (mrad)", fontsize=6.5)
    ax.set_title("a  What force adds", loc="left", fontsize=7.6, fontweight="bold", color=INK)
    clean(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.plot([], [], "o", color=FOUR, ms=4, label="four experts")
    ax.plot([], [], "s", color=TWO, ms=4, label="two experts")
    ax.legend(frameon=False, fontsize=6.2, loc="upper center", ncol=2, handletextpad=0.1, columnspacing=1.0, borderaxespad=0.0)

    # (b) data need
    ax = axes[1]
    x = np.array(N["hours"])
    series = (("four", FOUR, "o", "-", "four experts", 1.03), ("two", TWO, "s", "-", "two experts", 1.01),
              ("four_no_force", NOFORCE, "o", "-", "four plain, no force", 0.99), ("two_no_force", NOFORCE, "s", (0, (2.5, 1.5)), "two plain, no force", 0.97))
    for key, col, mk, ls, label, dx in series:
        r = N["scaling"][key]
        y = np.array([v["all"]["mean"] for v in r]) * 1000
        lo, hi = np.array([v["all"]["lo"] for v in r]) * 1000, np.array([v["all"]["hi"] for v in r]) * 1000
        ax.errorbar(x * dx, y, [y - lo, hi - y], color=col, marker=mk, ms=3.6, lw=1.3, ls=ls, capsize=0, elinewidth=0.6, mec="white", mew=0.5,
                    label=label, zorder=3 if key == "four" else 2)
    for key, col, mk, dy in (("two_experts", TWO, "s", 0.22), ("full", FOUR, "o", -0.22)):
        v = 1000 * N["act"][key]["all"]["mean"]
        ax.plot([x[-1] * 1.3], [v], mk, mfc="white", mec=col, ms=4.2, mew=1.1, zorder=4)
        ax.text(x[-1] * 1.42, v + dy, f"{v:.2f}", va="center", fontsize=6.2, color=INK)
    ax.plot([], [], "o", mfc="white", mec=MUTED, ms=4, mew=1.0, label="open: 50k steps")
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{h:.2f}" for h in x])
    ax.minorticks_off()
    ax.set_xlim(x[0] * 0.85, x[-1] * 2.0)
    ax.set_xlabel("hours of training data (10, 25, 50, 100 % of the episodes)", fontsize=6.5)
    ax.set_ylabel("held-out joint error (mrad)", fontsize=6.5)
    ax.set_title("b  Data need, 8k steps per point", loc="left", fontsize=7.6, fontweight="bold", color=INK)
    clean(ax)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right", handlelength=2.0, borderaxespad=0.0, labelspacing=0.25)

    # (c) input checks
    ax = axes[2]
    rows = [("all force inputs zero", "zero_force"), ("joint torque zero", "zero_torque"), ("fingertip force zero", "zero_fingertip"),
            ("instruction of another task", "swapped_instruction")]
    top = bottom = 0
    for k, (label, key) in enumerate(rows):
        for dy, m, col in ((0.19, "four", FOUR), (-0.19, "two", TWO)):
            r = N["checks"][m]["act"][key]["all"]
            y, d = len(rows) - 1 - k + dy, 1000 * r["diff"]
            ax.barh(y, d, height=0.32, color=col, zorder=2)
            ax.plot([1000 * r["diff_lo"], 1000 * r["diff_hi"]], [y, y], color=INK, lw=0.6, zorder=3)
            ax.text(max(1000 * r["diff_hi"], 0) + 0.4, y, f"+{d:.1f}" if d >= 0 else f"{d:.1f}", va="center", fontsize=6.3, color=INK)
            top, bottom = max(top, 1000 * r["diff_hi"]), min(bottom, 1000 * r["diff_lo"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(min(0, bottom * 1.1), top * 1.32)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("rise in joint error (mrad)", fontsize=6.5)
    ax.set_title("c  Input checks", loc="left", fontsize=7.6, fontweight="bold", color=INK)
    clean(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.barh([np.nan], [np.nan], color=FOUR, label="four experts")
    ax.barh([np.nan], [np.nan], color=TWO, label="two experts")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", handlelength=1.0, handletextpad=0.4, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    N = numbers()
    model_figure(OUT / "model.png")
    results_figure(N, OUT / "results.png")
    print("wrote", OUT / "numbers.json", "model.png", "results.png")
