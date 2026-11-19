"""Steps 4 and 5 of the brief from finished runs: data scaling, one task against three, input checks.

    python scripts/report_steps.py      -> reports/results/steps_4_5.md / .json, scaling.png

Needs the groups headline, three_tasks, scaling and eraser_force (run_experiments.py), the data report
(data_report.py, for hours per share) and the input-check evaluations of the released model headline/full:
    python scripts/tf_features.py rest=true                                      (once: no-contact tactile latents)
    python scripts/evaluate.py runs/headline/full/seed<k> zero_force=true        (also =tactile, =torque)
    python scripts/evaluate.py runs/headline/full/seed<k> swap_task=<task>
The scaling curve and the one-task models use the two-expert model at 8000 steps (groups scaling, three_tasks, eraser_force).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402
from sharpa.metrics import bootstrap, load_variant  # noqa: E402

RUNS, OUT = REPO_ROOT / "runs", REPO_ROOT / "reports" / "results"
SHARES = ("0.10", "0.25", "0.50", "1.00")
KEYS = ("act", "act_arm", "act_hand", "force", "act_hold", "force_persist", "contact")


def restrict(seeds: list[dict], uids: list[str]) -> list[dict]:
    """The same runs on a subset of held-out episodes, episodes renumbered in the order of `uids`."""
    out = []
    for s in seeds:
        new = {u: i for i, u in enumerate(uids)}
        ep = np.array([new.get(u, -1) for u in s["ep_uid"]])[s["ep"].astype(int)]
        m = ep >= 0
        out.append({k: s[k][m] for k in KEYS if k in s} | {"ep": ep[m].astype(np.float32), "ep_uid": np.array(uids)})
    return out


def cell(r, scale=1000, diff=False):
    if diff:
        return f"{r['diff'] * scale:+.2f} [{r['diff_lo'] * scale:+.2f}, {r['diff_hi'] * scale:+.2f}]" if "diff" in r else "–"
    return f"{r['mean'] * scale:.2f} [{r['lo'] * scale:.2f}, {r['hi'] * scale:.2f}]"


def main():
    cfg = load_config()
    ev, tasks = cfg["eval"], cfg["data"]["tasks"]
    boot = lambda v, key, ref=None: bootstrap(v, key, ref, ev["bootstrap_samples"], ev["ci"])
    summary = json.loads((REPO_ROOT / cfg["report"]["out_dir"] / "summary.json").read_text())
    sizes = {t: summary["tasks"][t]["split"]["train_share_sizes"] for t in tasks}
    hours = {p: sum(sizes[t][p]["hours"] for t in tasks) for p in SHARES}
    episodes = {p: sum(sizes[t][p]["episodes"] for t in tasks) for p in SHARES}
    L, J = ["# Steps 4 and 5: data scaling, one task against three, input checks", ""], {}

    # ---- step 4a: error against hours of training data --------------------------------------
    V = {}
    for p in SHARES:
        for kind in ("full", "no_force"):
            d = RUNS / "three_tasks" / kind if p == "1.00" else RUNS / "scaling" / f"{kind}_{int(float(p) * 100)}"
            V[f"{kind}_{p}"] = load_variant(d)
    scal = {}
    for p in SHARES:                                    # paired difference no_force - full at each share
        pair = {"full": V[f"full_{p}"], "no_force": V[f"no_force_{p}"]}
        scal[p] = {k: boot(pair, k, "full") for k in ("act", "act_arm", "act_hand")}
        scal[p]["force"] = boot({"full": pair["full"]}, "force")
    J["scaling"] = {"hours": hours, "episodes": episodes, "results": scal}
    L += ["## Data scaling (three tasks, all held-out episodes)", "",
          "Two-expert model (`three_tasks/full` and `no_force`). Nested subsets of the training episodes, the same held-out episodes and 8000 steps for every point. "
          "Δ is no-force minus with-force at the same share (positive = force helps).", "",
          "| share | episodes | hours | with force, all (mrad) | without force, all | Δ all | with force, contact | without force, contact | Δ contact | force error (N) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in SHARES:
        a = scal[p]["act"]
        L.append(f"| {int(float(p) * 100)} % | {episodes[p]} | {hours[p]:.2f} | {cell(a['full']['all'])} | {cell(a['no_force']['all'])} | "
                 f"{cell(a['no_force']['all'], diff=True)} | {cell(a['full']['contact'])} | {cell(a['no_force']['contact'])} | "
                 f"{cell(a['no_force']['contact'], diff=True)} | {cell(scal[p]['force']['full']['all'], 1)} |")
    x = np.array([hours[p] for p in SHARES])
    fit = {}
    for kind in ("full", "no_force"):                   # error ~ a * hours^b
        y = np.array([scal[p]["act"][kind]["all"]["mean"] for p in SHARES])
        b, a = np.polyfit(np.log(x), np.log(y), 1)
        fit[kind] = {"exponent": float(b), "per_doubling_percent": float((2 ** b - 1) * 100)}
    J["scaling"]["power_law"] = fit
    L += ["", f"Log-log slope: with force {fit['full']['exponent']:.3f} ({fit['full']['per_doubling_percent']:+.1f} % error per doubling of data), "
              f"without force {fit['no_force']['exponent']:.3f} ({fit['no_force']['per_doubling_percent']:+.1f} %)."]
    plot_scaling(x, scal, OUT / "scaling.png")

    # ---- step 4b: one task against three, on the eraser held-out episodes ---------------------
    one = {"full": load_variant(RUNS / "eraser_force" / "full_residual_force"), "no_force": load_variant(RUNS / "eraser_force" / "no_force")}
    uids = list(one["full"][0]["ep_uid"])
    three = {k: restrict(load_variant(RUNS / "three_tasks" / k), uids) for k in ("full", "no_force")}
    cmp = {"one_task": restrict(one["full"], uids), "three_tasks": three["full"],
           "one_task_no_force": restrict(one["no_force"], uids), "three_tasks_no_force": three["no_force"]}
    extra = RUNS / "three_tasks" / "full_3x_steps"
    if len(list(extra.glob("seed*/eval_heldout.npz"))) == 3:
        cmp["three_tasks_3x_steps"] = restrict(load_variant(extra), uids)
    for name, v in (("three_tasks_four_experts_50k", "full"), ("three_tasks_four_experts_50k_no_force", "no_force")):   # the released model and its counterpart
        if len(list((RUNS / "headline" / v).glob("seed*/eval_heldout.npz"))) == 3:
            cmp[name] = restrict(load_variant(RUNS / "headline" / v), uids)
    res = {k: boot(cmp, k, "one_task") for k in ("act", "act_arm", "act_hand", "force")}
    J["one_vs_three"] = {"episodes": len(uids), "results": res}
    L += ["", f"## One task against three, on the {len(uids)} eraser held-out episodes", "",
          f"One-task models saw {sizes[tasks[0]]['1.00']['hours']:.2f} h, three-task models {hours['1.00']:.2f} h; "
          "two experts and 8000 steps each unless noted. Δ is against the one-task model with force (negative = better).", "",
          "| model | all (mrad) | contact | free | Δ contact | Δ free | force error (N) |", "|---|---|---|---|---|---|---|"]
    for v, r in res["act"].items():
        f = res["force"][v]["all"]
        L.append(f"| {v} | {cell(r['all'])} | {cell(r['contact'])} | {cell(r['free'])} | {cell(r['contact'], diff=True)} | "
                 f"{cell(r['free'], diff=True)} | {'–' if 'no_force' in v else cell(f, 1)} |")

    # ---- step 5: input checks on the released model -------------------------------------------
    rel = RUNS / "headline" / "full"
    base = load_variant(rel)
    task_of = {u: u.split("/")[0] for u in base[0]["ep_uid"]}
    checks = {"zero_force": load_variant(rel, "heldout__zero_force-True"),
              "zero_tactile_only": load_variant(rel, "heldout__zero_force-tactile"),
              "zero_torque_only": load_variant(rel, "heldout__zero_force-torque")}
    for t in tasks:
        checks[f"instruction={t}"] = load_variant(rel, f"heldout__swap_task-{t}")
    checks = {k: v for k, v in checks.items() if len(v) == len(base)}
    J["input_checks"] = {}
    L += ["", "## Input checks (released model `headline/full`: four experts, 50k steps; no retraining)", "",
          "zero_force: fingertip force, joint torque and wrench history set to zero, and the tactile expert given the latents of no-contact fingertip images. "
          "zero_tactile_only keeps the torque, zero_torque_only keeps everything from the fingertips.", "",
          "Rows are held-out episodes of one task; Δ is the paired change against the normal evaluation of the same model.", "",
          "| held-out task | input | action, all (mrad) | Δ all | Δ contact | Δ free | force error (N) | Δ force |", "|---|---|---|---|---|---|---|---|"]
    for t in tasks + ["all"]:
        uids_t = [u for u in base[0]["ep_uid"] if t in ("all", task_of[u])]
        group = {"normal": restrict(base, uids_t)} | {k: restrict(v, uids_t) for k, v in checks.items() if k != f"instruction={t}"}
        ra, rf = boot(group, "act", "normal"), boot(group, "force", "normal")
        J["input_checks"][t] = {"episodes": len(uids_t), "act": ra, "force": rf}
        for k in group:
            if t == "all" and k.startswith("instruction"):
                continue
            a, f = ra[k], rf[k]
            dall = f"{a['all']['diff'] * 1000:+.2f} [{a['all']['diff_lo'] * 1000:+.2f}, {a['all']['diff_hi'] * 1000:+.2f}]" if "diff" in a["all"] else "–"
            df = f"{f['all']['diff']:+.2f} [{f['all']['diff_lo']:+.2f}, {f['all']['diff_hi']:+.2f}]" if "diff" in f["all"] else "–"
            L.append(f"| {t} ({len(uids_t)}) | {k} | {cell(a['all'])} | {dall} | {cell(a['contact'], diff=True)} | "
                     f"{cell(a['free'], diff=True)} | {cell(f['all'], 1)} | {df} |")
    hold = restrict(base, list(base[0]["ep_uid"]))[0]
    L += ["", f"Reference: holding the current joint state gives {hold['act_hold'].mean() * 1000:.1f} mrad, "
              f"keeping the current force gives {hold['force_persist'].mean():.2f} N."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "steps_4_5.md").write_text("\n".join(L) + "\n")
    (OUT / "steps_4_5.json").write_text(json.dumps(J, indent=1))
    print("\n".join(L))


def plot_scaling(x, scal, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True)
    style = {"full": ("#1f6fb5", "o", "with force"), "no_force": ("#c8641b", "s", "without force")}
    for ax, g in zip(axes, ("contact", "free")):
        for kind, (col, mk, name) in style.items():
            r = [scal[p]["act"][kind][g] for p in SHARES]
            y = np.array([v["mean"] for v in r]) * 1000
            lo, hi = np.array([v["lo"] for v in r]) * 1000, np.array([v["hi"] for v in r]) * 1000
            dx = 1.02 if kind == "full" else 0.98        # offset so the intervals do not hide each other
            ax.errorbar(x * dx, y, [y - lo, hi - y], color=col, marker=mk, ms=5, lw=2, capsize=2.5, elinewidth=1, label=name)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h:.2f} h\n{int(float(p) * 100)} %" for h, p in zip(x, SHARES)], fontsize=8)
        ax.minorticks_off()
        ax.set_title(f"{g} frames", fontsize=10)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("held-out action error (mrad)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.supxlabel("hours of training data (three tasks)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
