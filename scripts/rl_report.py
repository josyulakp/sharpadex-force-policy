"""Stage 2 (offline refinement) results on the held-out episodes.

    python scripts/rl_report.py [group] [overrides]      -> reports/results/<group>.md / .json   (group: refine or refine_luna)

Needs: rl_reward.py, rl_critic.py, `run_experiments.py refine`, and for every seed of refine/refined
    python scripts/evaluate.py runs/refine/refined/seed<k> guidance=<g>      for g in 0, 2, 3   (guidance 1 is the default evaluation)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402
from sharpa.metrics import bootstrap, load_variant  # noqa: E402

GROUP = sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else "refine"
RUNS, OUT = REPO_ROOT / "runs" / GROUP, REPO_ROOT / "reports" / "results"
ROWS = {"continued (no refinement)": ("../refine/continued", "heldout"), "refined, guidance 0 (plain branch)": ("refined", "heldout__guidance-0"),
        "refined, guidance 1 (tagged branch)": ("refined", "heldout"), "refined, guidance 2": ("refined", "heldout__guidance-2"),
        "refined, guidance 3": ("refined", "heldout__guidance-3")}
REF = "continued (no refinement)"

if __name__ == "__main__":
    cfg = load_config(overrides=[a for a in sys.argv[1:] if "=" in a])
    variants = {n: load_variant(RUNS / v, tag) for n, (v, tag) in ROWS.items()}
    variants = {n: s for n, s in variants.items() if s}
    label = variants["refined, guidance 1 (tagged branch)"][0]["adv_label"]
    by_label = {n: [dict(s, contact=label) for s in seeds] for n, seeds in variants.items()}   # "contact" group = positive credit
    for seeds in variants.values():
        for s in seeds:
            s["speed_ratio"] = s["speed"] / max(float(s["speed_recorded"].mean()), 1e-9)       # 1 = as fast as the operator on average
    kw = dict(n=cfg["eval"]["bootstrap_samples"], ci=cfg["eval"]["ci"])
    res = {"act": bootstrap(variants, "act", REF, **kw), "act_by_label": bootstrap(by_label, "act", REF, **kw),
           "force": bootstrap(variants, "force", REF, **kw), "speed_ratio": bootstrap(variants, "speed_ratio", REF, **kw),
           "force_peak": bootstrap(variants, "force_peak", REF, **kw),
           "critic": json.loads((Path(cfg["rl"]["out_dir"]) / "credit.json").read_text())}

    def c(x, s, nd):
        return f"{x['mean'] * s:.{nd}f} [{x['lo'] * s:.{nd}f}, {x['hi'] * s:.{nd}f}]"

    def d(x, s, nd):
        return f"{x['diff'] * s:+.{nd}f} [{x['diff_lo'] * s:+.{nd}f}, {x['diff_hi'] * s:+.{nd}f}]" if "diff" in x else "–"

    k = res["critic"]
    L = [f"# Stage 2: offline refinement of the flow policy (Facet-0 recipe, subgoal progress: {cfg['rl']['reward']['judge']})", "",
         f"Critic on held-out episodes: |value - return| {k['heldout_abs_error']:.3f} (constant guess {k['heldout_abs_error_constant']:.3f}), "
         f"correlation {k['heldout_correlation']:.3f}. Positive-credit frames: {k['train']['positive_rate']:.1%} of training, "
         f"{k['heldout']['positive_rate']:.1%} of held-out frames (same thresholds).", "",
         "Three seeds, 115 held-out episodes; 95 % bootstrap over episodes and seeds; Δ is paired against `continued`.", ""]
    for title, key, groups, s, nd in (
            ("Action error (mrad)", "act", ("all", "contact", "free"), 1000, 2),
            ("Action error by credit label (mrad)", "act_by_label", ("contact", "free"), 1000, 2),
            ("Force error (N)", "force", ("all", "contact", "free"), 1, 2),
            ("Commanded speed relative to the operator's mean (1 = same)", "speed_ratio", ("all", "contact", "free"), 1, 3),
            ("Predicted peak fingertip force in the chunk (N)", "force_peak", ("all", "contact", "free"), 1, 2)):
        names = ("positive credit", "other frames") if key == "act_by_label" else groups
        L += [f"## {title}", "", "| model | " + " | ".join(names) + " | " + " | ".join(f"Δ {n}" for n in names) + " |",
              "|---|" + "---|" * (2 * len(groups))]
        for n in variants:
            x = res[key][n]
            L.append(f"| {n} | " + " | ".join(c(x[g], s, nd) for g in groups) + " | " + " | ".join(d(x[g], s, nd) for g in groups) + " |")
        L.append("")
    (OUT / f"{GROUP}.md").write_text("\n".join(L))
    (OUT / f"{GROUP}.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))
