"""Flow-matching policy against direct regression at 8k / 16k / 24k / 25k training steps.

    python scripts/report_flow.py      -> reports/results/flow_steps.md

Needs the group `flow` (run_experiments.py flow), aggregated per checkpoint:
    python scripts/aggregate.py flow heldout__step8000      (also 16000, 24000, and no tag for 25k)
and the zero-noise evaluation of the final checkpoints:
    python scripts/evaluate.py runs/flow/<variant>/seed<k> flow_noise_scale=0
    python scripts/aggregate.py flow heldout__flow_noise_scale-0
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "reports" / "results"
TAGS = {"8,000": "__heldout__step8000", "16,000": "__heldout__step16000", "24,000": "__heldout__step24000",
        "25,000": "", "25,000, zero starting noise": "__heldout__flow_noise_scale-0"}
NAMES = {"full": "flow, with force", "no_force": "flow, without force",
         "full_wrench_expert": "flow, with force + wrench expert", "regression_25k": "regression, with force"}


def cell(r, k, scale, nd):
    return f"{r[k]['mean'] * scale:.{nd}f} [{r[k]['lo'] * scale:.{nd}f}, {r[k]['hi'] * scale:.{nd}f}]"


def diff(r, k, scale, nd):
    return f"{r[k]['diff'] * scale:+.{nd}f} [{r[k]['diff_lo'] * scale:+.{nd}f}, {r[k]['diff_hi'] * scale:+.{nd}f}]" if "diff" in r[k] else "–"


if __name__ == "__main__":
    res = {s: json.loads((OUT / f"flow{t}.json").read_text()) for s, t in TAGS.items()}
    L = ["# Flow-matching policy: three tasks, 115 held-out episodes, 3 seeds", "",
         "Intervals: 95 % bootstrap over episodes and seeds. Δ is the paired difference to `flow, with force` at the same step.",
         "Flow rows draw one chunk from noise (10 Euler steps, fixed seed) unless the row says zero starting noise.", ""]
    for title, key, scale, nd in (("Action error, all joints (mrad)", "act", 1000, 2), ("Force error (N)", "force", 1, 2)):
        L += [f"## {title}", "", "| steps | model | all | contact | free | Δ contact | Δ free |", "|---|---|---|---|---|---|---|"]
        for s, r in res.items():
            for v, n in NAMES.items():
                if v in r[key]:
                    x = r[key][v]
                    L.append(f"| {s} | {n} | {cell(x, 'all', scale, nd)} | {cell(x, 'contact', scale, nd)} | "
                             f"{cell(x, 'free', scale, nd)} | {diff(x, 'contact', scale, nd)} | {diff(x, 'free', scale, nd)} |")
        L.append("")
    (OUT / "flow_steps.md").write_text("\n".join(L))
    print("\n".join(L))
