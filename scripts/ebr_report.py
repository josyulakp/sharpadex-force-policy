"""Energy-based reasoning on the policy's slot latents: held-out results and test-time checks.

    python scripts/run_experiments.py energy      # trains runs/energy/<variant>/seed*
    python scripts/ebr_report.py                  -> reports/results/energy.md

For every variant: joint and force error after descent, and of the same model before descent (decoded from the start state).
For the variants with an energy, three checks that only an energy-based model allows:
  more steps      the descent is run for fewer / more steps than in training
  best of N       N noise draws, the one with the lowest final energy is kept
  verification    does the final energy tell which frames have a large joint error? (rank correlation; error of the fifth of the
                  frames with the lowest and the highest energy)
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402
from sharpa.engine import evaluate  # noqa: E402

GROUP = REPO_ROOT / "runs" / "energy"
BASE = REPO_ROOT / "runs" / "three_tasks" / "full"


def ev(run: Path, **overrides) -> dict:
    tag = "heldout" + "".join(f"__{k}-{v}" for k, v in overrides.items())
    f = run / f"eval_{tag}.json"
    return json.loads(f.read_text()) if f.exists() else evaluate(run, overrides, tag)


def mean(runs, f):
    return float(np.mean([f(r) for r in runs]))


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    base = sorted(BASE.glob("seed*"))
    lines = ["# Energy-based reasoning on the slot latents (three tasks, held-out episodes, mean of seeds)", "",
             "Joint error in mrad, force error in N. \"Before descent\" is the same model decoded from its start state.", "",
             "| Model | joint | in contact | free | force | joint before descent | force before descent | per seed (joint) |", "|---|---|---|---|---|---|---|---|",
             "| two experts, no energy (`three_tasks/full`) | " + " | ".join(f"{1000 * mean(base, lambda r: ev(r)['act'][k]):.2f}" for k in ("all", "contact", "free"))
             + f" | {mean(base, lambda r: ev(r)['force']['all']):.4f} | | | " + ", ".join(f"{1000 * ev(r)['act']['all']:.2f}" for r in base) + " |"]
    variants = {v.name: sorted(v.glob("seed*")) for v in sorted(GROUP.iterdir())}
    for name, runs in variants.items():
        s = [ev(r) for r in runs]
        start = "act_start" in s[0]
        lines.append(f"| `{name}` | " + " | ".join(f"{1000 * np.mean([x['act'][k] for x in s]):.2f}" for k in ("all", "contact", "free"))
                     + f" | {np.mean([x['force']['all'] for x in s]):.4f} | "
                     + (f"{1000 * np.mean([x['act_start']['all'] for x in s]):.2f} | {np.mean([x['force_start']['all'] for x in s]):.4f}" if start else " | ")
                     + " | " + ", ".join(f"{1000 * x['act']['all']:.2f}" for x in s) + " |")
    for name, runs in variants.items():
        cfg = yaml.safe_load((runs[0] / "config.yaml").read_text())["model"]["energy"]
        lines += ["", f"## `{name}`: test-time checks", "", "| | joint (mrad) | force (N) |", "|---|---|---|"]
        row = lambda label, **o: lines.append(f"| {label} | {1000 * mean(runs, lambda r: ev(r, **o)['act']['all']):.2f} | {mean(runs, lambda r: ev(r, **o)['force']['all']):.4f} |")  # noqa: E731
        for n in (1, 3, 6, 12, 24):
            row(f"{n} steps" + (" (as trained)" if n == cfg["steps"] else ""), **({} if n == cfg["steps"] else {"energy_steps": n}))
        row("as trained, no noise", energy_noise=0.0)
        if cfg["update"] == "gradient":
            for n in (4, 8):
                row(f"best of {n} by lowest energy", energy_samples=n)
            cor, lo, hi = [], [], []
            for r in runs:
                a = np.load(r / "eval_heldout.npz")
                e, err = a["energy"], a["act"]
                q = np.quantile(e, [0.2, 0.8])
                cor.append(rank_corr(e, err)); lo.append(1000 * err[e <= q[0]].mean()); hi.append(1000 * err[e >= q[1]].mean())
            lines += ["", f"Verification: rank correlation between final energy and joint error {np.mean(cor):+.3f}; joint error of the fifth of frames "
                      f"with the lowest energy {np.mean(lo):.2f} mrad, with the highest {np.mean(hi):.2f} mrad."]
    out = REPO_ROOT / "reports" / "results" / "energy.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
