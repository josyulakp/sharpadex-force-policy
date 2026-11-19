"""PrimeSim world model, step 8: the policy-in-the-video-model rollouts of every seed in one table.

    python scripts/ps_report.py [runs=headline/full model=sharpa_three_tasks]     (runs: comma-separated)     -> reports/results/world_model_policy.md / .json

Reads reports/world_model_primesim/policy_<run>_seed<k>_<model>_<task>/metrics.json (scripts/ps_policy.py). Per task and 3.2 s
block: the distance between the policy's commands and the operator's at the same time (mrad, all joints / arms) and the gap of
the force forecast to the recorded force (N); mean over episodes and seeds, with the per-seed means in brackets.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
RUNS, MODEL = ARGS.get("runs", "headline/full").split(","), ARGS.get("model", "sharpa_three_tasks")
SRC, OUT = REPO_ROOT / "reports" / "world_model_primesim", REPO_ROOT / "reports" / "results" / "world_model_policy.md"
ROWS = (("all joints (mrad)", "command_gap_mrad", "{:.0f}"), ("arms (mrad)", "arm_gap_mrad", "{:.0f}"), ("force forecast gap (N)", "force_gap_n", "{:.2f}"))


def main():
    tasks = load_config()["data"]["tasks"]
    L = [f"# Policies inside the video world model (PrimeSim `{MODEL}`), every seed", "",
         "Distance between the policy's commands and the operator's at the same time in the recording (mrad). It measures divergence, "
         "not error: after a few seconds the two need not be doing the same thing. A policy with force inputs gets them replayed from "
         "the recording. Each cell: mean over episodes and seeds (per-seed means in brackets).", ""]
    J = {}
    for run in RUNS:
        J[run] = {}
        for task in tasks:
            seeds = {}
            for d in sorted(SRC.glob(f"policy_{run.replace('/', '_')}_seed*_{MODEL}_{task}")):
                if (d / "metrics.json").exists():           # a rollout still running has none
                    seeds[d.name.split("_seed")[1].split("_")[0]] = json.loads((d / "metrics.json").read_text())
            if not seeds:
                continue
            n = min(len(r["command_gap_mrad"]) for rows in seeds.values() for r in rows)
            L += [f"## `{run}`, {task}: {len(seeds)} seeds, {len(next(iter(seeds.values())))} held-out episodes each", "",
                  "| | " + " | ".join(f"{3.2 * b:.1f}-{3.2 * (b + 1):.1f} s" for b in range(n)) + " |", "|---|" + "---|" * n]
            J[run][task] = {"seeds": len(seeds)}
            for label, key, fmt in ROWS:
                if not all(len(r[key]) >= n for rows in seeds.values() for r in rows):
                    continue
                per = np.array([[np.mean([r[key][b] for r in rows]) for b in range(n)] for rows in seeds.values()])     # [seed, block]
                if key == "force_gap_n" and not per.any():  # a policy without a force forecast
                    continue
                J[run][task][key] = {"mean": per.mean(0).tolist(), "per_seed": per.tolist()}
                L.append(f"| {label} | " + " | ".join(fmt.format(per[:, b].mean()) + " (" + ", ".join(fmt.format(v) for v in per[:, b]) + ")" for b in range(n)) + " |")
            L.append("")
    OUT.write_text("\n".join(L))
    OUT.with_suffix(".json").write_text(json.dumps(J, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
