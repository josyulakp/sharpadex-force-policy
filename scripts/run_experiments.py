"""Train + evaluate every variant x seed of an experiment group (configs/experiments.yaml).

    python scripts/run_experiments.py <group> [--jobs 3] [--only variant,variant] [extra key=value overrides for all runs]

Runs are named <group>/<variant>/seed<k> under runs/. Finished runs (eval_heldout.json present)
are skipped, so the command can be re-issued after an interruption.
"""
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    group = args.pop(0)
    jobs = int(args.pop(args.index("--jobs") + 1)) if "--jobs" in args else 3
    if "--jobs" in args:
        args.remove("--jobs")
    only = args.pop(args.index("--only") + 1).split(",") if "--only" in args else None
    if "--only" in args:
        args.remove("--only")
    spec = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text())[group]
    queue = []
    for name, overrides in spec["variants"].items():
        if only and name not in only:
            continue
        for seed in spec["seeds"]:
            run = REPO / "runs" / group / name / f"seed{seed}"
            if (run / "eval_heldout.json").exists():
                continue
            cmd = [sys.executable, str(REPO / "scripts" / "train.py"), *spec["common"], *overrides, *args,
                   f"train.name={group}/{name}", f"train.seed={seed}"]
            queue.append((run, cmd))
    print(f"{group}: {len(queue)} runs to do, {jobs} at a time")
    running, failed = [], []
    while queue or running:
        while queue and len(running) < jobs:
            run, cmd = queue.pop(0)
            run.mkdir(parents=True, exist_ok=True)
            running.append((run, subprocess.Popen(cmd, stdout=open(run / "log.txt", "w"), stderr=subprocess.STDOUT)))
            print("start", run.relative_to(REPO), flush=True)
        time.sleep(5)
        for item in list(running):
            code = item[1].poll()
            if code is not None:
                running.remove(item)
                print("done " if code == 0 else "FAILED", item[0].relative_to(REPO), flush=True)
                if code != 0:
                    failed.append(item[0])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
