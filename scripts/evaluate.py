"""Reproduce the held-out evaluation of a released checkpoint with one command.

    python scripts/evaluate.py runs/<name>/seed0                         # standard held-out evaluation
    python scripts/evaluate.py runs/<name>/seed0 zero_force=true         # step 5: force inputs set to zero
    python scripts/evaluate.py runs/<name>/seed0 swap_task=insert_plug_into_socket   # step 5: wrong instruction

Everything else (tasks, split seed, contact threshold, model) is read from the checkpoint's config.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.engine import evaluate  # noqa: E402

if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve()
    overrides = {k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    tag = "heldout" + "".join(f"__{k}-{v}" for k, v in overrides.items())
    print(json.dumps(evaluate(run, overrides, tag), indent=1))
