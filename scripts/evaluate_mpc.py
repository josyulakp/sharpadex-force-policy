"""MPC expert vs its C3 teacher, from a released checkpoint.

    python scripts/evaluate_mpc.py runs/eraser_force/full_plus_mpc_expert/seed0
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.engine import evaluate_mpc  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(evaluate_mpc(Path(sys.argv[1]).resolve()), indent=1))
