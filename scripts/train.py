"""Train one model and evaluate it on the held-out episodes.

    python scripts/train.py [--config configs/default.yaml] [key=value ...]
    e.g. python scripts/train.py train.name=no_force force.tactile_input=false force.torque_input=false \
             force.force_output=false force.decision_supervision=false train.seed=1
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import parse_cli  # noqa: E402
from sharpa.engine import evaluate, evaluate_mpc, train  # noqa: E402

if __name__ == "__main__":
    cfg = parse_cli(sys.argv[1:])
    run = train(cfg)
    if cfg["mpc"]["enabled"]:
        print(json.dumps(evaluate_mpc(run), indent=1))
    for step in cfg["train"].get("eval_at") or []:                          # e.g. results at 8k / 16k / 24k of one long run
        evaluate(run, tag=f"heldout__step{step}", ckpt=f"ckpt_step{step}.pt")
    print(json.dumps(evaluate(run), indent=1))
