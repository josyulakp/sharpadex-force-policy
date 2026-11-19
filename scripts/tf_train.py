"""Tactile world model, step 2: train TacForeSight's world model on the converted episodes (its own train.py, unmodified).

    python scripts/tf_train.py      -> <data_dir>/runs/<run_name>/checkpoints/best_model.pt
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402

T = load_config()["tactile_world_model"]
text = (REPO_ROOT / "configs" / "tacforesight" / "dynamics.yaml").read_text()
for k, v in {"DATA_DIR": T["data_dir"], "OUT_DIR": str(Path(T["data_dir"]) / "runs"), "RUN_NAME": T["run_name"], "HORIZON": T["horizon"],
             "EPOCHS": T["epochs"], "BATCH": T["batch_size"]}.items():
    text = text.replace(k, str(v))
cfg = Path(T["data_dir"]) / "dynamics.yaml"
cfg.write_text(text)
sys.path[:0] = [T["src"], T["packages"]]
os.environ["PYTHONPATH"] = os.pathsep.join([T["src"], T["packages"], os.environ.get("PYTHONPATH", "")])
os.chdir(T["src"])
sys.argv = ["train.py", "dynamics", "--config", str(cfg)]
runpy.run_path("train.py", run_name="__main__")
