"""PrimeSim world model, step 3: fine-tune on SharpaDex (LoRA on the video model, action inputs and view embeddings in full).

    cd <primesim src> && .venv/bin/python <this repo>/scripts/ps_train.py [hydra overrides, e.g. max_train_steps=20]
    ... ps_train.py tasks=<task,task,...> run_name=sharpa_three_tasks init=<step-N.safetensors> base=<file in checkpoints/>
    # several tasks, continued from a checkpoint. PrimeSim reads only the LoRA tensors from `init`; the 16 tensors trained in full (action
    # inputs, view embedding) must be copied into a copy of the base checkpoint first and passed as `base`.

Runs PrimeSim's own scripts/train.py with launch=finetune and the SharpaDex dataset config.
Checkpoints: <src>/models/<run_name, default sharpa_eraser>/step-<N>.safetensors
"""
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from sharpa.config import load_config  # noqa: E402

P = load_config()["primesim_world_model"]
SRC = Path(P["src"])
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO / "scripts"), str(SRC), os.environ.get("PYTHONPATH", "")])
import ps_transforms  # noqa: E402,F401  (registers SharpaStateAction)

if __name__ == "__main__":
    own = dict(a.split("=", 1) for a in sys.argv[1:] if a.split("=")[0] in ("tasks", "run_name", "init", "base"))
    sys.argv = [a for a in sys.argv if a.split("=")[0] not in own]
    tasks = tuple(own.get("tasks", P["task"]).split(","))
    repos = sorted(p.name for p in Path(P["data_dir"]).iterdir() if p.name.startswith(tasks) and not p.name.endswith("__heldout")
                   and (p / "meta" / "sharpa_episodes.json").exists())      # finished shards only
    text = (REPO / "configs" / "primesim" / "sharpa.yaml").read_text().replace("DATA_ROOT", P["data_dir"]).replace("REPO_IDS", str(repos))
    (SRC / "config" / "dataset" / "sharpa.yaml").write_text(text)
    os.chdir(SRC)
    s, ck = P["image_size"], SRC / "checkpoints"
    sys.argv = ["train.py", "launch=finetune", "dataset=sharpa", f"run_name={own.get('run_name', 'sharpa_eraser')}", *([f"lora.lora_checkpoint={own['init']}"] if "init" in own else []),
                f"action_dim={P['action_dim']}", f"state_dim={P['action_dim']}", f"img_height={s}", f"img_width={s}",
                f"model.checkpoint_path={ck}/{own.get('base', 'primesim_base_%d.safetensors' % P['action_dim'])}",
                f"model.text_encoder_path={ck}/models_t5_umt5-xxl-enc-bf16.pth", f"model.vae_path={ck}/Wan2.2_VAE.pth",
                f"model.tokenizer_path={ck}/google/umt5-xxl",
                "lora.lora_base_model=dit", "lora.lora_target_modules='q,k,v,o,ffn.0,ffn.2'", "lora.lora_rank=32",
                *[f"{k}={v}" for k, v in P["train"].items()], *sys.argv[1:]]
    runpy.run_path(str(SRC / "scripts" / "train.py"), run_name="__main__")
