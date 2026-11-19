"""PrimeSim world model, step 2: widen the action inputs of the released checkpoint from 31 to 75 dims.

    <primesim src>/.venv/bin/python scripts/ps_widen.py      -> <src>/checkpoints/primesim_base_75.safetensors

Two layers read the actions, four steps at a time ([step 0 dims, step 1 dims, ...]). The weights of the 31 shared dims
are kept in place; the columns of the 44 finger joints start at zero, so the widened model equals the released one
until training moves them.
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import load_config  # noqa: E402

if __name__ == "__main__":
    P = load_config()["primesim_world_model"]
    ck = Path(P["src"]) / "checkpoints"
    sd, new, old = load_file(ck / "primesim_base.safetensors"), P["action_dim"], 31
    done = []
    for k, w in sd.items():
        if w.ndim == 2 and w.shape[1] == 4 * old and k.startswith(("action_mlp1.", "frame_action_adapter.")):
            wide = torch.zeros(w.shape[0], 4, new, dtype=w.dtype)
            wide[:, :, :old] = w.view(w.shape[0], 4, old)
            sd[k] = wide.view(w.shape[0], 4 * new).contiguous()
            done.append(k)
    assert len(done) == 2, done
    save_file(sd, ck / f"primesim_base_{new}.safetensors")
    print("widened", done)
