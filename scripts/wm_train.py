"""Fine-tune the Cosmos-Predict2 2B action-conditioned video model on joint commands -> head-camera video.

    <cosmos src>/.venv/bin/python scripts/wm_train.py [steps=N] [batch_size=N] [lora_rank=N]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wm_common import STEPS, W, batch_dict, build_model, chunk_action, trainable_state

W.update({k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])})


class Chunks(torch.utils.data.Dataset):
    """Random 13-frame windows of the training episodes."""
    def __init__(self):
        root = Path(W["data_dir"])
        self.eps = []
        for task in W["tasks"]:
            for z in sorted((root / task).glob("*/*.npz")):
                d = np.load(z)
                if str(d["split"]) == "train" and len(d["state"]) > W["chunk_frames"]:
                    self.eps.append((z.with_suffix(".npy"), d["state"], d["action"]))

    def __len__(self):
        return 1_000_000

    def __getitem__(self, i):
        rng = np.random.default_rng()
        path, state, action = self.eps[rng.integers(len(self.eps))]
        t = rng.integers(len(state) - STEPS)
        frames = np.load(path, mmap_mode="r")[t:t + STEPS + 1]
        return torch.from_numpy(np.array(frames)), torch.from_numpy(chunk_action(action[t + 1:t + STEPS + 1], state[t]))


def main():
    out = Path(W["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(lora_rank=W.get("lora_rank", 32))
    params = [p for p in model.pipe.dit.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f} M parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=W["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    data = Chunks()
    print(f"{len(data.eps)} training episodes", flush=True)
    loader = iter(torch.utils.data.DataLoader(data, batch_size=W["batch_size"], num_workers=W["num_workers"], persistent_workers=True))
    log, t0, s = [], time.time(), W["train_size"]
    model.train()
    for step in range(1, W["steps"] + 1):
        frames, action = next(loader)
        v = frames.cuda().permute(0, 1, 4, 2, 3).flatten(0, 1).float()                # [B*T, 3, H, W]
        if v.shape[-1] != s:
            v = F.interpolate(v, size=(s, s), mode="bilinear", antialias=True)
        v = v.round().clamp(0, 255).to(torch.uint8).view(len(frames), -1, 3, s, s).permute(0, 2, 1, 3, 4)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model.training_step(batch_dict(v, action), step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append(float(loss))
        if step % 50 == 0 or step == 1:
            print(f"step {step} loss {np.mean(log[-50:]):.4f} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if step % W["save_every"] == 0 or step == W["steps"]:
            torch.save(trainable_state(model), out / "weights.pt")
            (out / "train_log.json").write_text(json.dumps({"step": step, "loss": log, "config": {k: v for k, v in W.items()}}))


if __name__ == "__main__":
    main()
