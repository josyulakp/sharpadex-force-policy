"""Stage 3, step 1: reference command streams of the frozen four-expert policy for every episode of the task.

    python scripts/s3_prepare.py      -> <stage3.data_dir>/streams/<episode>.npz   (training environment, GPU)

Per episode: the operator's commands and the frozen policy's commands (re-predicted every `execute_steps` frames from the
recorded observations, as in scripts/rollout.py), the hand-base poses both imply (learned forward kinematics, sim/fit_fk.py),
and what the policy exposes at every frame for the stage-3 actor: the pooled decision slots (the frozen token), the gate,
the contact and phase read-outs and the fingertip-force forecast.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sim_streams import hand_poses  # noqa: E402
from sharpa.config import REPO_ROOT  # noqa: E402
from sharpa.data.dataset import FrameData  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402
from sharpa.engine import load_checkpoint  # noqa: E402
from sim.fit_fk import FK  # noqa: E402


@torch.no_grad()
def main():
    s3 = yaml.safe_load((REPO_ROOT / "configs" / "stage3.yaml").read_text())
    model, ck = load_checkpoint(REPO_ROOT / s3["base_run"])
    cfg, task, E = ck["cfg"], s3["task"], s3["execute_steps"]
    sp = build_splits(cfg)[task]
    refs = sp["train_full"] + sp["heldout"]
    split = ["train"] * len(sp["train_full"]) + ["heldout"] * len(sp["heldout"])
    fkck = torch.load(REPO_ROOT / "sim" / f"fk_{task}.pt", map_location="cuda")
    fk = {}
    for side in ("left", "right"):
        fk[side] = FK().cuda().eval()
        fk[side].load_state_dict(fkck[side]["model"])
    out_dir = Path(s3["data_dir"]) / "streams"
    out_dir.mkdir(parents=True, exist_ok=True)
    slots = {}
    model.dec_norm.register_forward_hook(lambda m, i, o: slots.__setitem__("q", o))
    for c0 in range(0, len(refs), 64):                                   # 64 episodes at a time keeps the vision features in memory small
        part = refs[c0:c0 + 64]
        data = FrameData(cfg, part, ck["vocab"], cfg["data"]["tasks"])
        keep = {k: [] for k in ("act", "force", "token", "gate", "contact", "phase")}
        for s in range(0, len(data), 1024):
            b = data.batch(torch.arange(s, min(s + 1024, len(data))))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
                o = model(b)
            p = model.to_physical(o, b)
            keep["act"].append(p["action"].float().cpu()); keep["force"].append(p["force"].float().cpu())
            keep["token"].append(slots["q"][:, model.H:].float().mean(1).cpu())                     # [B, d]: the three decision slots, pooled
            keep["gate"].append(o["gate"].float().cpu())
            keep["contact"].append(torch.sigmoid(o["contact_logit"].float()).cpu())
            keep["phase"].append(torch.softmax(o["phase_logits"].float(), -1).cpu())
        keep = {k: torch.cat(v) for k, v in keep.items()}
        for k, ref in enumerate(part):
            g = torch.nonzero(data.ep_id == k).squeeze(1)
            T, t = len(g), torch.arange(len(g))
            t0 = t // E * E
            cmd = keep["act"][g][t0, t - t0].numpy()                                                # [T, 58] stitched joint targets
            fnext = keep["force"][g][:, 0].view(T, 10, 3).norm(dim=2).numpy()                       # [T, 10] forecast of |F| at the next frame
            st = np.stack(pd.read_parquet(ref.path, columns=["observation.state"])["observation.state"].values)
            torso, action = st[:, 58:63], data.action[g].numpy()
            np.savez_compressed(
                out_dir / (ref.uid.replace("/", "__") + ".npz"), uid=ref.uid, split=split[c0 + k],
                targets__base=cmd.astype(np.float32), poses__base=hand_poses(fk, cmd, torso).astype(np.float32),
                targets__recorded=action.astype(np.float32), poses__recorded=hand_poses(fk, action, torso).astype(np.float32),
                token=keep["token"][g].numpy().astype(np.float32), gate=keep["gate"][g].numpy().astype(np.float32),
                contact_prob=keep["contact"][g].numpy().astype(np.float32), phase=keep["phase"][g].numpy().astype(np.float32),
                force_pred=fnext.astype(np.float32),
                force_meas=data.tactile[g].view(T, 10, 6)[..., :3].norm(dim=2).numpy().astype(np.float32),
                err_mrad=np.float32(1000 * np.abs(cmd - action).mean()))
        print(f"{min(c0 + 64, len(refs))}/{len(refs)} episodes", flush=True)
        del data


if __name__ == "__main__":
    main()
