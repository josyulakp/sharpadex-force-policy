"""Command streams for the simulator: what the operator commanded and what each model variant commands.

    python scripts/sim_streams.py [key=value ...]      -> reports/sim/streams/<episode>.npz

Uses the same episodes, variants and stitching as scripts/rollout.py (config section `rollout`): the model
re-predicts every `rollout.execute_steps` frames from the recorded observations. Each stream is a [T, 58]
joint-target array plus the hand-base poses it implies (learned forward kinematics, sim/fit_fk.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rollout import predict  # noqa: E402
from sharpa.config import REPO_ROOT, parse_cli  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402
from sharpa.engine import load_checkpoint  # noqa: E402
from sim.fit_fk import ARM, FK  # noqa: E402


@torch.no_grad()
def hand_poses(fk: dict, targets: np.ndarray, torso: np.ndarray) -> np.ndarray:
    """[T, 58] joint targets + [T, 5] torso -> [T, 2, 12]: position (3) and rotation matrix (9), left then right."""
    out = []
    for side in ("left", "right"):
        q = torch.from_numpy(np.concatenate([targets[:, ARM[side]], torso], 1)).float().cuda()
        p, Rm = fk[side](q)
        out.append(torch.cat([p, Rm.flatten(1)], 1).cpu().numpy())
    return np.stack(out, 1)


def main():
    cfg = parse_cli(sys.argv[1:])
    r = cfg["rollout"]
    group = REPO_ROOT / "runs" / r["group"]
    listing = json.loads((REPO_ROOT / r["out_dir"] / "rollouts.json").read_text())["episodes"]
    run_cfg = load_checkpoint(group / r["action_variants"][0] / f"seed{r['seed']}")[1]["cfg"]
    task = run_cfg["data"]["tasks"][0]
    sp = build_splits(run_cfg)[task]
    by_uid = {x.uid: x for x in sp["train_full"] + sp["heldout"]}
    refs = [by_uid[e["uid"]] for e in listing]
    ck = torch.load(REPO_ROOT / "sim" / f"fk_{task}.pt", map_location="cuda")
    fk = {}
    for side in ("left", "right"):
        fk[side] = FK().cuda().eval()
        fk[side].load_state_dict(ck[side]["model"])
    preds = {v: predict(group / v / f"seed{r['seed']}", refs, r["execute_steps"]) for v in r["action_variants"]}
    out = REPO_ROOT / "reports" / "sim" / "streams"
    out.mkdir(parents=True, exist_ok=True)
    for k, (ref, e) in enumerate(zip(refs, listing)):
        first = preds[r["action_variants"][0]][k]

        torso = np.stack(pd.read_parquet(ref.path, columns=["observation.state"])["observation.state"].values)[:, 58:63]
        streams = {"recorded": first["action"]} | {v: preds[v][k]["cmd"] for v in preds}
        arrays = {"state": first["state"], "force_meas": first["force_meas"], "contact": first["contact"]}
        for name, tg in streams.items():
            arrays[f"targets__{name}"] = tg.astype(np.float32)
            arrays[f"poses__{name}"] = hand_poses(fk, tg, torso).astype(np.float32)
            if name in preds:
                arrays[f"force_pred__{name}"] = preds[name][k]["force"]
        np.savez_compressed(out / (ref.uid.replace("/", "__") + ".npz"), uid=ref.uid, label=e["label"], **arrays)
        print(ref.uid, e["label"], {n: a.shape for n, a in arrays.items() if n.startswith("targets")})


if __name__ == "__main__":
    main()
