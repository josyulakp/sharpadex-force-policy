"""What the trained model uses for which input, measured on the held-out frames.

    python scripts/activations.py [runs/eraser_force/full]     -> reports/architecture/activations.json

For every seed of the variant:
  gate        weight of each expert, by contact / free and by task phase
  experts     size of each expert's (gated) contribution to arm and hand joints, and how far apart the experts are
  attention   share of decoder cross-attention each query slot puts on each input group (seed 0)
  knockout    change of the predicted action when one input group is swapped for that of a random other frame
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402
from sharpa.data.dataset import FrameData  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402
from sharpa.engine import load_checkpoint  # noqa: E402

ARM = torch.tensor([i for i in range(58) if i < 7 or 29 <= i < 36])
HAND = torch.tensor([i for i in range(58) if not (i < 7 or 29 <= i < 36)])


def token_groups(cfg, model):
    n_vis = model.vis_pos.shape[1]
    g, s = {}, 0
    for cam in cfg["vision"]["cameras"]:
        g[f"camera {cam}"] = (s, s + n_vis); s += n_vis
    for name in ("joint state", "fingertip force", "joint torque"):
        g[name] = (s, s + model.K); s += model.K
    g["task"] = (s, s + 1)
    return g


@torch.no_grad()
def analyse(run: Path, with_attention: bool):
    model, ck = load_checkpoint(run)
    cfg = ck["cfg"]
    tasks = cfg["data"]["tasks"]
    refs = [r for t in tasks for r in build_splits(cfg)[t]["heldout"]]
    data = FrameData(cfg, refs, ck["vocab"], tasks)
    groups = token_groups(cfg, model)
    store = []
    if with_attention:
        for layer in model.decoder.layers:
            orig = layer.multihead_attn.forward
            def fwd(*a, _o=orig, **k):
                k["need_weights"], k["average_attn_weights"] = True, True
                out = _o(*a, **k)
                store.append(out[1].float())
                return out
            layer.multihead_attn.forward = fwd

    def run_model(b):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            cache = model.encode(b)
            q = model.dec_norm(model.decoder(model.slots.expand(len(cache), -1, -1), cache))
            chunk, dec = q[:, :model.H], q[:, model.H:]
            gate = torch.softmax(model.gate(dec.flatten(1)).float(), -1)
            ex = torch.stack([e(chunk).float() for e in model.experts], 1)       # [B, K, H, J] normalised
        return gate, ex

    rows = {k: [] for k in ("gate", "contact", "skill", "ex_arm", "ex_hand", "sep", "attn")}
    knock = {g: [] for g in ("cameras", "joint state", "fingertip force", "joint torque")}
    gen = torch.Generator().manual_seed(0)
    for s in range(0, len(data), 1024):
        idx = torch.arange(s, min(s + 1024, len(data)))
        b = data.batch(idx)
        store.clear()
        gate, ex = run_model(b)
        if with_attention:
            a = torch.stack(store).mean(0)                                        # [B, slots, tokens]
            a = torch.stack([a[..., lo:hi].sum(-1) for lo, hi in groups.values()], -1)
            rows["attn"].append(torch.cat([a[:, :model.H].mean(1, keepdim=True), a[:, model.H:]], 1).cpu())
        std = model.delta_std
        act = (gate[:, :, None, None] * ex).sum(1) * std                         # rad
        contrib = (gate[:, :, None, None] * ex * std).abs()                      # [B, K, H, J]
        rows["ex_arm"].append(contrib[..., ARM.cuda()].mean((2, 3)).cpu())
        rows["ex_hand"].append(contrib[..., HAND.cuda()].mean((2, 3)).cpu())
        rows["sep"].append((((ex[:, 0] - ex[:, 1]) * std).abs().mean((1, 2)) if ex.shape[1] > 1 else torch.zeros(len(idx))).cpu())
        rows["gate"].append(gate.cpu()); rows["contact"].append(b["contact_now"].cpu()); rows["skill"].append(b["skill"].cpu())
        other = data.batch(torch.randint(len(data), (len(idx),), generator=gen))
        for g, keys in (("cameras", ["feats"]), ("joint state", ["state"]), ("fingertip force", ["tactile"]),
                        ("joint torque", ["torque"])):
            g2, e2 = run_model(b | {k: other[k] for k in keys})
            knock[g].append((((g2[:, :, None, None] * e2).sum(1) * std - act).abs().mean((1, 2)) * 1000).cpu())
    R = {k: torch.cat(v) for k, v in rows.items() if v}
    c = R["contact"].bool()
    vocab = ck["vocab"]
    conds = {"contact": c, "free": ~c} | {f"phase: {n}": R["skill"] == i for i, n in enumerate(vocab) if (R["skill"] == i).any()}
    mean = lambda x, m: x[m].mean(0).tolist()  # noqa: E731
    out = {"conditions": {k: int(m.sum()) for k, m in conds.items()},
           "gate": {k: mean(R["gate"], m) for k, m in conds.items()},
           "expert_arm_mrad": {k: (np.array(mean(R["ex_arm"], m)) * 1000).tolist() for k, m in conds.items()},
           "expert_hand_mrad": {k: (np.array(mean(R["ex_hand"], m)) * 1000).tolist() for k, m in conds.items()},
           "expert_separation_mrad": {k: float(R["sep"][m].mean() * 1000) for k, m in conds.items()},
           "knockout_mrad": {g: {k: float(torch.cat(v)[m].mean()) for k, m in conds.items()} for g, v in knock.items()}}
    if with_attention:
        slots = ["action / force slots (16)", "contact-soon slot", "phase slot", "force-level slot"]
        out["attention"] = {"groups": list(groups), "tokens": [hi - lo for lo, hi in groups.values()],
                            "slots": {sn: {k: mean(R["attn"][:, i], m) for k, m in conds.items()} for i, sn in enumerate(slots)}}
        out["params"] = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()} | \
            {"total": sum(p.numel() for p in model.parameters())}
    return out


def main():
    variant = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/eraser_force/full")
    variant = variant if variant.is_absolute() else REPO_ROOT / variant
    res = {"variant": variant.name, "seeds": {}}
    for i, run in enumerate(sorted(variant.glob("seed*"))):
        res["seeds"][run.name] = analyse(run, with_attention=i == 0)
        print(run.name, json.dumps(res["seeds"][run.name]["gate"]), flush=True)
    out = REPO_ROOT / "reports" / "architecture"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"activations_{variant.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
