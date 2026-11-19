"""Offline replay of single episodes: what the model would command, next to what the operator commanded.

    python scripts/rollout.py [rollout.n_heldout=3 ...]     -> reports/rollouts/{rollouts.json, <id>.mp4}

The observations always come from the recording (nothing is executed on a robot). Every
`rollout.execute_steps` frames the model predicts a chunk from the recorded observation at that
frame; the chunk is followed until the next prediction. Episodes: the held-out episodes with the
lowest / median / highest action error of the first variant, plus training episodes of median length.
"""
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, parse_cli  # noqa: E402
from sharpa.data.dataset import FrameData  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402
from sharpa.engine import load_checkpoint  # noqa: E402

ARM = [i for i in range(58) if i < 7 or 29 <= i < 36]
NAMES = ([f"left arm j{i}" for i in range(7)] + [f"left hand j{i}" for i in range(22)] +
         [f"right arm j{i}" for i in range(7)] + [f"right hand j{i}" for i in range(22)])


@torch.no_grad()
def predict(run: Path, refs, execute: int):
    """Stitched replay per episode: list of dicts with command [T,J], force [T,30], decisions."""
    model, ck = load_checkpoint(run)
    cfg = ck["cfg"]
    data = FrameData(cfg, refs, ck["vocab"], cfg["data"]["tasks"])
    acts, forces, cprob, phase = [], [], [], []
    for s in range(0, len(data), 1024):
        b = data.batch(torch.arange(s, min(s + 1024, len(data))))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            o = model(b)
        p = model.to_physical(o, b)
        acts.append(p["action"].cpu()); forces.append(p["force"].cpu())
        cprob.append(torch.sigmoid(o["contact_logit"].float()).cpu()); phase.append(o["phase_logits"].argmax(-1).cpu())
    acts, forces, cprob, phase = (torch.cat(x) for x in (acts, forces, cprob, phase))
    out = []
    for k in range(len(refs)):
        g = torch.nonzero(data.ep_id == k).squeeze(1)
        T, t = len(g), torch.arange(len(g))
        t0 = t // execute * execute
        cmd = acts[g][t0, t - t0]                                        # [T, J]
        tf = (t - 1).clamp(min=0)
        f0 = tf // execute * execute
        frc = forces[g][f0, tf - f0].view(T, 10, 3).norm(dim=2)          # [T, 10] predicted fingertip |F|
        meas = data.tactile[g].view(T, 10, 6)[..., :3].norm(dim=2)
        out.append({"cmd": cmd.numpy(), "force": frc.numpy(), "force_meas": meas.numpy(),
                    "action": data.action[g].numpy(), "state": data.state[g].numpy(),
                    "contact": data.contact[g].numpy(), "skill": data.skill[g].numpy(),
                    "contact_prob": cprob[g].numpy(), "phase": phase[g].numpy(),
                    "decision_supervision": cfg["force"]["decision_supervision"], "vocab": ck["vocab"]})
    return out


def write_video(ref, cams, height, stride, fps, path: Path):
    base = Path(ref.path).parents[2] / "videos" / "chunk-000"
    ins = [av.open(str(base / f"observation.images.{c}" / f"episode_{ref.episode_index:06d}.mp4")) for c in cams]
    out = av.open(str(path), "w")
    st = out.add_stream("libx264", rate=fps // stride)
    st.width, st.height, st.pix_fmt = height * len(cams), height, "yuv420p"
    st.options = {"crf": "30", "preset": "slow", "movflags": "+faststart"}
    for i, frames in enumerate(zip(*(c.decode(video=0) for c in ins))):
        if i % stride:
            continue
        img = np.concatenate([f.reformat(height, height, "rgb24").to_ndarray() for f in frames], 1)
        for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    [c.close() for c in ins]


def main():
    cfg = parse_cli(sys.argv[1:])
    r = cfg["rollout"]
    group = REPO_ROOT / "runs" / r["group"]
    first = group / r["action_variants"][0] / f"seed{r['seed']}"
    run_cfg = load_checkpoint(first)[1]["cfg"]
    splits = build_splits(run_cfg)
    task = run_cfg["data"]["tasks"][0]
    held, train = splits[task]["heldout"], splits[task]["train"]

    ev = np.load(first / "eval_heldout.npz")
    ep = ev["ep"].astype(int)
    per_ep = np.bincount(ep, ev["act"]) / np.bincount(ep)
    order = np.argsort(per_ep)
    pick = [order[0], order[len(order) // 2], order[-1]] if r["n_heldout"] == 3 else \
        list(order[np.linspace(0, len(order) - 1, r["n_heldout"]).astype(int)])
    labels = ["lowest error", "median error", "highest error"] if r["n_heldout"] == 3 else [""] * len(pick)
    by_len = sorted(train, key=lambda x: x.num_frames)
    tr = [by_len[len(by_len) // 2 + i] for i in range(r["n_train"])]
    refs = tr + [held[i] for i in pick]
    meta = [("train", "training episode (seen)")] * len(tr) + [("held-out", f"held-out, {l}") for l in labels]

    variants = list(dict.fromkeys(r["action_variants"] + r["force_variants"]))
    preds = {v: predict(group / v / f"seed{r['seed']}", refs, r["execute_steps"]) for v in variants}
    out_dir = REPO_ROOT / r["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = lambda a, n=2: np.round(np.asarray(a, dtype=float), n).tolist()  # noqa: E731
    episodes = []
    for k, (ref, (split, label)) in enumerate(zip(refs, meta)):
        p0 = preds[variants[0]][k]
        act, c = p0["action"], p0["contact"]
        rng = act.max(0) - act.min(0)
        arm = sorted(ARM, key=lambda j: -rng[j])[:3]
        hand = sorted([j for j in range(58) if j not in ARM], key=lambda j: -rng[j])[:3]
        err = {v: np.abs(preds[v][k]["cmd"] - act).mean(1) * 1000 for v in r["action_variants"]}
        vid = f"ep{k}_{split.replace('-', '')}.mp4"
        write_video(ref, r["cameras"], r["video_height"], r["video_stride"], run_cfg["data"]["fps"], out_dir / vid)
        episodes.append({
            "uid": ref.uid, "split": split, "label": label, "video": vid, "frames": len(act), "fps": run_cfg["data"]["fps"],
            "contact": c.astype(int).tolist(),
            "phase_label": p0["skill"].tolist(), "phase_pred": p0["phase"].tolist(), "phases": p0["vocab"],
            "contact_prob": rd(p0["contact_prob"]),
            "err": {v: rd(e) for v, e in err.items()},
            "summary": {v: {"all": float(e.mean()), "contact": float(e[c].mean()) if c.any() else None,
                            "free": float(e[~c].mean()) if (~c).any() else None} for v, e in err.items()},
            "hold_err": float(np.abs(p0["state"][:, None][np.arange(len(act)) // r["execute_steps"] * r["execute_steps"], 0] - act).mean() * 1000),
            "joints": [{"name": NAMES[j], "recorded": rd(np.degrees(act[:, j])),
                        **{v: rd(np.degrees(preds[v][k]["cmd"][:, j])) for v in r["action_variants"]}} for j in arm + hand],
            "force": {"measured": rd(p0["force_meas"].max(1)),
                      **{v: rd(preds[v][k]["force"].max(1)) for v in r["force_variants"]}},
            "force_err": {v: float(np.abs(preds[v][k]["force"] - p0["force_meas"])[1:].mean()) for v in r["force_variants"]},
        })
        print(ref.uid, label, {v: round(s["all"], 1) for v, s in episodes[-1]["summary"].items()}, flush=True)
    (out_dir / "rollouts.json").write_text(json.dumps({"execute_steps": r["execute_steps"], "seed": r["seed"],
                                                       "contact_threshold_n": run_cfg["data"]["contact_threshold_n"],
                                                       "episodes": episodes}))


if __name__ == "__main__":
    main()
