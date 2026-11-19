"""Stage 2, steps 2-3: distributional critic on the frozen policy's features, then credit and positive labels.

    python scripts/rl_critic.py      -> <rl.out_dir>/credit.npz (per frame: value, credit, label, intensity, regime)
                                        <rl.out_dir>/credit.json (held-out check of the critic, thresholds, label rates)

Critic: categorical value distribution (C51 projection, target network) of h_t = [mean encoder token, decision slots]
of the frozen stage-1 flow policy; trained on training episodes only. Credit: reward over the next 20 frames + value
there - value now. Positive label: top 20 % of credit among training frames, separately for the contact regime
(wrench-change intensity = 1) and the rest; held-out frames are labelled with the same thresholds.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa import rl  # noqa: E402
from sharpa.config import REPO_ROOT, load_config  # noqa: E402
from sharpa.data.dataset import FrameData  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402
from sharpa.engine import load_checkpoint  # noqa: E402


@torch.no_grad()
def features(model, data, bs=2048):
    out = []
    for s in range(0, len(data), bs):
        b = data.batch(torch.arange(s, min(s + bs, len(data))))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cache = model.encode(b)
            dec = model.dec_norm(model.decoder(model.slots[model.H:].expand(len(cache), -1, -1), cache))
        out.append(torch.cat([cache.mean(1), dec.flatten(1)], -1).float())
    return torch.cat(out)


def project(probs, reward, done, atoms, gamma):
    """Categorical projection of reward + gamma * Z(next) onto the fixed atoms."""
    v_min, v_max, n = atoms[0], atoms[-1], len(atoms)
    b = ((reward[:, None] + gamma * (1 - done[:, None]) * atoms).clamp(v_min, v_max) - v_min) / (atoms[1] - atoms[0])
    lo, hi = b.floor().long().clamp(0, n - 1), b.ceil().long().clamp(0, n - 1)
    w_hi = b - lo
    out = torch.zeros_like(probs)
    out.scatter_add_(1, lo, probs * (1 - w_hi))
    out.scatter_add_(1, hi, probs * w_hi)
    return out


if __name__ == "__main__":
    cfg = load_config(overrides=sys.argv[1:])
    K, C, gamma = cfg["rl"]["critic"], cfg["rl"]["credit"], cfg["rl"]["reward"]["gamma"]
    out = Path(cfg["rl"]["out_dir"])
    tab = rl.Table(out / "reward.npz")
    model, ck = load_checkpoint(REPO_ROOT / K["encoder_run"])
    splits = build_splits(ck["cfg"])
    tasks = ck["cfg"]["data"]["tasks"]
    parts = {p: [r for t in tasks for r in splits[t][p]] for p in ("train", "heldout")}
    feats, cols = {}, {}
    for p, refs in parts.items():
        data = FrameData(ck["cfg"], refs, ck["vocab"], tasks)
        feats[p] = features(model, data)
        cols[p] = {k: torch.from_numpy(np.concatenate([tab.get(r.uid, k) for r in refs])).cuda() for k in ("reward", "ret")}
        cols[p]["last"] = (torch.arange(len(data)) == data.ep_end - 1).cuda()
        del data
    del model
    torch.manual_seed(K["seed"])
    h, mu, sd = feats["train"], feats["train"].mean(0), feats["train"].std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(h.shape[1], K["hidden"]), torch.nn.SiLU(), torch.nn.Linear(K["hidden"], K["hidden"]),
                              torch.nn.SiLU(), torch.nn.Linear(K["hidden"], K["atoms"])).cuda()
    target = copy.deepcopy(net).requires_grad_(False)
    atoms = torch.linspace(K["v_min"], K["v_max"], K["atoms"], device="cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=K["lr"])
    r, last = cols["train"]["reward"].float(), cols["train"]["last"].float()
    value = lambda x: (F.softmax(net((x - mu) / sd), -1) * atoms).sum(-1)  # noqa: E731
    for step in range(K["steps"]):
        i = torch.randint(len(h), (K["batch_size"],), device="cuda")
        nxt = (i + 1).clamp(max=len(h) - 1)                               # ignored where the episode ends (done = 1)
        with torch.no_grad():
            tgt = project(F.softmax(target((h[nxt] - mu) / sd), -1), r[i], last[i], atoms, gamma)
        loss = -(tgt * F.log_softmax(net((h[i] - mu) / sd), -1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for a, b in zip(target.parameters(), net.parameters()):
                a.lerp_(b, K["target_ema"])
        if step % 2000 == 0 or step == K["steps"] - 1:
            with torch.no_grad():
                v = value(feats["heldout"])
                g = cols["heldout"]["ret"]
            print(f"step {step:5d} loss {loss.item():.4f}  held-out: |value - return| {(v - g).abs().mean():.3f} "
                  f"(constant {(g - cols['train']['ret'].mean()).abs().mean():.3f}), correlation {torch.corrcoef(torch.stack([v, g]))[0, 1]:.3f}", flush=True)

    # credit, contact regime, labels
    with torch.no_grad():
        values = {p: value(feats[p]).cpu().numpy() for p in parts}
        g = cols["heldout"]["ret"]
        check = {"heldout_abs_error": float((value(feats["heldout"]) - g).abs().mean()),
                 "heldout_abs_error_constant": float((g - cols["train"]["ret"].mean()).abs().mean()),
                 "heldout_correlation": float(torch.corrcoef(torch.stack([value(feats["heldout"]), g]))[0, 1])}
    floor = float(np.quantile(np.concatenate([tab.get(r.uid, "change") for r in parts["train"]]), C["corpus_quantile"]))
    rows = {k: [] for k in ("value", "credit", "intensity", "regime", "label")}
    uid, length, split, thresholds = [], [], [], {}
    for p in ("train", "heldout"):
        refs, o = parts[p], 0
        inten = rl.contact_intensity([tab.get(r.uid, "change") for r in refs], floor, C["episode_quantile"])
        credit = []
        for r_, it in zip(refs, inten):
            v = values[p][o:o + r_.num_frames]
            o += r_.num_frames
            credit.append(rl.episode_credit(v.astype(np.float64), tab.get(r_.uid, "reward"), C["horizon"]))
            rows["value"].append(v); rows["intensity"].append(it.astype(np.float32)); rows["regime"].append(it >= 1.0)
        credit, regime = np.concatenate(credit), np.concatenate([it >= 1.0 for it in inten])
        if p == "train":
            thresholds = {"contact": float(np.quantile(credit[regime], 1 - C["positive_fraction"])),
                          "free": float(np.quantile(credit[~regime], 1 - C["positive_fraction"]))}
        label = credit > np.where(regime, thresholds["contact"], thresholds["free"])
        rows["credit"].append(credit.astype(np.float32)); rows["label"].append(label)
        uid += [r_.uid for r_ in refs]; length += [r_.num_frames for r_ in refs]; split += [p] * len(refs)
        check[p] = {"frames": int(len(label)), "contact_regime_rate": float(regime.mean()), "positive_rate": float(label.mean()),
                    "positive_rate_contact": float(label[regime].mean()), "positive_rate_free": float(label[~regime].mean())}
    np.savez_compressed(out / "credit.npz", uid=np.array(uid), length=np.array(length), split=np.array(split),
                        **{k: np.concatenate(v) for k, v in rows.items()})
    check["thresholds"] = thresholds
    (out / "credit.json").write_text(json.dumps(check, indent=1))
    print(json.dumps(check, indent=1))
