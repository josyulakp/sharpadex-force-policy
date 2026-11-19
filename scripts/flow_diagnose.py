"""Why is the flow-matching policy behind the regression policy? Measurements on one seed, no training.

    python scripts/flow_diagnose.py [seed=0 every=6]     -> reports/flow_diagnosis/diagnosis.json

flow = runs/flow/full, regression = runs/flow/regression_25k (same data, budget and schedule). On held-out frames (every `every`-th)
and an equally large random set of training frames:
  alignment    error next to the "do nothing" baselines, share of the real change explained, R^2, correlation
  sampling     flow error against the number of integration steps, the size of the starting noise, and the mean of K samples
  denoise      flow: error of the chunk recovered in ONE step from each noise level (tau = 1 is plain regression from noise)
  horizon      error along the 16 steps of the chunk
  layers       activations of every layer on the action path: size, unused units, number of directions in use
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402
from sharpa.data.dataset import FrameData  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402
from sharpa.engine import load_checkpoint  # noqa: E402

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
ARM = [i for i in range(58) if i < 7 or 29 <= i < 36]
HAND = [i for i in range(58) if i not in ARM]
BATCH = 1024


def batches(data, idx):
    for s in range(0, len(idx), BATCH):
        yield data.batch(idx[s:s + BATCH])


def run_model(model, cfg, b):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
        o = model(b)
    return model.to_physical(o, b), o


class Sums:
    """Running sums for error, baseline, R^2 and correlation of a predicted change against the real change."""

    def __init__(self):
        self.s = {}

    def add(self, name, pred, real, base, mask):
        """pred, real, base [..., D] in physical units; mask broadcastable: 1 = counted. The change is measured from `base`."""
        m = mask.expand_as(real).double()
        p, r = (pred - base).double(), (real - base).double()
        v = [m.sum(), ((p - r).abs() * m).sum(), (r.abs() * m).sum(), ((p - r) ** 2 * m).sum(), (r * m).sum(), (r * r * m).sum(),
             (p * m).sum(), (p * p * m).sum(), (p * r * m).sum(), (real.double().abs() * m).sum()]
        self.s[name] = self.s.get(name, 0) + torch.stack(v).cpu()

    def result(self):
        out = {}
        for k, (n, ae, ab, se, sr, srr, sp, spp, spr, mag) in self.s.items():
            var_r, var_p = srr / n - (sr / n) ** 2, spp / n - (sp / n) ** 2
            out[k] = {"error": float(ae / n), "baseline": float(ab / n), "explained": float(1 - ae / ab),
                      "r2": float(1 - (se / n) / var_r), "corr": float((spr / n - sp * sr / n ** 2) / (var_r * var_p).sqrt()),
                      "relative_to_level": float(ae / mag)}
        return out


@torch.no_grad()
def alignment(model, cfg, data, idx):
    S = Sums()
    horizon = torch.zeros(model.H, dtype=torch.float64)
    count = torch.zeros(model.H, dtype=torch.float64)
    for b in batches(data, idx):
        p, _ = run_model(model, cfg, b)
        av, fv, c = b["action_valid"][..., None], b["force_valid"][..., None], b["contact_now"][:, None, None]
        now_q = b["state"][:, -1:, :]
        now_f = b["tactile"][:, -1].view(-1, 10, 6)[..., :3].flatten(1)[:, None]
        for name, dims in (("joints", slice(None)), ("arm joints", ARM), ("hand joints", HAND)):
            S.add(name, p["action"][..., dims], b["action"][..., dims], now_q[..., dims], av)
        S.add("joints, in contact", p["action"], b["action"], now_q, av & c)
        S.add("joints, free", p["action"], b["action"], now_q, av & ~c)
        S.add("force", p["force"], b["force"], now_f, fv)
        S.add("force, in contact", p["force"], b["force"], now_f, fv & c)
        S.add("force, free", p["force"], b["force"], now_f, fv & ~c)
        e = ((p["action"] - b["action"]).abs().mean(-1) * av[..., 0]).double()
        horizon += e.sum(0).cpu(); count += av[..., 0].double().sum(0).cpu()
    return S.result(), (1000 * horizon / count).tolist()


@torch.no_grad()
def action_error(model, cfg, data, idx, average=1):
    """Mean |error| (mrad) of the mean of `average` samples, and the spread between samples (mrad)."""
    err = n = spread = 0.0
    for b in batches(data, idx):
        preds = []
        for k in range(average):
            model.flow_seed = k
            preds.append(run_model(model, cfg, b)[0]["action"])
        model.flow_seed = 0
        st = torch.stack(preds)
        av = b["action_valid"].float()
        err += (((st.mean(0) - b["action"]).abs().mean(-1) * av).sum()).item(); n += av.sum().item()
        if average > 1:
            spread += ((st.std(0).mean(-1) * av).sum()).item()
    return 1000 * err / n, 1000 * spread / n


def flow_parts(model, cfg, b):
    """Everything forward_flow computes before the noise: latent, gate, wrench tokens, and the normalised target chunk."""
    cache = model.encode(b)
    dec = model.dec_norm(model.decoder(model.slots[model.H:].expand(len(cache), -1, -1), cache))
    gate = F.softmax(model.gate(dec.flatten(1)), -1)
    toks = model.contact_tokens(b) if model.wrench_expert else None
    now = b["tactile"][:, -1].view(-1, 10, 6)[..., :3].flatten(1)[:, None]
    base = now if model.force_residual else torch.zeros_like(now)
    y = torch.cat([(b["action"] - b["state"][:, -1:, :]) / model.delta_std, (b["force"] - base) / model.force_scale], -1)
    return cache, gate, toks, y


@torch.no_grad()
def denoise(model, cfg, data, idx, taus):
    """Error (mrad) of the action chunk recovered in one step from noise level tau, and the velocity loss there."""
    g = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for tau in taus:
        err = n = loss = 0.0
        for b in batches(data, idx):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
                cache, gate, toks, y = flow_parts(model, cfg, b)
                noise = torch.randn(y.shape, device="cuda", generator=g)
                x = tau * noise + (1 - tau) * y
                v = model.velocity(cache, gate, x, torch.full((len(y),), tau, device="cuda"), toks)[0].float()
            y0 = (x - tau * v)[..., :model.J]                                               # implied clean chunk
            av = b["action_valid"].float()
            err += ((((y0 - y[..., :model.J]) * model.delta_std).abs().mean(-1) * av).sum()).item(); n += av.sum().item()
            loss += ((((v - (noise - y))[..., :model.J] ** 2).mean(-1) * av).sum()).item()
        out.append({"tau": tau, "one_step_error_mrad": 1000 * err / n, "velocity_loss": loss / n})
    return out


def layer_stats(x):
    """x [N, width] activations -> size, share of unused units, number of directions in use."""
    x = x.float()
    std = x.std(0)
    ev = torch.linalg.eigvalsh(torch.cov(x.T.double())).clamp(min=0).flip(0)
    cum = ev.cumsum(0) / ev.sum()
    return {"width": x.shape[1], "rms": float(x.pow(2).mean().sqrt()), "unused_share": float((std < 0.05 * std.median()).float().mean()),
            "effective_rank": float(ev.sum() ** 2 / (ev ** 2).sum()), "dims_90": int((cum < 0.90).sum()) + 1, "dims_99": int((cum < 0.99).sum()) + 1,
            "spectrum": (ev[:64] / ev[0]).tolist()}


@torch.no_grad()
def layers(model, cfg, data, idx, tau=None):
    """Activations on the action path. Regression: the normal forward pass. Flow: the velocity network at noise level tau."""
    H, store = model.H, {}

    def keep(name, take=lambda o: o):
        def hook(_, __, out):
            o = take(out)
            o = o[:, :H] if o.dim() == 3 else o
            store.setdefault(name, []).append(o.float().flatten(0, -2)[::7].cpu())
        return hook
    hooks = [model.enc_norm.register_forward_hook(keep("encoder output (all input tokens)", lambda o: o.mean(1)))]
    for i, layer in enumerate(model.decoder.layers):
        hooks.append(layer.register_forward_hook(keep(f"decoder layer {i + 1}")))
        hooks.append(layer.linear1.register_forward_hook(keep(f"decoder layer {i + 1}, feed-forward hidden", F.gelu)))
    hooks.append(model.dec_norm.register_forward_hook(keep("decoder output")))
    for k, e in enumerate(model.experts):
        hooks.append(e[1].register_forward_hook(keep(f"expert {k + 1} hidden")))
        hooks.append(e[2].register_forward_hook(keep(f"expert {k + 1} output (58 joints)")))
    if model.flow:
        hooks.append(model.x_in.register_forward_hook(keep("noisy chunk embedding")))
    g = torch.Generator(device="cuda").manual_seed(0)
    for b in batches(data, idx[::4]):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            if model.flow:
                cache, gate, toks, y = flow_parts(model, cfg, b)
                noise = torch.randn(y.shape, device="cuda", generator=g)
                model.velocity(cache, gate, tau * noise + (1 - tau) * y, torch.full((len(y),), tau, device="cuda"), toks)
            else:
                model(b)
    for h in hooks:
        h.remove()
    out = {}
    for name, chunks in store.items():
        if model.flow and name.startswith("decoder"):                                       # every batch: [decision pass, velocity pass]
            chunks = chunks[1::2]
        out[name] = layer_stats(torch.cat(chunks))
    return out


def main():
    seed, every = int(ARGS.get("seed", 0)), int(ARGS.get("every", 6))
    runs = {"flow": REPO_ROOT / "runs/flow/full" / f"seed{seed}", "regression": REPO_ROOT / "runs/flow/regression_25k" / f"seed{seed}"}
    result = {"seed": seed}
    sets = None
    for name, run in runs.items():
        model, ck = load_checkpoint(run)
        cfg = ck["cfg"]
        if sets is None:
            splits = build_splits(cfg)
            tasks = cfg["data"]["tasks"]
            held = FrameData(cfg, [r for t in tasks for r in splits[t]["heldout"]], ck["vocab"], tasks)
            train = FrameData(cfg, [r for t in tasks for r in splits[t]["train"]], ck["vocab"], tasks)
            hi = torch.arange(0, len(held), every)
            ti = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:len(hi)].sort().values
            sets = {"held-out": (held, hi), "training": (train, ti)}
            result["frames"] = len(hi)
        r = {"parameters": sum(p.numel() for p in model.parameters()), "width": cfg["model"]["d_model"]}
        for split, (data, idx) in sets.items():
            r[f"alignment {split}"], r[f"horizon {split}"] = alignment(model, cfg, data, idx)
        if model.flow:
            data, idx = sets["held-out"]
            r["steps"] = []
            for n in (1, 2, 5, 10, 20, 50):
                model.flow_steps = n
                r["steps"].append({"steps": n, "error": action_error(model, cfg, data, idx)[0]})
            model.flow_steps = cfg["model"]["flow"]["sample_steps"]
            r["noise"] = []
            for s in (0.0, 0.25, 0.5, 1.0):
                model.flow_noise = s
                r["noise"].append({"scale": s, "error": action_error(model, cfg, data, idx)[0],
                                   "error_training": action_error(model, cfg, *sets["training"])[0]})
            model.flow_noise = 1.0
            r["samples"] = [dict(zip(("error", "spread"), action_error(model, cfg, data, idx[::2], k)), k=k) for k in (1, 4, 16)]
            taus = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.02]
            r["denoise held-out"] = denoise(model, cfg, data, idx, taus)
            r["denoise training"] = denoise(model, cfg, *sets["training"], taus)
            r["layers"] = {f"tau {t}": layers(model, cfg, data, idx, t) for t in (1.0, 0.5, 0.1)}
        else:
            r["layers"] = {"forward": layers(model, cfg, *sets["held-out"])}
        result[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k not in ("layers",)}, indent=1)[:6000], flush=True)
    out = REPO_ROOT / "reports" / "flow_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis.json").write_text(json.dumps(result, indent=1))
    print("saved", out / "diagnosis.json")


if __name__ == "__main__":
    main()
