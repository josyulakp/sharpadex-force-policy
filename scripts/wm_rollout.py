"""Run joint-command streams through the video world model and save what it predicts the head camera sees.

    <cosmos src>/.venv/bin/python scripts/wm_rollout.py <streams npz> [names ...] [anchor=generated|real] [weights=...]
        -> reports/world_model/<episode>/<stream>__<anchor>.mp4 / .npz, summary.json

Streams come from scripts/sim_streams.py (targets__<name> [T, 58] at 30 Hz). The model predicts 12 frames (1.2 s) from
one frame and 12 joint commands; chunks are chained. anchor=generated: every chunk starts from the model's own last
frame (a free rollout from the first real frame). anchor=real: every chunk starts from the real frame (1.2 s look-ahead).
"""
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wm_common import REPO, STEPS, W, build_model, chunk_action


def resize(frames: np.ndarray, s: int) -> np.ndarray:
    if frames.shape[1] == s:
        return frames
    v = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return F.interpolate(v, size=(s, s), mode="bilinear", antialias=True).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def sample(pipe, frame: np.ndarray, act: np.ndarray, steps: int, seed: int) -> torch.Tensor:
    """One chunk: the pipeline's own sampling loop, fed a full-length clip (the conditioning frame repeated)
    because its single-frame entry point fails on the frame-count check. Returns [1, 3, 13, H, W] in [-1, 1]."""
    from cosmos_predict2.pipelines.video2world_action import misc

    vid = torch.from_numpy(frame).permute(2, 0, 1)[None, :, None].repeat(1, 1, STEPS + 1, 1, 1)
    batch = pipe._get_data_batch_input(vid, torch.from_numpy(act).to(torch.bfloat16)[None], "", "", num_latent_conditional_frames=1)
    pipe._normalize_video_databatch_inplace(batch)
    pipe._augment_image_dim_inplace(batch)
    _T, _H, _W = batch[pipe.input_video_key].shape[-3:]
    shape = [pipe.config.state_ch, pipe.tokenizer.get_latent_num_frames(_T),
             _H // pipe.tokenizer.spatial_compression_factor, _W // pipe.tokenizer.spatial_compression_factor]
    x0_fn = pipe.get_x0_fn_from_batch(batch, 0, is_negative_prompt=True)
    x = misc.arch_invariant_rand((1, *shape), torch.float32, "cuda", seed) * pipe.scheduler.config.sigma_max
    pipe.scheduler.set_timesteps(steps, device=x.device)
    x, prev = x.to(torch.float32), None
    for i, _ in enumerate(pipe.scheduler.timesteps):
        x0 = x0_fn(x, pipe.scheduler.sigmas[i].to(x.device, dtype=torch.float32).repeat(1))
        x, prev = pipe.scheduler.step(x0_pred=x0, i=i, sample=x, x0_prev=prev)
    return pipe.decode(x0_fn(x, pipe.scheduler.sigmas[-1].to(x.device, dtype=torch.float32).repeat(1)))


@torch.no_grad()
def rollout(model, real: np.ndarray, targets: np.ndarray, state: np.ndarray, anchor: str, seed: int = 0) -> np.ndarray:
    """real [N, S, S, 3] uint8 (10 Hz), targets / state [N, 58] at the same 10 Hz -> predicted frames [N', S, S, 3]."""
    out, frame = [real[0]], real[0]
    for i in range(0, len(real) - STEPS, STEPS):
        if anchor == "real":
            frame = real[i]
        act = chunk_action(targets[i + 1:i + STEPS + 1], state[i])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = sample(model.pipe, frame, act, W["sample_steps"], seed + i)
        chunk = ((video[0].float().permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()   # [13, S, S, 3]
        out += list(chunk[1:])
        frame = chunk[-1]
    return np.stack(out)


def main():
    args = [a for a in sys.argv[1:] if "=" not in a]
    opt = {"anchor": "generated", "weights": str(Path(W["out_dir"]) / "weights.pt")} | {
        k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    z = np.load(args[0], allow_pickle=False)
    uid = str(z["uid"])
    names = args[1:] or [k.split("__", 1)[1] for k in z.files if k.startswith("targets__")]
    k, s = W["frame_stride"], W["train_size"]
    task, season, ep = uid.split("/")
    real = resize(np.load(Path(W["data_dir"]) / task / season / f"{ep}.npy"), s)
    state = z["state"][::k]
    model = build_model(opt["weights"], lora_rank=opt.get("lora_rank", 32))
    model.eval()
    out = REPO / "reports" / "world_model" / uid.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for n in names:
        targets = z[f"targets__{n}"][::k]
        # joint state the stream would have at each chunk start: the real state shifted by the command difference
        est = state + (targets - z["targets__recorded"][::k])
        pred = rollout(model, real, targets, est, opt["anchor"])
        ref = real[:len(pred)].astype(np.float32)
        err = np.abs(pred.astype(np.float32) - ref).mean((1, 2, 3))                       # per frame, grey levels
        mse = ((pred.astype(np.float32) - ref) ** 2).mean((1, 2, 3))
        tag = f"{n}__{opt['anchor']}"
        iio.imwrite(out / f"{tag}.mp4", np.concatenate([real[:len(pred)], pred], 2), fps=30 // k, codec="libx264")
        np.savez_compressed(out / f"{tag}.npz", l1=err, psnr=10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6)))
        summary[tag] = {"frames": len(pred), "mean_abs_error_grey": float(err.mean()),
                        "psnr_db": float((10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6))).mean())}
        print(tag, json.dumps(summary[tag]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
