"""PrimeSim world model, step 4: generate held-out windows with a fine-tuned checkpoint and compare with the recording.

    cd <primesim src> && . ./env.sh && .venv2/bin/python <this repo>/scripts/ps_rollout.py step=3000 [episodes=8 windows=2 denoise_steps=30 videos=4]

Each window is 33 frames at 10 fps (first frame given, 32 generated) for the three cameras. Three command streams are compared:
recorded = the operator's commands, zero = all-zero commands, swapped = the commands of another window.
Baseline: the first frame repeated. Error = mean absolute pixel difference in grey levels (0-255).
Output: reports/world_model_primesim/step<N>/{summary.md, metrics.json, window_<k>.mp4 (recording | recorded | zero | swapped)}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from sharpa.config import load_config  # noqa: E402

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
P = load_config()["primesim_world_model"]
P["task"] = ARGS.get("task", P["task"])            # task=<name>: another task's held-out shard
MODEL = ARGS.get("model", "sharpa_eraser")              # model=<run name of ps_train.py>
SUFFIX = ("" if MODEL == "sharpa_eraser" else "_" + MODEL) + ("" if "task" not in ARGS else "_" + P["task"])   # of output folders
SRC = Path(P["src"])
sys.path.insert(0, str(SRC))
import ps_transforms  # noqa: E402,F401  (registers SharpaStateAction)

PROMPT = "Complete the bimanual manipulation task."
FRAMES, RAW_PER_FRAME = 33, 3                           # 10 fps windows over 30 Hz data
VIEWS = ["head", "wrist_left", "wrist_right"]
MOVING = 25.0                                           # a pixel "moves" if the recording differs from the first frame by more than this


def heldout_windows(n_episodes, per_episode):
    """The held-out shard through PrimeSim's own data pipeline (same transforms as training); fixed window starts inside episodes."""
    from mmengine import DATASETS, Config
    from omegaconf import OmegaConf
    from primesim.data.xemb_data_module import _episode_index_groups

    s = P["image_size"]
    text = (REPO / "configs" / "primesim" / "sharpa.yaml").read_text().replace("DATA_ROOT", P["data_dir"]).replace(
        "REPO_IDS", str([f"{P['task']}__heldout"]))
    cfg = OmegaConf.merge(OmegaConf.create(dict(num_frames=FRAMES, action_len=FRAMES - 1, img_height=s, img_width=s, fps=10,
                                               action_dim=P["action_dim"], batch_size=1, seed=42)), OmegaConf.create(text))
    loader = Config(OmegaConf.to_container(cfg, resolve=True)).datasets.data_loader
    loader.update(stride=1, shuffle=False, train_ratio=1.0, valid_ratio=0.0, num_workers=0)
    data = DATASETS.build(loader)._train_dataset
    groups = _episode_index_groups(data.datasets[0].dataset)
    span = (FRAMES - 1) * RAW_PER_FRAME + 1
    picks = []
    for g in np.linspace(0, len(groups) - 1, n_episodes).round().astype(int):
        _, episode, frames = groups[g]
        last = len(frames) - span
        if last < 0:
            continue
        for start in np.linspace(0, last, per_episode + 2)[1:-1].round().astype(int):      # interior of the episode
            picks.append((int(episode), int(start), frames[0] + int(start)))
    return data, picks


def build_pipeline(step):
    """Widened base model + the fine-tuned parts (LoRA and action inputs). The prompt is fixed, so the text encoder runs once."""
    from peft import LoraConfig, inject_adapter_in_model
    from primesim.modules.t5_text_encoder import T5TextEncoder
    from primesim.modules.wan_video_vae import WanVideoVAE38
    from primesim.pipeline.primesim_pipeline import HuggingfaceTokenizer, PrimeSimPipeline, _load_model, load_dit
    from safetensors.torch import load_file

    ck, dtype = SRC / "checkpoints", torch.bfloat16
    pipe = PrimeSimPipeline(device="cuda", torch_dtype=dtype)
    pipe.tokenizer = HuggingfaceTokenizer(name=str(ck / "google/umt5-xxl"), seq_len=512, clean="whitespace")
    pipe.text_encoder = _load_model(T5TextEncoder, str(ck / "models_t5_umt5-xxl-enc-bf16.pth"), dtype, "cuda")
    with torch.no_grad():
        context = pipe.encode_prompt([PROMPT]).detach()
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    pipe.encode_prompt = lambda prompt: context.expand(len(prompt), -1, -1)

    pipe.vae = _load_model(WanVideoVAE38, str(ck / "Wan2.2_VAE.pth"), dtype, "cuda", key_prefix="model.")
    dit = load_dit(str(ck / f"primesim_base_{P['action_dim']}.safetensors"), P["action_dim"], False, dtype, "cuda")
    dit = inject_adapter_in_model(LoraConfig(r=32, lora_alpha=32, target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"]), dit)
    weights = load_file(str(SRC / "models" / MODEL / f"step-{step}.safetensors"))
    result = dit.load_state_dict(weights, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys[:5]
    pipe.dit = dit.to(device="cuda", dtype=dtype).eval().requires_grad_(False)
    print(f"loaded step-{step}: {len(weights)} fine-tuned tensors")
    return pipe


def to_uint8(video):                                    # [C, T, H, W] in [-1, 1] -> [T, H, W, C] uint8
    return ((video.float().clamp(-1, 1) + 1) * 127.5).round().permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)


def errors(generated, real):
    """generated, real: [T, 3*H, W, C] uint8 (views stacked vertically). Per view: all pixels / moving pixels, early (<= 1.2 s) and whole."""
    h = real.shape[1] // len(VIEWS)
    diff = np.abs(generated[1:].astype(np.float32) - real[1:].astype(np.float32)).mean(-1)          # [T-1, 3H, W]
    moved = np.abs(real[1:].astype(np.float32) - real[:1].astype(np.float32)).mean(-1) > MOVING
    out = {}
    for v, name in enumerate(VIEWS):
        d, m = diff[:, v * h:(v + 1) * h], moved[:, v * h:(v + 1) * h]
        out[name] = dict(all=float(d.mean()), early=float(d[:12].mean()), moving=float(d[m].mean()) if m.any() else float("nan"),
                         moving_share=float(m.mean()))
    return out


def main():
    import imageio.v2 as imageio

    step = int(ARGS.get("step", 3000))
    data, picks = heldout_windows(int(ARGS.get("episodes", 8)), int(ARGS.get("windows", 2)))
    n_videos, denoise = int(ARGS.get("videos", 4)), int(ARGS.get("denoise_steps", 30))
    out_dir = REPO / "reports" / "world_model_primesim" / f"step{step}{SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(step)
    samples = [data[i] for _, _, i in picks]
    s, rows = P["image_size"], []
    for k, ((episode, start, _), sample) in enumerate(zip(picks, samples)):
        videos = sample["videos"]                                                            # [V, C, T, H, W]
        real = to_uint8(torch.cat(list(videos), dim=-2))                                     # views stacked vertically
        streams = {"recorded": sample["action"], "zero": torch.zeros_like(sample["action"]),
                   "swapped": samples[(k + len(samples) // 2) % len(samples)]["action"]}
        row = {"episode": episode, "start_frame": start, "first_frame": errors(np.repeat(real[:1], len(real), 0), real)}
        panels = [real]
        for name, action in streams.items():
            with torch.inference_mode():
                video = pipe([PROMPT], input_images=videos[None, :, :, 0].to("cuda", torch.bfloat16), seed=0, height=s, width=s,
                             num_frames=FRAMES, num_inference_steps=denoise, action=action[None].to("cuda", torch.bfloat16),
                             progress_bar_cmd=lambda x: x, output_type="floatpoint")
            generated = to_uint8(video[0])
            row[name] = errors(generated, real)
            row[name]["vs_recorded"] = float(np.abs(generated[1:].astype(np.float32) - panels[1][1:]).mean()) if len(panels) > 1 else 0.0
            panels.append(generated)
        rows.append(row)
        if k < n_videos:
            imageio.mimsave(out_dir / f"window_{k}.mp4", list(np.concatenate(panels, axis=2)), fps=10, codec="libx264",
                            macro_block_size=None)
        print(k, episode, start, {n: round(row[n]["head"]["all"], 1) for n in ["first_frame", *streams]}, flush=True)

    names = ["first_frame", "recorded", "zero", "swapped"]
    mean = lambda n, v, key: float(np.nanmean([r[n][v][key] for r in rows]))                # noqa: E731
    lines = [f"# PrimeSim on held-out eraser windows, checkpoint step {step}", "",
             f"{len(rows)} windows of 3.2 s from {len({r['episode'] for r in rows})} held-out episodes, {denoise} denoising steps. "
             "Mean absolute pixel error in grey levels (0-255) over the 32 generated frames; "
             f"\"moving\" = pixels where the recording differs from the first frame by more than {MOVING:.0f}.", ""]
    for key, title in [("all", "All pixels, 3.2 s"), ("early", "All pixels, first 1.2 s"), ("moving", "Moving pixels, 3.2 s")]:
        lines += [f"## {title}", "| Source | " + " | ".join(VIEWS) + " |", "|---|" + "---|" * len(VIEWS)]
        lines += [f"| {n} | " + " | ".join(f"{mean(n, v, key):.1f}" for v in VIEWS) + " |" for n in names] + [""]
    lines += ["## Difference from the video generated with the recorded commands (all views)",
              *[f"- {n}: {np.mean([r[n]['vs_recorded'] for r in rows]):.1f}" for n in ["zero", "swapped"]], "",
              f"Share of moving pixels: " + ", ".join(f"{v} {100 * mean('recorded', v, 'moving_share'):.0f}%" for v in VIEWS)]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
