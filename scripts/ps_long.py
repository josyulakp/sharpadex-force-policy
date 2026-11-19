"""PrimeSim world model, step 5: whole held-out episodes generated window after window (operator's commands).

    cd <primesim src> && . ./env.sh && .venv2/bin/python <this repo>/scripts/ps_long.py step=6000 [episodes=4 modes=chained,reanchored,receding videos=3 denoise_steps=30 skip=0 tag=]

Every window is 33 frames at 10 fps (the length the model was trained on). Only the first frame of the episode is real, except in `reanchored`.
chained:    all 32 generated frames are kept; the next window starts from the last one.
reanchored: like chained, but every window starts from the real frame (the 3.2 s test repeated along the episode).
receding:   only the first 5 frames (0.5 s) are kept and the commands after them are "hold the last command": what a policy
            that plans 0.5 s ahead can supply. The next window starts from generated frame 5.
skip = frames of the recording skipped before the start (the head and torso move at the very start, and their motion is not
part of the commands). Commands are the recorded ones, expressed relative to the recorded state at the start of each window.
Output: reports/world_model_primesim/long_step<N>/{summary.md, metrics.json, episode_<k>.mp4 (recording | chained | receding)}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ps_rollout as R  # noqa: E402
from ps_transforms import SharpaStateAction  # noqa: E402

RAW, FRAMES = R.RAW_PER_FRAME, R.FRAMES
BUCKET = 32                                             # error reported per 3.2 s of episode time
MODES = {"chained": (FRAMES - 1, False, False), "reanchored": (FRAMES - 1, True, False), "receding": (5, False, True),   # keep, anchor, hold
         "receding_known": (5, False, False), "receding16": (16, False, True)}       # diagnostics: later commands known / 1.6 s kept


def episode_spans(data, n):
    from primesim.data.xemb_data_module import _episode_index_groups
    groups = _episode_index_groups(data.datasets[0].dataset)
    return [(int(groups[g][1]), groups[g][2][0], len(groups[g][2])) for g in np.linspace(0, len(groups) - 1, n).round().astype(int)]


def raw_streams(data):
    """Absolute 75-D state and command of every 30 Hz frame of the held-out shard."""
    table = data.datasets[0].dataset._dataset.datasets[0].hf_dataset.with_format("numpy")[:]
    return torch.from_numpy(np.stack(table["observation.state"])).float(), torch.from_numpy(np.stack(table["action"])).float()


_REL = SharpaStateAction(FRAMES, FRAMES - 1)


def relative(command, reference):
    """command [32, 75] absolute, reference [75] absolute state -> the model's action input (same code as in training)."""
    return _REL({"state_vec": reference[None].repeat(FRAMES, 1), "action_vec": command})["action"]


def generate(pipe, first, action, size, denoise):
    """first [V, C, H, W] in [-1, 1], action [32, D] -> [V, C, 33, H, W] in [-1, 1]."""
    with torch.inference_mode():
        video = pipe([R.PROMPT], input_images=first[None].to("cuda", torch.bfloat16), seed=0, height=size, width=size, num_frames=FRAMES,
                     num_inference_steps=denoise, action=action[None].to("cuda", torch.bfloat16), progress_bar_cmd=lambda x: x,
                     output_type="floatpoint")[0].float().clamp(-1, 1).cpu()                 # [C, T, V*H, W]
    return torch.stack(video.split(size, dim=2))                                             # [V, C, T, H, W]


def window_commands(command, start, last, keep, hold):
    """Absolute commands of the 32 steps of the window starting at raw frame `start` (frames beyond the episode repeat the last one)."""
    idx = torch.clamp(start + RAW * torch.arange(FRAMES - 1), max=last)
    if hold:
        idx[keep:] = idx[keep - 1]
    return command[idx]


def rollout(pipe, data, streams, first_index, length, mode, size, denoise, skip=0):
    """One episode. Returns generated and real video [V, C, T, H, W] at 10 fps."""
    keep, anchor, hold = MODES[mode]
    state, command = streams
    generated, real, first = [], [], None
    for w in range((length - 1 - skip) // (keep * RAW)):
        i = first_index + skip + w * keep * RAW
        videos = data[i]["videos"]
        if first is None or anchor:
            first = videos[:, :, 0]
        action = relative(window_commands(command, i, first_index + length - 1, keep, hold), state[i])
        out = generate(pipe, first, action, size, denoise)[:, :, :keep + 1]
        first = out[:, :, -1]
        generated.append(out if w == 0 else out[:, :, 1:])
        real.append(videos[:, :, :keep + 1] if w == 0 else videos[:, :, 1:keep + 1])
    return torch.cat(generated, 2), torch.cat(real, 2)


def stacked(video):                                     # [V, C, T, H, W] -> [T, V*H, W, C] uint8
    return R.to_uint8(torch.cat(list(video), dim=-2))


def blocks(diff, h):                                    # [T-1, V*H, W] -> per view, mean per complete 3.2 s block
    return {v: [float(diff[b:b + BUCKET, i * h:(i + 1) * h].mean()) for b in range(0, BUCKET * (len(diff) // BUCKET), BUCKET)]
            for i, v in enumerate(R.VIEWS)}


def main():
    import imageio.v2 as imageio

    step, denoise = int(R.ARGS.get("step", 6000)), int(R.ARGS.get("denoise_steps", 30))
    modes = R.ARGS.get("modes", "chained,reanchored,receding").split(",")
    n_videos, size, skip = int(R.ARGS.get("videos", 3)), R.P["image_size"], int(R.ARGS.get("skip", 0))
    data, _ = R.heldout_windows(1, 1)
    streams = raw_streams(data)
    episodes = episode_spans(data, int(R.ARGS.get("episodes", 4)))
    out_dir = R.REPO / "reports" / "world_model_primesim" / (f"long_step{step}" + R.ARGS.get("tag", ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = R.build_pipeline(step)
    rows = []
    for k, (episode, first_index, length) in enumerate(episodes):
        row, panels = {"episode": episode, "seconds": length / 30}, []
        for mode in modes:
            generated, real = rollout(pipe, data, streams, first_index, length, mode, size, denoise, skip)
            g, r = stacked(generated), stacked(real)
            h = r.shape[1] // len(R.VIEWS)
            row[mode] = blocks(np.abs(g.astype(np.float32) - r).mean(-1)[1:], h)
            row.setdefault("first_frame", blocks(np.abs(r[:1].astype(np.float32) - r).mean(-1)[1:], h))
            if mode != "reanchored":
                panels += ([r] if not panels else []) + [g]
            print(k, episode, mode, [round(x, 1) for x in row[mode]["head"]], "first frame", [round(x, 1) for x in row["first_frame"]["head"]],
                  flush=True)
        rows.append(row)
        if k < n_videos and panels:
            t = min(len(p) for p in panels)
            imageio.mimsave(out_dir / f"episode_{k}.mp4", list(np.concatenate([p[:t] for p in panels], axis=2)), fps=10, codec="libx264",
                            macro_block_size=None)
    names = ["first_frame", *modes]
    n_blocks = min(len(r[n]["head"]) for r in rows for n in names)
    lines = [f"# PrimeSim on whole held-out eraser episodes, checkpoint step {step}", "",
             f"{len(rows)} held-out episodes ({min(r['seconds'] for r in rows):.0f}-{max(r['seconds'] for r in rows):.0f} s), operator's commands, "
             f"{denoise} denoising steps, started {skip / 30:.1f} s into the recording. Mean absolute pixel error in grey levels (0-255) per 3.2 s of episode time, for the first {n_blocks} "
             "blocks (present in every episode). first_frame = the first frame of the episode repeated. chained: 3.2 s windows, each started from "
             "the last generated frame. reanchored: every window started from the real frame. receding: 0.5 s kept per window, later commands held.", ""]
    for v in R.VIEWS:
        lines += [f"## {v}", "| Source | " + " | ".join(f"{3.2 * b:.1f}-{3.2 * (b + 1):.1f} s" for b in range(n_blocks)) + " |",
                  "|---|" + "---|" * n_blocks]
        lines += [f"| {n} | " + " | ".join(f"{np.mean([r[n][v][b] for r in rows]):.1f}" for b in range(n_blocks)) + " |" for n in names] + [""]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
