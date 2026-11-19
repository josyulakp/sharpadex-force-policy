"""Decode the camera frames the video world model trains on, once.

    python scripts/wm_prepare.py      -> <data_dir>/<task>/<season>/<episode>.npy  uint8 [T/stride, S, S, 3]
                                         <data_dir>/<task>/<season>/<episode>.npz  state, action [T/stride, 58] + uid, split
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import load_config  # noqa: E402
from sharpa.data.episodes import load_numeric  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402

cfg = load_config()
W = cfg["video_world_model"]


def one(job):
    ref, split = job
    out = Path(W["data_dir"]) / ref.task / ref.season / f"{ref.episode_index:06d}"
    if out.with_suffix(".npz").exists():
        return 0
    p = Path(ref.path)
    video = p.parents[2] / "videos" / p.parent.name / f"observation.images.{W['camera']}" / (p.stem + ".mp4")
    s, k = W["image_size"], W["frame_stride"]
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        frames = np.stack([f.to_ndarray(width=s, height=s, format="rgb24") for i, f in enumerate(c.decode(stream)) if i % k == 0])
    num = load_numeric(ref)
    state, action = num["observation.state"][::k, :58], num["action"][::k, :58]
    n = min(len(frames), len(state))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), frames[:n])
    np.savez(out.with_suffix(".npz"), state=state[:n], action=action[:n], uid=ref.uid, split=split)
    return n


if __name__ == "__main__":
    c = dict(cfg)
    c["data"] = dict(cfg["data"], tasks=W["tasks"])
    jobs = [(r, split) for t, s in build_splits(c).items() for split in ("train", "heldout") for r in s[split]]
    with Pool(32) as pool:
        n = sum(pool.imap_unordered(one, jobs, chunksize=2))
    print(f"{len(jobs)} episodes, {n} new frames -> {W['data_dir']}")
