"""Tactile world model, step 1: SharpaDex eraser episodes in TacForeSight's zarr layout.

    python scripts/tf_prepare.py      -> <tactile_world_model.data_dir>/{train,val}/replay_buffer.zarr   (val = held-out episodes)

data/tactile_left, tactile_right  [N, 700, 3] uint8   deformation images of the five fingertips of a hand, each reduced to 7 x 20, stacked
data/command_next_4x              [4N, 58]            commanded joints `horizon` frames later (what the hand is about to be told), each frame x 4
data/state_4x                     [4N, 58]            measured joints, each frame x 4 (TacForeSight expects 120 Hz streams; SharpaDex has 30 Hz)
data/wrench, torque, state, action [N, 60 / 58 / 58 / 58]   for the read-out and the checks (not used by TacForeSight)
meta/episode_ends, episode_ends_4x, and uids.json (episode order)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import load_config  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402

CFG = load_config()
T = CFG["tactile_world_model"]
sys.path.insert(0, T["packages"])
COLS = ["observation.state", "action", "observation.tactile", "observation.state.joint_torque"]


def episode(path):
    p = Path(path)
    video = p.parents[2] / "videos" / p.parent.name / "observation.images.tactile_deform" / (p.stem + ".mp4")
    rows, maps = T["map_rows"], []
    with av.open(str(video)) as c:
        c.streams.video[0].thread_type = "AUTO"
        for f in c.decode(video=0):
            g = f.to_ndarray(format="rgb24").reshape(2, 240, 5, 240, 3)                    # hand, y, finger, x, rgb
            g = g.reshape(2, rows, 240 // rows, 5, 20, 12, 3)[:, :, :240 // rows // 1].mean((2, 5)) if 240 % rows == 0 else None
            if g is None:                                                                  # 240 is not a multiple of rows: crop to the largest multiple
                k = 240 // rows
                g = f.to_ndarray(format="rgb24").reshape(2, 240, 5, 240, 3)[:, :rows * k].reshape(2, rows, k, 5, 20, 12, 3).mean((2, 5))
            maps.append(np.round(g.transpose(0, 2, 1, 3, 4).reshape(2, 5 * rows * 20, 3)).astype(np.uint8))   # per hand: finger, row, col
    t = pq.read_table(p, columns=COLS)
    state, action, wrench, torque = (np.stack(t.column(c).to_numpy(zero_copy_only=False)).astype(np.float32) for c in COLS)
    maps = np.stack(maps)
    assert len(maps) == len(state), (path, len(maps), len(state))
    return maps, state[:, :58], action[:, :58], wrench, torque[:, :58]


def main():
    import zarr
    splits = build_splits(CFG)[T["task"]]
    h = T["horizon"]
    for part, refs in (("val", splits["heldout"]), ("train", splits["train"])):
        with Pool(24) as pool:
            eps = pool.map(episode, [r.path for r in refs], chunksize=2)
        ends = np.cumsum([len(e[0]) for e in eps])
        root = zarr.open_group(str(Path(T["data_dir"]) / part / "replay_buffer.zarr"), mode="w")
        data, meta = root.create_group("data"), root.create_group("meta")
        cat = lambda i: np.concatenate([e[i] for e in eps])  # noqa: E731
        maps = cat(0)
        data.array("tactile_left", maps[:, 0], chunks=(128, 700, 3), compressor=None)      # small plain chunks: windows are read at random
        data.array("tactile_right", maps[:, 1], chunks=(128, 700, 3), compressor=None)
        nxt = np.concatenate([e[2][np.minimum(np.arange(len(e[2])) + h, len(e[2]) - 1)] for e in eps])
        data.array("command_next_4x", np.repeat(nxt, 4, 0), chunks=(1024, 58), compressor=None)
        data.array("state_4x", np.repeat(cat(1), 4, 0), chunks=(1024, 58), compressor=None)
        for name, i in (("state", 1), ("action", 2), ("wrench", 3), ("torque", 4)):
            data.array(name, cat(i), chunks=(1024, eps[0][i].shape[1]), compressor=None)
        meta.array("episode_ends", ends)
        meta.array("episode_ends_4x", ends * 4)
        (Path(T["data_dir"]) / part / "uids.json").write_text(json.dumps([r.uid for r in refs]))
        print(part, len(refs), "episodes,", int(ends[-1]), "frames; mean map value", float(maps.mean()), flush=True)


if __name__ == "__main__":
    main()
