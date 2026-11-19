"""PrimeSim world model, step 1: write SharpaDex episodes as LeRobot v3 datasets in PrimeSim's layout.

    <primesim src>/.venv/bin/python scripts/ps_prepare.py list [task]          -> shard names (one per season, plus `heldout`)
    <primesim src>/.venv/bin/python scripts/ps_prepare.py <shard> [task]       -> <primesim_world_model.data_dir>/<task>__<shard>/

Per frame: three cameras (head_left, wrist_left, wrist_right) resized to image_size, and a 75-D state and action:
    [ 0:31]  PrimeSim's shared layout: left hand pose xyz + quaternion xyzw, left "gripper" (mean finger angle),
             the same for the right hand; torso / head / waist / base stay zero (fixed here)
    [31:75]  the 44 finger joints (left 22, right 22), which the shared layout has no place for
State = measured, action = commanded (absolute; PrimeSim's pipeline makes them relative to the first frame).
"""
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import load_config  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402

CAMS = {"head": "head_left", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}
L_HAND, R_HAND = slice(7, 29), slice(36, 58)


def wide(pose12: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """tcp pose [T, 12] (xyz + rotation vector per hand) and joints [T, 65] -> [T, 75]."""
    out = np.zeros((len(pose12), 75), np.float32)
    for k, (o, hand) in enumerate(((0, L_HAND), (8, R_HAND))):
        p = pose12[:, 6 * k:6 * k + 6]
        out[:, o:o + 3] = p[:, :3]
        out[:, o + 3:o + 7] = Rotation.from_rotvec(p[:, 3:]).as_quat()   # xyzw
        out[:, o + 7] = joints[:, hand].mean(1)
    out[:, 31:53], out[:, 53:75] = joints[:, L_HAND], joints[:, R_HAND]
    return out


def main():
    cfg = load_config()
    P = cfg["primesim_world_model"]
    splits = build_splits(cfg)
    task = sys.argv[2] if len(sys.argv) > 2 else P["task"]
    seasons = sorted({r.season for r in splits[task]["train"]})
    if sys.argv[1] == "list":
        print(" ".join(seasons + ["heldout"]))
        return
    shard = sys.argv[1]
    refs = splits[task]["heldout"] if shard == "heldout" else [r for r in splits[task]["train"] if r.season == shard]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    s = P["image_size"]
    root = Path(P["data_dir"]) / f"{task}__{shard}"
    if root.exists():
        shutil.rmtree(root)
    feats = {f"observation.images.{k}": {"dtype": "video", "shape": (s, s, 3), "names": ["height", "width", "channels"]} for k in CAMS}
    feats |= {k: {"dtype": "float32", "shape": (75,), "names": None} for k in ("observation.state", "action")}
    ds = LeRobotDataset.create(root.name, fps=cfg["data"]["fps"], features=feats, root=root, robot_type="sharpa", image_writer_threads=12)
    uids = []
    for n, ref in enumerate(refs):
        p = Path(ref.path)
        t = pq.read_table(p, columns=["observation.state", "action", "observation.state.tcp_pose", "action.tcp_pose"])
        col = lambda c: np.stack(t.column(c).to_numpy(zero_copy_only=False)).astype(np.float32)  # noqa: E731
        state, action = wide(col("observation.state.tcp_pose"), col("observation.state")), wide(col("action.tcp_pose"), col("action"))
        instr = json.loads((p.parents[2] / "meta" / "tasks.jsonl").read_text().splitlines()[0])["task"].split("Instruction:")[1].split("|")[0].strip()
        streams = [av.open(str(p.parents[2] / "videos" / p.parent.name / f"observation.images.{c}" / (p.stem + ".mp4"))) for c in CAMS.values()]
        its = [c.decode(c.streams.video[0]) for c in streams]
        for i in range(len(state)):
            frame = {"observation.state": state[i], "action": action[i], "task": instr}
            for k, it in zip(CAMS, its):
                frame[f"observation.images.{k}"] = np.asarray(next(it).to_image().resize((s, s), Image.BICUBIC))
            ds.add_frame(frame)
        for c in streams:
            c.close()
        ds.save_episode()
        uids.append(ref.uid)
        if n % 10 == 0 or n == len(refs) - 1:
            print(f"{shard}: {n + 1}/{len(refs)} episodes", flush=True)
    ds.finalize()
    (root / "meta" / "sharpa_episodes.json").write_text(json.dumps(uids))       # episode_index -> SharpaDex episode


if __name__ == "__main__":
    main()
