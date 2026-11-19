"""Encode every camera frame once with the frozen vision encoder and cache the tokens.

    python scripts/cache_features.py [key=value ...]

Per episode it writes  <cache_dir>/<encoder>_g<grid>/<task>/<season>/episode_XXXXXX.npy
with shape [T, n_cameras, 1 + grid*grid, D] (float16): the CLS token plus the patch tokens
average-pooled to a grid x grid map. Training never touches the videos again.
"""
from __future__ import annotations

import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import parse_cli  # noqa: E402
from sharpa.data.episodes import index_task  # noqa: E402
from sharpa.data.features import feature_path  # noqa: E402


def decode_video(path: Path, size: int) -> np.ndarray:
    """All frames of one mp4, resized to size x size, uint8 [T, size, size, 3]."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return np.stack([f.to_ndarray(width=size, height=size, format="rgb24") for f in container.decode(stream)])


class EpisodeVideos(torch.utils.data.Dataset):
    def __init__(self, refs, cameras, size):
        self.refs, self.cameras, self.size = refs, cameras, size

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        ref = self.refs[i]
        p = Path(ref.path)                       # .../lerobot_v2.1/data/chunk-000/episode_000012.parquet
        vids = p.parents[2] / "videos" / p.parent.name
        frames = []
        for cam in self.cameras:
            v = decode_video(vids / f"observation.images.{cam}" / (p.stem + ".mp4"), self.size)
            if len(v) != ref.num_frames:
                raise RuntimeError(f"{ref.uid} {cam}: {len(v)} video frames vs {ref.num_frames} data frames")
            frames.append(torch.from_numpy(v))
        return i, torch.stack(frames)            # [C, T, H, W, 3] uint8


@torch.no_grad()
def encode(model, frames: torch.Tensor, grid: int, batch: int) -> np.ndarray:
    """[N, H, W, 3] uint8 -> [N, 1 + grid*grid, D] float16."""
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    out = []
    for s in range(0, len(frames), batch):
        x = frames[s:s + batch].cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        x = ((x - mean) / std).half()
        tok = model.forward_features(x)                       # [B, 1 + P, D]
        cls, patch = tok[:, :1], tok[:, model.num_prefix_tokens:]
        side = int(patch.shape[1] ** 0.5)
        patch = patch.transpose(1, 2).reshape(len(x), -1, side, side)
        patch = F.adaptive_avg_pool2d(patch.float(), grid).flatten(2).transpose(1, 2)
        out.append(torch.cat([cls.float(), patch], 1).half().cpu())
    return torch.cat(out).numpy()


def main():
    import timm
    cfg = parse_cli(sys.argv[1:])
    d, v = cfg["data"], cfg["vision"]
    refs = [r for t in d["tasks"] for r in index_task(d["root"], t, d["format"])]
    todo = [r for r in refs if not feature_path(cfg, r).exists()]
    print(f"{len(refs)} episodes, {len(todo)} to encode")
    if not todo:
        return
    model = timm.create_model(v["encoder"], pretrained=True, img_size=v["image_size"], num_classes=0).cuda().eval().half()
    loader = torch.utils.data.DataLoader(EpisodeVideos(todo, v["cameras"], v["image_size"]), batch_size=None,
                                         shuffle=False, num_workers=v["num_workers"], prefetch_factor=2)
    for n, (i, frames) in enumerate(loader):
        ref = todo[i]
        C, T = frames.shape[:2]
        feats = encode(model, frames.flatten(0, 1), v["pool_grid"], v["batch_size"])
        feats = feats.reshape(C, T, *feats.shape[1:]).transpose(1, 0, 2, 3)      # [T, C, tokens, D]
        out = feature_path(cfg, ref)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, feats)
        tmp.rename(out)                                                           # atomic: no half-written files
        if n % 25 == 0:
            print(f"[{n + 1}/{len(todo)}] {ref.uid} {feats.shape}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
