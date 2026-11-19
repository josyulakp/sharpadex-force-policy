"""Tactile world model, step 4: TacForeSight latents of every SharpaDex frame, as input tokens of the policy's tactile expert.

    python scripts/tf_features.py      -> <tactile_world_model.data_dir>/latents/<task>/<season>/episode_<n>.npy   [T, 16, 128] float16

For frame t of an episode (all tasks, training and held-out episodes), using only what the robot has at time t:
    tokens 0-7    latents of the fingertip deformation images of frames t-7 .. t (TacForeSight's tokenizer)
    tokens 8-15   latents TacForeSight predicts for frames t+1 .. t+8 (0.27 s ahead) from those, the measured joints, and the
                  last command sent (action[t-1]) held, because the commands of the predicted frames are what the policy has yet to decide
The world model is frozen and was trained on clean_plate_with_eraser only (scripts/tf_train.py).

    python scripts/tf_features.py rest=true   -> <data_dir>/latents_rest/...   held-out episodes only
The same with every fingertip image replaced by the image of the episode's frame with the lowest fingertip force (no contact),
joints and commands unchanged: the tactile expert's input when the evaluation sets the force to zero (evaluate.py zero_force=true).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_prepare as TP  # noqa: E402
import tf_readout as TR  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402

H = TR.H


def latent_path(cfg: dict, ref, name: str = "latents") -> Path:
    return Path(cfg["tactile_world_model"]["data_dir"]) / name / ref.task / ref.season / f"episode_{ref.episode_index:06d}.npy"


@torch.no_grad()
def features(model, norm, maps, state, action):
    T = len(maps)
    pack = lambda x: torch.from_numpy(x.reshape(T, 35, 20, 3)).float()  # noqa: E731
    tac = torch.cat([pack(maps[:, 0]), pack(maps[:, 1])], -1)
    tac = torch.cat([tac, tac[:, -1:]], 1).cuda()                                             # 35 -> 36 rows, [T, 36, 20, 6]
    idx = np.maximum(np.arange(T)[:, None] - np.arange(H - 1, -1, -1), 0)                     # frames t-7 .. t, padded at the start
    held = np.concatenate([state[:1], action[:-1]])                                           # the last command sent before frame t
    out = []
    for s in range(0, T, 256):
        i = idx[s:s + 256]
        b = {"tactile": tac[i], "force_4x": torch.from_numpy(np.repeat(np.repeat(held[s:s + 256, None], H, 1), 4, 1)).cuda(),
             "state_4x": torch.from_numpy(np.repeat(state[i], 4, 1)).cuda()}
        enc = model.encode({k: norm[k].normalize(v) for k, v in b.items()})
        out.append(torch.cat([enc["z"], model.predict(enc["z"], enc["cond"])], 1).half().cpu())
    return torch.cat(out).numpy()


def main():
    rest = TR.ARGS.get("rest", "false") == "true"
    name = "latents_rest" if rest else "latents"
    model, norm = TR.load_model()
    splits = build_splits(TR.CFG)
    refs = [r for t in TR.CFG["data"]["tasks"] for part in (("heldout",) if rest else ("train_full", "heldout")) for r in splits[t][part]]
    refs = [r for r in refs if not latent_path(TR.CFG, r, name).exists()]
    print(len(refs), "episodes to do", flush=True)
    with Pool(24) as pool:
        for n, (ref, (maps, state, action, wrench, _)) in enumerate(zip(refs, pool.imap(TP.episode, [r.path for r in refs], chunksize=2))):
            if rest:
                quiet = int(np.linalg.norm(wrench.reshape(len(wrench), 10, 6)[..., :3], axis=-1).sum(1).argmin())
                maps = np.repeat(maps[quiet:quiet + 1], len(maps), 0)
            f = features(model, norm, maps, state, action)
            p = latent_path(TR.CFG, ref, name)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p.with_suffix(".tmp.npy"), f)
            p.with_suffix(".tmp.npy").rename(p)
            if n % 100 == 0:
                print(n, ref.uid, f.shape, "latent std %.2f, predicted - last real: %.3f" % (f[:, :H].astype(np.float32).std(), np.abs(f[:, H:].astype(np.float32) - f[:, H - 1:H].astype(np.float32)).mean()), flush=True)


if __name__ == "__main__":
    main()
