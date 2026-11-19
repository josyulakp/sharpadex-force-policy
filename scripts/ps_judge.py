"""PrimeSim world model, step 8: can generated video be scored? The stage-2 judge (OpenAI API) counts finished subgoals.

    python scripts/ps_judge.py [policy=policy_three_tasks_no_force_seed0 replay=long_step6000 skip=96]

For the held-out episodes rolled out by scripts/ps_policy.py, once per second:
  recording        the judge's count on the real head-camera video (already stored by rl_judge_api.py)
  operator in WM   the same commands replayed in the video model (chained column of scripts/ps_long.py)
  policy in WM     the policy acting in the video model
next to the dataset's own annotation. Sends generated head-camera frames and the task text to api.openai.com; nothing else.
Output: reports/world_model_primesim/<policy>/judge.md + judge.json
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_judge_api as A  # noqa: E402
from rl_judge import Episodes, J, build_splits, cfg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WM = REPO / "reports" / "world_model_primesim"


def main():
    import os
    key = A.API_KEY or os.environ.get("OPENAI_API_KEY", "")
    policy_dir, replay_dir, skip = WM / J.get("policy", "policy_three_tasks_no_force_seed0"), WM / J.get("replay", "long_step6000"), int(J.get("skip", 96))
    rows = json.loads((policy_dir / "metrics.json").read_text())
    refs = {r.uid: r for t in cfg["data"]["tasks"] for r in build_splits(cfg)[t]["heldout"]}
    size, out = J["image_size"], []
    with ThreadPoolExecutor(A.A["workers"]) as pool:
        for k, row in enumerate(rows):
            ref = refs[row["uid"]]
            ep = Episodes([ref])[0]
            S = len(ep["steps"])
            text, first = A.question(ep)
            stored = np.load(Path(cfg["rl"]["judge_dir"]) / (A.A["model"] + "__api") / ref.task / ref.season / f"{ref.episode_index:06d}.npz")
            sources = {"policy in WM": np.load(policy_dir / f"episode_{k}.npz")["head"]}
            with av.open(str(replay_dir / f"episode_{k}.mp4")) as c:                         # columns: recording | chained | receding
                sources["operator in WM"] = np.stack([f.to_ndarray(format="rgb24")[:256, 256:512] for f in c.decode(video=0)])
            n = min(len(v) for v in sources.values())                                       # generated frames at 10 fps from `skip`
            ts = [t for t in ep["ts"] if skip < t <= skip + 3 * (n - 1)]
            res = {"episode": row["episode"], "uid": row["uid"], "steps": S, "t_seconds": [t / 30 for t in ts],
                   "annotation": [int(ep["truth"][ep["ts"].index(t)]) for t in ts],
                   "recording": [int(stored["done"][list(stored["t"]).index(t)]) for t in ts]}
            for name, video in sources.items():
                def images(t):
                    return [Image.fromarray(video[max(t - b - skip, 0) // 3]).resize((size, size), Image.BICUBIC) for b in [*J["lookback"], 0]]
                raw = np.array(list(pool.map(lambda t: A.ask(key, text, images(t), S, first), ts)))
                res[name] = np.maximum.accumulate(np.where(raw < 0, 0, raw)).tolist()
            out.append(res)
            print(k, {n_: res[n_] for n_ in ("annotation", "recording", "operator in WM", "policy in WM")}, flush=True)
    names = ["annotation", "recording", "operator in WM", "policy in WM"]
    lines = ["# Subgoals counted by the judge on generated video", "",
             f"Judge: {A.A['model']} (effort {A.A['effort']}), once per second, head camera. Count of finished subgoals at the end of the rollout, "
             "and share of judged seconds where the count equals the dataset's annotation.", "",
             "| Episode | Length judged | Steps in task | " + " | ".join(f"{n}: final" for n in names) + " | " +
             " | ".join(f"{n}: agreement" for n in names[1:]) + " |", "|---|---|---|" + "---|" * (len(names) * 2 - 1)]
    for r in out:
        ann = np.array(r["annotation"])
        lines.append(f"| {r['episode']} | {r['t_seconds'][-1] - r['t_seconds'][0] + 1:.0f} s | {r['steps']} | " + " | ".join(str(r[n][-1]) for n in names) + " | " +
                     " | ".join(f"{100 * (np.array(r[n]) == ann).mean():.0f}%" for n in names[1:]) + " |")
    (policy_dir / "judge.md").write_text("\n".join(lines) + "\n")
    (policy_dir / "judge.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
