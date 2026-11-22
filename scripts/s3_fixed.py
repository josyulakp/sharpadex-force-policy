"""Stage 3: is the gain a reaction to the scene, or a fixed adjustment that any scene would get?

    python scripts/s3_fixed.py [name=rl] [split=heldout] [jobs=6]     (after scripts/s3_eval.py with the same split)
        -> runs/stage3/<name>/fixed_<split>.json and a table on stdout

Runs the trained actor once on every scene of the split (first perturbation) and records the corrections it applies. Two
open-loop policies are built from them and evaluated on the perturbations of scripts/s3_eval.py (same seeds):
  constant    the actor's mean correction, held for the whole episode
  two_phase   its mean during the frames the recording labels as wiping, and its mean outside them
Both are measured on the scenes they are evaluated in, which favours them. Intervals: bootstrap over scenes, paired.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.s3_eval import interval  # noqa: E402
from scripts.s3_train import make_actor, run_episode, scenes  # noqa: E402


def episode(job: dict) -> dict:
    from sim.env import ResidualEnv
    torch.set_num_threads(1)
    run = REPO / "runs" / "stage3" / job["name"]
    saved = yaml.safe_load((run / "config.yaml").read_text())
    s3 = saved["stage3"]
    env = ResidualEnv(saved["sim"], s3, dict(np.load(run / "synergy.npz")))
    if job["policy"] == "stage3":
        ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
        actor = make_actor(s3, tuple(ck["dims"]))
        actor.load_state_dict(ck["actor"])
    else:
        fixed = {k: np.asarray(v) for k, v in job["fixed"].items()}

        def actor(obs):
            wiping = env.wiping[min(env.t, len(env.wiping) - 1)]
            return torch.from_numpy(fixed["all"] if job["policy"] == "constant" else fixed["wipe" if wiping else "other"])[None]
    i, k = job["scene"], job["repeat"]
    ep = run_episode(env, actor, {"scene": scenes(s3)[job["split"]][i], "seed": 1000 * i + k, "perturb": True, "noise": (0.0, 0.0), "zero": False,
                                  "stream": "base", "kind": job["policy"], "index": (i, k)})
    n = len(ep["act"])
    return {"policy": job["policy"], "scene": i, "repeat": k, "success": bool(ep["outcome"]["success"]), "wipe_path_smooth_cm": ep["outcome"]["wipe_path_smooth_cm"],
            "act": ep["act"], "wiping": env.wiping[: n * s3["env"]["decision_frames"]: s3["env"]["decision_frames"]][:n]}


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    name, split, jobs = args.get("name", "rl"), args.get("split", "heldout"), int(args.get("jobs", 6))
    run = REPO / "runs" / "stage3" / name
    ev = json.loads((run / f"eval_{split}.json").read_text())
    n, seeds = ev["scenes"], ev["seeds"]
    with ProcessPoolExecutor(jobs) as ex:
        probe = list(ex.map(episode, [{"name": name, "split": split, "policy": "stage3", "scene": i, "repeat": 1} for i in range(n)]))
        act, wiping = np.concatenate([p["act"] for p in probe]).astype(np.float64), np.concatenate([p["wiping"] for p in probe])
        fixed = {"all": act.mean(0).tolist(), "wipe": act[wiping].mean(0).tolist(), "other": act[~wiping].mean(0).tolist()}
        within = float(np.sqrt(np.mean([p["act"].var(0) for p in probe])))
        rows = list(ex.map(episode, [{"name": name, "split": split, "policy": p, "scene": i, "repeat": k, "fixed": fixed}
                                     for i in range(n) for k in range(1, seeds + 1) for p in ("constant", "two_phase")]))
    rows = [{k: v for k, v in r.items() if k not in ("act", "wiping")} for r in rows]
    rows += [{k: e[k] for k in ("policy", "scene", "repeat", "success", "wipe_path_smooth_cm")} for e in ev["episodes"] if e["repeat"] > 0]
    grid = lambda p, key: np.array([[float(next(r[key] for r in rows if r["policy"] == p and r["scene"] == i and r["repeat"] == k))  # noqa: E731
                                     for k in range(1, seeds + 1)] for i in range(n)])
    rng = np.random.default_rng(0)
    pols = ("frozen", "constant", "two_phase", "stage3")
    summary = {key: {p: interval(grid(p, key), rng) for p in pols} | {f"{p} - frozen": interval(grid(p, key) - grid("frozen", key), rng) for p in pols[1:]}
                    | {f"stage3 - {p}": interval(grid("stage3", key) - grid(p, key), rng) for p in pols[1:3]} for key in ("success", "wipe_path_smooth_cm")}
    out = {"name": name, "split": split, "scenes": n, "seeds": seeds, "fixed": fixed, "correction_mean_abs": float(np.abs(act).mean()),
           "correction_std_within_episode": within, "summary": summary, "episodes": rows}
    (run / f"fixed_{split}.json").write_text(json.dumps(out))
    print(f"corrections of the actor (units of the box): mean |correction| {out['correction_mean_abs']:.2f}, largest |mean| over the 12 dimensions "
          f"{np.abs(act.mean(0)).max():.2f}, standard deviation within an episode {within:.2f}")
    for key, s in summary.items():
        print(key)
        for p, v in s.items():
            print(f"  {p:22s} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")


if __name__ == "__main__":
    main()
