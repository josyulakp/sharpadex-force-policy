"""Stage 3: evaluate a trained actor against the frozen policy on scenes the actor never trained in.

    python scripts/s3_eval.py [name=rl] [ckpt=ckpt.pt] [seeds=5] [split=heldout] [limit=N] [workers=N]
        -> runs/stage3/<name>/eval_<split>.json and a table on stdout

Every scene is run as reconstructed and with `seeds` perturbations (the training perturbation, fixed seeds, the same for
every policy). Policies: the operator's recorded commands, the frozen four-expert policy, and the frozen policy with the
stage-3 actor's corrections (no exploration noise). Intervals: bootstrap over scenes; the difference is paired by scene.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.s3_train import load_cfg, scenes, worker  # noqa: E402


def interval(x: np.ndarray, rng, n: int = 2000) -> list[float]:
    """x [scenes, repeats] (or a paired difference): mean and 95 % interval, resampling scenes."""
    m = x.mean(1)
    b = m[rng.integers(len(m), size=(n, len(m)))].mean(1)
    return [float(m.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    name, ckpt_name, seeds, split = args.get("name", "rl"), args.get("ckpt", "ckpt.pt"), int(args.get("seeds", 5)), args.get("split", "heldout")
    run = REPO / "runs" / "stage3" / name
    saved = __import__("yaml").safe_load((run / "config.yaml").read_text())
    s3, sim = saved["stage3"], saved["sim"]
    sc = scenes(s3)[split]
    if "limit" in args:
        sc = sc[:: max(len(sc) // int(args["limit"]), 1)][: int(args["limit"])]
    ck = torch.load(run / ckpt_name, map_location="cpu", weights_only=False)
    actor = {k: v.cpu() for k, v in ck["actor"].items()}
    syn = dict(np.load(run / "synergy.npz"))
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(tasks, results, s3, sim, syn, tuple(ck["dims"])), daemon=True) for _ in range(int(args.get("workers", s3["learner"]["workers"])))]
    [p.start() for p in procs]
    policies = {"operator": ("recorded", True), "frozen": ("base", True), "stage3": ("base", False)}     # stream, zero correction
    n = 0
    for i, scene in enumerate(sc):
        for k in range(seeds + 1):                                        # k = 0: the scene as reconstructed
            for pol, (stream, zero) in policies.items():
                tasks.put({"scene": scene, "seed": 1000 * i + k, "perturb": k > 0, "noise": (0.0, 0.0), "zero": zero, "stream": stream,
                           "actor": None if zero else actor, "kind": pol, "index": (i, k)})
                n += 1
    rows = []
    for j in range(n):
        ep = results.get()
        if "error" in ep:
            print("error", ep["uid"], ep["error"], flush=True)
            continue
        rows.append({"policy": ep["kind"], "scene": ep["index"][0], "repeat": ep["index"][1], "uid": ep["uid"], "return": float(ep["rew"].sum()),
                     "mean_abs_correction": float(np.abs(ep["act"]).mean()), **ep["outcome"]})
        if (j + 1) % 100 == 0:
            print(f"{j + 1}/{n} episodes", flush=True)
    [tasks.put(None) for _ in procs]
    rng = np.random.default_rng(0)
    grid = lambda pol, key, rep: np.array([[float(next((r[key] for r in rows if r["policy"] == pol and r["scene"] == i and r["repeat"] == k), np.nan))  # noqa: E731
                                            for k in rep] for i in range(len(sc))])
    summary = {}
    for label, rep in (("as reconstructed", [0]), ("perturbed", list(range(1, seeds + 1)))):
        if not rep:
            continue
        summary[label] = {}
        for key in ("success", "success_strict", "return", "plate_lifted", "plate_placed", "plate_dropped", "wipe_path_cm", "wipe_path_smooth_cm", "wipe_extent_cm"):
            g = {pol: grid(pol, key, rep) for pol in policies}
            ok = ~np.isnan(np.stack(list(g.values()))).any((0, 2))
            summary[label][key] = {pol: interval(g[pol][ok], rng) for pol in policies} | {"stage3 - frozen": interval(g["stage3"][ok] - g["frozen"][ok], rng),
                                                                                           "scenes": int(ok.sum())}
    out = {"name": name, "ckpt": ckpt_name, "split": split, "scenes": len(sc), "seeds": seeds, "summary": summary, "episodes": rows}
    (run / f"eval_{split}{'' if ckpt_name == 'ckpt.pt' else '_' + Path(ckpt_name).stem}.json").write_text(json.dumps(out))
    for label, s in summary.items():
        print(f"\n{split} scenes, {label} ({s['success']['scenes']} scenes)")
        for key, v in s.items():
            print(f"  {key:20s} " + " | ".join(f"{pol} {x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]" for pol, x in v.items() if pol != "scenes"))


if __name__ == "__main__":
    main()
