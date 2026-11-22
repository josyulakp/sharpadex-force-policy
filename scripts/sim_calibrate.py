"""Calibrate the simulator on the one thing known about every scene: the operator's own commands completed the task.

    python scripts/sim_calibrate.py [--scenes 60] [--pop 24] [--gens 8]      -> <stage3.data_dir>/calibration/{log.jsonl, best.json}

Cross-entropy search over the scene and contact parameters that are estimated rather than measured (plate pose and
shape corrections, friction, mass, the stiffness of the hand-base pull and of the finger servos). Objective: the
stage-3 return of the operator's recorded command stream, averaged over a fixed set of training-split scenes. Only
the operator's stream is used; the frozen policy plays no part. Check the result on other scenes with
scripts/s3_scenes.py after copying best.json into configs/sim.yaml.
"""
import copy
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
S3 = yaml.safe_load((REPO / "configs" / "stage3.yaml").read_text())
SIM = yaml.safe_load((REPO / "configs" / "sim.yaml").read_text())
# name: (path in the simulator config, low, high)
PARAMS = {"plate_dz": (("calib", "plate_dz"), -0.012, 0.015), "plate_dx": (("calib", "plate_dx"), -0.02, 0.02),
          "plate_dy": (("calib", "plate_dy"), -0.02, 0.02), "plate_dr": (("calib", "plate_dr"), -0.015, 0.015),
          "table_dz": (("calib", "table_dz"), -0.008, 0.008), "rim_thickness": (("plate", "rim_thickness"), 0.004, 0.025),
          "foot_radius": (("plate", "foot_radius"), 0.02, 0.05), "plate_friction": (("plate", "friction"), 0.8, 3.0),
          "plate_mass": (("plate", "mass"), 0.08, 0.4), "eraser_friction": (("eraser", "friction"), 0.8, 3.0),
          "weld_time": (("weld_solref", 0), 0.012, 0.035), "hand_kp_scale": (("hand_kp_scale",), 0.6, 2.0)}


def apply(x: dict) -> dict:
    cfg = copy.deepcopy(SIM)
    cfg.setdefault("calib", {})
    for k, v in x.items():
        node, path = cfg, PARAMS[k][0]
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = float(v)
    return cfg


def episode_return(o: dict) -> float:
    rw = S3["reward"]
    return (rw["plate_lift"] * o["plate_lifted"] + rw["eraser_lift"] * o["eraser_lifted"]
            + rw["wipe"] * min(o["wipe_path_cm"] / 100 / rw["wipe_full_m"], 1.0)
            + (rw["dropped"] if o["plate_dropped"] else rw["placed"] * (o["plate_placed"] and o["plate_lifted"])))


def run(job):
    x, scene = job
    from sim.episode import Episode
    from sim.replay import Sim
    from sim.task import outcome
    z = np.load(Path(S3["data_dir"]) / "streams" / (scene["uid"].replace("/", "__") + ".npz"))
    try:
        sim = Sim(apply(x), Episode(scene["uid"]), scene["objects"])
        o = outcome(sim.run(z["poses__recorded"].astype(float), z["targets__recorded"].astype(float)), S3["outcome"])
        return episode_return(o), o["success"], o["plate_lifted"]
    except Exception:
        return 0.0, False, False


def main():
    arg = lambda n, d: int(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d  # noqa: E731
    n_scenes, pop, gens = arg("--scenes", 60), arg("--pop", 24), arg("--gens", 8)
    out = Path(S3["data_dir"]) / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((Path(S3["data_dir"]) / "scenes.json").read_text())
            if "error" not in r and r["split"] == "train" and r["frames"] <= S3["env"]["max_frames"]]
    scenes = rows[::2][:n_scenes]                                         # every second training scene; the others stay for the check
    names = list(PARAMS)
    lo, hi = np.array([PARAMS[k][1] for k in names]), np.array([PARAMS[k][2] for k in names])
    start = apply({})
    mean = np.zeros(len(names))
    for i, k in enumerate(names):                                         # start from the current configuration
        path, node = PARAMS[k][0], start
        try:
            for p in path:
                node = node[p]
            mean[i] = float(node)
        except (KeyError, TypeError):
            mean[i] = 1.0 if k == "hand_kp_scale" else 0.0
    std = (hi - lo) / 4
    rng = np.random.default_rng(0)
    log, best = (out / "log.jsonl").open("a"), (-1.0, None)
    with Pool(30) as pool:
        for g in range(gens):
            X = np.clip(mean + std * rng.standard_normal((pop, len(names))), lo, hi)
            X[0] = mean                                                   # the current mean is always evaluated
            res = pool.map(run, [(dict(zip(names, x)), s) for x in X for s in scenes], chunksize=1)
            R = np.array([r[0] for r in res]).reshape(pop, len(scenes)).mean(1)
            S = np.array([r[1] for r in res]).reshape(pop, len(scenes)).mean(1)
            P = np.array([r[2] for r in res]).reshape(pop, len(scenes)).mean(1)
            elite = np.argsort(-R)[: max(pop // 4, 3)]
            if R[elite[0]] > best[0]:
                best = (float(R[elite[0]]), dict(zip(names, map(float, X[elite[0]]))))
                (out / "best.json").write_text(json.dumps({"return": best[0], "success": float(S[elite[0]]), "generation": g, "params": best[1]}, indent=1))
            mean, std = X[elite].mean(0), np.maximum(X[elite].std(0), (hi - lo) / 40)
            row = {"generation": g, "mean_return": float(R[0]), "mean_success": float(S[0]), "best_return": float(R[elite[0]]),
                   "best_success": float(S[elite[0]]), "best_plate_lifted": float(P[elite[0]]), "population_return": float(R.mean()),
                   "mean": dict(zip(names, map(float, mean))), "std": dict(zip(names, map(float, std)))}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"gen {g}: current {R[0]:.3f} (success {S[0]:.2f}) | best {R[elite[0]]:.3f} (success {S[elite[0]]:.2f}, plate lifted {P[elite[0]]:.2f}) | "
                  f"population {R.mean():.3f} | " + " ".join(f"{k}={v:.4g}" for k, v in zip(names, mean)), flush=True)


if __name__ == "__main__":
    main()
