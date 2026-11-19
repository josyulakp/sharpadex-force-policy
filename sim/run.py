"""Run command streams of one episode in the simulator and score them.

    python sim/run.py <streams npz> [stream names ...] [--video]     (simulation environment, MUJOCO_GL=egl)
        -> reports/sim/<episode>/<stream>.npz (+ .mp4), summary printed as JSON
"""
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.episode import Episode  # noqa: E402
from sim.replay import Sim  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def score(log: dict, sim: Sim, ep: Episode, cfg: dict) -> dict:
    """Object-level outcome and fingertip forces of one simulated stream."""
    c = cfg["score"]
    z0 = {k: log[k][0, 2] for k in ("plate", "eraser")}
    lift = {k: log[k][:, 2] - z0[k] for k in z0}
    force = np.linalg.norm(log["tip_force"], axis=2)                      # [T, 10] N
    wipe = log["wipe_force"] > c["wipe_force_n"]                          # eraser pressed on the plate
    rel = log["eraser"][:, :3] - log["plate"][:, :3]
    path = float(np.linalg.norm(np.diff(rel, axis=0), axis=1)[wipe[1:]].sum())
    return {"plate_lifted": bool(lift["plate"].max() > c["lift_m"]), "eraser_lifted": bool(lift["eraser"].max() > c["lift_m"]),
            "plate_max_lift_cm": float(lift["plate"].max() * 100), "eraser_max_lift_cm": float(lift["eraser"].max() * 100),
            "wipe_s": float(wipe.sum() / 30), "wipe_path_cm": path * 100,
            "wipe_mean_force_n": float(log["wipe_force"][wipe].mean()) if wipe.any() else 0.0,
            "contact_share_sim": float((force.max(1) > 1).mean()), "contact_share_real": float((ep.force.max(1) > 1).mean()),
            "peak_force_sim_n": float(force.max()), "peak_force_real_n": float(ep.force.max()),
            "mean_contact_force_sim_n": float(force.max(1)[force.max(1) > 1].mean()) if (force.max(1) > 1).any() else 0.0,
            "mean_contact_force_real_n": float(ep.force.max(1)[ep.force.max(1) > 1].mean()),
            "hand_base_tracking_mm": float(log["base_err_mm"].mean()),
            "final_plate_drop_cm": float((log["plate"][-1, 2] - z0["plate"]) * 100)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    z = np.load(args[0], allow_pickle=False)
    names = args[1:] or [k.split("__", 1)[1] for k in z.files if k.startswith("targets__")]
    cfg = yaml.safe_load((REPO / "configs" / "sim.yaml").read_text())
    ep = Episode(str(z["uid"]))
    sim = Sim(cfg, ep)
    out = REPO / "reports" / "sim" / str(z["uid"]).replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    render = mujoco.Renderer(sim.m, cfg["video"]["height"], cfg["video"]["width"]) if "--video" in sys.argv else None
    res = {"objects": {k: (np.round(v, 4).tolist() if hasattr(v, "__len__") else v) for k, v in sim.objects.items()}}
    for n in names:
        log = sim.run(z[f"poses__{n}"], z[f"targets__{n}"], render, cfg["video"]["stride"])
        frames = log.pop("frames")
        if frames:
            iio.imwrite(out / f"{n}.mp4", np.stack(frames), fps=30 // cfg["video"]["stride"], codec="libx264")
        np.savez_compressed(out / f"{n}.npz", **log)
        res[n] = score(log, sim, ep, cfg)
        print(n, json.dumps(res[n]), flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
