"""Simulator results page: reports/sim/page_template.html + simulator logs -> reports/sim/page/index.html (+ v/*.mp4)

    python scripts/rollout.py && python scripts/sim_streams.py        # real videos and command streams
    MUJOCO_GL=egl <sim python> sim/run.py reports/sim/streams/<episode>.npz --video     # once per episode
    python scripts/build_sim_page.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402

STREAMS = ("recorded", "full", "no_force")
sim, cfg = REPO_ROOT / "reports" / "sim", load_config()
simcfg = yaml.safe_load((REPO_ROOT / "configs" / "sim.yaml").read_text())
roll_dir = REPO_ROOT / cfg["rollout"]["out_dir"]
rollouts = json.loads((roll_dir / "rollouts.json").read_text())
out = sim / "page"
(out / "v").mkdir(parents=True, exist_ok=True)
r2 = lambda a: np.round(np.asarray(a, float), 2).tolist()  # noqa: E731

episodes = []
for k, e in enumerate(rollouts["episodes"]):
    d = sim / e["uid"].replace("/", "__")
    z = np.load(sim / "streams" / f"{d.name}.npz")
    summary = json.loads((d / "summary.json").read_text())
    shutil.copy(roll_dir / e["video"], out / "v" / f"{k}_real.mp4")
    ep = {"uid": e["uid"], "label": e["label"], "frames": e["frames"], "fps": e["fps"], "contact": e["contact"],
          "real": f"v/{k}_real.mp4", "objects": summary["objects"], "score": {}, "video": {}, "plate": {}, "eraser": {}, "force": {},
          "force_real": r2(z["force_meas"].max(1)), "force_forecast": r2(z["force_pred__full"].max(1))}
    for s in STREAMS:
        log = np.load(d / f"{s}.npz")
        shutil.copy(d / f"{s}.mp4", out / "v" / f"{k}_{s}.mp4")
        ep["video"][s] = f"v/{k}_{s}.mp4"
        ep["score"][s] = summary[s]
        ep["plate"][s] = r2((log["plate"][:, 2] - log["plate"][0, 2]) * 100)
        ep["eraser"][s] = r2((log["eraser"][:, 2] - log["eraser"][0, 2]) * 100)
        ep["force"][s] = r2(np.linalg.norm(log["tip_force"], axis=2).max(1))
    episodes.append(ep)

data = {"episodes": episodes, "lift_cm": simcfg["score"]["lift_m"] * 100, "execute_steps": rollouts["execute_steps"],
        "seed": rollouts["seed"], "contact_threshold_n": rollouts["contact_threshold_n"]}
page = (sim / "page_template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
(out / "index.html").write_text(page)
print(out / "index.html", f"{len(page) / 1e6:.2f} MB;", f"{sum(p.stat().st_size for p in (out / 'v').iterdir()) / 1e6:.1f} MB video")
