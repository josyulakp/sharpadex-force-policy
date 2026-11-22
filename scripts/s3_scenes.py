"""Stage 3, step 2: build the simulated scene of every episode and replay two command streams in it.

    python scripts/s3_scenes.py [--jobs 30]      -> <stage3.data_dir>/scenes/<episode>.json, <stage3.data_dir>/scenes.json

Streams: the operator's recorded commands and the frozen four-expert policy's commands (scripts/s3_prepare.py). The
operator's stream is the check on the scene itself: where the operator's own commands do not complete the task in the
simulator, the reconstructed scene (object placement, plate shape, contact model) is not faithful enough to learn in.
"""
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[1]
S3 = yaml.safe_load((REPO / "configs" / "stage3.yaml").read_text())
SIM = yaml.safe_load((REPO / "configs" / "sim.yaml").read_text())
OUT = Path(S3["data_dir"]) / "scenes"


def survey(path: str) -> dict:
    from sim.episode import Episode
    from sim.replay import Sim
    from sim.task import outcome
    z = np.load(path, allow_pickle=False)
    uid = str(z["uid"])
    dst = OUT / (uid.replace("/", "__") + ".json")
    if dst.exists():
        return json.loads(dst.read_text())
    res = {"uid": uid, "split": str(z["split"]), "frames": int(len(z["targets__base"])), "base_err_mrad": float(z["err_mrad"])}
    try:
        ep = Episode(uid)
        sim = Sim(SIM, ep)
        res["objects"] = {k: (np.round(v, 4).tolist() if hasattr(v, "__len__") else v) for k, v in sim.objects.items()}
        for n in ("recorded", "base"):
            log = sim.run(z[f"poses__{n}"].astype(float), z[f"targets__{n}"].astype(float))
            res[n] = outcome(log, S3["outcome"]) | {"hand_base_tracking_mm": float(log["base_err_mm"].mean()),
                                                    "peak_tip_force_n": float(np.linalg.norm(log["tip_force"], axis=2).max())}
    except Exception as e:                                                # scene could not be built (e.g. no plate found in the first frame)
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-600:]
    dst.write_text(json.dumps(res, indent=1))
    return res


def main():
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 30
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in (Path(S3["data_dir"]) / "streams").glob("*.npz"))
    rows = []
    with Pool(jobs) as pool:
        for k, r in enumerate(pool.imap_unordered(survey, files), 1):
            rows.append(r)
            if k % 20 == 0 or k == len(files):
                ok = [x for x in rows if "error" not in x]
                print(f"{k}/{len(files)} scenes; built {len(ok)}; operator succeeds {sum(x['recorded']['success'] for x in ok)}; "
                      f"policy succeeds {sum(x['base']['success'] for x in ok)}", flush=True)
    rows.sort(key=lambda x: x["uid"])
    (Path(S3["data_dir"]) / "scenes.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
