"""Stage 3: side-by-side videos of the frozen policy and the frozen policy with the stage-3 corrections, same scene.

    python scripts/s3_video.py [name=rl] [ckpt=ckpt.pt] [split=heldout] [pick=6] [uids=a,b] [seed=0] [jobs=6]
        -> reports/stage3/videos/<episode>.mp4 and index.json (outcome of both runs per video)

Scenes: `uids`, or `pick` scenes of the split chosen from runs/stage3/<name>/eval_<split>.json so that every kind of
result is shown (corrections fix a failure, both succeed, both fail, corrections break a success). seed 0 = the scene as
reconstructed; any other seed = that perturbation of it.
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
from scripts.s3_train import make_actor, run_episode, scenes  # noqa: E402

OUT = REPO / "reports" / "stage3" / "videos"


def render_pair(job: dict) -> dict:
    import av
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from sim.env import ResidualEnv
    torch.set_num_threads(1)
    run = REPO / "runs" / "stage3" / job["name"]
    saved = yaml.safe_load((run / "config.yaml").read_text())
    s3, sim_cfg = saved["stage3"], saved["sim"]
    ck = torch.load(run / job["ckpt"], map_location="cpu", weights_only=False)
    env = ResidualEnv(sim_cfg, s3, dict(np.load(run / "synergy.npz")))
    actor = make_actor(s3, tuple(ck["dims"]))
    actor.load_state_dict(ck["actor"])
    W, H, stride = 480, 360, 2
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    clips, outcomes = {}, {}
    for label, zero in (("frozen four-expert policy", True), ("with stage-3 corrections", False)):
        frames, state = [], {}
        reset = env.reset

        def reset_and_hook(*a, **k):
            obs = reset(*a, **k)
            sim = env.sim
            state["r"] = mujoco.Renderer(sim.m, H, W)
            step, n = sim.step, [0]

            def step_and_render(pose, hand):
                out = step(pose, hand)
                if n[0] % stride == 0:
                    state["r"].update_scene(sim.d, camera="view")
                    frames.append(state["r"].render().copy())
                n[0] += 1
                return out
            sim.step = step_and_render
            return obs
        env.reset = reset_and_hook
        ep = run_episode(env, actor, {"scene": job["scene"], "seed": job["seed"], "perturb": job["seed"] > 0, "noise": (0.0, 0.0), "zero": zero,
                                      "stream": "base", "kind": label, "index": 0})
        env.reset = reset
        state["r"].close()
        o = ep["outcome"]
        outcomes[label] = o
        verdict = "success" if o["success"] else "plate dropped" if o["plate_dropped"] else "plate not lifted" if not o["plate_lifted"] else \
            "wiped too little" if o["wipe_path_cm"] < 100 * s3["outcome"]["wipe_path_m"] else "plate not put back"
        cap = Image.new("RGB", (W, 44), (22, 26, 31))
        dr = ImageDraw.Draw(cap)
        dr.text((10, 4), label, fill=(240, 240, 240), font=font)
        dr.text((10, 23), f"{verdict} · wiped {o['wipe_path_smooth_cm']:.0f} cm", fill=(120, 200, 140) if o["success"] else (235, 150, 110), font=font)
        clips[label] = [np.concatenate([np.asarray(cap), f]) for f in frames]
    a, b = clips.values()
    n = max(len(a), len(b))
    a, b = (c + [c[-1]] * (n - len(c)) for c in (a, b))                  # an episode that ended early (dropped plate) holds its last frame
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{job['scene']['uid'].replace('/', '__')}_seed{job['seed']}.mp4"
    out = av.open(str(f), "w")
    st = out.add_stream("libx264", rate=30 // stride)
    st.width, st.height, st.pix_fmt = 2 * W + 4, H + 44, "yuv420p"
    st.options = {"crf": "26", "preset": "slow", "movflags": "+faststart"}
    for x, y in zip(a, b):
        img = np.ascontiguousarray(np.concatenate([x, np.zeros((H + 44, 4, 3), np.uint8), y], 1))
        for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    return {"uid": job["scene"]["uid"], "seed": job["seed"], "video": f.name, "outcome": outcomes}


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    name, ckpt, split, seed = args.get("name", "rl"), args.get("ckpt", "ckpt.pt"), args.get("split", "heldout"), int(args.get("seed", 0))
    run = REPO / "runs" / "stage3" / name
    sc = scenes(yaml.safe_load((run / "config.yaml").read_text())["stage3"])[split]
    if "uids" in args:
        chosen = [s for s in sc if s["uid"] in args["uids"].split(",")]
    else:
        ev = json.loads((run / f"eval_{split}{'' if ckpt == 'ckpt.pt' else '_' + Path(ckpt).stem}.json").read_text())
        ok = {(r["policy"], r["uid"]): r["success"] for r in ev["episodes"] if r["repeat"] == seed}
        kinds = {}
        for s in sc:
            kinds.setdefault((ok.get(("frozen", s["uid"])), ok.get(("stage3", s["uid"]))), []).append(s)
        order, chosen = [(False, True), (True, True), (False, False), (True, False)], []
        while len(chosen) < int(args.get("pick", 6)) and any(kinds.get(k) for k in order):
            for k in order:
                if kinds.get(k) and len(chosen) < int(args.get("pick", 6)):
                    chosen.append(kinds[k].pop(0))
    jobs = [{"name": name, "ckpt": ckpt, "scene": s, "seed": seed} for s in chosen]
    import multiprocessing as mp
    with ProcessPoolExecutor(int(args.get("jobs", 6)), mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(render_pair, jobs))
    for r in rows:
        print(r["video"], {k: (v["success"], round(v["wipe_path_smooth_cm"], 1)) for k, v in r["outcome"].items()}, flush=True)
    (OUT / "index.json").write_text(json.dumps({"name": name, "ckpt": ckpt, "split": split, "videos": rows}, indent=1))


if __name__ == "__main__":
    main()
