"""Stage 2, step 1: judge columns from the dataset's subtask annotation and Facet-0's interaction reward.

    python scripts/rl_reward.py      -> <rl.out_dir>/reward.npz (per frame: subgoal, complete, elapsed, reward, return, wrench change)
                                        <rl.out_dir>/reward.json (time budgets, envelopes, summary)

Time budgets and wrench envelopes are quantiles over the training episodes only; held-out episodes get the same ones.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa import rl  # noqa: E402
from sharpa.config import load_config  # noqa: E402
from sharpa.data.episodes import load_numeric  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(overrides=sys.argv[1:])
    R, C = cfg["rl"]["reward"], cfg["rl"]["credit"]
    splits = build_splits(cfg)
    refs = [(r, part) for t in cfg["data"]["tasks"] for part in ("train", "heldout") for r in splits[t][part]]

    def load(item):
        ref, part = item
        e = load_numeric(ref)
        sub, complete, S = rl.judge_columns(ref, e["subtask_index"])
        if R["judge"] != "annotation":                                    # subgoal progress from a judge's 1 Hz answers
            z = np.load(Path(cfg["rl"]["judge_dir"]) / R["judge"] / ref.task / ref.season / f"{ref.episode_index:06d}.npz")
            sub = np.minimum(z["done"], S)[np.clip(np.searchsorted(z["t"], np.arange(len(sub)), "right") - 1, 0, None)]
            complete = np.r_[np.diff(sub), 0]                             # steps the judge counts as finished at this frame
        w = e["observation.tactile"].reshape(-1, 10, 6)
        return {"ref": ref, "part": part, "sub": sub, "complete": complete, "S": S, "elapsed": rl.elapsed_in_subgoal(sub),
                "wrench": w, "change": rl.wrench_change(e["observation.tactile"], C["filter_window"], C["lag"])}

    with ThreadPoolExecutor(16) as ex:
        eps = list(ex.map(load, refs))
    budgets, envelopes = {}, {}
    for task in cfg["data"]["tasks"]:
        tr = [e for e in eps if e["ref"].task == task and e["part"] == "train"]
        S = tr[0]["S"]
        used = [np.concatenate([e["elapsed"][(e["complete"] > 0) & (e["sub"] == k)] for e in tr]) for k in range(S)]
        fallback = float(np.median([len(e["sub"]) for e in tr]))         # a subgoal the judge never saw finished
        budgets[task] = [float(np.quantile(u, R["time_budget_quantile"])) if len(u) else fallback for u in used] + [1.0]
        every = np.concatenate([np.abs(e["wrench"]).max(1) for e in tr])
        per = [np.concatenate([np.abs(e["wrench"][e["sub"] == k]).max(1) for e in tr]) for k in range(S + 1)]
        envelopes[task] = [np.maximum(np.quantile(w if len(w) else every, R["envelope_quantile"], axis=0), 1e-3).tolist() for w in per]
    flat = {k: [] for k in ("subgoal", "complete", "elapsed", "reward", "ret", "change")}
    for e in eps:
        task = e["ref"].task
        r = rl.interaction_reward(e["complete"], e["elapsed"], e["wrench"], np.array(envelopes[task])[e["sub"]],
                                  np.array(budgets[task])[e["sub"]], R["alpha"], R["beta"])
        for k, v in (("subgoal", e["sub"]), ("complete", e["complete"]), ("elapsed", e["elapsed"]), ("reward", r),
                     ("ret", rl.discounted_returns(r, R["gamma"])), ("change", e["change"].astype(np.float32))):
            flat[k].append(v)
    out = Path(cfg["rl"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reward.npz", uid=np.array([e["ref"].uid for e in eps]), length=np.array([len(e["sub"]) for e in eps]),
                        split=np.array([e["part"] for e in eps]), **{k: np.concatenate(v) for k, v in flat.items()})
    total = np.array([r.sum() for r in flat["reward"]])
    pen = np.array([r[c == 0].sum() for r, c in zip(flat["reward"], flat["complete"])])
    done = {t: float(np.mean([e["complete"].sum() / e["S"] for e in eps if e["ref"].task == t])) for t in cfg["data"]["tasks"]}
    summary = {"judge": R["judge"], "share_of_subgoals_completed": done, "episodes": len(eps), "frames": int(sum(map(len, flat["reward"]))), "time_budget_frames": budgets, "envelopes": envelopes,
               "episode_reward": {"mean": float(total.mean()), "p05": float(np.quantile(total, 0.05)), "p95": float(np.quantile(total, 0.95))},
               "episode_wrench_penalty": {"mean": float(pen.mean()), "p05": float(np.quantile(pen, 0.05))},
               "frames_over_envelope": float(np.mean(np.concatenate([r[c == 0] < 0 for r, c in zip(flat["reward"], flat["complete"])])))}
    (out / "reward.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "envelopes"}, indent=1))
