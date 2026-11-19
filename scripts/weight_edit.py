"""Edit the force head's weights by hand, without gradients, to reduce the force error (after "Sparks of Weight-Space Programming").

    python scripts/weight_edit.py [group=flow variant=regression_25k]   -> runs/weight_edit/<variant>/seed*/ckpt.pt + reports/results/weight_edit.md

The force head ends in one linear layer with 30 rows, one per fingertip force component; the model predicts
force now + 10 N x row output. Two edits, both a rescaling of rows of that layer (weights and bias), found with forward passes only:
  1. rows that are worse than "force stays as it is" on the TRAINING data are set to zero (those outputs then predict "force stays");
  2. every other row is multiplied by one gain between 0 and 1.25, the one with the lowest error on the PRACTICE half of the
     held-out episodes (even episode numbers). The model is over-confident on episodes it has not seen; training data cannot show that.
The result is reported on the TEST half (odd episode numbers), which no edit has seen. Nothing else in the checkpoint changes, so the
joint commands are identical. The edited checkpoint loads with the unmodified code (scripts/evaluate.py works on it; its numbers
then include the practice half).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402
from sharpa.data.dataset import FrameData  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402
from sharpa.engine import load_checkpoint  # noqa: E402

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
GAINS = torch.arange(0, 1.2501, 0.05).cuda()
TIPS = [f"{hand} {finger}" for hand in ("left", "right") for finger in ("thumb", "index", "middle", "ring", "little")]


@torch.no_grad()
def passes(model, cfg, data, stride=1):
    """Forward passes -> predicted and true force change [N, H, 30] (N), valid steps, episode, contact, joint error (rad)."""
    rows = {k: [] for k in ("pred", "true", "valid", "ep", "contact", "act")}
    idx = torch.arange(0, len(data), stride)
    for s in range(0, len(idx), 1024):
        b = data.batch(idx[s:s + 1024])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            o = model(b)
        p, av = model.to_physical(o, b), b["action_valid"].float()
        rows["pred"].append(p["force"] - o["force_base"]); rows["true"].append(b["force"] - o["force_base"])
        rows["valid"].append(b["force_valid"].float()); rows["ep"].append(b["ep_id"]); rows["contact"].append(b["contact_now"] > 0.5)
        rows["act"].append(((p["action"] - b["action"]).abs().mean(-1) * av).sum(1) / av.sum(1))
    return {k: torch.cat(v) for k, v in rows.items()}


def row_error(r, mask, gain):
    """Mean absolute error of each of the 30 outputs (N) with its predicted change multiplied by `gain` ([30] or scalar)."""
    w = (r["valid"] * mask.float()[:, None])[..., None]
    return ((gain * r["pred"] - r["true"]).abs() * w).sum((0, 1)) / w.sum((0, 1))


def frame_error(r, mask, gain):
    """The force metric of sharpa.engine.evaluate: mean over frames of the mean absolute error over valid steps and outputs (N)."""
    e = ((gain * r["pred"] - r["true"]).abs().mean(-1) * r["valid"]).sum(1) / r["valid"].sum(1).clamp(min=1)
    return {"all": e[mask].mean().item(), "contact": e[mask & r["contact"]].mean().item(), "free": e[mask & ~r["contact"]].mean().item()}


def main():
    group, variant = ARGS.get("group", "flow"), ARGS.get("variant", "regression_25k")
    results = []
    for run in sorted((REPO_ROOT / "runs" / group / variant).glob("seed*")):
        model, ck = load_checkpoint(run)
        cfg, tasks = ck["cfg"], ck["cfg"]["data"]["tasks"]
        assert not model.flow and not model.world_model and model.force_residual
        splits = build_splits(cfg)
        frames = {part: FrameData(cfg, [r for t in tasks for r in splits[t][part]], ck["vocab"], tasks) for part in ("train", "heldout")}
        train = passes(model, cfg, frames["train"], stride=6)
        held = passes(model, cfg, frames["heldout"])
        practice, test, every = held["ep"] % 2 == 0, held["ep"] % 2 == 1, torch.ones_like(train["ep"], dtype=torch.bool)

        dead = row_error(train, every, 1.0) > row_error(train, every, 0.0)                  # edit 1, from training data
        gain = GAINS[torch.stack([row_error(held, practice, g) for g in GAINS]).argmin(0)]  # edit 2, from the practice half
        gain[dead] = 0
        train_gain = GAINS[torch.stack([row_error(train, every, g) for g in GAINS]).argmin(0)]   # what training data alone would choose

        state = ck["model"]
        state["force_head.2.weight"] = state["force_head.2.weight"] * gain[:, None].to(state["force_head.2.weight"].dtype)
        state["force_head.2.bias"] = state["force_head.2.bias"] * gain.to(state["force_head.2.bias"].dtype)
        out = REPO_ROOT / "runs" / "weight_edit" / variant / run.name
        out.mkdir(parents=True, exist_ok=True)
        ck["weight_edit"] = {"source": str(run.relative_to(REPO_ROOT)), "row_gain": gain.tolist()}
        torch.save(ck, out / "ckpt.pt")
        shutil.copy(run / "config.yaml", out / "config.yaml")

        edited, _ = load_checkpoint(out)                                                    # measured again through the edited checkpoint
        after = passes(edited, cfg, frames["heldout"])
        one = torch.ones(30, device="cuda")
        results.append({
            "seed": run.name, "row_gain": gain.tolist(), "dead_rows": dead.nonzero().flatten().tolist(), "train_only_gain": train_gain.tolist(),
            "changed_weights": int((gain != 1).sum()) * (state["force_head.2.weight"].shape[1] + 1), "weights": sum(v.numel() for v in state.values()),
            "test": {"original": frame_error(held, test, one), "edited": frame_error(after, test, one),
                     "dead_rows_only": frame_error(held, test, (~dead).float()), "train_only_gain": frame_error(held, test, train_gain),
                     "one_gain": frame_error(held, test, GAINS[int(np.argmin([frame_error(held, practice, g)["all"] for g in GAINS]))]),
                     "persist": frame_error(held, test, 0 * one)},
            "practice": {"original": frame_error(held, practice, one), "edited": frame_error(after, practice, one)},
            "train": {"original": frame_error(train, every, one), "edited": frame_error(train, every, gain)},
            "action_mrad": {"original": 1000 * held["act"][test].mean().item(), "edited": 1000 * after["act"][test].mean().item()},
            "test_frames": int(test.sum()), "practice_frames": int(practice.sum())})
        print(run.name, json.dumps(results[-1]["test"]), flush=True)
        del frames

    mean = lambda f: float(np.mean([f(r) for r in results]))  # noqa: E731
    cell = lambda part, k: " | ".join(f"{mean(lambda r: r[part][k][g]):.4f}" for g in ("all", "contact", "free"))  # noqa: E731
    gains = np.array([r["row_gain"] for r in results]).mean(0).reshape(10, 3).mean(1)
    r0 = results[0]
    lines = [f"# Force head edited by hand (`{group}/{variant}`, {len(results)} seeds)", "",
             f"Each of the 30 rows of the force head's last layer is multiplied by one number; {r0['changed_weights']:,} of {r0['weights']:,} weights "
             "change, no gradients. Rows worse than \"force stays as it is\" on the training data are zeroed; the other gains are chosen on the practice "
             f"half of the held-out episodes ({r0['practice_frames']:,} frames). Force error (N) on the test half ({r0['test_frames']:,} frames), mean of seeds.", "",
             "| | all | in contact | free |", "|---|---|---|---|",
             f"| original | {cell('test', 'original')} |", f"| zeroed rows only (training data) | {cell('test', 'dead_rows_only')} |",
             f"| gains chosen on training data only | {cell('test', 'train_only_gain')} |",
             f"| one gain for all rows (practice half) | {cell('test', 'one_gain')} |", f"| **edited: one gain per row** | {cell('test', 'edited')} |",
             f"| force stays as it is | {cell('test', 'persist')} |", "",
             "Per seed (test half, all frames): " + ", ".join(f"{r['seed']} {r['test']['original']['all']:.4f} -> {r['test']['edited']['all']:.4f}" for r in results) + ".",
             f"Practice half: {mean(lambda r: r['practice']['original']['all']):.4f} -> {mean(lambda r: r['practice']['edited']['all']):.4f}. "
             f"Training frames: {mean(lambda r: r['train']['original']['all']):.4f} -> {mean(lambda r: r['train']['edited']['all']):.4f}. "
             f"Joint error on the test half: {mean(lambda r: r['action_mrad']['original']):.2f} -> {mean(lambda r: r['action_mrad']['edited']):.2f} mrad.", "",
             "Gain per fingertip (mean of x, y, z and seeds):", "", "| " + " | ".join(TIPS) + " |", "|" + "---|" * 10,
             "| " + " | ".join(f"{g:.2f}" for g in gains) + " |"]
    (REPO_ROOT / "reports" / "results" / "weight_edit.md").write_text("\n".join(lines) + "\n")
    (REPO_ROOT / "reports" / "results" / "weight_edit.json").write_text(json.dumps(results, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
