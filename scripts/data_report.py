"""Task step 1: what is in the data, and the fixed held-out split.

    python scripts/data_report.py [--config configs/default.yaml] [key=value ...]

Writes   reports/data/summary.json, reports/data/data_report.md, reports/data/*.png
         splits/split_seed{S}.json  (held-out + training uids, and nested training-share subsets)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, parse_cli  # noqa: E402
from sharpa.data.episodes import (FINGERTIP_NAMES, L_ARM, L_HAND, R_ARM, R_HAND,  # noqa: E402
                                  fingertip_force_magnitude, load_numeric, to_jsonable)
from sharpa.data.split import build_splits, train_subset  # noqa: E402

SHARES = [0.10, 0.25, 0.50, 1.00]
MAX_LAG = 15


def subtask_table(ref) -> dict[int, str]:
    """subtask_index -> skill label; the season's catch-all index (global task text) -> 'none'."""
    meta = Path(ref.path).parents[2] / "meta" / "subtasks.jsonl"
    table = {}
    for line in meta.read_text().splitlines():
        s = json.loads(line)
        text = s["subtask"]
        table[s["subtask_index"]] = "none" if text.startswith("Task:") else text.rsplit("Skill:", 1)[-1].strip()
    return table


def episode_stats(args):
    """Everything the report needs from one episode, computed in a worker."""
    ref, thresholds, thr = args
    e = load_numeric(ref)
    mag = fingertip_force_magnitude(e["observation.tactile"])           # (T,10)
    peak = mag.max(1)
    contact = peak > thr
    onsets = int(np.sum(np.diff(contact.astype(np.int8)) == 1) + contact[0])
    dt = np.diff(e["timestamp"].astype(np.float64))
    s, a = e["observation.state"], e["action"]

    def lag_curve(sl):
        return [float(np.mean((a[: len(a) - k, sl] - s[k:, sl]) ** 2)) for k in range(MAX_LAG + 1)]

    skills = subtask_table(ref)
    sk = np.array([skills.get(int(i), "none") for i in e["subtask_index"]])
    per_skill = {k: (int((sk == k).sum()), int((contact & (sk == k)).sum())) for k in np.unique(sk)}
    return {
        "uid": ref.uid, "task": ref.task, "season": ref.season, "T": len(peak),
        "dt_max_dev_ms": float(np.abs(dt - 1 / 30).max() * 1000) if len(dt) else 0.0,
        "frame_index_ok": bool(e["frame_index"][0] == 0 and np.all(np.diff(e["frame_index"]) == 1)),
        "nonfinite": int(sum((~np.isfinite(e[c])).sum() for c in
                             ["observation.state", "action", "observation.state.joint_torque", "observation.tactile"])),
        "rest_force": float(np.median(mag[:15])),                          # first 0.5 s, hands open
        "sweep": [float(np.mean(peak > t)) for t in thresholds],
        "contact": int(contact.sum()),
        "contact_left": int((mag[:, :5].max(1) > thr).sum()),
        "contact_right": int((mag[:, 5:].max(1) > thr).sum()),
        "finger_contact": (mag > thr).sum(0).tolist(),
        "onsets": onsets,
        "peak_in_contact": peak[contact].tolist() if contact.any() else [],
        "lag_arm": lag_curve(np.r_[L_ARM, R_ARM]), "lag_hand": lag_curve(np.r_[L_HAND, R_HAND]),
        "per_skill": per_skill,
        "state_std": s.std(0).tolist(), "action_std": a.std(0).tolist(),
        "torque_std": e["observation.state.joint_torque"].std(0).tolist(),
    }


def example_episode_plot(ref, thr, path):
    e = load_numeric(ref)
    mag = fingertip_force_magnitude(e["observation.tactile"])
    skills = subtask_table(ref)
    sk = [skills.get(int(i), "none") for i in e["subtask_index"]]
    t = np.arange(len(mag)) / 30
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    for ax, (hand, sl) in zip(axes, [("left", slice(0, 5)), ("right", slice(5, 10))]):
        for j, name in enumerate(FINGERTIP_NAMES[sl]):
            ax.plot(t, mag[:, sl][:, j], lw=1, label=name.split("_")[1])
        ax.axhline(thr, color="k", ls="--", lw=0.8)
        ax.set_ylabel(f"{hand} |F| (N)")
        # shade subtask phases
        start = 0
        for i in range(1, len(sk) + 1):
            if i == len(sk) or sk[i] != sk[start]:
                if sk[start] != "none":
                    ax.axvspan(t[start], t[i - 1], alpha=0.08, color="C%d" % (hash(sk[start]) % 10))
                    if hand == "left":
                        ax.text((t[start] + t[i - 1]) / 2, ax.get_ylim()[1] * 0.92, sk[start], ha="center", fontsize=7)
                start = i
    axes[0].legend(fontsize=7, ncol=5, loc="upper right")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"{ref.uid}  (dashed = {thr} N contact threshold)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    cfg = parse_cli(sys.argv[1:])
    d, r = cfg["data"], cfg["report"]
    out = REPO_ROOT / r["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    thr, sweep = d["contact_threshold_n"], r["thresholds_sweep_n"]

    splits = build_splits(cfg)
    all_refs = [ref for t in d["tasks"] for part in ("train_full", "heldout") for ref in splits[t][part]]
    with ProcessPoolExecutor(r["num_workers"]) as ex:
        stats = {s["uid"]: s for s in ex.map(episode_stats, [(ref, sweep, thr) for ref in all_refs], chunksize=4)}

    fps = d["fps"]
    summary = {"config": {k: d[k] for k in ("tasks", "holdout_fraction", "split_seed", "subset_seed",
                                            "contact_threshold_n")}, "tasks": {}}
    split_file = {"split_seed": d["split_seed"], "subset_seed": d["subset_seed"],
                  "holdout_fraction": d["holdout_fraction"], "tasks": {}}
    for task in d["tasks"]:
        tr, ho = splits[task]["train_full"], splits[task]["heldout"]
        S = [stats[x.uid] for x in tr + ho]
        T = np.array([s["T"] for s in S])
        frames = int(T.sum())
        lag_arm = np.mean([s["lag_arm"] for s in S], 0)
        lag_hand = np.mean([s["lag_hand"] for s in S], 0)
        peaks = np.concatenate([s["peak_in_contact"] for s in S])
        skill_tot = {}
        for s in S:
            for k, (n, c) in s["per_skill"].items():
                a, b = skill_tot.get(k, (0, 0))
                skill_tot[k] = (a + n, b + c)
        state_std = np.mean([s["state_std"] for s in S], 0)
        torque_std = np.mean([s["torque_std"] for s in S], 0)
        shares = {}
        for p in SHARES:
            sub = train_subset(tr, task, p, d["subset_seed"])
            shares[f"{p:.2f}"] = {"episodes": len(sub), "hours": sum(x.num_frames for x in sub) / fps / 3600,
                                  "uids": [x.uid for x in sub]}
        ho_contact = sum(stats[x.uid]["contact"] for x in ho)
        ho_frames = sum(stats[x.uid]["T"] for x in ho)
        summary["tasks"][task] = {
            "seasons": sorted({x.season for x in tr + ho}),
            "episodes": len(S), "frames": frames, "hours": frames / fps / 3600,
            "episode_seconds": {"mean": T.mean() / fps, "median": float(np.median(T)) / fps,
                                "min": T.min() / fps, "max": T.max() / fps},
            "fps_max_timestamp_deviation_ms": max(s["dt_max_dev_ms"] for s in S),
            "episodes_bad_frame_index": sum(not s["frame_index_ok"] for s in S),
            "nonfinite_values": sum(s["nonfinite"] for s in S),
            "rest_force_n_median": float(np.median([s["rest_force"] for s in S])),
            "contact_fraction": sum(s["contact"] for s in S) / frames,
            "contact_fraction_left": sum(s["contact_left"] for s in S) / frames,
            "contact_fraction_right": sum(s["contact_right"] for s in S) / frames,
            "finger_contact_fraction": dict(zip(FINGERTIP_NAMES, (np.sum([s["finger_contact"] for s in S], 0) / frames).tolist())),
            "contact_sweep": dict(zip(map(str, sweep), (np.sum([np.array(s["sweep"]) * s["T"] for s in S], 0) / frames).tolist())),
            "contact_onsets_per_episode": float(np.mean([s["onsets"] for s in S])),
            "peak_force_in_contact_n": {"median": float(np.median(peaks)), "p95": float(np.percentile(peaks, 95))},
            "contact_fraction_by_skill": {k: {"frames": n, "contact": c / n} for k, (n, c) in sorted(skill_tot.items())},
            "action_to_state_lag_frames": {"arm": int(np.argmin(lag_arm)), "hand": int(np.argmin(lag_hand))},
            "torso_state_std_rad": state_std[58:].tolist(),
            "dead_torque_dims": [int(i) for i in np.where(torque_std < 1e-8)[0]],
            "dead_state_dims": [int(i) for i in np.where(state_std < 1e-8)[0]],
            "split": {"train_episodes": len(tr), "heldout_episodes": len(ho),
                      "train_hours": sum(x.num_frames for x in tr) / fps / 3600,
                      "heldout_hours": sum(x.num_frames for x in ho) / fps / 3600,
                      "heldout_contact_frames": ho_contact, "heldout_free_frames": ho_frames - ho_contact,
                      "heldout_by_season": {s: sum(x.season == s for x in ho) for s in sorted({x.season for x in tr + ho})},
                      "train_share_sizes": {k: {"episodes": v["episodes"], "hours": v["hours"]} for k, v in shares.items()}},
            "_lag_curves": {"arm": lag_arm.tolist(), "hand": lag_hand.tolist()},
        }
        split_file["tasks"][task] = {"heldout": to_jsonable(ho), "train": to_jsonable(tr),
                                     "train_shares": {k: v["uids"] for k, v in shares.items()}}

    split_path = REPO_ROOT / r["split_dir"] / f"split_seed{d['split_seed']}.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_file, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    make_plots(summary, d, stats, splits, out, thr, sweep)
    write_markdown(summary, out / "data_report.md", thr, split_path)
    print((out / "data_report.md").read_text())


def make_plots(summary, d, stats, splits, out, thr, sweep):
    tasks = d["tasks"]
    # 1. episode lengths
    fig, ax = plt.subplots(figsize=(6, 3))
    for t in tasks:
        T = [stats[x.uid]["T"] / d["fps"] for x in splits[t]["train_full"] + splits[t]["heldout"]]
        ax.hist(T, bins=40, alpha=0.55, label=t)
    ax.set_xlabel("episode length (s)"); ax.set_ylabel("episodes"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "episode_lengths.png", dpi=130); plt.close(fig)
    # 2. contact fraction vs threshold
    fig, ax = plt.subplots(figsize=(5, 3))
    for t in tasks:
        cs = summary["tasks"][t]["contact_sweep"]
        ax.plot([float(k) for k in cs], list(cs.values()), "o-", label=t)
    ax.axvline(thr, color="k", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("contact threshold (N, any fingertip)"); ax.set_ylabel("fraction of frames in contact")
    ax.set_ylim(0, 1); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "contact_vs_threshold.png", dpi=130); plt.close(fig)
    # 3. per-finger contact
    M = np.array([list(summary["tasks"][t]["finger_contact_fraction"].values()) for t in tasks])
    fig, ax = plt.subplots(figsize=(7, 2.2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(10), [n.replace("_", "\n") for n in FINGERTIP_NAMES], fontsize=7)
    ax.set_yticks(range(len(tasks)), tasks, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(10):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[i, j] < 0.5 else "k")
    fig.colorbar(im, ax=ax, label="frac. frames > thr")
    fig.tight_layout(); fig.savefig(out / "finger_contact.png", dpi=130); plt.close(fig)
    # 4. example episode per task (median length)
    for t in tasks:
        refs = splits[t]["train_full"]
        med = sorted(refs, key=lambda x: x.num_frames)[len(refs) // 2]
        example_episode_plot(med, thr, out / f"example_{t}.png")
    # 5. lag
    fig, ax = plt.subplots(figsize=(5, 3))
    for t in tasks:
        c = summary["tasks"][t]["_lag_curves"]
        ax.plot(np.array(c["arm"]) / c["arm"][0], label=f"{t} arm")
        ax.plot(np.array(c["hand"]) / c["hand"][0], ls="--", label=f"{t} hand")
    ax.set_xlabel("k (frames)"); ax.set_ylabel("MSE(action_t, state_t+k) / MSE at k=0"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "action_state_lag.png", dpi=130); plt.close(fig)


def write_markdown(summary, path, thr, split_path):
    S = summary["tasks"]
    tasks = list(S)
    L = ["# SharpaDex data report (step 1)", "",
         f"Contact = any fingertip |F| > {thr} N. Held-out = {summary['config']['holdout_fraction']:.0%} of episodes per task, "
         f"split_seed={summary['config']['split_seed']}. Split file: `{split_path.relative_to(REPO_ROOT)}`.", "",
         "| | " + " | ".join(tasks) + " | total |", "|---" * (len(tasks) + 2) + "|"]

    def row(name, f, total=None):
        L.append(f"| {name} | " + " | ".join(f(S[t]) for t in tasks) + f" | {total if total is not None else ''} |")

    row("seasons", lambda s: str(len(s["seasons"])), sum(len(S[t]["seasons"]) for t in tasks))
    row("episodes", lambda s: str(s["episodes"]), sum(S[t]["episodes"] for t in tasks))
    row("hours", lambda s: f"{s['hours']:.2f}", f"{sum(S[t]['hours'] for t in tasks):.2f}")
    row("episode length mean / min / max (s)", lambda s: "{mean:.1f} / {min:.1f} / {max:.1f}".format(**s["episode_seconds"]))
    row("rate", lambda s: f"30 Hz (max dev {s['fps_max_timestamp_deviation_ms']:.3f} ms)")
    row("non-finite values / bad frame index", lambda s: f"{s['nonfinite_values']} / {s['episodes_bad_frame_index']}")
    row("fingertip |F| at rest, median (N)", lambda s: f"{s['rest_force_n_median']:.3f}")
    row(f"frames in contact (> {thr} N)", lambda s: f"{s['contact_fraction']:.1%}")
    row("  left hand / right hand", lambda s: f"{s['contact_fraction_left']:.1%} / {s['contact_fraction_right']:.1%}")
    row("contact onsets per episode", lambda s: f"{s['contact_onsets_per_episode']:.1f}")
    row("peak fingertip |F| in contact, median / p95 (N)",
        lambda s: f"{s['peak_force_in_contact_n']['median']:.1f} / {s['peak_force_in_contact_n']['p95']:.1f}")
    row("action leads state by (frames) arm / hand",
        lambda s: f"{s['action_to_state_lag_frames']['arm']} / {s['action_to_state_lag_frames']['hand']}")
    row("train / held-out episodes", lambda s: f"{s['split']['train_episodes']} / {s['split']['heldout_episodes']}",
        f"{sum(S[t]['split']['train_episodes'] for t in tasks)} / {sum(S[t]['split']['heldout_episodes'] for t in tasks)}")
    row("train / held-out hours", lambda s: f"{s['split']['train_hours']:.2f} / {s['split']['heldout_hours']:.2f}")
    row("held-out frames contact / free", lambda s: f"{s['split']['heldout_contact_frames']} / {s['split']['heldout_free_frames']}")
    for k in ["0.10", "0.25", "0.50", "1.00"]:
        row(f"train share {float(k):.0%}: episodes (h)",
            lambda s, k=k: f"{s['split']['train_share_sizes'][k]['episodes']} ({s['split']['train_share_sizes'][k]['hours']:.2f})")
    L += ["", "## Contact by subtask phase", ""]
    for t in tasks:
        L.append(f"- **{t}**: " + ", ".join(f"{k} {v['contact']:.0%} ({v['frames']/30/60:.0f} min)"
                                           for k, v in S[t]["contact_fraction_by_skill"].items()))
    L += ["", "## Contact fraction vs threshold", "",
          "| threshold (N) | " + " | ".join(tasks) + " |", "|---" * (len(tasks) + 1) + "|"]
    for k in S[tasks[0]]["contact_sweep"]:
        L.append(f"| {k} | " + " | ".join(f"{S[t]['contact_sweep'][k]:.1%}" for t in tasks) + " |")
    L += ["", "## Held-out episodes by season", ""]
    for t in tasks:
        L.append(f"- **{t}**: " + ", ".join(f"{s[-26:-6]}: {n}" for s, n in S[t]["split"]["heldout_by_season"].items()))
    L += ["", "## Channels with no signal", ""]
    for t in tasks:
        L.append(f"- **{t}**: dead torque dims {S[t]['dead_torque_dims']}, dead state dims {S[t]['dead_state_dims']}, "
                 f"torso state std (rad) {np.round(S[t]['torso_state_std_rad'], 3).tolist()}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
