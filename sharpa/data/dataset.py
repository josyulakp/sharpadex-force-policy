"""In-memory frame dataset: numeric streams + cached vision tokens, indexed by global frame.

Everything for a set of episodes is concatenated into flat tensors; a sample is a frame index g.
History and future windows are built by index arithmetic and clamped to the episode bounds.
All values are returned raw (radians, newtons); normalisation lives in the model.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .episodes import EpisodeRef, load_numeric
from .features import feature_path
from .. import rl


def joint_dims(cfg: dict) -> int:
    return 65 if cfg["data"]["use_torso"] else 58


def skill_vocab(cfg: dict) -> list[str]:
    """Sorted skill labels over the configured tasks ('none' = frame outside any subtask segment)."""
    skills = {"none"}
    for task in cfg["data"]["tasks"]:
        for f in (Path(cfg["data"]["root"]) / task).glob(f"*/{cfg['data']['format']}/meta/subtasks.jsonl"):
            for line in f.read_text().splitlines():
                text = json.loads(line)["subtask"]
                if not text.startswith("Task:"):
                    skills.add(text.rsplit("Skill:", 1)[-1].strip())
    return sorted(skills)


def _episode_skills(ref: EpisodeRef, subtask_index: np.ndarray, vocab: list[str]) -> np.ndarray:
    table = {}
    for line in (Path(ref.path).parents[2] / "meta" / "subtasks.jsonl").read_text().splitlines():
        s = json.loads(line)
        text = s["subtask"]
        table[s["subtask_index"]] = "none" if text.startswith("Task:") else text.rsplit("Skill:", 1)[-1].strip()
    return np.array([vocab.index(table.get(int(i), "none")) for i in subtask_index], dtype=np.int64)


class FrameData:
    def __init__(self, cfg: dict, refs: list[EpisodeRef], vocab: list[str], task_names: list[str], latents: str = "latents"):
        self.cfg, self.refs, self.vocab, self.task_names = cfg, refs, vocab, task_names
        J = joint_dims(cfg)
        we = cfg["model"].get("wrench_expert", {}) or {}
        fw = int(we.get("filter_window", 5))

        def load(ref):
            e = load_numeric(ref)
            tac = e["observation.tactile"]
            c = np.cumsum(np.concatenate([np.zeros_like(tac[:1]), tac]), 0, dtype=np.float64)   # causal moving average
            lo = np.maximum(np.arange(len(tac)) + 1 - fw, 0)
            filt = ((c[1:] - c[lo]) / (np.arange(len(tac)) + 1 - lo)[:, None]).astype(np.float32)
            return (e["observation.state"][:, :J], e["action"][:, :J], e["observation.state.joint_torque"][:, :58],
                    tac, _episode_skills(ref, e["subtask_index"], vocab),
                    np.load(feature_path(cfg, ref)), filt)

        with ThreadPoolExecutor(16) as ex:
            eps = list(ex.map(load, refs))
        lens = np.array([len(e[0]) for e in eps])
        cat = lambda i: torch.from_numpy(np.concatenate([e[i] for e in eps]))  # noqa: E731
        self.state, self.action, self.torque, self.tactile, self.skill, self.feats = (cat(i) for i in range(6))
        self.wrench_on = bool(we.get("enabled", False))
        self.adv_on = bool((cfg["model"].get("advantage") or {}).get("enabled", False))
        if self.adv_on:                                                          # stage-2 labels from scripts/rl_critic.py
            tab = rl.Table(Path(cfg["rl"]["out_dir"]) / "credit.npz")
            self.adv_label = torch.from_numpy(np.concatenate([tab.get(r.uid, "label") for r in refs]))
            self.adv_intensity = torch.from_numpy(np.concatenate([tab.get(r.uid, "intensity") for r in refs]))
            assert len(self.adv_label) == len(self.state)
        self.tac_latent = None
        if (cfg["model"].get("tactile_expert") or {}).get("enabled", False):     # TacForeSight latents (scripts/tf_features.py), [N, 16, 128] float16
            base = Path(cfg["tactile_world_model"]["data_dir"]) / latents      # "latents_rest" = the zero-force input check
            assert base.exists(), f"{base} is missing: run scripts/tf_features.py" + (" rest=true" if latents != "latents" else "")
            self.tac_latent = torch.from_numpy(np.concatenate([np.load(base / r.task / r.season / f"episode_{r.episode_index:06d}.npy") for r in refs]))
            assert len(self.tac_latent) == len(self.state)
        self.tactile_filt = cat(6) if self.wrench_on else None                   # wrench targets of the wrench expert
        self.whist = -torch.arange(int(we.get("window", 10)) - 1, -1, -1)        # consecutive frames t-9 .. t
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        self.ep_id = torch.from_numpy(np.repeat(np.arange(len(refs)), lens))
        self.ep_start = torch.from_numpy(np.repeat(starts, lens))
        self.ep_end = torch.from_numpy(np.repeat(starts + lens, lens))          # exclusive
        self.task_id = torch.from_numpy(np.repeat([task_names.index(r.task) for r in refs], lens))
        peak = self.tactile.view(-1, 10, 6)[:, :, :3].norm(dim=2).amax(1)        # max fingertip |F| (N)
        self.peak_force = peak
        self.contact = peak > cfg["data"]["contact_threshold_n"]
        self.level = torch.bucketize(peak, torch.tensor(cfg["model"]["decision"]["force_levels_n"]))
        m = cfg["model"]
        self.hist = -torch.arange(m["obs_history"] - 1, -1, -1) * m["obs_stride"]   # e.g. [-10, ..., -2, 0]
        self.fut = torch.arange(m["chunk"])
        self.dec_h = m["decision"]["horizon"]

    def __len__(self):
        return len(self.state)

    def batch(self, g: torch.Tensor, device="cuda") -> dict:
        """g: [B] global frame indices (CPU)."""
        s, e = self.ep_start[g, None], self.ep_end[g, None]
        ih = torch.maximum(g[:, None] + self.hist, s)                    # [B, K] history frames, padded at start
        if_ = g[:, None] + self.fut                                      # [B, H] action frames t .. t+H-1
        iff = if_ + 1                                                    # [B, H] force frames t+1 .. t+H
        idec = torch.minimum(g + self.dec_h, self.ep_end[g] - 1)
        out = {
            "feats": self.feats[g],                                      # [B, C, tokens, D] float16
            "state": self.state[ih], "tactile": self.tactile[ih], "torque": self.torque[ih],
            "task_id": self.task_id[g],
            "action": self.action[torch.minimum(if_, e - 1)],            # [B, H, J] absolute joint targets
            "action_valid": if_ < e,
            "force": self.tactile[torch.minimum(iff, e - 1)].view(len(g), -1, 10, 6)[..., :3].flatten(2),  # [B,H,30] N
            "force_valid": iff < e,
            "contact_soon": self.contact[idec].float(), "level": self.level[idec], "skill": self.skill[g],
            "contact_now": self.contact[g], "ep_id": self.ep_id[g],
        }
        if self.wrench_on:
            out["wrench_hist"] = self.tactile[torch.maximum(g[:, None] + self.whist, s)]         # [B, W, 60]
            out["wrench_target"] = self.tactile_filt[torch.minimum(iff, e - 1)]                  # [B, H, 60] filtered, t+1 .. t+H
        if self.tac_latent is not None:
            out["tac_latent"] = self.tac_latent[g]
        if self.adv_on:
            out["adv_label"], out["adv_intensity"] = self.adv_label[g], self.adv_intensity[g]
        return {k: v.to(device, non_blocking=True) for k, v in out.items()}

    def stats(self) -> dict:
        """Normalisation statistics; call on the training set only."""
        def floor(std, frac=0.05):
            return torch.clamp(std, min=frac * std.max())

        idx = torch.arange(len(self))
        e = self.ep_end[idx, None]
        if_ = idx[:, None] + self.fut
        delta = self.action[torch.minimum(if_, e - 1)] - self.state[:, None]      # [N, H, J]
        valid = (if_ < e)[..., None].float()
        dstd = ((delta ** 2 * valid).sum((0, 1)) / valid.sum((0, 1))).sqrt()
        tac = self.tactile.view(-1, 10, 6)
        tac_std = torch.cat([floor(tac[..., :3].std(0)), floor(tac[..., 3:].std(0))], 1).flatten()
        # size of the force targets (N, one number): the future fingertip force itself, and its change from the force now
        force = tac[..., :3].flatten(1)                                           # [N, 30]
        iff = torch.minimum(if_ + 1, e - 1)
        fvalid = (if_ + 1 < e)[..., None].float()
        rms = lambda x: ((x ** 2 * fvalid).sum() / (fvalid.sum() * x.shape[-1])).sqrt()  # noqa: E731
        return {"state_mean": self.state.mean(0), "state_std": floor(self.state.std(0)),
                "torque_mean": self.torque.mean(0), "torque_std": floor(self.torque.std(0)),
                "tactile_std": tac_std, "delta_std": floor(dstd),
                "force_rms": rms(force[iff]), "force_change_rms": rms(force[iff] - force[:, None])}
