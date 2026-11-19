"""Teacher data of the C3 contact MPC (dairlib fork, examples/c3_python) as a second embodiment.

Each row is one control step of a closed-loop C3 rollout on the finger-ball model:
    obs       [9]      state (6), ball target (1), gait flag and clock (2)
    u_plan    [N, 2]   the input plan C3 returned (N = horizon), first row is what it executes
    lam_plan  [N, 4]   planned contact variables [gamma, lam+, lam-, lam_n]
Episodes are split 90 / 10 by a seed, like the SharpaDex episodes.
"""
from __future__ import annotations

import numpy as np
import torch


class MPCData:
    def __init__(self, cfg: dict, split: str):
        m, mc = cfg["model"], cfg["mpc"]
        d = np.load(mc["data"])
        eps = np.unique(d["episode"])
        rng = np.random.default_rng(mc["split_seed"])
        held = set(rng.permutation(eps)[:max(1, round(len(eps) * mc["holdout_fraction"]))].tolist())
        keep = np.array([(e in held) == (split == "heldout") for e in d["episode"]])
        self.obs = torch.from_numpy(d["obs"][keep])
        self.u = torch.from_numpy(d["u_plan"][keep])
        lam = torch.from_numpy(d["lam_plan"][keep])
        self.force = torch.stack([lam[..., 1] - lam[..., 2], lam[..., 3]], -1)       # friction on the ball, normal force
        ep = torch.from_numpy(d["episode"][keep].astype(np.int64))
        self.ep = ep
        first = torch.ones(len(ep), dtype=torch.bool)
        first[1:] = ep[1:] != ep[:-1]
        start = torch.nonzero(first).squeeze(1)
        self.ep_start = start[torch.cumsum(first.long(), 0) - 1]
        self.hist = -torch.arange(m["obs_history"] - 1, -1, -1) * m["obs_stride"]
        self.dec_h = min(m["decision"]["horizon"], self.u.shape[1] - 1)
        self.contact_thr = mc["contact_threshold_n"]
        self.episodes, self.horizon, self.dt = len(torch.unique(ep)), self.u.shape[1], float(d["dt"])

    def __len__(self):
        return len(self.obs)

    def batch(self, g: torch.Tensor, device="cuda") -> dict:
        ih = torch.maximum(g[:, None] + self.hist, self.ep_start[g, None])
        out = {"obs": self.obs[ih], "u": self.u[g], "force": self.force[g],
               "contact_soon": (self.force[g, self.dec_h, 1] > self.contact_thr).float(),
               "contact_now": self.force[g, 0, 1] > self.contact_thr, "ep_id": self.ep[g]}
        return {k: v.to(device) for k, v in out.items()}

    def stats(self) -> dict:
        return {"mpc_obs_mean": self.obs.mean(0), "mpc_obs_std": self.obs.std(0).clamp(min=1e-3),
                "mpc_u_std": self.u.flatten(0, 1).std(0), "mpc_force_std": self.force.flatten(0, 1).std(0)}
