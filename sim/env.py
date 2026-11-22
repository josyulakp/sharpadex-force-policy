"""Stage-3 environment: the frozen policy's command stream runs in the simulated scene, an actor adds a bounded correction.

One episode = one recorded episode's scene and the frozen four-expert policy's commands for it (scripts/s3_prepare.py).
Every `decision_frames` frames the actor outputs a correction in [-1, 1]^(6 + 2k):
    left / right hand-base position offset (root frame), then k synergy offsets per hand (principal components of the
    operator's hand joint commands), each scaled to its box and rate-limited.
The actor's observation is the list `env.actor_blocks` (see `_obs`), then its own last correction and the time. The
block `objects` is where plate and eraser are relative to the hands: the frozen policy reads the recorded cameras, which
do not show the simulated scene, so this block stands in for live vision. The critic gets the actor's observation, the
blocks in `env.critic_blocks` and the simulator's contact and progress state (privileged). Reward and outcome:
sim/task.py and configs/stage3.yaml.
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

from .episode import Episode
from .replay import Sim
from .task import outcome, rotation, tilt_deg

HAND = {"left": slice(0, 22), "right": slice(22, 44)}                  # within the 44 hand joints


def fit_synergies(hand_cmds: np.ndarray, k: int) -> dict:
    """hand_cmds [N, 44] operator hand joint commands -> per hand: mean [22], components [k, 22] (unit norm), std [k]."""
    out = {}
    for side, sl in HAND.items():
        x = hand_cmds[:, sl]
        mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        out[f"{side}_mean"], out[f"{side}_pc"], out[f"{side}_std"] = mean, vt[:k], s[:k] / np.sqrt(len(x))
    return out


class ResidualEnv:
    def __init__(self, sim_cfg: dict, s3: dict, synergy: dict):
        self.sim_cfg, self.s3, self.e, self.syn = sim_cfg, s3, s3["env"], synergy
        self.k = self.e["synergies"]
        self.act_dim = 6 + 2 * self.k
        std = np.concatenate([synergy["left_std"], synergy["right_std"]])
        self.bound = np.r_[np.full(6, self.e["pos_bound_m"]), self.e["synergy_bound_std"] * std]      # physical size of action 1.0
        self.rate = np.r_[np.full(6, self.e["pos_rate_m"]), self.e["synergy_rate_std"] * std] / self.bound
        self.pc = np.zeros((2 * self.k, 44))                                                       # synergy offsets -> 44 hand joints
        self.pc[:self.k, HAND["left"]], self.pc[self.k:, HAND["right"]] = synergy["left_pc"], synergy["right_pc"]

    # ---- episode ------------------------------------------------------------------------------------------
    def reset(self, stream: str, objects: dict, rng: np.random.Generator | None = None, zero_stream: str = "base"):
        """stream: npz of scripts/s3_prepare.py. objects: the scene (scenes.json). rng: perturb the scene if given."""
        z = np.load(stream, allow_pickle=False)
        self.z = {k: z[k] for k in ("token", "gate", "contact_prob", "phase", "force_pred")}
        self.uid = str(z["uid"])
        self.poses, self.targets = z[f"poses__{zero_stream}"].astype(float), z[f"targets__{zero_stream}"].astype(float)
        cfg, obj = copy.deepcopy(self.sim_cfg), copy.deepcopy(objects)
        if rng is not None:
            r = self.e["randomize"]
            for name in ("plate", "eraser"):
                obj[f"{name}_xyz"] = (np.asarray(obj[f"{name}_xyz"]) + np.r_[rng.uniform(-1, 1, 2) * r["object_xy_m"], 0]).tolist()
                cfg[name]["friction"] *= rng.uniform(*r["friction"])
                cfg[name]["mass"] *= rng.uniform(*r["mass"])
            obj["eraser_yaw"] += float(np.radians(rng.uniform(-1, 1) * r["eraser_yaw_deg"]))
        self.sim = sim = Sim(cfg, Episode(self.uid), obj)
        self.hand_ref = self.targets[:, ~sim.is_arm]                                            # [T, 44]
        D = self.e["decision_frames"]
        self.T = min(len(self.poses), self.e["max_frames"]) // D * D
        sim.reset(self.poses[0])
        self.t, self.res = 0, np.zeros(self.act_dim)
        self.rest = {n: sim.d.xpos[sim.obj[n]].copy() for n in ("plate", "eraser")}
        self.flags = {"plate": False, "eraser": False, "path": 0.0}
        self.wiping = sim.ep.subtask == self.s3["reward"]["wipe_subtask"]                         # [frames] the recording's wipe phase
        self.rel_prev, self.prev_obj, self.recent = None, None, []
        self.log = []
        self.last = sim.step(self.poses[0], self.hand_ref[0])                                    # one frame so that every signal exists
        self.log.append(self.last)
        self.t = 1
        return self._obs()

    def _command(self, t: int, res: np.ndarray):
        pose = self.poses[t].copy()
        pose[:, :3] += (res[:6] * self.bound[:6]).reshape(2, 3)
        return pose, self.hand_ref[t] + (res[6:] * self.bound[6:]) @ self.pc

    def step(self, action: np.ndarray):
        """action [act_dim] in [-1, 1]. Returns (actor obs, critic obs, reward, done, info); info["applied"] is the
        rate-limited correction that was executed (what the critic should be trained on)."""
        e, sim = self.e, self.sim
        new = np.clip(np.clip(action, -1, 1), self.res - self.rate, self.res + self.rate)
        D, reward, dropped = e["decision_frames"], 0.0, False
        for i in range(D):
            t = min(self.t + i, len(self.poses) - 1)
            self.last = sim.step(*self._command(t, self.res + (new - self.res) * (i + 1) / D))
            self.log.append(self.last)
            reward += self._progress(self.last, t)
            unstable = sim.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0     # MuJoCo reset the scene: the episode is void
            if unstable or self.last["plate"][2] - self.rest["plate"][2] < -e["drop_m"]:
                dropped = True
                break
        self.res, self.t = new, self.t + D
        done = dropped or self.t + D > self.T
        info = {"applied": new.copy()}
        if done:
            log = {k: np.stack([l[k] for l in self.log]) for k in ("plate", "eraser", "wipe_force", "base_err_mm", "plate_low")}
            log["plate"][0, :3], log["eraser"][0, :3] = self.rest["plate"], self.rest["eraser"]
            info["outcome"] = o = outcome(log, self.s3["outcome"])
            if sim.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
                o["success"] = o["success_strict"] = o["plate_placed"] = False
            rw = self.s3["reward"]
            reward += rw["dropped"] if dropped else rw["placed"] * float(o["plate_placed"] and self.flags["plate"])
            info["frames"] = len(self.log)
        return (*self._obs(), reward, done, info)

    def _progress(self, s: dict, t: int) -> float:
        """Reward earned in one frame (configs/stage3.yaml, section `reward`), with the outcome's own thresholds."""
        c, rw, f, r = self.s3["outcome"], self.s3["reward"], self.flags, 0.0
        lift = {n: s[n][2] - self.rest[n][2] for n in ("plate", "eraser")}
        for n in ("plate", "eraser"):
            if not f[n] and lift[n] > c["lift_m"]:
                f[n], r = True, r + rw[f"{n}_lift"]
        if self.wiping[min(t, len(self.wiping) - 1)] and lift["plate"] > c["lift_m"]:
            r += rw["hold"] / max(int(self.wiping[:self.T].sum()), 1)
        rel = s["eraser"][:3] - s["plate"][:3]
        if c.get("path", "raw") == "smooth":                                                    # as sim/task.py: plate frame, mean of the last frames
            n = int(c.get("smooth_frames", 9))
            local = rel @ rotation(s["plate"][None, 3:])[0]
            self.recent = (self.recent or [local] * n)[1 - n:] + [local] if n > 1 else [local]
            rel = np.mean(self.recent, 0)
        if self.rel_prev is not None and s["wipe_force"] > c["wipe_force_n"] and lift["plate"] > c["lift_m"]:
            new = min(f["path"] + float(np.linalg.norm(rel - self.rel_prev)), rw["wipe_full_m"])
            r, f["path"] = r + rw["wipe"] * (new - f["path"]) / rw["wipe_full_m"], new
        self.rel_prev = rel
        return r

    # ---- observations -------------------------------------------------------------------------------------
    def _obs(self):
        sim, s, t = self.sim, self.last, min(self.t, len(self.poses) - 1)
        D = self.e["decision_frames"]
        d, W = sim.d, sim.world_from_root
        Rw = W[:3, :3]
        base = np.stack([d.xpos[sim.base[side]] for side in ("left", "right")])                  # world
        target = np.stack([d.mocap_pos[sim.mocap[side]] for side in ("left", "right")])
        ahead = lambda n: self.poses[min(t + n, len(self.poses) - 1), :, :3] - self.poses[t, :, :3]  # noqa: E731
        syn_ref = np.r_[(self.hand_ref[t, HAND["left"]] - self.syn["left_mean"]) @ self.syn["left_pc"].T / self.syn["left_std"],
                        (self.hand_ref[t, HAND["right"]] - self.syn["right_mean"]) @ self.syn["right_pc"].T / self.syn["right_std"]]
        tz = self.z
        pos = {n: s[n][:3] for n in ("plate", "eraser")}
        vel = {n: (pos[n] - self.prev_obj[n]) * 30 / D if self.prev_obj else np.zeros(3) for n in pos}
        self.prev_obj = {n: pos[n].copy() for n in pos}
        Rb = [d.xmat[sim.base[side]].reshape(3, 3) for side in ("left", "right")]
        Ro = {n: rotation(s[n][None, 3:])[0] for n in pos}
        coords = lambda q: np.r_[(q[HAND["left"]] - self.syn["left_mean"]) @ self.syn["left_pc"].T / self.syn["left_std"],  # noqa: E731
                                 (q[HAND["right"]] - self.syn["right_mean"]) @ self.syn["right_pc"].T / self.syn["right_std"]]
        block = {
            "token": tz["token"][t],                                                             # frozen policy, from the recorded observations
            "phase": np.r_[tz["contact_prob"][t], tz["phase"][t]],
            "gate": tz["gate"][t], "force_forecast": tz["force_pred"][t] / 10,
            "hand_joints": s["hand_q"], "joint_error": (sim.hand - s["hand_q"]) * 5,             # the simulated hands
            "synergy": coords(s["hand_q"]), "synergy_error": coords(sim.hand) - coords(s["hand_q"]),
            "base_error": ((target - base) @ Rw).ravel() / 0.02, "tip_force": (s["tip_force"] @ Rw).ravel() / 10,
            "ahead": np.r_[ahead(D).ravel() / 0.02, ahead(3 * D).ravel() / 0.04], "synergy_ref": syn_ref,   # the reference motion
            "objects": np.concatenate([                                                          # what live cameras would show
                np.concatenate([(pos[n] - base[i]) @ Rb[i] for i in (0, 1) for n in ("plate", "eraser")]) / 0.1,   # objects in each hand's frame
                (pos["eraser"] - pos["plate"]) @ Ro["plate"] / 0.1,                               # eraser in the plate's frame
                Ro["plate"][:, 2] @ Rw, Ro["eraser"][:, 2] @ Rw,                                  # their up axes
                [(pos[n][2] - self.rest[n][2]) / 0.1 for n in ("plate", "eraser")], vel["plate"] @ Rw, vel["eraser"] @ Rw])}
        names = self.e.get("actor_blocks")
        if names is None:                                                                        # the first runs' fixed layout
            names = (["token"] if self.e.get("actor_token", True) else []) + ["phase", "gate", "force_forecast", "hand_joints", "joint_error",
                     "base_error", "tip_force", "ahead", "synergy_ref"] + (["objects"] if self.e.get("actor_objects", False) else [])
        actor = np.concatenate([*(block[n] for n in names), self.res, [self.t / self.T]]).astype(np.float32)
        extra = [block[n] for n in self.e.get("critic_blocks", [])]                              # more context for the critic only
        priv = np.concatenate([
            (pos["plate"] - base[0]) @ Rw / 0.1, (pos["eraser"] - base[1]) @ Rw / 0.1, (pos["eraser"] - pos["plate"]) @ Rw / 0.1,
            [(pos["plate"][2] - self.rest["plate"][2]) / 0.1, (pos["eraser"][2] - self.rest["eraser"][2]) / 0.1,
             tilt_deg(s["plate"][3:]) / 90, tilt_deg(s["eraser"][3:]) / 90, s["wipe_force"] / 20,
             self.flags["path"] / self.s3["reward"]["wipe_full_m"], float(self.flags["plate"]), float(self.flags["eraser"])],
            vel["plate"], vel["eraser"]]).astype(np.float32)
        return actor, np.concatenate([actor, *extra, priv]).astype(np.float32)
