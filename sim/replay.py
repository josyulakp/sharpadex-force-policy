"""Execute a command stream (hand-base poses + hand joint targets) in the simulated scene.

The two hands float: each hand base is pulled to its commanded pose by a weld (sim/scene.py explains why the arms
are left out). The 44 hand joints are position-controlled with the model's own servo gains. World frame = the
robot's root frame (the base is fixed), so recorded poses and camera-located objects need no fitting.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .episode import Episode, pose_matrix
from .robot import ARM_DIMS, HAND_DIMS, joint_names, qpos_adr
from .scene import FINGERS, FingertipForces, build

SIDES = ("left", "right")
TCP = {"left": slice(0, 6), "right": slice(6, 12)}


def _pose(d: mujoco.MjData, body: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d.xmat[body].reshape(3, 3), d.xpos[body]
    return T


class Sim:
    def __init__(self, cfg: dict, ep: Episode, objects: dict | None = None):
        """objects: where things are, as returned by locate_objects (from an earlier build, or perturbed); located if None."""
        self.cfg, self.ep = cfg, ep
        self.objects = objects if objects is not None else self.locate_objects()
        o, c = dict(self.objects), cfg.get("calib") or {}                 # calibrated corrections of the located scene (configs/sim.yaml)
        o["plate_xyz"] = np.asarray(o["plate_xyz"], float) + [c.get("plate_dx", 0.0), c.get("plate_dy", 0.0), 0.0]
        o["plate_top_z"], o["plate_radius"] = o["plate_top_z"] + c.get("plate_dz", 0.0), o["plate_radius"] + c.get("plate_dr", 0.0)
        o["table_z"] = o["table_z"] + c.get("table_dz", 0.0)
        self.m = m = build(cfg, plate_pose=(o["plate_xyz"], 0.0), eraser_pose=(o["eraser_xyz"], o["eraser_yaw"]),
                           table_z=o["table_z"], plate_radius=o["plate_radius"], plate_height=o["plate_top_z"] - o["table_z"])
        self.d = mujoco.MjData(m)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        mujoco.mj_kinematics(m, self.d)                                   # world <- root (static)
        self.world_from_root = _pose(self.d, bid("north"))
        self.link7 = {s: _pose(self.d, bid(f"{s}_arm_link_7")) for s in SIDES}       # static parents of the floating hand mounts
        self.is_arm = np.zeros(58, bool)                                  # within the 58 dataset joints
        self.is_arm[ARM_DIMS["left"]] = self.is_arm[ARM_DIMS["right"]] = True
        names = [n for n, arm in zip(joint_names(), self.is_arm) if not arm]             # the 44 hand joints
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.adr = qpos_adr(m, names)
        act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]): i for i in range(m.nu)}
        self.act = np.array([act[n] for n in names])
        self.free = {s: (np.array([m.jnt_qposadr[jid(f"{s}_base_{a}")] for a in "xyz"]), m.jnt_qposadr[jid(f"{s}_base_rot")]) for s in SIDES}
        self.mount_from_base = {}
        for s in SIDES:                                                   # hand base in its mount (constant)
            Tm, Tb = _pose(self.d, bid(f"{s}_hand_mount")), _pose(self.d, bid(f"{s}_hand_base"))
            self.mount0 = getattr(self, "mount0", {}) | {s: np.linalg.inv(self.link7[s]) @ Tm}
            self.mount_from_base[s] = np.linalg.inv(Tm) @ Tb
        self.mocap = {s: m.body_mocapid[bid(f"{s}_target")] for s in SIDES}
        self.base = {s: bid(f"{s}_hand_base") for s in SIDES}
        self.root = bid("north")
        self.obj = {n: bid(n) for n in ("plate", "eraser")}
        self.plate_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj["plate"]]   # cylinders
        self.forces = FingertipForces(m)
        self.substeps = round(1 / 30 / m.opt.timestep)
        self.tips = [bid(f"{s}_{f}_DP") for s in SIDES for f in FINGERS]

    # ---- where things are ---------------------------------------------------------------------------------
    def locate_objects(self) -> dict:
        """Plate from the stereo cameras; eraser from where the right hand first closes on it; table under the plate."""
        m = build(self.cfg)
        d = mujoco.MjData(m)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        names = joint_names()
        adr = {side: qpos_adr(m, names[HAND_DIMS[side]]) for side in SIDES}
        mujoco.mj_kinematics(m, d)
        W = _pose(d, bid("north"))                                        # world <- root

        def tips(side, t):                                                # world positions of one hand's fingertips
            d.qpos[:] = m.qpos0
            d.qpos[adr[side]] = self.ep.state[t, HAND_DIMS[side]]
            mujoco.mj_kinematics(m, d)
            b = bid(f"{side}_hand_base")
            Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
            T = W @ pose_matrix(self.ep.tcp[t, TCP[side]])
            return np.array([T[:3, :3] @ (Rb.T @ (d.xpos[bid(f"{side}_{f}_DP")] - pb)) + T[:3, 3] for f in FINGERS])

        right_tips = lambda t: tips("right", t)  # noqa: E731
        low = np.array([[tips(s, t)[:, 2].min() for s in SIDES] for t in range(0, len(self.ep), 3)])
        plate = self.ep.locate_plate()
        centre = (W @ np.r_[plate["centre_root"], 1])[:3]
        p = self.cfg["plate"]
        # table height: the lowest point any fingertip body reaches, minus the fingertip half-thickness
        table_z = float(np.percentile(low.min(1), 1)) - self.cfg["fingertip_radius"]
        # the rim is where the left thumb (above) and index finger (below) first close on it: this fixes the plate
        # height and radius better than stereo depth, whose error lies along the downward viewing direction
        fl = self.ep.force[:, :5]
        t_rim = int(np.argmax((fl[:, :2].min(1) > self.cfg["grasp_force_n"]) & (self.ep.subtask == self.cfg["plate_subtask"])))
        lt = tips("left", t_rim)
        rim = (lt[0] + lt[1]) / 2
        plate_top = float(rim[2] + p["rim_thickness"] / 2)
        # radius: the pinch point when it agrees with the stereo disc; otherwise the plate was pushed before the pinch
        radius = float(np.linalg.norm(rim[:2] - centre[:2]) + self.cfg["rim_overlap"])
        if abs(radius - self.cfg["rim_overlap"] - plate["radius"]) > self.cfg["rim_radius_tolerance"]:
            radius, plate_top = float(plate["radius"]), float(centre[2])
        f = self.ep.force[:, 5:]
        t_grasp = int(np.argmax((f.max(1) > self.cfg["grasp_force_n"]) & (self.ep.subtask == self.cfg["eraser_subtask"])))
        tp = right_tips(t_grasp)
        touching = f[t_grasp] > 0.3 * f[t_grasp].max()
        fingers = tp[1:][touching[1:]] if touching[1:].any() else tp[1:3]
        mid = (tp[0] + fingers.mean(0)) / 2                               # between the thumb and the closing fingers
        across = (fingers.mean(0) - tp[0])[:2]                            # the hand closes across the short side
        return {"plate_xyz": centre, "plate_radius": radius, "plate_radius_stereo": plate["radius"], "plate_grasp_frame": t_rim,
                "table_z": float(table_z), "eraser_xyz": mid,
                "eraser_yaw": float(np.arctan2(across[1], across[0]) + np.pi / 2), "eraser_grasp_frame": t_grasp,
                "eraser_grasp_tip_z": float(tp[:, 2].min()),
                "lowest_tip_z_left_right": low.min(0).tolist(), "plate_top_z": plate_top, "plate_top_z_stereo": float(centre[2])}

    # ---- execution ----------------------------------------------------------------------------------------
    def _world_targets(self, pose: np.ndarray):
        """pose [2, 12]: position + rotation matrix of the left / right hand base in the root frame -> world position [2, 3]
        and quaternion [2, 4] (w, x, y, z)."""
        W = self.world_from_root
        pos = pose[:, :3] @ W[:3, :3].T + W[:3, 3]
        quat = np.roll(R.from_matrix(W[:3, :3] @ pose[:, 3:].reshape(2, 3, 3)).as_quat(), 1, axis=1)
        return pos, quat

    def _set_targets(self, pos: np.ndarray, quat: np.ndarray):
        for k, s in enumerate(SIDES):
            self.d.mocap_pos[self.mocap[s]] = pos[k]
            self.d.mocap_quat[self.mocap[s]] = quat[k] / np.linalg.norm(quat[k])

    def _physics(self, hand_target: np.ndarray):
        m, d = self.m, self.d
        lo, hi = m.actuator_ctrlrange[self.act].T
        d.ctrl[self.act] = np.clip(hand_target, lo, hi)
        mujoco.mj_step(m, d)

    def _place_bases(self):
        """Put each hand base on its current target: slides and ball joint of the hand mount, relative to its static parent."""
        d = self.d
        for s in SIDES:
            T = np.eye(4)
            T[:3, :3] = R.from_quat(np.roll(d.mocap_quat[self.mocap[s]], -1)).as_matrix()
            T[:3, 3] = d.mocap_pos[self.mocap[s]]
            M = np.linalg.inv(self.link7[s]) @ T @ np.linalg.inv(self.mount_from_base[s])        # mount in link 7
            slide, ball = self.free[s]
            d.qpos[slide] = np.linalg.inv(self.mount0[s][:3, :3]) @ (M[:3, 3] - self.mount0[s][:3, 3])
            d.qpos[ball:ball + 4] = np.roll(R.from_matrix(np.linalg.inv(self.mount0[s][:3, :3]) @ M[:3, :3]).as_quat(), 1)

    def reset(self, pose0: np.ndarray):
        """Recorded hand joints, hand bases on the first commanded poses, then a short settle."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.adr] = self.ep.state[0, :58][~self.is_arm]
        self.target, self.hand = self._world_targets(pose0), self.ep.state[0, :58][~self.is_arm].copy()
        self._set_targets(*self.target)
        self._place_bases()
        mujoco.mj_forward(m, d)
        for _ in range(int(self.cfg["settle_s"] / m.opt.timestep)):
            self._physics(self.hand)

    def step(self, pose: np.ndarray, hand: np.ndarray) -> dict:
        """One 30 Hz frame: move the hand-base targets to `pose` [2, 12] and the hand joint targets to `hand` [44]."""
        fsum = np.zeros((10, 3))
        (p0, q0), (p1, q1) = self.target, self._world_targets(pose)
        q1 = q1 * np.where((q0 * q1).sum(1, keepdims=True) < 0, -1.0, 1.0)  # same hemisphere, then a normalised linear blend
        for k in range(self.substeps):
            a = (k + 1) / self.substeps
            self._set_targets(p0 + a * (p1 - p0), q0 + a * (q1 - q0))
            self._physics(self.hand + a * (hand - self.hand))
            fsum += self.forces.read(self.d)
        self.target, self.hand = (p1, q1), hand.copy()
        d = self.d
        out = {"hand_q": d.qpos[self.adr].copy(), "tip_force": fsum / self.substeps,
               "tip_pos": d.xpos[self.tips].copy(),
               "base_err_mm": np.array([np.linalg.norm(d.xpos[self.base[s]] - d.mocap_pos[self.mocap[s]]) * 1000 for s in SIDES])}
        for name, b in self.obj.items():
            out[name] = np.r_[d.xpos[b], d.xquat[b]]
        out["wipe_force"] = self._pair_force("eraser", "plate")
        az = d.geom_xmat[self.plate_geoms].reshape(-1, 3, 3)[:, 2, 2]                              # z of each cylinder's axis
        r, h = self.m.geom_size[self.plate_geoms, 0], self.m.geom_size[self.plate_geoms, 1]
        out["plate_low"] = float((d.geom_xpos[self.plate_geoms, 2] - h * np.abs(az) - r * np.sqrt(np.clip(1 - az ** 2, 0, 1))).min())  # its lowest point
        return out

    def _pair_force(self, a: str, b: str) -> float:
        """Normal contact force between two objects at the end of the frame (N)."""
        m, d, f6, total = self.m, self.d, np.zeros(6), 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} == {self.obj[a], self.obj[b]}:
                mujoco.mj_contactForce(m, d, i, f6)
                total += f6[0]
        return total

    def run(self, poses: np.ndarray, targets: np.ndarray, render=None, stride: int = 2) -> dict:
        """poses [T, 2, 12], targets [T, 58] (only the hand joints are used)."""
        self.reset(poses[0])
        log, frames = [], []
        for t in range(len(poses)):
            log.append(self.step(poses[t].astype(float), targets[t][~self.is_arm].astype(float)))
            if render is not None and t % stride == 0:
                render.update_scene(self.d, camera="view")
                frames.append(render.render().copy())
        out = {k: np.stack([l[k] for l in log]) for k in log[0]}
        out["frames"] = frames
        return out
