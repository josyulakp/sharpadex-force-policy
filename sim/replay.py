"""Execute a command stream (hand-base poses + hand joint targets) in the simulated scene.

The hand bases are pulled to the commanded poses by soft welds; the arms follow passively (gravity
compensated). The 44 hand joints are position-controlled with the model's own servo gains. World frame =
the robot's root frame (the base is fixed), so recorded poses and camera-located objects need no fitting.
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


class Sim:
    def __init__(self, cfg: dict, ep: Episode):
        self.cfg, self.ep = cfg, ep
        self.torso = ep.state[0, 58:63]
        self.objects = o = self.locate_objects()
        self.m = m = build(cfg, self.torso, plate_pose=(o["plate_xyz"], 0.0), eraser_pose=(o["eraser_xyz"], o["eraser_yaw"]),
                           table_z=o["table_z"], plate_radius=o["plate_radius"], plate_height=o["plate_top_z"] - o["table_z"])
        self.d = mujoco.MjData(m)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        names = joint_names()
        self.adr = qpos_adr(m, names)
        self.dof = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
        act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]): i for i in range(m.nu)}
        self.act = np.array([act[n] for n in names])
        self.is_arm = np.zeros(58, bool)
        self.is_arm[ARM_DIMS["left"]] = self.is_arm[ARM_DIMS["right"]] = True
        self.kd = np.tile(cfg["arm_kd"], 2)
        self.mocap = {s: m.body_mocapid[bid(f"{s}_target")] for s in SIDES}
        self.base = {s: bid(f"{s}_hand_base") for s in SIDES}
        self.root = bid("north")
        self.obj = {n: bid(n) for n in ("plate", "eraser")}
        self.forces = FingertipForces(m)
        self.substeps = round(1 / 30 / m.opt.timestep)
        self.tips = [bid(f"{s}_{f}_DP") for s in SIDES for f in FINGERS]

    # ---- where things are ---------------------------------------------------------------------------------
    def locate_objects(self) -> dict:
        """Plate from the stereo cameras; eraser from where the right hand first closes on it; table under the plate."""
        m = build(self.cfg, self.torso)
        d = mujoco.MjData(m)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        adr = qpos_adr(m, joint_names())
        mujoco.mj_kinematics(m, d)
        W = np.eye(4)                                                     # world <- root
        W[:3, :3], W[:3, 3] = d.xmat[bid("north")].reshape(3, 3), d.xpos[bid("north")]
        self.world_from_root = W

        def tips(side, t):                                                # world positions of one hand's fingertips
            d.qpos[:] = m.qpos0
            d.qpos[adr[HAND_DIMS[side]]] = self.ep.state[t, HAND_DIMS[side]]
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
    def _set_targets(self, pose: np.ndarray):
        """pose [2, 12]: position + rotation matrix of the left / right hand base in the root frame."""
        W = self.world_from_root
        for k, s in enumerate(SIDES):
            self.d.mocap_pos[self.mocap[s]] = W[:3, :3] @ pose[k, :3] + W[:3, 3]
            self.d.mocap_quat[self.mocap[s]] = np.roll(R.from_matrix(W[:3, :3] @ pose[k, 3:].reshape(3, 3)).as_quat(), 1)

    def _physics(self, hand_target: np.ndarray):
        m, d, arm = self.m, self.d, self.is_arm
        d.ctrl[self.act[arm]] = d.qfrc_bias[self.dof[arm]] - self.kd * d.qvel[self.dof[arm]]
        lo, hi = m.actuator_ctrlrange[self.act[~arm]].T
        d.ctrl[self.act[~arm]] = np.clip(hand_target, lo, hi)
        mujoco.mj_step(m, d)

    def _ik(self, iters: int = 200):
        """Arm joints that put each hand base on its current target (damped least squares, kinematics only)."""
        m, d = self.m, self.d
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for s, sl in ARM_DIMS.items():
            dof, adr = self.dof[sl], self.adr[sl]
            lo, hi = m.jnt_range[m.dof_jntid[dof]].T
            for _ in range(iters):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                b, k = self.base[s], self.mocap[s]
                e_pos = d.mocap_pos[k] - d.xpos[b]
                q_err = np.zeros(4)
                mujoco.mju_mulQuat(q_err, d.mocap_quat[k], d.xquat[b] * [1, -1, -1, -1])
                e_rot = np.zeros(3)
                mujoco.mju_quat2Vel(e_rot, q_err, 1.0)
                mujoco.mj_jacBody(m, d, jp, jr, b)
                J, e = np.vstack([jp[:, dof], jr[:, dof]]), np.r_[e_pos, e_rot]
                d.qpos[adr] = np.clip(d.qpos[adr] + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e) * 0.5, lo, hi)

    def reset(self, pose0: np.ndarray):
        """Recorded torso and hand joints, arms solved onto the first commanded hand poses, then a short settle."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        jadr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
        for j, v in zip(self.cfg["locked_joints"], self.torso):
            d.qpos[jadr(j)] = v
        d.qpos[self.adr[~self.is_arm]] = self.ep.state[0, :58][~self.is_arm]
        d.qpos[self.adr[self.is_arm]] = np.tile(self.cfg["arm_start"], 2)
        self.pose, self.hand = pose0.copy(), self.ep.state[0, :58][~self.is_arm].copy()
        self._set_targets(pose0)
        self._ik()
        mujoco.mj_forward(m, d)
        for _ in range(int(self.cfg["settle_s"] / m.opt.timestep)):
            self._physics(self.hand)

    def step(self, pose: np.ndarray, hand: np.ndarray) -> dict:
        """One 30 Hz frame: move the hand-base targets to `pose` [2, 12] and the hand joint targets to `hand` [44]."""
        fsum = np.zeros((10, 3))
        for k in range(self.substeps):
            a = (k + 1) / self.substeps
            self._set_targets(self.pose + a * (pose - self.pose))        # small steps: linear blend of the matrices
            self._physics(self.hand + a * (hand - self.hand))
            fsum += self.forces.read(self.d)
        self.pose, self.hand = pose.copy(), hand.copy()
        d = self.d
        out = {"hand_q": d.qpos[self.adr[~self.is_arm]].copy(), "tip_force": fsum / self.substeps,
               "tip_pos": d.xpos[self.tips].copy(),
               "base_err_mm": np.array([np.linalg.norm(d.xpos[self.base[s]] - d.mocap_pos[self.mocap[s]]) * 1000 for s in SIDES])}
        for name, b in self.obj.items():
            out[name] = np.r_[d.xpos[b], d.xquat[b]]
        out["wipe_force"] = self._pair_force("eraser", "plate")
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
