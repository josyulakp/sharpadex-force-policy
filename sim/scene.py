"""MuJoCo scene for clean_plate_with_eraser: the two hands of the D01 robot, a table, a plate and an eraser.

Objects are simple shapes (the plate is a foot disc plus a wider rim disc, so fingers can reach under the
rim as in the videos). All sizes, masses and friction values live in configs/sim.yaml.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .robot import HAND, SCENE_XML

OBJ_BIT = 1 << 13                      # collision bit of table / plate / eraser (robot geoms use bits 0-12)
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


HAND_BASE_IN_LINK7 = {"left": ([0.055, 0, 0], [0, -1.5708, 3.1416]), "right": ([0.055, 0, 0], [0, 1.5708, 0])}   # URDF, rpy


def build(cfg: dict, plate_pose=None, eraser_pose=None, table_z=None, plate_radius=None, plate_height=None) -> mujoco.MjModel:
    """plate_pose / eraser_pose: (xyz, yaw) of the object on the table in the world frame. table_z / plate_radius /
    plate_height: values measured for the episode.

    Only the two hands move. The public arm model cannot reach the recorded hand poses (the left arm is at full stretch
    about 5 cm short of the plate, see sim/calibrate_arms.py), so each hand is cut loose from its wrist (three slides and a
    ball joint on the hand mount) and pulled to the commanded hand-base pose by a weld. The rest of the robot is static,
    hidden and does not collide."""
    s = mujoco.MjSpec.from_file(str(SCENE_XML))
    s.option.timestep = cfg["timestep"]
    s.option.impratio = cfg["impratio"]
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    hand_joints = {f"{side}_{j}" for side in HAND_BASE_IN_LINK7 for j in HAND}
    for x in list(s.equalities) + list(s.sensors) + [a for a in s.actuators if a.target not in hand_joints]:
        s.delete(x)                                                       # brakes (released by Sharpa's app at start-up), sensors, body actuators
    for j in [j for j in s.joints if j.name not in hand_joints]:
        s.delete(j)
    for a in s.actuators:                                                 # finger servo stiffness (position gain), calibrated
        a.gainprm[0] *= cfg.get("hand_kp_scale", 1.0)
        a.biasprm[1] *= cfg.get("hand_kp_scale", 1.0)
    w, t, p, e = s.worldbody, dict(cfg["table"]), dict(cfg["plate"]), cfg["eraser"]
    if table_z is not None:
        t["z"] = table_z
    if plate_radius is not None:
        p["radius"] = plate_radius
    if plate_height is not None:                                          # rim top above the table, measured
        p["foot_height"] = max(plate_height - p["rim_thickness"], 0.005)
    hand_bodies = set()
    for side, (xyz, rpy) in HAND_BASE_IN_LINK7.items():
        mount = s.body(f"{side}_hand_mount")
        for b in [mount] + list(mount.find_all(mujoco.mjtObj.mjOBJ_BODY)):
            b.gravcomp = 1.0                                              # the arm carries the hand's weight
            hand_bodies.add(b.name)
        for k, axis in enumerate(np.eye(3)):
            mount.add_joint(name=f"{side}_base_{'xyz'[k]}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
        mount.add_joint(name=f"{side}_base_rot", type=mujoco.mjtJoint.mjJNT_BALL)
        Rb = R.from_euler("xyz", rpy)                                     # hand base in link 7 (URDF) -> in the hand mount
        mount.add_body(name=f"{side}_hand_base", pos=np.asarray(xyz) - mount.pos, quat=np.roll(Rb.as_quat(), 1))
        w.add_body(name=f"{side}_target", mocap=True)
        q = s.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=f"{side}_target", name2=f"{side}_hand_base",
                           objtype=mujoco.mjtObj.mjOBJ_BODY, solref=cfg["weld_solref"])
        q.data = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    for b in w.find_all(mujoco.mjtObj.mjOBJ_BODY):
        if b.name not in hand_bodies:
            for g in b.geoms:
                g.contype, g.conaffinity, g.group = 0, 0, 4               # no collisions, not rendered
    col = dict(contype=OBJ_BIT, conaffinity=OBJ_BIT | (OBJ_BIT - 1), condim=4)
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t["half_x"], t["half_y"], 0.02],
               pos=[t["x"], t["y"], t["z"] - 0.02], rgba=[0.1, 0.12, 0.2, 1], friction=[t["friction"], 0.01, 0.001], **col)
    if plate_pose is not None:
        b = w.add_body(name="plate", pos=[*plate_pose[0][:2], t["z"] + 1e-3], quat=_yaw(plate_pose[1]))
        b.add_freejoint(name="plate_free")
        f = [p["friction"], 0.01, 0.001]
        height = p["foot_height"] + p["rim_thickness"]
        if p["steps"] == 1:                                               # a flat disc on a stem: fingers slide under the disc
            parts = [(p["foot_radius"], 0.0, p["foot_height"], 0.3), (p["radius"], p["foot_height"], height, 0.7)]
        else:                                                             # stacked discs widening from the foot to the rim
            n = p["steps"]
            parts = [(p["foot_radius"] + (p["radius"] - p["foot_radius"]) * (i / (n - 1)) ** p["flare"], height * i / n,
                      height * (i + 1) / n, 1 / n) for i in range(n)]
        for i, (r, z0, z1, share) in enumerate(parts):                    # (radius, bottom, top, share of the mass)
            b.add_geom(name=f"plate_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, (z1 - z0) / 2, 0], pos=[0, 0, (z0 + z1) / 2],
                       mass=p["mass"] * share, rgba=[0.95, 0.95, 0.92, 1], friction=f, **col)
    if eraser_pose is not None:
        h = e["half_size"]
        b = w.add_body(name="eraser", pos=[*eraser_pose[0][:2], t["z"] + h[2] + 1e-3], quat=_yaw(eraser_pose[1]))
        b.add_freejoint(name="eraser_free")
        f = [e["friction"], 0.01, 0.001]
        b.add_geom(name="eraser_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=h, mass=e["mass"] * 0.8, rgba=[0.93, 0.92, 0.88, 1],
                   friction=f, **col)
        g = e["handle_half_size"]                                         # the grip on top that the fingers pinch
        b.add_geom(name="eraser_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=g, pos=[0, 0, h[2] + g[2]], mass=e["mass"] * 0.2,
                   rgba=[0.8, 0.8, 0.76, 1], friction=f, **col)
    cam = cfg["camera"]
    w.add_body(name="view_target", pos=cam["target"])
    w.add_camera(name="view", pos=cam["pos"], mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="view_target")
    w.add_light(pos=[0.5, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    return s.compile()


def _yaw(a: float) -> list[float]:
    return [float(np.cos(a / 2)), 0.0, 0.0, float(np.sin(a / 2))]


class FingertipForces:
    """Contact force on each of the ten fingertip bodies (dataset order: left then right; thumb..little), in N."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{f}_DP") for side in ("left", "right") for f in FINGERS]
        self.tip_of_body = {b: i for i, b in enumerate(tips)}
        self.obj_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] & OBJ_BIT}

    def read(self, d: mujoco.MjData) -> np.ndarray:
        out, f6 = np.zeros((10, 3)), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for g, other, sign in ((c.geom1, c.geom2, 1.0), (c.geom2, c.geom1, -1.0)):
                tip = self.tip_of_body.get(int(self.m.geom_bodyid[g]))
                if tip is not None and other in self.obj_geoms:
                    mujoco.mj_contactForce(self.m, d, i, f6)
                    out[tip] += sign * (c.frame.reshape(3, 3).T @ f6[:3])          # world frame
        return out
