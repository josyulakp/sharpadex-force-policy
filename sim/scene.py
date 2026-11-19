"""MuJoCo scene for clean_plate_with_eraser: the D01 robot on a fixed base, a table, a plate and an eraser.

Objects are simple shapes (the plate is a foot disc plus a wider rim disc, so fingers can reach under the
rim as in the videos). All sizes, masses and friction values live in configs/sim.yaml.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .robot import SCENE_XML

OBJ_BIT = 1 << 13                      # collision bit of table / plate / eraser (robot geoms use bits 0-12)
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


HAND_BASE_IN_LINK7 = {"left": ([0.055, 0, 0], [0, -1.5708, 3.1416]), "right": ([0.055, 0, 0], [0, 1.5708, 0])}   # URDF, rpy


def build(cfg: dict, torso=None, plate_pose=None, eraser_pose=None, table_z=None, plate_radius=None,
          plate_height=None) -> mujoco.MjModel:
    """torso: the five lower-body joint angles of the recording (locked). plate_pose / eraser_pose: (xyz, yaw)
    of the object on the table in the world frame. table_z / plate_radius: values measured for the episode."""
    s = mujoco.MjSpec.from_file(str(SCENE_XML))
    s.option.timestep = cfg["timestep"]
    s.option.impratio = cfg["impratio"]
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.delete(s.joint("north_freejoint"))                                  # fixed base
    w, t, p, e = s.worldbody, dict(cfg["table"]), dict(cfg["plate"]), cfg["eraser"]
    if table_z is not None:
        t["z"] = table_z
    if plate_radius is not None:
        p["radius"] = plate_radius
    if plate_height is not None:                                          # rim top above the table, measured
        p["foot_height"] = plate_height - p["rim_thickness"]
    for q in s.equalities:                                                # the model ships with every joint braked at zero;
        if "brake" in q.name and "chassis" not in q.name:                 # Sharpa's app releases them at start-up
            q.active = False
    lock = dict(zip(cfg["locked_joints"], list(torso if torso is not None else np.zeros(5)) + [0.0] * 8))
    for j, value in lock.items():                                         # torso at the recorded pose, neck at zero
        q = s.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=j, objtype=mujoco.mjtObj.mjOBJ_JOINT)
        q.data = [float(value), 0, 0, 0, 0] + [0.0] * 6
    for side, (xyz, rpy) in HAND_BASE_IN_LINK7.items():                   # hand bases follow a commanded pose
        quat = np.roll(R.from_euler("xyz", rpy).as_quat(), 1)
        s.body(f"{side}_arm_link_7").add_body(name=f"{side}_hand_base", pos=xyz, quat=quat)
        w.add_body(name=f"{side}_target", mocap=True)
        q = s.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=f"{side}_target", name2=f"{side}_hand_base",
                           objtype=mujoco.mjtObj.mjOBJ_BODY, solref=cfg["weld_solref"])
        q.data = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    col = dict(contype=OBJ_BIT, conaffinity=OBJ_BIT | (OBJ_BIT - 1), condim=4)
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t["half_x"], t["half_y"], 0.02],
               pos=[t["x"], t["y"], t["z"] - 0.02], rgba=[0.1, 0.12, 0.2, 1], friction=[t["friction"], 0.01, 0.001], **col)
    if plate_pose is not None:
        b = w.add_body(name="plate", pos=[*plate_pose[0][:2], t["z"] + 1e-3], quat=_yaw(plate_pose[1]))
        b.add_freejoint(name="plate_free")
        f = [p["friction"], 0.01, 0.001]
        # a dish on a foot: stacked discs widening from the foot to the rim, so fingers meet a sloping underside
        n, height = p["steps"], p["foot_height"] + p["rim_thickness"]
        for i in range(n):
            r = p["foot_radius"] + (p["radius"] - p["foot_radius"]) * (i / (n - 1)) ** p["flare"]
            b.add_geom(name=f"plate_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, height / n / 2, 0],
                       pos=[0, 0, height * (i + 0.5) / n], mass=p["mass"] / n, rgba=[0.95, 0.95, 0.92, 1], friction=f, **col)
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
    w.add_camera(name="view", pos=cam["pos"], mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="lower_body_link_5")
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
