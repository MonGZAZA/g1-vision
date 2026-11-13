"""Robot geometry for the vision module: where the head lidar and the camera are, from the measured joint angles.

The robot frame is the one used everywhere in exp05 (pointing, shop.json): origin under the pelvis on the floor,
x forward (into the shop), y to the robot's left, z up. The legs stay in Unitree's standing posture (exp05 never
moves them), so the pelvis height comes from the model standing on its feet. The head (lidar and camera) is on
torso_link, which turns with the waist yaw; the arms move through the lidar's view, so the robot's own body is
removed from the point cloud with the posed model (self_mask).

Joint angles come from the exp05 server (GET /api/status: arm.q = waist yaw + 2 x 7 arm joints, imu_quat = pelvis
attitude); without them the rest pose is used.
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

EXP05 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP05 / "g1/deploy"))
from robot_model import ARM_REST, MODEL, MOTORS, STAND_HEIGHT, STAND_LEGS, UPPER, lowest_point  # noqa: E402

# MID-360 on the head: mounted upside down (roll 180°), 0.45 m above torso_link. Same as exp04's
# head_sensors.LIDAR_MOUNT, checked on our G1 on 2026-10-30 (lidar 1.29-1.31 m above the floor, within 2-4°).
LIDAR_MOUNT = {"parent": "torso_link", "pos": [0.0, 0.0, 0.45], "rpy_deg": [180.0, 0.0, 0.0]}

# Camera mounts in torso_link. "head_d435" = Unitree's URDF d435_joint (g1_29dof_lock_waist_rev_1_0): pitched
# 47.6° down (made for manipulation: it sees the floor and the hands, a standing person only from the hips down
# within ~1.5 m). "level" = the same place looking straight ahead (a re-aimed or extra camera on the head): sees
# faces from ~1.2 m. Optical axis = +x of the mount; image x to the right (-y), image y down (-z).
CAMERA_MOUNTS = {
    "head_d435": {"parent": "torso_link", "pos": [0.0576, 0.0175, 0.4299], "rpy_deg": [0.0, 47.6, 0.0]},
    "level": {"parent": "torso_link", "pos": [0.0576, 0.0, 0.4299], "rpy_deg": [0.0, 0.0, 0.0]},
    "tilt10": {"parent": "torso_link", "pos": [0.0576, 0.0, 0.4299], "rpy_deg": [0.0, 10.0, 0.0]},
    # laptop webcam on the desk, no robot: camera at 1.0 m looking along +x (distance from the image only)
    "laptop": {"parent": None, "pos": [0.0, 0.0, 1.0], "rpy_deg": [0.0, 0.0, 0.0]},
}


def rpy_matrix(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])  # fmt: skip


def parse_mount(text, table):
    """A name from table, or 'x,y,z,roll,pitch,yaw' (m, deg) on torso_link."""
    if text in table:
        return dict(table[text])
    v = [float(x) for x in text.split(",")]
    if len(v) != 6:
        raise ValueError(f"mount must be one of {sorted(table)} or x,y,z,roll,pitch,yaw")
    return {"parent": "torso_link", "pos": v[:3], "rpy_deg": v[3:]}


def quat_level(q):
    """Pelvis attitude quaternion (w, x, y, z) with its yaw removed: roll/pitch only (the robot frame has no yaw)."""
    w, x, y, z = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(-yaw / 2), math.sin(-yaw / 2)  # q_level = q_z(-yaw) * q
    out = np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])
    return out / np.linalg.norm(out)


class RobotGeometry:
    """The G1 model standing at the robot-frame origin, posed from the measured upper-body joints."""

    def __init__(self, model_path=MODEL):
        self.m = m = mujoco.MjModel.from_xml_path(str(model_path))
        self.d = d = mujoco.MjData(m)
        names = [m.joint(j).name for j in range(m.njnt)]
        self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in MOTORS if n in names}
        for n, a in self.qadr.items():
            for k, v in STAND_LEGS.items():
                if k in n:
                    d.qpos[a] = v
            d.qpos[a] = ARM_REST.get(n, d.qpos[a])
        d.qpos[2], d.qpos[3] = STAND_HEIGHT, 1.0
        mujoco.mj_kinematics(m, d)
        d.qpos[2] -= lowest_point(m, d)  # feet on the floor (z = 0), as in sim_robot.py
        mujoco.mj_kinematics(m, d)
        self.pelvis_z = float(d.qpos[2])
        self.root = m.body("pelvis").id if "pelvis" in [m.body(b).name for b in range(m.nbody)] else 1
        # robot geoms for the self mask: every geom of a robot body, as a box in its own frame (+ margin)
        self.geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
        self.aabb = m.geom_aabb[self.geoms].copy()  # (G, 6): centre (3), half size (3) in the geom frame
        self.q_upper = None

    def update(self, q_upper=None, imu_quat=None):
        """Pose the model: q_upper = 15 values in UPPER order (or None = rest), imu_quat = pelvis (w,x,y,z)."""
        d = self.d
        if q_upper is not None and len(q_upper) == len(UPPER):
            for n, v in zip(UPPER, q_upper):
                if n in self.qadr:
                    d.qpos[self.qadr[n]] = v
            self.q_upper = list(q_upper)
        if imu_quat is not None:
            d.qpos[3:7] = quat_level(imu_quat)
        mujoco.mj_kinematics(self.m, self.d)

    def body_pose(self, name):
        b = self.m.body(name).id
        return self.d.xmat[b].reshape(3, 3).copy(), self.d.xpos[b].copy()

    def mount_pose(self, mount):
        """(R, p) of a sensor mount in the robot frame."""
        Rm, pm = rpy_matrix(mount["rpy_deg"]), np.array(mount["pos"], float)
        if not mount.get("parent"):
            return Rm, pm
        Rb, pb = self.body_pose(mount["parent"])
        return Rb @ Rm, pb + Rb @ pm

    def self_mask(self, pts, margin=0.04):
        """True for points ON the robot (inside a posed geom box grown by margin)."""
        on = np.zeros(len(pts), bool)
        if not len(pts):
            return on
        near = np.linalg.norm(pts[:, :2], axis=1) < 1.3  # nothing of the robot reaches further from its centre
        idx = np.flatnonzero(near)
        if not len(idx):
            return on
        p = pts[idx]
        hit = np.zeros(len(idx), bool)
        for k, g in enumerate(self.geoms):
            R, c = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g]
            local = (p - c) @ R - self.aabb[k, :3]
            hit |= np.all(np.abs(local) <= self.aabb[k, 3:] + margin, axis=1)
        on[idx] = hit
        return on
