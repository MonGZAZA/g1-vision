"""Shared facts about exp05's G1 model and DDS interface (used by the sim robot, the server and the motion code)."""

import math
import os
from pathlib import Path

import mujoco
import numpy as np

G1 = Path(__file__).resolve().parents[1]
EXP05 = G1.parent
# Fitted hands: "dex3" (Dex3-1, 3 fingers: the robot now) or "inspire" (RH56, 5 fingers: planned). One setting for the
# sim, the server and the motion code (the wai and the gestures are solved on the matching model); run_sim.sh /
# run_robot.sh export it from HANDS=... "none" (no hands driven) uses the Dex3 model.
HANDS = os.environ.get("EXP05_HANDS", "dex3")
if HANDS not in ("dex3", "inspire", "none"):
    raise SystemExit(f"EXP05_HANDS must be dex3, inspire or none (got {HANDS!r})")
HAND_MODEL = "inspire" if HANDS == "inspire" else "dex3"
MODEL = G1 / f"model/g1_{HAND_MODEL}.xml"  # built by tools/build_model.py --hands ...
WAI_CLIP = EXP05 / f"clips/wai_{HAND_MODEL}.json"  # learned wai (tools/make_wai_clip.py), per hand geometry
POSE_CACHE = EXP05 / f"cache/poses_{HAND_MODEL}.json"  # gesture IK cache (gestures.py)
HAND_BODY_PREFIX = {"inspire": {"left": "L_", "right": "R_"},
                    "dex3": {"left": "left_hand_", "right": "right_hand_"}}[HAND_MODEL]

# Unitree DDS motor order (rt/lowstate, rt/lowcmd, rt/arm_sdk)
MOTORS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip
NUM_MOTOR = 29
LOCKED = ("waist_roll_joint", "waist_pitch_joint")  # our robot: g1_29dof_lock_waist
WEIGHT_SLOT = 29  # rt/arm_sdk: motor_cmd[29].q = weight 0..1 (Unitree's example: "kNotUsedJoint")
ARM_SDK_MOTORS = [i for i in range(12, 29) if MOTORS[i] not in LOCKED]  # waist yaw + 2 x 7 arm joints
UPPER = [MOTORS[i] for i in ARM_SDK_MOTORS]  # 15 joint names exp05 drives

STAND_LEGS = {"hip_pitch": -0.1, "knee": 0.3, "ankle_pitch": -0.2}
STAND_HEIGHT = 0.793  # pelvis height before the feet are put on the floor (the sim adjusts it)
# arms hanging, elbows a little bent; stands in for Unitree's own posture (same as exp02/exp04's default pose).
# Elbow: 0 = right angle (forearm forward), positive straightens (straight ~1.45), negative raises the forearm.
# Dex3: shoulder roll 0.18 instead of 0.12, else the Dex3 thumbs touch the hips (hip_roll_link); with the thumb
# folded ("relaxed" below) the hands then clear the hips by ~2 cm.
_ROLL = 0.18 if HAND_MODEL == "dex3" else 0.12
ARM_REST = {"left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": _ROLL, "right_shoulder_roll_joint": -_ROLL,
            "left_elbow_joint": 0.9, "right_elbow_joint": 0.9}  # fmt: skip

# Inspire RH56: per hand 6 DoF in this order; value 1 = open, 0 = closed (Unitree's Inspire DDS service)
INSPIRE_DOFS = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]

# Dex3-1 (3 fingers, 7 motors per hand: thumb 0-2, middle 0-1, index 0-1) on rt/dex3/{left,right}/cmd|state.
# The motion layer always speaks Inspire values (1 = open); they map to Dex3 as closure fractions of each joint's
# range end: thumb bend -> thumb 1/2, index -> index 0/1, mean(middle, ring, pinky) -> middle 0/1, thumb rotation 0.
# Ranges = unitree_rl_gym g1_29dof_with_hand(_rev_1_0) (checked against model/g1_dex3.xml in tools/sim_check.py).
DEX3_MOTORS = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
DEX3_FAR = {"left": np.array([0.0, 1.0472, 1.74533, -1.5708, -1.74533, -1.5708, -1.74533]),
            "right": np.array([0.0, -1.0472, -1.74533, 1.5708, 1.74533, 1.5708, 1.74533])}


def inspire_to_dex3(v6, side):
    """Inspire 6 values (pinky, ring, middle, index, thumb bend, thumb rotation; 1 = open) -> 7 Dex3 targets (rad)."""
    c = 1.0 - np.clip(np.asarray(v6, float), 0, 1)
    frac = np.array([0.0, c[4], c[4], np.mean(c[0:3]), np.mean(c[0:3]), c[3], c[3]])
    return frac * DEX3_FAR[side]


def dex3_to_inspire(q7, side):
    far = DEX3_FAR[side]
    f = np.clip(np.asarray(q7, float)[1:] / far[1:], 0, 1)
    mid, idx, th = (f[2] + f[3]) / 2, (f[4] + f[5]) / 2, (f[0] + f[1]) / 2
    return 1.0 - np.array([mid, mid, mid, idx, th, 0.0])


# Hand poses (Inspire values per hand, 1 = open). "flat" = fingers straight for the wai and the open-palm gestures:
# the Inspire thumb lies along the index. The Dex3 thumb FOLDS across the palm (thumb bend 0) in "flat" and "relaxed":
# an open Dex3 thumb sticks out 8.8 cm on the palm side, so the two thumbs would hit before the palms meet in the wai
# (folded: 2.9 cm) and the hanging hands would touch the hips. Folded, the thumb stays 5 mm clear of the fingers.
_DEX3 = HAND_MODEL == "dex3"
HAND_POSES = {
    "open": [1, 1, 1, 1, 1, 1],
    "flat": [1, 1, 1, 1, 0.0, 0.0] if _DEX3 else [1, 1, 1, 1, 1, 0.3],
    "relaxed": [0.75, 0.75, 0.8, 0.85, 0.0, 0.0] if _DEX3 else [0.75, 0.75, 0.8, 0.85, 0.8, 0.5],
    "closed": [0, 0, 0, 0, 0.3, 0.3],
}
assert all(len(v) == len(INSPIRE_DOFS) for v in HAND_POSES.values())


def set_hand_qpos(m, d, pose="flat"):
    """Finger joints of model m (data d) to a HAND_POSES pose (kinematic models: IK, gap checks)."""
    v6 = HAND_POSES[pose]
    for side in ("left", "right"):
        if HAND_MODEL == "dex3":
            for k, v in zip(DEX3_MOTORS, inspire_to_dex3(v6, side)):
                d.qpos[m.jnt_qposadr[m.joint(f"{side}_hand_{k}_joint").id]] = v
        else:
            for dof, v in zip(INSPIRE_DOFS, v6):
                j = m.actuator(f"{side}_hand_{dof}").trnid[0]
                d.qpos[m.jnt_qposadr[j]] = inspire_to_joint(v, m.jnt_range[j])


def inspire_to_joint(v, rng):
    """Inspire value (1 open .. 0 closed) -> model joint angle (open = lower limit)."""
    lo, hi = rng
    return lo + (1.0 - float(v)) * (hi - lo)


def joint_to_inspire(q, rng):
    lo, hi = rng
    return float(np.clip(1.0 - (q - lo) / (hi - lo), 0.0, 1.0))


def lowest_point(m, d, body_suffix="ankle_roll_link"):
    """Lowest z of the collision geoms of bodies whose name ends with body_suffix (mesh vertices included)."""
    z = math.inf
    for g in range(m.ngeom):
        if not m.body(m.geom_bodyid[g]).name.endswith(body_suffix) or not m.geom_contype[g]:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            z = min(z, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
        else:
            z = min(z, float(d.geom_xpos[g][2] - m.geom_rbound[g]))
    return z


def world_to_robot(shop, x, y):
    """Shop frame (m) -> robot frame (robot at origin facing +x, y to its left)."""
    r = shop["robot"]
    yaw = math.radians(r["yaw_deg"])
    dx, dy = x - r["x"], y - r["y"]
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)


def shop_booths_robot_frame(shop):
    """Booths as robot-frame boxes: centre, size, yaw, aim point, colour."""
    out = []
    ryaw = math.radians(shop["robot"]["yaw_deg"])
    for b in shop.get("booths", []):
        rc = b["rect"]
        a = math.radians(rc.get("angle_deg", 0.0))
        cx = rc["x"] + (rc["w"] / 2) * math.cos(a) - (rc["h"] / 2) * math.sin(a)
        cy = rc["y"] + (rc["w"] / 2) * math.sin(a) + (rc["h"] / 2) * math.cos(a)
        px, py = world_to_robot(shop, cx, cy)
        ax, ay = world_to_robot(shop, b["aim"]["x"], b["aim"]["y"])
        col = b.get("color", "#888888").lstrip("#")
        rgba = [int(col[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0]
        out.append({"id": b["id"], "cx": px, "cy": py, "w": rc["w"], "h": rc["h"], "yaw": a - ryaw,
                    "aim": [ax, ay, b["aim"].get("z", 1.0)], "rgba": rgba})
    return out
