"""Build exp05's MuJoCo models of our G1: 29-DoF body with the waist roll/pitch LOCKED, with either hand:
  --hands dex3     Dex3-1 3-finger hands (what the robot has now)  -> model/g1_dex3.xml
  --hands inspire  Inspire RH56DFQ 5-finger hands (planned)          -> model/g1_inspire.xml

Sources (see third_party/THIRD_PARTY.md): unitree_g1_dex3/g1_29dof_with_hand_rev_1_0.urdf (unitree_rl_gym) and
unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf.
The URDF is edited as text before MuJoCo imports it:
  - a floating base (world -> pelvis), so the robot can stand (sim mode B) or be pinned (mode A, a weld the sim
    switches on);
  - waist_roll_joint and waist_pitch_joint become fixed (our robot: g1_29dof_lock_waist, found 2026-10-26);
  - finger mimic joints become MuJoCo joint equalities (done by MuJoCo's URDF importer).
Then added with MjSpec: Unitree motor armature/friction on the body joints, one PD actuator per body motor (the
sim sets kp/kd per command, like the real motor driver: implicit PD through affine bias, integrator implicitfast),
hand actuators (Inspire: 6 per hand, pinky, ring, middle, index, thumb bend, thumb rotation; Dex3: 7 per hand,
one per motor), sites for the palms, shoulders and head, a floor, lights and the pelvis pin.

The palm site ("<side>_palm") has the SAME pose on the wrist yaw link for both hands (the Inspire hand base frame:
41.5 mm past the wrist yaw joint; fingers along local -y, palm normal along local -x), so the IK of pointing,
gestures and the wai gives the same arm angles; only the finger geometry differs.

Run from exp05/g1:
    .venv/bin/python tools/build_model.py --hands dex3      (and --hands inspire)
"""

import argparse
import math
import re
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"inspire": ROOT / "third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf",
           "dex3": ROOT / "third_party/unitree_g1_dex3/g1_29dof_with_hand_rev_1_0.urdf"}
LOCKED = ("waist_roll_joint", "waist_pitch_joint")

# Unitree DDS motor order (29 slots; the two locked waist slots stay in the list, they report q = 0)
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
# Inspire order per hand (Unitree's Inspire service: right 0-5, left 6-11; value 1 = open, 0 = closed)
INSPIRE = [("pinky", "pinky_proximal_joint"), ("ring", "ring_proximal_joint"), ("middle", "middle_proximal_joint"),
           ("index", "index_proximal_joint"), ("thumb_bend", "thumb_proximal_pitch_joint"),
           ("thumb_rotation", "thumb_proximal_yaw_joint")]  # fmt: skip
# Dex3-1 motor order per hand (rt/dex3/<side>/cmd motor_cmd[0..6])
DEX3 = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
# palm site pose on <side>_wrist_yaw_link = the Inspire URDF's <L|R>_base_link_joint (pos, rpy)
PALM_SITE = {"left": ([0.0415, 0, 0], [0, 0, math.pi / 2]), "right": ([0.0415, 0, 0], [math.pi, 0, -math.pi / 2])}


def hand_bodies(spec, side, hands):
    """Names of the bodies of one hand (palm/base + fingers)."""
    pre = {"inspire": {"left": "L_", "right": "R_"}, "dex3": {"left": "left_hand_", "right": "right_hand_"}}[hands][side]
    return [b.name for b in spec.bodies if b.name.startswith(pre)]


def edited_urdf(SRC):
    text = SRC.read_text()
    for j in LOCKED:
        text, n = re.subn(rf'(<joint\s+name="{j}"\s+type=)"revolute"', r'\1"fixed"', text)
        assert n == 1, j
    text = text.replace('discardvisual="false"', 'discardvisual="false" fusestatic="false"', 1)  # keep hand/head links
    float_base = ('<link name="world"/>\n  <joint name="floating_base_joint" type="floating">'
                  '<parent link="world"/><child link="pelvis"/></joint>\n  ')
    text = text.replace('<link name="pelvis">', float_base + '<link name="pelvis">', 1)
    efforts = {m.group(1): float(m.group(2)) for m in re.finditer(
        r'<joint\s+name="([^"]+)"\s+type="revolute">.*?<limit[^>]*effort="([0-9.]+)"', text, re.S)}
    return text, efforts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", choices=["dex3", "inspire"], required=True)
    hands = ap.parse_args().hands
    SRC, OUT = SOURCES[hands], ROOT / f"model/g1_{hands}.xml"
    text, efforts = edited_urdf(SRC)
    tmp = SRC.parent / "_exp05_tmp.urdf"  # next to the meshes so the relative mesh paths resolve
    tmp.write_text(text)
    try:
        spec = mujoco.MjSpec.from_file(str(tmp))
        spec.meshdir = ""
        spec.compile()  # check
    finally:
        tmp.unlink()

    spec.modelname = f"g1_29dof_lock_waist_{hands}"
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 0.002
    mesh_rel = Path("..") / SRC.parent.relative_to(ROOT)
    spec.meshdir = str(SRC.parent)  # absolute while compiling here; relative in the written file

    joints = {j.name: j for j in spec.joints}
    for name in MOTORS:
        if name in LOCKED:
            continue
        j = joints[name]
        j.armature, j.frictionloss = 0.01, 0.2
        j.damping[0] = 0.05  # Unitree motor values (exp02 finding)
        a = spec.add_actuator(name=name.replace("_joint", ""), target=name)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = 1.0  # force = ctrl - kp q - kd dq; kp/kd (bias) are set by the sim per command
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        e = efforts.get(name, 25.0)
        a.forcerange = [-e, e]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE  # ctrl = kp q* + kd dq* + tau_ff, unbounded
    hand_acts = [(f"{s_}_hand_{dof}", ("R_" if s_ == "right" else "L_") + jn) for s_ in ("right", "left") for dof, jn in INSPIRE]
    if hands == "dex3":
        hand_acts = [(f"{s_}_hand_{k}", f"{s_}_hand_{k}_joint") for s_ in ("right", "left") for k in DEX3]
    for aname, jn in hand_acts:
        j = joints[jn]
        j.armature = 0.001
        j.damping[0] = 0.02
        a = spec.add_actuator(name=aname, target=jn)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0], a.biasprm[1], a.biasprm[2] = 2.0, -2.0, -0.05
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-2.0, 2.0]

    bodies = {b.name: b for b in spec.bodies}
    # fingers touch their own palm when closing; skip those contacts (the real hand has no such collision issue)
    for side in ("left", "right"):
        hb = hand_bodies(spec, side, hands)
        for i, a in enumerate(hb):
            for b in hb[i + 1:]:
                spec.add_exclude(bodyname1=a, bodyname2=b)
    # palm site: same pose on the wrist yaw link for both hands (see the docstring); checked in tests
    for side in ("left", "right"):
        bodies[f"{side}_shoulder_pitch_link"].add_site(name=f"{side}_shoulder", size=[0.015] * 3, rgba=[0, 0.6, 1, 0.5])
        pos, rpy = PALM_SITE[side]
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array(rpy, float), "XYZ")
        bodies[f"{side}_wrist_yaw_link"].add_site(name=f"{side}_palm", pos=pos, quat=quat, size=[0.012] * 3,
                                                  rgba=[1, 0.3, 0, 0.5])
    bodies["torso_link"].add_site(name="head", pos=[0.06, 0, 0.45], size=[0.015] * 3, rgba=[0, 1, 0, 0.5])
    bodies["torso_link"].add_site(name="imu_torso", size=[0.01] * 3)
    bodies["pelvis"].add_site(name="imu_pelvis", size=[0.01] * 3)

    # pelvis pin for sim mode A (active in the sim when pinned; relpose taken at activation)
    eq = spec.add_equality(name="pin_pelvis", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1="pelvis", name2="world")
    eq.active = False

    for typ, name, obj in ((mujoco.mjtSensor.mjSENS_FRAMEQUAT, "imu_quat", "imu_pelvis"),
                           (mujoco.mjtSensor.mjSENS_GYRO, "imu_gyro", "imu_pelvis"),
                           (mujoco.mjtSensor.mjSENS_ACCELEROMETER, "imu_acc", "imu_pelvis")):
        sen = spec.add_sensor(name=name, type=typ, objname=obj)
        sen.objtype = mujoco.mjtObj.mjOBJ_SITE

    wb = spec.worldbody
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.85, 0.88, 0.92], rgb2=[0.55, 0.6, 0.68], width=512, height=3072)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.82, 0.82, 0.8], rgb2=[0.74, 0.74, 0.72], width=512, height=512)
    mat = spec.add_material(name="floor")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [1, 1]
    mat.texuniform = True
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor")
    wb.add_light(name="sun", pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    wb.add_light(name="fill", pos=[2, -2, 3], dir=[-0.5, 0.5, -1], diffuse=[0.3, 0.3, 0.3])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720

    pelvis = bodies["pelvis"]
    pelvis.pos = [0, 0, 0.793]
    m = spec.compile()
    OUT.parent.mkdir(exist_ok=True)
    xml = spec.to_xml().replace(f'meshdir="{SRC.parent}', f'meshdir="{mesh_rel}', 1)
    assert str(SRC.parent) not in xml
    OUT.write_text(xml)
    m2 = mujoco.MjModel.from_xml_path(str(OUT))  # reload check
    print(f"wrote {OUT.relative_to(ROOT)}: nq {m2.nq}, nu {m2.nu}, neq {m2.neq}, mass {m2.body_subtreemass[1]:.1f} kg")
    assert m2.nu == 27 + (14 if hands == "dex3" else 12), m2.nu
    print("locked:", [n for n in LOCKED], "-> fixed; actuators:", [m2.actuator(i).name for i in range(m2.nu)][:3], "...")
    _ = np  # noqa


if __name__ == "__main__":
    main()
