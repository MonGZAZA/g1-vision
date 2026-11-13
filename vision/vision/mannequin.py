"""Simulated people for the exp05 sensor sim: articulated mannequins (MuJoCo) with a drawn face.

Each person is a body with a free joint (root at the floor, facing its own +x) and hinge joints for hips, knees,
shoulders and elbows, posed kinematically (qpos set directly, no physics). The head carries a cube texture: a
procedurally drawn face on the front, hair elsewhere, so a face detector finds a face only when the person looks
toward the camera. Proportions are those of a ~1.70 m adult (scale for others).
"""

import math

import cv2
import mujoco
import numpy as np

SHIRTS = [(0.15, 0.35, 0.70), (0.70, 0.20, 0.20), (0.20, 0.55, 0.30), (0.85, 0.75, 0.25), (0.45, 0.30, 0.60),
          (0.95, 0.95, 0.95), (0.20, 0.20, 0.22), (0.90, 0.50, 0.20)]  # fmt: skip
TROUSERS = [(0.12, 0.14, 0.25), (0.25, 0.22, 0.20), (0.35, 0.35, 0.38), (0.10, 0.10, 0.10)]
SKINS = [(0.87, 0.70, 0.58), (0.76, 0.57, 0.45), (0.60, 0.43, 0.32), (0.93, 0.78, 0.67)]
HAIR = (0.12, 0.10, 0.10)
JOINTS = ["hip_l", "hip_r", "knee_l", "knee_r", "sho_l", "sho_r", "elb_l", "elb_r"]


def face_image(size=128, skin=(0.87, 0.70, 0.58), hair=HAIR):
    """RGB face for the front of the head cube (row 0 = top)."""
    s = size
    sk = tuple(int(255 * c) for c in skin)
    hr = tuple(int(255 * c) for c in hair)
    im = np.full((s, s, 3), sk, np.uint8)
    cv2.ellipse(im, (s // 2, int(s * 0.04)), (int(s * 0.55), int(s * 0.22)), 0, 0, 360, hr, -1)
    for x in (0.31, 0.69):
        cv2.ellipse(im, (int(s * x), int(s * 0.32)), (int(s * 0.11), int(s * 0.03)), 0, 180, 360, (40, 35, 35), 3)
        cv2.ellipse(im, (int(s * x), int(s * 0.42)), (int(s * 0.08), int(s * 0.045)), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(im, (int(s * x), int(s * 0.42)), int(s * 0.035), (45, 35, 30), -1)
    dark = tuple(int(0.7 * c) for c in sk)
    cv2.line(im, (s // 2, int(s * 0.45)), (int(s * 0.45), int(s * 0.63)), dark, 3)
    cv2.line(im, (int(s * 0.45), int(s * 0.63)), (int(s * 0.56), int(s * 0.64)), dark, 3)
    cv2.ellipse(im, (s // 2, int(s * 0.77)), (int(s * 0.15), int(s * 0.05)), 0, 0, 180, (150, 60, 70), 4)
    return cv2.GaussianBlur(im, (3, 3), 0)




def head_texture(skin, size=128):
    """Cube texture, faces stacked R L U D F B = +x -x +y -y +z -z of the head: face on +x (the person's front),
    hair on the back and top, hair over skin on the sides, skin below."""
    hr = np.full((size, size, 3), [int(255 * c) for c in HAIR], np.uint8)
    sk = np.full((size, size, 3), [int(255 * c) for c in skin], np.uint8)
    side = hr.copy()
    side[int(size * 0.45):] = sk[int(size * 0.45):]  # ears / cheeks on the lower half
    # rotations found by rendering the head from each side (tools: sensor_check renders)
    return np.vstack([np.rot90(face_image(size, skin), 1), hr, np.rot90(side, 2), side, hr, sk]).copy()


def add_person(spec, k, height=1.70, seed=None):
    """Add mannequin k (name prefix f'p{k}_') to an MjSpec; returns the root body."""
    rng = np.random.default_rng(k if seed is None else seed)
    shirt, trousers = SHIRTS[k % len(SHIRTS)], TROUSERS[k % len(TROUSERS)]
    skin = SKINS[int(rng.integers(len(SKINS)))]
    s = height / 1.70
    tex = spec.add_texture(name=f"p{k}_headtex", type=mujoco.mjtTexture.mjTEXTURE_CUBE)
    img = head_texture(skin)
    tex.width, tex.height, tex.nchannel = img.shape[1], img.shape[0], 3
    tex.data = img.tobytes()
    mat = spec.add_material(name=f"p{k}_headmat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"p{k}_headtex"

    def geom(body, **kw):
        g = body.add_geom(contype=0, conaffinity=0, group=1, **kw)
        return g

    root = spec.worldbody.add_body(name=f"p{k}_root", pos=[50 + 3 * k, 50, 0])
    root.add_freejoint(name=f"p{k}_free")
    # torso + pelvis + neck + head
    geom(root, type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.11 * s, 0.17 * s, 0.24 * s], pos=[0, 0, 1.20 * s],
         rgba=[*shirt, 1])
    geom(root, type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.10 * s, 0.16 * s, 0.10 * s], pos=[0, 0, 0.95 * s],
         rgba=[*trousers, 1])
    geom(root, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.045 * s, 0.05 * s], pos=[0, 0, 1.47 * s],
         rgba=[*skin, 1])
    head = root.add_body(name=f"p{k}_head", pos=[0, 0, 1.61 * s])
    geom(head, type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.095 * s, 0.078 * s, 0.11 * s], material=f"p{k}_headmat")
    for side, sy in (("l", 1), ("r", -1)):
        hip = root.add_body(name=f"p{k}_thigh_{side}", pos=[0, sy * 0.085 * s, 0.92 * s])
        hip.add_joint(name=f"p{k}_hip_{side}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        geom(hip, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -0.42 * s], size=[0.065 * s, 0, 0],
             rgba=[*trousers, 1])
        shin = hip.add_body(name=f"p{k}_shin_{side}", pos=[0, 0, -0.42 * s])
        shin.add_joint(name=f"p{k}_knee_{side}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        geom(shin, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -0.42 * s], size=[0.05 * s, 0, 0],
             rgba=[*trousers, 1])
        geom(shin, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.12 * s, 0.045 * s, 0.035 * s],
             pos=[0.05 * s, 0, -0.46 * s], rgba=[0.1, 0.1, 0.1, 1])
        arm = root.add_body(name=f"p{k}_upper_{side}", pos=[0, sy * 0.20 * s, 1.40 * s])
        arm.add_joint(name=f"p{k}_sho_{side}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        geom(arm, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, sy * 0.02 * s, -0.28 * s],
             size=[0.045 * s, 0, 0], rgba=[*shirt, 1])
        fore = arm.add_body(name=f"p{k}_fore_{side}", pos=[0, sy * 0.02 * s, -0.28 * s])
        fore.add_joint(name=f"p{k}_elb_{side}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        geom(fore, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -0.25 * s], size=[0.038 * s, 0, 0],
             rgba=[*skin, 1])
        geom(fore, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.045 * s, 0, 0], pos=[0, 0, -0.29 * s],
             rgba=[*skin, 1])
    return root


class PersonPose:
    """qpos addresses of one mannequin; set(x, y, yaw, phase, moving) poses it (walk cycle when moving)."""

    def __init__(self, m, k):
        self.free = m.jnt_qposadr[m.joint(f"p{k}_free").id]
        self.j = {n: m.jnt_qposadr[m.joint(f"p{k}_{n}").id] for n in JOINTS}
        self.body = m.body(f"p{k}_root").id

    def set(self, qpos, x, y, yaw, phase=0.0, amp=0.0, visible=True):
        qpos[self.free:self.free + 3] = [x, y, 0.0] if visible else [x + 100.0, y + 100.0, -50.0]
        qpos[self.free + 3:self.free + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        sw = amp * math.sin(phase)
        qpos[self.j["hip_l"]], qpos[self.j["hip_r"]] = -0.45 * sw, 0.45 * sw
        qpos[self.j["knee_l"]] = 0.6 * amp * max(0.0, math.sin(phase + 1.2))
        qpos[self.j["knee_r"]] = 0.6 * amp * max(0.0, -math.sin(phase + 1.2))
        qpos[self.j["sho_l"]], qpos[self.j["sho_r"]] = 0.35 * sw, -0.35 * sw
        qpos[self.j["elb_l"]] = qpos[self.j["elb_r"]] = -0.25 - 0.15 * amp
