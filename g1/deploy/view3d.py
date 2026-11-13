"""3D preview for the operator page: the robot in its measured pose (sim or real), the booths of the active shop
around it, and optionally the line from the shoulder to a booth's aim point. Rendered offscreen (EGL) as JPEG by one
render thread (an EGL context belongs to one thread); requests wait for their frame.

Cameras: "behind" (over the robot's shoulder, looking into the shop: what a customer next to it sees),
"front" (facing the robot), "top" (bird's eye of the shop around the robot).
"""

import io
import math
import queue
import threading

import mujoco
import numpy as np
from PIL import Image

import sim_robot  # build(shop): the robot model + booth boxes
from pointing import CENTER_ARM, CENTER_DEG, MAX_AZIMUTH_DEG
from robot_model import (DEX3_MOTORS, HAND_MODEL, INSPIRE_DOFS, MOTORS, inspire_to_dex3, inspire_to_joint, lowest_point,
                         shop_booths_robot_frame)

CAMERAS = {  # lookat, distance, azimuth, elevation
    "behind": ([1.5, 0.0, 1.0], 4.2, 0.0, -14.0),  # azimuth = viewing direction: 0 looks along +x (into the shop)
    "front": ([0.0, 0.0, 0.95], 2.6, 180.0, -8.0),
    "top": ([9.0, 0.0, 0.0], 30.0, 0.0, -62.0),
}
SIZE = (480, 640)  # h, w


class View3D:
    def __init__(self):
        self.q = queue.Queue()
        self.shop = None
        self._new_shop = None
        threading.Thread(target=self._run, name="view3d", daemon=True).start()

    def set_shop(self, shop):
        self._new_shop = shop

    def render(self, q29, hands12, cam="behind", booth=None, timeout=3.0):
        """JPEG bytes, or None if the render thread is busy for longer than timeout."""
        out = queue.Queue(maxsize=1)
        self.q.put((q29, hands12, cam, booth, out))
        try:
            return out.get(timeout=timeout)
        except queue.Empty:
            return None

    def _build(self, shop):
        self.m = m = sim_robot.build(shop)
        self.d = mujoco.MjData(m)
        self.r = mujoco.Renderer(m, *SIZE)
        self.qadr = {i: m.jnt_qposadr[m.joint(n).id] for i, n in enumerate(MOTORS) if n in
                     [m.joint(j).name for j in range(m.njnt)]}
        self.hand = []
        self.dex3_adr = {s: [m.jnt_qposadr[m.joint(f"{s}_hand_{k}_joint").id] for k in DEX3_MOTORS]
                         for s in ("left", "right")} if HAND_MODEL == "dex3" else None
        for side, off in (("right", 0), ("left", 6)):
            for k, dof in enumerate(INSPIRE_DOFS if self.dex3_adr is None else []):
                a = m.actuator(f"{side}_hand_{dof}").id
                j = m.actuator_trnid[a, 0]
                self.hand.append((off + k, m.jnt_qposadr[j], m.jnt_range[j].copy()))
        self.booths = {b["id"]: b for b in shop_booths_robot_frame(shop)} if shop else {}
        self.shoulder = {s: m.site(f"{s}_shoulder").id for s in ("left", "right")}
        self.palm = {s: m.site(f"{s}_palm").id for s in ("left", "right")}

    def _run(self):
        self._build(None)
        while True:
            q29, hands12, cam, booth, out = self.q.get()
            try:
                if self._new_shop is not None:
                    self.shop, self._new_shop = self._new_shop, None
                    self.r.close()
                    self._build(self.shop)
                out.put(self._frame(q29, hands12, cam, booth))
            except Exception as e:  # never kill the render thread
                print(f"[VIEW3D] {e}", flush=True)
                out.put(None)

    def _frame(self, q29, hands12, cam, booth):
        m, d = self.m, self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        if q29 is not None:
            for i, a in self.qadr.items():
                d.qpos[a] = q29[i]
        for idx, a, rng in self.hand:
            d.qpos[a] = inspire_to_joint(hands12[idx] if hands12 is not None else 1.0, rng)
        if self.dex3_adr is not None:  # hands12 in Inspire values (the arm channel converts the Dex3 state)
            h = np.ones(12) if hands12 is None else np.asarray(hands12, float)
            for side, sl in (("right", slice(0, 6)), ("left", slice(6, 12))):
                for a, v in zip(self.dex3_adr[side], inspire_to_dex3(h[sl], side)):
                    d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        d.qpos[2] = -lowest_point(m, d)  # feet on the floor
        mujoco.mj_forward(m, d)
        look, dist, az, el = CAMERAS.get(cam, CAMERAS["behind"])
        c = mujoco.MjvCamera()
        c.lookat[:], c.distance, c.azimuth, c.elevation = look, dist, az, el
        self.r.update_scene(d, c)
        b = self.booths.get(booth) if booth else None
        if b is not None:
            az = math.degrees(math.atan2(b["aim"][1], b["aim"][0]))
            side = "left" if az > CENTER_DEG else "right" if az < -CENTER_DEG else CENTER_ARM  # the pointing arm
            p0 = d.site_xpos[self.shoulder[side]].copy()
            self._line(p0, np.array(b["aim"]), [1.0, 0.15, 0.1, 0.9], 0.012)
            # the target line's ground projection helps reading the direction from above
            self._line(np.array([p0[0], p0[1], 0.01]), np.array([b["aim"][0], b["aim"][1], 0.01]), [1, 0.5, 0.1, 0.6], 0.02)
        img = self.r.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=80)
        return buf.getvalue()

    def _line(self, a, b, rgba, width):
        scn = self.r.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scn.ngeom += 1


def booth_geometry(shop, booth_id):
    """Azimuth (deg, + = robot's left), distance (m), torso turn (deg) for a booth of the shop."""
    st = shop.get("settings", {})
    for b in shop_booths_robot_frame(shop):
        if b["id"] == booth_id:
            x, y, _ = b["aim"]
            az = math.degrees(math.atan2(y, x))
            k, mx = st.get("torso_k", 0.25), st.get("torso_max_deg", 15.0)
            return {"azimuth_deg": az, "distance_m": math.hypot(x, y), "torso_deg": max(-mx, min(mx, k * az)),
                    "pointable": abs(az) <= MAX_AZIMUTH_DEG,
                    "side": "left" if az > CENTER_DEG else "right" if az < -CENTER_DEG else CENTER_ARM}
    return None
