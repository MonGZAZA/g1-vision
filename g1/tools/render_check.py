"""Render the model in a given pose to a PNG (offscreen), for visual checks. Usage:
    MUJOCO_GL=egl .venv/bin/python tools/render_check.py out.png [joint=value ...] [--az 150] [--el -10] [--dist 2.2]
"""
import sys

import mujoco
import numpy as np
from PIL import Image

STAND = {"hip_pitch": -0.1, "knee": 0.3, "ankle_pitch": -0.2}


def main():
    out, args = sys.argv[1], sys.argv[2:]
    az, el, dist, look = 150.0, -10.0, 2.2, [0, 0, 0.9]
    q = {}
    it = iter(args)
    for a in it:
        if a == "--az": az = float(next(it))
        elif a == "--el": el = float(next(it))
        elif a == "--dist": dist = float(next(it))
        elif a == "--look": look = [float(v) for v in next(it).split(",")]
        else:
            k, v = a.split("="); q[k] = float(v)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    d.qpos[2] = 0.793
    d.qpos[3] = 1
    for s in ("left", "right"):
        for j, v in STAND.items():
            d.qpos[m.jnt_qposadr[m.joint(f"{s}_{j}_joint").id]] = v
    for k, v in q.items():
        d.qpos[m.jnt_qposadr[m.joint(k).id]] = v
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 600, 600)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 1
    r.update_scene(d, cam, opt)
    Image.fromarray(r.render()).save(out)


if __name__ == "__main__":
    main()
