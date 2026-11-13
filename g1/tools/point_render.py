"""Contact sheet of the pointing poses for a shop's booths (offscreen), for visual review.

    MUJOCO_GL=egl .venv/bin/python tools/point_render.py out.png [--shop ../shops/mockup/shop.json] [--cam behind|front|top]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from arm_channel import HAND_POSES  # noqa: E402
from pointing import Pointer  # noqa: E402
from robot_model import MOTORS, UPPER, shop_booths_robot_frame  # noqa: E402
from view3d import View3D  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--shop", default=str(Path(__file__).resolve().parents[2] / "shops/mockup/shop.json"))
    p.add_argument("--cam", default="behind")
    a = p.parse_args()
    shop = json.loads(Path(a.shop).read_text())
    ptr, view = Pointer(), View3D()
    view.set_shop(shop)
    tiles = []
    for b in shop_booths_robot_frame(shop):
        r = ptr.solve(b["aim"], shop.get("settings"))
        q29 = np.zeros(29)
        if "q" in r:
            for n, v in zip(UPPER, r["q"]):
                q29[MOTORS.index(n)] = v
        hands = np.array(HAND_POSES["relaxed"] * 2, float)
        if r.get("side") == "right":
            hands[0:6] = HAND_POSES["flat"]
        elif r.get("side") == "left":
            hands[6:12] = HAND_POSES["flat"]
        img = Image.open(__import__("io").BytesIO(view.render(q29, hands, a.cam, b["id"], timeout=20)))
        dr = ImageDraw.Draw(img)
        txt = (f"{b['id']}  az {r['azimuth_deg']:+.0f}  d {r['distance_m']:.1f} m  torso {r['torso_deg']:+.0f}  "
               f"{r.get('side', '')}  elev {r['elev_deg']:+.0f}")
        txt2 = f"err az {r.get('azimuth_error_deg', 0):+.1f} el {r.get('elev_error_deg', 0):+.1f}  {'OK' if r['ok'] else r['reason']}"
        dr.rectangle([0, 0, 640, 40], fill=(0, 0, 0))
        dr.text((6, 4), txt, fill=(255, 255, 255))
        dr.text((6, 22), txt2, fill=(120, 255, 120) if r["ok"] else (255, 120, 120))
        tiles.append(img)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w // 2, rows * h // 2), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t.resize((w // 2, h // 2)), ((i % cols) * w // 2, (i // cols) * h // 2))
    sheet.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
