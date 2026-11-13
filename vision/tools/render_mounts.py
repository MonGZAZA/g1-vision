"""Render what each camera mount sees (sensor sim, people at 1.2 / 2 / 3.5 m) -> docs/camera_mounts.jpg."""
import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import sim_sensors as ss  # noqa: E402

out = []
for mount in ("level", "head_d435"):
    a = argparse.Namespace(shop="", scenario="empty", camera_mount=mount, hfov=69.0, size=[640, 480], port=0,
                           lidar_port=0, lidar_rays=0, joints_url="")
    sim = ss.SensorSim(a)
    sim.set_scenario({"people": [{"start": [1.2, 0.35], "steps": [{"wait": 99, "face": "robot"}]},
                                 {"start": [2.0, -0.6], "steps": [{"wait": 99, "face": "robot"}]},
                                 {"start": [3.5, 0.5], "steps": [{"wait": 99, "face": "robot"}]}]})
    sim._pose(1.0)
    r = mujoco.Renderer(sim.m, 480, 640)
    r.update_scene(sim.d, camera="vision_cam")
    img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
    cv2.putText(img, f"camera mount: {mount}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    out.append(img)
    r.close()
dst = Path(__file__).resolve().parents[1] / "docs/camera_mounts.jpg"
cv2.imwrite(str(dst), np.hstack(out), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(dst)
