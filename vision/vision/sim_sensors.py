"""Sensor sim for the vision module: people walking around the simulated G1, its head lidar and a camera.

A separate process next to exp05's sim (sim_robot.py is not changed): it builds the same G1 model standing at the
robot-frame origin, the booths of the shop, and up to MAX_PEOPLE mannequins (vision/mannequin.py) that follow a
scenario. The robot's waist and arms mirror the exp05 server (GET /api/status), so the lidar sees the arms as on
the real robot (the vision module must remove them). Outputs, in the real robot's formats:

    lidar   Livox MID-360 point packets to UDP 127.0.0.1:--lidar-port (default 56305), 10 Hz, raycast from the head
            (upside down, -7..52° in the lidar frame, random pattern, 2 cm range noise)
    camera  GET /color.jpg (port --port, default 8110), rendered from --camera-mount on torso_link
    truth   GET /truth: every person's position, yaw, speed, facing the robot (for checks)
    joints  GET /api/status: the joint angles used ({"arm": {"q", "imu_quat"}}), POST /joints {"q": [15]} overrides
            them (checks), {"q": null} goes back to mirroring
    scene   POST /scenario {"people": [...]} (see SCENARIOS), POST /scenario {"name": "demo"}

    .venv/bin/python -m vision.sim_sensors [--shop ../shops/mockup/shop.json] [--scenario demo]
"""

import argparse
import io
import json
import math
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .geometry import CAMERA_MOUNTS, LIDAR_MOUNT, RobotGeometry, parse_mount, rpy_matrix  # noqa: E402
from .lidar import SIM_PORT, pack_points  # noqa: E402
from .mannequin import PersonPose, add_person  # noqa: E402
from robot_model import MODEL, UPPER, shop_booths_robot_frame  # noqa: E402  (path set by .geometry)

MAX_PEOPLE = 6
RAY_GROUPS = np.array([1, 1, 1, 1, 1, 0], np.uint8)  # group 5 = head shell (around the lidar)
FACING_DEG = 40.0

# A scenario: people, each a list of steps. Step: {"to": [x, y], "speed": m/s, "face": "motion"|"robot"|deg}
# or {"wait": s, "face": ...}. "start": [x, y], "delay": s before appearing, "loop": restart at the end.
# Robot frame: x ahead (into the shop), y to the robot's left.
SCENARIOS = {
    "demo": [
        {"name": "customer", "start": [6.0, -2.5], "delay": 0.0, "loop": True, "steps": [
            {"to": [1.4, -0.2], "speed": 1.1, "face": "motion"}, {"wait": 6.0, "face": "robot"},
            {"to": [5.0, 3.5], "speed": 1.2, "face": "motion"}, {"wait": 6.0, "face": "motion"}]},
        {"name": "passer-by", "start": [3.2, -6.0], "delay": 3.0, "loop": True, "steps": [
            {"to": [3.2, 6.0], "speed": 1.3, "face": "motion"}, {"wait": 4.0, "face": 90},
            {"to": [3.2, -6.0], "speed": 1.3, "face": "motion"}, {"wait": 4.0, "face": -90}]},
        {"name": "side visitor", "start": [-1.0, 4.0], "delay": 10.0, "loop": True, "steps": [
            {"to": [0.15, 0.85], "speed": 0.7, "face": "motion"}, {"wait": 4.0, "face": 0},
            {"to": [-1.0, 4.0], "speed": 0.7, "face": "motion"}, {"wait": 12.0, "face": 0}]},
    ],
    "empty": [],
}


class Person:
    def __init__(self, spec):
        self.spec = spec
        self.name = spec.get("name", "")
        self.height = float(spec.get("height", 1.70))
        self.segments, t, p = [], 0.0, np.array(spec["start"], float)
        for st in spec["steps"]:
            if "to" in st:
                q = np.array(st["to"], float)
                dur = float(np.linalg.norm(q - p)) / max(float(st.get("speed", 1.2)), 0.05)
                self.segments.append((t, t + dur, p, q, st.get("face", "motion")))
                t, p = t + dur, q
            else:
                self.segments.append((t, t + float(st["wait"]), p, p, st.get("face", "robot")))
                t += float(st["wait"])
        self.period = t
        self.delay = float(spec.get("delay", 0.0))
        self.loop = bool(spec.get("loop", False))
        self.yaw = 0.0

    def state(self, t):
        """(visible, x, y, yaw, vx, vy, walked distance)."""
        t -= self.delay
        if t < 0 or (not self.loop and t > self.period) or self.period <= 0:
            return False, 0, 0, 0, 0, 0, 0
        t = t % self.period if self.loop else t
        walked = 0.0
        for t0, t1, p, q, face in self.segments:
            if t <= t1 or (t0, t1) == self.segments[-1][:2]:
                a = 0.0 if t1 <= t0 else min(1.0, (t - t0) / (t1 - t0))
                x, y = p + a * (q - p)
                v = (q - p) / (t1 - t0) if t1 > t0 else np.zeros(2)
                walked += a * float(np.linalg.norm(q - p))
                if face == "motion":
                    if np.linalg.norm(v) > 1e-6:
                        self.yaw = math.atan2(v[1], v[0])
                elif face == "robot":
                    self.yaw = math.atan2(-y, -x)
                else:
                    self.yaw = math.radians(float(face))
                return True, float(x), float(y), self.yaw, float(v[0]), float(v[1]), walked
            walked += float(np.linalg.norm(q - p))
        return False, 0, 0, 0, 0, 0, 0


def camera_quat(R_mount):
    """MuJoCo camera (looks along -z, y up) from a mount rotation (x forward, y left, z up)."""
    Rc = R_mount @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rc.flatten())
    return q


class SensorSim:
    def __init__(self, args):
        self.args = args
        spec = mujoco.MjSpec.from_file(str(MODEL))
        spec.visual.headlight.ambient = [0.35] * 3
        spec.visual.headlight.diffuse = [0.45] * 3
        for g in spec.body("head_link").geoms:  # the lidar sits inside the head shell: not seen by the raycast
            g.group = 5
        if args.shop:
            shop = json.loads(open(args.shop).read())
            for b in shop_booths_robot_frame(shop):
                body = spec.worldbody.add_body(name=f"booth_{b['id']}", pos=[b["cx"], b["cy"], 0.5],
                                               quat=[math.cos(b["yaw"] / 2), 0, 0, math.sin(b["yaw"] / 2)])
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[b["w"] / 2, b["h"] / 2, 0.5],
                              rgba=b["rgba"][:3] + [1.0], contype=0, conaffinity=0)
        for k in range(MAX_PEOPLE):
            add_person(spec, k, height=[1.70, 1.62, 1.78, 1.55, 1.85, 1.68][k])
        self.cam_mount = parse_mount(args.camera_mount, CAMERA_MOUNTS)
        w, h = args.size
        fovy = math.degrees(2 * math.atan(math.tan(math.radians(args.hfov) / 2) * h / w))
        R = rpy_matrix(self.cam_mount["rpy_deg"])
        parent = spec.body(self.cam_mount["parent"]) if self.cam_mount.get("parent") else spec.worldbody
        cam = parent.add_camera(name="vision_cam", pos=self.cam_mount["pos"], quat=camera_quat(R), fovy=fovy)
        del cam
        self.m = m = spec.compile()
        m.vis.map.zfar = 60.0
        self.d = mujoco.MjData(m)
        # standing robot: same posing code as the vision module
        self.geom = RobotGeometry()
        self.qadr_robot = {n: m.jnt_qposadr[m.joint(n).id] for n in self.geom.qadr}
        self.q_rest = [float(self.geom.d.qpos[self.geom.qadr[n]]) for n in UPPER]  # used without a joint source
        self.free_robot = m.jnt_qposadr[0]
        self.poses = [PersonPose(m, k) for k in range(MAX_PEOPLE)]
        self.torso = m.body("torso_link").id
        self.lidar_mount = LIDAR_MOUNT
        self.lock = threading.Lock()
        self.people = []
        self.t0 = time.time()
        self.set_scenario({"name": args.scenario})
        self.q_override = None
        self.q_server, self.q_used = None, None
        self.jpg = None
        self.truth = []
        self.udp = __import__("socket").socket(2, 2)  # AF_INET, SOCK_DGRAM
        self.lidar_addr = ("127.0.0.1", args.lidar_port)
        self.frame_cnt = 0
        self.rng = np.random.default_rng(0)
        if args.joints_url:
            threading.Thread(target=self._joints, daemon=True).start()
        threading.Thread(target=self._loop, name="sensors", daemon=True).start()

    def set_scenario(self, body):
        people = body.get("people")
        if people is None:
            people = SCENARIOS[body.get("name", "demo")]
        with self.lock:
            self.people = [Person(p) for p in people[:MAX_PEOPLE]]
            self.t0 = time.time()

    def _joints(self):
        while True:
            try:
                with urllib.request.urlopen(self.args.joints_url, timeout=1.0) as r:
                    q = json.loads(r.read()).get("arm", {}).get("q")
                self.q_server = q if q and len(q) == len(UPPER) else None
            except Exception:  # noqa: BLE001 (server not running: rest pose)
                self.q_server = None
            time.sleep(0.05)

    def _pose(self, t):
        d = self.d
        self.geom.update(self.q_override or self.q_server)
        self.q_used = list(self.geom.q_upper) if self.geom.q_upper else self.q_rest
        d.qpos[self.free_robot:self.free_robot + 7] = self.geom.d.qpos[:7]
        for n, a in self.qadr_robot.items():
            d.qpos[a] = self.geom.d.qpos[self.geom.qadr[n]]
        truth = []
        with self.lock:
            people = list(self.people)
        for k, pose in enumerate(self.poses):
            if k >= len(people):
                pose.set(d.qpos, 0, 0, 0, visible=False)
                continue
            vis, x, y, yaw, vx, vy, walked = people[k].state(t)
            speed = math.hypot(vx, vy)
            pose.set(d.qpos, x, y, yaw, phase=walked / 0.7 * math.pi, amp=0.6 if speed > 0.05 else 0.0, visible=vis)
            if vis:
                to_robot = math.atan2(-y, -x)
                off = math.degrees(math.remainder(yaw - to_robot, 2 * math.pi))
                truth.append({"k": k, "name": people[k].name, "x": round(x, 3), "y": round(y, 3),
                              "distance_m": round(math.hypot(x, y), 3),
                              "bearing_deg": round(math.degrees(math.atan2(y, x)), 2),
                              "yaw_deg": round(math.degrees(yaw), 1), "vx": round(vx, 3), "vy": round(vy, 3),
                              "speed_mps": round(speed, 3), "facing_robot": abs(off) < FACING_DEG,
                              "face_offset_deg": round(off, 1)})  # fmt: skip
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        self.truth = truth

    def _lidar(self, n_rays):
        m, d = self.m, self.d
        Rt, pt = d.xmat[self.torso].reshape(3, 3), d.xpos[self.torso]
        Rl = Rt @ rpy_matrix(self.lidar_mount["rpy_deg"])
        origin = pt + Rt @ np.array(self.lidar_mount["pos"])
        el = np.radians(self.rng.uniform(-7.0, 52.0, n_rays))
        az = self.rng.uniform(-math.pi, math.pi, n_rays)
        dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
        vec = dirs @ Rl.T
        geomid = np.zeros(n_rays, np.int32)
        dist = np.zeros(n_rays)
        mujoco.mj_multiRay(m, d, origin, vec.flatten(), RAY_GROUPS, True, self.torso, geomid, dist, None, n_rays, 40.0)
        hit = (geomid >= 0) & (dist > 0.1) & np.isfinite(dist)
        r = dist[hit] + self.rng.normal(0, 0.02, hit.sum())
        pts = dirs[hit] * r[:, None]
        self.frame_cnt += 1
        ts = time.time_ns()
        for i in range(0, len(pts), 96):
            self.udp.sendto(pack_points(pts[i:i + 96], i // 96, self.frame_cnt, ts), self.lidar_addr)

    def _loop(self):
        w, h = self.args.size
        renderer = mujoco.Renderer(self.m, h, w) if self.args.port else None
        tick, t_next = 0, time.time()
        while True:
            t = time.time() - self.t0
            self._pose(t)
            if tick % 3 == 0 and self.args.lidar_port:
                self._lidar(self.args.lidar_rays)
            if renderer is not None and tick % 2 == 0:
                renderer.update_scene(self.d, camera="vision_cam")
                img = renderer.render()
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, "JPEG", quality=85)
                self.jpg = buf.getvalue()
            tick += 1
            t_next += 1 / 30
            time.sleep(max(0.0, t_next - time.time()))
            if time.time() - t_next > 0.5:
                t_next = time.time()


def serve(sim, port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/color.jpg":
                return self._send(200, sim.jpg, "image/jpeg") if sim.jpg else self._send(503, {"error": "no frame"})
            if u.path == "/truth":
                return self._send(200, {"time": time.time(), "t": round(time.time() - sim.t0, 3),
                                        "people": sim.truth})
            if u.path == "/api/status":
                return self._send(200, {"arm": {"q": sim.q_used, "imu_quat": [1.0, 0.0, 0.0, 0.0]}})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            u = urlparse(self.path)
            if u.path == "/scenario":
                sim.set_scenario(body)
                return self._send(200, {"ok": True})
            if u.path == "/joints":
                q = body.get("q")
                sim.q_override = list(q) if q else None
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    srv.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shop", default="")
    p.add_argument("--scenario", default="demo", choices=sorted(SCENARIOS))
    p.add_argument("--camera-mount", default="level")
    p.add_argument("--hfov", type=float, default=69.0)
    p.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    p.add_argument("--port", type=int, default=8110, help="HTTP: camera image, truth, scenario")
    p.add_argument("--lidar-port", type=int, default=SIM_PORT)
    p.add_argument("--lidar-rays", type=int, default=8000, help="rays per 0.1 s frame (real MID-360: ~20000)")
    p.add_argument("--joints-url", default="http://127.0.0.1:8100/api/status",
                   help="mirror the robot's joints from the exp05 server ('' = rest pose)")
    args = p.parse_args()
    sim = SensorSim(args)
    print(f"[sensor-sim] camera http://127.0.0.1:{args.port}/color.jpg ({args.camera_mount}), lidar UDP "
          f"127.0.0.1:{args.lidar_port}, {len(sim.people)} people (scenario {args.scenario})", flush=True)
    serve(sim, args.port)


if __name__ == "__main__":
    main()
