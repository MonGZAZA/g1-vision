"""exp05 vision service: people around the robot for the behaviour layer (HTTP, default port 8107).

    GET  /people                 {"time", "people": [...], "proximity": {...}, "events_seq"}; ?all=1 also unconfirmed
    GET  /events?after=N&wait=S  events with seq > N (long poll up to S s): person_new, person_near, person_far,
                                 person_dwell, person_facing, person_not_facing, person_left, too_close, clear
    GET  /status                 sources, rates, timings, settings
    GET  /frame.jpg  /top.jpg    annotated camera image / top view (debug);  GET /  a small page with both

A person (robot frame: x ahead, y left, metres; bearing deg, positive to the left):
    id, x, y, distance_m (from the robot centre), gap_m (nearest lidar point of the person), bearing_deg,
    speed_mps, radial_mps (negative = approaching), vx, vy, heading_deg (direction of walking), moving, approaching,
    facing (true / false while the camera sees the person, null when it cannot), head_yaw_deg, body_yaw_deg,
    face_score, in_camera_view, seen_by_camera, source ("lidar+camera" | "camera" | "lidar"), confirmed,
    near, near_s, still_s, age_s, last_seen_s

Run (see ../README.md): .venv/bin/python -m vision.service --camera http://127.0.0.1:8110/color.jpg --lidar-port 56305
"""

import argparse
import json
import math
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .age_gender import AgeGenderEstimator
from .camera import Camera
from .detector import PersonDetector
from .geometry import CAMERA_MOUNTS, LIDAR_MOUNT, RobotGeometry, parse_mount, rpy_matrix
from .lidar import LidarPeople, LidarReceiver
from .tracker import CLOSE_M, NEAR_M, CameraModel, Events, Tracker, match_clusters

PRESENCE_M = 1.5  # --presence-url: a confirmed, facing person closer than this
CAMERA_MAX_AGE_S = 1.5  # detections of older frames are dropped (the detector can take ~0.4 s under GPU load)
SECTORS = (("front", -45, 45), ("left", 45, 135), ("back", 135, 225), ("right", 225, 315))

POSE_CONNECTIONS = (
    # Face / Head
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)


def _cluster_in_direction(clusters, used, wedge, tol=10.0):
    """Nearest unused person-like cluster whose bearing is within tol deg of the camera wedge."""
    b_lo, b_c, b_hi = wedge
    best = None
    for k, c in enumerate(clusters):
        if k in used or not c["person_like"] or not (b_lo - tol <= c["bearing"] <= b_hi + tol):
            continue
        if best is None or c["range"] < clusters[best]["range"]:
            best = k
    return best


class Joints:
    """Upper-body joint angles + pelvis attitude from the exp05 server (GET /api/status), 10 Hz."""

    def __init__(self, url):
        self.url, self.q, self.imu, self.t, self.error = url, None, None, 0.0, "not polled yet"
        if url:
            threading.Thread(target=self._run, name="joints", daemon=True).start()
        else:
            self.error = "no joint source (rest pose)"

    def _run(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=1.0) as r:
                    st = json.loads(r.read())
                arm = st.get("arm", {})
                if arm.get("q"):
                    self.q, self.imu, self.t, self.error = arm["q"], arm.get("imu_quat"), time.time(), ""
                else:
                    self.error = "server has no joint state yet"
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"[:120]
            time.sleep(0.1)

    def ok(self):
        return self.q is not None and time.time() - self.t < 1.0


class Vision:
    def __init__(self, args):
        self.args = args
        self.events = Events()
        self.tracker = Tracker(self.events)
        self.geom = RobotGeometry() if args.camera_mount != "laptop" or args.lidar_port else None
        self.cam_mount = parse_mount(args.camera_mount, CAMERA_MOUNTS)
        self.lidar_mount = parse_mount(args.lidar_mount, {"default": LIDAR_MOUNT}) if args.lidar_port else None
        self.joints = Joints(args.joints_url if (self.geom and args.joints_url and args.joints_url.lower() not in ("", "none")) else "")
        self.camera = Camera(args.camera, args.max_fps) if args.camera else None
        self.detector = PersonDetector(args.model, gpu=not args.cpu, backend=getattr(args, "pose_backend", "auto")) if args.camera else None
        self.age_gender = AgeGenderEstimator() if args.camera else None
        self.lidar = LidarReceiver(args.lidar_port) if args.lidar_port else None
        self.lp = LidarPeople()
        self.cam = CameraModel(args.hfov, (640, 480))
        self.lock = threading.Lock()
        self.dets, self.det_t, self.det_new = [], 0.0, False
        self.frame = None
        self.view = {"pts": np.zeros((0, 3)), "clusters": [], "prox": {}}
        self.people, self.prox = [], {}
        self.clusters, self.kept, self.last_lidar_t = [], np.zeros((0, 3)), 0.0
        self.loop_ms = 0.0
        self.floor_z = None
        self.presence_posts, self.presence_error = 0, ""
        if args.presence_url:
            threading.Thread(target=self._presence_loop, name="presence", daemon=True).start()
        if self.camera:
            threading.Thread(target=self._camera_loop, name="detect", daemon=True).start()
        threading.Thread(target=self._fusion_loop, name="fusion", daemon=True).start()

    # ---- optional bridge to the exp05 listener (2.2): "a customer is in front of the robot"
    def _presence_loop(self):
        """Every second while a confirmed person is within PRESENCE_M and facing the robot:
        POST {"present": true, "hold_s": 2} to --presence-url (the server's /api/listen/presence)."""
        while True:
            time.sleep(1.0)
            with self.lock:
                hit = [p for p in self.people if p["confirmed"] and p["facing"] and p["distance_m"] < PRESENCE_M]
            if not hit:
                continue
            body = json.dumps({"present": True, "hold_s": 2, "id": hit[0]["id"],
                               "distance_m": hit[0]["distance_m"]}).encode()
            try:
                req = urllib.request.Request(self.args.presence_url, body, {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=1.0).read()
                self.presence_posts += 1
                self.presence_error = ""
            except Exception as e:  # noqa: BLE001 (server down: keep trying)
                self.presence_error = f"{type(e).__name__}: {e}"[:120]

    # ---- camera: detections in pixels (placed in the robot frame by the fusion loop)
    def _camera_loop(self):
        last = -1
        while True:
            img, t, n = self.camera.latest()
            if img is None or n == last:
                time.sleep(0.005)
                continue
            last = n
            dets = self.detector.detect(img)
            with self.lock:
                self.dets, self.det_t, self.det_new, self.frame = dets, t, True, img

    # ---- 10 Hz: lidar clusters + camera detections -> tracks -> events
    def _fusion_loop(self):
        period = 0.1
        t_next = time.time()
        while True:
            t0 = time.time()
            try:
                self._fuse(t0)
            except Exception as e:  # noqa: BLE001 (keep serving; show the error)
                print(f"[vision] fusion error: {type(e).__name__}: {e}", flush=True)
                import traceback

                traceback.print_exc()
            self.loop_ms = 0.8 * self.loop_ms + 0.2 * (time.time() - t0) * 1000
            t_next += period
            time.sleep(max(0.0, t_next - time.time()))
            if time.time() - t_next > 1.0:
                t_next = time.time()

    def _fuse(self, now):
        if self.geom is not None:
            ok = self.joints.ok()
            self.geom.update(self.joints.q if ok else None, self.joints.imu if ok else None)
            self.cam.set_pose(*self.geom.mount_pose(self.cam_mount))
        else:
            self.cam.set_pose(rpy_matrix(self.cam_mount["rpy_deg"]), self.cam_mount["pos"])
        # new lidar frame? (each frame is used once; the camera may run faster or slower)
        new_lidar, t_meas = False, now
        if self.lidar is not None:
            t_l, pts_l = self.lidar.latest()
            if t_l != self.last_lidar_t and now - t_l < 0.5:
                self.last_lidar_t = t_l
                new_lidar, t_meas = True, t_l - self.lidar.frame_s / 2  # middle of the accumulated frame
                R, p = self.geom.mount_pose(self.lidar_mount)
                pts = pts_l @ R.T + p
                pts = pts[~self.geom.self_mask(pts)]
                floor = pts[(np.abs(pts[:, 2]) < 0.15) & (np.hypot(pts[:, 0], pts[:, 1]) < 4)]
                if len(floor) > 50:
                    self.floor_z = float(np.median(floor[:, 2]))
                freeze = [(t.x[0], t.x[1]) for t in self.tracker.tracks if t.confirmed]
                self.clusters, self.kept = self.lp.clusters(pts, freeze)
            elif now - t_l >= 0.5:
                self.clusters, self.kept = [], np.zeros((0, 3))
        clusters = self.clusters
        with self.lock:
            dets, det_t, new_cam = self.dets, self.det_t, self.det_new
            self.det_new = False
            if self.frame is not None:
                self.cam.set_size((self.frame.shape[1], self.frame.shape[0]))
        new_cam = new_cam and now - det_t < CAMERA_MAX_AGE_S
        if not new_lidar and not new_cam:
            with self.lock:
                self.people = self.tracker.people(now)
            return
        if not new_lidar:
            t_meas = det_t
        meas, used = [], set()
        if new_cam:
            for det, (k, wedge, pos_cam, how) in zip(dets, match_clusters(dets, clusters, self.cam)):
                det["wedge"], det["cam_xy"], det["cam_how"] = wedge, pos_cam, how
                if k is None:  # nothing in the wedge: the nearest unused person-sized cluster in this direction
                    k = _cluster_in_direction(clusters, used, wedge)
                if k is not None and (new_lidar or k not in used):
                    c = clusters[k]
                    used.add(k)
                    meas.append({"xy": np.array([c["x"], c["y"]]), "sigma": 0.08, "source": "lidar+camera",
                                 "det": det, "gap": c["near"], "lidar": new_lidar})
                elif pos_cam is not None:
                    meas.append({"xy": np.asarray(pos_cam), "sigma": max(0.2, 0.15 * float(np.hypot(*pos_cam))),
                                 "source": "camera", "det": det})
        if new_lidar:
            for k, c in enumerate(clusters):
                if k in used or not c["person_like"] or c["static"]:
                    continue
                if any("det" in m and np.hypot(*(m["xy"] - [c["x"], c["y"]])) < 0.5 for m in meas):
                    continue
                meas.append({"xy": np.array([c["x"], c["y"]]), "sigma": 0.08, "source": "lidar", "gap": c["near"]})
        self.tracker.step(t_meas, meas, self.cam if self.camera else None,
                          lidar_active=self.lidar is not None and now - self.last_lidar_t < 0.5)
        for m in meas:  # tracks matched to detections: give the detection its track id (drawing)
            if "det" in m:
                tr_id = m["det"].get("track")
                if tr_id is not None and self.age_gender is not None and self.frame is not None and m["det"].get("face_box"):
                    fb = m["det"]["face_box"]
                    h, w = self.frame.shape[:2]
                    fw, fh = fb[2] - fb[0], fb[3] - fb[1]
                    px0 = max(0, int(fb[0] - fw * 0.2))
                    py0 = max(0, int(fb[1] - fh * 0.2))
                    px1 = min(w, int(fb[2] + fw * 0.2))
                    py1 = min(h, int(fb[3] + fh * 0.2))
                    if px1 > px0 + 16 and py1 > py0 + 16:
                        face_crop = self.frame[py0:py1, px0:px1]
                        bx = m["det"]["box"]
                        bx0, by0 = max(0, int(bx[0])), max(0, int(bx[1]))
                        bx1, by1 = min(w, int(bx[2])), min(h, int(bx[3]))
                        body_crop = self.frame[by0:by1, bx0:bx1] if bx1 > bx0 + 16 and by1 > by0 + 16 else None
                        self.age_gender.submit(tr_id, face_crop, body_crop)

        if self.age_gender is not None:
            active_ids = [t.id for t in self.tracker.tracks]
            self.age_gender.purge_tracks(active_ids)
            for tr in self.tracker.tracks:
                g, a = self.age_gender.get(tr.id)
                if g is not None:
                    tr.gender = g
                if a is not None:
                    tr.age = a
        if new_lidar:
            sectors = {name: None for name, _, _ in SECTORS}
            for c in clusters:
                if c["static"] or (not c["person_like"] and c["size"] > 1.4):
                    continue  # walls, shelves, booths: the arm-safety check is about people
                az = c["bearing"] % 360
                for name, lo, hi in SECTORS:
                    lo, hi = lo % 360, hi % 360
                    if (lo <= az < hi) if lo < hi else (az >= lo or az < hi):
                        sectors[name] = c["near"] if sectors[name] is None else min(sectors[name], c["near"])
            self.prox = self.tracker.proximity({k: (round(v, 3) if v is not None else None)
                                                for k, v in sectors.items()}, now)
        with self.lock:
            self.people = self.tracker.people(now)
            self.view = {"pts": self.kept, "clusters": clusters, "dets": dets}

    # ---- outputs
    def snapshot(self, all_tracks=False):
        with self.lock:
            people = [p for p in self.people if all_tracks or p["confirmed"]]
            return {"time": round(time.time(), 3), "people": people, "proximity": self.prox,
                    "events_seq": self.events.seq}

    def status(self):
        a = self.args
        return {
            "camera": None if not self.camera else {"source": a.camera, "fps": round(self.camera.fps, 1),
                                                    "error": self.camera.error, "mount": a.camera_mount,
                                                    "hfov_deg": a.hfov, "size": [self.cam.w, self.cam.h]},
            "detector": None if not self.detector else {"model": a.model, "delegate": self.detector.delegate,
                                                        "ms": round(self.detector.ms, 1)},
            "lidar": None if not self.lidar else {"port": a.lidar_port, "frames": self.lidar.frames,
                                                  "points_per_s": round(self.lidar.rate),
                                                  "floor_z_m": None if self.floor_z is None else round(self.floor_z, 3)},
            "joints": {"url": self.joints.url, "ok": self.joints.ok(), "error": self.joints.error},
            "presence": {"url": a.presence_url, "posts": self.presence_posts, "error": self.presence_error},
            "fusion_ms": round(self.loop_ms, 1), "near_m": NEAR_M, "close_m": CLOSE_M,
        }  # fmt: skip

    def frame_jpg(self):
        img = None
        if self.camera is not None:
            raw_img, _, _ = self.camera.latest()
            if raw_img is not None:
                img = raw_img.copy()
        with self.lock:
            if img is None and self.frame is not None:
                img = self.frame.copy()
            dets, people = list(self.view.get("dets", [])), {p["id"]: p for p in self.people}
        if img is None:
            img = np.full((240, 320, 3), 40, np.uint8)
            cv2.putText(img, "no camera image", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        for det in dets:
            # Draw MediaPipe skeleton
            kpts = det.get("px")
            vis = det.get("vis")
            if kpts is not None and vis is not None:
                for i, j in POSE_CONNECTIONS:
                    if i < len(vis) and j < len(vis) and vis[i] > 0.4 and vis[j] > 0.4:
                        pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                        pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                        if i in (11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31) and j in (11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31):
                            line_col = (0, 255, 100)  # Left side (Greenish)
                        elif i in (12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32) and j in (12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
                            line_col = (255, 150, 0)  # Right side (Blue/Cyan)
                        else:
                            line_col = (0, 220, 255)  # Torso / Head (Yellow/Gold)
                        cv2.line(img, pt1, pt2, line_col, 2)
                for i in range(len(vis)):
                    if vis[i] > 0.4:
                        joint_pt = (int(kpts[i][0]), int(kpts[i][1]))
                        cv2.circle(img, joint_pt, 3, (0, 0, 255), -1)
                        cv2.circle(img, joint_pt, 4, (255, 255, 255), 1)

            x0, y0, x1, y1 = (int(v) for v in det["box"])
            p = people.get(det.get("track"))
            col = (0, 200, 0) if det["facing"] else (0, 160, 255)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            if det.get("face_box"):
                fx0, fy0, fx1, fy1 = (int(v) for v in det["face_box"])
                cv2.rectangle(img, (fx0, fy0), (fx1, fy1), (255, 255, 0) if det["face_score"] else (90, 90, 90), 1)
            if det.get("lip_pts"):
                lp = det["lip_pts"]
                lip_col = (0, 255, 0) if (p and p.get("talking")) else (0, 255, 255)
                cv2.line(img, tuple(lp[0]), tuple(lp[1]), lip_col, 2)
                cv2.line(img, tuple(lp[2]), tuple(lp[3]), (255, 200, 0), 1)
            label = (p.get("facing_state") if p and p.get("facing_state") else ("facing" if det["facing"] else "not facing"))
            if p:
                t_str = " [TALKING]" if p.get("talking") else ""
                gest = p.get("gesture", "still")
                g_str = f" [{gest.upper()}]" if gest in ("wave", "point") else ""
                label = f"#{p['id']} {p['distance_m']:.2f} m {p['speed_mps']:.1f} m/s {label}{t_str}{g_str}"
                if p.get("head_yaw_deg") is not None:
                    label += f" yaw:{p['head_yaw_deg']:.0f}°"
                g = p.get("gender")
                a = p.get("age")
                if g or a:
                    label += f" [{' '.join(filter(None, [g, a]))}]"
            cv2.putText(img, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if (p and p.get("talking")) else col, 1)
            y_extra = y1 + 18
            if p and p.get("talking"):
                cv2.putText(img, "TALKING", (x0, min(img.shape[0] - 6, y_extra)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(img, "TALKING", (x0, min(img.shape[0] - 6, y_extra)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                y_extra += 18
            if p and p.get("gesture") in ("wave", "point"):
                g_val = p.get("gesture")
                g_col = (0, 165, 255) if g_val == "point" else (255, 100, 0)
                g_txt = g_val.upper()
                if g_val == "point" and p.get("pointing_arm"):
                    g_txt += f" ({p['pointing_arm'].upper()})"
                cv2.putText(img, g_txt, (x0, min(img.shape[0] - 6, y_extra)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(img, g_txt, (x0, min(img.shape[0] - 6, y_extra)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, g_col, 2)
            if p and p.get("gesture") == "point" and p.get("pointing_ray_2d"):
                ray = p["pointing_ray_2d"]
                p_start = (int(ray["start"][0]), int(ray["start"][1]))
                p_end = (int(ray["end"][0]), int(ray["end"][1]))
                cv2.arrowedLine(img, p_start, p_end, (0, 0, 0), 6, tipLength=0.18)
                cv2.arrowedLine(img, p_start, p_end, (0, 240, 255), 3, tipLength=0.18)
                cv2.circle(img, p_start, 5, (0, 140, 255), -1)
                cv2.circle(img, p_start, 6, (255, 255, 255), 1)

                v_info = ""
                if p.get("pointing_yaw_deg") is not None and p.get("pointing_pitch_deg") is not None:
                    v_info = f"yaw:{p['pointing_yaw_deg']:.0f}° pitch:{p['pointing_pitch_deg']:.0f}°"
                if v_info:
                    tx = max(10, min(img.shape[1] - 180, p_end[0] + 6))
                    ty = max(18, min(img.shape[0] - 8, p_end[1]))
                    cv2.putText(img, v_info, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                    cv2.putText(img, v_info, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def top_jpg(self, size=560, span=7.0):
        """Top view, robot at the centre facing up: lidar points, clusters, tracks with velocity, camera view."""
        img = np.full((size, size, 3), 250, np.uint8)
        s = size / (2 * span)

        def px(x, y):  # robot x up, y left
            return int(size / 2 - y * s), int(size / 2 - x * s)

        for r, col in ((NEAR_M, (200, 230, 200)), (CLOSE_M, (190, 190, 245))):
            cv2.circle(img, px(0, 0), int(r * s), col, 2)
        for r in range(1, int(span) + 1):
            cv2.circle(img, px(0, 0), int(r * s), (225, 225, 225), 1)
        with self.lock:
            pts, clusters, people = self.view["pts"], self.view["clusters"], list(self.people)
        if self.camera:
            for u in (0, self.cam.w):
                d = self.cam.ray(u, self.cam.h / 2)
                h = math.hypot(d[0], d[1]) or 1
                cv2.line(img, px(self.cam.p[0], self.cam.p[1]), px(d[0] / h * span * 1.5, d[1] / h * span * 1.5),
                         (230, 200, 150), 1)
        for x, y in pts[::2, :2]:
            cv2.circle(img, px(x, y), 1, (150, 150, 150), -1)
        for c in clusters:
            cv2.circle(img, px(c["x"], c["y"]), max(3, int(c["size"] / 2 * s)),
                       (0, 190, 230) if c["person_like"] else (180, 180, 180), 1)
        cv2.rectangle(img, px(0.12, 0.2), px(-0.12, -0.2), (60, 60, 60), -1)
        cv2.arrowedLine(img, px(0, 0), px(0.45, 0), (255, 255, 255), 2)
        for p in people:
            col = (0, 150, 0) if p["confirmed"] else (160, 160, 160)
            if p["facing"]:
                col = (0, 200, 0)
            c = px(p["x"], p["y"])
            cv2.circle(img, c, 9, col, 2)
            cv2.arrowedLine(img, c, px(p["x"] + p["vx"], p["y"] + p["vy"]), (200, 60, 0), 2)
            if p.get("gesture") == "point" and p.get("pointing_vector"):
                pv = p["pointing_vector"]
                p_vec_end = px(p["x"] + pv[0] * 1.0, p["y"] + pv[1] * 1.0)
                cv2.arrowedLine(img, c, p_vec_end, (0, 220, 255), 2)
            t = f"#{p['id']} {p['distance_m']:.1f}m {p['speed_mps']:.1f}m/s" + (" F" if p["facing"] else "") + (" T" if p.get("talking") else "") + (" W" if p.get("gesture") == "wave" else (" P" if p.get("gesture") == "point" else ""))
            cv2.putText(img, t, (c[0] + 11, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
        if self.prox.get("too_close"):
            cv2.putText(img, "TOO CLOSE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>exp05 vision</title>
<style>body{font:14px system-ui;margin:12px;background:#f4f4f4}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
#f{max-width:640px;width:auto;height:auto;border:1px solid #ccc;background:#000;display:block}
#t{max-width:480px;width:auto;height:auto;border:1px solid #ccc;background:#fff;display:block}
table{border-collapse:collapse;background:#fff;margin-top:12px}td,th{padding:4px 8px;border:1px solid #ddd}
pre{background:#fff;padding:8px;max-height:240px;overflow:auto}</style></head><body>
<h3>exp05 vision &middot; people around the robot</h3>
<div class="row"><img id="f" src="/frame.jpg"><img id="t" src="/top.jpg"></div>
<table id="tab"></table><h4>Events</h4><pre id="ev"></pre><pre id="st"></pre>
<script>
let seq=0;const ev=document.getElementById('ev');
let f_busy=false, t_busy=false;
function loadFrame(){
  if(f_busy) return;
  f_busy = true;
  const im = document.getElementById('f');
  const next = new Image();
  next.onload = () => { im.src = next.src; f_busy = false; };
  next.onerror = () => { f_busy = false; };
  next.src = '/frame.jpg?' + Date.now();
}
function loadTop(){
  if(t_busy) return;
  t_busy = true;
  const im = document.getElementById('t');
  const next = new Image();
  next.onload = () => { im.src = next.src; t_busy = false; };
  next.onerror = () => { t_busy = false; };
  next.src = '/top.jpg?' + Date.now();
}
function tickTable(){
 fetch('/people?all=1').then(r=>r.json()).then(d=>{const k=['id','distance_m','bearing_deg','speed_mps','radial_mps','facing','facing_state','talking','gesture','pointing_vec','head_yaw_deg','gender','age','source','confirmed','near_s','still_s'];
  document.getElementById('tab').innerHTML='<tr>'+k.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+d.people.map(p=>'<tr>'+k.map(x=>{
    const v=p[x];if(v===null||v===undefined||v==='')return '<td>-</td>';
    if(x==='talking'&&v===true)return '<td style="color:#0a0;font-weight:bold">true</td>';
    if(x==='gesture'&&v==='wave')return '<td style="color:#00c;font-weight:bold">wave</td>';
    if(x==='gesture'&&v==='point')return '<td style="color:#d50;font-weight:bold">point ('+(p.pointing_arm||'').toUpperCase()+')</td>';
    if(x==='pointing_vec'){
      if(p.gesture==='point'&&p.pointing_vector){
        const pv=p.pointing_vector;
        const y=p.pointing_yaw_deg!==null?' yaw:'+p.pointing_yaw_deg+'°':'';
        const pi=p.pointing_pitch_deg!==null?' pitch:'+p.pointing_pitch_deg+'°':'';
        return '<td style="color:#d50;font-weight:bold">['+pv.join(', ')+']'+y+pi+'</td>';
      }
      return '<td>-</td>';
    }
    return '<td>'+v+'</td>';}).join('')+'</tr>').join('')
  +'<tr><td colspan=17>proximity: '+JSON.stringify(d.proximity)+'</td></tr>';});}
function poll(){fetch('/events?after='+seq+'&wait=10').then(r=>r.json()).then(d=>{for(const e of d.events){seq=e.seq;
  ev.textContent=new Date(e.time*1000).toLocaleTimeString()+' '+e.type+' '+(e.person?('#'+e.person.id+' '+e.person.distance_m+' m facing='+e.person.facing):JSON.stringify(e.sectors||''))+'\\n'+ev.textContent;}
  poll();}).catch(()=>setTimeout(poll,1000));}
setInterval(loadFrame,33);
setInterval(loadTop,100);
setInterval(tickTable,100);
setInterval(()=>fetch('/status').then(r=>r.json()).then(s=>document.getElementById('st').textContent=JSON.stringify(s,null,1)),1000);poll();
</script></body></html>"""


def _plain(o):
    """numpy scalars / arrays -> JSON."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def serve(vision, port, host):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, default=_plain).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/people":
                    return self._send(200, vision.snapshot(all_tracks=q.get("all", ["0"])[0] == "1"))
                if u.path == "/events":
                    after = int(q.get("after", ["0"])[0])
                    wait = min(float(q.get("wait", ["0"])[0]), 30.0)
                    evs, seq = vision.events.wait(after, wait)
                    return self._send(200, {"events": evs, "seq": seq})
                if u.path == "/debug":  # clusters and camera detections of the last fusion step
                    with vision.lock:
                        v = vision.view
                        dets = [{k: d.get(k) for k in ("box", "wedge", "cam_xy", "cam_how", "track", "facing",
                                                        "face_score", "head_yaw_deg", "gender", "age")} for d in v.get("dets", [])]
                        return self._send(200, {"clusters": v["clusters"], "dets": dets})
                if u.path == "/status":
                    return self._send(200, vision.status())
                if u.path == "/frame.jpg":
                    return self._send(200, vision.frame_jpg(), "image/jpeg")
                if u.path == "/top.jpg":
                    return self._send(200, vision.top_jpg(), "image/jpeg")
                if u.path in ("/", "/index.html"):
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                return self._send(404, {"error": "not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    print(f"[vision] http://{host}:{port}/  (people: /people, events: /events?after=0&wait=10)", flush=True)
    srv.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="", help="webcam:0 | http://.../color.jpg | unitree:http://.../capture | "
                                                "file:VIDEO (empty: no camera)")
    p.add_argument("--camera-mount", default="level",
                   help=f"{'|'.join(CAMERA_MOUNTS)} or x,y,z,roll,pitch,yaw (m, deg) on torso_link")
    p.add_argument("--hfov", type=float, default=69.0, help="camera horizontal field of view, deg (D435i colour 69)")
    p.add_argument("--max-fps", type=float, default=15.0, help="HTTP camera polling rate")
    p.add_argument("--model", default="full", choices=["lite", "full", "heavy"], help="pose landmarker model")
    p.add_argument("--cpu", action="store_true", help="pose landmarker on the CPU (default: GPU if available)")
    p.add_argument("--lidar-port", type=int, default=0, help="Livox UDP port (real 56301 via exp04 lidar_relay, "
                                                             "sim 56305; 0 = no lidar)")
    p.add_argument("--lidar-mount", default="default", help="'default' (upside down, 0.45 m above torso_link) or "
                                                            "x,y,z,roll,pitch,yaw")
    p.add_argument("--joints-url", default="http://127.0.0.1:8100/api/status", nargs="?", const="",
                   help="exp05 server status (joint angles for the sensor poses and the self filter)")
    p.add_argument("--presence-url", default="", help="optional: POST presence (2.2 listener gate), e.g. "
                                                       "http://127.0.0.1:8100/api/listen/presence")
    p.add_argument("--pose-backend", default="auto", choices=["auto", "rcnn", "mediapipe"], help="pose detector backend")
    p.add_argument("--port", type=int, default=8107)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve other machines (the camera image too)")
    args = p.parse_args()
    if not args.camera and not args.lidar_port:
        p.error("give --camera and/or --lidar-port")
    v = Vision(args)
    serve(v, args.port, args.host)


if __name__ == "__main__":
    main()
