"""Fusion of camera people and lidar clusters, tracking (walking speed) and events for the behaviour layer.

Distance comes from the lidar: a camera person gives a bearing wedge (left/right edge of the body in the image);
the lidar cluster inside that wedge (closest to the image's own rough distance, else the nearest) gives the
position. Without a lidar match the image's own estimate is used (source "camera"). Person-sized lidar clusters
the camera does not see (beside or behind the robot, or too close for the camera) become "lidar" tracks; they are
"confirmed" people only once the camera has seen them or they have moved > 0.6 m.

Tracks: constant-velocity Kalman filter on (x, y) in the robot frame -> speed, radial speed (negative =
approaching), heading of motion. Facing: true / false while the camera sees the person, null when it cannot.
"""

import itertools
import math
import threading
import time

import numpy as np

from .detector import ANKLE_Z, TORSO_M
from .gesture_ml import GestureClassifier, estimate_pointing_vector_3d, extract_arm_features

_GESTURE_CLF = GestureClassifier()

NEAR_M, NEAR_EXIT_M = 2.0, 2.3  # person_near zone (+ hysteresis)
ENGAGE_M = 3.5  # person_facing events only inside this range
DWELL_S, STILL_MPS = 2.0, 0.3  # person_dwell: in the near zone and slower than STILL_MPS for DWELL_S
CLOSE_M, CLEAR_M, CLEAR_S = 1.0, 1.2, 1.0  # too_close / clear (any lidar point, robot removed)
FACING_HOLD_S = 0.6  # facing stays true this long after the last frontal face (detector ~0.1-0.4 s/frame)
LIDAR_ONLY_M = 8.0  # lidar-only tracks are started within this range (unconfirmed until seen or moving)
CONFIRM_MOVE_M = 0.6
BEARING_TOL = 6.0  # deg: camera wedge vs lidar cluster bearing


def calibrate_camera_distance(raw_d):
    """Calibrate camera-only distance for 165cm person standing at 1m, 2m, 3m:
    Raw -> True:
    1.252m -> 1.0m
    1.745m -> 2.0m
    2.653m -> 3.0m
    """
    if raw_d <= 0.0:
        return 0.0
    if raw_d <= 1.252:
        return raw_d * (1.0 / 1.252)
    elif raw_d <= 1.745:
        return 1.0 + (raw_d - 1.252) * ((2.0 - 1.0) / (1.745 - 1.252))
    elif raw_d <= 2.653:
        return 2.0 + (raw_d - 1.745) * ((3.0 - 2.0) / (2.653 - 1.745))
    else:
        return 3.0 + (raw_d - 2.653) * ((3.0 - 2.0) / (2.653 - 1.745))


class CameraModel:
    """Pinhole camera on a mount: pixel -> ray in the robot frame. hfov in degrees, square pixels."""

    def __init__(self, hfov_deg, size_wh):
        self.hfov = hfov_deg
        self.set_size(size_wh)
        self.R, self.p = np.eye(3), np.zeros(3)

    def set_size(self, size_wh):
        self.w, self.h = size_wh
        self.f = (self.w / 2) / math.tan(math.radians(self.hfov) / 2)

    def set_pose(self, R, p):
        self.R, self.p = R, np.asarray(p, float)

    def ray(self, u, v):
        d = self.R @ np.array([1.0, -(u - self.w / 2) / self.f, -(v - self.h / 2) / self.f])
        return d / np.linalg.norm(d)

    def bearing(self, u, v):
        d = self.ray(u, v)
        return math.degrees(math.atan2(d[1], d[0]))

    def project(self, xyz):
        """Robot-frame point -> (u, v) or None if behind the camera."""
        c = self.R.T @ (np.asarray(xyz, float) - self.p)
        if c[0] <= 0.05:
            return None
        return self.w / 2 - self.f * c[1] / c[0], self.h / 2 - self.f * c[2] / c[0]

    def in_view(self, x, y, z=1.2, margin=10):
        uv = self.project([x, y, z])
        return uv is not None and margin <= uv[0] <= self.w - margin

    def place(self, det):
        """Robot-frame (x, y) of a detection from the image alone, and how: feet on the floor, else torso length."""
        if det.get("ankle_px"):
            d = self.ray(*det["ankle_px"])
            if d[2] < -0.05:
                t = (ANKLE_Z - self.p[2]) / d[2]
                if 0 < t < 15:
                    q = self.p + t * d
                    raw_r = float(np.hypot(q[0], q[1]))
                    cal_r = calibrate_camera_distance(raw_r)
                    q[:2] = q[:2] * (cal_r / max(raw_r, 1e-6))
                    return q[:2], "feet"
        if det.get("torso_px"):
            dist = self.f * TORSO_M / max(det["torso_px"], 1.0)
            if dist < 15:
                q = self.p + dist * self.ray(*det["centre"])
                raw_r = float(np.hypot(q[0], q[1]))
                cal_r = calibrate_camera_distance(raw_r)
                q[:2] = q[:2] * (cal_r / max(raw_r, 1e-6))
                return q[:2], "torso"
        if det.get("shoulder_px"):
            dist = self.f * 0.40 / max(det["shoulder_px"], 1.0)
            if 0.3 < dist < 15:
                q = self.p + dist * self.ray(*det["centre"])
                raw_r = float(np.hypot(q[0], q[1]))
                cal_r = calibrate_camera_distance(raw_r)
                q[:2] = q[:2] * (cal_r / max(raw_r, 1e-6))
                return q[:2], "shoulders"
        if det.get("face_box"):
            fb = det["face_box"]
            fh = max(abs(fb[3] - fb[1]), 1.0)
            dist = self.f * 0.20 / fh
            if 0.3 < dist < 15:
                q = self.p + dist * self.ray(*det["centre"])
                raw_r = float(np.hypot(q[0], q[1]))
                cal_r = calibrate_camera_distance(raw_r)
                q[:2] = q[:2] * (cal_r / max(raw_r, 1e-6))
                return q[:2], "face"
        return None, None


def match_clusters(dets, clusters, cam):
    """For each detection: (index of the lidar cluster, bearing wedge, image-only position)."""
    out = []
    for det in dets:
        x0, _, x1, _ = det["box"]
        v = det["centre"][1]
        b_hi, b_lo = cam.bearing(x0, v), cam.bearing(x1, v)  # image left = robot left = larger bearing
        b_c = cam.bearing(*det["centre"])
        pos_cam, how = cam.place(det)
        r_cam = float(np.hypot(*pos_cam)) if pos_cam is not None else None
        best, best_cost = None, math.inf
        for k, c in enumerate(clusters):
            if c["size"] > 1.4 or c["bearing_lo"] is None:
                continue
            # tolerance: the image is ~0.1-0.2 s older than the lidar frame (a walker moves a few degrees)
            if not (b_lo - BEARING_TOL <= c["bearing"] <= b_hi + BEARING_TOL):
                continue
            if r_cam is not None:
                err = abs(c["range"] - r_cam)
                if err > max(0.8, 0.45 * r_cam):
                    continue
                cost = err + 0.02 * abs(c["bearing"] - b_c)
            else:
                cost = c["range"] + 0.02 * abs(c["bearing"] - b_c)
            if cost < best_cost:
                best, best_cost = k, cost
        out.append((best, (b_lo, b_c, b_hi), pos_cam, how))
    return out


def _is_waving(hist, sh_w=80.0, min_samples=4, min_reversals=2):
    if len(hist) < min_samples:
        return False
    xs = np.array([x for _, x in hist])
    min_span = max(8.0, 0.12 * sh_w)
    min_total_path = max(14.0, 0.22 * sh_w)
    if np.ptp(xs) < min_span:
        return False
    diffs = np.diff(xs)
    if np.sum(np.abs(diffs)) < min_total_path:
        return False
    signs = []
    accum = 0.0
    thresh = max(3.5, 0.045 * sh_w)
    for d in diffs:
        accum += d
        if abs(accum) > thresh:
            s = 1 if accum > 0 else -1
            if not signs or signs[-1] != s:
                signs.append(s)
            accum = 0.0
    return max(0, len(signs) - 1) >= min_reversals


def _check_point_arm_2d(sh, elb, wr, hip, torso_h, sh_w=80.0):
    v1 = sh - elb
    v2 = wr - elb
    cos_elb = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    elb_angle = np.degrees(np.arccos(np.clip(cos_elb, -1.0, 1.0)))

    l_se = np.linalg.norm(elb - sh)
    l_ew = np.linalg.norm(wr - elb)
    l_sw = np.linalg.norm(wr - sh)
    reach = l_sw / (l_se + l_ew + 1e-6)

    v_torso = hip - sh
    cos_torso = np.dot(wr - sh, v_torso) / (l_sw * np.linalg.norm(v_torso) + 1e-6)
    torso_angle = np.degrees(np.arccos(np.clip(cos_torso, -1.0, 1.0)))

    elevated_or_out = (hip[1] - wr[1] > 0.02 * torso_h) or (wr[1] < elb[1] + 35.0) or (abs(wr[0] - sh[0]) > 0.45 * sh_w)
    extended = (reach >= 0.55) and (l_sw >= 0.48 * sh_w)
    return (elb_angle >= 115.0) and extended and (torso_angle >= 22.0) and elevated_or_out


def _check_point_arm_3d(sh, elb, wr, hip):
    v1 = sh - elb
    v2 = wr - elb
    cos_elb = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    elb_angle = np.degrees(np.arccos(np.clip(cos_elb, -1.0, 1.0)))

    l_se = np.linalg.norm(elb - sh)
    l_ew = np.linalg.norm(wr - elb)
    l_sw = np.linalg.norm(wr - sh)
    reach = l_sw / (l_se + l_ew + 1e-6)

    v_torso = hip - sh
    cos_torso = np.dot(wr - sh, v_torso) / (l_sw * np.linalg.norm(v_torso) + 1e-6)
    torso_angle = np.degrees(np.arccos(np.clip(cos_torso, -1.0, 1.0)))

    elevated_or_forward = (wr[1] < hip[1] - 0.02) or (wr[2] < sh[2] - 0.10) or (abs(wr[0] - sh[0]) > 0.30)
    return (elb_angle >= 115.0) and (reach >= 0.55) and (torso_angle >= 22.0) and elevated_or_forward and (l_sw >= 0.22)


class Track:
    _ids = itertools.count(1)

    def __init__(self, z, t, sigma):
        self.id = next(Track._ids)
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([sigma**2, sigma**2, 1.0, 1.0])
        self.t = self.t_seen = self.t_born = t
        self.hits = 1
        self.start = np.array(z, float)
        self.moved = 0.0
        self.camera_seen_t = None  # last camera detection
        self.camera_ever = False
        self.facing_t = None
        self.head_yaw = self.body_yaw = None
        self.face_score = 0.0
        self.source = ""
        self.gap = None  # nearest lidar point of the person (m from the robot centre)
        self.gender = None
        self.age = None
        self.talking = False
        self.talking_t = None
        self.mouth_history = []
        self.latest_openness = 0.0
        self.talking_sent = False
        self.gesture = "still"
        self.gesture_hold_t = None
        self.gesture_hold_type = None
        self.l_wrist_hist = []
        self.r_wrist_hist = []
        self.l_arm_hist = []
        self.r_arm_hist = []
        self.l_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.r_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.gesture_sent = "still"
        self.pointing_arm = None
        self.pointing_vector = None
        self.pointing_yaw_deg = None
        self.pointing_pitch_deg = None
        self.pointing_ray_2d = None
        self.in_view = False
        self.near = False
        self.near_t = self.still_t = None
        self.dwell_sent = False
        self.facing_sent = False
        self.confirmed_sent = False

    def predict(self, t, accel=1.5):
        dt = max(0.0, t - self.t)
        if dt == 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = accel**2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q
        self.t = t

    def update(self, z, sigma):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1
        S = H @ self.P @ H.T + np.eye(2) * sigma**2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1
        self.t_seen = self.t
        self.moved = max(self.moved, float(np.linalg.norm(self.x[:2] - self.start)))

    @property
    def confirmed(self):
        return self.hits >= 3 and (self.camera_ever or self.moved > CONFIRM_MOVE_M)

    def facing(self, now):
        if self.camera_seen_t is None or now - self.camera_seen_t > FACING_HOLD_S:
            return None  # the camera does not see this person now
        return self.facing_t is not None and now - self.facing_t <= FACING_HOLD_S

    def update_mouth(self, openness, dist_m, is_facing, now):
        # Limit talking detection to 2.2m (calibrated distance) to eliminate distant noise
        if dist_m is not None and dist_m > 2.2:
            self.talking = False
            self.mouth_history.clear()
            self.latest_openness = 0.0
            return

        if not is_facing:
            if self.facing_t is None or (now - self.facing_t > 0.4):
                self.talking = False
                self.mouth_history.clear()
                self.latest_openness = 0.0
                return

        self.latest_openness = openness
        self.mouth_history.append((now, openness))
        self.mouth_history = [(t, v) for t, v in self.mouth_history if now - t <= 1.0]

        if len(self.mouth_history) >= 4:
            vals = [v for _, v in self.mouth_history]
            max_v = max(vals)
            min_v = min(vals)
            range_v = max_v - min_v
            vel = sum(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals)))
            std_v = float(np.std(vals))

            diffs = np.diff(vals)
            signs = []
            for d in diffs:
                if abs(d) > 0.025:
                    s = 1 if d > 0 else -1
                    if not signs or signs[-1] != s:
                        signs.append(s)
            reversals = max(0, len(signs) - 1)

            # Speech dynamics: clear mouth opening and syllable reversals
            if max_v >= 0.16 and range_v >= 0.10 and vel >= 0.15 and std_v >= 0.03 and reversals >= 1:
                self.talking_t = now

        # Hold talking status for 0.5s so sentences don't flicker between words
        self.talking = (self.talking_t is not None) and (now - self.talking_t <= 0.5)

    def update_gesture(self, px, vis, world_lms, now):
        sh_w = float(np.linalg.norm(px[11] - px[12])) if (vis[11] > 0.3 and vis[12] > 0.3) else 80.0
        torso_h = 1.3 * sh_w
        if len(vis) > 24 and vis[23] > 0.3 and vis[24] > 0.3:
            torso_h = max(torso_h, float(np.linalg.norm((px[11] + px[12]) / 2 - (px[23] + px[24]) / 2)))
        scale = max(sh_w, 0.7 * torso_h, 20.0)

        l_hip_2d = px[23] if (len(vis) > 23 and vis[23] > 0.3) else (px[11] + np.array([0.0, torso_h]))
        r_hip_2d = px[24] if (len(vis) > 24 and vis[24] > 0.3) else (px[12] + np.array([0.0, torso_h]))

        l_ok = vis[11] > 0.25 and vis[13] > 0.25 and vis[15] > 0.25
        r_ok = vis[12] > 0.25 and vis[14] > 0.25 and vis[16] > 0.25

        sh_both_ok = vis[11] > 0.3 and vis[12] > 0.3
        l_out = (1.0 if px[11][0] >= px[12][0] else -1.0) if sh_both_ok else 1.0
        r_out = -l_out

        # Maintain temporal arm histories for ML classifier
        if l_ok:
            self.l_arm_hist.append({
                "t": now, "sh": px[11].copy(), "elb": px[13].copy(), "wr": px[15].copy(),
                "hip": l_hip_2d.copy(), "scale": scale, "out_sign": l_out,
            })
        else:
            self.l_arm_hist = [h for h in self.l_arm_hist if now - h["t"] <= 0.3]
        self.l_arm_hist = [h for h in self.l_arm_hist if now - h["t"] <= 0.85]

        if r_ok:
            self.r_arm_hist.append({
                "t": now, "sh": px[12].copy(), "elb": px[14].copy(), "wr": px[16].copy(),
                "hip": r_hip_2d.copy(), "scale": scale, "out_sign": r_out,
            })
        else:
            self.r_arm_hist = [h for h in self.r_arm_hist if now - h["t"] <= 0.3]
        self.r_arm_hist = [h for h in self.r_arm_hist if now - h["t"] <= 0.85]

        if _GESTURE_CLF.loaded:
            if l_ok and len(self.l_arm_hist) >= 2:
                lf = extract_arm_features(self.l_arm_hist)
                lp = _GESTURE_CLF.predict_proba(lf)
                self.l_probs = 0.55 * self.l_probs + 0.45 * lp
            else:
                self.l_probs = 0.70 * self.l_probs + 0.30 * np.array([1.0, 0.0, 0.0], dtype=np.float32)

            if r_ok and len(self.r_arm_hist) >= 2:
                rf = extract_arm_features(self.r_arm_hist)
                rp = _GESTURE_CLF.predict_proba(rf)
                self.r_probs = 0.55 * self.r_probs + 0.45 * rp
            else:
                self.r_probs = 0.70 * self.r_probs + 0.30 * np.array([1.0, 0.0, 0.0], dtype=np.float32)

            p_point_l, p_point_r = float(self.l_probs[1]), float(self.r_probs[1])
            p_wave_l, p_wave_r = float(self.l_probs[2]), float(self.r_probs[2])
            p_point = max(p_point_l, p_point_r)
            p_wave = max(p_wave_l, p_wave_r)

            enter_th = 0.55
            exit_th = 0.38
            l_point = l_ok and (p_point_l >= (exit_th if self.gesture == "point" and self.pointing_arm == "left" else enter_th))
            r_point = r_ok and (p_point_r >= (exit_th if self.gesture == "point" and self.pointing_arm == "right" else enter_th))
            point_active = l_point or r_point
            wave_active = (not point_active) and (p_wave >= (exit_th if self.gesture == "wave" else enter_th))
        else:
            # Fallback geometric rules if gesture_mlp.npz is absent
            l_point = False
            if l_ok:
                if _check_point_arm_2d(px[11], px[13], px[15], l_hip_2d, torso_h, sh_w):
                    l_point = True
                elif world_lms is not None and len(world_lms) > 23:
                    hip_3d = world_lms[23] if vis[23] > 0.3 else (world_lms[11] + np.array([0.0, 0.45, 0.0]))
                    if _check_point_arm_3d(world_lms[11], world_lms[13], world_lms[15], hip_3d):
                        l_point = True

            r_point = False
            if r_ok:
                if _check_point_arm_2d(px[12], px[14], px[16], r_hip_2d, torso_h, sh_w):
                    r_point = True
                elif world_lms is not None and len(world_lms) > 24:
                    hip_3d = world_lms[24] if vis[24] > 0.3 else (world_lms[12] + np.array([0.0, 0.45, 0.0]))
                    if _check_point_arm_3d(world_lms[12], world_lms[14], world_lms[16], hip_3d):
                        r_point = True

            point_active = l_point or r_point
            l_bent = l_ok and (np.degrees(np.arccos(np.clip(
                np.dot(px[11] - px[13], px[15] - px[13]) / (np.linalg.norm(px[11] - px[13]) * np.linalg.norm(px[15] - px[13]) + 1e-6), -1.0, 1.0))) < 140.0)
            r_bent = r_ok and (np.degrees(np.arccos(np.clip(
                np.dot(px[12] - px[14], px[16] - px[14]) / (np.linalg.norm(px[12] - px[14]) * np.linalg.norm(px[16] - px[14]) + 1e-6), -1.0, 1.0))) < 140.0)

            if l_ok and not l_point and l_bent:
                self.l_wrist_hist.append((now, float(px[15][0] - px[13][0]) + 0.3 * float(px[15][1] - px[13][1])))
            else:
                self.l_wrist_hist.clear()
            self.l_wrist_hist = [(t, x) for t, x in self.l_wrist_hist if now - t <= 0.9]

            if r_ok and not r_point and r_bent:
                self.r_wrist_hist.append((now, float(px[16][0] - px[14][0]) + 0.3 * float(px[16][1] - px[14][1])))
            else:
                self.r_wrist_hist.clear()
            self.r_wrist_hist = [(t, x) for t, x in self.r_wrist_hist if now - t <= 0.9]

            l_waving = l_bent and not l_point and (px[15][1] < px[13][1] + 25.0) and _is_waving(self.l_wrist_hist, sh_w, min_reversals=2)
            r_waving = r_bent and not r_point and (px[16][1] < px[14][1] + 25.0) and _is_waving(self.r_wrist_hist, sh_w, min_reversals=2)
            wave_active = l_waving or r_waving

        now_point = point_active
        now_wave = wave_active and not point_active

        # Prevent transient wave trigger right after holding a point
        if self.gesture_hold_type == "point" and self.gesture_hold_t is not None and (now - self.gesture_hold_t <= 0.45):
            if not now_point and now_wave:
                now_wave = False

        if now_point:
            if l_point and r_point:
                arm = "right" if float(self.r_probs[1]) >= float(self.l_probs[1]) else "left"
            else:
                arm = "right" if r_point else "left"
            sh_idx = 12 if arm == "right" else 11
            elb_idx = 14 if arm == "right" else 13
            wr_idx = 16 if arm == "right" else 15
            finger_idx = 20 if arm == "right" else 19

            tip_2d = px[finger_idx] if (len(vis) > finger_idx and vis[finger_idx] > 0.35) else px[wr_idx]
            tip_idx = finger_idx if (len(vis) > finger_idx and vis[finger_idx] > 0.35) else wr_idx

            sh_other_idx = 11 if arm == "right" else 12
            hip_2d = r_hip_2d if arm == "right" else l_hip_2d
            v_rob, u2d = estimate_pointing_vector_3d(
                px[sh_idx], px[elb_idx], tip_2d, sh_w, world_lms=world_lms, elb_idx=elb_idx, wr_idx=tip_idx,
                sh_other_2d=px[sh_other_idx], hip_2d=hip_2d, torso_h=torso_h,
            )
            if float(np.linalg.norm(u2d)) < 1e-3:
                u2d = np.array([1.0, 0.0]) if arm == "right" else np.array([-1.0, 0.0])

            ray_len = 180.0
            ray_start = px[wr_idx].tolist()
            ray_end = (px[wr_idx] + u2d * ray_len).tolist()
            self.pointing_ray_2d = {"start": ray_start, "end": ray_end, "dir": [round(float(u2d[0]), 3), round(float(u2d[1]), 3)]}

            yaw_deg = math.degrees(math.atan2(v_rob[1], v_rob[0]))
            pitch_deg = math.degrees(math.atan2(v_rob[2], math.hypot(v_rob[0], v_rob[1])))

            self.pointing_arm = arm
            self.pointing_vector = [round(float(v_rob[0]), 3), round(float(v_rob[1]), 3), round(float(v_rob[2]), 3)]
            self.pointing_yaw_deg = round(yaw_deg, 1)
            self.pointing_pitch_deg = round(pitch_deg, 1)
            self.gesture = "point"
            self.gesture_hold_t = now
            self.gesture_hold_type = "point"
            self.l_wrist_hist.clear()
            self.r_wrist_hist.clear()
        elif now_wave:
            self.gesture = "wave"
            self.gesture_hold_t = now
            self.gesture_hold_type = "wave"
            self.pointing_arm = None
            self.pointing_vector = None
            self.pointing_yaw_deg = None
            self.pointing_pitch_deg = None
            self.pointing_ray_2d = None
        elif self.gesture_hold_t is not None and (now - self.gesture_hold_t <= (0.45 if self.gesture_hold_type == "point" else 0.55)):
            self.gesture = self.gesture_hold_type
        else:
            self.gesture = "still"
            self.gesture_hold_type = None
            self.pointing_arm = None
            self.pointing_vector = None
            self.pointing_yaw_deg = None
            self.pointing_pitch_deg = None
            self.pointing_ray_2d = None

    def facing_state(self, now, speed, radial):
        """Granular semantic facing+motion state: walking_facing, walking_past_facing, standing_facing, walking_past, etc."""
        f = self.facing(now)
        is_moving = speed > STILL_MPS
        is_approaching = radial < -0.25
        is_receding = radial > 0.25
        if f is True:
            if is_approaching:
                return "walking_facing"
            if is_moving:
                return "walking_past_facing"
            return "standing_facing"
        if f is False:
            if is_approaching:
                return "walking_approaching"
            if is_receding:
                return "walking_away"
            if is_moving:
                return "walking_past"
            return "standing_away"
        if is_approaching:
            return "lidar_approaching"
        if is_moving:
            return "lidar_moving"
        return "lidar_still"

    def summary(self, now):
        x, y, vx, vy = self.x
        ahead = min(max(now - self.t, 0.0), 0.4)  # report where the person is now (measurements are ~0.1 s old)
        x, y = x + vx * ahead, y + vy * ahead
        r = math.hypot(x, y)
        speed = math.hypot(vx, vy)
        radial = (x * vx + y * vy) / r if r > 1e-3 else 0.0
        cam_active = bool(self.camera_seen_t and now - self.camera_seen_t <= 0.8)
        gest = self.gesture if cam_active else "still"
        is_pt = (gest == "point")
        return {
            "id": self.id, "x": round(x, 3), "y": round(y, 3), "distance_m": round(r, 3),
            "gap_m": None if self.gap is None else round(self.gap, 3),
            "bearing_deg": round(math.degrees(math.atan2(y, x)), 1),
            "speed_mps": round(speed, 3), "radial_mps": round(radial, 3),
            "vx": round(vx, 3), "vy": round(vy, 3),
            "heading_deg": round(math.degrees(math.atan2(vy, vx)), 1) if speed > 0.1 else None,
            "moving": speed > STILL_MPS, "approaching": radial < -0.25,
            "facing": self.facing(now),
            "facing_state": self.facing_state(now, speed, radial),
            "talking": bool(self.talking and cam_active and (r <= 2.2) and self.facing(now)),
            "gesture": gest,
            "pointing_arm": self.pointing_arm if is_pt else None,
            "pointing_vector": self.pointing_vector if is_pt else None,
            "pointing_yaw_deg": self.pointing_yaw_deg if is_pt else None,
            "pointing_pitch_deg": self.pointing_pitch_deg if is_pt else None,
            "pointing_ray_2d": self.pointing_ray_2d if is_pt else None,
            "head_yaw_deg": self.head_yaw, "body_yaw_deg": self.body_yaw,
            "gender": self.gender, "age": self.age,
            "face_score": self.face_score, "in_camera_view": self.in_view, "seen_by_camera": self.camera_ever,
            "source": self.source, "confirmed": self.confirmed,
            "near": self.near, "near_s": round(now - self.near_t, 2) if self.near_t else 0.0,
            "still_s": round(now - self.still_t, 2) if self.still_t else 0.0,
            "age_s": round(now - self.t_born, 2), "last_seen_s": round(now - self.t_seen, 2),
        }  # fmt: skip


class Events:
    """Numbered event log; wait(after, timeout) returns the events with seq > after (long poll)."""

    def __init__(self, keep=500):
        self.cv = threading.Condition()
        self.items, self.seq, self.keep = [], 0, keep

    def emit(self, kind, **data):
        with self.cv:
            self.seq += 1
            ev = {"seq": self.seq, "type": kind, "time": round(time.time(), 3), **data}
            self.items.append(ev)
            del self.items[:-self.keep]
            self.cv.notify_all()
        print(f"[vision] event {kind} " + " ".join(f"{k}={v}" for k, v in data.items() if k != "person"),
              (f"id={data['person']['id']} d={data['person']['distance_m']} facing={data['person']['facing']}"
               if "person" in data else ""), flush=True)  # fmt: skip
        return ev

    def wait(self, after, timeout):
        end = time.time() + timeout
        with self.cv:
            while True:
                new = [e for e in self.items if e["seq"] > after]
                left = end - time.time()
                if new or left <= 0:
                    return new, self.seq
                self.cv.wait(left)


class Tracker:
    def __init__(self, events):
        self.tracks = []
        self.events = events
        self.close = {"active": False, "since": None, "clear_t": None}

    def step(self, now, meas, cam=None, lidar_active=False):
        """meas: list of dicts {xy, sigma, source, det?, gap?}. Updates tracks, emits events."""
        for tr in self.tracks:
            tr.predict(now)
        pairs = []
        for i, tr in enumerate(self.tracks):
            S = tr.P[:2, :2]
            for j, m in enumerate(meas):
                e = np.asarray(m["xy"]) - tr.x[:2]
                if m["source"] == "camera":  # image-only range is poor: weigh the error across the ray more
                    r = np.asarray(m["xy"]) / max(np.linalg.norm(m["xy"]), 1e-3)
                    e = np.array([0.35 * (e @ r), e @ np.array([-r[1], r[0]])])
                d = float(np.linalg.norm(e))
                cov = S + np.eye(2) * m["sigma"] ** 2
                maha = float(e @ np.linalg.solve(cov, e)) if m["source"] != "camera" else (d / 0.5) ** 2
                max_d = 1.2
                if d < max_d and maha < 20.0:  # 4-5 sigma
                    pairs.append((maha, i, j))
        used_t, used_m = set(), set()
        for d, i, j in sorted(pairs):
            if i in used_t or j in used_m:
                continue
            used_t.add(i)
            used_m.add(j)
            self._apply(self.tracks[i], meas[j], now)
        for j, m in enumerate(meas):
            if j in used_m:
                continue
            if m["source"] == "camera" and lidar_active:
                # the lidar would see a real person: an image-only position (rough range) must not start a second
                # track; give its facing etc. to the track in the same direction, if any
                b = math.atan2(m["xy"][1], m["xy"][0])
                tr = min(self.tracks, key=lambda t: abs(math.remainder(math.atan2(t.x[1], t.x[0]) - b, math.tau)),
                         default=None)
                if tr is not None and abs(math.remainder(math.atan2(tr.x[1], tr.x[0]) - b, math.tau)) < math.radians(8):
                    self._apply(tr, m, now, new=True)  # attributes only, no position update
                continue
            if m["source"] != "lidar" or np.hypot(*m["xy"]) < LIDAR_ONLY_M:
                tr = Track(m["xy"], now, m["sigma"])
                self._apply(tr, m, now, new=True)
                self.tracks.append(tr)
        for tr in self.tracks:
            if cam is not None:
                tr.in_view = cam.in_view(tr.x[0], tr.x[1])
        keep = []
        for tr in self.tracks:
            timeout = (1.5 if tr.hits > 3 else 0.8) if not lidar_active else (2.0 if tr.hits > 3 else 0.8)
            lost = now - tr.t_seen > timeout
            if lost:
                if tr.confirmed_sent:
                    self.events.emit("person_left", id=tr.id, person=tr.summary(now))
                continue
            keep.append(tr)
        self.tracks = keep
        for tr in self.tracks:
            self._events(tr, now)

    def _apply(self, tr, m, now, new=False):
        if not new:
            tr.update(m["xy"], m["sigma"])
            tr.source = m["source"]
        elif not tr.source:
            tr.source = m["source"]
        tr.gap = m.get("gap", tr.gap if m["source"] == "camera" else None)
        det = m.get("det")
        if det is not None:
            det["track"] = tr.id
            tr.camera_seen_t = now
            tr.camera_ever = True
            tr.head_yaw, tr.body_yaw, tr.face_score = det["head_yaw_deg"], det["body_yaw_deg"], det["face_score"]
            if det["facing"]:
                tr.facing_t = now
            if det.get("mouth_openness") is not None:
                dist_m = float(np.hypot(tr.x[0], tr.x[1]))
                tr.update_mouth(det["mouth_openness"], dist_m, det["facing"], now)
            if det.get("px") is not None and det.get("vis") is not None:
                tr.update_gesture(det["px"], det["vis"], det.get("world_lms"), now)

    def _events(self, tr, now):
        s = tr.summary(now)
        if not tr.confirmed:
            return
        if not tr.confirmed_sent:
            tr.confirmed_sent = True
            self.events.emit("person_new", id=tr.id, person=s)
        r = s["distance_m"]
        if not tr.near and r < NEAR_M:
            tr.near, tr.near_t, tr.dwell_sent = True, now, False
            s = tr.summary(now)
            self.events.emit("person_near", id=tr.id, person=s)
        elif tr.near and r > NEAR_EXIT_M:
            tr.near, tr.near_t, tr.still_t = False, None, None
            self.events.emit("person_far", id=tr.id, person=tr.summary(now))
        if tr.near:
            if s["speed_mps"] < STILL_MPS:
                tr.still_t = tr.still_t or now
            else:
                tr.still_t = None
            if tr.still_t and now - tr.still_t >= DWELL_S and not tr.dwell_sent:
                tr.dwell_sent = True
                self.events.emit("person_dwell", id=tr.id, person=tr.summary(now))
        f = s["facing"]
        if f and r < ENGAGE_M and not tr.facing_sent:
            tr.facing_sent = True
            self.events.emit("person_facing", id=tr.id, person=s)
        elif f is False and tr.facing_sent:
            tr.facing_sent = False
            self.events.emit("person_not_facing", id=tr.id, person=s)
        is_talk = s["talking"]
        if is_talk and not tr.talking_sent:
            tr.talking_sent = True
            self.events.emit("person_talking", id=tr.id, person=s)
        elif not is_talk and tr.talking_sent:
            tr.talking_sent = False
            self.events.emit("person_stopped_talking", id=tr.id, person=s)
        g = s.get("gesture", "still")
        if g != tr.gesture_sent:
            prev = tr.gesture_sent
            tr.gesture_sent = g
            if g == "wave":
                self.events.emit("person_wave", id=tr.id, person=s)
            elif g == "point":
                self.events.emit("person_point", id=tr.id, pointing_arm=s.get("pointing_arm"),
                                 pointing_vector=s.get("pointing_vector"),
                                 pointing_yaw_deg=s.get("pointing_yaw_deg"),
                                 pointing_pitch_deg=s.get("pointing_pitch_deg"), person=s)
            elif prev in ("wave", "point"):
                self.events.emit("person_gesture_stopped", id=tr.id, person=s)

    def proximity(self, sectors, now):
        """too_close / clear from the nearest lidar point per sector (robot removed)."""
        near = {k: v for k, v in sectors.items() if v is not None and v < CLOSE_M}
        c = self.close
        if near and not c["active"]:
            c.update(active=True, since=now, clear_t=None)
            self.events.emit("too_close", sectors=near)
        elif c["active"]:
            if all(v is None or v > CLEAR_M for v in sectors.values()):
                c["clear_t"] = c["clear_t"] or now
                if now - c["clear_t"] >= CLEAR_S:
                    c.update(active=False, since=None, clear_t=None)
                    self.events.emit("clear")
            else:
                c["clear_t"] = None
        return {"too_close": c["active"], "nearest_m": sectors, "close_m": CLOSE_M}

    def people(self, now, confirmed_only=False):
        return [tr.summary(now) for tr in sorted(self.tracks, key=lambda t: np.hypot(*t.x[:2]))
                if tr.confirmed or not confirmed_only]
