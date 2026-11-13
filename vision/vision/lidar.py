"""Head lidar (Livox MID-360) for the vision module: UDP point packets -> 0.1 s frames -> people-sized clusters.

Packets (Livox SDK2, same format as exp04/g1/deploy/head_sensors.py, copied here so exp05 stays self-contained):
    header 36 bytes: version u8, length u16, time_interval u16, dot_num u16, udp_cnt u16, frame_cnt u8,
                     data_type u8, time_type u8, reserved 12, crc32 u32, timestamp 8
    data_type 1: per point int32 x, y, z (mm), uint8 reflectivity, uint8 tag      (14 bytes)
    data_type 2: per point int16 x, y, z (cm), uint8 reflectivity, uint8 tag      (8 bytes)
On the real G1 the lidar sends to PC2; exp04's lidar_relay.sh copies the packets to this laptop's port 56301.
The sim (vision/sim_sensors.py) sends the same packets from a raycast of the scene.
"""

import socket
import struct
import threading
import time

import cv2
import numpy as np

HEADER = struct.Struct("<BHHHHBBB12sI8s")  # 36 bytes
POINT_DTYPES = {1: (np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("r", "u1"), ("t", "u1")]), 0.001),
                2: (np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("r", "u1"), ("t", "u1")]), 0.01)}
REAL_PORT, SIM_PORT = 56301, 56305  # real (exp04 relay target) / exp05 sensor sim
FRAME_S = 0.1
BODY_HALF_DEPTH = 0.08  # m: torso half depth (centre of a person behind its visible surface)


def pack_points(xyz_m, udp_cnt, frame_cnt, ts_ns=None):
    """One data_type 1 packet (used by the sensor sim)."""
    n = len(xyz_m)
    body = np.zeros(n, dtype=POINT_DTYPES[1][0])
    mm = np.round(np.asarray(xyz_m) * 1000).astype(np.int32)
    body["x"], body["y"], body["z"], body["r"] = mm[:, 0], mm[:, 1], mm[:, 2], 100
    head = HEADER.pack(0, HEADER.size + body.nbytes, 0, n, udp_cnt & 0xFFFF, frame_cnt & 0xFF, 1, 0, b"\0" * 12, 0,
                       struct.pack("<Q", time.time_ns() if ts_ns is None else ts_ns))  # fmt: skip
    return head + body.tobytes()


def parse_packets(packets):
    """Points (N, 3) in metres (lidar frame) from many packets at once; (0,0,0) = no return, dropped."""
    bufs = {1: [], 2: []}
    for data in packets:
        if len(data) < HEADER.size or data[10] not in bufs:  # data_type at byte 10
            continue
        dt = POINT_DTYPES[data[10]][0]
        n = min(struct.unpack_from("<H", data, 5)[0], (len(data) - HEADER.size) // dt.itemsize)
        bufs[data[10]].append(data[HEADER.size:HEADER.size + n * dt.itemsize])
    out = [np.zeros((0, 3), np.float32)]
    for kind, parts in bufs.items():
        if parts:
            dt, scale = POINT_DTYPES[kind]
            a = np.frombuffer(b"".join(parts), dtype=dt)
            p = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float32) * scale
            out.append(p[np.any(p != 0, axis=1)])
    return np.concatenate(out)


class LidarReceiver:
    """Receives packets on a UDP port; latest() = (time, points (N,3) lidar frame) of the last full 0.1 s frame."""

    def __init__(self, port=REAL_PORT, bind="0.0.0.0", frame_s=FRAME_S):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.sock.bind((bind, port))
        self.sock.settimeout(0.2)
        self.port, self.frame_s = port, frame_s
        self.lock = threading.Lock()
        self.frame = (0.0, np.zeros((0, 3), np.float32))
        self.packets = self.frames = 0
        self.rate = 0.0  # points per second
        threading.Thread(target=self._run, name="lidar", daemon=True).start()

    def _run(self):
        buf, t0 = [], time.time()
        n_pts, t_rate = 0, time.time()
        while True:
            try:
                buf.append(self.sock.recv(65535))
                self.packets += 1
            except socket.timeout:
                pass
            now = time.time()
            if now - t0 >= self.frame_s:
                pts = parse_packets(buf) if buf else np.zeros((0, 3), np.float32)
                buf, t0 = [], now
                with self.lock:
                    self.frame = (now, pts)
                    self.frames += 1
                n_pts += len(pts)
                if now - t_rate > 2.0:
                    self.rate, n_pts, t_rate = n_pts / (now - t_rate), 0, now

    def latest(self):
        with self.lock:
            return self.frame


class LidarPeople:
    """Lidar points (robot frame, robot removed) -> clusters on a 2D grid; person-like ones get a flag.

    Kept: points 0.25-2.1 m above the floor (legs to head; the floor and low clutter go). Cells of 5 cm, dilated by
    one cell so a person's two legs / torso become one cluster. A person seen by the MID-360 (upside down at ~1.3 m:
    elevation +7..-52°) shows up from ~0.2 m away (torso) to > 10 m (legs, then the whole body).
    """

    def __init__(self, max_range=10.0, cell=0.05, z_min=0.25, z_max=2.1, min_points=6, bg_s=10.0, frame_s=FRAME_S):
        self.max_range, self.cell, self.z_min, self.z_max, self.min_points = max_range, cell, z_min, z_max, min_points
        self.n = int(2 * max_range / cell)
        # static background (walls, booths, poles): 10 cm cells, running average of "occupied" over ~bg_s.
        # The MID-360 pattern is random, so a static cell is hit in only part of the frames: threshold BG_THRESHOLD.
        self.nb = self.n // 2
        self.bg = np.zeros((self.nb, self.nb), np.float32)
        self.bg_frames, self.bg_rate = 0, frame_s / bg_s

    def _learn(self, ijb, freeze_xy):
        """Running average of occupancy per 10 cm cell (not under confirmed people)."""
        occ = np.zeros((self.nb, self.nb), np.float32)
        occ[ijb[:, 1], ijb[:, 0]] = 1.0
        occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
        a = max(1.0 / (self.bg_frames + 1), self.bg_rate)
        upd = self.bg * (1 - a) + occ * a
        if len(freeze_xy):
            mask = np.zeros((self.nb, self.nb), np.uint8)
            for x, y in freeze_xy:
                c = (int((x + self.max_range) / (2 * self.cell)), int((y + self.max_range) / (2 * self.cell)))
                cv2.circle(mask, c, int(0.6 / (2 * self.cell)), 1, -1)
            upd[mask > 0] = self.bg[mask > 0]
        self.bg = upd
        self.bg_frames += 1

    def _cluster(self, p, ij, is_static):
        """Connected 5 cm cells (dilated by one) of one point set -> cluster dicts."""
        grid = np.zeros((self.n, self.n), np.uint8)
        grid[ij[:, 1], ij[:, 0]] = 1
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
        n_lab, lab = cv2.connectedComponents(grid, connectivity=8)
        lab_pt = lab[ij[:, 1], ij[:, 0]]
        order = np.argsort(lab_pt, kind="stable")
        bounds = np.searchsorted(lab_pt[order], np.arange(1, n_lab + 1))
        out = []
        for k in range(1, n_lab):
            sel = order[bounds[k - 1]:bounds[k]]
            if len(sel) < self.min_points:
                continue
            c = p[sel]
            xy = c[:, :2]
            rr = np.hypot(xy[:, 0], xy[:, 1])
            ext = xy.max(axis=0) - xy.min(axis=0)
            size = float(np.hypot(*ext))
            if size > 1.5:  # walls, shelves, booths: only what the proximity check needs
                centre = xy.mean(axis=0)
                az_c = math_atan2(centre)
                out.append({"x": float(centre[0]), "y": float(centre[1]), "range": float(np.hypot(*centre)),
                            "near": float(rr.min()), "bearing": float(np.degrees(az_c)), "bearing_lo": None,
                            "bearing_hi": None, "size": size, "z_top": None, "z_low": None, "n": int(len(sel)),
                            "person_like": False, "static": is_static})  # fmt: skip
                continue
            centre = np.array([_q(xy[:, 0], 0.5), _q(xy[:, 1], 0.5)])
            # the lidar sees the near side of a body: the centre is ~BODY_HALF_DEPTH further along the ray
            rc = float(np.hypot(*centre))
            if rc > 0.05:
                centre = centre * (rc + BODY_HALF_DEPTH) / rc
            az = np.arctan2(xy[:, 1], xy[:, 0])
            az_c = math_atan2(centre)
            daz = np.angle(np.exp(1j * (az - az_c)))
            out.append({
                "x": float(centre[0]), "y": float(centre[1]), "range": float(np.hypot(*centre)),
                "near": _q(rr, 0.05), "bearing": float(np.degrees(az_c)),
                "bearing_lo": float(np.degrees(az_c + daz.min())), "bearing_hi": float(np.degrees(az_c + daz.max())),
                "size": size, "z_top": _q(c[:, 2], 0.95), "z_low": float(c[:, 2].min()), "n": int(len(sel)),
                "person_like": bool(0.08 <= size <= 1.1 and float(c[:, 2].max()) > 0.7), "static": is_static,
            })  # fmt: skip
        return out

    def clusters(self, pts, freeze_xy=()):
        """pts (N,3) in the robot frame (self-filtered). Returns a list of dicts and the kept points.
        freeze_xy: positions of confirmed people (the background does not learn within 0.6 m of them)."""
        r = np.hypot(pts[:, 0], pts[:, 1])
        keep = (pts[:, 2] > self.z_min) & (pts[:, 2] < self.z_max) & (r < self.max_range - self.cell) & (r > 0.15)
        p = pts[keep]
        if not len(p):
            return [], p
        ij = np.floor((p[:, :2] + self.max_range) / self.cell).astype(int)
        ijb = ij // 2
        static_pt = self.bg[ijb[:, 1], ijb[:, 0]] > BG_THRESHOLD  # before this frame is learned
        self._learn(ijb, freeze_xy)
        # people next to walls/shelves: cluster the points on the learned background separately, so they do not
        # merge a person standing beside a booth into the booth
        out = []
        for is_static in (False, True):
            sub = np.flatnonzero(static_pt == is_static)
            if len(sub):
                out += self._cluster(p[sub], ij[sub], is_static)
        return out, p


BG_THRESHOLD = 0.3


def _q(a, f):
    """Quantile f of a 1-D array (no interpolation; np.percentile is ~10x slower for small arrays)."""
    k = int(f * (len(a) - 1))
    return float(np.partition(a, k)[k])


def math_atan2(xy):
    return float(np.arctan2(xy[1], xy[0]))


def sector_min_range(pts, sectors=(("front", -45, 45), ("left", 45, 135), ("back", 135, 225), ("right", 225, 315))):
    """Nearest horizontal range per sector (bearing deg, 0 = ahead, positive to the left)."""
    if not len(pts):
        return {s[0]: None for s in sectors}
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360
    r = np.hypot(pts[:, 0], pts[:, 1])
    out = {}
    for name, lo, hi in sectors:
        lo, hi = lo % 360, hi % 360
        sel = (az >= lo) & (az < hi) if lo < hi else (az >= lo) | (az < hi)
        out[name] = round(float(r[sel].min()), 3) if sel.any() else None
    return out
