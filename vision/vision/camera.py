"""Camera sources: one background thread keeps the newest frame (BGR) and its time.

    webcam:0                                  a local camera through OpenCV (laptop webcam, USB camera)
    http://HOST:PORT/color.jpg                JPEG snapshots (exp04 camera_streamer.py on PC2, the exp05 sensor sim)
    unitree:http://192.168.123.164:8000/capture   Unitree's camera server on PC2 ({"image_base64": JPEG})
    file:VIDEO.mp4                            a recorded video, played at its own frame rate (loops)
"""

import base64
import http.client
import json
import threading
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np


class Camera:
    def __init__(self, spec, max_fps=15.0):
        self.spec, self.max_fps = spec, max_fps
        self.lock = threading.Lock()
        self.frame, self.t, self.count, self.error = None, 0.0, 0, ""
        self.fps = 0.0
        threading.Thread(target=self._run, name="camera", daemon=True).start()

    def latest(self):
        with self.lock:
            return self.frame, self.t, self.count

    def _set(self, img):
        with self.lock:
            self.frame, self.t = img, time.time()
            self.count += 1
            self.error = ""

    def _run(self):
        kind, _, rest = self.spec.partition(":")
        if kind == "webcam":
            self._run_cv(int(rest), live=True)
        elif kind == "file":
            self._run_cv(rest, live=False)
        elif kind in ("http", "https"):
            self._run_http(self.spec, unitree=False)
        elif kind == "unitree":
            self._run_http(rest, unitree=True)
        else:
            self.error = f"unknown camera source {self.spec!r}"

    def _tick_fps(self, state):
        n, t0 = state
        n += 1
        if time.time() - t0 > 2.0:
            self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
        return n, t0

    def _run_cv(self, src, live):
        state = (0, time.time())
        while True:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self.error = f"cannot open {src}"
                time.sleep(2.0)
                continue
            if live:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 14.0)
                period = 1.0 / (self.max_fps if (self.max_fps and self.max_fps > 0) else 14.0)
            else:
                period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
            t_next = time.time()
            while True:
                t_start = time.time()
                ok, img = cap.read()
                if not ok:
                    break  # end of the file: loop; camera lost: reopen
                if img.shape[1] != 640 or img.shape[0] != 480:
                    img = cv2.resize(img, (640, 480))
                self._set(img)
                state = self._tick_fps(state)
                if live:
                    time.sleep(max(0.0, period - (time.time() - t_start)))
                else:
                    t_next += period
                    time.sleep(max(0.0, t_next - time.time()))
            cap.release()
            if live:
                self.error = "camera stopped, reopening"
                time.sleep(1.0)

    def _run_http(self, url, unitree):
        state = (0, time.time())
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path + ("?" + parsed.query if parsed.query else "")
        if not path:
            path = "/"

        conn = None
        while True:
            t0 = time.time()
            try:
                if conn is None:
                    if parsed.scheme == "https":
                        import ssl
                        conn = http.client.HTTPSConnection(host, port, timeout=2.0, context=ssl._create_unverified_context())
                    else:
                        conn = http.client.HTTPConnection(host, port, timeout=2.0)
                conn.request("GET", path)
                resp = conn.getresponse()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                body = resp.read()
                if unitree:
                    body = base64.b64decode(json.loads(body)["image_base64"])
                img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("not an image")
                self._set(img)
                state = self._tick_fps(state)
            except Exception as e:  # noqa: BLE001 (network, JSON, decode: keep trying)
                self.error = f"{type(e).__name__}: {e}"[:200]
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                time.sleep(0.05)
            sleep_t = max(0.0, 1.0 / self.max_fps - (time.time() - t0))
            if sleep_t > 0:
                time.sleep(sleep_t)
