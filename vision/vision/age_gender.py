import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parents[1] / "models"

GENDERS = ["Male", "Female"]
FAIRFACE_AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AgeGenderEstimator:
    """Async, instance-bound age and gender estimator.

    Uses FairFace for gender classification and MiVOLO v2 for fine-grained continuous age
    estimation (especially detailed in the 1-20 range). Each prediction is strictly keyed by
    track ID so that results from an old instance cannot contaminate a new instance row.
    """

    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.fairface, self.ff_in = None, None
        ff_path = MODELS / "fairface.onnx"
        if ff_path.exists():
            try:
                self.fairface = ort.InferenceSession(str(ff_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self.ff_in = self.fairface.get_inputs()[0].name
                print("[vision] FairFace loaded for gender classification", flush=True)
            except Exception as e:
                print(f"[vision] FairFace load error: {e}", flush=True)

        self.mivolo, self.mv_in = None, None
        mv_path = MODELS / "mivolo_v2_age.onnx"
        if mv_path.exists():
            try:
                self.mivolo = ort.InferenceSession(str(mv_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self.mv_in = self.mivolo.get_inputs()[0].name
                print("[vision] MiVOLO loaded for detailed age estimation", flush=True)
            except Exception as e:
                print(f"[vision] MiVOLO load error: {e}", flush=True)

        self.queue = queue.Queue(maxsize=3)
        self.lock = threading.Lock()
        self.results = {}  # {track_id: {"gender": str, "age": str, "age_val": float, "updated_at": float}}
        self.history = {}  # {track_id: [float]}
        self.pending = set()

        self.worker_thread = threading.Thread(target=self._worker, name="age_gender_worker", daemon=True)
        self.worker_thread.start()

    def submit(self, track_id, face_bgr, body_bgr=None):
        if track_id is None or face_bgr is None or face_bgr.size == 0:
            return
        with self.lock:
            if track_id in self.pending:
                return  # already in queue or processing
            if self.queue.full():
                try:
                    dropped_id, _, _ = self.queue.get_nowait()
                    self.pending.discard(dropped_id)
                except queue.Empty:
                    pass
            self.pending.add(track_id)
            self.queue.put((track_id, face_bgr.copy(), None if body_bgr is None else body_bgr.copy()))

    def get(self, track_id):
        """Strictly get prediction for this instance ID. Returns (gender, age) or (None, None)."""
        if track_id is None:
            return None, None
        with self.lock:
            res = self.results.get(track_id)
            if res is not None:
                return res["gender"], res["age"]
            return None, None

    def purge_tracks(self, active_track_ids):
        """Remove instance data for any tracks that are no longer active."""
        active = set(active_track_ids)
        with self.lock:
            dead = [tid for tid in self.results if tid not in active]
            for tid in dead:
                self.results.pop(tid, None)
                self.history.pop(tid, None)
                self.pending.discard(tid)

    def _worker(self):
        while True:
            try:
                task = self.queue.get()
                if task is None:
                    break
                track_id, face_bgr, body_bgr = task
                try:
                    gender, age_str, age_val = self._predict(face_bgr, body_bgr, track_id)
                    with self.lock:
                        if gender is not None or age_str is not None:
                            self.results[track_id] = {
                                "gender": gender,
                                "age": age_str,
                                "age_val": age_val,
                                "updated_at": time.time(),
                            }
                finally:
                    with self.lock:
                        self.pending.discard(track_id)
                    self.queue.task_done()
            except Exception:
                time.sleep(0.01)

    def _prep_mivolo(self, patch):
        if patch is None or patch.size == 0:
            patch = np.zeros((384, 384, 3), dtype=np.uint8)
        ph, pw = patch.shape[:2]
        s = 384.0 / max(ph, pw)
        nw, nh = max(1, int(round(pw * s))), max(1, int(round(ph * s)))
        r = cv2.resize(patch, (nw, nh))
        canvas = np.zeros((384, 384, 3), dtype=np.uint8)
        canvas[(384 - nh) // 2 : (384 - nh) // 2 + nh, (384 - nw) // 2 : (384 - nw) // 2 + nw] = r
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(norm, (2, 0, 1))

    def _predict(self, face_bgr, body_bgr, track_id):
        gender = None
        age_str = None
        age_val = None

        # 1. FairFace for gender
        if self.fairface is not None:
            try:
                rgb224 = cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
                norm224 = (rgb224 - IMAGENET_MEAN) / IMAGENET_STD
                blob = np.transpose(norm224, (2, 0, 1))[np.newaxis, ...]
                out_ff = self.fairface.run(None, {self.ff_in: blob})[0][0]
                g_idx = int(np.argmax(out_ff[7:9]))
                gender = GENDERS[g_idx]
                a_idx = int(np.argmax(out_ff[9:18]))
                age_str = FAIRFACE_AGES[a_idx]
            except Exception:
                pass

        # 2. MiVOLO for fine-grained continuous age
        if self.mivolo is not None:
            try:
                face_in = self._prep_mivolo(face_bgr)
                body_in = self._prep_mivolo(body_bgr if body_bgr is not None else face_bgr)
                inp = np.concatenate([face_in, body_in], axis=0)[np.newaxis, ...]
                out_mv = self.mivolo.run(None, {self.mv_in: inp})
                raw_age = float(out_mv[0][0][0])
                if raw_age > 0:
                    with self.lock:
                        hist = self.history.setdefault(track_id, [])
                        hist.append(raw_age)
                        if len(hist) > 5:
                            hist.pop(0)
                        smoothed_age = float(np.median(hist))
                    age_val = smoothed_age
                    if 0 < smoothed_age <= 21.0:
                        age_str = f"{smoothed_age:.1f}"
                    else:
                        age_str = f"{round(smoothed_age)}"
            except Exception:
                pass

        return gender, age_str, age_val
