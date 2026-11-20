"""People in a camera image: MediaPipe pose landmarker (several people) + YuNet face on the head crop.

For each person: the image box, the torso point, a rough distance from the image alone (feet on the floor, else torso
length), facing (a frontal face found on the upscaled head crop, head turned < FACING_YAW_DEG), the head yaw from the
face landmarks and the body yaw from the 3D pose. The pose model alone cannot tell a back from a front reliably;
the face can (no face seen from behind or from the side).
"""

import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

MODELS = Path(__file__).resolve().parents[1] / "models"
FACING_YAW_DEG = 20.0
NOSE_DEPTH = 0.47  # nose tip forward of the eye line / eye distance: yaw = atan(nose offset / eye dist / 0.47)
TORSO_M = 0.50  # shoulder centre to hip centre of an adult (1.70 m)
ANKLE_Z = 0.08
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK, NOSE = 11, 12, 23, 24, 27, 28, 0


def _box_iou(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# Mapping from COCO 17 keypoints -> MediaPipe 33 pose landmark indices
_COCO_TO_MP = (
    (0, 0),    # nose
    (1, 2),    # left_eye
    (2, 5),    # right_eye
    (3, 7),    # left_ear
    (4, 8),    # right_ear
    (5, 11),   # left_shoulder
    (6, 12),   # right_shoulder
    (7, 13),   # left_elbow
    (8, 14),   # right_elbow
    (9, 15),   # left_wrist
    (10, 16),  # right_wrist
    (11, 23),  # left_hip
    (12, 24),  # right_hip
    (13, 25),  # left_knee
    (14, 26),  # right_knee
    (15, 27),  # left_ankle
    (16, 28),  # right_ankle
)


class PersonDetector:
    def __init__(self, model="full", max_people=35, gpu=True, min_conf=0.65, backend="auto"):
        self.rcnn = None
        self.rcnn_device = None
        self.landmarker, self.delegate = None, ""
        self.min_conf = min_conf
        self.max_people = max_people

        # Prefer TorchVision Keypoint R-CNN (BSD-3-Clause) + fine-tuned crowded heads
        if backend in ("auto", "rcnn"):
            try:
                import torch
                from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

                self.rcnn_device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
                if self.rcnn_device.type == "cuda":
                    torch.backends.cudnn.benchmark = True
                self.rcnn = keypointrcnn_resnet50_fpn(
                    weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
                    min_size=400,
                    max_size=560,
                    box_score_thresh=min_conf,
                    box_nms_thresh=0.55,
                    box_detections_per_img=max_people,
                    rpn_pre_nms_top_n_test=200,
                    rpn_post_nms_top_n_test=200,
                )
                heads_pt = MODELS / "crowd_keypointrcnn_heads.pt"
                if heads_pt.exists():
                    head_sd = torch.load(heads_pt, map_location="cpu", weights_only=True)
                    self.rcnn.load_state_dict({k: v.float() for k, v in head_sd.items()}, strict=False)
                    tag = "Crowd-KeypointRCNN"
                else:
                    tag = "KeypointRCNN"
                self.rcnn.to(self.rcnn_device).eval()
                if self.rcnn_device.type == "cuda":
                    with torch.no_grad(), torch.amp.autocast("cuda"):
                        self.rcnn([torch.zeros(3, 480, 640, device=self.rcnn_device)])
                self.delegate = f"{tag}-{self.rcnn_device.type.upper()}"
                print(f"[vision] Loaded {tag} (BSD-3-Clause) on {self.delegate}", flush=True)
            except Exception as e:
                print(f"[vision] KeypointRCNN backend unavailable, falling back to MediaPipe: {e}", flush=True)
                self.rcnn = None

        if self.rcnn is None:
            path = str(MODELS / f"pose_landmarker_{model}.task")
            for deleg in ([BaseOptions.Delegate.GPU] if gpu else []) + [BaseOptions.Delegate.CPU]:
                try:
                    opts = mpv.PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=path, delegate=deleg),
                        running_mode=mpv.RunningMode.VIDEO, num_poses=max_people,
                        min_pose_detection_confidence=0.30, min_pose_presence_confidence=0.30,
                        min_tracking_confidence=0.30)  # fmt: skip
                    self.landmarker = mpv.PoseLandmarker.create_from_options(opts)
                    self.delegate = f"MP-{deleg.name}"
                    break
                except Exception as e:  # noqa: BLE001 (no GPU delegate on this machine: use the CPU)
                    print(f"[vision] pose landmarker {deleg.name} not available: {e}", flush=True)

        self.face = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (160, 160), 0.6)
        self.face_mesh = None
        fl_path = MODELS / "face_landmarker.task"
        if fl_path.exists():
            try:
                opts_fl = mpv.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(fl_path), delegate=BaseOptions.Delegate.CPU),
                    running_mode=mpv.RunningMode.IMAGE,
                    output_face_blendshapes=True,
                    num_faces=1,
                )
                self.face_mesh = mpv.FaceLandmarker.create_from_options(opts_fl)
                print("[vision] FaceLandmarker loaded for talking detection", flush=True)
            except Exception as e:
                print(f"[vision] FaceLandmarker load error: {e}", flush=True)
        self.t_last = 0
        self.ms = 0.0

    def detect(self, bgr):
        """List of detections (dicts with pixel quantities) for one BGR frame."""
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        candidates = []

        if self.rcnn is not None:
            import torch

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(rgb).to(self.rcnn_device).permute(2, 0, 1).float() / 255.0
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(self.rcnn_device.type == "cuda")):
                pred = self.rcnn([img_t])[0]

            keep = (pred["labels"] == 1) & (pred["scores"] >= self.min_conf)
            boxes = pred["boxes"][keep].cpu().numpy()
            scores = pred["scores"][keep].cpu().numpy()
            kpts = pred["keypoints"][keep].cpu().numpy()  # (N, 17, 3) -> x, y, visibility
            kpt_scores = pred["keypoints_scores"][keep].cpu().numpy() if "keypoints_scores" in pred else None

            for idx, (box, score, kp) in enumerate(zip(boxes, scores, kpts)):
                px = np.zeros((33, 2), dtype=float)
                vis = np.zeros(33, dtype=float)
                ks = kpt_scores[idx] if kpt_scores is not None else kp[:, 2]
                for ci, mi in _COCO_TO_MP:
                    px[mi] = kp[ci, :2]
                    # KeypointRCNN keypoints_scores are logits; map > 0 to high conf via sigmoid
                    if kpt_scores is not None:
                        vis[mi] = float(1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, float(ks[ci]))))))
                    else:
                        vis[mi] = 0.9 if kp[ci, 2] > 0 else 0.0
                ok = vis > 0.35
                if ok.sum() < 3:
                    continue
                torso = [i for i in (L_SH, R_SH, L_HIP, R_HIP) if ok[i]]
                if torso:
                    centre = px[torso].mean(axis=0)
                else:
                    centre = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
                det = {
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "centre": centre.tolist(),
                    "px": px,
                    "vis": vis,
                    "score": float(score),
                    "torso_px": None,
                    "ankle_px": None,
                    "shoulder_px": None,
                    "body_yaw_deg": None,
                    "world_lms": None,
                }
                if ok[L_SH] and ok[R_SH]:
                    det["shoulder_px"] = float(np.linalg.norm(px[L_SH] - px[R_SH]))
                if ok[L_SH] and ok[R_SH] and ok[L_HIP] and ok[R_HIP]:
                    det["torso_px"] = float(np.linalg.norm((px[L_SH] + px[R_SH]) / 2 - (px[L_HIP] + px[R_HIP]) / 2))
                ank = [i for i in (L_ANK, R_ANK) if vis[i] > 0.45 and 0 <= px[i][1] < h - 2]
                if ank:
                    det["ankle_px"] = px[ank].mean(axis=0).tolist()
                det.update(self._face(bgr, px, vis, ok))
                candidates.append(det)
        else:
            ts = max(self.t_last + 1, int(time.monotonic() * 1000))
            self.t_last = ts
            res = self.landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), ts)
            for k, lms in enumerate(res.pose_landmarks):
                px = np.array([[lm.x * w, lm.y * h] for lm in lms])
                vis = np.array([min(lm.visibility, lm.presence) for lm in lms])
                ok = vis > 0.30
                if ok.sum() < 4:
                    continue
                torso = [i for i in (L_SH, R_SH, L_HIP, R_HIP) if ok[i]]
                centre = px[torso].mean(axis=0) if torso else px[ok].mean(axis=0)
                box = np.concatenate([px[ok].min(axis=0), px[ok].max(axis=0)])
                det = {"box": box.tolist(), "centre": centre.tolist(), "px": px, "vis": vis,
                       "score": float(vis[ok].mean()), "torso_px": None, "ankle_px": None, "shoulder_px": None}  # fmt: skip
                if ok[L_SH] and ok[R_SH]:
                    det["shoulder_px"] = float(np.linalg.norm(px[L_SH] - px[R_SH]))
                if ok[L_SH] and ok[R_SH] and ok[L_HIP] and ok[R_HIP]:
                    det["torso_px"] = float(np.linalg.norm((px[L_SH] + px[R_SH]) / 2 - (px[L_HIP] + px[R_HIP]) / 2))
                ank = [i for i in (L_ANK, R_ANK) if vis[i] > 0.6 and 0 <= px[i][1] < h - 2]
                if ank:
                    det["ankle_px"] = px[ank].mean(axis=0).tolist()
                det["body_yaw_deg"] = None
                det["world_lms"] = None
                if res.pose_world_landmarks and k < len(res.pose_world_landmarks):
                    wl = res.pose_world_landmarks[k]
                    det["world_lms"] = np.array([[lm.x, lm.y, lm.z] for lm in wl])
                    v = np.array([wl[L_SH].x - wl[R_SH].x, wl[L_SH].z - wl[R_SH].z])
                    if np.linalg.norm(v) > 0.1:
                        det["body_yaw_deg"] = round(math.degrees(math.atan2(v[1], v[0])), 1)
                det.update(self._face(bgr, px, vis, ok))
                candidates.append(det)

        # NMS deduplication for crowded scenes
        candidates.sort(key=lambda d: d["score"], reverse=True)
        out = []
        for det in candidates:
            if any(_box_iou(det["box"], kept["box"]) > 0.55 for kept in out):
                continue
            out.append(det)

        self.ms = 0.8 * self.ms + 0.2 * (time.perf_counter() - t0) * 1000 if self.ms else (time.perf_counter() - t0) * 1000
        return out

    def _face(self, bgr, px, vis, ok):
        """Face on the head crop (upscaled so YuNet sees ~100 px faces at any distance)."""
        h, w = bgr.shape[:2]
        head = [i for i in range(11) if vis[i] > 0.3]  # nose, eyes, ears, mouth
        if not head:
            return {"facing": False, "face_score": 0.0, "head_yaw_deg": None, "face_box": None,
                    "jaw_open": 0.0, "mar": 0.0, "mouth_openness": 0.0, "lip_pts": None}
        c = px[head].mean(axis=0)
        span = np.ptp(px[head], axis=0).max() if len(head) > 1 else 0
        sh = np.linalg.norm(px[L_SH] - px[R_SH]) if ok[L_SH] and ok[R_SH] else 0
        half = max(span * 1.2, sh * 0.55, 10.0)
        x0, x1 = int(max(c[0] - half, 0)), int(min(c[0] + half, w))
        y0, y1 = int(max(c[1] - 1.25 * half, 0)), int(min(c[1] + 1.1 * half, h))
        if x1 - x0 < 6 or y1 - y0 < 6:
            return {"facing": False, "face_score": 0.0, "head_yaw_deg": None, "face_box": None,
                    "jaw_open": 0.0, "mar": 0.0, "mouth_openness": 0.0, "lip_pts": None}
        crop = bgr[y0:y1, x0:x1]
        s = 160.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(8, int(crop.shape[1] * s)), max(8, int(crop.shape[0] * s))))
        self.face.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self.face.detect(crop)
        if faces is None or not len(faces):
            return {"facing": False, "face_score": 0.0, "head_yaw_deg": None, "face_box": [x0, y0, x1, y1],
                    "jaw_open": 0.0, "mar": 0.0, "mouth_openness": 0.0, "lip_pts": None}
        f = faces[np.argmax(faces[:, 14])]
        re, le, nose = f[4:6], f[6:8], f[8:10]
        eye = abs(le[0] - re[0])
        yaw = math.degrees(math.atan(((nose[0] - (re[0] + le[0]) / 2) / max(eye, 1e-3)) / NOSE_DEPTH))
        fb = [x0 + f[0] / s, y0 + f[1] / s, x0 + (f[0] + f[2]) / s, y0 + (f[1] + f[3]) / s]
        fx0, fy0, fx1, fy1 = fb
        fw, fh = fx1 - fx0, fy1 - fy0
        facing = bool(abs(yaw) < FACING_YAW_DEG)
        if max(fw, fh) < 48 or min(fw, fh) < 36 or abs(yaw) > FACING_YAW_DEG * 1.3:
            # Face is too small, distant (>2m), or turned away: skip face mesh to prevent distance noise
            return {"facing": facing, "face_score": round(float(f[14]), 3),
                    "head_yaw_deg": round(yaw, 1), "face_box": [round(v, 1) for v in fb],
                    "jaw_open": 0.0, "mar": 0.0, "mouth_openness": 0.0, "lip_pts": None}

        jaw_open, mar, lip_pts = 0.0, 0.0, None
        has_blendshapes = False
        if self.face_mesh is not None:
            try:
                cx0 = max(0, int(fx0 - fw * 0.2))
                cy0 = max(0, int(fy0 - fh * 0.2))
                cx1 = min(w, int(fx1 + fw * 0.2))
                cy1 = min(h, int(fy1 + fh * 0.2))
                if cx1 > cx0 + 16 and cy1 > cy0 + 16:
                    crop_fl = bgr[cy0:cy1, cx0:cx1]
                    res_fl = self.face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                           data=cv2.cvtColor(crop_fl, cv2.COLOR_BGR2RGB)))
                    if res_fl.face_blendshapes and len(res_fl.face_blendshapes[0]):
                        has_blendshapes = True
                        for bs in res_fl.face_blendshapes[0]:
                            if bs.category_name == "jawOpen":
                                jaw_open = float(bs.score)
                                break
                    if res_fl.face_landmarks and len(res_fl.face_landmarks[0]) > 308:
                        lms = res_fl.face_landmarks[0]
                        cw, ch = crop_fl.shape[1], crop_fl.shape[0]
                        p13 = np.array([lms[13].x * cw, lms[13].y * ch])
                        p14 = np.array([lms[14].x * cw, lms[14].y * ch])
                        p78 = np.array([lms[78].x * cw, lms[78].y * ch])
                        p308 = np.array([lms[308].x * cw, lms[308].y * ch])
                        mar = float(np.linalg.norm(p14 - p13) / max(np.linalg.norm(p308 - p78), 1.0))
                        lip_pts = [
                            [int(cx0 + p13[0]), int(cy0 + p13[1])],
                            [int(cx0 + p14[0]), int(cy0 + p14[1])],
                            [int(cx0 + p78[0]), int(cy0 + p78[1])],
                            [int(cx0 + p308[0]), int(cy0 + p308[1])],
                        ]
            except Exception:
                pass

        if has_blendshapes:
            openness = jaw_open
        else:
            openness = max(0.0, (mar - 0.12) / 0.18)

        return {"facing": bool(abs(yaw) < FACING_YAW_DEG), "face_score": round(float(f[14]), 3),
                "head_yaw_deg": round(yaw, 1), "face_box": [round(v, 1) for v in fb],
                "jaw_open": round(jaw_open, 3), "mar": round(mar, 3),
                "mouth_openness": round(openness, 3), "lip_pts": lip_pts}
        return {"facing": bool(abs(yaw) < FACING_YAW_DEG), "face_score": round(float(f[14]), 3),
                "head_yaw_deg": round(yaw, 1), "face_box": [round(v, 1) for v in fb],
                "jaw_open": round(jaw_open, 3), "mar": round(mar, 3),
                "mouth_openness": round(openness, 3), "lip_pts": lip_pts}
