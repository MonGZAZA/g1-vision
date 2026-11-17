# Changelog — G1 Vision

All notable changes, trained models, benchmarks, and hardware/network updates for **Unitree G1 Vision**.

---

## [v2.0.0] — 2026-11-17 — Crowded Pose, Learned ML Gestures, 3D Pointing, `facing_state`, & LAN+LiDAR

### Added
* **Crowded Human & Pose Detector (`Crowd-KeypointRCNN`, BSD-3-Clause)**:
  * Integrated TorchVision `KeypointRCNN_ResNet50_FPN` (`BSD-3-Clause`, CUDA/CPU) as the primary detector in `vision/vision/detector.py`, replacing AGPL-3.0 dependencies and MediaPipe BlazePose (`--pose-backend mediapipe` remains available as fallback).
  * Created `vision/training/prepare_and_train_crowd_pose.py` to curate **794 crowded COCO 2017 images** ($\ge 3$ people with keypoints per frame; 652 train / 142 held-out crowded validation) with horizontal-flip keypoint swapping, lighting jitter, and random occlusion patches.
  * Saved FP16 fine-tuned head weights to `vision/models/crowd_keypointrcnn_heads.pt` (71.38 MB, under GitHub's 100 MB limit).
* **Temporal ML Gesture Classifier (`GestureNet`, MIT)**:
  * Created `vision/vision/gesture_ml.py` with a 30-feature scale-, yaw-, and FPS-invariant kinematic extractor (`extract_arm_features`) and NumPy MLP inference (`GestureClassifier`).
  * Created `vision/training/train_gesture.py` generating **36,000 train / 6,000 val** 3D$\rightarrow$2D pinhole-projected upper-body sequences (`still`, `point`, `wave`), including foreshortened forward pointing, arm-lift-to-point transitions, and hard negatives (crossed arms, hands on hips, phone/head, conversational hands).
  * Exported compact weights to `vision/models/gesture_mlp.npz` (32.5 KB).
* **Learned 3D Pointing Direction Regressor (`PointingDirNet`, MIT)**:
  * Added 18-feature 2D upper-body geometry extractor (`extract_pointing_dir_features`) and `PointingDirRegressor` in `vision/vision/gesture_ml.py`.
  * Trained `PointingDirNet` on **30,000 train / 5,000 val** 3D pointing poses (`-105°..+105°` azimuth, `-45°..+50°` elevation, `0.9–4.5 m`, $\pm 50^\circ$ body yaw) using cosine-similarity loss.
  * Exported compact weights to `vision/models/pointing_dir_mlp.npz` (34.1 KB).
* **Granular `facing_state` Classification**:
  * Added `Track.facing_state()` in `vision/vision/tracker.py` returning semantic motion+facing states: `walking_facing`, `walking_past_facing`, `standing_facing`, `walking_approaching`, `walking_away`, `walking_past`, `standing_away`, `lidar_approaching`, `lidar_moving`, `lidar_still` (while keeping boolean `facing` for backward compatibility).
  * Surfaced `facing_state` in `/people?all=1`, camera bounding-box overlays, and the web UI table in `vision/vision/service.py`.
* **G1 LAN + Livox MID-360 LiDAR Launcher**:
  * Added `vision/run_robot_vision.ps1` and `vision/run_robot_vision.bat` to verify `Ethernet` static IP `192.168.123.222/24`, check `192.168.123.100` reachability, configure/verify Windows Firewall for **Livox MID-360 LiDAR UDP port `56301`**, and start `vision.service` with camera + LiDAR fusion.
* **Commercial Licensing & Citations**:
  * Added `MODEL_CARDS_AND_LICENSES.md` documenting all models, datasets, BibTeX citations, and commercial licenses (`BSD-3-Clause`, `Apache-2.0`, `MIT`, `CC BY 4.0`).

### Changed & Improved
* **Crowded Scene Validation (`score >= 0.45`, 142 held-out crowded COCO2017 images)**:
  * Baseline `KeypointRCNN`: `P=0.3999`, `R=0.9567`, `F1=0.5640` (`TP=905, FP=1358, FN=41`).
  * Fine-Tuned `Crowd-KeypointRCNN`: `P=0.5068`, `R=0.9482`, **`F1=0.6605`** (`TP=897, FP=873, FN=49`) — **+9.65 F1** and **35.7% fewer false positives** at ~95% recall.
  * On a 12-person crowded test image (`000000001000.jpg`), `Crowd-KeypointRCNN-CUDA` (`min_conf=0.65`) detected **15 people** vs. **2 people** with MediaPipe Pose.
* **Gesture & 3D Pointing Accuracy**:
  * `GestureNet` achieved **100.0% validation accuracy** across 6,000 held-out sequences (`still`: 3115/3115, `point`: 1828/1828, `wave`: 1057/1057), with EMA smoothing and hysteresis (`enter >= 0.55`, `exit < 0.38`) in `vision/vision/tracker.py`.
  * `PointingDirNet` achieved **6.79° median / 9.24° mean 3D angular error** across 5,000 held-out 3D pointing vectors.
* **Documentation & `.gitignore`**:
  * Updated `README.md` and `.gitignore` to track the commercially permissive trained models (`crowd_keypointrcnn_heads.pt`, `gesture_mlp.npz`, `pointing_dir_mlp.npz`).

---

## [v1.0.0] — 2026-11-13 — Initial G1 Vision Pipeline & Calibration
* Persistent HTTP Keep-Alive streaming from Unitree G1 front camera (`192.168.123.100:8000/capture`) at ~30 FPS.
* Native camera aspect ratio preservation across detector and web dashboard.
* Piecewise linear distance calibration for a 165 cm person (`1.252 m -> 1.0 m`, `1.745 m -> 2.0 m`, `2.653 m -> 3.0 m`).
* Fixed talking detection baseline noise using MediaPipe `jaw_open` blendshapes and distance cutoff (`> 2.2 m`).
* Added FairFace gender & MiVOLO age estimation workers and initial `README.md` / `vision.md`.
