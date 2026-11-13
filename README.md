# G1 Vision: Unitree G1 Multimodal Perception & Tracking

Comprehensive perception, tracking, gesture recognition, distance calibration, and high-framerate streaming pipeline for the **Unitree G1 humanoid robot**.

---

## 1. System Overview

The G1 vision system delivers real-time multimodal perception:
* **High-Framerate Streaming**: Smooth ~30 FPS live video streaming from the robot's front eye-level camera.
* **Aspect Ratio Preservation**: Displays and processes frames in their native camera aspect ratio without stretching or squishing.
* **Calibrated Distance Estimation**: Precise distance measurement calibrated against ground-truth targets (1.0 m, 2.0 m, and 3.0 m for a 165 cm person).
* **Robust Multi-Person Tracking**: Kalman filter-based tracking with greedy 1-to-1 association and Non-Maximum Suppression (NMS) to eliminate duplicate detections in crowded environments.
* **Speech / Talking Detection**: MediaPipe blendshape-driven jaw motion tracking with distance filtering and speech dynamic thresholds.
* **Gesture Recognition & Pointing Vector**:
  * **Pointing**: Arm extension detection, 2D pointing ray in image space, and 3D pointing vector $(X, Y, Z)$ with yaw and pitch angles in the robot frame.
  * **Waving**: Body-scaled oscillatory wrist movement detection with bent-elbow constraint.
  * **Disambiguation**: Anti-confusion state machine preventing pointing motions from triggering false waves.
* **Demographics**: FairFace gender classification and MiVOLO age estimation per tracked individual.
* **Web Interface & REST API**: Live web dashboard, top-down radar display, and long-polling event bus on port `8107`.

---

## 2. Hardware & Network Setup

### Network Configuration
The Unitree G1 robot operates on a dedicated static subnet: `192.168.123.0/24`.

1. **Connect Cable**: Plug an RJ-45 Ethernet cable directly between the laptop's Ethernet port and the robot's network port.
2. **Laptop IP Address**: Configure the laptop's Ethernet adapter with a static IP:
   * **IP Address**: `192.168.123.222`
   * **Subnet Mask**: `255.255.255.0` (`/24`)
   * **Gateway / DNS**: (Leave blank)
3. **Verify Connectivity**:
   ```powershell
   # In Windows PowerShell:
   Get-NetAdapter -Name 'Ethernet'
   Test-Connection -ComputerName 192.168.123.100 -Count 2
   ```

### Robot Camera Endpoints
The Unitree G1 exposes two camera HTTP streaming endpoints on port 8000 (serving JSON base64-encoded JPEGs):
* **Front Eye-Level Camera (Used)**: `http://192.168.123.100:8000/capture`
  * Mount: `level` (height: 1.26 m, pitch: $0^\circ$, looking straight ahead at people and the room)
  * Horizontal Field of View (HFOV): $69^\circ$
* **Waist RealSense D435**: `http://192.168.123.164:8000/capture`
  * Mount: `d435` (height: 0.85 m, pitched $47.6^\circ$ down towards the floor/shoes)

---

## 3. How to Run

### Option A: Standard Run with Robot Camera (Recommended)
Open PowerShell and run:
```powershell
cd vision
python -m vision.service --camera unitree:http://192.168.123.100:8000/capture --camera-mount level --hfov 69 --max-fps 30 --port 8107 --joints-url none
```

### Option B: Local USB Webcam Fallback (Testing without Robot)
If developing away from the robot, run with your laptop webcam:
```powershell
cd vision
python -m vision.service --camera 0 --camera-mount level --hfov 69 --max-fps 30 --port 8107 --joints-url none
```

### Option C: Running with Full LiDAR & Joint Telemetry (Simulation / Robot Pack)
```powershell
cd vision
python -m vision.service --camera unitree:http://192.168.123.100:8000/capture --camera-mount level --hfov 69 --lidar-port COM3 --joints-url http://192.168.123.100:8080/joints --port 8107
```

---

## 4. Web Interface & Endpoints

Once running, navigate to:
* **Web Dashboard**: [http://localhost:8107/](http://localhost:8107/)
  * Displays the live annotated camera view, top-down radar view, and real-time state table for all tracked people.

### Available API Endpoints
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `http://localhost:8107/` | `GET` | Interactive web dashboard. |
| `http://localhost:8107/frame.jpg` | `GET` | Latest annotated camera JPEG frame (~30 FPS). |
| `http://localhost:8107/top.jpg` | `GET` | Top-down 2D radar view of the robot and surrounding people/clusters. |
| `http://localhost:8107/people?all=1` | `GET` | JSON snapshot of all active tracks, positions, velocities, gestures, talking states, and demographics. |
| `http://localhost:8107/status` | `GET` | JSON diagnostic report (camera FPS, detector latency, errors). |
| `http://localhost:8107/events?after=0&wait=10` | `GET` | Long-polling HTTP event stream (`person_new`, `person_near`, `person_dwell`, `person_talking`, `person_gesture`, `person_left`). |

---

## 5. Technical Modifications & Changelog

### 1. High-Framerate Optimization (Jump from ~13 FPS to ~30 FPS)
* **Persistent HTTP Keep-Alive**: Replaced per-frame `urllib.request.urlopen` requests with persistent `http.client.HTTPConnection` in `vision/camera.py`. Raw fetch latency dropped from ~45 ms to ~9 ms.
* **Decoupled Video Pipeline**: Decoupled browser frame serving in `vision/service.py`. `/frame.jpg` now serves annotated frames over the latest camera buffer immediately without waiting for the slower CPU detector inference loop.
* **Client Pipelining**: Upgraded the web interface JavaScript in `vision/service.py` from a fixed 300 ms timer (`setInterval`, 3.3 FPS) to an event-driven `Image.onload` pipelining loop operating at 33 ms intervals (~30 FPS).

### 2. Native Camera Aspect Ratio Preservation
* **Removed Forced Square Resize**: Removed `cv2.resize(img, (640, 480))` from `camera.py` for network streams.
* **Dynamic Camera Model**: `CameraModel.set_size((w, h))` automatically adapts focal length and projection rays to any native sensor resolution (e.g. 16:9 widescreen or 4:3).
* **Responsive Web Layout**: Added `object-fit: contain` and `height: auto` to `#f` in `vision/service.py` to prevent browser CSS from stretching the feed into a square.

### 3. Distance Calibration (165 cm Person Ground Truth)
* **Empirical Ground Truth**: Calibrated using test captures of a 165 cm tall person standing at 1.0 m, 2.0 m, and 3.0 m:
  * Measured Raw: `1.252 m` $\implies$ Calibrated True: `1.000 m`
  * Measured Raw: `1.745 m` $\implies$ Calibrated True: `2.000 m`
  * Measured Raw: `2.653 m` $\implies$ Calibrated True: `3.000 m`
* **Piecewise Linear Model**: Implemented `calibrate_camera_distance(raw_d)` in `vision/tracker.py` providing monotonic mapping across the operating range.
* **Fixed Method Scoping**: Fixed an indentation bug in `tracker.py` where `calibrate_camera_distance` had enclosed `place()`, properly restoring `CameraModel.place()` across `feet`, `torso`, `shoulders`, and `face` estimators.

### 4. Talking Detection Overhaul
* **Eliminated False Baseline Noise**:
  * *Previous Bug*: `openness = max(jaw_open, mar * 1.5)` resulted in a permanent openness floor of $\ge 0.15$ because resting closed lips naturally have `mar \approx 0.10`.
  * *Fix*: `detector.py` now directly consumes MediaPipe `jaw_open` blendshapes (closed mouth: `0.00 - 0.03`, talking: `0.20 - 0.60`).
* **Distance Noise Suppression**: Facial landmarking is automatically bypassed for faces smaller than 48 px or farther than 2.2 m, completely eliminating distance-induced pixel jitter false alarms.
* **Speech Dynamics Criteria**: In `tracker.py`, talking requires active syllable movement (`max_v >= 0.16`, `range_v >= 0.10`, `vel >= 0.15`, `reversals >= 1`). Hold duration is tuned to 0.5 s to eliminate syllable flickering without lingering.

### 5. Pointing vs. Waving Disambiguation & 3D Pointing Vector
* **Anti-Confusion State Machine**:
  * Moving an arm into a pointing pose previously triggered waving due to transient wrist displacement.
  * Pointing is now checked first and takes priority. When pointing or extending the arm, the waving wrist history is immediately purged.
  * Active point holds (`0.6 s`) cannot be preempted by wave gestures.
* **Posture Requirements**:
  * **Pointing**: Arm extended (elbow angle $\ge 115^\circ$, reach $\ge 0.55$, torso angle $\ge 22^\circ$).
  * **Waving**: Forearm bent (elbow angle $< 140^\circ$), wrist raised near or above shoulder height, and cyclic oscillation ($\ge 2$ reversals: left-right-left).
* **3D Pointing Vector Calculation**:
  * Computes normalized 3D ray $(X_{fwd}, Y_{left}, Z_{up})$ in the Robot Frame.
  * Calculates `pointing_yaw_deg` and `pointing_pitch_deg`.
  * Visualized with an arrowed line on the 2D camera feed and a 3D ray on the top-down radar view.

### 6. Crowded Scene & Multi-Person Tracking Stability
* **Non-Maximum Suppression (NMS)**: Added `_box_iou()` filtering in `detector.py` (IoU threshold $0.55$) to eliminate duplicate detections of the same person.
* **Direct Kalman Association**: Replaced fragile secondary distance searching in `service.py` by binding detections directly to their Kalman track ID in `tracker.py` `_apply()`.
* **Track Retention**: Relaxed association gating (`max_d = 1.2 m`, `maha < 20.0`) and raised lost-track timeout to 1.5 s to maintain track persistence through occlusion and rapid movement.

---

## 6. Code Architecture Reference

```
exp05/vision/vision/
├── camera.py       # Camera grabber (HTTP Keep-Alive for Unitree & OpenCV device grabber)
├── detector.py     # MediaPipe Pose Landmarker, YuNet Face, FaceLandmarker (blendshapes), NMS
├── tracker.py      # Kalman filter, distance calibration, talking & gesture arbitration, events
├── age_gender.py   # FairFace & MiVOLO background demographic estimation workers
├── lidar.py        # 2D LiDAR driver and spatial clustering (when hardware is connected)
└── service.py      # HTTP server, frame drawing, web dashboard, and REST API (/people, /events)
```
