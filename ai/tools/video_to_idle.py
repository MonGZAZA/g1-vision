"""Keep-alive / talking style from a video of a person talking calmly -> exp05/clips/idle.json.

MediaPipe Pose per frame (every 2nd frame): wrist positions relative to the shoulders, in shoulder widths.
  - idle amplitude: median over 4 s windows of the wrist position spread while the hands are calm;
  - idle speed: dominant frequency of the calm wrist motion (0.05-1 Hz);
  - gesture rate: hand-speed peaks above a threshold per minute (beats / gestures while talking).
These scale the robot's idle layer (gestures.idle) and how often it adds a beat (behaviour.py), relative to the
defaults (REF values = the procedural defaults, tuned by eye).

    cd exp05/ai && .venv/bin/python tools/video_to_idle.py VIDEO [--start 20 --end 80] [--install]
        -> exp05/clips/source/<video>_idle.json; --install also makes it the robot's style (clips/idle.json)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

EXP05 = Path(__file__).resolve().parents[2]
REF_SPREAD = 0.035  # shoulder widths: calm hand drift that the default idle amplitude corresponds to
REF_FREQ = 0.15  # Hz: the default idle's main frequency
REF_RATE = 6.0  # gestures/beats per minute that the default beat scheduling gives


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0)
    p.add_argument("--install", action="store_true", help="use this style on the robot (clips/idle.json)")
    a = p.parse_args()
    cap = cv2.VideoCapture(str(Path(a.video).resolve()))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    pose = mp.solutions.pose.Pose(model_complexity=1)
    T, W = [], []
    i = 0
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok = cap.grab()
        if not ok or (a.end and t > a.end):
            break
        i += 1
        if i % 2:
            continue
        ok, fr = cap.retrieve()
        r = pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if not r.pose_landmarks:
            continue
        L = r.pose_landmarks.landmark
        if min(L[15].visibility, L[16].visibility, L[11].visibility, L[12].visibility) < 0.5:
            continue
        sw = abs(L[11].x - L[12].x) or 1e-3
        cx, cy = (L[11].x + L[12].x) / 2, (L[11].y + L[12].y) / 2
        T.append(t)
        W.append([(L[15].x - cx) / sw, (L[15].y - cy) / sw, (L[16].x - cx) / sw, (L[16].y - cy) / sw])
    T, W = np.array(T), np.array(W)
    if len(T) == 0 and not cap.isOpened():
        raise SystemExit("the video could not be opened")
    if len(T) < 50:
        raise SystemExit(f"hands visible in only {len(T)} frames: use a video with both hands in the picture")
    dt = np.median(np.diff(T))
    speed = np.linalg.norm(np.gradient(W, T, axis=0).reshape(len(T), 2, 2), axis=2).max(axis=1)  # shoulder widths / s
    calm = speed < np.percentile(speed, 60)
    spreads = []
    for t0 in np.arange(T[0], T[-1] - 4, 2.0):
        k = (T >= t0) & (T < t0 + 4) & calm
        if k.sum() > 10:
            spreads.append(float(np.mean(np.std(W[k], axis=0))))
    spread = float(np.median(spreads)) if spreads else REF_SPREAD
    # dominant frequency of the calm motion (uniformly resampled)
    tu = np.arange(T[0], T[-1], dt)
    x = np.interp(tu, T, W[:, 1] + W[:, 3])
    x = x - np.convolve(x, np.ones(int(10 / dt)) / int(10 / dt), mode="same")  # remove drift slower than 0.1 Hz
    f = np.fft.rfftfreq(len(x), dt)
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    band = (f > 0.05) & (f < 1.0)
    freq = float(f[band][np.argmax(P[band])]) if band.any() else REF_FREQ
    thr = max(0.8, np.percentile(speed, 90))
    peaks = int(np.sum((speed[1:-1] > thr) & (speed[1:-1] >= speed[:-2]) & (speed[1:-1] >= speed[2:])))
    minutes = (T[-1] - T[0]) / 60
    rate = peaks / max(minutes, 1e-3)
    out = {"source": str(Path(a.video).resolve().relative_to(EXP05)), "segment_s": [round(T[0], 1), round(T[-1], 1)],
           "made": time.strftime("%Y-%m-%d %H:%M"), "frames_used": int(len(T)),
           "measured": {"calm_spread_sw": round(spread, 4), "idle_freq_hz": round(freq, 3), "gestures_per_min": round(rate, 1)},
           "idle_amp_scale": round(float(np.clip(spread / REF_SPREAD, 0.5, 1.6)), 2),
           "idle_speed_scale": round(float(np.clip(freq / REF_FREQ, 0.6, 1.6)), 2),
           "beat_every": int(np.clip(round(2 * REF_RATE / max(rate, 0.5)), 1, 4)),
           "enabled": True}
    (EXP05 / "clips/source").mkdir(parents=True, exist_ok=True)
    dst = EXP05 / "clips/source" / f"{Path(a.video).stem}_idle.json"
    dst.write_text(json.dumps(out, indent=1))
    if a.install:
        (EXP05 / "clips/idle.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(dst, "(installed: clips/idle.json)" if a.install else "(not installed; --install to use it)")


if __name__ == "__main__":
    main()
