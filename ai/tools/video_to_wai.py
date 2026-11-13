"""Wai from a video -> measurements for the robot (step 1 of 2; step 2: g1/tools/make_wai_clip.py).

MediaPipe Holistic (pose + both hands; models ship with the pip package, no registration) per frame, then:
  - finds the wai holds: both hands visible, wrists close together, hands raised above the elbows;
  - progress s(t) of each wai (0 = hands down, 1 = wai), from the wrists' height, low-pass 4 Hz;
  - hold height: fingertips relative to the shoulders (0) and the chin (1);
  - forward tilt of the hands (from the hand landmarks' depth), finger straightness (Inspire 0..1 per finger);
  - a contact sheet of the frames used (for the user to judge).
Monocular close-up videos usually cut off the elbows, so the robot's arm comes from IK on these measurements,
not from the human's joint angles.

    cd exp05/ai && .venv/bin/python tools/video_to_wai.py ../videos/examples/x07hnDvNcxM.mp4 [--start 26 --end 40]
        -> exp05/clips/source/<video>_wai.json (+ _sheet.jpg)
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

EXP05 = Path(__file__).resolve().parents[2]
OUT = EXP05 / "clips/source"
FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12), "ring": (13, 14, 16), "pinky": (17, 18, 20)}


def lowpass(x, fps, fc=4.0):
    """Zero-phase first-order low-pass (forward + backward)."""
    a = 1.0 / (1.0 + fps / (2 * np.pi * fc))
    y = np.array(x, float)
    for rng in (range(1, len(y)), range(len(y) - 2, -1, -1)):
        for i in rng:
            j = i - 1 if rng.step == 1 else i + 1
            y[i] = y[j] + a * (y[i] - y[j])
    return y


def angle(a, b, c):
    v1, v2 = a - b, c - b
    return np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))


def measure(path, t_start, t_end):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000)
    hol = mp.solutions.holistic.Holistic(model_complexity=1, refine_face_landmarks=False)
    rows, frames = [], {}
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, fr = cap.read()
        if not ok or (t_end and t > t_end):
            break
        h, w = fr.shape[:2]
        r = hol.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        row = {"t": t}
        if r.pose_landmarks:
            P = np.array([[p.x * w, p.y * h, p.visibility] for p in r.pose_landmarks.landmark])
            row.update(sh=(P[11, :2] + P[12, :2]) / 2, sw=abs(P[11, 0] - P[12, 0]), nose=P[0, :2],
                       mouth=(P[9, :2] + P[10, :2]) / 2, lw=P[15], rw=P[16], le=P[13], re=P[14])
        for side, hl in (("L", r.left_hand_landmarks), ("R", r.right_hand_landmarks)):
            if hl:
                row[side] = np.array([[p.x * w, p.y * h, p.z * w] for p in hl.landmark])
        rows.append(row)
        frames[round(t, 2)] = fr
    return fps, rows, frames


def analyse(fps, rows):
    t = np.array([r["t"] for r in rows])
    ok = np.array(["sh" in r for r in rows])
    if ok.sum() < 10:
        raise SystemExit("no person found")
    sw = np.median([r["sw"] for r in rows if "sw" in r])
    both = np.array(["L" in r and "R" in r for r in rows])
    # wrist height above the shoulder line, in shoulder widths (hands: wrist landmark 0; else pose wrists)
    wy, close = np.full(len(rows), np.nan), np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        if "sh" not in r:
            continue
        if "L" in r and "R" in r:
            wl, wr = r["L"][0, :2], r["R"][0, :2]
            close[i] = np.linalg.norm(wl - wr) < 0.45 * sw
        elif min(r["lw"][2], r["rw"][2]) > 0.3:
            wl, wr = r["lw"][:2], r["rw"][:2]
        else:
            continue  # hands out of the picture: down
        wy[i] = (r["sh"][1] - (wl[1] + wr[1]) / 2) / sw
    low = np.nanpercentile(wy, 5) if np.isfinite(wy).any() else -1.5
    wy = np.where(np.isfinite(wy), wy, min(low, -1.2))
    wy = lowpass(wy, fps)
    hold = both & close & (wy > np.percentile(wy, 50))
    # hold segments (>= 0.4 s)
    segs, s0 = [], None
    for i, hflag in enumerate(hold):
        if hflag and s0 is None:
            s0 = i
        if (not hflag or i == len(hold) - 1) and s0 is not None:
            if t[i] - t[s0] >= 0.4:
                segs.append((s0, i))
            s0 = None
    if not segs:
        raise SystemExit("no wai found (both hands together, raised) in this part of the video")
    merged = [segs[0]]
    for a, b in segs[1:]:  # landmark drop-outs split one hold: join holds less than 0.8 s apart
        if t[a] - t[merged[-1][1]] < 0.8:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    segs = merged
    out = []
    for a, b in segs:
        y_hold = np.median(wy[a:b])
        # extend to the hands-down level on both sides
        lo_i, hi_i = a, b
        while lo_i > 0 and wy[lo_i] > low + 0.15 * (y_hold - low) and t[a] - t[lo_i] < 3.0:
            lo_i -= 1
        while hi_i < len(wy) - 1 and wy[hi_i] > low + 0.15 * (y_hold - low) and t[hi_i] - t[b] < 3.0:
            hi_i += 1
        s = np.clip((wy[lo_i:hi_i + 1] - low) / max(y_hold - low, 1e-3), 0, 1)
        # geometry during the hold
        tips, tipf, tilt, flat, thumb = [], [], [], {k: [] for k in FINGERS}, []
        for r in rows[a:b]:
            if not ("L" in r and "R" in r and "sh" in r):
                continue
            chin = r["mouth"] + (r["mouth"] - r["nose"]) * 1.1
            tip_y = (r["L"][12, 1] + r["R"][12, 1]) / 2
            tips.append((r["sh"][1] - tip_y) / max(r["sh"][1] - chin[1], 1e-3))
            tipf.append((chin[1] - tip_y) / max(chin[1] - r["nose"][1], 1e-3))  # 0 = chin, 1 = nose
            for H in (r["L"], r["R"]):
                d = H[9] - H[0]  # wrist -> middle finger base
                tilt.append(np.degrees(np.arctan2(-d[2], -d[1])))  # + = fingers lean away from the body
                for k, (m_, p_, tp) in FINGERS.items():
                    flat[k].append(np.clip((angle(H[m_], H[p_], H[tp]) - 90) / 90, 0, 1))
                thumb.append(np.linalg.norm(H[4] - H[5]) / (np.linalg.norm(H[9] - H[0]) + 1e-9))
        out.append({
            "t_start": round(float(t[lo_i]), 3), "t_hold": [round(float(t[a]), 3), round(float(t[b]), 3)],
            "t_end": round(float(t[hi_i]), 3),
            "progress": {"t": [round(float(x - t[lo_i]), 3) for x in t[lo_i:hi_i + 1]], "s": [round(float(x), 4) for x in s]},
            "tip_height": round(float(np.median(tips)), 3) if tips else None,  # 0 = shoulder line, 1 = chin
            "tip_face": round(float(np.median(tipf)), 3) if tipf else None,  # 0 = chin, 1 = nose (close-ups)
            "tilt_deg": round(float(np.median(tilt)), 1) if tilt else 0.0,
            "fingers": {k: round(float(np.median(v)), 3) for k, v in flat.items() if v},
            "thumb_to_index": round(float(np.median(thumb)), 3) if thumb else None,
            "hands_visible": round(float(both[a:b].mean()), 2),
        })
    return out


def contact_sheet(frames, wai, path):
    ts = sorted(frames)
    picks = np.linspace(wai["t_start"], wai["t_end"], 8)
    ims = []
    for p in picks:
        k = min(ts, key=lambda x: abs(x - p))
        im = frames[k]
        im = cv2.resize(im, (320, int(320 * im.shape[0] / im.shape[1])))
        cv2.putText(im, f"{k:.1f}s", (5, 20), 0, 0.6, (0, 0, 255), 2)
        ims.append(im)
    cv2.imwrite(str(path), np.vstack([np.hstack(ims[:4]), np.hstack(ims[4:])]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0)
    a = p.parse_args()
    v = Path(a.video).resolve()
    fps, rows, frames = measure(v, a.start, a.end)
    if not rows:
        raise SystemExit("no frames could be decoded (AV1 video? re-encode or download as H.264 / avc1)")
    wais = analyse(fps, rows)
    if not wais:
        raise SystemExit("no wai found: both hands must come together in front of the chest and be visible")
    best = max(wais, key=lambda w: (w["hands_visible"], w["t_hold"][1] - w["t_hold"][0]))
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{v.stem}_{int(best['t_start'])}s_wai"
    contact_sheet(frames, best, OUT / f"{stem}_sheet.jpg")
    res = {"source": str(v.relative_to(EXP05)) if v.resolve().is_relative_to(EXP05) else str(v), "fps": fps,
           "chosen": wais.index(best), "wais": wais}
    (OUT / f"{stem}.json").write_text(json.dumps(res, indent=1))
    for i, w in enumerate(wais):
        print(f"wai {i}: {w['t_start']:.1f}-{w['t_end']:.1f} s, hold {w['t_hold']}, tip height {w['tip_height']}, "
              f"tilt {w['tilt_deg']} deg, fingers {w['fingers']}{'  <- chosen' if w is best else ''}")
    print(OUT / f"{stem}.json")


if __name__ == "__main__":
    main()
