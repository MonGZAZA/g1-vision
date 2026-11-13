"""Wai measurements from a video -> robot wai clip (step 2 of 2; step 1: ai/tools/video_to_wai.py).

  - wai pose by IK at the video's hold height (fingertips between the shoulder line and the chin, mapped to the
    robot) and forward tilt; palms mirrored with a 1.5 cm gap (no pressing);
  - raise and lower follow the video's progress curves (monotonic, low-passed), slowed down where the robot would
    exceed PEAK_FRACTION of the arm channel's speed limit at speed scale 1.0;
  - fingers from the video's finger straightness (Inspire values), thumb as in the "flat" hand pose (Inspire: along
    the index; Dex3: folded across the palm, else the thumbs collide);
  - solved on the model of EXP05_HANDS (default dex3); one clip per hand model: clips/wai_<dex3|inspire>.json;
  - checks: palm gap along the whole path, joint limits, left/right symmetry, peak speed;
  - comparison sheet: video frames next to the robot at the same progress (for the user to judge).
The hold length is not taken from the video: the robot holds while it says "สวัสดีค่ะ"/"Hello" (behaviour.py).

    .venv/bin/python tools/make_wai_clip.py ../clips/source/x07hnDvNcxM_29s_wai.json [--tip-z 1.17] [--no-install]
        -> exp05/clips/wai_<hands>.json (used by the "Sawasdee + introduce" action), <source>_compare_<hands>.jpg
    EXP05_HANDS=inspire .venv/bin/python tools/make_wai_clip.py ...     (the clip for the Inspire hands)
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
G1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G1 / "deploy"))

import numpy as np  # noqa: E402

from arm_channel import MAX_SPEED  # noqa: E402
from gestures import hand_gap, rest_q, solve_wai  # noqa: E402
from motion import PEAK_FRACTION  # noqa: E402
from pointing import Pointer  # noqa: E402
from robot_model import HAND_MODEL, HAND_POSES, MOTORS, UPPER, WAI_CLIP  # noqa: E402

EXP05 = G1.parent
SHOULDER_Z, CHIN_Z = 1.08, 1.20  # robot: shoulder line and chin height (m), to map the video's hold height
CHIN_NOSE_M = 0.06  # robot "chin to nose" scale for the face-relative height
TIP_Z_RANGE = (1.10, 1.24)
DT = 0.02
S_PRE = 0.8


def mirror_err(q):
    """Largest left/right asymmetry (rad) of a 15-joint pose (roll/yaw joints change sign)."""
    L = [i for i, n in enumerate(UPPER) if n.startswith("left")]
    R = [i for i, n in enumerate(UPPER) if n.startswith("right")]
    sign = np.array([-1 if any(k in UPPER[i] for k in ("roll", "yaw")) else 1 for i in L])
    return float(np.max(np.abs(q[L] - sign * q[R])))


def profile(t, s, rising):
    """Monotonic, resampled to DT; returns (times, s)."""
    t, s = np.asarray(t, float), np.asarray(s, float)
    s = np.maximum.accumulate(s) if rising else np.minimum.accumulate(s)
    s = (s - s.min()) / max(s.max() - s.min(), 1e-6)
    tt = np.arange(0.0, t[-1] - t[0] + 1e-9, DT)
    return tt, np.interp(tt + t[0], t, s)


def retime(tt, s, dq_max):
    """Stretch time where |ds/dt| * dq_max would exceed the speed budget at speed scale 1."""
    vmax = PEAK_FRACTION * MAX_SPEED
    ds = np.abs(np.gradient(s, tt))
    stretch = np.maximum(1.0, ds * dq_max / vmax)
    new_t = np.concatenate([[0.0], np.cumsum(np.diff(tt) * (stretch[1:] + stretch[:-1]) / 2)])
    out_t = np.arange(0.0, new_t[-1] + 1e-9, DT)
    return out_t, np.interp(out_t, new_t, s)


def minjerk_s(n):
    x = np.linspace(0, 1, n)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--tip-z", type=float, default=None, help="override the hold height (m, fingertips)")
    p.add_argument("--no-install", action="store_true", help="do not write clips/wai_<hands>.json")
    p.add_argument("--out", default="", help="write the clip here instead (also when the checks fail, ok=false): "
                                            "a candidate the operator accepts or not (server: video upload)")
    a = p.parse_args()
    src = json.loads(Path(a.source).read_text())
    w = src["wais"][src["chosen"]]
    k = w["tip_height"] if w["tip_height"] is not None else 1.0
    if w.get("tip_face") is not None:  # face landmarks are reliable even in close-ups (shoulders often are not)
        z = CHIN_Z + CHIN_NOSE_M * w["tip_face"]
    else:
        z = SHOULDER_Z + k * (CHIN_Z - SHOULDER_Z)
    tip_z = a.tip_z or float(np.clip(z, *TIP_Z_RANGE))
    tilt = float(np.clip(w["tilt_deg"], -10, 25))
    ptr = Pointer()
    q_wai, info = solve_wai(ptr, tip_z, tilt)
    q_pre, _ = solve_wai(ptr, tip_z - 0.02, tilt, gap=0.08)  # hands 8 cm apart: the palms close only at the end
    q_rest = rest_q()
    dq = np.abs(q_wai - q_rest)
    dq_max = float(dq.max())

    pr_t, pr_s = np.array(w["progress"]["t"]), np.array(w["progress"]["s"])
    th0 = w["t_hold"][0] - w["t_start"]
    th1 = w["t_hold"][1] - w["t_start"]
    up = pr_t <= th0 + 1e-6
    dn = pr_t >= th1 - 1e-6
    if up.sum() >= 4 and pr_s[up][0] < 0.5:
        rt, rs = retime(*profile(pr_t[up], pr_s[up], True), dq_max)
        raise_src = "video"
    else:  # the raise is not visible (hands came from outside the picture): min-jerk
        rt = np.arange(0, 1.0 + 1e-9, DT)
        rs = minjerk_s(len(rt))
        rt, rs = retime(rt, rs, dq_max)
        raise_src = "min-jerk (raise not visible in the video)"
    if dn.sum() >= 4 and pr_s[dn][-1] < 0.5:
        lt, ls = profile(pr_t[dn], pr_s[dn], False)
        lt, ls = retime(lt, 1 - ls, dq_max)
        ls = 1 - ls
        lower_src = "video"
    else:
        lt = np.arange(0, 1.1 + 1e-9, DT)
        ls = 1 - minjerk_s(len(lt))
        lt, ls = retime(lt, 1 - ls, dq_max)
        ls = 1 - ls
        lower_src = "min-jerk (lowering not visible in the video)"

    def path(s_):
        """rest -> pre-wai (hands apart) for s < S_PRE, then the palms close; keeps the hands from touching."""
        s_ = np.asarray(s_)[:, None]
        a_ = np.clip(s_ / S_PRE, 0, 1)
        b_ = np.clip((s_ - S_PRE) / (1 - S_PRE), 0, 1)
        return q_rest + a_ * (q_pre - q_rest) + b_ * (q_wai - q_pre)

    raise_q, lower_q = path(rs), path(ls)
    for _ in range(3):  # retime on the real joint path until the peak speed fits
        pk_r = float(np.max(np.abs(np.diff(raise_q, axis=0)))) / DT
        pk_l = float(np.max(np.abs(np.diff(lower_q, axis=0)))) / DT
        if max(pk_r, pk_l) <= PEAK_FRACTION * MAX_SPEED:
            break
        if pk_r > PEAK_FRACTION * MAX_SPEED:
            rt, rs = retime(rt, rs, 1.05 * pk_r / float(np.max(np.abs(np.gradient(rs, rt)))))  # effective dq of the path
            raise_q = path(rs)
        if pk_l > PEAK_FRACTION * MAX_SPEED:
            lt, l1 = retime(lt, 1 - ls, 1.05 * pk_l / float(np.max(np.abs(np.gradient(ls, lt)))))
            ls = 1 - l1
            lower_q = path(ls)
    f = w.get("fingers", {})
    hand = [float(np.clip(f.get(k, 1.0), 0.85, 1.0)) for k in ("pinky", "ring", "middle", "index")] + HAND_POSES["flat"][4:]
    relaxed = HAND_POSES["relaxed"]
    hands_raise = [list(np.round(np.array(relaxed) + min(1, 2 * s_) * (np.array(hand) - relaxed), 3)) * 2 for s_ in rs]

    # ---- checks
    gaps = [hand_gap(ptr, raise_q[i]) for i in range(0, len(rs), max(1, len(rs) // 12))] + [hand_gap(ptr, q_wai)]
    lo = np.array([ptr.rng[n][0] for n in UPPER])
    hi = np.array([ptr.rng[n][1] for n in UPPER])
    in_lim = bool(np.all(q_wai >= lo) and np.all(q_wai <= hi))
    peak = max(float(np.max(np.abs(np.diff(raise_q, axis=0)))) / DT, float(np.max(np.abs(np.diff(lower_q, axis=0)))) / DT)
    checks = {"palm_gap_min_m": round(min(gaps), 4), "in_joint_limits": in_lim, "mirror_err_deg": round(np.degrees(mirror_err(q_wai)), 2),
              "peak_speed_rad_s": round(peak, 2), "speed_budget_rad_s": PEAK_FRACTION * MAX_SPEED,
              "ik": info}
    ok = min(gaps) >= 0.005 and in_lim and peak <= PEAK_FRACTION * MAX_SPEED * 1.02 and np.degrees(mirror_err(q_wai)) < 3
    clip = {"name": "wai", "enabled": True, "source": src["source"], "segment_s": [w["t_start"], w["t_end"]],
            "made": __import__("time").strftime("%Y-%m-%d %H:%M"), "hand_model": HAND_MODEL,
            "params": {"tip_z_m": round(tip_z, 3), "tilt_deg": round(tilt, 1), "video_tip_height": k,
                       "raise": raise_src, "lower": lower_src, "video_hold_s": round(th1 - th0, 2)},
            "checks": checks, "ok": bool(ok),
            "t": [round(float(x), 3) for x in rt], "q": np.round(raise_q, 4).tolist(), "hands": hands_raise,
            "phases": [[0.0, "raise"], [round(float(rt[-1]), 3), "hold"], [round(float(rt[-1]), 3), "lower"]],
            "wai_q": np.round(q_wai, 4).tolist(), "wai_hands": hand * 2,
            "lower_t": [round(float(x), 3) for x in lt], "lower_q": np.round(lower_q, 4).tolist()}
    print(json.dumps({k: clip[k] for k in ("params", "checks", "ok")}, indent=1))
    print(f"raise {rt[-1]:.2f} s, lower {lt[-1]:.2f} s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(clip))
        print("candidate", a.out)
    elif not a.no_install:
        if not ok:
            print(f"checks failed: {WAI_CLIP.relative_to(EXP05)} not written")
        else:
            out = WAI_CLIP
            out.write_text(json.dumps(clip))
            print(out)
    compare(src, w, rt, rs, raise_q, clip, Path(a.source))


def compare(src, w, rt, rs, raise_q, clip, src_path):
    """Video frames next to the robot at the same progress (raise, hold)."""
    try:
        import cv2
    except ImportError:  # opencv lives in the ai env; the g1 env renders the robot only
        cv2 = None
    from PIL import Image, ImageDraw

    import view3d
    from view3d import View3D
    view3d.CAMERAS["cmp"] = ([0.15, 0.0, 1.05], 1.5, 180.0, -3.0)
    v = View3D()
    picks = [0.1, 0.4, 0.7, 1.0]
    tiles = []
    for s_ in picks:
        i = int(np.argmin(np.abs(rs - s_))) if s_ < 1.0 else len(rs) - 1
        q29 = np.zeros(29)
        for n, x in zip(UPPER, raise_q[i] if s_ < 1.0 else clip["wai_q"]):
            q29[MOTORS.index(n)] = x
        im = Image.open(io.BytesIO(v.render(q29, np.array(clip["hands"][i] if s_ < 1.0 else clip["wai_hands"]), "cmp", None, timeout=30)))
        im = im.crop((160, 40, 480, 440)).resize((240, 300))
        frame = None
        vid = EXP05 / src["source"] if not Path(src["source"]).is_absolute() else Path(src["source"])
        if cv2 is not None and vid.exists():
            pt, ps = np.array(w["progress"]["t"]), np.array(w["progress"]["s"])
            th0 = w["t_hold"][0] - w["t_start"]
            j = int(np.argmin(np.abs(ps[pt <= th0 + 0.3] - s_))) if s_ < 1.0 else int(np.searchsorted(pt, th0 + 0.5))
            cap = cv2.VideoCapture(str(vid))
            cap.set(cv2.CAP_PROP_POS_MSEC, (w["t_start"] + pt[min(j, len(pt) - 1)]) * 1000)
            ok, fr = cap.read()
            if ok:
                frame = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                frame = frame.resize((int(300 * frame.width / frame.height), 300))
        tile = Image.new("RGB", ((frame.width if frame else 0) + 240, 320), "white")
        if frame:
            tile.paste(frame, (0, 0))
        tile.paste(im, (frame.width if frame else 0, 0))
        ImageDraw.Draw(tile).text((6, 302), f"progress {s_:.1f}" if s_ < 1 else "hold", fill=(0, 0, 0))
        tiles.append(tile)
    W = sum(t.width for t in tiles)
    sheet = Image.new("RGB", (W, 320), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    out = src_path.with_name(src_path.stem + f"_compare_{HAND_MODEL}.jpg")
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
