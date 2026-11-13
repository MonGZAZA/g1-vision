"""Procedural upper-body poses for the greeter: the wai and the talking gestures, plus the idle (keep-alive) layer.

Poses are solved once by IK on the robot model (hand frame: fingers along local -y, palm normal along local -x) and
cached in exp05/cache/poses_<hand model>.json. Robot frame: x forward, y left, z up from the floor; the robot stands at the origin.

Wai (task 1): palms together in front of the chest, fingers up, thumbs toward the chest; a gap of WAI_GAP between
the palms (no pressing: protects the fingers). Fingertips at WAI_TIP_Z (about chin height). No bow, no knee dip.
Dex3: the hands are also turned WAI_HAND_TURN_DEG (35) inward about the vertical, because the folded thumb bases
would touch first; the finger edges close in front instead (Inspire: 0, palm to palm).
A clip learned from a video (clips/wai_<hand model>.json, tools/make_wai_clip.py) replaces the procedural one when present.

Gestures (task 2, while speaking), all small and in front of the body (customers are close):
  present  one open palm, forearm forward-out, palm up ("here is ...")
  welcome  both open palms out to the sides, palms up ("welcome")
  both     both hands in front, palms up-in ("explaining")
  beat     a small forearm lift with a relaxed hand, at a phrase start
Idle: slow sums of sines on shoulders, elbows and wrists (2-4 deg), a breathing component, a rare small waist
glance, small finger drift. The robot is never frozen while it stands taken over.
"""

import json
import math
from pathlib import Path

import numpy as np

from pointing import hand_error, hand_frame, solve_hand
from robot_model import ARM_REST, HAND_BODY_PREFIX, HAND_MODEL, POSE_CACHE, UPPER, set_hand_qpos

CACHE = POSE_CACHE
POSE_VERSION = 5
WAI_TIP_Z = 1.17  # m, fingertips (about chin height on the G1)
WAI_X = 0.27  # m, wrists in front of the robot's centre
WAI_GAP = 0.015  # m between the palm surfaces
# m, palm surface (Dex3: the folded thumb) to the wrist centre along the palm normal; wrist to fingertips (flat hand).
# Start values only: solve_wai widens the wai until the measured gap between the hand geoms is WAI_GAP.
PALM_HALF, FINGER_LEN = {"dex3": (0.046, 0.165), "inspire": (0.038, 0.20)}[HAND_MODEL]
# Dex3: the thumb base stands ~3.2 cm proud of the finger pads (any thumb angle), so with the palms facing each other
# the thumb bases touch first and the fingers stay ~9 cm apart. Turning each hand about the vertical by this angle
# (the finger edges come together in front, the thumb bases move back toward the chest) closes the fingers to ~3.7 cm.
# User's choice 2026-11-13 (35 deg). Inspire hands meet palm to palm: 0.
WAI_HAND_TURN_DEG = {"dex3": 35.0, "inspire": 0.0}[HAND_MODEL]
IX = {n: i for i, n in enumerate(UPPER)}
SIDE_IX = {s: [i for i, n in enumerate(UPPER) if n.startswith(s)] for s in ("left", "right")}


def rest_q():
    return np.array([ARM_REST.get(n, 0.0) for n in UPPER])


def wai_targets(yw=PALM_HALF + WAI_GAP / 2, tip_z=WAI_TIP_Z, tilt_deg=3.0):
    """Wrist targets of the wai: fingers up, leaning tilt_deg away from the body, fingertips at tip_z."""
    tl = math.radians(tilt_deg)
    f = [math.sin(tl), 0.0, math.cos(tl)]
    zt = tip_z - FINGER_LEN * math.cos(tl)
    xw = WAI_X - FINGER_LEN * math.sin(tl) / 2  # the fingertips lean forward, the wrists move back a little
    a = math.radians(WAI_HAND_TURN_DEG)
    return {"left": ([xw, yw, zt], f, [math.sin(a), -math.cos(a), 0]),
            "right": ([xw, -yw, zt], f, [math.sin(a), math.cos(a), 0])}


def mirror_arm(q15, src):
    """Copy the src arm's joints to the other arm, mirrored (roll and yaw joints change sign)."""
    dst = "right" if src == "left" else "left"
    q = np.asarray(q15, float).copy()
    for i in SIDE_IX[src]:
        n = UPPER[i]
        q[IX[n.replace(src, dst, 1)]] = -q[i] if ("roll" in n or "yaw" in n) else q[i]
    return q


def arm_asymmetry(q15):
    """Largest left/right difference (rad) of a pose after mirroring."""
    return float(np.max(np.abs(np.asarray(q15, float) - mirror_arm(q15, "left"))))


def solve_wai(ptr, tip_z=WAI_TIP_Z, tilt_deg=3.0, gap=WAI_GAP):
    """Wai pose by IK, widened until the real hand geometry keeps WAI_GAP between the palms. -> (q15, info)"""
    def solve(tg):
        """Both orders (left arm first, then right first); the second arm starts from the mirror of the first and
        keeps it (the arm has one redundant DoF: unseeded, the two arms can settle on different elbow swivels, e.g.
        17 deg asymmetry on the Dex3 model). The pair closer to symmetric (then: lower IK error) wins."""
        best = None
        for order in (("left", "right"), ("right", "left")):
            q, err = rest_q(), {}
            for k, side in enumerate(order):
                p, f, n = tg[side]
                if k == 1:
                    q = mirror_arm(q, order[0])
                    pe, re_, coll = hand_error(ptr, side, q, np.array(p, float), hand_frame(f, n))
                    other = order[0]
                    coll = {c for c in coll if not any(x.startswith((other, HAND_BODY_PREFIX[other])) for x in c)}
                    if pe < 0.002 and re_ < 1.0 and not coll:  # the exact mirror reaches the target: keep it
                        err[side] = [round(pe, 4), round(re_, 1)]
                        continue
                q, pe, re_ = solve_hand(ptr, side, np.array(p, float), hand_frame(f, n), q, seed_base=k == 1,
                                        ignore_other_arm=True)
                err[side] = [round(pe, 4), round(re_, 1)]
            cost = arm_asymmetry(q) + sum(e[0] + 0.002 * e[1] for e in err.values())
            if best is None or cost < best[0]:
                best = (cost, q, err)
        return best[1], best[2]

    yw = PALM_HALF + gap / 2
    q, err = solve(wai_targets(yw, tip_z, tilt_deg))
    for _ in range(6):
        g = hand_gap(ptr, q)
        if g >= gap * 0.8:
            break
        yw += (gap - g) / 2 + 0.002
        q, err = solve(wai_targets(yw, tip_z, tilt_deg))
    err["gap_m"] = round(hand_gap(ptr, q), 4)
    return q, err


def _targets():
    """name -> {side: (wrist position, finger direction, palm normal)} (the wai is solved by solve_wai)"""
    t = {
        "present_right": {"right": ([0.30, -0.25, 0.90], [0.75, -0.55, 0.05], [-0.15, 0.35, 0.92])},
        "present_left": {"left": ([0.30, 0.25, 0.90], [0.75, 0.55, 0.05], [-0.15, -0.35, 0.92])},
        "welcome": {"left": ([0.22, 0.36, 0.88], [0.45, 0.85, -0.05], [0.1, -0.35, 0.93]),
                    "right": ([0.22, -0.36, 0.88], [0.45, -0.85, -0.05], [0.1, 0.35, 0.93])},
        "both": {"left": ([0.30, 0.15, 0.92], [0.9, 0.15, 0.25], [0, -0.6, 0.8]),
                 "right": ([0.30, -0.15, 0.92], [0.9, -0.15, 0.25], [0, 0.6, 0.8])},
    }
    return t


def solve_poses(ptr, force=False):
    """IK of all gesture poses (cached). Returns {name: {"q": 15 joints, "err": [pos m, rot deg] per side}}."""
    if CACHE.exists() and not force:
        d = json.loads(CACHE.read_text())
        if d.get("version") == POSE_VERSION:
            return d["poses"]
    q, err = solve_wai(ptr)
    out = {"wai": {"q": [round(float(v), 4) for v in q], "err": err}}
    for name, tg in _targets().items():
        q = rest_q()
        err = {}
        for side, (p, f, n) in tg.items():
            q, pe, re_ = solve_hand(ptr, side, np.array(p, float), hand_frame(f, n), q)
            err[side] = [round(pe, 4), round(re_, 1)]
        out[name] = {"q": [round(float(v), 4) for v in q], "err": err}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"version": POSE_VERSION, "poses": out}, indent=1))
    return out


def hand_gap(ptr, q15):
    """Smallest distance between the left and right hand collision geoms (m) in pose q15, hands in the "flat" pose."""
    import mujoco
    m, d = ptr.m, ptr.d
    with ptr.lock:
        set_hand_qpos(m, d, "flat")
        ptr._apply(np.asarray(q15, float))
        mujoco.mj_forward(m, d)
        L, R = ([g for g in range(m.ngeom) if m.geom_contype[g]
                 and m.body(m.geom_bodyid[g]).name.startswith(HAND_BODY_PREFIX[s])] for s in ("left", "right"))
        fr = np.zeros(6)
        gap = min(mujoco.mj_geomDistance(m, d, a, b, 0.2, fr) for a in L for b in R)
        ptr._apply(ptr.rest)
    return float(gap)


# ------------------------------------------------------------------ idle (keep-alive) layer
IDLE_AMP = {  # rad, per joint name fragment: (amplitude, frequency Hz, phase)
    "shoulder_pitch": [(0.035, 0.11, 0.0), (0.015, 0.29, 1.3)],
    "shoulder_roll": [(0.02, 0.13, 2.1)],
    "shoulder_yaw": [(0.025, 0.09, 0.7)],
    "elbow": [(0.04, 0.12, 0.4), (0.015, 0.33, 2.5)],
    "wrist_roll": [(0.05, 0.10, 1.9)],
    "wrist_pitch": [(0.04, 0.14, 0.2)],
    "wrist_yaw": [(0.03, 0.17, 2.8)],
}
BREATH = (0.012, 0.23)  # shoulder pitch, both arms together
STYLE_FILE = CACHE.parents[1] / "clips/idle.json"  # learned from a keep-alive video (ai/tools/video_to_idle.py)
_style = {"mtime": None, "amp": 1.0, "speed": 1.0, "beat_every": 2}


def style():
    """Idle amplitude / speed scale and beat spacing; from clips/idle.json when present (re-read when it changes)."""
    try:
        mt = STYLE_FILE.stat().st_mtime
    except OSError:
        _style.update(mtime=None, amp=1.0, speed=1.0, beat_every=2)
        return _style
    if mt != _style["mtime"]:
        try:
            d = json.loads(STYLE_FILE.read_text())
            if d.get("enabled", True):
                _style.update(amp=float(d["idle_amp_scale"]), speed=float(d["idle_speed_scale"]),
                              beat_every=int(d["beat_every"]))
        except (ValueError, KeyError):
            pass
        _style["mtime"] = mt
    return _style


def idle(t, scale=1.0, glance=True):
    """Additive idle offsets (15 joints) at time t (s). Left and right use different phases so they never mirror."""
    st = style()
    scale, t = scale * st["amp"], t * st["speed"]
    out = np.zeros(len(UPPER))
    for i, n in enumerate(UPPER):
        if n == "waist_yaw_joint":
            if glance:  # a slow small look around: mostly still, a 3 deg glance every ~20 s
                g = math.sin(2 * math.pi * t / 21.0)
                out[i] = math.radians(3.0) * math.copysign(max(0.0, abs(g) - 0.6) / 0.4, g) * scale
            continue
        ph = 0.0 if n.startswith("left") else 1.7
        for key, comps in IDLE_AMP.items():
            if key in n:
                out[i] = sum(a * math.sin(2 * math.pi * f * t + p + ph) for a, f, p in comps) * scale
        if "shoulder_pitch" in n:
            out[i] += BREATH[0] * math.sin(2 * math.pi * BREATH[1] * t) * scale
    return out


def idle_hands(t, base12, scale=1.0):
    """Small finger drift around a hand goal (12 values, 1 = open)."""
    h = np.asarray(base12, float).copy()
    for k in range(12):
        h[k] = np.clip(h[k] + 0.04 * scale * math.sin(2 * math.pi * (0.07 + 0.01 * k) * t + k), 0.0, 1.0)
    return h
