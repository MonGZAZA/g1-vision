"""Motion clips = action timelines for the upper body: time -> (15 upper joint goals, 12 hand goals or None), plus
speech items (audio + captions) on the same clock. The server samples the active clip at 50 Hz, adds the idle
(keep-alive) layer and hands the goals to the arm channel (which still clips and rate-limits them).

Transitions are minimum-jerk. Their duration follows the largest joint move so the peak speed stays under the arm
channel's speed limit (PEAK_FRACTION of it), so the channel never has to cut the motion short.
"""

import math

import numpy as np

from arm_channel import HAND_POSES, MAX_SPEED
from robot_model import UPPER

PEAK_FRACTION = 0.8
MIN_MOVE_S = 0.6
WAIST = UPPER.index("waist_yaw_joint")
SIDE_IX = {s: [i for i, n in enumerate(UPPER) if n.startswith(s)] for s in ("left", "right")}
ARMS = SIDE_IX["left"] + SIDE_IX["right"]
SIDE_HAND = {"right": slice(0, 6), "left": slice(6, 12)}


def minjerk(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def move_time(q0, q1, speed_scale, minimum=MIN_MOVE_S):
    """Duration so that the min-jerk peak speed (1.875 * dq / T) stays under PEAK_FRACTION of the channel limit."""
    dq = float(np.max(np.abs(np.asarray(q1) - np.asarray(q0)))) if np.size(q0) else 0.0
    return max(minimum, 1.875 * dq / (PEAK_FRACTION * MAX_SPEED * speed_scale))


def hands_with(base12, side, pose):
    h = np.asarray(base12, float).copy()
    sides = ("left", "right") if side == "both" else (side,)
    for s in sides:
        h[SIDE_HAND[s]] = HAND_POSES[pose] if isinstance(pose, str) else pose
    return h


class Clip:
    """Per-joint segments + hand events + speech items.

    seg(joints, t0, t1, q_to): min-jerk from the joint's value at t0 to q_to[j] at t1;
    dense(joints, t0, times, Q): sampled trajectory (from a video clip), linear between samples;
    hands(t, v12): hand goals switch at t (the arm channel rate-limits the fingers);
    speak(t, utterance): audio starts at t, captions follow its phrases;
    phase(t, name): phase marks for the UI;
    idle_scale / glance: how much of the server's idle layer is added (pointing and wai keep it small).
    """

    def __init__(self, q_start, name, info=None):
        self.q0 = np.asarray(q_start, float).copy()
        self.name, self.info = name, info or {}
        self.segs = [[] for _ in UPPER]  # per joint: ("m", t0, t1, a, b) or ("d", t0, t1, times, values)
        self.hand_events = []
        self.phases = []
        self.speech = []  # [t, utterance]
        self.marks = {}
        self.idle_scale, self.glance = 0.6, False
        self.duration = 0.0

    def seg(self, joints, t0, t1, q_to):
        for j in joints:
            self.segs[j].append(("m", t0, t1, self.at(j, t0), float(q_to[j])))
            self.segs[j].sort(key=lambda s: s[1])
        self.duration = max(self.duration, t1)

    def dense(self, joints, t0, times, Q):
        times = np.asarray(times, float) + t0
        Q = np.asarray(Q, float)
        for j in joints:
            self.segs[j].append(("d", float(times[0]), float(times[-1]), times, Q[:, j].copy()))
            self.segs[j].sort(key=lambda s: s[1])
        self.duration = max(self.duration, float(times[-1]))

    def at(self, j, t):
        v = self.q0[j]
        for s in self.segs[j]:
            kind, t0, t1 = s[0], s[1], s[2]
            if t >= t1:
                v = s[4] if kind == "m" else s[4][-1]
            elif t > t0:
                if kind == "m":
                    a, b = s[3], s[4]
                    return a + (b - a) * minjerk((t - t0) / (t1 - t0))
                return float(np.interp(t, s[3], s[4]))
            else:
                break
        return v

    def q_at(self, t):
        return np.array([self.at(j, t) for j in range(len(UPPER))])

    def hands(self, t, v12):
        self.hand_events.append((t, np.asarray(v12, float).copy()))
        self.hand_events.sort(key=lambda e: e[0])

    def hand_at(self, t, default):
        h = default
        for te, v in self.hand_events:
            if te <= t:
                h = v
        return h

    def phase(self, t, name):
        self.phases.append((t, name))
        self.phases.sort(key=lambda e: e[0])

    def speak(self, t, utt):
        self.speech.append([t, utt])
        self.duration = max(self.duration, t + utt.duration)

    def sample(self, t):
        q = self.q_at(t)
        h = None
        for te, v in self.hand_events:
            if te <= t:
                h = v
        ph = ""
        for tp, name in self.phases:
            if tp <= t:
                ph = name
        return q, h, ph

    def caption(self, t):
        """(utterance, phrase index, phrase text) spoken at time t, or None."""
        for ts, u in reversed(self.speech):
            if t >= ts:
                c = u.caption_at(t - ts)
                return (u, c[0], c[1]) if c else None
        return None

    def done(self, t):
        return t >= self.duration


def move_clip(q_start, q_goal, speed_scale, name="move", hands_v12=None):
    c = Clip(q_start, name)
    T = move_time(q_start, q_goal, speed_scale)
    c.seg(range(len(UPPER)), 0.0, T, np.asarray(q_goal, float))
    if hands_v12 is not None:
        c.hands(0.0, hands_v12)
    c.phase(0.0, "moving")
    c.phase(T, "done")
    c.idle_scale = 1.0
    return c


def point_clip(q_start, q_point, q_rest, side, hands_now, speed_scale, hold_s=4.0, info=None, utt=None):
    """Torso starts turning -> arm rises (min-jerk) with the hand opening flat -> hold (the sentence is spoken as the
    arm arrives) -> hand relaxes, arm and torso return to rest. hold_s = None holds until the next action."""
    q_start, q_point, q_rest = (np.asarray(v, float) for v in (q_start, q_point, q_rest))
    c = Clip(q_start, "point", info)
    c.idle_scale = 0.4  # pointing stays readable: small idle, no waist glance
    arm = SIDE_IX[side]
    other = SIDE_IX["left" if side == "right" else "right"]
    t_arm = move_time(q_start[arm], q_point[arm], speed_scale)
    t_w = max(0.6, move_time(q_start[[WAIST]], q_point[[WAIST]], speed_scale))
    lead = 0.25  # the torso (head) leads, so the turn reads as "looking there"
    c.seg([WAIST], 0.0, t_w, q_point)
    c.seg(arm, lead, lead + t_arm, q_point)
    c.seg(other, 0.0, move_time(q_start[other], q_rest[other], speed_scale), q_rest)
    h = hands_with(hands_now, side, "flat")
    h = hands_with(h, "left" if side == "right" else "right", "relaxed")
    c.hands(lead, h)
    t_up = max(t_w, lead + t_arm)
    c.marks["arm_up"] = t_up
    c.phase(0.0, "raise")
    c.phase(t_up, "hold")
    if utt is not None:
        t_say = max(0.0, t_up - 0.4)  # start speaking as the arm arrives
        c.speak(t_say, utt)
        c.marks["speech"] = t_say
        if hold_s is not None:
            hold_s = max(hold_s, t_say + utt.duration + 0.5 - t_up)
    if hold_s is None:
        c.duration = math.inf
        return c
    t_ret = t_up + hold_s
    c.hands(t_ret, hands_with(h, side, "relaxed"))
    t_back = move_time(q_point[arm], q_rest[arm], speed_scale)
    c.seg(arm, t_ret, t_ret + t_back, q_rest)
    c.seg([WAIST], t_ret + 0.2, t_ret + 0.2 + max(0.6, move_time(q_point[[WAIST]], q_rest[[WAIST]], speed_scale)), q_rest)
    c.phase(t_ret, "return")
    c.phase(c.duration, "done")
    return c
