"""Actions -> timelines (motion + speech + captions on one clock).

  intro    wai ("สวัสดีค่ะ" / "Hello" spoken in the wai's hold), then the self-introduction with gestures
  talk     any text with optional gesture tags (service introduction, answers): gestures at phrase starts
  point    torso turn + open-palm pointing, the booth sentence spoken as the arm arrives
The idle layer is added by the server on top of every clip (and when no action runs), so the robot never freezes.

Gesture scheduling: a [tag] in the text starts that gesture at the next phrase; phrases without a tag that last
over BEAT_MIN_S get a small beat (alternating arms, every second phrase); a gesture is held while its phrase is
spoken, then the arms go back to rest (or straight to the next gesture).
"""

import json
from pathlib import Path

import numpy as np

import scripts
from arm_channel import HAND_POSES
from gestures import rest_q, solve_poses, style
from motion import ARMS, SIDE_IX, WAIST, Clip, hands_with, move_time, point_clip
from robot_model import UPPER, WAI_CLIP

BEAT_MIN_S = 1.0
GESTURE_SIDES = {"present_right": "right", "present_left": "left", "welcome": "both", "both": "both"}
ELBOW = {s: UPPER.index(f"{s}_elbow_joint") for s in ("left", "right")}
WRIST_P = {s: UPPER.index(f"{s}_wrist_pitch_joint") for s in ("left", "right")}


class Behaviour:
    def __init__(self, ptr):
        self.ptr = ptr
        self.poses = {k: np.array(v["q"]) for k, v in solve_poses(ptr).items()}
        self.rest = rest_q()
        self.relaxed = np.array(HAND_POSES["relaxed"] * 2, float)

    # ---------------------------------------------------------------- wai
    def wai_clip_file(self):
        """A wai learned from video (clips/wai_<hand model>.json, see tools/make_wai_clip.py), or None."""
        p = WAI_CLIP
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return d if d.get("enabled", True) else None

    def add_wai(self, c, t0, speed, hold_min=1.2):
        """Wai from the current pose at t0 -> {"t_hold", "t_lower" (earliest lowering), "wai_q", "clip"}.
        With clips/wai_<hand model>.json (learned from a video) the raise follows the video's curve; else min-jerk to the
        procedural wai pose. Clip times are at speed scale 1.0 and stretched for slower speeds."""
        q_from = c.q_at(t0)
        vid = self.wai_clip_file()
        c.phase(t0, "wai: raise")
        if vid is not None:
            k = 1.0 / max(speed, 0.1)
            times, Q = np.array(vid["t"]) * k, np.array(vid["q"])
            t_in = move_time(q_from, Q[0], speed, minimum=0.3)  # from wherever the arms are to the clip's start
            c.seg(range(len(UPPER)), t0, t0 + t_in, Q[0])
            c.dense(range(len(UPPER)), t0 + t_in, times, Q)
            for tt, h in list(zip(times, vid["hands"]))[::5]:
                c.hands(t0 + t_in + tt, np.array(h))
            c.hands(t0 + t_in + times[-1], np.array(vid["wai_hands"]))
            t_hold = t0 + t_in + float(times[-1])
            wai_q = np.array(vid["wai_q"])
        else:
            wai_q = self.poses["wai"].copy()
            wai_q[WAIST] = 0.0
            t_in = move_time(q_from, wai_q, speed, minimum=0.9)
            c.seg(range(len(UPPER)), t0, t0 + t_in, wai_q)
            c.hands(t0 + 0.15, hands_with(self.relaxed, "both", "flat"))
            t_hold = t0 + t_in
        c.phase(t_hold, "wai: hold")
        return {"t_hold": t_hold, "t_lower": t_hold + hold_min, "wai_q": wai_q, "clip": vid}

    def lower_from_wai(self, c, t_lower, w, speed):
        vid = w["clip"]
        c.phase(t_lower, "wai: lower")
        c.hands(t_lower + 0.3, self.relaxed)
        if vid is not None and "lower_t" in vid:
            times = np.array(vid["lower_t"]) / max(speed, 0.1)
            c.dense(range(len(UPPER)), t_lower, times, np.array(vid["lower_q"]))
            return t_lower + float(times[-1])
        t_dn = move_time(w["wai_q"], self.rest, speed, minimum=0.9)
        c.seg(range(len(UPPER)), t_lower, t_lower + t_dn, self.rest)
        return t_lower + t_dn

    # ---------------------------------------------------------------- talking gestures
    def add_gestures(self, c, t_speech, utt, spoken, tags, t_free, speed):
        """Gestures for an utterance starting at t_speech; nothing starts before t_free (e.g. while the wai lowers)."""
        tag_at = {}
        for pos, tag in tags:
            i = scripts.phrase_index(utt.phrases, spoken, pos)
            tag_at.setdefault(i, tag)
        t_busy, beat_side, beat_n, last_present = t_free, "right", 0, "left"
        cur = None  # gesture currently held: (name, until)
        for i, p in enumerate(utt.phrases):
            ts, te = t_speech + p["start"], t_speech + p["end"]
            tag = tag_at.get(i)
            if tag == "present":
                last_present = "right" if last_present == "left" else "left"
                tag = f"present_{last_present}"
            if tag in self.poses and tag != "wai":
                t0 = max(ts - 0.35, t_busy)  # the gesture leads the words slightly
                q_from = c.q_at(t0)
                goal = self.rest.copy()
                for s in ("left", "right") if GESTURE_SIDES[tag] == "both" else (GESTURE_SIDES[tag],):
                    goal[SIDE_IX[s]] = self.poses[tag][SIDE_IX[s]]
                goal[WAIST] = 0.0
                T = move_time(q_from, goal, speed, minimum=0.7)
                c.seg(range(len(UPPER)), t0, t0 + T, goal)
                c.hands(t0, hands_with(self.relaxed, GESTURE_SIDES[tag], "flat"))
                hold_until = max(t0 + T + 0.6, min(te + 0.2, t0 + T + 2.5))
                cur = (tag, hold_until, goal)
                t_busy = hold_until
                c.phase(t0, f"gesture: {tag.split('_')[0]}")
                continue
            if cur is not None and ts >= cur[1]:  # back to rest before the next beat
                T = move_time(cur[2], self.rest, speed, minimum=0.7)
                c.seg(range(len(UPPER)), cur[1], cur[1] + T, self.rest)
                c.hands(cur[1], self.relaxed)
                t_busy, cur = cur[1] + T, None
            if cur is None and p["end"] - p["start"] >= BEAT_MIN_S and ts - 0.2 >= t_busy:
                beat_n += 1
                if (beat_n - 1) % style()["beat_every"] == 0:
                    self.add_beat(c, ts - 0.2, beat_side, speed)
                    t_busy = ts + 1.0
                    beat_side = "left" if beat_side == "right" else "right"
        if cur is not None:
            T = move_time(cur[2], self.rest, speed, minimum=0.7)
            t_back = max(cur[1], t_speech + utt.duration - 0.2)
            c.seg(range(len(UPPER)), t_back, t_back + T, self.rest)
            c.hands(t_back, self.relaxed)

    def add_beat(self, c, t0, side, speed):
        """Small forearm lift and drop (elbow 0.25 rad, wrist 0.15 rad) on one arm."""
        q0 = c.q_at(t0)
        up = q0.copy()
        up[ELBOW[side]] -= 0.25  # negative elbow raises the forearm
        up[WRIST_P[side]] -= 0.15
        c.seg([ELBOW[side], WRIST_P[side]], t0, t0 + 0.4, up)
        c.seg([ELBOW[side], WRIST_P[side]], t0 + 0.4, t0 + 1.0, q0)
        c.hands(t0, hands_with(self.relaxed, side, [0.9, 0.9, 0.92, 0.95, 0.9, 0.5]))
        c.hands(t0 + 1.0, self.relaxed)

    # ---------------------------------------------------------------- actions
    def intro(self, q_now, lang, shop, speech, voice, speed):
        text = scripts.fill(shop, "intro", lang)
        spoken, tags = scripts.split_tags(text)
        utt = speech.render(spoken, lang, voice)
        c = Clip(q_now, "intro", {"lang": lang})
        c.idle_scale = 0.5
        w = self.add_wai(c, 0.0, speed)
        # the first phrase ("สวัสดีค่ะ" / "Hello!") lands in the wai's hold
        t_say = w["t_hold"] + 0.15
        c.speak(t_say, utt)
        first = utt.phrases[0] if utt.phrases else {"end": 0.8}
        t_lower = max(w["t_lower"], t_say + first["end"] + 0.35)
        t_free = self.lower_from_wai(c, t_lower, w, speed)
        tags = [(p, t) for p, t in tags if scripts.phrase_index(utt.phrases, spoken, p) > 0]
        self.add_gestures(c, t_say, utt, spoken, tags, t_free + 0.2, speed)
        c.phase(t_free, "introduce")
        c.phase(c.duration, "done")
        c.marks.update(wai_hold=w["t_hold"], wai_lower=t_lower, speech=t_say)
        return c

    def talk(self, q_now, text, lang, speech, voice, speed, name="talk", info=None):
        spoken, tags = scripts.split_tags(text)
        utt = speech.render(spoken, lang, voice)
        c = Clip(q_now, name, {"lang": lang, **(info or {})})
        c.idle_scale = 0.7
        T = move_time(q_now, self.rest, speed, minimum=0.5)
        c.seg(range(len(UPPER)), 0.0, T, self.rest)
        c.hands(0.0, self.relaxed)
        t_say = 0.2
        c.speak(t_say, utt)
        self.add_gestures(c, t_say, utt, spoken, tags, T, speed)
        c.phase(0.0, "speaking")
        c.phase(c.duration, "done")
        c.marks["speech"] = t_say
        return c

    def point(self, q_now, plan, booth, lang, shop, speech, voice, speed, hold_s, hands_now, text=None):
        """text: what to say while pointing (default: the shop's booth sentence, e.g. an answer instead)."""
        name = booth.get(f"name_{lang}") or booth["id"]
        utt = None
        if speech is not None:
            utt = speech.render(text or scripts.fill(shop, "point", lang, name=name), lang, voice)
        info = {"booth": booth["id"], "side": plan["side"], "azimuth_deg": round(plan["azimuth_deg"], 1),
                "distance_m": round(plan["distance_m"], 2), "torso_deg": round(plan["torso_deg"], 1),
                "elev_deg": round(plan["elev_deg"], 1), "lang": lang}
        return point_clip(q_now, plan["q"], self.rest, plan["side"], hands_now, speed, hold_s, info, utt)
