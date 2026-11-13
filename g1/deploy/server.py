"""exp05 operator server (port 8100): robot link, upper-body channel, shops, and the operator page.

S0: status, take-over / release / E-STOP, speed scale, test poses, hand poses. S1: shops (editor, 3D view).
S2: pointing at booths. Wai, introduction, voice and captions come in S3-S6.

    .venv/bin/python deploy/server.py            # simulation: DDS domain 5 on lo (start deploy/sim_robot.py first)
    .venv/bin/python deploy/server.py --net <if> # real robot (later, S7): domain 0 on the robot's interface

HTTP API
    GET  /                    operator page (static/index.html)
    GET  /api/status          link, arm channel, hands, balance (sim)
    POST /api/arm             {"action": "takeover" | "release" | "estop"}
    POST /api/speed           {"scale": 0.1 .. 1.0}
    POST /api/pose            {"name": <TEST_POSES key>}   (arms must be taken over)
    POST /api/hands           {"pose": <HAND_POSES key>, "side": "left" | "right" | "both"}
    POST /api/point           {"booth": ID, "hold_s": 4.0 | null}   point at a booth of the active shop (takes over the
                              arms if needed); hold_s null = hold until the next action
    POST /api/rest            arms back to the rest pose (stay taken over)
    GET  /api/point/plans     pointing plan per booth of the active shop (side, torso, elevation, IK result)
    GET  /api/shops           {"shops": [...], "active": name}
    GET  /api/shop?name=N     shop.json of N
    POST /api/shop/save       {"shop": {...}}         validated, previous version kept as shop.prev.json
    POST /api/shop/new        {"name": N, "copy_from": M?}
    POST /api/shop/active     {"name": N}             the shop the robot works in (pointing, 3D preview)
    POST /api/shop/background?name=N   body = image bytes (PNG/JPEG/WebP, <= 30 MB)
    GET  /shops/N/<image>     shop images
    GET  /api/view.jpg?cam=behind|front|top&booth=ID   3D preview of the measured pose + booths
    POST /api/settings        also {"speech_rate": 0.8-1.2} (TTS speed; the motion follows the measured phrase times)
                              and {"volume": 0-100} (for the current audio output; robot = its speaker); both saved
    POST /api/video/upload?kind=wai|idle&name=FILE&start=S&end=E   body = video bytes (<= 500 MB): learn the wai /
                              keep-alive style in the background (video_jobs.py); GET /api/video/job = progress
    POST /api/video/decide    {"use": true|false}     install the processed candidate (old one -> clips/backup/) or not
    POST /api/video/restore   {"kind": "wai"|"idle"}  put back the version before the last "use"
    POST /api/listen          {"mode": "off"|"ptt"|"auto", "reply": "auto"|"review", "relevance": bool,
                              "presence_gate": bool}   speech recognition (deploy/listen.py); GET /api/listen = state,
                              mic level, last recognised lines (also in /api/status "listen")
    POST /api/listen/ptt      {"down": bool}           push-to-talk button (mode ptt)
    POST /api/listen/presence {"present": bool, "hold_s": 2}   customer in front (vision, 2.3; for the presence gate)
    POST /api/listen/audio    body = 16-bit mono WAV: heard as if from the microphone (tests, recorded questions)
"""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen 3D preview

import numpy as np

import qa
import shops
import scripts
from arm_channel import HAND_POSES, ArmChannel
from listen import Listener
from behaviour import Behaviour
from gestures import idle, idle_hands
from motion import move_clip
from pointing import Pointer
from speech import Speech, SpeechError, make_outputs
from robot_model import ARM_REST, HAND_MODEL, HANDS, UPPER, WAI_CLIP, shop_booths_robot_frame
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from urllib.parse import parse_qs, urlparse
from video_jobs import VideoError, VideoJobs
from view3d import View3D, booth_geometry

HERE = Path(__file__).resolve().parent
EXP05 = HERE.parents[1]
ACTIVE_FILE = shops.SHOPS / ".active"
AUDIO_SETTINGS = EXP05 / "cache/audio_settings.json"  # speaking speed + volume per output


def pose(**kw):
    q = np.array([ARM_REST.get(n, 0.0) for n in UPPER])
    for k, v in kw.items():
        q[UPPER.index(k + "_joint")] = v
    return q


TEST_POSES = {  # S0 only: to see the channel work; real motions come from the motion layer
    "rest": pose(),
    "left_forward": pose(left_shoulder_pitch=-1.2, left_elbow=1.2),
    "right_forward": pose(right_shoulder_pitch=-1.2, right_elbow=1.2),
    "turn_left_15": pose(waist_yaw=np.radians(15)),
    "turn_right_15": pose(waist_yaw=np.radians(-15)),
}


class Server:
    def __init__(self, args):
        ChannelFactoryInitialize(args.domain, args.net)
        self.sim = args.net == "lo"
        self.arm = ArmChannel(hands=args.hands)
        self.balance = None
        if self.sim:
            ChannelSubscriber("rt/sim/balance", String_).Init(lambda m: setattr(self, "balance", json.loads(m.data)), 10)
        self.goal = None
        self.clip, self.clip_t0, self.pending, self.last_hands = None, 0.0, None, None
        self.action = {"name": "", "phase": ""}
        self.said = set()  # speech items of the current clip already started
        self.caption, self.captions = None, []  # current caption, recent ones (monitor)
        self.hand_base = np.array(HAND_POSES["relaxed"] * 2, float)
        self.t_idle0 = time.monotonic()
        self.pointer = Pointer()
        self.behaviour = Behaviour(self.pointer)
        self.speech = Speech(args.ai_url)
        self.videos = VideoJobs()
        self.outputs = make_outputs(real_robot=not self.sim)
        self.settings = {"lang": "th", "audio": "laptop" if self.sim else "robot", "voice": "default", "idle": True}
        self._load_audio_settings()
        self.ai = None
        threading.Thread(target=self._watch_ai, daemon=True).start()
        self.plans, self.plans_lock, self.solve_lock = {}, threading.Lock(), threading.Lock()
        self.view = View3D()
        self.shop = None
        name = ACTIVE_FILE.read_text().strip() if ACTIVE_FILE.exists() else "mockup"
        try:
            self.set_active(name)
        except shops.ShopError:
            pass
        threading.Thread(target=self._hold, daemon=True).start()
        self.listener = Listener(self, args.listen_source, args.listen_device, args.net)

    def speaking(self):
        """True while the robot speaks or is about to (an action is queued, or the active clip has a speech item that has
        not ended yet, e.g. the arm rising before an answer): the microphone must not take the robot's own voice as a
        question, and a new question must not interrupt an answer being given."""
        if self.pending is not None:
            return True
        c = self.clip
        if c is None:
            return False
        t = time.monotonic() - self.clip_t0
        return any(t < ts + u.duration for ts, u in c.speech)

    def set_active(self, name):
        self.shop = shops.load(name)
        self.view.set_shop(self.shop)
        ACTIVE_FILE.write_text(name + "\n")
        with self.plans_lock:
            self.plans = {}
        threading.Thread(target=self.all_plans, daemon=True).start()  # warm the cache (~0.1 s per booth)
        threading.Thread(target=self.prerender, daemon=True).start()

    # ---- speech
    def fixed_texts(self, shop=None):
        """Every fixed text of the shop (both languages): introductions, service, booth sentences, filler."""
        shop = shop or self.shop
        out = []
        for lang in ("th", "en"):
            for key in ("intro", "service", "filler", "unknown"):
                out.append((scripts.split_tags(scripts.fill(shop, key, lang))[0], lang))
            for b in shop.get("booths", []):
                out.append((scripts.fill(shop, "point", lang, name=b.get(f"name_{lang}") or b["id"]), lang))
        return out

    # ---- speaking speed + volume (saved in cache/audio_settings.json; volume per output)
    def _load_audio_settings(self):
        try:
            d = json.loads(AUDIO_SETTINGS.read_text())
        except (OSError, ValueError):
            d = {}
        self.speech.rate = float(np.clip(float(d.get("speech_rate", 1.0)), 0.8, 1.2))
        self.settings["speech_rate"] = self.speech.rate
        vols = d.get("volume", {})
        self.settings["volume"] = {}
        for name, o in self.outputs.items():
            if name == "off":
                continue
            if name in vols:
                if name == "robot":  # DDS call to the robot: not in the start-up path
                    threading.Thread(target=o.set_volume, args=(vols[name],), daemon=True).start()
                else:
                    o.set_volume(vols[name])
                self.settings["volume"][name] = int(vols[name])
            elif name == "robot":  # not set from exp05 yet: show the robot's own level once it can be read
                self.settings["volume"][name] = None
                threading.Thread(target=self._read_robot_volume, daemon=True).start()
            else:
                self.settings["volume"][name] = o.get_volume()

    def _read_robot_volume(self):
        for _ in range(10):
            v = self.outputs["robot"].get_volume()
            if v is not None:
                self.settings["volume"]["robot"] = v
                return
            time.sleep(3)

    def _save_audio_settings(self):
        AUDIO_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        AUDIO_SETTINGS.write_text(json.dumps({"speech_rate": self.speech.rate, "volume": {
            k: v for k, v in self.settings["volume"].items() if v is not None}}, indent=1))

    def prerender(self):
        """Render the fixed texts in the background so the buttons start without waiting for the TTS."""
        shop, voice, rate = self.shop, self.settings["voice"], self.speech.rate

        def changed():
            return shop is not self.shop or voice != self.settings["voice"] or rate != self.speech.rate

        for _ in range(120):  # the AI service may still be starting: retry for ~10 min
            if changed():
                return
            try:
                for text, lang in self.fixed_texts(shop):
                    if changed():
                        return
                    self.speech.render(text, lang, voice)
                return
            except SpeechError:
                time.sleep(5)

    def _watch_ai(self):
        while True:
            self.ai = self.speech.health()
            time.sleep(3)

    def out(self):
        return self.outputs.get(self.settings["audio"], self.outputs["off"])

    def stop_audio(self):
        for o in self.outputs.values():
            o.stop()

    # ---- pointing plans (IK per booth, cached per active shop)
    def plan(self, booth_id):
        with self.solve_lock:  # one solver at a time; the warm-up thread and requests share the cache
            shop = self.shop
            with self.plans_lock:
                if booth_id in self.plans:
                    return self.plans[booth_id]
            b = next((b for b in shop_booths_robot_frame(shop) if b["id"] == booth_id), None) if shop else None
            if b is None:
                return None
            r = self.pointer.solve(b["aim"], shop.get("settings"))
            r["booth"] = booth_id
            with self.plans_lock:
                if shop is self.shop:  # the shop did not change meanwhile
                    self.plans[booth_id] = r
            return r

    def all_plans(self):
        if not self.shop:
            return []
        out = []
        for b in self.shop["booths"]:
            r = self.plan(b["id"])
            if r is not None:
                out.append({k: v for k, v in r.items() if k != "q"} | {"name_en": b["name_en"], "name_th": b["name_th"]})
        return out

    # ---- motion: the active clip is sampled at 50 Hz; new actions start from the current goal (smooth)
    def start(self, builder, action):
        """Queue a clip builder(q_now) -> Clip; it starts once the arms are fully taken over."""
        if self.arm.status()["mode"] in ("released", "releasing"):
            if not self.arm.takeover():
                return False, self.arm.reason
        self.pending = (builder, action)
        return True, ""

    def _hold(self):
        """50 Hz: while the arms are taken over, send the active clip's sample (else the last goal) plus the idle
        layer; start the clip's speech items on its clock; keep the caption."""
        while True:
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001  never let the motion thread die (the channel would release)
                print(f"[SERVER] motion tick: {type(e).__name__}: {e}", flush=True)
            time.sleep(0.02)

    def _tick(self):
        st = self.arm.status()["mode"]
        now = time.monotonic()
        if st not in ("takeover", "active"):
            if self.clip is not None or self.goal is not None:
                self.stop_audio()
            self.goal, self.clip, self.last_hands, self.caption = None, None, None, None
            if st == "released":
                self.pending = None
            if self.action.get("phase") not in ("", "done", "stopped"):
                self.action = {**self.action, "phase": "stopped"}
            return
        if self.goal is None:
            q = self.arm.upper_measured()
            self.goal = q if q is not None else TEST_POSES["rest"]
        if st == "active" and self.pending is not None:
            (builder, action), self.pending = self.pending, None
            try:
                clip = builder(self.goal.copy())
            except Exception as e:  # noqa: BLE001
                self.action = {**action, "phase": "failed", "reason": str(e)}
                clip = None
            if clip is not None:
                self.stop_audio()
                self.clip, self.clip_t0, self.said = clip, now, set()
                self.action = {**action, "marks": {k: round(v, 3) for k, v in clip.marks.items()}}
        scale, glance = 1.0, True
        if self.clip is not None:
            c, t = self.clip, now - self.clip_t0
            q, h, ph = c.sample(t)
            self.goal = q
            scale, glance = c.idle_scale, c.glance
            if h is not None:
                self.hand_base = h
            lat = self.out().LATENCY
            for k, (ts, u) in enumerate(c.speech):
                if k not in self.said and t >= ts - lat:
                    self.said.add(k)
                    self.out().play(u)
            cap = c.caption(t)
            self._set_caption(cap)
            self.action = {**self.action, "phase": ph, "t": round(t, 2),
                           "duration": None if c.duration == float("inf") else round(c.duration, 2)}
            if c.done(t):
                self.clip = None
                self.action = {**self.action, "phase": "done"}
                self._set_caption(None)
        if self.settings["idle"]:
            ti = now - self.t_idle0
            self.arm.set_goal(self.goal + idle(ti, scale, glance))
            h = idle_hands(ti, self.hand_base, scale)
        else:
            self.arm.set_goal(self.goal)
            h = self.hand_base
        if self.last_hands is None or np.max(np.abs(h - self.last_hands)) > 0.004:
            self.arm.set_hands(h)
            self.last_hands = h

    def _set_caption(self, cap):
        if cap is None:
            self.caption = None
            return
        u, i, text = cap
        if self.caption is None or self.caption["text"] != text or self.caption["i"] != i:
            self.caption = {"text": text, "i": i, "n": len(u.phrases), "lang": u.lang, "full": u.text,
                            "time": time.time()}
            self.captions = (self.captions + [self.caption])[-30:]

    def clips_info(self):
        """Motion learned from video: the wai clip and the keep-alive style (clips/*.json)."""
        out = {"ok": True, "wai": None, "idle": None, "hand_model": HAND_MODEL}
        p = WAI_CLIP
        if p.exists():
            d = json.loads(p.read_text())
            src = Path(d.get("source", "")).stem
            cmp_ = sorted((EXP05 / "clips/source").glob(f"{src}_*_wai_compare_{HAND_MODEL}.jpg"))
            out["wai"] = {k: d.get(k) for k in ("source", "segment_s", "made", "params", "checks", "ok", "enabled")}
            out["wai"]["compare"] = cmp_[-1].name if cmp_ else None
        p = EXP05 / "clips/idle.json"
        if p.exists():
            out["idle"] = json.loads(p.read_text())
        return out

    def live(self):
        return {"action": self.action, "caption": self.caption, "mode": self.arm.status()["mode"],
                "settings": self.settings, "shop": self.shop["name"] if self.shop else None, "time": time.time()}

    def status(self):
        return {"sim": self.sim, "hands": self.arm.hand_type, "arm": self.arm.status(), "joints": UPPER, "balance": self.balance,
                "shop": self.shop["name"] if self.shop else None, "action": self.action,
                "caption": self.caption, "settings": self.settings, "ai": self.ai,
                "audio_outputs": list(self.outputs), "listen": self.listener.status(),
                "test_poses": list(TEST_POSES), "hand_poses": list(HAND_POSES), "time": time.time()}

    def view_jpg(self, cam, booth):
        with self.arm.lock:
            q = None if self.arm.q_meas is None else self.arm.q_meas.copy()
            h = None if self.arm.hand_meas is None else self.arm.hand_meas.copy()
        return self.view.render(q, h, cam, booth)

    def post(self, path, body):
        if path.startswith("/api/shop/"):
            try:
                if path == "/api/shop/save":
                    shop = shops.save(body.get("shop"))
                    if self.shop and shop["name"] == self.shop["name"]:
                        self.set_active(shop["name"])
                    return {"ok": True, "shop": shop}
                if path == "/api/shop/new":
                    return {"ok": True, "shop": shops.new(body.get("name"), body.get("copy_from") or None)}
                if path == "/api/shop/active":
                    self.set_active(body.get("name"))
                    return {"ok": True}
            except shops.ShopError as e:
                return {"ok": False, "reason": str(e)}
        if path == "/api/arm":
            a = body.get("action")
            if a == "takeover":
                return {"ok": self.arm.takeover(), "reason": self.arm.reason}
            if a in ("release", "estop"):
                self.arm.release("E-STOP" if a == "estop" else "operator")
                return {"ok": True}
        elif path == "/api/speed":
            self.arm.speed_scale = float(np.clip(float(body.get("scale", 0.5)), 0.1, 1.0))
            return {"ok": True, "scale": self.arm.speed_scale}
        elif path == "/api/pose":
            name = body.get("name")
            if name not in TEST_POSES:
                return {"ok": False, "reason": f"unknown pose {name}"}
            if self.arm.status()["mode"] not in ("takeover", "active"):
                return {"ok": False, "reason": "take over the arms first"}
            ok, why = self.start(lambda q: move_clip(q, TEST_POSES[name], self.arm.speed_scale, "pose"),
                                 {"name": f"pose {name}"})
            return {"ok": ok, "reason": why}
        elif path in ("/api/rest", "/api/stop"):  # stop = stop speaking + arms back to rest (they stay taken over)
            self.stop_audio()
            rest_h = np.array(HAND_POSES["relaxed"] * 2, float)
            if self.arm.status()["mode"] not in ("takeover", "active"):
                return {"ok": True}
            ok, why = self.start(lambda q: move_clip(q, TEST_POSES["rest"], self.arm.speed_scale, "rest", rest_h),
                                 {"name": "rest"})
            return {"ok": ok, "reason": why}
        elif path == "/api/point":
            bid = str(body.get("booth", ""))
            hold = body.get("hold_s", 4.0)
            hold = None if hold is None else float(np.clip(float(hold), 0.5, 60.0))
            lang = body.get("lang") or self.settings["lang"]
            r = self.plan(bid)
            booth = next((b for b in self.shop["booths"] if b["id"] == bid), None) if self.shop else None
            if r is None or booth is None:
                return {"ok": False, "reason": f"no booth '{bid}' in the active shop"}
            if not r["ok"]:
                return {"ok": False, "reason": f"{bid}: {r['reason']}"}
            text = body.get("text")  # e.g. an answer spoken while pointing; default: the booth sentence
            warn = ""
            speak = body.get("speak", True)
            try:
                if speak:
                    self.speech.render(text if text else scripts.fill(self.shop, "point", lang,
                                       name=booth.get(f"name_{lang}") or bid), lang, self.settings["voice"])
            except SpeechError as e:
                warn, speak = str(e), False
            shop, voice, sp = self.shop, self.settings["voice"], self.speech if speak else None

            def build(q, r=r):
                return self.behaviour.point(q, r, booth, lang, shop, sp, voice, self.arm.speed_scale, hold,
                                            self.hand_base, text=text)

            ok, why = self.start(build, {"name": "point", "booth": bid, "side": r["side"],
                                         "azimuth_deg": round(r["azimuth_deg"], 1), "distance_m": round(r["distance_m"], 2),
                                         "torso_deg": round(r["torso_deg"], 1), "elev_deg": round(r["elev_deg"], 1),
                                         "lang": lang})
            return {"ok": ok, "reason": why, "warning": warn, "plan": {k: v for k, v in r.items() if k != "q"}}
        elif path in ("/api/intro", "/api/service", "/api/say"):
            lang = body.get("lang") or self.settings["lang"]
            if lang not in ("th", "en"):
                return {"ok": False, "reason": "lang must be th or en"}
            voice, shop, speed = self.settings["voice"], self.shop, self.arm.speed_scale
            if path == "/api/say":
                text = str(body.get("text", "")).strip()
                if not text:
                    return {"ok": False, "reason": "nothing to say"}
            else:
                text = scripts.fill(shop, "intro" if path == "/api/intro" else "service", lang)
            try:
                self.speech.render(scripts.split_tags(text)[0], lang, voice)  # TTS now, not in the motion thread
            except SpeechError as e:
                return {"ok": False, "reason": str(e)}
            if path == "/api/intro":
                def build(q):
                    return self.behaviour.intro(q, lang, shop, self.speech, voice, speed)
            else:
                name = "service" if path == "/api/service" else "say"

                def build(q, name=name):
                    return self.behaviour.talk(q, text, lang, self.speech, voice, speed, name)
            ok, why = self.start(build, {"name": path.rsplit("/", 1)[1], "lang": lang})
            return {"ok": ok, "reason": why}
        elif path == "/api/listen":
            return self.listener.configure(body)
        elif path == "/api/listen/ptt":
            return self.listener.push_to_talk(body.get("down", False))
        elif path == "/api/listen/presence":
            return self.listener.presence(body.get("present", False), body.get("hold_s", 2.0))
        elif path == "/api/ask":
            q = str(body.get("question", "")).strip()[:500]
            if not q:
                return {"ok": False, "reason": "empty question"}
            if not self.shop:
                return {"ok": False, "reason": "no active shop"}
            t = time.time()
            r = qa.answer(self.shop, q, self.speech)
            r["seconds"] = round(time.time() - t, 2)
            try:  # render now, so "Speak" starts at once
                self.speech.render(r["answer"], r["lang"], self.settings["voice"])
            except SpeechError as e:
                r["warning"] = str(e)
            return r
        elif path == "/api/settings":
            for k in ("lang", "audio", "voice"):
                if k in body:
                    v = str(body[k])
                    if k == "lang" and v not in ("th", "en") or k == "audio" and v not in self.outputs:
                        return {"ok": False, "reason": f"bad {k}: {v}"}
                    if k == "audio":
                        self.stop_audio()
                    self.settings[k] = v
                    if k == "voice":
                        threading.Thread(target=self.prerender, daemon=True).start()
            if "speech_rate" in body:  # 0.8-1.2: the TTS speaks slower / faster; motion follows the phrase times
                try:
                    r = round(float(np.clip(float(body["speech_rate"]), 0.8, 1.2)), 2)
                except (TypeError, ValueError):
                    return {"ok": False, "reason": "speech_rate must be a number 0.8-1.2"}
                if r != self.speech.rate:
                    self.speech.rate = self.settings["speech_rate"] = r
                    self._save_audio_settings()
                    threading.Thread(target=self.prerender, daemon=True).start()
            if "volume" in body:  # 0-100 % for the current output (laptop: stream volume; robot: its speaker)
                try:
                    v = int(np.clip(float(body["volume"]), 0, 100))
                except (TypeError, ValueError):
                    return {"ok": False, "reason": "volume must be a number 0-100"}
                name = self.settings["audio"]
                if name != "off":
                    if not self.out().set_volume(v):
                        return {"ok": False, "reason": "the robot did not accept the volume (audio service reachable?)",
                                "settings": self.settings}
                    self.settings["volume"][name] = v
                    self._save_audio_settings()
            if "idle" in body:
                self.settings["idle"] = bool(body["idle"])
            if "audio_latency_s" in body:  # measured per output in S7 (robot speaker), compensated by the scheduler
                self.out().LATENCY = float(np.clip(float(body["audio_latency_s"]), 0.0, 1.0))
            self.settings["audio_latency_s"] = round(self.out().LATENCY, 3)
            return {"ok": True, "settings": self.settings}
        elif path in ("/api/video/decide", "/api/video/restore"):
            try:
                if path == "/api/video/decide":
                    return self.videos.decide(bool(body.get("use")))
                return self.videos.restore_previous(body.get("kind", "wai"))
            except VideoError as e:
                return {"ok": False, "reason": str(e)}
        elif path == "/api/clips":
            p = WAI_CLIP
            if "wai_enabled" in body and p.exists():
                d = json.loads(p.read_text())
                d["enabled"] = bool(body["wai_enabled"])
                p.write_text(json.dumps(d))
            return self.clips_info()
        elif path == "/api/scripts":
            if not self.shop:
                return {"ok": False, "reason": "no active shop"}
            shop = json.loads(json.dumps(self.shop))
            shop["scripts"] = scripts.clean({**scripts.get(self.shop), **(body.get("scripts") or {})})
            try:
                shops.save(shop)
            except shops.ShopError as e:
                return {"ok": False, "reason": str(e)}
            self.set_active(shop["name"])
            return {"ok": True, "scripts": scripts.get(self.shop)}
        elif path == "/api/hands":
            p = HAND_POSES.get(body.get("pose"))
            if p is None:
                return {"ok": False, "reason": "unknown hand pose"}
            v = self.hand_base.copy()
            side = body.get("side", "both")
            if side in ("right", "both"):
                v[0:6] = p
            if side in ("left", "both"):
                v[6:12] = p
            self.hand_base = v
            if self.arm.status()["mode"] not in ("takeover", "active"):
                self.arm.set_hands(v)
            return {"ok": True}
        return None


def wav_to_pcm16k(data):
    import io
    import wave
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{w.getsampwidth() * 8}-bit, {w.getnchannels()} channels")
        sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2")
    if sr != 16000:
        x = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x.astype(float)).astype("<i2")
    return x


def make_handler(srv):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, data, ctype="application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/api/shop":
                try:
                    shop = shops.load(qs.get("name", ""))
                    geo = {b["id"]: booth_geometry(shop, b["id"]) for b in shop["booths"]}
                    return self._send(200, {"ok": True, "shop": shop, "geometry": geo})
                except shops.ShopError as e:
                    return self._send(200, {"ok": False, "reason": str(e)})
            if u.path.startswith("/shops/"):
                parts = u.path.split("/")
                try:
                    p = shops.file_path(parts[2], parts[3]) if len(parts) == 4 else None
                except shops.ShopError:
                    p = None
                if p is None:
                    return self._send(404, {"error": "not found"})
                ctype = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}[p.suffix[1:]]
                return self._send(200, p.read_bytes(), ctype)
            if u.path == "/api/view.jpg":
                jpg = srv.view_jpg(qs.get("cam", "behind"), qs.get("booth") or None)
                return self._send(200, jpg, "image/jpeg") if jpg else self._send(503, {"error": "busy"})
            if u.path == "/api/point/plans":
                return self._send(200, {"ok": True, "shop": srv.shop["name"] if srv.shop else None,
                                        "plans": srv.all_plans()})
            if u.path == "/api/listen":
                return self._send(200, srv.listener.status())
            if u.path == "/api/live":
                return self._send(200, srv.live())
            if u.path == "/api/scripts":
                return self._send(200, {"ok": True, "scripts": scripts.get(srv.shop), "keys": scripts.KEYS})
            if u.path == "/api/clips":
                return self._send(200, srv.clips_info())
            if u.path == "/api/video/job":
                return self._send(200, srv.videos.status())
            if u.path.startswith("/clips/source/"):
                name = u.path.rsplit("/", 1)[1]
                p = EXP05 / "clips/source" / name
                if not (name.endswith(".jpg") and "/" not in name and ".." not in name and p.exists()):
                    return self._send(404, {"error": "not found"})
                return self._send(200, p.read_bytes(), "image/jpeg")
            if u.path == "/api/voices":
                return self._send(200, srv.speech.get("/voices"))
            if u.path == "/monitor":
                return self._send(200, (HERE / "static/monitor.html").read_bytes(), "text/html; charset=utf-8")
            if self.path in ("/", "/index.html"):
                self._send(200, (HERE / "static/index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._send(200, srv.status())
            elif self.path == "/api/shops":
                self._send(200, {"shops": shops.list_shops(), "active": srv.shop["name"] if srv.shop else None})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            u = urlparse(self.path)
            if u.path == "/api/voice/upload":  # raw audio body; name + transcript in the query
                if n > 20 * 1024 * 1024:
                    return self._send(413, {"ok": False, "reason": "audio larger than 20 MB"})
                data = self.rfile.read(n)
                return self._send(200, srv.speech.proxy("/voice/upload?" + u.query, data,
                                                        self.headers.get("Content-Type", "application/octet-stream")))
            if u.path == "/api/video/upload":  # raw video body; kind, file name, start/end (s) in the query
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                try:
                    st = srv.videos.start(q.get("kind", ""), q.get("name", ""), self.rfile, n,
                                          float(q.get("start") or 0), float(q.get("end") or 0))
                    return self._send(200, st)
                except (VideoError, ValueError) as e:
                    self.close_connection = True  # the unread body must not be parsed as the next request
                    return self._send(200, {"ok": False, "reason": str(e)})
            if u.path == "/api/listen/audio":  # WAV body (16-bit, mono); fed to the listener as if from the mic
                if n > 10 * 1024 * 1024:
                    return self._send(413, {"ok": False, "reason": "audio larger than 10 MB"})
                try:
                    pcm = wav_to_pcm16k(self.rfile.read(n))
                except (ValueError, EOFError) as e:
                    return self._send(200, {"ok": False, "reason": f"need a 16-bit mono WAV ({e})"})
                srv.listener.inject(pcm)  # real time: returns when the audio has been "heard"
                return self._send(200, {"ok": True, "seconds": round(len(pcm) / 16000, 2)})
            if u.path == "/api/shop/background":
                if n > shops.MAX_IMAGE_BYTES:
                    return self._send(413, {"ok": False, "reason": "image larger than 30 MB"})
                data = self.rfile.read(n)
                try:
                    name = parse_qs(u.query).get("name", [""])[0]
                    shop = shops.upload_background(name, data, self.headers.get("Content-Type", ""))
                    if srv.shop and srv.shop["name"] == name:
                        srv.set_active(name)
                    return self._send(200, {"ok": True, "shop": shop})
                except shops.ShopError as e:
                    return self._send(200, {"ok": False, "reason": str(e)})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "bad json"})
            out = srv.post(self.path, body)
            self._send(200 if out is not None else 404, out if out is not None else {"error": "not found"})

    return H


def main():
    global AUDIO_SETTINGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", default="lo", help="network interface: lo = simulation")
    p.add_argument("--domain", type=int, default=None, help="DDS domain (default: 5 on lo, 0 on the robot)")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--hands", default=HANDS, choices=["inspire", "dex3", "none"],
                   help="fitted hands (default: EXP05_HANDS, else dex3 = the robot now); must match EXP05_HANDS, "
                        "which picks the model the wai and gestures are solved on")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets a remote machine open the monitor view")
    p.add_argument("--audio-settings", default="", help="speaking speed + volume file (default cache/audio_settings.json; "
                                                     "tests use their own)")
    p.add_argument("--ai-url", default="http://127.0.0.1:8105", help="AI service (TTS, LLM, speech recognition)")
    p.add_argument("--listen-source", default="laptop", choices=["laptop", "robot", "none"],
                   help="microphone for speech recognition: laptop (default input via PulseAudio), robot (G1 mic "
                        "multicast, untested), none (only audio posted to /api/listen/audio, tests)")
    p.add_argument("--listen-device", default="", help="PulseAudio source name for --listen-source laptop "
                                                       "(pactl list sources short); default: the system's default input")
    args = p.parse_args()
    if args.hands != "none" and args.hands != HANDS:
        p.error(f"--hands {args.hands} but EXP05_HANDS={HANDS}: start with EXP05_HANDS={args.hands} (run_sim.sh / "
                f"run_robot.sh set it from HANDS=...), so the motion is solved on the {args.hands} model")
    if args.audio_settings:
        AUDIO_SETTINGS = Path(args.audio_settings)
    if args.domain is None:
        args.domain = 5 if args.net == "lo" else 0
    import signal

    def on_term(*_):  # SIGTERM (run scripts) stops like Ctrl+C: the arms are released before the process ends
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    srv = Server(args)
    http = ThreadingHTTPServer((args.host, args.port), make_handler(srv))
    print(f"[SERVER] exp05 on http://localhost:{args.port}  ({'simulation' if srv.sim else 'robot on ' + args.net}, "
          f"DDS domain {args.domain})", flush=True)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.arm.release("server stop")  # hand the arms back before the stream ends
        t0 = time.time()
        while srv.arm.status()["mode"] != "released" and time.time() - t0 < 3.0:
            time.sleep(0.05)
        srv.stop_audio()
        srv.listener.close()
        print(f"[SERVER] stopped; arms {srv.arm.status()['mode']} after {time.time() - t0:.1f} s", flush=True)
        # exit now: the interpreter teardown segfaulted (DDS threads; seen with the sim running, also without the
        # speech recognition), and nothing is left to clean up after the arms are released
        os._exit(0)


if __name__ == "__main__":
    main()
