"""Speech recognition for the robot server (guide 2.2): microphone -> voice activity detection -> ASR -> Q&A -> reply.

    mic (laptop: parec 16 kHz mono | robot: G1 mic multicast | none) ─┐
    injected audio (POST /api/listen/audio, tests) ───────────────────┴─▶ 32 ms frames
      -> muted while the robot speaks (plus MUTE_TAIL_S): the robot must not hear itself (half-duplex)
      -> Silero VAD v6 (ONNX, CPU): a segment = speech until END_SILENCE_S of silence (max MAX_SEGMENT_S)
      -> AI service POST /asr (Typhoon Whisper Turbo; Thai and English in one model)
      -> filters: too short, empty, low confidence, phantom phrases Whisper invents on noise, and in auto mode
         "not a question for the robot" (relevance)
      -> reply "auto": Q&A (qa.answer) then say it, pointing at the booth when the answer names one;
         reply "review": the text only goes to the operator page (question box), the operator presses Speak.

Modes: off (mic closed) | ptt (push-to-talk: listen only while the operator holds the button) | auto (always
listening). Auto with the presence gate on listens only while a customer is present (POST /api/listen/presence, for
the vision module 2.3; until it exists the gate is off) and for FOLLOWUP_S after each answer.

Privacy (PDPA): audio stays in memory and is dropped after recognition; only the text is kept (last 20 lines).
"""

import base64
import collections
import json
import queue
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

import qa
import scripts
from qa import lang_of

SR = 16000
FRAME = 512  # 32 ms, the Silero VAD frame at 16 kHz
VAD_MODEL = Path(__file__).resolve().parents[1] / "third_party/silero_vad/silero_vad.onnx"
START_P, END_P = 0.5, 0.35  # speech starts above START_P, silence is below END_P
PREROLL_S = 0.3  # audio kept from before the speech start (first syllable)
END_SILENCE_S = 0.7  # Thai sentence-final particles (ครับ/ค่ะ) come after short pauses: not shorter
MIN_SPEECH_S = 0.35
MAX_SEGMENT_S = 12.0
MUTE_TAIL_S = 0.4  # still muted after the robot stops speaking (laptop audio buffer, room echo)
FOLLOWUP_S = 8.0
MIN_CONFIDENCE = 0.45
ROBOT_MIC = ("239.168.123.161", 5555)  # G1 EDU mic stream (unitree_sdk2_python PR #133); untested here

# Whisper's phantom phrases (subtitle credits it learned from video): never a customer's question
PHANTOM = ["ขอบคุณที่รับชม", "ติดตามตอนต่อไป", "ซับไทย", "คำบรรยายไทย", "กดติดตาม", "กดไลค์", "subtitles by",
           "thank you for watching", "thanks for watching", "please subscribe", "amara.org"]
# question / request cues for the relevance filter (auto mode)
CUES_TH = ["ไหม", "มั้ย", "หรือเปล่า", "อะไร", "ที่ไหน", "ตรงไหน", "ยังไง", "อย่างไร", "เท่าไร", "เท่าไหร่", "กี่โมง", "กี่บาท",
           "เมื่อไร", "เมื่อไหร่", "ทำไม", "ขอถาม", "ขอทราบ", "ช่วย", "อยากได้", "อยากซื้อ", "หาซื้อ", "สวัสดี", "ไปทาง",
           "ราคา", "แนะนำ"]
CUES_EN = ["?", "what", "where", "how", "when", "why", "who", "which", "can i", "could", "do you", "is there",
           "are there", "please", "looking for", "i want", "i need", "hello", "hi ", "excuse me", "help"]


class VAD:
    """Silero VAD v6 through onnxruntime (no torch in this env). Feed 512-sample float32 frames at 16 kHz."""

    def __init__(self):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(VAD_MODEL), sess_options=o, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros(64, np.float32)  # the model sees the last 64 samples of the previous frame

    def __call__(self, frame):
        x = np.concatenate([self.ctx, frame]).astype(np.float32)[None]
        out, self.state = self.s.run(None, {"input": x, "state": self.state, "sr": np.array(SR, np.int64)})
        self.ctx = frame[-64:]
        return float(out[0, 0])


# ---------------------------------------------------------------- microphones (16 kHz mono int16 chunks)
class LaptopMic:
    """A laptop input through PulseAudio's parec (the sound server on this laptop). Without `device`: the default
    input, unless it is dead silent (exact zeros: e.g. the headset-jack mic with nothing plugged in, which is the
    default on this laptop), then the next input that is not (the built-in "Digital Microphone")."""

    def __init__(self, device=""):
        self.device, self.proc, self.name = device, None, device

    def available(self):
        return shutil.which("parec") is not None

    @staticmethod
    def _sources():
        try:
            out = subprocess.run(["pactl", "list", "sources", "short"], capture_output=True, text=True, timeout=3).stdout
            default = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True, timeout=3).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return []
        names = [ln.split("\t")[1] for ln in out.splitlines() if "\t" in ln and not ln.split("\t")[1].endswith(".monitor")]
        return ([default] if default in names else []) + [n for n in names if n != default]

    @staticmethod
    def _description(dev):
        """'sof-hda-dsp Digital Microphone' for a PulseAudio source name, or ''."""
        try:
            out = subprocess.run(["pactl", "list", "sources"], capture_output=True, text=True, timeout=3).stdout
        except (OSError, subprocess.TimeoutExpired):
            return ""
        m = re.search(r"Name: " + re.escape(dev) + r"\s*\n\s*Description: (.+)", out)
        return m.group(1).strip() if m else ""

    def _start(self, dev):
        cmd = ["parec", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1", "--latency-msec=30"]
        if dev:
            cmd.append(f"--device={dev}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def open(self):
        if self.device:
            self.name = self.device
            return self._start(self.device)
        for dev in self._sources() or [""]:
            self._start(dev)
            x = np.concatenate([self.read() for _ in range(16)])  # 0.5 s
            if np.any(x):
                self.name = self._description(dev) or dev or "default input"
                return
            self.close()
        raise OSError("every microphone input is silent (plug in a mic or check the input in the sound settings)")

    def read(self):
        b = self.proc.stdout.read(FRAME * 2)
        if not b:
            raise OSError("parec stopped (no input device?)")
        return np.frombuffer(b, "<i2")

    def close(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class RobotMic:
    """G1 EDU microphone: 16 kHz mono 16-bit PCM multicast by the robot (239.168.123.161:5555). Reported to stream
    only with the voice assistant in "Wake-up Conversation Mode" (Unitree app -> Device -> Data -> Voice Assistant).
    NOT TESTED on our robot yet (S7)."""

    def __init__(self, iface_ip):
        self.ip, self.sock, self.buf = iface_ip, None, b""

    def available(self):
        return bool(self.ip)

    def open(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", ROBOT_MIC[1]))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     struct.pack("=4s4s", socket.inet_aton(ROBOT_MIC[0]), socket.inet_aton(self.ip)))
        s.settimeout(3.0)
        self.sock, self.buf = s, b""

    def read(self):
        while len(self.buf) < FRAME * 2:
            try:
                d, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                raise OSError("no audio from the robot microphone (voice assistant in wake-up mode?)") from None
            self.buf += d[: len(d) // 2 * 2]
        b, self.buf = self.buf[: FRAME * 2], self.buf[FRAME * 2:]
        return np.frombuffer(b, "<i2")

    def close(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = None


def iface_ip(name):
    """IPv4 address of a network interface (for joining the robot's multicast group), or ''."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", name], capture_output=True, text=True).stdout
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out)
        return m.group(1) if m else ""
    except OSError:
        return ""


# ---------------------------------------------------------------- the listener
class Listener:
    """srv: the robot Server (speaking(), shop, speech, settings, post()). source: laptop | robot | none."""

    def __init__(self, srv, source="laptop", device="", net=""):
        self.srv = srv
        self.source = source
        self.mic = {"laptop": lambda: LaptopMic(device), "robot": lambda: RobotMic(iface_ip(net)),
                    "none": lambda: None}[source]()
        self.settings = {"mode": "off", "reply": "auto", "relevance": True, "presence_gate": False}
        self.state, self.reason = "off", ""
        self.level_db = -90.0
        self.ptt = False
        self.present_until = 0.0  # customer present (vision 2.3) until this time
        self.followup_until = 0.0
        self.await_followup = False  # an answer was started: open the follow-up window when it has been spoken
        self.lines = collections.deque(maxlen=20)
        self.seq = 0
        self.frames = queue.Queue(maxsize=400)  # ~13 s of audio
        self.segments = queue.Queue(maxsize=4)
        self.vad = None
        self.vad_error = ""
        try:
            self.vad = VAD()
        except Exception as e:  # noqa: BLE001
            self.vad_error = f"VAD not available: {e}"
        self.lock = threading.Lock()
        self.mic_open = False
        self.closing = False
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._capture, self._process,
                                                                           self._recognise_loop)]
        for t in self.threads:
            t.start()

    def close(self):
        """Server shutdown: stop the threads before the interpreter tears down (onnxruntime in a daemon thread at exit
        segfaulted)."""
        self.closing = True
        self.settings["mode"] = "off"
        self.frames.put((None, "stop"))
        self.segments.put((None, 0.0, "stop"))
        for t in self.threads:
            t.join(2.0)
        self.vad = None

    # ---- API
    def configure(self, body):
        for k in ("mode", "reply"):
            if k in body:
                ok = {"mode": ("off", "ptt", "auto"), "reply": ("auto", "review")}[k]
                if body[k] not in ok:
                    return {"ok": False, "reason": f"{k} must be one of {', '.join(ok)}"}
                self.settings[k] = body[k]
        for k in ("relevance", "presence_gate"):
            if k in body:
                self.settings[k] = bool(body[k])
        if "mode" in body:
            self.ptt = False
        return {"ok": True, "listen": self.status()}

    def push_to_talk(self, down):
        if self.settings["mode"] != "ptt":
            return {"ok": False, "reason": "push-to-talk needs mode ptt"}
        self.ptt = bool(down)
        return {"ok": True}

    def presence(self, present, hold_s=2.0):
        """Vision (2.3) reports a customer in front; it must repeat this while the customer stays."""
        self.present_until = time.monotonic() + float(np.clip(hold_s, 0.2, 30.0)) if present else 0.0
        return {"ok": True}

    def inject(self, pcm):
        """Audio from a file or a test (int16, 16 kHz mono), as if it came from the microphone (real time)."""
        pcm = np.asarray(pcm, "<i2")
        for i in range(0, len(pcm) - FRAME + 1, FRAME):
            self.frames.put((pcm[i: i + FRAME], "inject"), timeout=5)
            time.sleep(FRAME / SR * 0.98)

    def status(self):
        state = self.state
        if self.settings["mode"] != "off" and state not in ("recognising", "answering") and self.srv.speaking():
            state = "muted"  # (the frame loop also mutes; this shows it even when no audio is arriving)
        return {**self.settings, "source": self.source, "state": state, "reason": self.reason,
                "level_db": round(self.level_db, 1), "ptt": self.ptt, "seq": self.seq,
                "present": time.monotonic() < self.present_until,
                "followup_s": round(max(0.0, self.followup_until - time.monotonic()), 1),
                "mic": self.mic_open, "mic_name": getattr(self.mic, "name", "") if self.mic_open else "",
                "error": self.vad_error, "lines": list(self.lines)}

    # ---- capture: the mic is open only while a mode is on
    def _capture(self):
        while not self.closing:
            want = self.settings["mode"] != "off" and self.mic is not None and self.vad is not None
            if not want:
                time.sleep(0.1)
                continue
            if not self.mic.available():
                self.vad_error = f"microphone '{self.source}' not available"
                time.sleep(1.0)
                continue
            try:
                self.mic.open()
                self.mic_open, self.vad_error = True, ""
                while self.settings["mode"] != "off":
                    f = self.mic.read()
                    try:
                        self.frames.put_nowait((f, "mic"))
                    except queue.Full:
                        pass
            except OSError as e:
                self.vad_error = str(e)
                time.sleep(1.0)
            finally:
                self.mic.close()
                self.mic_open = False

    def _gate(self, now):
        """-> (listening?, state when not)."""
        mode = self.settings["mode"]
        if mode == "off":
            return False, "off"
        if self.srv.speaking():
            self.mute_until = now + MUTE_TAIL_S
            return False, "muted"
        if now < getattr(self, "mute_until", 0.0):
            return False, "muted"
        if self.await_followup:  # the answer has been spoken: follow-up questions need no trigger
            self.await_followup = False
            self.followup_until = now + FOLLOWUP_S
        if self.state in ("recognising", "answering"):
            return False, self.state
        if mode == "ptt":
            return self.ptt, "waiting for the button"
        if self.settings["presence_gate"] and now >= self.present_until and now >= self.followup_until:
            return False, "waiting for a customer"
        return True, ""

    def _process(self):
        pre = collections.deque(maxlen=int(PREROLL_S * SR / FRAME))
        seg, silence, speech_frames, in_speech, was_ptt = [], 0, 0, False, False
        while True:
            f, _ = self.frames.get()
            if f is None:
                return
            x = f.astype(np.float32) / 32768
            rms = float(np.sqrt(np.mean(x * x))) + 1e-9
            self.level_db = max(20 * np.log10(rms), self.level_db - 3.0)  # meter: fast up, slow down
            if self.vad is None:
                continue
            now = time.monotonic()
            listening, why = self._gate(now)
            ptt_mode = self.settings["mode"] == "ptt"
            if not listening:
                if ptt_mode and was_ptt and seg:  # button released: recognise what was said
                    self._finish(seg, speech_frames, "ptt")
                elif in_speech and not ptt_mode and why != "muted":
                    pass  # (recognising/answering: the segment was already handed over)
                seg, silence, speech_frames, in_speech, was_ptt = [], 0, 0, False, False
                pre.clear()
                self.vad.reset()
                if self.state not in ("recognising", "answering"):
                    self.state = why
                continue
            p = self.vad(x)
            if ptt_mode:  # everything while the button is held is the segment; VAD only counts the speech
                was_ptt = True
                seg.append(f)
                speech_frames += p >= START_P
                self.state = "hearing" if p >= START_P else "listening (button held)"
                if len(seg) * FRAME / SR >= MAX_SEGMENT_S:
                    self._finish(seg, speech_frames, "ptt")
                    seg, speech_frames = [], 0
                continue
            if not in_speech:
                pre.append(f)
                self.state = "listening"
                if p >= START_P:
                    in_speech, seg, silence, speech_frames = True, list(pre), 0, 1
                    self.state = "hearing"
                continue
            seg.append(f)
            if p >= END_P:
                silence, speech_frames = 0, speech_frames + 1
            else:
                silence += 1
            if silence * FRAME / SR >= END_SILENCE_S or len(seg) * FRAME / SR >= MAX_SEGMENT_S:
                self._finish(seg[: len(seg) - max(0, silence - int(0.2 * SR / FRAME))], speech_frames, "vad")
                seg, silence, speech_frames, in_speech = [], 0, 0, False
                pre.clear()
                self.vad.reset()

    def _finish(self, seg, speech_frames, how):
        speech_s = speech_frames * FRAME / SR
        if speech_s < MIN_SPEECH_S:
            if how == "ptt":
                self._line({"text": "", "ignored": "no speech heard", "speech_s": round(speech_s, 2)})
            return
        try:
            self.segments.put_nowait((np.concatenate(seg), speech_s, how))
            self.state = "recognising"
        except queue.Full:
            pass

    # ---- recognition and reply
    def _line(self, d):
        with self.lock:
            self.seq += 1
            self.lines.appendleft({"id": self.seq, "time": time.strftime("%H:%M:%S"), **d})
        return self.seq

    def asr(self, pcm):
        req = urllib.request.Request(self.srv.speech.ai + "/asr",
                                     json.dumps({"pcm_b64": base64.b64encode(pcm.astype("<i2").tobytes()).decode(),
                                                 "lang": "th"}).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except OSError as e:
            return {"ok": False, "reason": f"AI service not reachable ({e})"}

    def _recognise_loop(self):
        while True:
            pcm, speech_s, how = self.segments.get()
            if pcm is None:
                return
            self.state = "recognising"
            try:
                self._recognise(pcm, speech_s, how)
            except Exception as e:  # noqa: BLE001
                self._line({"text": "", "ignored": f"error: {type(e).__name__}: {e}"})
            finally:
                self.state = "listening" if self.settings["mode"] == "auto" else "waiting for the button"
                if self.settings["mode"] == "off":
                    self.state = "off"

    def _recognise(self, pcm, speech_s, how):
        r = self.asr(pcm)
        base = {"speech_s": round(speech_s, 2), "audio_s": round(len(pcm) / SR, 2), "how": how}
        if not r.get("ok"):
            self._line({**base, "text": "", "ignored": r.get("reason", "ASR failed")})
            return
        text, conf = r["text"].strip(), r["confidence"]
        line = {**base, "text": text, "lang": lang_of(text), "confidence": conf, "asr_s": r["seconds"]}
        why = self.reject(text, conf, how)
        if why:
            self._line({**line, "ignored": why})
            return
        if self.settings["reply"] == "review":
            self._line({**line, "action": "review"})
            return
        self.state = "answering"
        line_id = self._line({**line, "action": "answering"})
        if not self.srv.shop:
            self._update(line_id, action="", ignored="no active shop")
            return
        t = time.time()
        a = qa.answer(self.srv.shop, text, self.srv.speech)
        body = {"text": a["answer"], "lang": a["lang"]}
        if a.get("booth"):
            if a.get("source") == "fallback":  # no real answer, but the question names a booth: "X is this way"
                body["text"] = None
                b = next((b for b in self.srv.shop["booths"] if b["id"] == a["booth"]), {})
                a["answer"] = scripts.fill(self.srv.shop, "point", a["lang"], name=b.get(f"name_{a['lang']}", a["booth"]))
            res = self.srv.post("/api/point", {**body, "booth": a["booth"], "hold_s": 2.0})
            act = f"point {a['booth']}"
        else:
            res = self.srv.post("/api/say", body)
            act = "say"
        self.await_followup = bool(res and res.get("ok"))
        self._update(line_id, action=act if res and res.get("ok") else f"{act} failed: {(res or {}).get('reason', '')}",
                     answer=a["answer"], source=a.get("source", ""), answer_s=round(time.time() - t, 2))

    def _update(self, line_id, **kw):
        with self.lock:
            for ln in self.lines:
                if ln["id"] == line_id:
                    ln.update(kw)

    def reject(self, text, conf, how):
        """Why a recognised text is ignored, or ''."""
        t = text.lower().strip()
        core = re.sub(r"[\s.,!?…]+", "", t)
        if len(core) < 2:
            return "nothing recognised"
        if any(p in t for p in PHANTOM):
            return "phantom phrase (noise)"
        if conf < MIN_CONFIDENCE:
            return f"unclear (confidence {conf:.2f})"
        if how == "vad" and self.settings["relevance"] and not self.relevant(t):
            return "not a question for the robot"
        return ""

    def relevant(self, t):
        shop = self.srv.shop
        if shop:
            if qa.find_booth(shop, t):
                return True
            if qa.match_faq(shop, t, lang_of(t))[1] >= 0.5:
                return True
            s = scripts.get(shop)
            if any(len(s[k]) > 1 and s[k].lower() in t for k in ("robot_th", "robot_en", "shop_th", "shop_en")):
                return True
        cues = CUES_TH if lang_of(t) == "th" else CUES_EN
        return any(c in t + " " for c in cues)
