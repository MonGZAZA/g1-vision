"""Speech for the robot server: TTS through the exp05 AI service (with a file cache) and audio output.

Utterance = one text in one language and voice: a 24 kHz WAV (laptop), a 16 kHz WAV (robot speaker), its duration
and phrase timing [{text, start, end}] (s from the audio start). Captions are the phrases.

Cache: exp05/cache/tts/<sha1>.json + .wav + .16k.wav, keyed by (engine version, voice, lang, text, speaking speed). Fixed texts
(booth sentences, introductions) are pre-rendered in the background when a shop becomes active, so the buttons
start without waiting.

Audio outputs (one active, chosen in the UI):
  laptop  paplay (else pw-play / aplay) of the 24 kHz file (testing, sim); volume = paplay stream volume
  robot   G1 AudioClient.PlayStream: 16 kHz mono PCM in 1 s chunks, sent ahead of time (deployment);
          volume = the robot speaker's own level (AudioClient.SetVolume)
  off     silent: captions and timing run as if audio played (tests, night work)
"""

import base64
import hashlib
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parents[2] / "cache/tts"
ENGINE_VERSION = "v1"  # bump to invalidate the cache when the TTS setup changes
AI_URL = "http://127.0.0.1:8105"


class SpeechError(RuntimeError):
    pass


class Utterance:
    def __init__(self, meta, wav, wav16):
        self.text, self.lang, self.voice = meta["text"], meta["lang"], meta["voice"]
        self.duration, self.phrases = meta["duration"], meta["phrases"]
        self.wav, self.wav16 = wav, wav16

    def caption_at(self, t):
        """Phrase being spoken at time t (s from the audio start), or None between phrases / outside."""
        for i, p in enumerate(self.phrases):
            if p["start"] <= t < p["end"] + 0.12:
                return i, p["text"]
        return None

    def info(self):
        return {"text": self.text, "lang": self.lang, "duration": self.duration, "phrases": self.phrases}


class Speech:
    def __init__(self, ai_url=AI_URL):
        self.ai = ai_url
        # speaking speed 0.8-1.2, applied by the TTS engines while generating (phrase times are measured on that
        # audio, so motion and captions follow); part of the cache key, so another speed is never reused
        self.rate = 1.0
        CACHE.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()  # one TTS request at a time (the service serialises anyway)

    def health(self):
        try:
            with urllib.request.urlopen(self.ai + "/health", timeout=1.5) as r:
                return json.loads(r.read())
        except OSError:
            return None

    def key(self, text, lang, voice, rate=1.0):
        r = "" if abs(rate - 1.0) < 1e-6 else f"|rate{rate:.2f}"  # 1.0: the key of the cache before speeds existed
        return hashlib.sha1(f"{ENGINE_VERSION}|{voice}|{lang}|{text.strip()}{r}".encode()).hexdigest()

    def cached(self, text, lang, voice="default", rate=None):
        k = self.key(text, lang, voice, self.rate if rate is None else rate)
        meta = CACHE / f"{k}.json"
        if meta.exists():
            return Utterance(json.loads(meta.read_text(encoding="utf-8")), CACHE / f"{k}.wav", CACHE / f"{k}.16k.wav")
        return None

    def render(self, text, lang, voice="default", timeout=60):
        text = text.strip()
        if not text:
            raise SpeechError("empty text")
        rate = self.rate
        u = self.cached(text, lang, voice, rate)
        if u is not None:
            return u
        with self.lock:
            u = self.cached(text, lang, voice, rate)
            if u is not None:
                return u
            req = urllib.request.Request(self.ai + "/tts", json.dumps({"text": text, "lang": lang, "voice": voice,
                                                                       "speed": rate}).encode(),
                                         {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read())
            except OSError as e:
                raise SpeechError(f"AI service not reachable ({e}); start it with exp05/ai/run_ai.sh") from None
            if not d.get("ok"):
                raise SpeechError(d.get("reason", "TTS failed"))
            k = self.key(text, lang, voice, rate)
            (CACHE / f"{k}.wav").write_bytes(base64.b64decode(d["wav_b64"]))
            (CACHE / f"{k}.16k.wav").write_bytes(base64.b64decode(d["wav16_b64"]))
            meta = {"text": text, "lang": lang, "voice": voice, "rate": rate, "duration": d["duration"], "phrases": d["phrases"],
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            (CACHE / f"{k}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            return self.cached(text, lang, voice, rate)

    def ask(self, question, context, lang, timeout=60):
        req = urllib.request.Request(self.ai + "/llm", json.dumps({"question": question, "context": context, "lang": lang}).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except OSError as e:
            return {"ok": False, "reason": f"AI service not reachable ({e})"}

    def proxy(self, path, body, ctype, timeout=120):
        """Forward a raw request to the AI service (voice upload)."""
        req = urllib.request.Request(self.ai + path, body, {"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except OSError as e:
            return {"ok": False, "reason": f"AI service not reachable ({e})"}

    def get(self, path, timeout=3):
        try:
            with urllib.request.urlopen(self.ai + path, timeout=timeout) as r:
                return json.loads(r.read())
        except OSError as e:
            return {"ok": False, "reason": f"AI service not reachable ({e})"}


# ---------------------------------------------------------------- audio outputs
class LaptopAudio:
    """Plays a WAV on this computer. LATENCY: measured time from play() to sound (subtracted by the scheduler)."""
    LATENCY = 0.08

    def __init__(self):
        # paplay first: it works on PulseAudio and on PipeWire (pipewire-pulse); pw-play fails silently
        # ("no node available") when PipeWire is installed but PulseAudio is the sound server (this laptop)
        self.cmd = next(([c] for c in ("paplay", "pw-play", "aplay") if shutil.which(c)), None)
        self.proc = None
        self.volume = 100  # % of the system volume (paplay / pw-play stream volume; aplay: not adjustable)

    def set_volume(self, v):
        self.volume = int(v)
        return True

    def get_volume(self):
        return self.volume

    def available(self):
        return self.cmd is not None

    def play(self, u):
        self.stop()
        if self.cmd:
            vol = {"paplay": [f"--volume={round(65536 * self.volume / 100)}"],
                   "pw-play": [f"--volume={self.volume / 100:.2f}"]}.get(self.cmd[0], [])
            self.proc = subprocess.Popen(self.cmd + vol + [str(u.wav)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


class RobotAudio:
    """G1 speaker through AudioClient.PlayStream (16 kHz mono 16-bit PCM). Needs the DDS link to the real robot."""
    LATENCY = 0.25  # first estimate; measured on the robot in S7
    APP = "exp05"
    CHUNK_S = 1.0

    def __init__(self):
        self.client = None
        self.stop_ev = threading.Event()
        self.thread = None

    def _client(self):
        if self.client is None:
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            c = AudioClient()
            c.SetTimeout(3.0)
            c.Init()
            self.client = c
        return self.client

    def available(self):
        return True

    def set_volume(self, v):
        """The robot speaker's own volume (AudioClient.SetVolume, 0-100). -> True if the robot accepted it."""
        try:
            return self._client().SetVolume(int(v)) == 0
        except Exception as e:  # noqa: BLE001
            print(f"[AUDIO] robot volume: {e}", flush=True)
            return False

    def get_volume(self):
        """The robot speaker's volume (0-100), or None if it cannot be read."""
        try:
            code, d = self._client().GetVolume()
            return int(d.get("volume")) if code == 0 and isinstance(d, dict) and "volume" in d else None
        except Exception:  # noqa: BLE001
            return None

    def play(self, u):
        self.stop()
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self._stream, args=(u,), daemon=True)
        self.thread.start()

    def _stream(self, u):
        import wave
        try:
            with wave.open(str(u.wav16)) as w:
                pcm = w.readframes(w.getnframes())
            c = self._client()
            sid = str(int(time.time() * 1000))
            step = int(16000 * 2 * self.CHUNK_S)
            t0 = time.monotonic()
            for i, off in enumerate(range(0, len(pcm), step)):
                if self.stop_ev.is_set():
                    break
                c.PlayStream(self.APP, sid, pcm[off: off + step])
                # keep ~1 chunk queued on the robot: send chunk i+1 when chunk i starts playing
                wait = t0 + i * self.CHUNK_S - time.monotonic()
                if wait > 0 and self.stop_ev.wait(wait):
                    break
        except Exception as e:  # noqa: BLE001
            print(f"[AUDIO] robot speaker: {e}", flush=True)

    def stop(self):
        self.stop_ev.set()
        if self.client is not None:
            try:
                self.client.PlayStop(self.APP)
            except Exception:  # noqa: BLE001
                pass


class SilentAudio:
    LATENCY = 0.0

    def set_volume(self, v):
        return True

    def get_volume(self):
        return 0

    def available(self):
        return True

    def play(self, u):
        pass

    def stop(self):
        pass


def make_outputs(real_robot):
    out = {"laptop": LaptopAudio(), "off": SilentAudio()}
    if real_robot:
        out["robot"] = RobotAudio()
    return out
