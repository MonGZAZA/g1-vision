"""exp05 AI service (127.0.0.1:8105): Thai/English female TTS with phrase timing, and the local LLM for Q&A.

Runs in its own uv env (torch); the robot server (g1 env) calls it over HTTP. Models live in exp05/ai/models.

    cd exp05/ai && HF_HOME=models/hf .venv/bin/python service.py [--port 8105] [--no-llm]

API
    GET  /health                       models loaded, GPU memory
    POST /tts   {"text", "lang": "th"|"en", "voice": "default"|<uploaded voice name>, "speed": 0.8-1.2 (default 1)}
                -> {"ok", "sr", "wav_b64" (24 kHz 16-bit mono WAV), "wav16_b64" (16 kHz copy), "duration",
                    "phrases": [{"text", "start", "end"}]}
    POST /llm   {"question", "context", "lang"} -> {"ok", "answer", "seconds"}
    POST /asr   {"pcm_b64": 16 kHz mono 16-bit little-endian PCM, "lang": "th" (default) | "en"}
                -> {"ok", "text", "confidence" (0-1, mean token probability), "seconds", "audio_s"}

Speech recognition: typhoon-ai/typhoon-whisper-turbo (Whisper large-v3-turbo fine-tuned for Thai, MIT, ~1.7 GB GPU).
With language "th" it also writes English words and English sentences in English (tested 2026-11-13), so one
model serves both; "en" only adds English punctuation. Whisper invents text on silence or noise (here
"สวัสดีครับ"): callers must send only speech found by a VAD (the robot server's deploy/listen.py does).

Voices: "default" = VIZINTZOR/MMS-TTS-THAI-FEMALEV2 (Thai, VITS) + Kokoro-82M af_heart (English). An uploaded
reference voice (voices/<name>/ref.wav + ref.txt, POST /voice/upload) speaks Thai through F5-TTS-THAI
(clone_engine.py); English stays on the default voice (the clone model is Thai-only).
    GET  /voices                       default + uploaded voices
    POST /voice/upload?name=N&text=T   body = audio (WAV/MP3/FLAC/OGG), the reference recording and its transcript
"""

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / "models/hf"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

SR = 24000
TH_MODEL = "VIZINTZOR/MMS-TTS-THAI-FEMALEV2"
EN_VOICE = "af_heart"
LLM_MODEL = "scb10x/typhoon2.5-qwen3-4b"
ASR_MODEL = "typhoon-ai/typhoon-whisper-turbo"
PAUSE = {"sentence": 0.32, "phrase": 0.16}  # s of silence after a phrase
VOICES = HERE / "voices"


def split_phrases(text, lang):
    """Phrases with the pause that follows each. Thai: spaces and punctuation separate phrases (Thai writes no spaces
    between words, a space marks a pause); English: sentence punctuation and commas."""
    text = re.sub(r"\s+", " ", text.strip())
    out = []
    if lang == "th":
        for sent in re.split(r"(?<=[.!?。])\s+|\n", text):
            parts = [p for p in re.split(r"[ ,]+", sent) if p.strip()]
            # very short pieces (one or two syllables) join the next piece: VITS sounds odd on tiny inputs
            merged = []
            for p in parts:
                if merged and len(merged[-1]) < 6:
                    merged[-1] += " " + p
                else:
                    merged.append(p)
            for i, p in enumerate(merged):
                out.append((p, PAUSE["sentence"] if i == len(merged) - 1 else PAUSE["phrase"]))
    else:
        for sent in re.split(r"(?<=[.!?])\s+", text):
            parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", sent) if p.strip()]
            for i, p in enumerate(parts):
                out.append((p, PAUSE["sentence"] if i == len(parts) - 1 else PAUSE["phrase"]))
    return out


class TTS:
    def __init__(self, device):
        from kokoro import KPipeline
        from transformers import AutoTokenizer, VitsModel
        self.device = device
        self.th = VitsModel.from_pretrained(TH_MODEL).to(device).eval()
        self.th_tok = AutoTokenizer.from_pretrained(TH_MODEL)
        self.th_sr = self.th.config.sampling_rate
        self.en = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
        self.lock = threading.Lock()
        self.clone = None
        try:
            import clone_engine  # optional F5-TTS-THAI voice cloning (own env, see clone_engine.py)
            self.clone = clone_engine.CloneEngine()
        except Exception as e:  # noqa: BLE001
            print(f"[AI] voice cloning not available ({e}); uploaded voices fall back to default", flush=True)

    def _th(self, text, speed=1.0):
        torch.manual_seed(0)  # VITS samples durations/noise: fixed seed = same audio for the same text
        inp = self.th_tok(text, return_tensors="pt").to(self.device)
        self.th.speaking_rate = speed  # VITS length scale = 1 / speaking_rate (phrase timing is measured after)
        with torch.no_grad():
            w = self.th(**inp).waveform[0].float().cpu().numpy()
        return resample_poly(w, SR, self.th_sr) if self.th_sr != SR else w

    def _en(self, text, speed=1.0):
        return np.concatenate([r.audio.cpu().numpy() for r in self.en(text, voice=EN_VOICE, speed=speed)])

    def say(self, text, lang, voice="default", speed=1.0):
        """Phrase by phrase at the given speaking speed (0.8-1.2; the pauses scale too). Each phrase's start/end is
        measured on the generated audio, so motion and captions scheduled on them follow any speed."""
        speed = float(np.clip(speed, 0.8, 1.2))
        phrases, chunks, t = [], [], 0.0
        ref = VOICES / voice if voice != "default" else None
        with self.lock:
            for p, pause in split_phrases(text, lang):
                w = None
                if ref is not None and self.clone is not None and (ref / "ref.wav").exists():
                    w = self.clone.say(p, lang, ref, speed)  # None for English (Thai-only model)
                if w is None:
                    w = self._th(p, speed) if lang == "th" else self._en(p, speed)
                w = _trim(w)
                phrases.append({"text": p, "start": round(t, 3), "end": round(t + len(w) / SR, 3)})
                pause /= speed
                chunks += [w, np.zeros(int(pause * SR), np.float32)]
                t += len(w) / SR + pause
        audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = (audio / peak * 0.89).astype(np.float32)
        return audio, phrases


def _trim(w, thr=0.01):
    """Cut leading/trailing silence (keeps 30 ms) so phrase timing is the speech itself."""
    idx = np.where(np.abs(w) > thr * (np.max(np.abs(w)) or 1))[0]
    if len(idx) == 0:
        return w.astype(np.float32)
    pad = int(0.03 * SR)
    return w[max(0, idx[0] - pad): idx[-1] + pad].astype(np.float32)


class LLM:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.m = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_quant_type="nf4"))
        self.lock = threading.Lock()

    def ask(self, question, context, lang):
        if lang == "th":
            sys_msg = ("คุณคือหุ่นยนต์ต้อนรับผู้หญิงที่หน้าร้าน ตอบเป็นภาษาไทยเท่านั้น สั้นๆ 1-2 ประโยค สุภาพ "
                       "ลงท้ายด้วย ค่ะ หรือ นะคะ ใช้เฉพาะข้อมูลร้านด้านล่าง ถ้าไม่มีข้อมูล ให้บอกว่าไม่ทราบและแนะนำให้ติดต่อ"
                       "จุดบริการข้อมูล ห้ามแต่งข้อมูลเอง\n\nข้อมูลร้าน:\n" + context)
        else:
            sys_msg = ("You are the female greeter robot at the shop entrance. Answer in English only, in 1-2 short polite "
                       "sentences. Use only the shop information below. If it is not there, say you do not know and "
                       "suggest the information desk. Never invent facts.\n\nShop information:\n" + context)
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": question}]
        with self.lock:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                               enable_thinking=False).to("cuda")
            out = self.m.generate(ids, max_new_tokens=160, do_sample=False, repetition_penalty=1.05)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


class ASR:
    def __init__(self, device):
        import transformers
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        transformers.logging.set_verbosity_error()  # "output_scores may be ignored": they are not (confidence)
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = AutoProcessor.from_pretrained(ASR_MODEL)
        self.m = AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL, torch_dtype=self.dtype).to(device).eval()
        self.lock = threading.Lock()
        self.transcribe(np.zeros(16000, np.float32))  # warm-up: the first call is ~1 s slower (CUDA kernels)

    def transcribe(self, pcm, lang="th"):
        """pcm: float32 16 kHz mono (at most 30 s). -> (text, confidence = mean token probability)."""
        f = self.proc(pcm, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, self.dtype)
        with self.lock, torch.no_grad():
            o = self.m.generate(f, language=lang, task="transcribe", max_new_tokens=128,
                                return_dict_in_generate=True, output_scores=True)
            sc = self.m.compute_transition_scores(o.sequences, o.scores, normalize_logits=True)
        text = self.proc.batch_decode(o.sequences, skip_special_tokens=True)[0].strip()
        conf = float(torch.exp(sc[0].float().mean())) if sc.numel() else 0.0
        return text, conf


def upload_voice(path, data):
    """Reference voice for cloning: any audio soundfile reads (WAV, FLAC, OGG, MP3) -> voices/<name>/ref.wav
    (24 kHz mono, at most 12 s, 0.5 s silence at the end, as F5-TTS wants) + ref.txt (its transcript)."""
    from urllib.parse import parse_qs, urlparse
    q = {k: v[0] for k, v in parse_qs(urlparse(path).query).items()}
    name, text = q.get("name", ""), q.get("text", "").strip()
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", name) or name == "default":
        return {"ok": False, "reason": "voice name: 1-40 letters, digits, - or _ (not 'default')"}
    if not text:
        return {"ok": False, "reason": "the transcript of the recording is needed"}
    try:
        w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"cannot read the audio ({e}); use WAV or MP3"}
    w = w.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr)
        w = resample_poly(w, SR // g, sr // g)
    w = _trim(w)
    if len(w) < 2 * SR:
        return {"ok": False, "reason": "recording too short (needs 5-12 s of speech)"}
    w = w[: 12 * SR]
    w = np.concatenate([w / (np.max(np.abs(w)) or 1) * 0.9, np.zeros(SR // 2, np.float32)])
    d = VOICES / name
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "ref.wav", w, SR, subtype="PCM_16")
    (d / "ref.txt").write_text(text, encoding="utf-8")
    secs = round(len(w) / SR, 1)
    (d / "meta.json").write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"), "seconds": secs,
                                             "consent": True}), encoding="utf-8")
    return {"ok": True, "name": name, "seconds": secs}


def wav_bytes(audio, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8105)
    p.add_argument("--no-llm", action="store_true", help="TTS only (no GPU memory for the LLM)")
    p.add_argument("--cpu", action="store_true", help="TTS on the CPU")
    p.add_argument("--no-asr", action="store_true", help="no speech recognition (saves ~1.7 GB GPU)")
    a = p.parse_args()
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    t0 = time.time()
    tts = TTS(dev)
    state = {"llm": None, "llm_error": "", "asr": None, "asr_error": ""}
    print(f"[AI] TTS ready on {dev} ({time.time() - t0:.0f} s)", flush=True)

    def load_llm():
        try:
            t = time.time()
            state["llm"] = LLM()
            print(f"[AI] LLM {LLM_MODEL} ready ({time.time() - t:.0f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            state["llm_error"] = str(e)
            print(f"[AI] LLM failed: {e}", flush=True)

    def load_asr():
        try:
            t = time.time()
            state["asr"] = ASR(dev)
            print(f"[AI] ASR {ASR_MODEL} ready ({time.time() - t:.0f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            state["asr_error"] = str(e)
            print(f"[AI] ASR failed: {e}", flush=True)

    def load_models():  # one after the other: loading both at once doubles the peak CPU RAM
        if not a.no_asr:
            load_asr()
        if not a.no_llm:
            load_llm()

    threading.Thread(target=load_models, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/voices":
                vs = [{"name": "default", "note": "built-in (Thai MMS female, English Kokoro af_heart)"}]
                for d in sorted(VOICES.glob("*/meta.json")):
                    m = json.loads(d.read_text())
                    vs.append({"name": d.parent.name, "note": f"uploaded {m.get('created', '')}, {m.get('seconds', 0)} s"})
                return self._send({"ok": True, "voices": vs, "clone": tts.clone is not None})
            if self.path == "/health":
                gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                return self._send({"ok": True, "tts": True, "device": dev, "llm": state["llm"] is not None,
                                   "llm_loading": not a.no_llm and state["llm"] is None and not state["llm_error"],
                                   "llm_error": state["llm_error"], "clone": tts.clone is not None,
                                   "asr": state["asr"] is not None, "asr_error": state["asr_error"],
                                   "asr_loading": not a.no_asr and state["asr"] is None and not state["asr_error"],
                                   "gpu_gb": round(gpu, 2), "models": {"th": TH_MODEL, "en": f"Kokoro-82M {EN_VOICE}",
                                                                      "llm": LLM_MODEL, "asr": ASR_MODEL}})
            self._send({"ok": False, "reason": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if self.path.startswith("/voice/upload"):
                return self._send(upload_voice(self.path, self.rfile.read(n)))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send({"ok": False, "reason": "bad json"}, 400)
            try:
                if self.path == "/tts":
                    text, lang = str(body.get("text", ""))[:2000], body.get("lang", "th")
                    if lang not in ("th", "en") or not text.strip():
                        return self._send({"ok": False, "reason": "text and lang th|en needed"})
                    t = time.time()
                    try:
                        speed = float(body.get("speed", 1.0))
                    except (TypeError, ValueError):
                        return self._send({"ok": False, "reason": "speed must be a number (0.8-1.2)"})
                    audio, phrases = tts.say(text, lang, str(body.get("voice", "default")), speed)
                    return self._send({"ok": True, "sr": SR, "duration": round(len(audio) / SR, 3), "phrases": phrases,
                                       "seconds": round(time.time() - t, 2),
                                       "wav_b64": base64.b64encode(wav_bytes(audio)).decode(),
                                       # 16 kHz copy for the robot speaker (G1 AudioClient.PlayStream wants 16 kHz mono)
                                       "wav16_b64": base64.b64encode(wav_bytes(resample_poly(audio, 2, 3), 16000)).decode()})
                if self.path == "/asr":
                    if state["asr"] is None:
                        return self._send({"ok": False, "reason": state["asr_error"] or (
                            "speech recognition is off (--no-asr)" if a.no_asr else "speech recognition still loading")})
                    lang = body.get("lang", "th")
                    if lang not in ("th", "en"):
                        return self._send({"ok": False, "reason": "lang must be th or en"})
                    pcm = np.frombuffer(base64.b64decode(body.get("pcm_b64", "")), dtype="<i2").astype(np.float32) / 32768
                    if len(pcm) < 1600 or len(pcm) > 30 * 16000:
                        return self._send({"ok": False, "reason": "audio must be 0.1-30 s of 16 kHz mono PCM"})
                    t = time.time()
                    text, conf = state["asr"].transcribe(pcm, lang)
                    return self._send({"ok": True, "text": text, "confidence": round(conf, 3),
                                       "seconds": round(time.time() - t, 3), "audio_s": round(len(pcm) / 16000, 2)})
                if self.path == "/llm":
                    if state["llm"] is None:
                        return self._send({"ok": False, "reason": state["llm_error"] or "LLM still loading"})
                    t = time.time()
                    ans = state["llm"].ask(str(body.get("question", ""))[:500], str(body.get("context", ""))[:12000],
                                           body.get("lang", "th"))
                    return self._send({"ok": True, "answer": ans, "seconds": round(time.time() - t, 2)})
            except Exception as e:  # noqa: BLE001
                return self._send({"ok": False, "reason": f"{type(e).__name__}: {e}"})
            self._send({"ok": False, "reason": "not found"}, 404)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"[AI] http://127.0.0.1:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
