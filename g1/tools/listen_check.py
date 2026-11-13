"""2.2 check: speech recognition (deploy/listen.py + AI service /asr) end to end, without a microphone.

Questions are spoken by the AI service's own TTS (16 kHz), padded with silence and light noise, and posted to
/api/listen/audio, which the listener hears as if from the microphone (in real time). Starts its own sim + server
(DDS domain 56, port 8109, --listen-source none, audio off; by PID, nothing else is touched).

    .venv/bin/python tools/listen_check.py [--ai-url http://127.0.0.1:8105]

Needs the AI service with speech recognition loaded (exp05/ai/run_ai.sh). With --no-llm the questions that are not
in the FAQ get the "unknown" answer, which is enough here.
Note: the TTS voice is synthetic, so recognition is checked loosely (similarity); real accuracy needs real voices.
"""

import argparse
import base64
import difflib
import io
import json
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

G1 = Path(__file__).resolve().parents[1]
DOMAIN, PORT = 56, 8109
URL = f"http://127.0.0.1:{PORT}"
res = []


def check(name, ok, info=""):
    res.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)


def get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, body, timeout=90, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data, {"Content-Type": "audio/wav" if raw else "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tts16(ai, text, lang):
    req = urllib.request.Request(ai + "/tts", json.dumps({"text": text, "lang": lang}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    with wave.open(io.BytesIO(base64.b64decode(d["wav16_b64"]))) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


RNG = np.random.default_rng(0)


def wav(x, lead=0.6, tail=1.3, noise_db=-50):
    """Speech at a realistic level (peak -6 dB) with silence around it and light background noise."""
    x = x / (np.max(np.abs(x)) or 1) * 0.5
    x = np.concatenate([np.zeros(int(lead * 16000)), x, np.zeros(int(tail * 16000))])
    x = x + RNG.normal(0, 10 ** (noise_db / 20), len(x))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def hear(audio, wait=6.0):
    """Inject audio; -> new listener lines (after recognition and any answer has settled)."""
    before = get("/api/listen")["seq"]
    post("/api/listen/audio", None, raw=audio)
    t0 = time.time()
    while time.time() - t0 < wait:
        L = get("/api/listen")
        if L["state"] not in ("recognising", "answering", "hearing") and (L["seq"] > before or time.time() - t0 > 1.5):
            break
        time.sleep(0.1)
    time.sleep(0.3)
    L = get("/api/listen")
    return [ln for ln in L["lines"] if ln["id"] > before]


def sim(a, b):
    n = lambda s: "".join(c for c in s.lower() if c.isalnum())  # noqa: E731
    return difflib.SequenceMatcher(None, n(a), n(b)).ratio()


def wait_idle(limit=30):
    t0 = time.time()
    while time.time() - t0 < limit:
        s = get("/api/status")
        if s["action"].get("phase") in ("done", "stopped", "failed", "") and not s["caption"]:
            return True
        time.sleep(0.2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-url", default="http://127.0.0.1:8105")
    a = ap.parse_args()
    h = json.loads(urllib.request.urlopen(a.ai_url + "/health", timeout=3).read())
    if not h.get("asr"):
        sys.exit(f"the AI service at {a.ai_url} has no speech recognition loaded: {h.get('asr_error') or 'still loading?'}")
    procs = [subprocess.Popen([sys.executable, "deploy/sim_robot.py", "--headless", "--domain", str(DOMAIN),
                               "--shop", "../shops/mockup/shop.json"], cwd=G1,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(2)
    procs.append(subprocess.Popen([sys.executable, "deploy/server.py", "--audio-settings", "/dev/null", "--domain",
                                   str(DOMAIN), "--port", str(PORT), "--host", "127.0.0.1", "--listen-source", "none",
                                   "--ai-url", a.ai_url], cwd=G1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    try:
        for _ in range(60):
            try:
                if get("/api/status")["arm"]["state_ok"]:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        post("/api/settings", {"audio": "off"})
        post("/api/shop/active", {"name": "mockup"})
        get("/api/point/plans", timeout=300)  # pointing plans of a fresh server: the first answer must not wait for IK
        print("rendering test questions with the TTS ...", flush=True)
        # (the built-in Thai voice slurs some words, e.g. "ห้องน้ำ" is heard as "ห้องนอน" even straight from the TTS:
        # these sentences come out clearly)
        q_faq = wav(tts16(a.ai_url, "อยากทราบว่าร้านเปิดกี่โมงคะ", "th"))
        q_booth = wav(tts16(a.ai_url, "ขอโทษค่ะ ร้านกาแฟอยู่ที่ไหนคะ", "th"))
        q_en = wav(tts16(a.ai_url, "Where can I buy a phone case?", "en"))
        chatter = wav(tts16(a.ai_url, "เมื่อวานฝนตกหนักมากเลยนะ รถติดมาก", "th"))
        noise = wav(np.zeros(16000), noise_db=-32)

        L = get("/api/listen")
        check("listener present, mode off, VAD loaded", L["mode"] == "off" and not L["error"], L.get("error", ""))
        check("mode off: nothing is heard", not hear(q_faq, wait=2))

        post("/api/listen", {"mode": "auto", "reply": "review", "relevance": True, "presence_gate": False})
        new = hear(q_faq)
        ln = new[0] if new else {}
        check("auto/review: Thai FAQ question recognised", len(new) == 1 and ln.get("action") == "review"
              and sim(ln.get("text", ""), "อยากทราบว่าร้านเปิดกี่โมงคะ") >= 0.8,
              f"{ln.get('text')!r} conf {ln.get('confidence')} ASR {ln.get('asr_s')} s")
        check("ASR latency under 1.5 s for a short question", ln.get("asr_s", 9) < 1.5,
              f"{ln.get('asr_s')} s (~0.3 s on an idle GPU; the sim and other jobs share it)")
        check("segment trimmed by VAD (not the whole padded clip)", ln.get("audio_s", 99) < 3.3, f"{ln.get('audio_s')} s")

        new = hear(q_en)
        ln = new[0] if new else {}
        check("English question recognised (same model)", len(new) == 1 and ln.get("lang") == "en"
              and sim(ln.get("text", ""), "Where can I buy a phone case") >= 0.8, repr(ln.get("text")))

        check("noise only: VAD hears no speech", not hear(noise, wait=2))

        new = hear(chatter)
        ln = new[0] if new else {}
        check("chatter (not a question) is ignored in auto mode", len(new) <= 1 and (not new or ln.get("ignored") ==
              "not a question for the robot"), f"{ln.get('text')!r} {ln.get('ignored')}")

        post("/api/listen", {"reply": "auto"})
        new = hear(q_booth, wait=30)
        ln = new[0] if new else {}
        check("auto reply: booth question -> answer + point at the cafe", ln.get("action") == "point cafe",
              f"{ln.get('text')!r} -> {ln.get('action')} / {ln.get('answer')}")
        if ln.get("source") == "fallback":  # AI service without the LLM
            check("no LLM answer: the booth sentence is said instead of 'I don't know'",
                  "ทางนี้" in (ln.get("answer") or ""), ln.get("answer", ""))
        t0 = time.time()
        while time.time() - t0 < 8:  # the arms are taken over first (ramp), then the clip starts
            s = get("/api/status")
            if s["action"].get("name") == "point" and s["action"].get("t") is not None:
                break
            time.sleep(0.1)
        check("the robot is speaking the answer (point action running)", s["action"].get("name") == "point"
              and s["action"].get("phase") not in ("failed",), f"{s['action'].get('phase')} after {time.time() - t0:.1f} s")
        L = get("/api/listen")
        check("listener muted while the robot speaks", L["state"] == "muted", L["state"])
        new = hear(q_faq, wait=3)
        check("a question during the robot's speech is not heard (half-duplex)", not new, str([x.get("text") for x in new]))
        wait_idle()
        time.sleep(0.8)

        post("/api/listen", {"mode": "ptt", "reply": "review"})
        check("ptt, button up: nothing is heard", not hear(q_faq, wait=2))
        post("/api/listen/ptt", {"down": True})
        before = get("/api/listen")["seq"]
        post("/api/listen/audio", None, raw=q_en)
        post("/api/listen/ptt", {"down": False})
        post("/api/listen/audio", None, raw=wav(np.zeros(1600), lead=0.1, tail=0.2))  # frames after the release
        t0 = time.time()
        while time.time() - t0 < 5 and get("/api/listen")["seq"] == before:
            time.sleep(0.1)
        time.sleep(0.3)
        new = [x for x in get("/api/listen")["lines"] if x["id"] > before]
        ln = new[0] if new else {}
        check("ptt, button held: question recognised after release", len(new) == 1 and
              sim(ln.get("text", ""), "Where can I buy a phone case") >= 0.8, repr(ln.get("text")))

        post("/api/listen", {"mode": "auto", "presence_gate": True})
        check("presence gate, nobody there: nothing is heard", not hear(q_faq, wait=2))
        post("/api/listen/presence", {"present": True, "hold_s": 10})
        new = hear(q_faq)
        check("presence gate, customer present: question heard", len(new) == 1 and new[0].get("action") == "review",
              str([x.get("text") for x in new]))
        r = post("/api/listen", {"mode": "loud"})
        check("bad mode refused", r.get("ok") is False, r.get("reason", ""))
        post("/api/listen", {"mode": "off"})
        time.sleep(0.3)
        check("mode off again: microphone closed", get("/api/listen")["mic"] is False)
    finally:
        for p in procs[::-1]:
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
    print(f"\n{sum(res)}/{len(res)} passed", flush=True)
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
