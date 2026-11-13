"""S3/S5/S6 check: voice + captions + timeline, introduction (wai) and service with gestures, keep-alive, Q&A.

Needs the AI service (exp05/ai/run_ai.sh, port 8105) running. Starts its own sim + server (DDS domain 54, port 8104,
by PID; nothing else is touched). Audio output "off" (silent) unless --audio laptop.

    .venv/bin/python tools/behaviour_check.py [--audio laptop] [--rate 0.8]
--rate: speaking speed (0.8-1.2) for the whole run; all timing checks must hold at any speed (the motion follows the
measured phrase times), and the speech must really be longer / shorter than at 1.0.
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

G1 = Path(__file__).resolve().parents[1]
DOMAIN, PORT = 54, 8104
URL = f"http://127.0.0.1:{PORT}"
SHOP = G1.parent / "shops/mockup/shop.json"
res = []


def check(name, ok, info=""):
    res.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)


def get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, body, timeout=90):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_action(path, body, limit=60):
    """Start an action and record (wall time, status) at ~20 Hz until it is done."""
    r = post(path, body)
    if not r.get("ok"):
        return r, []
    log, t0, started = [], time.time(), False
    while time.time() - t0 < limit:
        s = get("/api/status")
        a = s["action"]
        if a.get("name") == path.rsplit("/", 1)[1] and a.get("t") is not None \
                and a.get("phase") not in ("done", "stopped", "failed"):
            started = True  # (the previous action of the same kind may still show "done")
        if started:
            log.append(s)
            if a.get("phase") in ("done", "stopped", "failed"):
                break
        time.sleep(0.05)
    return r, log


def first_caption_t(log):
    for s in log:
        if s["caption"]:
            return s["action"]["t"], s["caption"]
    return None, None


def frozen_windows(log, win=1.0):
    """Number of 1 s windows in which no upper-body target changed by more than 0.3 deg."""
    if len(log) < 3:
        return 99
    ts = np.array([s["action"]["t"] for s in log])
    Q = np.array([s["arm"]["target"] for s in log])
    n = 0
    for t in np.arange(ts[0], ts[-1] - win, win):
        k = (ts >= t) & (ts < t + win)
        if k.sum() >= 3 and np.max(np.ptp(Q[k], axis=0)) < np.radians(0.3):
            n += 1
    return n


def main():
    audio = sys.argv[sys.argv.index("--audio") + 1] if "--audio" in sys.argv else "off"
    rate = float(sys.argv[sys.argv.index("--rate") + 1]) if "--rate" in sys.argv else 1.0
    h = None
    for _ in range(60):  # the LLM loads in the background after the TTS (up to ~1 min)
        try:
            h = json.loads(urllib.request.urlopen("http://127.0.0.1:8105/health", timeout=2).read())
            if h["llm"] or h["llm_error"]:
                break
        except OSError:
            pass
        time.sleep(2)
    if h is None:
        print("AI service not running: start exp05/ai/run_ai.sh first")
        sys.exit(2)
    check("AI service: TTS + LLM loaded", h["tts"] and h["llm"], f"GPU {h['gpu_gb']} GB, clone {h['clone']}")
    procs = [subprocess.Popen([sys.executable, "deploy/sim_robot.py", "--headless", "--domain", str(DOMAIN),
                               "--shop", str(SHOP)], cwd=G1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(2)
    procs.append(subprocess.Popen([sys.executable, "deploy/server.py", "--audio-settings", "/dev/null", "--domain", str(DOMAIN), "--port", str(PORT),
                                   "--host", "127.0.0.1"], cwd=G1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    try:
        for _ in range(80):
            try:
                if get("/api/status")["arm"]["state_ok"]:
                    break
            except OSError:
                pass
            time.sleep(0.25)
        if get("/api/shops")["active"] != "mockup":
            post("/api/shop/active", {"name": "mockup"})
        post("/api/settings", {"audio": audio, "idle": True, "speech_rate": rate})
        if rate != 1.0:  # the TTS itself speaks at that speed (not a faster playback)
            def dur(sp):
                req = urllib.request.Request("http://127.0.0.1:8105/tts", json.dumps(
                    {"text": "สวัสดีค่ะ ยินดีต้อนรับสู่ร้านของเรานะคะ ถ้าต้องการความช่วยเหลือ สอบถามได้เลยค่ะ", "lang": "th",
                     "speed": sp}).encode(), {"Content-Type": "application/json"})
                d = json.loads(urllib.request.urlopen(req, timeout=60).read())
                return d["duration"], d["phrases"]
            (d1, p1), (dr, pr) = dur(1.0), dur(rate)
            ratio = dr / d1
            check(f"speech at {rate}x: duration scales, same phrases", abs(ratio - 1 / rate) < 0.12 and len(p1) == len(pr),
                  f"{d1:.2f} s at 1.0 -> {dr:.2f} s (x{ratio:.2f}, expected ~x{1 / rate:.2f})")
        s_ = get("/api/status")["settings"]
        check("speaking speed set", abs(s_.get("speech_rate", 0) - rate) < 1e-6, f"{s_.get('speech_rate')}")
        post("/api/speed", {"scale": 1.0})
        time.sleep(1.0)

        # ---- pointing with the booth sentence, both languages: speech starts as the arm arrives
        for lang, booth in (("th", "cafe"), ("en", "cashier")):
            r, log = run_action("/api/point", {"booth": booth, "lang": lang, "hold_s": 1.0})
            mk = log[0]["action"].get("marks", {}) if log else {}
            tc, cap = first_caption_t(log)
            err = None if tc is None else tc - mk.get("speech", 0)
            caps = [s["caption"]["text"] for s in log if s["caption"]]
            check(f"point {booth} ({lang}): caption starts with the speech, as the arm arrives",
                  r.get("ok") and err is not None and -0.06 <= err <= 0.12 and cap["lang"] == lang,
                  f"caption at {tc}, speech mark {mk.get('speech')}, arm up {mk.get('arm_up')}: {caps[:1]}")
            time.sleep(0.5)

        # ---- introduction: wai with "สวัสดีค่ะ"/"Hello" in its hold, then gestures; never frozen
        for lang, hello in (("th", "สวัสดีค่ะ"), ("en", "Hello!")):
            r, log = run_action("/api/intro", {"lang": lang})
            phases = []
            for s in log:
                p = s["action"].get("phase")
                if p and (not phases or phases[-1] != p):
                    phases.append(p)
            hold_caps = {s["caption"]["text"] for s in log if s["caption"] and s["action"].get("phase") == "wai: hold"}
            gest = [p for p in phases if p.startswith("gesture")]
            fr = frozen_windows(log)
            margins = [s["balance"]["zmp_margin"] for s in log if s["balance"]]
            worst = min((s for s in log if s["balance"]), key=lambda s: s["balance"]["zmp_margin"])
            check(f"intro ({lang}): wai raise -> hold -> lower, '{hello}' spoken in the hold",
                  r.get("ok") and phases[:3] == ["wai: raise", "wai: hold", "wai: lower"] and hello in hold_caps,
                  f"phases {phases}, captions in hold {sorted(hold_caps)}")
            check(f"intro ({lang}): gestures while speaking, never frozen, balance kept",
                  len(gest) >= 2 and fr == 0 and min(margins) > 0.02,
                  f"gestures {gest}, frozen 1 s windows {fr}, min ZMP margin {min(margins) * 100:.1f} cm (in {worst['action'].get('phase')}), "
                  f"{log[-1]['action']['t']:.1f} s")
            time.sleep(0.5)
        r, log = run_action("/api/service", {"lang": "en"})
        gest = {s["action"].get("phase") for s in log if str(s["action"].get("phase")).startswith("gesture")}
        check("service (en): spoken with gestures, never frozen", r.get("ok") and len(gest) >= 2 and frozen_windows(log) == 0,
              f"{sorted(gest)}, {log[-1]['action']['t']:.1f} s" if log else str(r))

        # ---- keep-alive while standing with no action
        time.sleep(1.0)
        qs = []
        for _ in range(40):
            qs.append(get("/api/status")["arm"]["target"])
            time.sleep(0.05)
        rng = np.degrees(np.max(np.ptp(np.array(qs), axis=0)))
        check("standing between actions: keep-alive idle moves the arms (never frozen)", rng > 0.5, f"{rng:.2f} deg in 2 s")

        # ---- Q&A: language, FAQ, LLM from the shop data, booth hand-off
        cases = [("ร้านเปิดกี่โมงคะ", "th", None, "FAQ"), ("Where is the restroom?", "en", "restroom", None),
                 ("ร้านกาแฟอยู่ตรงไหนคะ", "th", "cafe", None), ("Can I fix my phone here?", "en", "service", None)]
        for q, lang, booth, src in cases:
            r = post("/api/ask", {"question": q})
            ok = r.get("ok") and r["lang"] == lang and (booth is None or r.get("booth") == booth) \
                and (src is None or r["source"].startswith(src)) and r["source"] != "fallback"
            check(f"ask '{q}'", ok, f"-> [{r.get('source')}, {r.get('seconds')} s, booth {r.get('booth')}] {r.get('answer')}")
        r = post("/api/ask", {"question": "Where is the restroom?"})
        r2, log = run_action("/api/point", {"booth": r["booth"], "text": r["answer"], "lang": "en", "hold_s": 0.5})
        caps = " ".join(dict.fromkeys(s["caption"]["text"] for s in log if s["caption"]))
        check("answer spoken while pointing at the booth", r2.get("ok") and log and log[0]["action"].get("booth") == "restroom"
              and caps.split(" ")[0] in r["answer"], caps[:80])

        # ---- Stop during speech, E-STOP
        post("/api/service", {"lang": "th"})
        time.sleep(3.0)
        post("/api/stop", {})
        time.sleep(1.5)
        s = get("/api/status")
        check("Stop: speech ends, arms to rest", s["caption"] is None and s["action"]["name"] == "rest", str(s["action"].get("name")))
        post("/api/intro", {"lang": "th"})
        time.sleep(3.0)
        post("/api/arm", {"action": "estop"})
        time.sleep(2.0)
        s = get("/api/status")
        check("E-STOP during the intro: released, caption cleared", s["arm"]["mode"] == "released" and s["caption"] is None
              and s["action"]["phase"] == "stopped", f"{s['arm']['mode']}, {s['action'].get('phase')}")
        live = get("/api/live")
        mon = urllib.request.urlopen(URL + "/monitor", timeout=3).read()
        check("monitor page + live endpoint", b"greeter monitor" in mon and "caption" in live)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=6)
            except subprocess.TimeoutExpired:
                p.kill()
    print(f"{sum(res)}/{len(res)} passed")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
