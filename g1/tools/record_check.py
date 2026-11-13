"""Check of the voice recording in the page (Scripts & voice -> "Upload a reference voice"), in real Chrome.

Starts its own sim + server (private DDS domain and port), opens the page in headless Chrome with a FAKE microphone
(Chrome's test tone) through the DevTools protocol, and checks:
  1. the reading scripts are listed and picking one fills the transcript (so the transcript matches the reading)
  2. Record -> Stop gives a WAV (the AI service cannot read the browser's webm), playable, with its length shown
  3. Upload -> the AI service stores the voice (ref.wav length about the recorded time, ref.txt = the script)
  4. a recording longer than 12 s stops by itself
The test voice is deleted at the end. Needs the AI service running (it stores the voices), like behaviour_check.
Usage: .venv/bin/python tools/record_check.py
"""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import wave

import websockets

G1 = Path(__file__).resolve().parents[1]
EXP05 = G1.parent
DOMAIN, PORT, CDP = 55, 8103, 9233
VOICE = "zz_check_rec"
VOICE_DIR = EXP05 / "ai/voices" / VOICE


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def js(self, expr, timeout=30):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": "Runtime.evaluate",
                                       "params": {"expression": expr, "awaitPromise": True, "returnByValue": True}}))
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = json.loads(await asyncio.wait_for(self.ws.recv(), timeout))
            if m.get("id") == self.n:
                r = m["result"]
                if "exceptionDetails" in r:
                    raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
                return r["result"].get("value")
        raise TimeoutError(expr)


async def run(check):
    url = f"http://127.0.0.1:{PORT}/"
    for _ in range(60):
        try:
            targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{CDP}/json", timeout=2).read())
            tgt = next(t for t in targets if t["type"] == "page")
            break
        except (OSError, StopIteration):
            await asyncio.sleep(0.5)
    async with websockets.connect(tgt["webSocketDebuggerUrl"], max_size=None) as ws:
        p = Page(ws)
        await p.js(f"location.href = {json.dumps(url)}")
        await asyncio.sleep(3)
        await p.js("document.querySelector('.tab[data-page=scripts]').click()")
        await asyncio.sleep(1.5)
        n = await p.js("document.getElementById('recScript').options.length")
        await p.js("document.getElementById('recScript').value = '1'; pickScript()")
        txt = await p.js("document.getElementById('vText').value")
        shown = await p.js("document.getElementById('recText').textContent")
        check("reading scripts listed, picking one fills the transcript", n == 4 and txt == shown and txt.startswith("ห้องน้ำ"),
              f"{n} scripts")

        await p.js("toggleRecord()")
        await asyncio.sleep(6.5)
        await p.js("toggleRecord()")
        await asyncio.sleep(2.0)
        info = await p.js("(async () => recBlob ? {size: recBlob.size, type: recBlob.type, head: "
                          "new TextDecoder().decode((await recBlob.slice(0, 12).arrayBuffer())), "
                          "label: document.getElementById('recTime').textContent, "
                          "player: document.getElementById('recPlay').style.display} : null)()")
        check("record -> WAV, playable, length shown", info and info["type"] == "audio/wav" and info["head"].startswith("RIFF")
              and info["head"].endswith("WAVE") and info["player"] == "" and "recorded" in info["label"],
              f"{info['size'] // 1024 if info else 0} KB, '{info['label'] if info else ''}'")

        await p.js(f"document.getElementById('vName').value = '{VOICE}'; document.getElementById('vConsent').checked = true;"
                   "uploadVoice()")
        msg = ""
        for _ in range(40):
            await asyncio.sleep(0.5)
            msg = await p.js("document.getElementById('vmsg').textContent")
            if msg and msg != "uploading…":
                break
        ok = VOICE_DIR.exists() and (VOICE_DIR / "ref.txt").read_text(encoding="utf-8") == shown
        secs = 0.0
        if ok:
            with wave.open(str(VOICE_DIR / "ref.wav")) as w:
                secs = w.getnframes() / w.getframerate()
        check("upload: voice stored with the script as transcript", ok and 4.5 <= secs <= 7.5 and "uploaded" in msg,
              f"'{msg}', ref.wav {secs:.1f} s (with 0.5 s silence added)")
        listed = await p.js("[...document.getElementById('voiceSel').options].map(o => o.value)")
        check("new voice in the voice list", VOICE in (listed or []))

        await p.js("toggleRecord()")
        await asyncio.sleep(13.5)
        label = await p.js("document.getElementById('recTime').textContent")
        btn = await p.js("document.getElementById('recBtn').textContent")
        check("recording stops by itself at 12 s", "recorded" in label and "Record" in btn, f"'{label}'")


def main():
    try:
        urllib.request.urlopen("http://127.0.0.1:8105/health", timeout=3)
    except OSError:
        sys.exit("the AI service is not running (start ./run_sim.sh or ai/run_ai.sh first)")
    if VOICE_DIR.exists():
        shutil.rmtree(VOICE_DIR)
    procs = [subprocess.Popen([sys.executable, "deploy/sim_robot.py", "--headless", "--domain", str(DOMAIN)], cwd=G1,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(2)
    procs.append(subprocess.Popen([sys.executable, "deploy/server.py", "--audio-settings", "/dev/null", "--domain", str(DOMAIN), "--port", str(PORT),
                                   "--host", "127.0.0.1"], cwd=G1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    results = []

    def check(name, ok, info=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

    prof = tempfile.mkdtemp()
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--disable-gpu", f"--user-data-dir={prof}",
                               f"--remote-debugging-port={CDP}", "--use-fake-device-for-media-stream",
                               "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required",
                               "--window-size=1500,950", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(4)
        asyncio.run(run(check))
    except Exception as e:  # noqa: BLE001
        check("page test ran", False, f"{type(e).__name__}: {e}")
    finally:
        chrome.terminate()
        chrome.wait(timeout=10)
        shutil.rmtree(prof, ignore_errors=True)
        for p in reversed(procs):
            p.terminate()
            p.wait(timeout=10)
        shutil.rmtree(VOICE_DIR, ignore_errors=True)
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
