"""Screenshots of the operator page (headless Chrome) against a private sim + server (DDS domain 53, port 8101).

    .venv/bin/python tools/ui_shot.py OUT_DIR [fragment ...] [--point BOOTH]    e.g. "" "#shop"
--point: point at a booth (through the API) first, the screenshots show the action running.
"""
import json
import urllib.request
import subprocess
import sys
import tempfile
import time
from pathlib import Path

G1 = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    booth = None
    if "--point" in args:
        i = args.index("--point")
        booth = args[i + 1]
        del args[i:i + 2]
    out = Path(args[0])
    frags = args[1:] or ["", "#shop"]
    procs = [subprocess.Popen([sys.executable, "deploy/sim_robot.py", "--headless", "--domain", "53"], cwd=G1,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(2)
    procs.append(subprocess.Popen([sys.executable, "deploy/server.py", "--audio-settings", "/dev/null", "--domain", "53", "--port", "8101", "--host",
                                   "127.0.0.1"], cwd=G1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(4)
    try:
        if booth:
            urllib.request.urlopen("http://127.0.0.1:8101/api/point/plans", timeout=60).read()
            req = urllib.request.Request("http://127.0.0.1:8101/api/point", json.dumps({"booth": booth, "hold_s": None}).encode(),
                                         {"Content-Type": "application/json"})
            print("point:", json.loads(urllib.request.urlopen(req, timeout=10).read())["ok"])
            time.sleep(5)
        with tempfile.TemporaryDirectory() as prof:
            for i, f in enumerate(frags):
                png = out / f"ui_{i}.png"
                r = subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", f"--user-data-dir={prof}",
                                "--enable-logging=stderr", "--v=0",
                                "--window-size=1500,950", "--virtual-time-budget=6000", f"--screenshot={png}",
                                f"http://127.0.0.1:8101/{f}"], timeout=60, capture_output=True, text=True)
                errs = [ln for ln in r.stderr.splitlines() if "CONSOLE" in ln or "Uncaught" in ln]
                print(png, png.exists(), "console:", errs or "clean")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=6)


if __name__ == "__main__":
    main()
