"""S0 check of the operator server through its HTTP API (own sim + server on DDS domain 53, port 8101; both are
started and stopped here by PID, nothing else is touched).

    .venv/bin/python tools/server_check.py
"""

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

G1 = Path(__file__).resolve().parents[1]
DOMAIN, PORT = 53, 8101
URL = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(URL + path, timeout=5) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main():
    procs = [subprocess.Popen([sys.executable, "deploy/sim_robot.py", "--headless", "--domain", str(DOMAIN),
                               "--shop", "../shops/mockup/shop.json"], cwd=G1,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)]
    time.sleep(2)
    procs.append(subprocess.Popen([sys.executable, "deploy/server.py", "--audio-settings", "/dev/null", "--domain", str(DOMAIN), "--port", str(PORT),
                                   "--host", "127.0.0.1"], cwd=G1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    res = []

    def check(name, ok, info=""):
        res.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

    try:
        for _ in range(50):
            try:
                s = get("/api/status")
                if s["arm"]["state_ok"]:
                    break
            except OSError:
                pass
            time.sleep(0.2)
        check("server up, robot state ok", s["arm"]["state_ok"])
        post("/api/settings", {"audio": "off"})  # silent tests
        check("page served", b"exp05" in urllib.request.urlopen(URL + "/").read())
        check("shops listed", "mockup" in get("/api/shops")["shops"])
        # ---- shops (a temporary copy of the mockup, removed at the end)
        tmp = "zz_check_tmp"
        shutil.rmtree(G1.parent / "shops" / tmp, ignore_errors=True)  # left over from an interrupted run
        r = post("/api/shop/new", {"name": tmp, "copy_from": "mockup"})
        check("shop copy", r["ok"] and len(r["shop"]["booths"]) == 12, r.get("reason", ""))
        check("bad shop name refused", post("/api/shop/new", {"name": "../x"})["ok"] is False)
        shop = get(f"/api/shop?name={tmp}")["shop"]
        shop["robot"]["yaw_deg"] = 45.0
        shop["booths"][0]["name_en"] = "Changed"
        r = post("/api/shop/save", {"shop": shop})
        back = get(f"/api/shop?name={tmp}")
        check("shop save + reload", r["ok"] and back["shop"]["robot"]["yaw_deg"] == 45.0
              and back["shop"]["booths"][0]["name_en"] == "Changed" and "info" in back["geometry"])
        bad = json.loads(json.dumps(shop))
        bad["booths"][1]["id"] = bad["booths"][0]["id"]
        check("duplicate booth id refused", post("/api/shop/save", {"shop": bad})["ok"] is False)
        img = (G1.parent / "shops/mockup/map.png").read_bytes()
        req = urllib.request.Request(f"{URL}/api/shop/background?name={tmp}", img, {"Content-Type": "image/png"})
        r = json.loads(urllib.request.urlopen(req, timeout=5).read())
        check("background upload", r["ok"] and r["shop"]["background"]["size_px"] == [950, 825]
              and r["shop"]["background"]["resolution_m_per_px"] == 0.04, str(r.get("reason", "")))
        req = urllib.request.Request(f"{URL}/api/shop/background?name={tmp}", b"not an image", {"Content-Type": "image/png"})
        check("non-image refused", json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"] is False)
        im = urllib.request.urlopen(f"{URL}/shops/{tmp}/{r['shop']['background']['image']}").read()
        check("background served", im == img)
        check("active shop switch", post("/api/shop/active", {"name": tmp})["ok"] and get("/api/shops")["active"] == tmp)
        jpg = urllib.request.urlopen(f"{URL}/api/view.jpg?cam=top&booth=cafe", timeout=10).read()
        check("3D view rendered", jpg[:2] == b"\xff\xd8" and len(jpg) > 5000, f"{len(jpg)} bytes")
        st = post("/api/settings", {"audio": "laptop", "volume": 60, "speech_rate": 0.9})["settings"]
        hi = post("/api/settings", {"volume": 150, "speech_rate": 2.0})["settings"]
        bad = post("/api/settings", {"volume": "loud"})
        off = post("/api/settings", {"audio": "off", "volume": 10})["settings"]
        back = post("/api/settings", {"audio": "laptop", "speech_rate": 1.0})["settings"]
        check("volume + speaking speed settings", st["volume"]["laptop"] == 60 and st["speech_rate"] == 0.9
              and hi["volume"]["laptop"] == 100 and hi["speech_rate"] == 1.2 and not bad["ok"]
              and back["volume"]["laptop"] == 100 and off.get("volume", {}).get("off") is None,
              "set, clamped to 0-100 / 0.8-1.2, bad value refused, volume per output")
        post("/api/shop/active", {"name": "mockup"})
        shutil.rmtree(G1.parent / "shops" / tmp)

        r = post("/api/pose", {"name": "left_forward"})
        check("pose refused before take-over", r["ok"] is False, r.get("reason", ""))
        check("take-over", post("/api/arm", {"action": "takeover"})["ok"])
        time.sleep(2.0)
        a = get("/api/status")["arm"]
        check("active after the ramp", a["mode"] == "active" and a["weight"] == 1.0, f"{a['mode']} w={a['weight']}")
        post("/api/speed", {"scale": 1.0})
        post("/api/pose", {"name": "left_forward"})
        post("/api/hands", {"pose": "flat", "side": "both"})
        time.sleep(2.5)
        s = get("/api/status")
        a = s["arm"]
        i = s["joints"].index("left_shoulder_pitch_joint")
        check("pose reached", abs(a["q"][i] - (-1.2)) < 0.05, f"left shoulder pitch {a['q'][i]:.3f} rad")
        check("hands flat", a["hands"] is not None and min(a["hands"][:4]) > 0.9, f"{a['hands'][:6]}")
        b = s["balance"]
        check("balance reported", b is not None and b["zmp_margin"] > 0.02, f"ZMP margin {b and b['zmp_margin']}")
        post("/api/arm", {"action": "estop"})
        time.sleep(0.3)
        a = get("/api/status")["arm"]
        check("E-STOP -> releasing", a["mode"] == "releasing", a["reason"])
        time.sleep(2.0)
        s = get("/api/status")
        a = s["arm"]
        check("released, arm back at rest", a["mode"] == "released" and abs(a["q"][i] - 0.3) < 0.05,
              f"{a['mode']}, left shoulder pitch {a['q'][i]:.3f}")
        post("/api/arm", {"action": "takeover"})
        time.sleep(2.0)
        procs[0].terminate()  # robot state disappears
        procs[0].wait(timeout=5)
        time.sleep(1.0)
        a = get("/api/status")["arm"]
        check("state lost -> release", a["mode"] in ("releasing", "released") and not a["state_ok"], f"{a['mode']}, {a['reason']}")
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
