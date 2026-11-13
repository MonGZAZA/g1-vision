"""S2 check: pointing.

A. IK only (no robot): the mockup shop's booths + a grid of virtual booths (azimuth -100..+100 deg, distance 1-45 m):
   azimuth error of the ray shoulder -> palm <= 5 deg, near lower than far, no self-collision, inside joint limits.
B. Live (own sim + server on DDS domain 52, port 8102, started and stopped here by PID): point at a near, a side and
   a straight-ahead booth through the HTTP API, measure the robot's ray, the hands, the balance margin, the return.

    .venv/bin/python tools/point_check.py [--ik-only]
"""

import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

G1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G1 / "deploy"))

import numpy as np  # noqa: E402

from pointing import AZ_TOL_DEG, Pointer  # noqa: E402
from robot_model import ARM_REST, UPPER, shop_booths_robot_frame  # noqa: E402

DOMAIN, PORT = 52, 8102
URL = f"http://127.0.0.1:{PORT}"
SHOP = G1.parent / "shops/mockup/shop.json"
res = []


def check(name, ok, info=""):
    res.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)


def part_a(ptr):
    shop = json.loads(SHOP.read_text())
    lim = {n: ptr.rng[n] for n in UPPER}
    bad, worst = [], 0.0
    for b in shop_booths_robot_frame(shop):
        r = ptr.solve(b["aim"], shop.get("settings"))
        worst = max(worst, abs(r.get("azimuth_error_deg", 99)))
        if not r["ok"]:
            bad.append(f"{b['id']}: {r['reason']}")
    check("mockup booths: all pointable, azimuth error <= 5 deg", not bad, f"worst {worst:.2f} deg {bad}")

    thetas, dists = range(-100, 101, 10), [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 45.0]
    fails, worst, coll, limits, order = [], 0.0, [], [], []
    for th in thetas:
        elevs = []
        for d in dists:
            a = math.radians(th)
            r = ptr.solve([d * math.cos(a), d * math.sin(a), 1.0])
            worst = max(worst, abs(r["azimuth_error_deg"]))
            if not r["ok"]:
                fails.append(f"({th},{d}) {r['reason']}")
            if r["collisions"]:
                coll.append((th, d))
            for n, v in zip(UPPER, r["q"]):
                if not lim[n][0] - 1e-6 <= v <= lim[n][1] + 1e-6:
                    limits.append((th, d, n))
            elevs.append(r["ray_elev_deg"])
        if any(e2 < e1 - 0.5 for e1, e2 in zip(elevs, elevs[1:])):
            order.append((th, [round(e) for e in elevs]))
    n = len(thetas) * len(dists)
    check(f"grid {n} targets: azimuth error <= {AZ_TOL_DEG:.0f} deg", worst <= AZ_TOL_DEG and not fails,
          f"worst {worst:.2f} deg, failures {fails[:4]}")
    check("grid: near always lower than far", not order, str(order[:3]))
    check("grid: no self-collision", not coll, str(coll[:5]))
    check("grid: inside joint limits", not limits, str(limits[:5]))


def get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def part_b(ptr):
    procs = [subprocess.Popen([sys.executable, "deploy/sim_robot.py", "--headless", "--domain", str(DOMAIN),
                               "--shop", str(SHOP)], cwd=G1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(2)
    procs.append(subprocess.Popen([sys.executable, "deploy/server.py", "--audio-settings", "/dev/null", "--domain", str(DOMAIN), "--port", str(PORT),
                                   "--host", "127.0.0.1"], cwd=G1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    try:
        for _ in range(60):
            try:
                if get("/api/status")["arm"]["state_ok"]:
                    break
            except OSError:
                pass
            time.sleep(0.25)
        post("/api/settings", {"audio": "off"})  # silent tests
        active = get("/api/shops")["active"]
        if active != "mockup":
            post("/api/shop/active", {"name": "mockup"})
        plans = {p["booth"]: p for p in get("/api/point/plans", timeout=60)["plans"]}
        check("plans for all booths", len(plans) == 12 and all(p["ok"] for p in plans.values()),
              f"{sum(p['ok'] for p in plans.values())}/{len(plans)}")
        check("unknown booth refused", post("/api/point", {"booth": "nope"})["ok"] is False)
        post("/api/speed", {"scale": 1.0})
        ui = {n: i for i, n in enumerate(UPPER)}
        for bid in ("new", "cafe", "robots"):
            p = plans[bid]
            r = post("/api/point", {"booth": bid, "hold_s": 2.5})  # takes over the arms on the first call
            if not r["ok"]:
                check(f"point {bid}", False, r.get("reason", ""))
                continue
            margins, t0, phase, s, t_hold = [], time.time(), "", None, None
            while time.time() - t0 < 15:
                s = get("/api/status")
                phase = s["action"].get("phase", "")
                if s["balance"]:
                    margins.append(s["balance"]["zmp_margin"])
                if phase == "hold" and t_hold is None:
                    t_hold = time.time()
                if t_hold is not None and time.time() - t_hold > 1.0:
                    break  # 1 s into the hold
                time.sleep(0.1)
            a = s["arm"]
            az, el = ptr.measure(a["q"], p["side"])
            err = (az - p["target_azimuth_deg"] + 180) % 360 - 180
            hs = slice(0, 4) if p["side"] == "right" else slice(6, 10)
            flat = a["hands"] is not None and min(a["hands"][hs]) > 0.9
            torso = math.degrees(a["q"][ui["waist_yaw_joint"]])
            check(f"point {bid} ({p['side']}, az {p['azimuth_deg']:+.0f}, {p['distance_m']:.1f} m): robot ray on target",
                  phase == "hold" and abs(err) <= AZ_TOL_DEG and abs(el - p["elev_deg"]) <= 6.0 and flat
                  and abs(torso - p["torso_deg"]) < 2.5,
                  f"az err {err:+.1f} deg, elev {el:.1f} (plan {p['elev_deg']:.1f}), torso {torso:.1f} (plan "
                  f"{p['torso_deg']:.1f}), hand flat {flat}")
            while time.time() - t0 < 20:
                s = get("/api/status")
                if s["balance"]:
                    margins.append(s["balance"]["zmp_margin"])
                if s["action"].get("phase") == "done":
                    break
                time.sleep(0.1)
            time.sleep(0.6)
            a = get("/api/status")["arm"]
            back = max(abs(a["q"][ui[n]] - ARM_REST.get(n, 0.0)) for n in UPPER)
            check(f"point {bid}: returns to rest, balance kept", s["action"].get("phase") == "done" and back < 0.06
                  and min(margins) > 0.02, f"max joint off rest {back:.3f} rad, min ZMP margin {min(margins) * 100:.1f} cm")
        # hold until the next action, then a new action replaces it smoothly
        post("/api/point", {"booth": "cashier", "hold_s": None})
        time.sleep(4)
        s1 = get("/api/status")["action"]
        post("/api/rest", {})
        time.sleep(3.5)
        s2 = get("/api/status")
        back = max(abs(s2["arm"]["q"][ui[n]] - ARM_REST.get(n, 0.0)) for n in UPPER)
        check("hold until next action, then Rest", s1.get("phase") == "hold" and s1.get("duration") is None
              and s2["action"]["name"] == "rest" and back < 0.06, f"{s1.get('phase')}, then {s2['action']}, off rest {back:.3f}")
        post("/api/point", {"booth": "phones", "hold_s": None})
        time.sleep(1.5)
        post("/api/arm", {"action": "estop"})
        time.sleep(2.0)
        s = get("/api/status")
        check("E-STOP during pointing -> released, action stopped", s["arm"]["mode"] == "released"
              and s["action"]["phase"] == "stopped", f"{s['arm']['mode']}, {s['action'].get('phase')}")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=6)
            except subprocess.TimeoutExpired:
                p.kill()


def main():
    ptr = Pointer()
    t = time.time()
    part_a(ptr)
    print(f"  (IK part {time.time() - t:.0f} s)")
    if "--ik-only" not in sys.argv:
        part_b(ptr)
    print(f"{sum(res)}/{len(res)} passed")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
