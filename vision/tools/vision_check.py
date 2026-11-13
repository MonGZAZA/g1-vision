"""Checks of the vision module against the sensor sim's ground truth (own processes on free ports).

Starts vision.sim_sensors (mockup shop, scripted people) and vision.service, plays scenarios and compares
GET /people and /events with GET /truth:
  approach      a customer walks up and waits facing the robot: events, distance, speed, facing, bearing, one id
  facing away   a person stands in front, back to the robot: seen, not facing, no person_facing event
  passer-by     crosses at 3 m: speed, not approaching, never "near", no person_facing
  side visitor  comes beside the robot (outside the camera view) to 0.8 m: lidar track, too_close left, then clear
  two people    side by side at 2.5 m: two tracks, both facing
  self filter   nobody, arms pointing into the lidar view: no track, no too_close
  real video    (if exp05/videos/examples has a video) the detector on real frames: person found, facing

    cd exp05/vision && .venv/bin/python tools/vision_check.py [--keep-logs]
"""

import argparse
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP05 = HERE.parent
PY = str(HERE / ".venv/bin/python")
RESULTS = []


def free_port(kind=socket.SOCK_STREAM):
    s = socket.socket(socket.AF_INET, kind)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def get(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3.0) as r:
        return json.loads(r.read())


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


class Run:
    def __init__(self, sim, vis):
        self.sim, self.vis = sim, vis

    def play(self, people, seconds, q=None, hz=5.0):
        """Set the scene, record (truth, people) samples and the events of this scene."""
        post(f"{self.sim}/joints", {"q": q})
        t_end = time.time() + 5.0  # tracks of the previous scene are dropped ~1.2 s after it ends
        time.sleep(1.0)
        while get(f"{self.vis}/people?all=1")["people"] and time.time() < t_end:
            time.sleep(0.2)
        seq0 = get(f"{self.vis}/people")["events_seq"]
        post(f"{self.sim}/scenario", {"people": people})
        samples, t0 = [], time.time()
        while time.time() - t0 < seconds:
            tr = get(f"{self.sim}/truth")
            pe = get(f"{self.vis}/people?all=1")
            samples.append((tr["t"], tr["people"], pe["people"], pe["proximity"]))
            time.sleep(1 / hz)
        evs = get(f"{self.vis}/events?after={seq0}&wait=0")["events"]
        post(f"{self.sim}/scenario", {"people": []})
        return samples, evs


def nearest(people, t, max_d=0.8, confirmed=True):
    best = min((p for p in people if p["confirmed"] or not confirmed),
               key=lambda p: math.hypot(p["x"] - t["x"], p["y"] - t["y"]), default=None)
    if best is None or math.hypot(best["x"] - t["x"], best["y"] - t["y"]) > max_d:
        return None
    return best


def med(v):
    return statistics.median(v) if v else float("nan")


def case_approach(run):
    print("approach: a customer walks up and waits facing the robot")
    people = [{"name": "customer", "start": [6.0, -1.0], "steps": [
        {"to": [1.5, 0.0], "speed": 1.0, "face": "motion"}, {"wait": 6.0, "face": "robot"}]}]
    walk_s = math.hypot(4.5, 1.0) / 1.0
    samples, evs = run.play(people, walk_s + 6.0)
    first_seen, ids, d_err, b_err, spd, rad, facing = None, set(), [], [], [], [], []
    for t, truth, pe, _ in samples:
        if not truth:
            continue
        tr = truth[0]
        p = nearest(pe, tr)
        if p is None:
            continue
        first_seen = first_seen if first_seen is not None else t
        ids.add(p["id"])
        if t > walk_s + 1.5:  # standing
            d_err.append(abs(p["distance_m"] - tr["distance_m"]))
            facing.append(p["facing"] is True)
            b_err.append(abs(p["bearing_deg"] - tr["bearing_deg"]))
        elif 1.5 < t < walk_s - 0.5:  # walking steadily
            spd.append(p["speed_mps"])
            rad.append(p["radial_mps"])
    check("confirmed within 2 s at ~6 m", first_seen is not None and first_seen < 2.0, f"first at {first_seen}")
    kinds = [e["type"] for e in evs]
    want = ["person_new", "person_facing", "person_near", "person_dwell"]
    check("events new, facing, near, dwell", all(k in kinds for k in want), " ".join(kinds))
    check("distance error standing < 0.10 m", med(d_err) < 0.10, f"median {med(d_err):.3f} m, n={len(d_err)}")
    check("bearing error standing < 3 deg", med(b_err) < 3.0, f"median {med(b_err):.2f} deg")
    check("walking speed 1.0 +- 0.2 m/s", abs(med(spd) - 1.0) < 0.2, f"median {med(spd):.2f}")
    check("approaching (radial < -0.6 m/s)", med(rad) < -0.6, f"median radial {med(rad):.2f}")
    check("facing while waiting >= 90 %", facing and sum(facing) / len(facing) >= 0.9,
          f"{sum(facing)}/{len(facing)}")
    check("one track id", len(ids) == 1, f"ids {sorted(ids)}")
    near = next((e for e in evs if e["type"] == "person_near"), None)
    if near:
        check("person_near carries distance + facing", "distance_m" in near["person"] and "facing" in near["person"],
              f"d={near['person']['distance_m']} facing={near['person']['facing']}")


def case_facing_away(run):
    print("facing away: a person stands in front with the back to the robot")
    people = [{"name": "back", "start": [2.2, 0.4], "steps": [{"wait": 7.0, "face": 0}]}]
    samples, evs = run.play(people, 7.0)
    seen, facing = 0, []
    for t, truth, pe, _ in samples:
        if t < 2.0 or not truth:
            continue
        p = nearest(pe, truth[0])
        if p is not None:
            seen += 1
            if p["facing"] is not None:
                facing.append(p["facing"])
    check("tracked (camera + lidar)", seen >= 0.8 * max(1, len(samples) - 10), f"{seen} samples")
    check("not facing >= 90 %", facing and facing.count(False) / len(facing) >= 0.9,
          f"{facing.count(False)}/{len(facing)} false")
    check("no person_facing event", not any(e["type"] == "person_facing" for e in evs))


def case_passer_by(run):
    print("passer-by: crosses in front at 3 m, 1.4 m/s, looking ahead")
    people = [{"name": "passer", "start": [3.0, -4.0], "steps": [
        {"to": [3.0, 4.0], "speed": 1.4, "face": "motion"}, {"wait": 2.0, "face": 90}]}]
    samples, evs = run.play(people, 8.0 / 1.4 + 1.0)
    spd, rad, d_err = [], [], []
    for t, truth, pe, _ in samples:
        if not truth or not (1.2 < t < 8.0 / 1.4 - 0.5):
            continue
        p = nearest(pe, truth[0], confirmed=False)
        if p is not None:
            spd.append(p["speed_mps"])
            d_err.append(abs(p["distance_m"] - truth[0]["distance_m"]))
            if abs(truth[0]["y"]) < 1.0:
                rad.append(p["radial_mps"])
    check("speed 1.4 +- 0.25 m/s", abs(med(spd) - 1.4) < 0.25, f"median {med(spd):.2f}, n={len(spd)}")
    check("distance error walking < 0.30 m", med(d_err) < 0.30, f"median {med(d_err):.2f}")
    check("not approaching while in front", not rad or abs(med(rad)) < 0.4, f"median radial {med(rad):.2f}")
    kinds = [e["type"] for e in evs]
    if os.environ.get("VCHECK_DEBUG"):
        for e in evs:
            print("    ", e["type"], {k: e.get("person", {}).get(k) for k in ("id", "x", "y", "source", "speed_mps")})
    check("never near, never facing", "person_near" not in kinds and "person_facing" not in kinds, " ".join(kinds))


def case_side(run):
    print("side visitor: comes beside the robot (outside the camera view) to 0.8 m, then leaves")
    people = [{"name": "side", "start": [-0.5, 3.5], "steps": [
        {"to": [0.1, 0.8], "speed": 0.8, "face": "motion"}, {"wait": 4.0, "face": 0},
        {"to": [-0.5, 3.5], "speed": 0.8, "face": "motion"}, {"wait": 4.0, "face": 0}]}]
    leg = math.hypot(0.6, 2.7) / 0.8
    samples, evs = run.play(people, 2 * leg + 4.0 + 3.5)
    tracked, src, fac, d_err = 0, set(), set(), []
    for t, truth, pe, _ in samples:
        if not truth:
            continue
        p = nearest(pe, truth[0])
        if p is not None:
            tracked += 1
            src.add(p["source"])
            fac.add(p["facing"])
            if leg + 1.0 < t < leg + 4.0:
                d_err.append(abs(p["distance_m"] - truth[0]["distance_m"]))
    close = [e for e in evs if e["type"] == "too_close"]
    check("tracked by the lidar (confirmed by walking)", tracked > 0 and src <= {"lidar"}, f"sources {src}")
    check("facing unknown (null) outside the camera", fac <= {None}, f"facing values {fac}")
    check("distance error at 0.8 m < 0.15 m", med(d_err) < 0.15, f"median {med(d_err):.3f}")
    check("too_close (left)", close and "left" in close[0]["sectors"], json.dumps(close[0]["sectors"]) if close else "none")
    check("clear after leaving", any(e["type"] == "clear" for e in evs))


def case_two(run):
    print("two people: side by side at 2.5 m, facing the robot")
    people = [{"name": "a", "start": [2.5, 0.6], "steps": [{"wait": 7.0, "face": "robot"}]},
              {"name": "b", "start": [2.5, -0.6], "steps": [{"wait": 7.0, "face": "robot"}]}]
    samples, evs = run.play(people, 7.0)
    ok_n, both_facing = 0, 0
    late = [s for s in samples if s[0] > 3.0]
    if os.environ.get("VCHECK_DEBUG"):
        for t, truth, pe, _ in late:
            print("    ", t, [(p["id"], p["x"], p["y"], p["source"], p["confirmed"], p["facing"]) for p in pe])
    for t, truth, pe, _ in late:
        ps = [nearest(pe, tr) for tr in truth]
        if all(ps) and ps[0]["id"] != ps[1]["id"]:
            ok_n += 1
            both_facing += all(p["facing"] for p in ps)
    check("two separate tracks", ok_n >= 0.9 * len(late), f"{ok_n}/{len(late)}")
    check("both facing", both_facing >= 0.8 * len(late), f"{both_facing}/{len(late)}")


def case_self(run):
    print("self filter: nobody around, arms pointing forward and to the side")
    q = [0.3, -1.5, 1.4, 0, 1.4, 0, 0, 0, -1.5, 0, 0, 1.45, 0, 0, 0]
    samples, evs = run.play([], 5.0, q=q)
    n = max(len(pe) for _, _, pe, _ in samples)
    if os.environ.get("VCHECK_DEBUG"):
        for t, _, pe, _ in samples:
            if pe:
                print("    ", t, [{k: p[k] for k in ("id", "x", "y", "source", "confirmed", "age_s")} for p in pe])
    check("no track from the robot's own arms", n == 0, f"max tracks {n}")
    check("no too_close", not any(pr.get("too_close") for *_, pr in samples) and
          not any(e["type"] == "too_close" for e in evs))


def case_video():
    vids = sorted((EXP05 / "videos/examples").glob("*.mp4"))
    if not vids:
        print("real video: no example video, skipped")
        return
    print(f"real video: detector on frames of {vids[0].name}")
    import cv2

    sys.path.insert(0, str(HERE))
    from vision.detector import PersonDetector

    det = PersonDetector("full")
    cap = cv2.VideoCapture(str(vids[0]))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found = facing = total = 0
    for k in range(30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (0.1 + 0.8 * k / 30)))
        ok, img = cap.read()
        if not ok:
            continue
        total += 1
        d = det.detect(img)
        found += bool(d)
        facing += bool(d) and d[0]["facing"]
    check("person found in >= 90 % of frames", total and found >= 0.9 * total, f"{found}/{total}")
    check("presenter facing the camera in >= 70 %", total and facing >= 0.7 * total, f"{facing}/{total}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-logs", action="store_true")
    ap.add_argument("--only", default="", help="comma list of cases: approach,away,passer,side,two,self,video")
    args = ap.parse_args()
    sim_port, vis_port, udp = free_port(), free_port(), free_port(socket.SOCK_DGRAM)
    logdir = EXP05 / "logs" / ("vision_check_" + time.strftime("%Y-%m-%d_%H-%M-%S"))
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    procs = [
        subprocess.Popen([PY, "-m", "vision.sim_sensors", "--shop", str(EXP05 / "shops/mockup/shop.json"),
                          "--scenario", "empty", "--port", str(sim_port), "--lidar-port", str(udp),
                          "--joints-url", ""], cwd=HERE, env=env, stdout=open(logdir / "sim.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True),
        subprocess.Popen([PY, "-m", "vision.service", "--camera", f"http://127.0.0.1:{sim_port}/color.jpg",
                          "--lidar-port", str(udp), "--joints-url", f"http://127.0.0.1:{sim_port}/api/status",
                          "--port", str(vis_port), "--host", "127.0.0.1"], cwd=HERE, env=env,
                         stdout=open(logdir / "vision.log", "w"), stderr=subprocess.STDOUT, start_new_session=True),
    ]  # fmt: skip
    sim, vis = f"http://127.0.0.1:{sim_port}", f"http://127.0.0.1:{vis_port}"
    print(f"sensor sim {sim}, vision {vis}, lidar UDP {udp}, logs {logdir}")
    try:
        t0 = time.time()
        while True:
            try:
                st = get(f"{vis}/status")
                if st["camera"]["fps"] > 0 and st["lidar"]["frames"] > 5 and st["joints"]["ok"]:
                    break
            except OSError:
                pass
            if time.time() - t0 > 90 or any(p.poll() is not None for p in procs):
                raise SystemExit(f"processes did not start (see {logdir})")
            time.sleep(1.0)
        time.sleep(3.0)  # background map: the booths
        st = get(f"{vis}/status")
        print(f"ready in {time.time() - t0:.0f} s: camera {st['camera']['fps']} fps, detector "
              f"{st['detector']['delegate']} {st['detector']['ms']} ms, lidar {st['lidar']['points_per_s']} pts/s")
        run = Run(sim, vis)
        cases = {"approach": case_approach, "away": case_facing_away, "passer": case_passer_by, "side": case_side,
                 "two": case_two, "self": case_self}
        only = [c for c in args.only.split(",") if c]
        for k, fn in cases.items():
            if not only or k in only:
                fn(run)
        st = get(f"{vis}/status")
        check("fusion loop < 60 ms", st["fusion_ms"] < 60, f"{st['fusion_ms']} ms; detector {st['detector']['ms']} ms")
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
    if not args.only or "video" in args.only:
        case_video()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
