"""Check of the operator's video upload (page card "Motion learned from video", deploy/video_jobs.py).

Starts its own sim + server (private DDS domain and port) and, with the example videos:
  1. wrong uploads are refused (not a video, empty, bad time range) and the server stays usable after a refusal
  2. wai video -> processed in the background: measured, solved for the current hands, checks + comparison sheet
  3. "Keep the old one" leaves the installed wai untouched
  4. the same video again -> "Use this" installs it (old one in clips/backup/), then Undo puts the old one back
  5. keep-alive video -> measured style; "Use this", then Undo (back to the defaults if there was no style before)
The clips the robot uses are snapshotted first and restored at the end (the check never leaves a change behind).
Usage: .venv/bin/python tools/video_check.py      (~2-4 min, CPU: MediaPipe)
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

G1 = Path(__file__).resolve().parents[1]
EXP05 = G1.parent
sys.path.insert(0, str(G1 / "deploy"))
from robot_model import WAI_CLIP  # noqa: E402

DOMAIN, PORT = 54, 8102
URL = f"http://127.0.0.1:{PORT}"
WAI_VIDEO, WAI_RANGE = EXP05 / "videos/examples/x07hnDvNcxM.mp4", (26, 40)
IDLE_VIDEO, IDLE_RANGE = EXP05 / "videos/examples/4ffl4T5FYcY.mp4", (20, 60)
IDLE_CLIP = EXP05 / "clips/idle.json"


def get(path):
    with urllib.request.urlopen(URL + path, timeout=10) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def upload(kind, name, data, start=0, end=0):
    req = urllib.request.Request(f"{URL}/api/video/upload?kind={kind}&name={name}&start={start}&end={end}", data,
                                 {"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def wait_job(timeout=400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = get("/api/video/job")["job"]
        if j["status"] != "running":
            return j
        time.sleep(2)
    return get("/api/video/job")["job"]


def main():
    snap = {p: (p.read_bytes() if p.exists() else None) for p in (WAI_CLIP, IDLE_CLIP)}
    dirs = [EXP05 / "clips/backup", EXP05 / "videos/uploads", EXP05 / "clips/source", EXP05 / "clips/candidates"]
    before = {d_: set(d_.glob("*")) if d_.exists() else set() for d_ in dirs}  # only files the check adds are removed
    procs = [subprocess.Popen([sys.executable, "deploy/sim_robot.py", "--headless", "--domain", str(DOMAIN)], cwd=G1,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(2)
    procs.append(subprocess.Popen([sys.executable, "deploy/server.py", "--audio-settings", "/dev/null", "--domain", str(DOMAIN), "--port", str(PORT),
                                   "--host", "127.0.0.1"], cwd=G1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    results = []

    def check(name, ok, info=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

    try:
        for _ in range(60):
            try:
                get("/api/video/job")
                break
            except OSError:
                time.sleep(0.5)
        r1 = upload("wai", "notes.txt", b"hello")
        r2 = upload("wai", "clip.mp4", b"")
        r3 = upload("wai", "clip.mp4", b"x" * 1000, start=10, end=5)
        check("bad uploads refused", not (r1["ok"] or r2["ok"] or r3["ok"]),
              f"{r1['reason']} | {r2['reason']} | {r3['reason']}")
        check("server usable after a refusal", get("/api/video/job")["ok"])

        data = WAI_VIDEO.read_bytes()
        r = upload("wai", WAI_VIDEO.name, data, *WAI_RANGE)
        check("wai video accepted", r["ok"] and r["job"]["status"] == "running", r.get("reason", ""))
        busy = upload("idle", IDLE_VIDEO.name, b"x" * 1000)
        check("second upload while busy refused", not busy["ok"], busy.get("reason", ""))
        t0 = time.time()
        j = wait_job()
        res = j.get("result") or {}
        check("wai processed", j["status"] == "ready" and res.get("ok"),
              f"{time.time() - t0:.0f} s, {j['hand_model']} hands, gap {res.get('checks', {}).get('palm_gap_min_m')}, "
              f"symmetry {res.get('checks', {}).get('mirror_err_deg')} deg {j.get('reason', '')}")
        check("comparison sheet served", res.get("compare") and
              urllib.request.urlopen(f"{URL}/clips/source/{res['compare']}", timeout=10).status == 200)
        wai_before = WAI_CLIP.read_bytes() if WAI_CLIP.exists() else None
        d = post("/api/video/decide", {"use": False})
        check("keep the old one: installed wai unchanged", d["ok"] and d["job"]["status"] == "discarded"
              and (WAI_CLIP.read_bytes() if WAI_CLIP.exists() else None) == wai_before)

        upload("wai", WAI_VIDEO.name, data, *WAI_RANGE)
        j = wait_job()
        d = post("/api/video/decide", {"use": True})
        new = json.loads(WAI_CLIP.read_text()) if WAI_CLIP.exists() else {}
        check("use this: wai installed", d["ok"] and d["job"]["status"] == "installed" and new.get("ok")
              and "videos/uploads/" in new.get("source", ""), new.get("source", ""))
        clips = get("/api/clips")
        check("page shows the new wai", (clips.get("wai") or {}).get("source") == new.get("source"))
        u = post("/api/video/restore", {"kind": "wai"})
        check("undo: previous wai back", u["ok"] and (WAI_CLIP.read_bytes() if WAI_CLIP.exists() else None) == wai_before,
              u.get("restored", u.get("reason", "")))

        idle_before = IDLE_CLIP.read_bytes() if IDLE_CLIP.exists() else None
        upload("idle", IDLE_VIDEO.name, IDLE_VIDEO.read_bytes(), *IDLE_RANGE)
        j = wait_job()
        res = j.get("result") or {}
        check("keep-alive video processed", j["status"] == "ready",
              f"amp x{res.get('idle_amp_scale')}, speed x{res.get('idle_speed_scale')}, beat every "
              f"{res.get('beat_every')} {j.get('reason', '')}")
        d = post("/api/video/decide", {"use": True})
        check("keep-alive style installed", d["ok"] and IDLE_CLIP.exists()
              and "videos/uploads/" in json.loads(IDLE_CLIP.read_text())["source"])
        u = post("/api/video/restore", {"kind": "idle"})
        check("undo: keep-alive style as before", u["ok"] and (IDLE_CLIP.read_bytes() if IDLE_CLIP.exists() else None)
              == idle_before, "defaults" if idle_before is None else "")
    finally:
        for p in reversed(procs):
            p.terminate()
            p.wait(timeout=10)
        for p, b in snap.items():  # leave the robot's clips exactly as they were
            if b is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(b)
        for d_, before_set in before.items():
            if d_.exists():
                for f in set(d_.glob("*")) - before_set:
                    f.unlink()
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
