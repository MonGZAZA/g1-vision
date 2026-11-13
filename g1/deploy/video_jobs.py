"""Operator video uploads -> motion learned from video (the page's "Motion learned from video" card).

Two kinds, each the same tools as on the command line (README "Learn from video"), run in the background:
  wai   ai/tools/video_to_wai.py VIDEO  ->  g1/tools/make_wai_clip.py <measurements> --out <candidate>
        (solved on the hand model of EXP05_HANDS; the comparison sheet video vs robot comes with it)
  idle  ai/tools/video_to_idle.py VIDEO  (keep-alive / talking style)
The result is a CANDIDATE: nothing changes on the robot until the operator presses "Use this". Then the current
clip is copied to clips/backup/ and the candidate takes its place (behaviour.py and gestures.py re-read the files,
so the next action uses it; no restart). "Keep the old one" discards the candidate. One job at a time.

Files: videos/uploads/<time>_<name>.<ext> (the upload), clips/source/ (measurements, sheets), clips/candidates/.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from robot_model import EXP05, HAND_MODEL, WAI_CLIP

UPLOADS = EXP05 / "videos/uploads"
SOURCE = EXP05 / "clips/source"
CANDIDATES = EXP05 / "clips/candidates"
BACKUP = EXP05 / "clips/backup"
IDLE_CLIP = EXP05 / "clips/idle.json"
AI_PY = EXP05 / "ai/.venv/bin/python"
G1_PY = EXP05 / "g1/.venv/bin/python"
MAX_BYTES = 500 * 1024 * 1024
EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")
TIMEOUT_S = 20 * 60


class VideoError(Exception):
    pass


class VideoJobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.job = None  # the latest job (dict), see status()

    # ---------------------------------------------------------------- upload
    def start(self, kind, filename, stream, length, start_s=0.0, end_s=0.0):
        """Save the uploaded video (read from stream in chunks) and start processing it."""
        if kind not in ("wai", "idle"):
            raise VideoError("kind must be wai or idle")
        ext = Path(filename or "").suffix.lower()
        if ext not in EXTS:
            raise VideoError(f"video file needed ({', '.join(EXTS)})")
        if not 0 < length <= MAX_BYTES:
            raise VideoError("empty file" if length <= 0 else f"video larger than {MAX_BYTES // 2 ** 20} MB")
        if start_s < 0 or end_s < 0 or (end_s and end_s <= start_s):
            raise VideoError("end must be after start (0 = to the end)")
        with self.lock:
            if self.job and self.job["status"] == "running":
                raise VideoError("another video is still being processed")
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem)[:40].strip("_") or "video"
            jid = time.strftime("%Y%m%d-%H%M%S")
            UPLOADS.mkdir(parents=True, exist_ok=True)
            path = UPLOADS / f"{jid}_{stem}{ext}"
            self.job = {"id": jid, "kind": kind, "status": "running", "step": "uploading", "video": path.name,
                        "hand_model": HAND_MODEL, "log": [], "result": None, "reason": "", "started": time.time()}
        try:
            with open(path, "wb") as f:
                left = length
                while left > 0:
                    chunk = stream.read(min(1 << 20, left))
                    if not chunk:
                        raise VideoError("upload interrupted")
                    f.write(chunk)
                    left -= len(chunk)
        except Exception as e:  # noqa: BLE001
            path.unlink(missing_ok=True)
            self._fail(str(e))
            raise VideoError(str(e)) from e
        threading.Thread(target=self._run, args=(path, kind, start_s, end_s), daemon=True).start()
        return self.status()

    # ---------------------------------------------------------------- processing
    def _step(self, name, cmd, cwd):
        """Run one tool; its output goes to the job log. -> (return code, output lines)."""
        with self.lock:
            self.job["step"] = name
        env = dict(os.environ, PYTHONUNBUFFERED="1", MUJOCO_GL="egl")
        p = subprocess.Popen(["nice", "-n", "10", *map(str, cmd)], cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        out, t0 = [], time.time()
        for line in p.stdout:
            line = line.rstrip()
            if not line or "multicast" in line or "Warning" in line or line.startswith(("  warnings.warn", "W0000", "I0000")):
                continue
            out.append(line)
            with self.lock:
                self.job["log"] = (self.job["log"] + [line])[-40:]
            if time.time() - t0 > TIMEOUT_S:
                p.kill()
                out.append("timed out")
                break
        return p.wait(), out

    def _run(self, path, kind, start_s, end_s):
        try:
            rng = (["--start", start_s] if start_s else []) + (["--end", end_s] if end_s else [])
            if kind == "wai":
                rc, out = self._step("measuring the hands (MediaPipe)", [AI_PY, "tools/video_to_wai.py", path, *rng],
                                     EXP05 / "ai")
                meas = next((ln for ln in reversed(out) if ln.endswith("_wai.json")), None)
                if rc != 0 or not meas:
                    return self._fail(out[-1] if out else "measurement failed")
                cand = CANDIDATES / f"wai_{HAND_MODEL}_{self.job['id']}.json"
                rc, out = self._step(f"solving the robot's wai ({HAND_MODEL} hands)",
                                     [G1_PY, "tools/make_wai_clip.py", meas, "--out", cand], EXP05 / "g1")
                if rc != 0 or not cand.exists():
                    return self._fail(out[-1] if out else "wai solving failed")
                clip = json.loads(cand.read_text())
                stem = Path(meas).stem
                result = {"candidate": str(cand.relative_to(EXP05)), "ok": bool(clip["ok"]), "checks": clip["checks"],
                          "params": clip["params"], "segment_s": clip.get("segment_s"),
                          "compare": f"{stem}_compare_{HAND_MODEL}.jpg", "sheet": f"{stem}_sheet.jpg"}
                if not (SOURCE / result["compare"]).exists():
                    result["compare"] = None
            else:
                rc, out = self._step("measuring the keep-alive style (MediaPipe)",
                                     [AI_PY, "tools/video_to_idle.py", path, *rng], EXP05 / "ai")
                src = SOURCE / f"{path.stem}_idle.json"
                if rc != 0 or not src.exists():
                    return self._fail(out[-1] if out else "measurement failed")
                CANDIDATES.mkdir(parents=True, exist_ok=True)
                cand = CANDIDATES / f"idle_{self.job['id']}.json"
                shutil.copy(src, cand)
                d = json.loads(cand.read_text())
                result = {"candidate": str(cand.relative_to(EXP05)), "ok": True,
                          "measured": d["measured"], "segment_s": d.get("segment_s"),
                          "idle_amp_scale": d["idle_amp_scale"], "idle_speed_scale": d["idle_speed_scale"],
                          "beat_every": d["beat_every"]}
            with self.lock:
                self.job.update(status="ready", step="ready: use it or keep the old one", result=result)
        except Exception as e:  # noqa: BLE001
            self._fail(f"{type(e).__name__}: {e}")

    def _fail(self, reason):
        with self.lock:
            if self.job:
                self.job.update(status="failed", step="failed", reason=str(reason)[-300:])

    # ---------------------------------------------------------------- decision
    def decide(self, use):
        """use=True: install the candidate (the current clip goes to clips/backup/); False: discard it."""
        with self.lock:
            j = self.job
            if not j or j["status"] != "ready":
                raise VideoError("no processed video waiting for a decision")
            cand = EXP05 / j["result"]["candidate"]
            if not use:
                cand.unlink(missing_ok=True)
                j.update(status="discarded", step="discarded: the old one stays")
                return self.status_locked()
            if not j["result"]["ok"]:
                raise VideoError("the checks failed: this wai is not safe to use")
            if j["kind"] == "wai" and j["hand_model"] != HAND_MODEL:
                raise VideoError("solved for other hands")
            dst = WAI_CLIP if j["kind"] == "wai" else IDLE_CLIP
            BACKUP.mkdir(parents=True, exist_ok=True)
            bk = BACKUP / f"{dst.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
            if dst.exists():
                shutil.copy(dst, bk.with_suffix(".json"))
            else:  # nothing installed before (procedural wai / default style): undo removes the clip again
                bk.with_suffix(".none").write_text("")
            d = json.loads(cand.read_text())
            d["enabled"] = True
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(dst)  # atomic: an action starting now reads either the old or the new clip
            cand.unlink(missing_ok=True)
            j.update(status="installed", step=f"in use: {dst.relative_to(EXP05)}")
            return self.status_locked()

    def restore_previous(self, kind):
        """Put back the newest backup of the wai / keep-alive clip (undo the last "Use this")."""
        dst = WAI_CLIP if kind == "wai" else IDLE_CLIP
        backups = sorted(b for b in BACKUP.glob(f"{dst.stem}_*") if b.suffix in (".json", ".none")) if BACKUP.exists() else []
        if not backups:
            raise VideoError("no earlier version saved")
        if backups[-1].suffix == ".none":
            dst.unlink(missing_ok=True)  # back to the procedural wai / default keep-alive style
        else:
            tmp = dst.with_suffix(".tmp")
            shutil.copy(backups[-1], tmp)
            tmp.replace(dst)
        backups[-1].unlink()
        return {"ok": True, "restored": backups[-1].name}

    # ---------------------------------------------------------------- status
    def status_locked(self):
        j = self.job
        return {"ok": True, "job": None if j is None else {**j, "elapsed_s": round(time.time() - j["started"])}}

    def status(self):
        with self.lock:
            return self.status_locked()
