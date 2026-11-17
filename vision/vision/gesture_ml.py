"""ML-based temporal gesture classifier (still / point / wave) and 3D pointing vector recovery.

Uses scale- and FPS-invariant upper-body kinematic features computed over a sliding window
so gesture recognition works identically at 1 m, 2 m, 3 m, or 4 m and at any camera framerate.
"""

import math
from pathlib import Path

import numpy as np

GESTURE_LABELS = ("still", "point", "wave")
FEATURE_DIM = 30
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gesture_mlp.npz"


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees at vertex b."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-5 or n2 < 1e-5:
        return 90.0
    cos_t = float(np.dot(v1, v2)) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def _count_reversals(vals: np.ndarray, deadband: float = 0.06) -> int:
    if len(vals) < 3:
        return 0
    diffs = np.diff(vals)
    signs = []
    accum = 0.0
    for d in diffs:
        accum += float(d)
        if abs(accum) > deadband:
            s = 1 if accum > 0 else -1
            if not signs or signs[-1] != s:
                signs.append(s)
            accum = 0.0
    return max(0, len(signs) - 1)


def extract_arm_features(history: list[dict]) -> np.ndarray | None:
    """Extract 30 dimensionless temporal-kinematic features from an arm's recent pose history.

    Each item in `history` must be a dict:
      {"t": float, "sh": (2,), "elb": (2,), "wr": (2,), "hip": (2,), "scale": float, "out_sign": float}
    Ordered oldest -> newest. Returns float32 array of shape (30,) or None if history is empty.
    """
    if not history:
        return None

    cur = history[-1]
    scale = max(float(cur["scale"]), 15.0)
    out = float(cur["out_sign"])

    sh = cur["sh"]
    elb = cur["elb"]
    wr = cur["wr"]
    hip = cur["hip"]

    elb_rel_x = (elb[0] - sh[0]) * out / scale
    elb_rel_y = (elb[1] - sh[1]) / scale
    wr_rel_x = (wr[0] - sh[0]) * out / scale
    wr_rel_y = (wr[1] - sh[1]) / scale
    fa_x = (wr[0] - elb[0]) * out / scale
    fa_y = (wr[1] - elb[1]) / scale

    l_se = float(np.linalg.norm(elb - sh))
    l_ew = float(np.linalg.norm(wr - elb))
    l_sw = float(np.linalg.norm(wr - sh))
    reach = l_sw / (l_se + l_ew + 1e-5)
    sw_norm = l_sw / scale

    elb_ang = _angle_deg(sh, elb, wr) / 180.0
    torso_ang = _angle_deg(wr, sh, hip) / 180.0

    wr_above_elb = (elb[1] - wr[1]) / scale
    wr_above_sh = (sh[1] - wr[1]) / scale
    wr_above_hip = (hip[1] - wr[1]) / scale

    # Temporal statistics over window
    n = len(history)
    wr_xs = np.array([(h["wr"][0] - h["sh"][0]) * h["out_sign"] / max(h["scale"], 15.0) for h in history])
    wr_ys = np.array([(h["wr"][1] - h["sh"][1]) / max(h["scale"], 15.0) for h in history])
    fa_xs = np.array([(h["wr"][0] - h["elb"][0]) * h["out_sign"] / max(h["scale"], 15.0) for h in history])
    fa_ys = np.array([(h["wr"][1] - h["elb"][1]) / max(h["scale"], 15.0) for h in history])
    ts = np.array([h["t"] for h in history])

    elb_angs = np.array([_angle_deg(h["sh"], h["elb"], h["wr"]) / 180.0 for h in history])
    reaches = np.array([
        np.linalg.norm(h["wr"] - h["sh"]) / (np.linalg.norm(h["elb"] - h["sh"]) + np.linalg.norm(h["wr"] - h["elb"]) + 1e-5)
        for h in history
    ])
    sw_norms = np.array([np.linalg.norm(h["wr"] - h["sh"]) / max(h["scale"], 15.0) for h in history])

    dt_span = max(float(ts[-1] - ts[0]), 0.1) if n > 1 else 0.1
    if n > 1:
        d_wr = np.hypot(np.diff(wr_xs), np.diff(wr_ys))
        d_fa = np.hypot(np.diff(fa_xs), np.diff(fa_ys))
        path_rate = float(np.sum(d_wr)) / dt_span
        fa_path_rate = float(np.sum(d_fa)) / dt_span
        still_frac = float(np.mean(d_fa < 0.08))
        recent_mask = (ts[-1] - ts[1:]) <= 0.28
        if np.any(recent_mask):
            recent_dt = max(float(ts[-1] - ts[np.argmax(recent_mask)]), 0.05)
            recent_speed = float(np.sum(d_fa[recent_mask])) / recent_dt
        else:
            recent_speed = float(d_fa[-1]) / max(float(ts[-1] - ts[-2]), 0.03)
    else:
        path_rate = 0.0
        fa_path_rate = 0.0
        still_frac = 1.0
        recent_speed = 0.0

    rev_x = float(_count_reversals(fa_xs, deadband=0.06))
    rev_y = float(_count_reversals(fa_ys, deadband=0.06))

    feats = np.array([
        elb_rel_x, elb_rel_y,
        wr_rel_x, wr_rel_y,
        fa_x, fa_y,
        elb_ang, reach, sw_norm, torso_ang,
        wr_above_elb, wr_above_sh, wr_above_hip,
        float(np.mean(wr_xs)), float(np.mean(wr_ys)),
        float(np.std(wr_xs)), float(np.std(wr_ys)),
        float(np.ptp(wr_xs)), float(np.ptp(wr_ys)),
        path_rate,
        float(np.ptp(fa_xs)), float(np.ptp(fa_ys)),
        fa_path_rate,
        rev_x, rev_y,
        float(np.mean(elb_angs)), float(np.min(elb_angs)), float(np.max(elb_angs)),
        float(np.mean(reaches)),
        still_frac,
    ], dtype=np.float32)
    # Note: keep recent_speed folded or check len == 30
    assert feats.shape[0] == FEATURE_DIM
    return feats


class GestureClassifier:
    """Fast NumPy MLP inference for arm gesture classification (still / point / wave)."""

    def __init__(self, model_path: Path | str | None = None):
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.loaded = False
        if path.exists():
            data = np.load(path)
            self.mean = data["mean"].astype(np.float32)
            self.std = np.maximum(data["std"].astype(np.float32), 1e-5)
            self.w1 = data["w1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.w2 = data["w2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)
            self.w3 = data["w3"].astype(np.float32)
            self.b3 = data["b3"].astype(np.float32)
            self.loaded = True

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        """Return probabilities [p_still, p_point, p_wave]."""
        if not self.loaded:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x = (feats - self.mean) / self.std
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        logits = h2 @ self.w3 + self.b3
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        return (exps / np.sum(exps)).astype(np.float32)


def estimate_pointing_vector_3d(sh_2d, elb_2d, wr_2d, sh_w: float, world_lms=None,
                                elb_idx: int = 0, wr_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Recover 3D pointing vector in Robot Frame (X forward, Y left, Z up) + 2D unit ray.

    Uses analytical inverse foreshortening from 2D forearm length vs anthropometric
    forearm-to-shoulder ratio (~0.65 * sh_w), or MediaPipe world landmarks when high-confidence.
    """
    v2d = wr_2d - elb_2d
    n2d = float(np.linalg.norm(v2d))
    if n2d > 1e-3:
        u2d = v2d / n2d
    else:
        u2d = np.array([0.0, 0.0], dtype=float)

    if world_lms is not None and len(world_lms) > max(elb_idx, wr_idx):
        v3_mp = world_lms[wr_idx] - world_lms[elb_idx]
        n3 = float(np.linalg.norm(v3_mp))
        if n3 > 0.08:
            v3_mp = v3_mp / n3
            v_rob = np.array([v3_mp[2], -v3_mp[0], -v3_mp[1]], dtype=float)
            v_rob /= (np.linalg.norm(v_rob) + 1e-6)
            return v_rob, u2d

    # Analytical inverse foreshortening from 2D image geometry:
    # Expected un-foreshortened forearm length ~ 0.65 * shoulder_width
    expected_fa = max(0.65 * sh_w, 12.0)
    cos_depth = min(1.0, n2d / expected_fa)
    # Person is facing camera, so pointing foreshortening points toward camera (-X in robot frame)
    vx = -math.sqrt(max(0.0, 1.0 - cos_depth * cos_depth))
    vy = -float(u2d[0]) * cos_depth
    vz = -float(u2d[1]) * cos_depth
    v_rob = np.array([vx, vy, vz], dtype=float)
    n_rob = float(np.linalg.norm(v_rob))
    if n_rob < 1e-5:
        v_rob = np.array([-1.0, 0.0, 0.0], dtype=float)
    else:
        v_rob /= n_rob
    return v_rob, u2d
