"""Train temporal GestureNet (still/point/wave) + 3D PointingDirNet (3D unit vector regressor).

Generates biomechanically realistic 3D upper-body motion sequences projected through the Unitree G1
camera pinhole model across distances (0.9 - 4.5 m), body yaws (-50 to +50 deg), and FPS (10 - 30 Hz).
Trains both models on GPU/CPU and saves compact NumPy weights:
  - vision/models/gesture_mlp.npz
  - vision/models/pointing_dir_mlp.npz
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

VISION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISION_DIR))

from vision.gesture_ml import (
    DEFAULT_DIR_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DIR_FEATURE_DIM,
    FEATURE_DIM,
    GESTURE_LABELS,
    extract_arm_features,
    extract_pointing_dir_features,
)


def _project_3d_to_2d(pts3d: np.ndarray, dist_m: float, yaw_rad: float, hfov_deg: float = 69.0, w: int = 640, h: int = 480) -> np.ndarray:
    """Rotate local skeleton points (x=left, y=up, z=forward_to_cam) by yaw and project to 2D pixels."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    xr = pts3d[:, 0] * c + pts3d[:, 2] * s
    yr = pts3d[:, 1]
    zr = -pts3d[:, 0] * s + pts3d[:, 2] * c

    xc = xr
    yc = -yr
    zc = dist_m - zr

    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u = w / 2.0 + f * (xc / np.maximum(zc, 0.3))
    v = h / 2.0 + f * (yc / np.maximum(zc, 0.3))
    return np.stack([u, v], axis=-1)


def _limb_from_angles(sh3: np.ndarray, upper_dir: np.ndarray, fore_dir: np.ndarray, l_ua: float, l_fa: float):
    u1 = upper_dir / (np.linalg.norm(upper_dir) + 1e-6)
    u2 = fore_dir / (np.linalg.norm(fore_dir) + 1e-6)
    elb3 = sh3 + l_ua * u1
    wr3 = elb3 + l_fa * u2
    return elb3, wr3


def simulate_sequence(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    dist_m = rng.uniform(0.45, 4.5)
    yaw_rad = math.radians(rng.uniform(-50.0, 50.0))
    fps = rng.uniform(12.0, 30.0)
    n_frames = int(rng.uniform(7, 15))
    dt = 1.0 / fps

    sh_half = rng.uniform(0.18, 0.23)
    torso_m = rng.uniform(0.45, 0.54)
    l_ua = rng.uniform(0.27, 0.32)
    l_fa = rng.uniform(0.24, 0.28)
    out_sign = 1.0 if rng.random() > 0.5 else -1.0
    noise_px = rng.uniform(1.0, 3.2)

    sh_l3 = np.array([sh_half * out_sign, 0.25, 0.0])
    sh_r3 = np.array([-sh_half * out_sign, 0.25, 0.0])
    hip3 = np.array([sh_half * 0.7 * out_sign, 0.25 - torso_m, 0.0])

    phase0 = rng.uniform(0, 2 * math.pi)
    freq = rng.uniform(1.3, 4.5)

    if kind == "point_side":
        az = rng.uniform(math.radians(25), math.radians(95)) * out_sign
        el = rng.uniform(math.radians(-40), math.radians(40))
        dir_p = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        bend = rng.uniform(0.0, 0.35)
    elif kind == "point_forward":
        az = rng.uniform(math.radians(-30), math.radians(30))
        el = rng.uniform(math.radians(-30), math.radians(35))
        dir_p = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        bend = rng.uniform(0.0, 0.30)
    elif kind == "point_casual":
        # Seated or relaxed pointing: upper arm relaxed/angled down, forearm extended steadily toward target
        az = rng.uniform(math.radians(-40), math.radians(75)) * out_sign
        el = rng.uniform(math.radians(-15), math.radians(35))
        dir_p = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        ua_casual = np.array([0.35 * out_sign * math.sin(abs(az)), rng.uniform(-0.75, -0.25), rng.uniform(0.30, 0.70)])
    elif kind == "point_transition":
        az = rng.uniform(math.radians(15), math.radians(85)) * out_sign
        el = rng.uniform(math.radians(-25), math.radians(35))
        dir_p = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
    elif kind == "wave_high":
        amp = rng.uniform(0.09, 0.26)
        base_x = rng.uniform(0.16, 0.42) * out_sign
        base_y = rng.uniform(0.02, 0.38)
    elif kind == "wave_chest":
        amp = rng.uniform(0.11, 0.28)
        base_x = rng.uniform(0.12, 0.38) * out_sign
        base_y = rng.uniform(-0.18, 0.08)

    history = []
    for k in range(n_frames):
        t = k * dt
        frac = k / max(n_frames - 1, 1)

        if kind == "still_idle":
            u1 = np.array([0.08 * out_sign, -1.0, 0.05 * math.sin(2 * math.pi * 0.8 * t + phase0)])
            u2 = np.array([0.06 * out_sign, -0.95, 0.15 + 0.08 * math.sin(2 * math.pi * 0.8 * t + phase0)])
            label = 0
        elif kind == "still_crossed":
            u1 = np.array([0.05 * out_sign, -0.95, 0.25])
            u2 = np.array([-0.9 * out_sign, 0.05, 0.35])
            label = 0
        elif kind == "still_hip":
            u1 = np.array([0.55 * out_sign, -0.75, -0.1])
            u2 = np.array([-0.65 * out_sign, -0.55, 0.15])
            label = 0
        elif kind == "still_phone":
            u1 = np.array([0.25 * out_sign, -0.85, 0.35])
            u2 = np.array([-0.25 * out_sign, 0.90, 0.20 + 0.03 * math.sin(4 * t)])
            label = 0
        elif kind == "still_talking":
            u1 = np.array([0.12 * out_sign, -0.95, 0.15])
            u2 = np.array([
                (-0.10 + 0.18 * math.sin(2 * math.pi * 1.6 * t + phase0)) * out_sign,
                -0.35 + 0.16 * math.cos(2 * math.pi * 1.6 * t + phase0),
                0.30,
            ])
            label = 0
        elif kind in ("point_side", "point_forward"):
            jitter_dir = dir_p + np.array([0.04 * math.sin(1.5 * t + phase0), 0.03 * math.cos(1.2 * t + phase0), 0.0])
            u1 = jitter_dir
            u2 = jitter_dir + np.array([0.0, bend * 0.5, 0.0])
            label = 1
        elif kind == "point_casual":
            u1 = ua_casual
            u2 = dir_p + np.array([0.03 * math.sin(1.2 * t + phase0), 0.02 * math.cos(1.0 * t + phase0), 0.0])
            label = 1
        elif kind == "point_transition":
            alpha = min(1.0, frac * 1.6)
            u_rest = np.array([0.1 * out_sign, -1.0, 0.1])
            u_cur = (1.0 - alpha) * u_rest + alpha * dir_p
            u1 = u_cur
            u2 = u_cur
            label = 1 if alpha > 0.72 else 0
        elif kind in ("wave_high", "wave_chest"):
            u1 = np.array([base_x, base_y, 0.20])
            osc = math.sin(2 * math.pi * freq * t + phase0)
            u2 = np.array([
                (0.15 + amp * 3.0 * osc) * out_sign,
                0.95 + 0.15 * math.cos(2 * math.pi * freq * t + phase0),
                0.20,
            ])
            label = 2
        else:
            raise ValueError(kind)

        elb3, wr3 = _limb_from_angles(sh_l3, u1, u2, l_ua, l_fa)
        pts3 = np.stack([sh_l3, sh_r3, elb3, wr3, hip3], axis=0)
        pts2 = _project_3d_to_2d(pts3, dist_m, yaw_rad) + rng.normal(0.0, noise_px, size=(5, 2))

        sh2, sh_other2, elb2, wr2, hip2 = pts2
        sh_w_px = float(np.linalg.norm(sh2 - sh_other2))
        torso_h_px = float(np.linalg.norm(sh2 - hip2))
        scale = max(sh_w_px, 0.7 * torso_h_px, 15.0)
        inferred_out = 1.0 if sh2[0] >= sh_other2[0] else -1.0

        history.append({
            "t": t, "sh": sh2, "elb": elb2, "wr": wr2, "hip": hip2,
            "scale": scale, "out_sign": inferred_out,
        })

    return extract_arm_features(history), label


def simulate_pointing_3d_sample(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate one 2D feature vector + ground-truth 3D pointing vector in Robot Frame."""
    dist_m = rng.uniform(0.9, 4.5)
    yaw_rad = math.radians(rng.uniform(-50.0, 50.0))
    sh_half = rng.uniform(0.18, 0.23)
    torso_m = rng.uniform(0.45, 0.54)
    l_ua = rng.uniform(0.27, 0.32)
    l_fa = rng.uniform(0.24, 0.28)
    out_sign = 1.0 if rng.random() > 0.5 else -1.0
    noise_px = rng.uniform(0.8, 2.8)

    sh_l3 = np.array([sh_half * out_sign, 0.25, 0.0])
    sh_r3 = np.array([-sh_half * out_sign, 0.25, 0.0])
    hip3 = np.array([sh_half * 0.7 * out_sign, 0.25 - torso_m, 0.0])

    # Sample pointing direction in world/robot frame around the person
    # Robot frame: X away from robot, Y left of robot, Z up.
    # Person faces roughly toward robot (-X_rob), pointing azimuth in [-105, +105] deg around -X_rob
    az = rng.uniform(math.radians(-105), math.radians(105))
    el = rng.uniform(math.radians(-45), math.radians(50))
    gt_rob = np.array([
        -math.cos(az) * math.cos(el),
        -math.sin(az) * math.cos(el),
        math.sin(el),
    ], dtype=np.float32)
    gt_rob /= np.linalg.norm(gt_rob)

    # Convert gt_rob back into local person skeleton frame before body yaw rotation:
    # In _project_3d_to_2d:
    #   robot_X = -zr_cam = -(-x*sin + z*cos)  => cam_x = xr = x*cos + z*sin
    #   robot_Y = -cam_x = -xr
    #   robot_Z = yr = y
    cam_x = -gt_rob[1]
    cam_z_to_cam = -gt_rob[0]  # zr in _project_3d_to_2d
    cam_y = gt_rob[2]

    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # Inverse rotation around Y: x_local = cam_x*c - cam_z*s, z_local = cam_x*s + cam_z*c
    dir_local = np.array([cam_x * c - cam_z_to_cam * s, cam_y, cam_x * s + cam_z_to_cam * c], dtype=float)
    dir_local /= np.linalg.norm(dir_local)

    # Slight natural elbow bend around pointing direction
    elb3, wr3 = _limb_from_angles(sh_l3, dir_local + rng.normal(0, 0.05, 3), dir_local, l_ua, l_fa)
    pts3 = np.stack([sh_l3, sh_r3, elb3, wr3, hip3], axis=0)
    pts2 = _project_3d_to_2d(pts3, dist_m, yaw_rad) + rng.normal(0.0, noise_px, size=(5, 2))

    sh2, sh_other2, elb2, wr2, hip2 = pts2
    torso_h_px = float(np.linalg.norm(sh2 - hip2))
    feats = extract_pointing_dir_features(sh2, sh_other2, elb2, wr2, hip2, torso_h_px)
    return feats, gt_rob


class GestureNet(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(96, 48), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class PointingDirNet(nn.Module):
    def __init__(self, in_dim: int = DIR_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(96, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def build_gesture_dataset(n_total: int = 36000, seed: int = 42):
    rng = np.random.default_rng(seed)
    kinds_weights = [
        ("still_idle", 0.16), ("still_crossed", 0.08), ("still_hip", 0.07),
        ("still_phone", 0.08), ("still_talking", 0.09),
        ("point_side", 0.12), ("point_forward", 0.09), ("point_casual", 0.08), ("point_transition", 0.06),
        ("wave_high", 0.10), ("wave_chest", 0.07),
    ]
    kinds = [k for k, _ in kinds_weights]
    probs = np.array([w for _, w in kinds_weights])
    probs /= probs.sum()
    X = np.zeros((n_total, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.int64)
    for i, k in enumerate(rng.choice(kinds, size=n_total, p=probs)):
        X[i], y[i] = simulate_sequence(k, rng)
    return X, y


def build_pointing_dir_dataset(n_total: int = 30000, seed: int = 123):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_total, DIR_FEATURE_DIM), dtype=np.float32)
    y = np.zeros((n_total, 3), dtype=np.float32)
    for i in range(n_total):
        X[i], y[i] = simulate_pointing_3d_sample(rng)
    return X, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train GestureNet
    print("[gesture] Training GestureNet (36,000 train / 6,000 val)...", flush=True)
    X_tr, y_tr = build_gesture_dataset(36000, seed=42)
    X_va, y_va = build_gesture_dataset(6000, seed=999)
    mean, std = X_tr.mean(axis=0), np.maximum(X_tr.std(axis=0), 1e-5)

    loader = DataLoader(TensorDataset(torch.from_numpy((X_tr - mean) / std), torch.from_numpy(y_tr)), batch_size=256, shuffle=True)
    model = GestureNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    crit = nn.CrossEntropyLoss()

    x_va_t = torch.from_numpy((X_va - mean) / std).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)
    best_acc, best_state = 0.0, None
    for ep in range(1, 26):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(x_va_t).argmax(dim=1) == y_va_t).float().mean().item())
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    print(f"[gesture] Best val_acc = {best_acc:.4f}", flush=True)

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DEFAULT_MODEL_PATH, mean=mean, std=std,
        w1=best_state["net.0.weight"].T, b1=best_state["net.0.bias"],
        w2=best_state["net.3.weight"].T, b2=best_state["net.3.bias"],
        w3=best_state["net.6.weight"].T, b3=best_state["net.6.bias"],
    )

    # 2. Train PointingDirNet (3D direction regressor)
    print("\n[pointing-3d] Training PointingDirNet (30,000 train / 5,000 val)...", flush=True)
    Xd_tr, yd_tr = build_pointing_dir_dataset(30000, seed=123)
    Xd_va, yd_va = build_pointing_dir_dataset(5000, seed=777)
    d_mean, d_std = Xd_tr.mean(axis=0), np.maximum(Xd_tr.std(axis=0), 1e-5)

    d_loader = DataLoader(TensorDataset(torch.from_numpy((Xd_tr - d_mean) / d_std), torch.from_numpy(yd_tr)), batch_size=256, shuffle=True)
    d_model = PointingDirNet().to(device)
    d_opt = torch.optim.AdamW(d_model.parameters(), lr=2e-3, weight_decay=1e-4)
    d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=30)

    xd_va_t = torch.from_numpy((Xd_va - d_mean) / d_std).to(device)
    yd_va_t = torch.from_numpy(yd_va).to(device)
    best_err, best_d_state = 180.0, None
    for ep in range(1, 31):
        d_model.train()
        for xb, yb in d_loader:
            xb, yb = xb.to(device), yb.to(device)
            d_opt.zero_grad()
            pred = d_model(xb)
            loss = (1.0 - (pred * yb).sum(dim=-1)).mean()
            loss.backward()
            d_opt.step()
        d_sched.step()
        d_model.eval()
        with torch.no_grad():
            cos = torch.clamp((d_model(xd_va_t) * yd_va_t).sum(dim=-1), -1.0, 1.0)
            ang = torch.rad2deg(torch.acos(cos))
            mean_ang = float(ang.mean().item())
            med_ang = float(ang.median().item())
        if mean_ang < best_err:
            best_err = mean_ang
            best_med = med_ang
            best_d_state = {k: v.detach().cpu().numpy() for k, v in d_model.state_dict().items()}
        if ep % 10 == 0 or ep == 30:
            print(f"  Epoch {ep:2d}/30 | val_mean_err={mean_ang:.2f} deg | val_median_err={med_ang:.2f} deg", flush=True)

    np.savez(
        DEFAULT_DIR_MODEL_PATH, mean=d_mean, std=d_std,
        w1=best_d_state["net.0.weight"].T, b1=best_d_state["net.0.bias"],
        w2=best_d_state["net.3.weight"].T, b2=best_d_state["net.3.bias"],
        w3=best_d_state["net.5.weight"].T, b3=best_d_state["net.5.bias"],
    )
    print(f"[pointing-3d] Saved -> {DEFAULT_DIR_MODEL_PATH} ({DEFAULT_DIR_MODEL_PATH.stat().st_size / 1024:.1f} KB), best mean={best_err:.2f} deg, median={best_med:.2f} deg")


if __name__ == "__main__":
    main()
