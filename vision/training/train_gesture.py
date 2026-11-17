"""Train a temporal ML gesture classifier (still / point / wave) using 3D->2D kinematic sequences.

Generates biomechanically realistic 3D upper-body motion sequences projected through the Unitree G1
camera pinhole model across distances (0.9 - 4.5 m), body yaws (-50 to +50 deg), and FPS (10 - 30 Hz),
including hard negatives (arms crossed, phone/head, walking swing, moving-into-point transitions,
and foreshortened forward pointing). Trains a PyTorch MLP on GPU/CPU and saves weights to
vision/models/gesture_mlp.npz.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

VISION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISION_DIR))

from vision.gesture_ml import DEFAULT_MODEL_PATH, FEATURE_DIM, GESTURE_LABELS, extract_arm_features


def _project_3d_to_2d(pts3d: np.ndarray, dist_m: float, yaw_rad: float, hfov_deg: float = 69.0, w: int = 640, h: int = 480) -> np.ndarray:
    """Rotate local skeleton points (x=left, y=up, z=forward_to_cam) by yaw and project to 2D pixels."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # Rotate around Y (vertical)
    xr = pts3d[:, 0] * c + pts3d[:, 2] * s
    yr = pts3d[:, 1]
    zr = -pts3d[:, 0] * s + pts3d[:, 2] * c

    # Place person at (0, 0, dist_m) in camera coords (Z away from cam)
    xc = xr
    yc = -yr  # image Y down
    zc = dist_m - zr  # pts3d z>0 is toward camera

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
    """Simulate a single 0.8s arm motion sequence and return (features[30], label_idx)."""
    dist_m = rng.uniform(0.9, 4.5)
    yaw_rad = math.radians(rng.uniform(-50.0, 50.0))
    fps = rng.uniform(12.0, 30.0)
    n_frames = int(rng.uniform(7, 15))
    dt = 1.0 / fps

    sh_half = rng.uniform(0.18, 0.23)
    torso_m = rng.uniform(0.45, 0.54)
    l_ua = rng.uniform(0.27, 0.32)
    l_fa = rng.uniform(0.24, 0.28)
    out_sign = 1.0 if rng.random() > 0.5 else -1.0  # +1 = left arm, -1 = right arm
    noise_px = rng.uniform(1.0, 3.2)

    sh_l3 = np.array([sh_half * out_sign, 0.25, 0.0])
    sh_r3 = np.array([-sh_half * out_sign, 0.25, 0.0])
    hip3 = np.array([sh_half * 0.7 * out_sign, 0.25 - torso_m, 0.0])

    # Random base parameters per sequence kind
    phase0 = rng.uniform(0, 2 * math.pi)
    freq = rng.uniform(1.8, 4.2)  # Hz for wave/walk

    if kind == "point_side":
        az = rng.uniform(math.radians(25), math.radians(85)) * out_sign
        el = rng.uniform(math.radians(-25), math.radians(35))
        dir_p = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        bend = rng.uniform(0.0, 0.28)
    elif kind == "point_forward":
        az = rng.uniform(math.radians(-25), math.radians(25))
        el = rng.uniform(math.radians(-20), math.radians(30))
        dir_p = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        bend = rng.uniform(0.0, 0.22)
    elif kind == "point_transition":
        az = rng.uniform(math.radians(15), math.radians(80)) * out_sign
        el = rng.uniform(math.radians(-15), math.radians(30))
        dir_p = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
    elif kind == "wave_high":
        amp = rng.uniform(0.10, 0.24)
        base_x = rng.uniform(0.18, 0.40) * out_sign
        base_y = rng.uniform(0.05, 0.35)
    elif kind == "wave_chest":
        amp = rng.uniform(0.12, 0.26)
        base_x = rng.uniform(0.14, 0.36) * out_sign
        base_y = rng.uniform(-0.15, 0.08)

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
            u1 = np.array([0.15 * out_sign, -0.90, 0.20])
            u2 = np.array([
                (0.15 + 0.12 * math.sin(2 * math.pi * 1.5 * t + phase0)) * out_sign,
                -0.15 + 0.12 * math.cos(2 * math.pi * 1.5 * t + phase0),
                0.75,
            ])
            label = 0
        elif kind in ("point_side", "point_forward"):
            # Slight natural hand drift/adjustment while pointing
            jitter_dir = dir_p + np.array([
                0.04 * math.sin(1.5 * t + phase0),
                0.03 * math.cos(1.2 * t + phase0),
                0.0,
            ])
            u1 = jitter_dir
            u2 = jitter_dir + np.array([0.0, bend * 0.5, 0.0])
            label = 1
        elif kind == "point_transition":
            # Smoothly lift arm from resting down into pointing pose and hold in second half
            alpha = min(1.0, frac * 1.6)
            u_rest = np.array([0.1 * out_sign, -1.0, 0.1])
            u_cur = (1.0 - alpha) * u_rest + alpha * dir_p
            u1 = u_cur
            u2 = u_cur
            label = 1 if alpha > 0.75 else 0
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
            "t": t,
            "sh": sh2,
            "elb": elb2,
            "wr": wr2,
            "hip": hip2,
            "scale": scale,
            "out_sign": inferred_out,
        })

    feats = extract_arm_features(history)
    return feats, label


class GestureNet(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_dataset(n_total: int = 36000, seed: int = 42):
    rng = np.random.default_rng(seed)
    kinds_weights = [
        ("still_idle", 0.18),
        ("still_crossed", 0.08),
        ("still_hip", 0.07),
        ("still_phone", 0.08),
        ("still_talking", 0.10),
        ("point_side", 0.14),
        ("point_forward", 0.10),
        ("point_transition", 0.07),
        ("wave_high", 0.11),
        ("wave_chest", 0.07),
    ]
    kinds = [k for k, _ in kinds_weights]
    probs = np.array([w for _, w in kinds_weights])
    probs /= probs.sum()

    X = np.zeros((n_total, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.int64)
    chosen = rng.choice(kinds, size=n_total, p=probs)
    for i, k in enumerate(chosen):
        X[i], y[i] = simulate_sequence(k, rng)
    return X, y


def main():
    print("[gesture] Generating 36,000 training + 6,000 validation 3D->2D kinematic sequences...")
    X_train, y_train = build_dataset(36000, seed=42)
    X_val, y_val = build_dataset(6000, seed=999)

    mean = X_train.mean(axis=0)
    std = np.maximum(X_train.std(axis=0), 1e-5)

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gesture] Training GestureNet on {device}...")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train)),
                              batch_size=256, shuffle=True)
    model = GestureNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    x_val_t = torch.from_numpy(X_val_n).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    for epoch in range(1, 26):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = model(x_val_t).argmax(dim=1)
            acc = float((preds == y_val_t).float().mean().item())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 25:
            print(f"  Epoch {epoch:2d}/25 | val_acc={acc:.4f} (best={best_acc:.4f})")

    # Confusion matrix & per-class F1 on held-out val set
    model.load_state_dict({k: torch.from_numpy(v) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds = model(x_val_t).argmax(dim=1).cpu().numpy()

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_val, preds):
        cm[t, p] += 1
    print("\n[gesture] Validation Confusion Matrix (rows=true, cols=pred):")
    print("          still  point   wave")
    for idx, name in enumerate(GESTURE_LABELS):
        tp = cm[idx, idx]
        prec = tp / max(cm[:, idx].sum(), 1)
        rec = tp / max(cm[idx, :].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        print(f"  {name:>5s}:  {cm[idx,0]:5d}  {cm[idx,1]:5d}  {cm[idx,2]:5d}  | P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # Export compact NumPy weights (zero torch dependency needed at runtime!)
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DEFAULT_MODEL_PATH,
        mean=mean,
        std=std,
        w1=best_state["net.0.weight"].T,
        b1=best_state["net.0.bias"],
        w2=best_state["net.3.weight"].T,
        b2=best_state["net.3.bias"],
        w3=best_state["net.6.weight"].T,
        b3=best_state["net.6.bias"],
    )
    print(f"\n[gesture] Saved trained model -> {DEFAULT_MODEL_PATH} ({DEFAULT_MODEL_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
