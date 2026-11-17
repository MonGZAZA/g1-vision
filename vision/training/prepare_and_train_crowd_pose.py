"""Fine-tune TorchVision Keypoint R-CNN (BSD-3-Clause) on crowded COCO2017 scenes (CC BY 4.0).

Uses the COCO2017 validation split (>= 3 people per frame) with crowd/occlusion augmentations
(random horizontal flip with keypoint swap, lighting jitter, and random occlusion patches) to
improve multi-person recall and pose accuracy in dense/occluded rooms.

Saves ONLY the fine-tuned FPN/RPN/RoI head weights in FP16 to:
  vision/models/crowd_keypointrcnn_heads.pt  (< 100 MB, commercially permissive BSD-3-Clause).
"""

import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

VISION_DIR = Path(__file__).resolve().parents[1]
EXP05_DIR = VISION_DIR.parent
CACHE_DIR = EXP05_DIR / "cache" / "datasets" / "coco_crowd"
MODELS_DIR = VISION_DIR / "models"

COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def download_file(url: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"[dataset] Using cached {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    print(f"[dataset] Downloading {url} -> {dst.name} ...", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        last_pct = -10
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 10:
                        last_pct = pct
                        print(f"[dataset]   {dst.name}: {pct}%", flush=True)
    tmp.replace(dst)


def collect_crowded_items():
    raw_dir = CACHE_DIR / "raw"
    val_zip = raw_dir / "coco2017val.zip"
    lbl_zip = raw_dir / "coco2017labels-pose.zip"

    download_file("https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip", val_zip)
    download_file("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip", lbl_zip)

    img_src = raw_dir / "coco" / "images" / "val2017"
    if not img_src.exists():
        img_src = raw_dir / "images" / "val2017"
    if not img_src.exists():
        print("[dataset] Extracting coco2017val.zip...", flush=True)
        with zipfile.ZipFile(val_zip, "r") as zf:
            zf.extractall(raw_dir)
        img_src = raw_dir / "images" / "val2017"

    lbl_src = raw_dir / "coco-pose" / "labels" / "val2017"
    if not lbl_src.exists():
        print("[dataset] Extracting coco2017labels-pose.zip (coco-pose/labels/val2017)...", flush=True)
        with zipfile.ZipFile(lbl_zip, "r") as zf:
            members = [m for m in zf.namelist() if "coco-pose/labels/val2017/" in m and m.endswith(".txt")]
            zf.extractall(raw_dir, members=members)

    print("[dataset] Filtering crowded pose scenes (>= 3 people with keypoints per image)...", flush=True)
    crowded = []
    for lbl_file in sorted(lbl_src.glob("*.txt")):
        lines = [ln.strip() for ln in lbl_file.read_text(encoding="utf-8").splitlines() if len(ln.split()) >= 56]
        if len(lines) >= 3:
            jpg = img_src / f"{lbl_file.stem}.jpg"
            if jpg.exists():
                crowded.append((jpg, lbl_file, len(lines)))

    very_crowded = [x for x in crowded if x[2] >= 4]
    rng = random.Random(42)
    rng.shuffle(very_crowded)
    n_val = max(120, int(0.18 * len(crowded)))
    val_stems = {x[0].stem for x in very_crowded[:n_val]}

    train_items = [x for x in crowded if x[0].stem not in val_stems]
    val_items = [x for x in crowded if x[0].stem in val_stems]
    print(f"[dataset] Crowded COCO2017 (>=3 people): total={len(crowded)}, train={len(train_items)}, val={len(val_items)}", flush=True)
    return train_items, val_items


class CrowdPoseDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        jpg_path, lbl_path, _ = self.items[idx]
        bgr = cv2.imread(str(jpg_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]

        boxes, kpts_list = [], []
        for ln in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = ln.strip().split()
            if len(parts) < 5 + 51:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x0 = max(0.0, (cx - bw / 2.0) * iw)
            y0 = max(0.0, (cy - bh / 2.0) * ih)
            x1 = min(float(iw), (cx + bw / 2.0) * iw)
            y1 = min(float(ih), (cy + bh / 2.0) * ih)
            if x1 - x0 < 6 or y1 - y0 < 8:
                continue
            kp = np.zeros((17, 3), dtype=np.float32)
            raw_kp = list(map(float, parts[5:5 + 51]))
            for k in range(17):
                kx, ky, kv = raw_kp[3 * k], raw_kp[3 * k + 1], raw_kp[3 * k + 2]
                if kv > 0:
                    kp[k, 0] = max(0.0, min(float(iw - 1), kx * iw))
                    kp[k, 1] = max(0.0, min(float(ih - 1), ky * ih))
                    kp[k, 2] = 1.0 if kv >= 1 else 0.0
            boxes.append([x0, y0, x1, y1])
            kpts_list.append(kp)

        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                for b in boxes:
                    ox0, ox1 = b[0], b[2]
                    b[0], b[2] = iw - ox1, iw - ox0
                for kp in kpts_list:
                    vis_m = kp[:, 2] > 0
                    kp[vis_m, 0] = iw - kp[vis_m, 0]
                    kp[:] = kp[COCO_FLIP_IDX]
            # Random brightness/contrast jitter
            if random.random() < 0.5:
                alpha = random.uniform(0.80, 1.25)
                beta = random.uniform(-18, 18)
                rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            # Random occlusion patch (teaches robustness to crowded foreground occlusions)
            if random.random() < 0.45:
                oh = int(ih * random.uniform(0.10, 0.22))
                ow = int(iw * random.uniform(0.10, 0.22))
                oy = random.randint(0, max(0, ih - oh))
                ox = random.randint(0, max(0, iw - ow))
                rgb[oy:oy + oh, ox:ox + ow] = random.randint(40, 180)

        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "keypoints": torch.tensor(np.array(kpts_list), dtype=torch.float32),
        }
        return img_t, target


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6)


@torch.no_grad()
def evaluate_crowd(model, val_loader, device, score_thresh=0.45):
    model.eval()
    tp_total, fp_total, fn_total = 0, 0, 0
    for imgs, targets in val_loader:
        imgs = [im.to(device) for im in imgs]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            preds = model(imgs)
        for pred, tgt in zip(preds, targets):
            keep = (pred["labels"] == 1) & (pred["scores"] >= score_thresh)
            p_boxes = pred["boxes"][keep].cpu().numpy()
            g_boxes = tgt["boxes"].numpy()
            if len(g_boxes) == 0:
                fp_total += len(p_boxes)
                continue
            if len(p_boxes) == 0:
                fn_total += len(g_boxes)
                continue
            iou = _box_iou_matrix(p_boxes, g_boxes)
            matched_g = set()
            tp = 0
            for pi in range(len(p_boxes)):
                best_g = int(np.argmax(iou[pi]))
                if iou[pi, best_g] >= 0.50 and best_g not in matched_g:
                    matched_g.add(best_g)
                    tp += 1
            tp_total += tp
            fp_total += len(p_boxes) - tp
            fn_total += len(g_boxes) - tp
    prec = tp_total / max(tp_total + fp_total, 1)
    rec = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return prec, rec, f1, tp_total, fp_total, fn_total


def save_crowd_heads(model, dst: Path):
    """Save FPN + RPN + RoI heads in FP16 (< 75 MB, under GitHub 100 MB limit)."""
    head_state = {
        k: v.detach().cpu().half()
        for k, v in model.state_dict().items()
        if not k.startswith("backbone.body.")
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head_state, dst)
    print(f"[train] Saved FP16 crowd heads -> {dst} ({dst.stat().st_size / 1e6:.2f} MB)", flush=True)


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    train_items, val_items = collect_crowded_items()
    train_ds = CrowdPoseDataset(train_items, augment=True)
    val_ds = CrowdPoseDataset(val_items, augment=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}", flush=True)

    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_nms_thresh=0.55,
        box_detections_per_img=60,
        rpn_post_nms_top_n_test=800,
    ).to(device)

    p0, r0, f1_0, tp0, fp0, fn0 = evaluate_crowd(model, val_loader, device)
    print(f"[baseline] Crowded Val (score>=0.45): P={p0:.4f} R={r0:.4f} F1={f1_0:.4f} (TP={tp0} FP={fp0} FN={fn0})", flush=True)

    # Freeze ResNet50 backbone body; fine-tune FPN, RPN, Box & Keypoint heads
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith("backbone.body.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1 = f1_0
    dst_heads = MODELS_DIR / "crowd_keypointrcnn_heads.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (imgs, targets) in enumerate(train_loader, 1):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())

        scheduler.step()
        p, r, f1, tp, fp, fn = evaluate_crowd(model, val_loader, device)
        print(
            f"  Epoch {epoch}/{epochs} | loss={total_loss/len(train_loader):.4f} | "
            f"Val P={p:.4f} R={r:.4f} F1={f1:.4f} (TP={tp} FP={fp} FN={fn})",
            flush=True,
        )
        if f1 >= best_f1:
            best_f1 = f1
            save_crowd_heads(model, dst_heads)

    if not dst_heads.exists():
        save_crowd_heads(model, dst_heads)


if __name__ == "__main__":
    main()
