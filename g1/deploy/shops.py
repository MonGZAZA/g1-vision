"""Shop files: exp05/shops/<name>/shop.json + one background image (floor plan or map), uploaded from the UI.

shop.json (format "exp05-shop", version 1), units m and deg:
  frame        x right, y into the shop, origin = whatever the operator sets (mockup: inside south-west corner);
               yaw 0 = +x, 90 = +y (counter-clockwise)
  background   {image, resolution_m_per_px, size_px [w, h], origin_bottom_left_m [x, y]}: the world position of the
               image's bottom-left corner. World (x, y) of pixel (u, v) (v down from the top):
               x = ox + u * res, y = oy + (h - v) * res
  robot        {x, y, yaw_deg}
  booths[]     {id, name_th, name_en, rect {x, y, w, h, angle_deg} (lower-left corner + size, rotated about that
               corner), aim {x, y, z}, description_th, description_en, labels [], color}
  faq[]        {q_th, q_en, a_th, a_en, booth?}
  settings     {torso_k, torso_max_deg} (pointing; defaults below)
"""

import io
import json
import re
import shutil
import time
from pathlib import Path

from PIL import Image

import scripts

SHOPS = Path(__file__).resolve().parents[2] / "shops"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_SIDE_M = 200.0  # sanity bound for coordinates
DEFAULT_SETTINGS = {"torso_k": 0.25, "torso_max_deg": 15.0}
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


class ShopError(ValueError):
    pass


def check_name(name):
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ShopError("shop name: 1-40 letters, digits, '-' or '_'")
    return name


def list_shops():
    return sorted(p.parent.name for p in SHOPS.glob("*/shop.json"))


def load(name):
    p = SHOPS / check_name(name) / "shop.json"
    if not p.exists():
        raise ShopError(f"no shop '{name}'")
    shop = json.loads(p.read_text(encoding="utf-8"))
    shop.setdefault("settings", {})
    for k, v in DEFAULT_SETTINGS.items():
        shop["settings"].setdefault(k, v)
    return shop


def _num(v, what, lo=-MAX_SIDE_M, hi=MAX_SIDE_M):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ShopError(f"{what}: not a number") from None
    if not lo <= f <= hi:
        raise ShopError(f"{what}: {f} outside {lo}..{hi}")
    return round(f, 4)


def validate(shop):
    """Clean copy of a shop dict from the UI, or ShopError."""
    if not isinstance(shop, dict):
        raise ShopError("shop: not an object")
    out = {"format": "exp05-shop", "version": 1, "name": check_name(shop.get("name", "")),
           "note": str(shop.get("note", ""))[:500],
           "frame": "x right, y into the shop; yaw 0 = +x, 90 = +y (counter-clockwise); metres, degrees"}
    bg = shop.get("background") or {}
    if bg.get("image"):
        if not re.match(r"^[A-Za-z0-9_.-]{1,80}$", bg["image"]):
            raise ShopError("background image name")
        sz = bg.get("size_px") or [0, 0]
        out["background"] = {"image": bg["image"], "resolution_m_per_px": _num(bg.get("resolution_m_per_px"), "resolution", 1e-4, 10),
                             "size_px": [int(sz[0]), int(sz[1])],
                             "origin_bottom_left_m": [_num(bg["origin_bottom_left_m"][0], "origin x", -10 * MAX_SIDE_M, 10 * MAX_SIDE_M),
                                                      _num(bg["origin_bottom_left_m"][1], "origin y", -10 * MAX_SIDE_M, 10 * MAX_SIDE_M)]}
    if "shop_size_m" in shop:
        out["shop_size_m"] = [_num(shop["shop_size_m"][0], "shop width", 0), _num(shop["shop_size_m"][1], "shop depth", 0)]
    r = shop.get("robot") or {}
    out["robot"] = {"x": _num(r.get("x", 0), "robot x"), "y": _num(r.get("y", 0), "robot y"),
                    "yaw_deg": _num(r.get("yaw_deg", 90), "robot yaw", -360, 360)}
    booths, ids = [], set()
    for b in shop.get("booths") or []:
        bid = str(b.get("id", ""))
        if not ID_RE.match(bid) or bid in ids:
            raise ShopError(f"booth id '{bid}': must be unique, letters/digits/-/_")
        ids.add(bid)
        rc, aim = b.get("rect") or {}, b.get("aim") or {}
        color = str(b.get("color", "#9ecae1"))
        if not re.match(r"^#[0-9a-fA-F]{6}$", color):
            color = "#9ecae1"
        booths.append({
            "id": bid, "name_th": str(b.get("name_th", ""))[:80], "name_en": str(b.get("name_en", ""))[:80],
            "rect": {"x": _num(rc.get("x"), f"{bid} x"), "y": _num(rc.get("y"), f"{bid} y"),
                     "w": _num(rc.get("w"), f"{bid} width", 0.1, MAX_SIDE_M), "h": _num(rc.get("h"), f"{bid} depth", 0.1, MAX_SIDE_M),
                     "angle_deg": _num(rc.get("angle_deg", 0), f"{bid} angle", -360, 360)},
            "aim": {"x": _num(aim.get("x"), f"{bid} aim x"), "y": _num(aim.get("y"), f"{bid} aim y"),
                    "z": _num(aim.get("z", 1.0), f"{bid} aim height", 0, 5)},
            "description_th": str(b.get("description_th", ""))[:1000], "description_en": str(b.get("description_en", ""))[:1000],
            "labels": [str(x)[:40] for x in (b.get("labels") or [])][:20], "color": color})
    out["booths"] = booths
    out["faq"] = [{k: str(f.get(k, ""))[:1000] for k in ("q_th", "q_en", "a_th", "a_en", "booth") if f.get(k)}
                  for f in (shop.get("faq") or []) if isinstance(f, dict)][:200]
    st = shop.get("settings") or {}
    out["settings"] = {"torso_k": _num(st.get("torso_k", DEFAULT_SETTINGS["torso_k"]), "torso k", 0, 1),
                       "torso_max_deg": _num(st.get("torso_max_deg", DEFAULT_SETTINGS["torso_max_deg"]), "torso max", 0, 45)}
    if "scripts" in shop:  # texts the robot speaks (deploy/scripts.py)
        out["scripts"] = scripts.clean(shop["scripts"])
    return out


def save(shop):
    shop = validate(shop)
    d = SHOPS / shop["name"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / "shop.json"
    if p.exists():  # keep one previous version
        shutil.copy2(p, d / "shop.prev.json")
    tmp = d / "shop.json.tmp"
    tmp.write_text(json.dumps(shop, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)
    return shop


def new(name, copy_from=None):
    check_name(name)
    if (SHOPS / name / "shop.json").exists():
        raise ShopError(f"shop '{name}' exists")
    if copy_from:
        src = load(copy_from)
        src["name"] = name
        bg = src.get("background")
        if bg:
            (SHOPS / name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(SHOPS / copy_from / bg["image"], SHOPS / name / bg["image"])
        return save(src)
    return save({"name": name, "robot": {"x": 0, "y": 0, "yaw_deg": 90}, "booths": [], "faq": [],
                 "note": f"created {time.strftime('%Y-%m-%d %H:%M')}"})


def upload_background(name, data, content_type):
    """Store an uploaded image as the shop's background; keeps the scale if there was one, else 0.05 m/px."""
    shop = load(name)
    if len(data) > MAX_IMAGE_BYTES:
        raise ShopError("image larger than 30 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise ShopError("not an image (PNG, JPEG or WebP)") from None
    fmt = (img.format or "").lower()
    ext = {"png": "png", "jpeg": "jpg", "webp": "webp"}.get(fmt)
    if ext is None:
        raise ShopError(f"image format {fmt or '?'}: use PNG, JPEG or WebP")
    w, h = img.size
    if w * h > 60e6:
        raise ShopError(f"image {w}x{h}: too many pixels (max 60 Mpx)")
    d = SHOPS / name
    old = (shop.get("background") or {}).get("image")
    fname = f"background_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
    (d / fname).write_bytes(data)
    bg = shop.get("background") or {}
    res = bg.get("resolution_m_per_px", 0.05)
    shop["background"] = {"image": fname, "resolution_m_per_px": res, "size_px": [w, h],
                          "origin_bottom_left_m": bg.get("origin_bottom_left_m", [0.0, 0.0])}
    shop = save(shop)
    if old and old != fname and old.startswith("background_") and (d / old).exists():
        (d / old).unlink()  # replaced upload; a hand-made map.png (mockup) is kept
    return shop


def file_path(name, fname):
    """Path of a shop file for serving (images only), or None."""
    check_name(name)
    if not re.match(r"^[A-Za-z0-9_.-]{1,80}\.(png|jpg|jpeg|webp)$", fname):
        return None
    p = SHOPS / name / fname
    return p if p.exists() else None
