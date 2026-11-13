"""Make the exp05 mockup shop: a 2D floor-plan background (map.png), the layout file (shop.json) and a preview
with the robot, booths and pointing directions (preview.png).

The real shop will be uploaded later through the UI; this mockup only stands in for it. map.png is drawn like a
plan someone would upload (walls, shelves, booth areas with names, scale bar), without the robot.

World frame (metres): x to the right (east), y into the shop (north), origin = inside corner of the south-west
wall. The robot stands in the entrance on the south wall and faces +y (yaw 90 deg).

Run: /usr/bin/python3 make_mockup.py   (needs matplotlib + a Thai font, e.g. Loma from fonts-tlwg)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrow, Polygon, Rectangle, Wedge

HERE = Path(__file__).resolve().parent
RES = 0.04                       # m per pixel of map.png
X0, X1, Y0, Y1 = -1.0, 37.0, -2.0, 31.0   # world extent of the image (m); shop interior is 0..36 x 0..30
W_SHOP, H_SHOP = 36.0, 30.0
ENTRANCE = (16.0, 20.0)          # door opening on the south wall (x from..to)
ROBOT = {"x": 18.0, "y": 1.2, "yaw_deg": 90.0}
AIM_Z = 1.0                      # default aim height: display/counter height (m)

FONT = "Loma"
if not any(FONT in f.name for f in font_manager.fontManager.ttflist):
    FONT = "Garuda"

# id, Thai name, English name, x, y, w, h (lower-left corner + size, m), colour, short descriptions
BOOTHS = [
    ("info", "จุดบริการข้อมูล", "Information", 22.0, 2.5, 3.0, 2.0, "#9ecae1",
     "สอบถามข้อมูลสินค้าและโปรโมชั่น", "Product and promotion information"),
    ("cashier", "แคชเชียร์", "Cashier", 8.5, 2.5, 5.0, 2.0, "#fdd0a2",
     "ชำระเงิน รับเงินสด บัตร และ QR", "Payment: cash, card and QR"),
    ("cafe", "คาเฟ่", "Cafe", 29.0, 2.0, 6.0, 5.0, "#d9b38c",
     "กาแฟ เครื่องดื่ม และขนม", "Coffee, drinks and snacks"),
    ("restroom", "ห้องน้ำ", "Restrooms", 0.0, 0.0, 4.0, 4.5, "#d9d9d9",
     "ห้องน้ำชายและหญิง", "Men's and women's restrooms"),
    ("new", "สินค้ามาใหม่", "New Arrivals", 15.2, 3.2, 1.8, 1.1, "#f768a1",
     "โต๊ะสินค้าใหม่และโปรโมชั่นประจำสัปดาห์", "New products and this week's promotions"),
    ("phones", "โทรศัพท์และแท็บเล็ต", "Phones & Tablets", 4.0, 8.0, 7.0, 4.0, "#a1d99b",
     "สมาร์ทโฟน แท็บเล็ต และอุปกรณ์เสริม", "Smartphones, tablets and accessories"),
    ("computers", "คอมพิวเตอร์", "Computers", 25.0, 8.0, 7.0, 4.0, "#bcbddc",
     "โน้ตบุ๊ก คอมพิวเตอร์ตั้งโต๊ะ และจอ", "Laptops, desktops and monitors"),
    ("robots", "โชว์รูมหุ่นยนต์", "Robot Showroom", 14.5, 14.0, 7.0, 5.0, "#fc9272",
     "หุ่นยนต์บริการและหุ่นยนต์การศึกษา", "Service and education robots"),
    ("smarthome", "บ้านอัจฉริยะ", "Smart Home", 2.0, 19.5, 7.0, 5.0, "#c7e9c0",
     "อุปกรณ์บ้านอัจฉริยะและกล้องวงจรปิด", "Smart home devices and cameras"),
    ("audio", "เครื่องเสียง", "Audio", 28.0, 18.5, 6.0, 5.0, "#fdae6b",
     "หูฟัง ลำโพง และเครื่องเสียง", "Headphones, speakers and audio"),
    ("gaming", "โซนเกม", "Gaming Zone", 8.0, 25.5, 8.0, 3.5, "#9e9ac8",
     "เครื่องเล่นเกมและอุปกรณ์เกมมิ่ง", "Game consoles and gaming gear"),
    ("service", "ศูนย์บริการซ่อม", "Service Center", 22.5, 25.5, 8.0, 3.5, "#6baed6",
     "ซ่อมและรับประกันสินค้า", "Repairs and warranty"),
]

# Shelves / racks (x, y, w, h): background only, not booths
SHELVES = [(13.0, 7.5, 1.0, 4.5), (22.0, 7.5, 1.0, 4.5),
           (11.5, 13.5, 0.8, 5.0), (23.7, 13.5, 0.8, 5.0),
           (11.0, 20.5, 5.0, 0.8), (20.0, 20.5, 5.0, 0.8)]

FAQ = [
    {"q_th": "ร้านเปิดกี่โมง", "q_en": "What are the opening hours?",
     "a_th": "ร้านเปิดทุกวัน 10 โมงเช้าถึง 2 ทุ่มค่ะ", "a_en": "We are open every day from 10 a.m. to 8 p.m."},
    {"q_th": "มีที่จอดรถไหม", "q_en": "Is there parking?",
     "a_th": "มีที่จอดรถชั้นใต้ดิน ซื้อครบ 500 บาทจอดฟรี 2 ชั่วโมงค่ะ",
     "a_en": "There is basement parking, free for 2 hours with a purchase of 500 baht."},
    {"q_th": "ซ่อมโทรศัพท์ได้ไหม", "q_en": "Can you repair my phone?",
     "a_th": "ได้ค่ะ ที่ศูนย์บริการซ่อมด้านในสุดทางขวาค่ะ", "a_en": "Yes, at the Service Center at the back on the right.",
     "booth": "service"},
]


def to_px(x, y):
    """World (m) -> pixel (col, row) of map.png, row 0 at the top."""
    return (x - X0) / RES, (Y1 - y) / RES


def robot_frame(b):
    """Booth aim point -> (azimuth deg, + = left of the robot's facing; horizontal distance m)."""
    ax, ay = b["aim"]["x"], b["aim"]["y"]
    dx, dy = ax - ROBOT["x"], ay - ROBOT["y"]
    heading = math.radians(ROBOT["yaw_deg"])
    fwd = dx * math.cos(heading) + dy * math.sin(heading)
    left = -dx * math.sin(heading) + dy * math.cos(heading)
    return math.degrees(math.atan2(left, fwd)), math.hypot(dx, dy)


def draw_plan(ax):
    ax.set_facecolor("white")
    wall = 0.3
    ex0, ex1 = ENTRANCE
    walls = [(-wall, -wall, ex0 + wall, wall), (ex1, -wall, W_SHOP - ex1 + wall, wall),   # south, with door gap
             (-wall, H_SHOP, W_SHOP + 2 * wall, wall),                                     # north
             (-wall, -wall, wall, H_SHOP + 2 * wall), (W_SHOP, -wall, wall, H_SHOP + 2 * wall)]  # west, east
    for x, y, w, h in walls:
        ax.add_patch(Rectangle((x, y), w, h, color="#333333", lw=0))
    # glass doors (sliding) in the entrance
    ax.plot([ex0, ex1], [-wall / 2, -wall / 2], color="#7fb3d5", lw=2, ls="--")
    ax.text((ex0 + ex1) / 2, -1.2, "ทางเข้า / ENTRANCE", ha="center", va="center", fontsize=11,
            family=FONT, weight="bold")
    # floor grid every 5 m (light), like a CAD plan
    for gx in range(0, int(W_SHOP) + 1, 5):
        ax.plot([gx, gx], [0, H_SHOP], color="#eeeeee", lw=0.6, zorder=0)
    for gy in range(0, int(H_SHOP) + 1, 5):
        ax.plot([0, W_SHOP], [gy, gy], color="#eeeeee", lw=0.6, zorder=0)
    for x, y, w, h in SHELVES:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#bdbdbd", edgecolor="#636363", lw=0.8))
    for bid, th, en, x, y, w, h, col, *_ in BOOTHS:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor="#444444", lw=1.0, alpha=0.85))
        ax.text(x + w / 2, y + h / 2 + 0.35, th, ha="center", va="center", fontsize=9, family=FONT)
        ax.text(x + w / 2, y + h / 2 - 0.45, en, ha="center", va="center", fontsize=8, family=FONT,
                color="#333333")
    # scale bar 5 m, bottom right outside the shop
    sx, sy = 30.0, -1.3
    ax.add_patch(Rectangle((sx, sy - 0.12), 5.0, 0.24, color="black"))
    ax.text(sx + 2.5, sy - 0.55, "5 m", ha="center", va="center", fontsize=8, family=FONT)
    ax.text(1.0, -1.2, "MOCKUP SHOP 36 × 30 m (exp05)", ha="left", va="center", fontsize=8, family=FONT,
            color="#555555")


def new_fig():
    wpx, hpx = round((X1 - X0) / RES), round((Y1 - Y0) / RES)
    dpi = 100
    fig = plt.figure(figsize=(wpx / dpi, hpx / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax, dpi, (wpx, hpx)


LABELS = {'info': ['ประชาสัมพันธ์', 'สอบถาม', 'help desk', 'information desk'],
          'cashier': ['จ่ายเงิน', 'ชำระเงิน', 'แคชเชียร์', 'pay', 'checkout'],
          'cafe': ['กาแฟ', 'ร้านกาแฟ', 'เครื่องดื่ม', 'coffee', 'drink'],
          'restroom': ['ห้องน้ำ', 'สุขา', 'toilet', 'restroom', 'bathroom'],
          'new': ['สินค้าใหม่', 'ของใหม่', 'new product'],
          'phones': ['โทรศัพท์', 'มือถือ', 'แท็บเล็ต', 'phone', 'tablet', 'iphone', 'smartphone'],
          'computers': ['คอม', 'โน้ตบุ๊ก', 'แล็ปท็อป', 'laptop', 'notebook', 'computer', 'pc'],
          'robots': ['หุ่นยนต์', 'robot'],
          'smarthome': ['สมาร์ทโฮม', 'บ้านอัจฉริยะ', 'smart home', 'iot'],
          'audio': ['หูฟัง', 'ลำโพง', 'เครื่องเสียง', 'headphone', 'speaker', 'earbuds'],
          'gaming': ['เกม', 'เกมมิ่ง', 'game', 'gaming', 'console'],
          'service': ['ซ่อม', 'ศูนย์บริการ', 'เคลม', 'repair', 'fix', 'warranty']}  # synonyms: Q&A finds the booth to point at


def main():
    booths = []
    for bid, th, en, x, y, w, h, col, dth, den in BOOTHS:
        booths.append({"id": bid, "name_th": th, "name_en": en,
                       "rect": {"x": x, "y": y, "w": w, "h": h, "angle_deg": 0.0},
                       "aim": {"x": round(x + w / 2, 2), "y": round(y + h / 2, 2), "z": AIM_Z},
                       "description_th": dth, "description_en": den, "labels": LABELS.get(bid, []), "color": col})

    # 1) background map (what a user would upload)
    fig, ax, dpi, (wpx, hpx) = new_fig()
    draw_plan(ax)
    fig.savefig(HERE / "map.png", dpi=dpi)
    plt.close(fig)

    # 2) layout file
    shop = {
        "format": "exp05-shop", "version": 1, "name": "mockup",
        "note": "Mockup shop for exp05 until the real shop is uploaded. Units: metres, degrees.",
        "frame": "x right/east, y into the shop/north, origin = inside corner of the south-west wall; "
                 "yaw 0 = +x, 90 = +y (counter-clockwise)",
        "background": {"image": "map.png", "resolution_m_per_px": RES, "size_px": [wpx, hpx],
                       "origin_bottom_left_m": [X0, Y0]},
        "shop_size_m": [W_SHOP, H_SHOP],
        "robot": ROBOT,
        "booths": booths,
        "faq": FAQ,
    }
    (HERE / "shop.json").write_text(json.dumps(shop, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 3) preview: robot, pointing rays, small torso turn (plan's rule), table of azimuth/distance
    fig, ax, dpi, _ = new_fig()
    draw_plan(ax)
    rx, ry, yaw = ROBOT["x"], ROBOT["y"], math.radians(ROBOT["yaw_deg"])
    ax.add_patch(Wedge((rx, ry), 6.0, ROBOT["yaw_deg"] - 90, ROBOT["yaw_deg"] + 90, color="#fff7bc",
                       alpha=0.5, zorder=0.5))
    ax.add_patch(Polygon([(rx + 0.6 * math.cos(yaw), ry + 0.6 * math.sin(yaw)),
                          (rx + 0.4 * math.cos(yaw + 2.4), ry + 0.4 * math.sin(yaw + 2.4)),
                          (rx + 0.4 * math.cos(yaw - 2.4), ry + 0.4 * math.sin(yaw - 2.4))],
                         color="#d62728", zorder=5))
    ax.text(rx, ry + 0.95, "G1", ha="center", fontsize=9, color="#d62728", weight="bold", family=FONT, zorder=5)
    rows = []
    for b in booths:
        az, d = robot_frame(b)
        torso = max(-15.0, min(15.0, 0.25 * az))
        rows.append((b["id"], az, d, torso))
        ax.add_patch(FancyArrow(rx, ry, b["aim"]["x"] - rx, b["aim"]["y"] - ry, width=0.04,
                                head_width=0.4, length_includes_head=True, color="#d62728", alpha=0.5,
                                zorder=4))
    txt = "booth        az°    dist m  torso°\n" + "\n".join(
        f"{i:<11}{a:+6.0f}  {d:7.1f}  {t:+6.1f}" for i, a, d, t in rows)
    ax.text(35.7, 18.1, txt, ha="right", va="top", fontsize=7, family="monospace",
            bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9), zorder=6)
    fig.savefig(HERE / "preview.png", dpi=dpi)
    plt.close(fig)
    for r in rows:
        print(f"{r[0]:<11} az {r[1]:+6.1f} deg  dist {r[2]:5.1f} m  torso {r[3]:+5.1f} deg")


if __name__ == "__main__":
    main()
