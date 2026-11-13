"""Texts the robot speaks, per shop (shop.json "scripts"), with defaults. Edited in the UI (Shop tab, Scripts).

Placeholders: {robot} robot name, {shop} shop name, {name} booth name (point sentences).
Gesture tags in square brackets (e.g. "[welcome]", "[present]") pick a gesture at that point (S5) and are not spoken.
"""

import re

DEFAULTS = {
    "robot_th": "มะลิ", "robot_en": "Mali",
    "shop_th": "เทคโนมาร์ท", "shop_en": "Techno Mart",
    "intro_th": "สวัสดีค่ะ [welcome] ดิฉันชื่อ{robot} เป็นหุ่นยนต์ต้อนรับของ{shop}ค่ะ ยินดีต้อนรับนะคะ "
                "[present] ถ้าต้องการสอบถามอะไร ถามดิฉันได้เลยค่ะ",
    "intro_en": "Hello! [welcome] My name is {robot}, the greeter robot of {shop}. Welcome! "
                "[present] If you have any questions, please ask me.",
    "service_th": "[present] ร้านของเรามีสินค้าไอทีครบครันค่ะ ทั้งโทรศัพท์ คอมพิวเตอร์ เครื่องเสียง และอุปกรณ์สมาร์ทโฮม "
                  "[both] มีโซนเกมให้ทดลองเล่น และศูนย์บริการซ่อมด้านในค่ะ [welcome] ขอให้สนุกกับการช้อปปิ้งนะคะ",
    "service_en": "[present] We have everything in IT: phones, computers, audio and smart home devices. "
                  "[both] There is a gaming zone to try games, and a service center at the back. "
                  "[welcome] Enjoy your shopping!",
    "point_th": "{name}อยู่ทางนี้ค่ะ",
    "point_en": "{name} is this way, please.",
    "filler_th": "สักครู่นะคะ",
    "filler_en": "One moment, please.",
    "unknown_th": "ขอโทษค่ะ ดิฉันไม่มีข้อมูลเรื่องนี้ รบกวนสอบถามที่จุดบริการข้อมูลนะคะ",
    "unknown_en": "Sorry, I don't have that information. Please ask at the information desk.",
}
KEYS = list(DEFAULTS)
TAG_RE = re.compile(r"\[([a-z_]+)\]")


def get(shop):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in ((shop or {}).get("scripts") or {}).items() if k in DEFAULTS and isinstance(v, str)})
    return s


def clean(scripts):
    """Validated copy for shop.json (known keys, strings, length limit)."""
    return {k: str(v)[:2000] for k, v in (scripts or {}).items() if k in DEFAULTS}


def fill(shop, key, lang, **kw):
    s = get(shop)
    return s[f"{key}_{lang}"].format(robot=s[f"robot_{lang}"], shop=s[f"shop_{lang}"], **kw)


def split_tags(text):
    """Text with gesture tags -> (spoken text, [(char index in the spoken text, tag)])."""
    parts = TAG_RE.split(text)  # text, tag, text, tag, ...
    spoken, tags = "", []
    for i, p in enumerate(parts):
        if i % 2:
            tags.append((len(spoken), p))
            continue
        piece = re.sub(r"\s+", " ", p)
        if (not spoken or spoken.endswith(" ")) and piece.startswith(" "):
            piece = piece[1:]
        spoken += piece
    end = len(spoken.rstrip())
    return spoken[:end], [(min(i, end), t) for i, t in tags]


def phrase_index(phrases, spoken, char_pos):
    """Index of the phrase (from the TTS timing) that starts at or after char_pos of the spoken text."""
    pos = 0
    for i, p in enumerate(phrases):
        k = spoken.find(p["text"].split(" ")[0], pos)
        k = pos if k < 0 else k
        if k >= char_pos - 1:
            return i
        pos = k + 1
    return len(phrases) - 1
