"""Typed Q&A (task 2): question -> answer (+ the booth to point at), from the shop data only.

  1. language: Thai if the question has Thai letters, else English (the answer uses the same language);
  2. FAQ: a close match (string similarity >= FAQ_MIN) answers directly with the stored answer;
  3. otherwise the local LLM (AI service) phrases a short answer from the shop context: shop and robot names, the
     service text, every booth with its description and where it is as seen from the entrance, and the FAQ;
  4. booth: the FAQ entry's booth, else a booth whose name (or label) appears in the question or the answer.
The operator sees the answer, can edit it, then has it spoken (optionally while pointing at the booth).
"""

import difflib
import re

import scripts
from view3d import booth_geometry

FAQ_MIN = 0.72
THAI = re.compile(r"[฀-๿]")


def lang_of(text):
    return "th" if THAI.search(text or "") else "en"


def _norm(s):
    return re.sub(r"[\s?？!.,ๆ]+|ครับ|คะ|ค่ะ|ไหม|มั้ย|หรือเปล่า|please", "", (s or "").lower())


def where(shop, b, lang):
    g = booth_geometry(shop, b["id"]) or {}
    az, d = g.get("azimuth_deg", 0.0), g.get("distance_m", 0.0)
    if lang == "th":
        side = "ตรงไปข้างหน้า" if abs(az) < 12 else ("ทางซ้าย" if az > 0 else "ทางขวา")
        if abs(az) > 70:
            side = "ทางซ้ายมือสุด" if az > 0 else "ทางขวามือสุด"
        dist = "ใกล้ทางเข้า" if d < 6 else ("ตรงกลางร้าน" if d < 18 else "ด้านในสุดของร้าน")
        return f"{side} {dist} (ประมาณ {d:.0f} เมตร)"
    side = "straight ahead" if abs(az) < 12 else ("on the left" if az > 0 else "on the right")
    if abs(az) > 70:
        side = "far to the left" if az > 0 else "far to the right"
    dist = "near the entrance" if d < 6 else ("in the middle of the shop" if d < 18 else "at the back of the shop")
    return f"{side}, {dist} (about {d:.0f} m)"


def context(shop, lang):
    s = scripts.get(shop)
    lines = []
    if lang == "th":
        lines.append(f"ชื่อร้าน: {s['shop_th']}  ชื่อของคุณ: {s['robot_th']}")
        lines.append("บริการ: " + scripts.split_tags(scripts.fill(shop, "service", "th"))[0])
        lines.append("จุดต่างๆ ในร้าน (มองจากทางเข้า):")
        for b in shop.get("booths", []):
            lines.append(f"- {b['name_th']}: {b.get('description_th', '')} อยู่{where(shop, b, 'th')}")
        lines.append("คำถามที่พบบ่อย:")
        for f in shop.get("faq", []):
            if f.get("q_th") and f.get("a_th"):
                lines.append(f"- ถาม: {f['q_th']} ตอบ: {f['a_th']}")
    else:
        lines.append(f"Shop name: {s['shop_en']}  Your name: {s['robot_en']}")
        lines.append("Service: " + scripts.split_tags(scripts.fill(shop, "service", "en"))[0])
        lines.append("Places in the shop (seen from the entrance):")
        for b in shop.get("booths", []):
            lines.append(f"- {b['name_en']}: {b.get('description_en', '')}; {where(shop, b, 'en')}")
        lines.append("Frequent questions:")
        for f in shop.get("faq", []):
            if f.get("q_en") and f.get("a_en"):
                lines.append(f"- Q: {f['q_en']} A: {f['a_en']}")
    return "\n".join(lines)


def find_booth(shop, *texts):
    """Booth whose Thai/English name or a label occurs in one of the texts (longest name wins)."""
    best, best_len = None, 0
    for t in texts:
        tl = (t or "").lower()
        for b in shop.get("booths", []):
            for name in [b.get("name_th", ""), b.get("name_en", ""), *b.get("labels", [])]:
                n = name.lower().strip()
                if len(n) < 2 or len(n) <= best_len:
                    continue
                # English: whole words ("pc" must not match inside "pick"); Thai has no spaces: substring
                hit = re.search(rf"\b{re.escape(n)}s?\b", tl) if n.isascii() else n in tl
                if hit:
                    best, best_len = b["id"], len(n)
    return best


def match_faq(shop, question, lang):
    qn = _norm(question)
    best, score = None, 0.0
    for f in shop.get("faq", []):
        for key in ("q_th", "q_en"):
            r = difflib.SequenceMatcher(None, qn, _norm(f.get(key, ""))).ratio()
            if r > score:
                best, score = f, r
    if best is not None and score >= FAQ_MIN and best.get(f"a_{lang}"):
        return best, score
    return None, score


def answer(shop, question, speech):
    lang = lang_of(question)
    f, score = match_faq(shop, question, lang)
    if f is not None:
        a = f[f"a_{lang}"]
        return {"ok": True, "lang": lang, "answer": a, "source": f"FAQ ({score:.2f})",
                "booth": f.get("booth") or find_booth(shop, question, a)}
    r = speech.ask(question, context(shop, lang), lang)
    if not r.get("ok"):
        return {"ok": True, "lang": lang, "answer": scripts.fill(shop, "unknown", lang), "source": "fallback",
                "booth": find_booth(shop, question), "warning": r.get("reason", "")}
    a = r["answer"].strip().strip('"')
    if lang_of(a) != lang:  # the model drifted into the other language: say we don't know instead of mixing
        return {"ok": True, "lang": lang, "answer": scripts.fill(shop, "unknown", lang), "source": "fallback",
                "booth": find_booth(shop, question), "warning": f"LLM answered in the wrong language: {a}"}
    return {"ok": True, "lang": lang, "answer": a, "source": f"LLM ({r.get('seconds', 0)} s)",
            "booth": find_booth(shop, question, a)}
