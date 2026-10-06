"""Offline OCR (RapidOCR / PaddleOCR models on ONNX Runtime, Apache-2.0).
Reads horizontal and vertical (rotated) text and classifies it as
room names or dimension strings."""
import re
import cv2
import numpy as np

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def _run(img):
    res, _ = _engine()(img)
    out = []
    for box, txt, score in (res or []):
        out.append((np.array(box, dtype=np.float32), str(txt), float(score)))
    return out


def read_text(img_bgr, min_score=0.45):
    """Return list of dicts: text, score, box(4x2, original coords), center, vertical(bool)."""
    H0, W0 = img_bgr.shape[:2]
    up = 2.0 if max(H0, W0) < 1400 else 1.0      # small drawings: OCR at 2x for tiny text
    if up != 1.0:
        big = cv2.resize(img_bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        items = read_text_raw(big, min_score)
        for it in items:
            it["box"] = it["box"] / up; it["center"] = it["center"] / up; it["w"] /= up; it["h"] /= up
        return items
    return read_text_raw(img_bgr, min_score)


def read_text_raw(img_bgr, min_score=0.45):
    H, W = img_bgr.shape[:2]
    items = []
    for box, txt, sc in _run(img_bgr):
        if sc < min_score:
            continue
        w = np.ptp(box[:, 0]); h = np.ptp(box[:, 1])
        items.append(dict(text=txt, score=sc, box=box, vertical=False, w=w, h=h))
    # vertical text: rotate both ways, keep boxes that are tall in the original
    for rot, inv in ((cv2.ROTATE_90_CLOCKWISE, "cw"), (cv2.ROTATE_90_COUNTERCLOCKWISE, "ccw")):
        r = cv2.rotate(img_bgr, rot)
        for box, txt, sc in _run(r):
            if sc < min_score:
                continue
            b = box.copy()
            if inv == "cw":   # rotated (x', y') -> original (x, y) = (y', H-1-x')
                ob = np.c_[b[:, 1], H - 1 - b[:, 0]]
            else:             # ccw: original (x, y) = (W-1-y', x')
                ob = np.c_[W - 1 - b[:, 1], b[:, 0]]
            w = np.ptp(ob[:, 0]); h = np.ptp(ob[:, 1])
            if h > 1.3 * w:
                items.append(dict(text=txt, score=sc, box=ob, vertical=True, w=w, h=h))
    for it in items:
        it["center"] = it["box"].mean(0)
    # de-duplicate overlapping detections (keep best score)
    items.sort(key=lambda d: -d["score"])
    keep = []
    for it in items:
        dup = False
        for k in keep:
            if np.linalg.norm(it["center"] - k["center"]) < 0.5 * max(k["w"], k["h"], 8):
                dup = True; break
        if not dup:
            keep.append(it)
    for it in keep:
        it["text"] = tidy_text(it["text"])
        it["dim"] = parse_dimension(it["text"])
        it["dim_cands"] = dimension_candidates(it["text"])
        if it["dim"] is None and it["dim_cands"]:
            it["dim"] = it["dim_cands"][0][0]
    return keep


def tidy_text(t):
    """Restore spaces OCR sometimes drops: 'LivingRoomandKitchen' -> 'Living Room and Kitchen'."""
    t = re.sub(r"([a-z])(and|with|cum)([A-Z])", r"\1 \2 \3", t)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    return re.sub(r"\s+", " ", t).strip()


def dimension_candidates(t):
    """All plausible readings of a (possibly OCR-damaged) dimension string as
    (metres, weight) pairs. weight 1.0 = clean, explicit reading; 0.5 = repaired
    reading (missing foot/inch marks or decimal point). The calibration picks the
    readings that agree with each other."""
    out = []
    s = t.strip()
    low = s.lower()
    if re.search(r"[a-ln-z]", low.replace("mm", "").replace("m", "")):
        return out                                   # contains words -> not a dimension
    p = parse_dimension(s)
    has_imp = bool(re.search(r"['′’`´\"″”°]", s))
    has_met = bool(re.search(r"\d\s*(mm|m|cm)\b", low)) or bool(re.search(r"\d[.,]\d", s))
    if p:
        out.append((p, 1.0 if (has_imp or has_met) else 0.5))
    digits = re.sub(r"[^0-9]", "", s)
    if 2 <= len(digits) <= 4:
        if not has_met:                              # imperial repairs: '123*' -> 12'3"
            for k in range(1, len(digits)):
                ft, inch = int(digits[:k]), int(digits[k:])
                if 0 < ft < 100 and inch < 12 and not (len(digits[k:]) == 2 and digits[k] == "0" and inch != 0):
                    out.append(((ft * 12 + inch) * 0.0254, 0.5))
        if not has_imp:                              # metric repairs: '217m' -> 2.17, '1594' -> 15.94, '3250' -> 3.25
            v = int(digits)
            for div in (100.0, 1000.0, 10.0):
                m = v / div
                if 0.4 <= m <= 60:
                    out.append((m, 0.5))
    seen = []
    for v, w in out:
        if all(abs(v - u) > 1e-6 for u, _ in seen):
            seen.append((v, w))
    return seen


_FTIN = re.compile(r"^\s*(\d{1,3})\s*['′’`´°\"*]{1,2}\s*[-–]?\s*(\d{1,2}(?:\.\d+)?)?\s*(?:[\"″”*°']{0,2})?\s*(?:\d/\d)?\s*$")


def parse_dimension(t):
    """Parse a dimension string. Returns metres (float) or None.
    Handles 18' 3", 17′-9", 6'8, 15°10° (OCR noise), 3.25 m, 3250 mm, 3.25"""
    s = t.strip().replace(",", ".").replace("O", "0").replace("o", "0")
    s = s.replace("I", "1").replace("l", "1")
    m = _FTIN.match(s)
    if m:
        ft = int(m.group(1)); inch = float(m.group(2)) if m.group(2) else 0.0
        if inch < 12 and 0 < ft < 300:
            return (ft * 12 + inch) * 0.0254
    m = re.match(r"^\s*(\d{1,3})\s*[\"″]\s*(\d{1,2})\s*[\"″*]?\s*$", s)  # 17" 9" OCR confusion
    if m and int(m.group(2)) < 12:
        return (int(m.group(1)) * 12 + int(m.group(2))) * 0.0254
    m = re.match(r"^\s*(\d{1,2}\.\d{1,3})\s*(m|M)?\s*$", s)
    if m:
        v = float(m.group(1))
        if 0.3 < v < 100:
            return v
    m = re.match(r"^\s*(\d{3,5})\s*(mm)?\s*$", s)
    if m:
        v = float(m.group(1))
        if 300 <= v <= 60000:
            return v / 1000.0
    m = re.match(r"^\s*(\d{2,4})\s*cm\s*$", s)
    if m:
        return float(m.group(1)) / 100
    return None


ROOM_KEYWORDS = [
    # (type, keywords) -- order matters (first match wins)
    ("living_kitchen", [("living", "kitchen"), ("kitchen", "dining"), ("kitchen", "living"), ("great room",)]),
    ("master_bedroom", [("master",), ("m. bed",), ("m.bed",), ("primary bed",)]),
    ("bedroom", [("bed",), ("br",), ("guest",), ("nursery",), ("kids",), ("chamber",)]),
    ("bathroom", [("bath",), ("wc",), ("w.c",), ("toilet",), ("shower",), ("ensuite",), ("en-suite",), ("powder",), ("lav",)]),
    ("kitchen", [("kitchen",), ("kit.",), ("kitchenette",), ("pantry",)]),
    ("dining", [("dining",), ("dinning",)]),
    ("living", [("living",), ("lounge",), ("family",), ("drawing",), ("sitting",), ("hall ",), ("salon",)]),
    ("hallway", [("hallway",), ("corridor",), ("passage",), ("foyer",), ("entry",), ("entrance",), ("lobby",), ("hall",)]),
    ("balcony", [("balcony",), ("terrace",), ("deck",), ("patio",), ("veranda",), ("porch",), ("sit out",)]),
    ("storage", [("storage",), ("store",), ("closet",), ("w.i.c",), ("wic",), ("wardrobe",), ("utility",), ("laundry",), ("cupboard",)]),
    ("office", [("office",), ("study",), ("den",), ("work",)]),
    ("stair", [("stair",), ("up",), ("dn",)]),
    ("garage", [("garage",), ("parking",), ("car",)]),
    ("dressing", [("dress",)]),
    ("pooja", [("pooja",), ("puja",), ("prayer",)]),
]


def room_type_from_text(t):
    s = " " + t.lower().strip() + " "
    for typ, groups in ROOM_KEYWORDS:
        for g in groups:
            if all(k in s for k in g):
                if typ == "bedroom" and g == ("br",) and not re.search(r"\bbr\b", s):
                    continue
                if typ in ("stair",) and not re.search(r"\b(up|dn|stairs?)\b", s):
                    continue
                if typ == "office" and g in (("den",), ("work",)) and not re.search(r"\b(den|work ?room)\b", s):
                    continue
                return typ
    return None
