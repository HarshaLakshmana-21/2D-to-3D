"""Output #1: the annotated 2D detection image."""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOM_COLORS = {  # RGB
    "living": (255, 205, 130), "living_kitchen": (255, 190, 120), "kitchen": (255, 150, 90),
    "dining": (255, 220, 150), "bedroom": (140, 200, 255), "master_bedroom": (110, 170, 245),
    "bathroom": (130, 230, 220), "balcony": (170, 225, 140), "hallway": (215, 190, 255),
    "storage": (210, 210, 150), "stair": (200, 160, 220), "office": (250, 170, 210),
    "garage": (180, 180, 180), "dressing": (230, 200, 230), "pooja": (255, 230, 120), "room": (220, 220, 220),
}
WALL_EXT = (200, 30, 30); WALL_INT = (235, 110, 40); DOOR = (20, 160, 60); WINDOW = (30, 110, 230)


def _font(sz, bold=False):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    cands = [os.path.join(here, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
             "arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
             "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]
    for c in cands:
        try:
            return ImageFont.truetype(c, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_detection(img_bgr, res, cls=None, scale=2):
    H, W = img_bgr.shape[:2]
    base = cv2.resize(img_bgr, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)
    base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = 255 - (255 - base) * 0.45            # fade the drawing
    over = base.copy()
    sx, sy = res["scale"]["px_per_m_x"], res["scale"]["px_per_m_y"]
    ox, oy = res["image"]["origin_px"]
    P = lambda p: (int(round(((p[0] * sx) + ox) * scale)), int(round(((p[1] * sy) + oy) * scale)))

    for r in res["rooms"]:
        pts = np.array([P(p) for p in r["polygon"]], np.int32)
        cv2.fillPoly(over, [pts], ROOM_COLORS.get(r["type"], (220, 220, 220)))
    img = (0.55 * base + 0.45 * over).astype(np.uint8)
    for r in res["rooms"]:
        pts = np.array([P(p) for p in r["polygon"]], np.int32)
        cv2.polylines(img, [pts], True, tuple(int(c * 0.6) for c in ROOM_COLORS.get(r["type"], (200, 200, 200))), 2, cv2.LINE_AA)
    for w in res["walls"]:
        a, b = np.array(P(w["start"]), float), np.array(P(w["end"]), float)
        t = w["thickness_m"] * (sy if w["orientation"] == "horizontal" else sx) * scale
        d = b - a; L = np.linalg.norm(d) + 1e-9; n = np.array([-d[1], d[0]]) / L * t / 2
        quad = np.array([a + n, b + n, b - n, a - n], np.int32)
        col = WALL_EXT if w["type"] == "exterior" else WALL_INT
        cv2.fillPoly(img, [quad], col)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
    for o in res["openings"]:
        x, y, w, h = o["_px"]["bbox"] if "_px" in o else (0, 0, 0, 0)
        col = DOOR if o["type"] == "door" else WINDOW
        cv2.rectangle(img, (x * scale, y * scale), ((x + w) * scale, (y + h) * scale), col, -1)
        cv2.rectangle(img, (x * scale, y * scale), ((x + w) * scale, (y + h) * scale), (255, 255, 255), 1)

    pil = Image.fromarray(img); dr = ImageDraw.Draw(pil)
    f1, f2, f3 = _font(15 * scale // 2, True), _font(12 * scale // 2), _font(10 * scale // 2, True)
    for r in res["rooms"]:
        cx, cy = P(r["centroid"])
        lines = [(r["name"], f1), (f"[{r['type']}]", f2),
                 (f"{r['width_ft_in']} x {r['depth_ft_in']}", f2),
                 (f"{r['area_ft2']:.0f} ft²  /  {r['area_m2']:.1f} m²", f2)]
        hs = [dr.textbbox((0, 0), t, font=f) for t, f in lines]
        tot = sum(b[3] - b[1] + 4 for b in hs)
        bw = max(b[2] - b[0] for b in hs) + 10
        y = cy - tot / 2
        dr.rounded_rectangle([cx - bw / 2, y - 4, cx + bw / 2, y + tot + 2], 6, fill=(255, 255, 255), outline=(90, 90, 90))
        for (t, f), b in zip(lines, hs):
            dr.text((cx - (b[2] - b[0]) / 2, y - b[1]), t, fill=(20, 20, 20), font=f); y += b[3] - b[1] + 4
    for o in res["openings"]:
        if "_px" not in o:
            continue
        x, y, w, h = o["_px"]["bbox"]
        t = f"{o['id']} {o['width_ft_in']}"
        col = DOOR if o["type"] == "door" else WINDOW
        b = dr.textbbox((0, 0), t, font=f3)
        tx = (x + w / 2) * scale - (b[2] - b[0]) / 2
        ty = (y + h) * scale + 3 if o["orientation"] == "horizontal" else (y + h / 2) * scale - (b[3] - b[1]) / 2
        if o["orientation"] == "vertical":
            tx = (x + w) * scale + 4
        dr.rectangle([tx - 2, ty - 1, tx + b[2] - b[0] + 2, ty + b[3] - b[1] + 3], fill=(255, 255, 255))
        dr.text((tx, ty - b[1]), t, fill=col, font=f3)
    for w in res["walls"]:
        if w["length_m"] < 1.0:
            continue
        a, b = P(w["start"]), P(w["end"])
        t = f"{w['id']}: {w['length_ft_in']}"
        bb = dr.textbbox((0, 0), t, font=f3)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if w["orientation"] == "horizontal":
            tx, ty = mx - (bb[2] - bb[0]) / 2, my - (bb[3] - bb[1]) - 8 * scale // 2 - 2
        else:
            tx, ty = mx + 6 * scale // 2, my - (bb[3] - bb[1]) / 2
        dr.text((tx, ty - bb[1]), t, fill=(150, 20, 20), font=f3)

    # legend / info panel
    pw = 360 * scale // 2
    canvas = Image.new("RGB", (pil.width + pw, max(pil.height, 420 * scale // 2)), (255, 255, 255))
    canvas.paste(pil, (0, 0)); d2 = ImageDraw.Draw(canvas)
    x0 = pil.width + 14; y = 14
    d2.text((x0, y), "Detection result", fill=(0, 0, 0), font=_font(18 * scale // 2, True)); y += 34 * scale // 2
    b = res["building"]
    info = [f"Scale: {res['scale']['px_per_ft_x']:.2f} px/ft  ({res['scale']['method']})",
            f"Footprint: {b['width_ft_in']} x {b['depth_ft_in']}",
            f"  = {b['width_m']:.2f} m x {b['depth_m']:.2f} m",
            f"Interior area: {b['floor_area_m2']:.1f} m² / {b['floor_area_m2'] * 10.7639:.0f} ft²",
            f"Rooms {b['room_count']}  Walls {b['wall_count']}  Doors {b['door_count']}  Windows {b['window_count']}"]
    for t in info:
        d2.text((x0, y), t, fill=(30, 30, 30), font=f2); y += 20 * scale // 2
    y += 8
    for lab, col in (("Exterior wall", WALL_EXT), ("Interior wall", WALL_INT), ("Door", DOOR), ("Window", WINDOW)):
        d2.rectangle([x0, y, x0 + 26, y + 14 * scale // 2], fill=col); d2.text((x0 + 36, y - 2), lab, fill=(0, 0, 0), font=f2); y += 20 * scale // 2
    seen = []
    for r in res["rooms"]:
        if r["type"] not in seen:
            seen.append(r["type"])
    for t in seen:
        d2.rectangle([x0, y, x0 + 26, y + 14 * scale // 2], fill=ROOM_COLORS.get(t, (220, 220, 220)), outline=(90, 90, 90))
        d2.text((x0 + 36, y - 2), t, fill=(0, 0, 0), font=f2); y += 20 * scale // 2
    ann = [a for a in res["scale"].get("annotations", [])]
    if ann:
        y += 8; d2.text((x0, y), "Dimension check (annotated -> measured)", fill=(0, 0, 0), font=f3); y += 18 * scale // 2
        for a in ann:
            d2.text((x0, y), f"{a['text']:>8}  {a['annotated_ft_in']:>8} -> {a['measured_m'] / 0.3048:6.2f} ft  ({a['error_pct']:+.1f}%)",
                    fill=(30, 30, 30) if a["used"] else (160, 160, 160), font=f3); y += 16 * scale // 2
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
