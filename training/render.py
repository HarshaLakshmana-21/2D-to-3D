import os
"""Synthetic floor-plan renderer: turns ResPlan vector plans (CC BY 4.0) into
architectural-drawing style raster images + pixel-perfect label masks.

Classes:
 0 background/outside, 1 wall, 2 door, 3 window, 4 living, 5 kitchen,
 6 bedroom, 7 bathroom, 8 balcony, 9 storage, 10 stair
"""
import glob
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely import affinity

CLASSES = ["background", "wall", "door", "window", "living", "kitchen",
           "bedroom", "bathroom", "balcony", "storage", "stair"]
ROOM_KEYS = {"living": 4, "kitchen": 5, "bedroom": 6, "bathroom": 7,
             "balcony": 8, "storage": 9, "stair": 10}

FONTS = [f for f in glob.glob("/usr/share/fonts/truetype/**/*.ttf", recursive=True)
         if any(k in f for k in ["DejaVuSans.ttf", "DejaVuSans-Bold", "LiberationSans-R",
                                 "LiberationSans-B", "FreeSans.ttf", "Arial", "Carlito",
                                 "LiberationSansNarrow-R", "Poppins-Regular", "DejaVuSansCondensed.ttf"])]
if not FONTS:
    FONTS = glob.glob("/usr/share/fonts/truetype/dejavu/*.ttf")

NAMES = {
    "living": ["Living Room", "Living", "LIVING", "Lounge", "Living Room and Kitchen", "Family Room",
               "Great Room", "Living / Dining", "Dining", "LIVING ROOM", "Hall", "Hallway", "Corridor"],
    "kitchen": ["Kitchen", "KITCHEN", "Kit.", "Kitchenette", "Kitchen/Dining"],
    "bedroom": ["Bedroom", "BEDROOM", "Master Bedroom", "Bedroom 2", "Bed", "BR 1", "Guest Room", "Bedroom 3", "M. Bedroom"],
    "bathroom": ["Bathroom", "Bath", "WC", "Toilet", "BATH", "Ensuite", "Powder", "Shower", "W.C."],
    "balcony": ["Balcony", "BALCONY", "Terrace", "Deck", "Patio"],
    "storage": ["Storage", "Closet", "Store", "Utility", "Laundry", "Pantry", "W.I.C.", "Wardrobe"],
    "stair": ["Stairs", "UP", "DN", "Stair"],
}


def geoms(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        out = []
        for q in g.geoms:
            out += geoms(q)
        return out
    return []


def ft_in(m):
    tot = m / 0.0254
    f = int(tot // 12)
    i = int(round(tot - f * 12))
    if i == 12:
        f, i = f + 1, 0
    return f"{f}' {i}\"" if random.random() < 0.7 else f"{f}'-{i}\""


def dim_text(m, unit):
    if unit == "ft":
        return ft_in(m)
    if unit == "m":
        return f"{m:.2f}" if random.random() < 0.5 else f"{m:.2f} m"
    return f"{int(round(m * 1000))}"


class Ctx:
    def __init__(self, s, ox, oy, SS):
        self.s, self.ox, self.oy, self.SS = s, ox, oy, SS

    def pts(self, coords, ss=True):
        a = np.asarray(coords, dtype=np.float64)
        f = self.s if ss else self.s / self.SS
        o = 1.0 if ss else 1.0 / self.SS
        return np.round(np.c_[(a[:, 0] - self.ox) * f, (a[:, 1] - self.oy) * f]).astype(np.int32)


def fill_poly(img, ctx, poly, color, ss=True):
    ext = ctx.pts(poly.exterior.coords, ss)
    holes = [ctx.pts(h.coords, ss) for h in poly.interiors]
    cv2.fillPoly(img, [ext] + holes, color, lineType=cv2.LINE_8)


def outline_poly(img, ctx, poly, color, th):
    cv2.polylines(img, [ctx.pts(poly.exterior.coords)], True, color, th, cv2.LINE_AA)
    for h in poly.interiors:
        cv2.polylines(img, [ctx.pts(h.coords)], True, color, th, cv2.LINE_AA)


def rect_axes(poly):
    """min-rotated-rect: returns center, long axis unit vec, length, short width."""
    r = poly.minimum_rotated_rectangle
    c = list(r.exterior.coords)[:4]
    e1 = np.subtract(c[1], c[0]); e2 = np.subtract(c[2], c[1])
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if l1 >= l2:
        u, L, W = e1 / (l1 + 1e-9), l1, l2
    else:
        u, L, W = e2 / (l2 + 1e-9), l2, l1
    cen = np.mean(c, axis=0)
    return cen, u, L, W


# ---------------- furniture -----------------
def place_box(room, w, h, tries=15):
    inner = room.buffer(-0.2 * max(w, h) * 0.1)
    if inner.is_empty:
        return None
    x0, y0, x1, y1 = inner.bounds
    for _ in range(tries):
        rot = random.random() < 0.5
        ww, hh = (h, w) if rot else (w, h)
        if x1 - x0 < ww or y1 - y0 < hh:
            continue
        x = random.uniform(x0, x1 - ww); y = random.uniform(y0, y1 - hh)
        # snap to a wall sometimes
        if random.random() < 0.6:
            if random.random() < 0.5:
                x = x0 if random.random() < 0.5 else x1 - ww
            else:
                y = y0 if random.random() < 0.5 else y1 - hh
        b = box(x, y, x + ww, y + hh)
        if inner.contains(b):
            return b, rot
    return None


def draw_furniture(img, ctx, room, rtype, upm, col, th):
    """upm = units per metre"""
    n = random.randint(1, 4)
    for _ in range(n):
        kind = random.choice({
            "bedroom": ["bed", "bed", "wardrobe", "desk", "plant"],
            "living": ["sofa", "table", "roundtable", "chair", "plant", "rug", "tv"],
            "kitchen": ["counter", "counter", "table", "roundtable"],
            "bathroom": ["toilet", "sink", "tub", "shower"],
            "balcony": ["roundtable", "plant", "chair"],
            "storage": ["shelf"], "stair": ["stairs"],
        }.get(rtype, ["table"]))
        dims = {"bed": (1.6, 2.0), "wardrobe": (0.6, 1.8), "desk": (0.6, 1.2), "plant": (0.7, 0.7),
                "sofa": (0.9, 2.2), "table": (1.0, 1.8), "roundtable": (1.1, 1.1), "chair": (0.8, 0.8),
                "rug": (1.6, 2.2), "tv": (0.4, 1.6), "counter": (0.6, 2.4), "toilet": (0.45, 0.7),
                "sink": (0.5, 0.6), "tub": (0.75, 1.7), "shower": (0.9, 0.9), "shelf": (0.4, 1.2),
                "stairs": (1.0, 2.6)}[kind]
        w, h = dims[0] * upm * random.uniform(0.8, 1.15), dims[1] * upm * random.uniform(0.8, 1.15)
        r = place_box(room, w, h)
        if r is None:
            continue
        b, rot = r
        x0, y0, x1, y1 = b.bounds
        p0 = ctx.pts([(x0, y0)])[0]; p1 = ctx.pts([(x1, y1)])[0]
        X0, Y0, X1, Y1 = int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])
        W, H = X1 - X0, Y1 - Y0
        cx, cy = (X0 + X1) // 2, (Y0 + Y1) // 2
        L = cv2.LINE_AA
        if kind in ("plant",):
            rr = min(W, H) // 2
            for k in range(random.randint(14, 30)):
                a = random.uniform(0, 2 * math.pi); l = rr * random.uniform(0.4, 1.0)
                cv2.line(img, (cx, cy), (int(cx + l * math.cos(a)), int(cy + l * math.sin(a))), (20, 20, 20), max(1, th), L)
            cv2.circle(img, (cx, cy), max(2, rr // 4), (10, 10, 10), -1, L)
            continue
        if kind in ("roundtable", "toilet", "shower") and kind != "shower":
            if kind == "roundtable":
                rr = min(W, H) // 3
                cv2.circle(img, (cx, cy), rr, col, th, L)
                for k in range(random.choice([0, 2, 3, 4])):
                    a = k * 2 * math.pi / 4 + 0.3
                    cv2.rectangle(img, (int(cx + 1.3 * rr * math.cos(a)) - rr // 2, int(cy + 1.3 * rr * math.sin(a)) - rr // 2),
                                  (int(cx + 1.3 * rr * math.cos(a)) + rr // 2, int(cy + 1.3 * rr * math.sin(a)) + rr // 2), col, th, L)
            else:
                cv2.ellipse(img, (cx, cy), (max(2, W // 2 - 2), max(2, H // 3)), 0, 0, 360, col, th, L)
                cv2.rectangle(img, (X0, Y0), (X1, Y0 + max(2, H // 4)), col, th, L)
            continue
        cv2.rectangle(img, (X0, Y0), (X1, Y1), col, th, L)
        if kind == "bed":
            k = H // 5 if not rot else W // 5
            if not rot:
                cv2.rectangle(img, (X0 + 3, Y0 + 3), (cx - 2, Y0 + k), col, th, L)
                cv2.rectangle(img, (cx + 2, Y0 + 3), (X1 - 3, Y0 + k), col, th, L)
                cv2.line(img, (X0, Y0 + int(1.6 * k)), (X1, Y0 + int(1.6 * k)), col, th, L)
            else:
                cv2.rectangle(img, (X0 + 3, Y0 + 3), (X0 + k, cy - 2), col, th, L)
                cv2.rectangle(img, (X0 + 3, cy + 2), (X0 + k, Y1 - 3), col, th, L)
        elif kind == "sofa":
            n2 = 3
            for k in range(1, n2):
                if W > H:
                    cv2.line(img, (X0 + k * W // n2, Y0), (X0 + k * W // n2, Y1), col, th, L)
                else:
                    cv2.line(img, (X0, Y0 + k * H // n2), (X1, Y0 + k * H // n2), col, th, L)
            cv2.rectangle(img, (X0 + W // 8, Y0 + H // 8), (X1 - W // 8, Y1 - H // 8), col, th, L)
        elif kind == "table":
            s2 = max(4, min(W, H) // 3)
            for k in range(random.choice([2, 3])):
                if H > W:
                    yy = Y0 + (k + 1) * H // 4
                    cv2.rectangle(img, (X0 - s2 - 2, yy - s2 // 2), (X0 - 2, yy + s2 // 2), col, th, L)
                    cv2.rectangle(img, (X1 + 2, yy - s2 // 2), (X1 + s2 + 2, yy + s2 // 2), col, th, L)
                else:
                    xx = X0 + (k + 1) * W // 4
                    cv2.rectangle(img, (xx - s2 // 2, Y0 - s2 - 2), (xx + s2 // 2, Y0 - 2), col, th, L)
                    cv2.rectangle(img, (xx - s2 // 2, Y1 + 2), (xx + s2 // 2, Y1 + s2 + 2), col, th, L)
        elif kind == "counter":
            # stove
            rr = max(2, min(W, H) // 6)
            for dx in (-1, 1):
                for dy in (-1, 1):
                    if W > H:
                        cv2.circle(img, (cx + dx * rr * 2, cy + dy * rr), rr, col, th, L)
                    else:
                        cv2.circle(img, (cx + dx * rr, cy + dy * rr * 2), rr, col, th, L)
            cv2.rectangle(img, (X0 + W // 10, Y0 + H // 10), (X0 + W // 3, Y0 + H // 3), col, th, L)
        elif kind == "tub":
            cv2.rectangle(img, (X0 + 4, Y0 + 4), (X1 - 4, Y1 - 4), col, th, L)
        elif kind == "shower":
            cv2.line(img, (X0, Y0), (X1, Y1), col, th, L); cv2.line(img, (X1, Y0), (X0, Y1), col, th, L)
        elif kind == "sink":
            cv2.ellipse(img, (cx, cy), (max(2, W // 3), max(2, H // 3)), 0, 0, 360, col, th, L)
        elif kind == "stairs":
            n3 = random.randint(8, 14)
            for k in range(1, n3):
                if H > W:
                    cv2.line(img, (X0, Y0 + k * H // n3), (X1, Y0 + k * H // n3), col, th, L)
                else:
                    cv2.line(img, (X0 + k * W // n3, Y0), (X0 + k * W // n3, Y1), col, th, L)


# ---------------- main render -----------------
def prepare(plan):
    """Thin walls / openings consistently, returns dict of geometry lists."""
    wall = unary_union(geoms(plan.get("wall")))
    door = unary_union(geoms(plan.get("door")) + geoms(plan.get("front_door")))
    win = unary_union(geoms(plan.get("window")))
    wd = float(plan.get("wall_depth", 4.0) or 4.0)
    shrink = random.choice([0, 0, random.uniform(0.05, 0.35)]) * wd
    solid = unary_union([wall, door, win])
    if shrink > 0:
        solid2 = solid.buffer(-shrink, join_style=2)
        if solid2.is_empty or solid2.area < 0.3 * solid.area:
            solid2 = solid
    else:
        solid2 = solid
    d2 = solid2.intersection(door.buffer(0.01, join_style=2))
    w2 = solid2.intersection(win.buffer(0.01, join_style=2)).difference(d2)
    wl2 = solid2.difference(door).difference(win)
    return wl2, d2, w2


def render(plan, max_side=None, SS=2, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed % (2 ** 31))
    inner = plan.get("inner")
    net = float(plan.get("area") or 80)
    m_per_unit = math.sqrt(net / max(inner.area, 1e-6)) if inner is not None and not inner.is_empty else 0.05
    upm = 1.0 / m_per_unit
    wall, door, win = prepare(plan)
    rooms = []
    for k, c in ROOM_KEYS.items():
        for g in geoms(plan.get(k)):
            rooms.append((k, c, g))
    allg = unary_union([wall, door, win] + [g for _, _, g in rooms])
    x0, y0, x1, y1 = allg.bounds
    px_per_m = random.uniform(28, 80)
    if max_side:
        px_per_m = min(px_per_m, max_side / ((max(x1 - x0, y1 - y0) * m_per_unit) + 4))
    s_final = px_per_m * m_per_unit  # px per unit at final res
    margin_u = random.uniform(1.0, 3.2) * upm  # dimension-line margin in metres
    ox, oy = x0 - margin_u, y0 - margin_u
    Wf = int(math.ceil((x1 - x0 + 2 * margin_u) * s_final))
    Hf = int(math.ceil((y1 - y0 + 2 * margin_u) * s_final))
    ctx = Ctx(s_final * SS, ox, oy, SS)
    img = np.full((Hf * SS, Wf * SS, 3), 255, np.uint8)
    lab = np.zeros((Hf, Wf), np.uint8)

    # ---- labels (final resolution) ----
    if inner is not None:
        for g in geoms(inner):
            fill_poly(lab, ctx, g, 4, ss=False)
    for k, c, g in sorted(rooms, key=lambda r: -r[2].area):
        fill_poly(lab, ctx, g, c, ss=False)
    for g in geoms(wall):
        fill_poly(lab, ctx, g, 1, ss=False)
    for g in geoms(win):
        fill_poly(lab, ctx, g, 3, ss=False)
    for g in geoms(door):
        fill_poly(lab, ctx, g, 2, ss=False)

    # ---- drawing ----
    th = max(1, int(round(SS * random.choice([1, 1, 1, 1.5, 2]))))
    fcol = tuple([random.randint(0, 110)] * 3)
    colored = random.random() < 0.2
    for k, c, g in rooms:
        if colored and k != "balcony":
            col = tuple(int(v) for v in np.random.randint(200, 256, 3))
            fill_poly(img, ctx, g, col)
        elif random.random() < 0.05:
            # hatch
            sub = np.zeros(img.shape[:2], np.uint8); fill_poly(sub, ctx, g, 255)
            step = random.randint(6, 14) * SS
            for t in range(-img.shape[0], img.shape[1], step):
                cv2.line(img, (t, 0), (t + img.shape[0], img.shape[0]), (215, 215, 215), 1) if False else None
    if random.random() < 0.25:
        grass = random.choice([(225, 240, 225), (235, 235, 235)])
        for g in geoms(plan.get("garden")):
            fill_poly(img, ctx, g, grass)
    # balcony railing
    for k, c, g in rooms:
        if k == "balcony":
            st = random.random()
            if st < 0.5:
                outline_poly(img, ctx, g, (40, 40, 40), th)
                gi = g.buffer(-0.08 * upm, join_style=2)
                for q in geoms(gi):
                    outline_poly(img, ctx, q, (40, 40, 40), th)
            elif st < 0.8:
                outline_poly(img, ctx, g, (30, 30, 30), th)
            else:
                fill_poly(img, ctx, g, (230, 230, 230)); outline_poly(img, ctx, g, (60, 60, 60), th)
    if random.random() < 0.85:
        for k, c, g in rooms:
            if random.random() < 0.85:
                draw_furniture(img, ctx, g, k, upm, fcol, th)

    # walls
    wstyle = random.random()
    wc = random.randint(0, 40)
    for g in geoms(wall):
        if wstyle < 0.65:
            fill_poly(img, ctx, g, (wc, wc, wc))
        elif wstyle < 0.8:
            v = random.randint(70, 140); fill_poly(img, ctx, g, (v, v, v)); outline_poly(img, ctx, g, (0, 0, 0), th)
        elif wstyle < 0.9:
            fill_poly(img, ctx, g, (255, 255, 255)); outline_poly(img, ctx, g, (0, 0, 0), th)
        else:
            sub = np.zeros(img.shape[:2], np.uint8)
            fill_poly(sub, ctx, g, 255)
            hatch = np.full_like(img, 255)
            step = max(4, int(0.12 * upm * ctx.s / upm * m_per_unit * upm)) if False else 5 * SS
            for t in range(-img.shape[0], img.shape[1], step):
                cv2.line(hatch, (t, img.shape[0]), (t + img.shape[0], 0), (0, 0, 0), max(1, SS // 2))
            img[sub > 0] = hatch[sub > 0]
            outline_poly(img, ctx, g, (0, 0, 0), th)
    # windows
    wst = random.random()
    for g in geoms(win):
        fill_poly(img, ctx, g, (255, 255, 255))
        cen, u, L, Wd = rect_axes(g)
        n = np.array([-u[1], u[0]])
        if wst < 0.4:
            outline_poly(img, ctx, g, (60, 60, 60), th)
            p = ctx.pts([cen - u * L / 2, cen + u * L / 2]); cv2.line(img, tuple(p[0]), tuple(p[1]), (60, 60, 60), th, cv2.LINE_AA)
        elif wst < 0.7:
            for off in (-0.25, 0.25):
                p = ctx.pts([cen - u * L / 2 + n * Wd * off, cen + u * L / 2 + n * Wd * off])
                cv2.line(img, tuple(p[0]), tuple(p[1]), (110, 110, 110), th, cv2.LINE_AA)
            for e in (-1, 1):
                p = ctx.pts([cen + e * u * L / 2 - n * Wd / 2, cen + e * u * L / 2 + n * Wd / 2])
                cv2.line(img, tuple(p[0]), tuple(p[1]), (40, 40, 40), th, cv2.LINE_AA)
        elif wst < 0.85:
            fill_poly(img, ctx, g, random.choice([(230, 210, 170), (200, 200, 200), (240, 230, 200)]))
            outline_poly(img, ctx, g, (0, 0, 0), th)
        else:
            outline_poly(img, ctx, g, (0, 0, 0), th)
            for off in (-0.15, 0.15):
                p = ctx.pts([cen - u * L / 2 + n * Wd * off, cen + u * L / 2 + n * Wd * off])
                cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 0, 0), th, cv2.LINE_AA)
    # doors
    for g in geoms(door):
        fill_poly(img, ctx, g, (255, 255, 255))
        cen, u, L, Wd = rect_axes(g)
        n = np.array([-u[1], u[0]])
        st = random.random()
        side = random.choice([-1, 1]); end = random.choice([-1, 1])
        dcol = (random.randint(0, 80),) * 3
        if st < 0.6:
            hinge = cen + end * u * L / 2 + side * n * Wd / 2
            tip = hinge + side * n * L
            p = ctx.pts([hinge, tip]); cv2.line(img, tuple(p[0]), tuple(p[1]), dcol, th, cv2.LINE_AA)
            pts = []
            for a in np.linspace(0, math.pi / 2, 24):
                v = math.cos(a) * (-end * u) + math.sin(a) * (side * n)
                pts.append(hinge + v * L)
            cv2.polylines(img, [ctx.pts(pts)], False, dcol, max(1, th // 2 if random.random() < 0.5 else th), cv2.LINE_AA)
        elif st < 0.72:  # bifold zigzag
            a0 = cen - u * L / 2; zz = [a0]
            for k in range(1, 5):
                zz.append(cen - u * L / 2 + u * L * k / 4 + side * n * (L / 6 if k % 2 else 0))
            cv2.polylines(img, [ctx.pts(zz)], False, dcol, th, cv2.LINE_AA)
        elif st < 0.84:  # sliding
            for off, a, b in ((-0.2, -0.5, 0.1), (0.2, -0.1, 0.5)):
                p = ctx.pts([cen + u * L * a + n * Wd * off, cen + u * L * b + n * Wd * off])
                cv2.line(img, tuple(p[0]), tuple(p[1]), dcol, th, cv2.LINE_AA)
        # else: plain opening
        if random.random() < 0.3:
            for e in (-1, 1):
                p = ctx.pts([cen + e * u * L / 2 - n * Wd / 2, cen + e * u * L / 2 + n * Wd / 2])
                cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 0, 0), th, cv2.LINE_AA)

    # ---- dimension lines ----
    texts = []  # (x,y,text,size,angle) in SS px
    unit = random.choice(["ft", "ft", "m", "mm"])
    font_px = int(random.uniform(0.25, 0.42) * upm * ctx.s / upm * m_per_unit * upm) if False else int(random.uniform(10, 17) * SS * px_per_m / 55)
    font_px = max(8 * SS, min(font_px, 26 * SS))
    if random.random() < 0.75:
        wb = wall.bounds
        xs = sorted(set(round(c[0], 1) for g in geoms(wall) for c in g.exterior.coords))
        ys = sorted(set(round(c[1], 1) for g in geoms(wall) for c in g.exterior.coords))
        for side in random.sample(["top", "bottom", "left", "right"], random.randint(1, 4)):
            vals = xs if side in ("top", "bottom") else ys
            lo, hi = (wb[0], wb[2]) if side in ("top", "bottom") else (wb[1], wb[3])
            k = random.randint(1, 3)
            br = sorted(set([lo, hi] + random.sample(vals, min(k, len(vals)))))
            br = [b for i, b in enumerate(br) if i == 0 or b - br[i - 1] > 1.2 * upm]
            off = random.uniform(0.35, 0.8) * margin_u
            for a, b in zip(br[:-1], br[1:]):
                if side == "top":
                    P, Q = (a, wb[1] - off), (b, wb[1] - off)
                elif side == "bottom":
                    P, Q = (a, wb[3] + off), (b, wb[3] + off)
                elif side == "left":
                    P, Q = (wb[0] - off, a), (wb[0] - off, b)
                else:
                    P, Q = (wb[2] + off, a), (wb[2] + off, b)
                p = ctx.pts([P, Q])
                cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 0, 0), max(1, SS // 2 + 1), cv2.LINE_AA)
                vert = side in ("left", "right")
                tk = 5 * SS
                for q in p:
                    if random.random() < 0.5:
                        cv2.line(img, (int(q[0] - tk), int(q[1] - tk)), (int(q[0] + tk), int(q[1] + tk)), (0, 0, 0), max(1, SS // 2 + 1), cv2.LINE_AA)
                    else:
                        if vert:
                            cv2.line(img, (int(q[0] - tk), int(q[1])), (int(q[0] + tk), int(q[1])), (0, 0, 0), max(1, SS // 2 + 1), cv2.LINE_AA)
                        else:
                            cv2.line(img, (int(q[0]), int(q[1] - tk)), (int(q[0]), int(q[1] + tk)), (0, 0, 0), max(1, SS // 2 + 1), cv2.LINE_AA)
                L = abs(b - a) * m_per_unit
                mid = (p[0] + p[1]) / 2
                t = dim_text(L, unit)
                if vert:
                    texts.append((mid[0] + (-1 if side == "left" else 1) * font_px * 0.9, mid[1], t, int(font_px * 0.9), 90))
                else:
                    texts.append((mid[0], mid[1] - font_px * 0.9, t, int(font_px * 0.9), 0))
    # room names
    if random.random() < 0.85:
        upper = random.random() < 0.3
        for k, c, g in rooms:
            if random.random() < 0.85 and g.area > (1.5 * upm) ** 2:
                nm = random.choice(NAMES[k])
                nm = nm.upper() if upper else nm
                pt = g.representative_point()
                pp = ctx.pts([(pt.x, pt.y)])[0]
                texts.append((pp[0], pp[1], nm, font_px, 0))
                if random.random() < 0.35:
                    bb = g.bounds
                    a = (bb[2] - bb[0]) * m_per_unit; b = (bb[3] - bb[1]) * m_per_unit
                    sub = (f"{a:.1f} x {b:.1f} m" if unit != "ft" else f"{ft_in(a)} x {ft_in(b)}") if random.random() < 0.6 else f"{g.area * m_per_unit ** 2:.1f} m²"
                    texts.append((pp[0], pp[1] + font_px * 1.3, sub, int(font_px * 0.8), 0))
    if texts:
        pil = Image.fromarray(img)
        fontfile = random.choice(FONTS)
        dr = ImageDraw.Draw(pil)
        for (x, y, t, sz, ang) in texts:
            f = ImageFont.truetype(fontfile, max(8, sz))
            l, tp, r, bt = f.getbbox(t)
            if ang == 0:
                dr.text((x - (r - l) / 2, y - (bt - tp) / 2 - tp), t, fill=(0, 0, 0), font=f)
            else:
                ti = Image.new("L", (r - l + 4, bt + 4), 0)
                ImageDraw.Draw(ti).text((2 - l, 2), t, fill=255, font=f)
                ti = ti.rotate(90, expand=True)
                X, Y = int(x - ti.width / 2), int(y - ti.height / 2)
                blk = Image.new("RGB", ti.size, (0, 0, 0))
                pil.paste(blk, (X, Y), ti)
        img = np.array(pil)

    img = cv2.resize(img, (Wf, Hf), interpolation=cv2.INTER_AREA)
    meta = {"px_per_m": px_per_m, "m_per_unit": m_per_unit}
    return img, lab, meta


if __name__ == "__main__":
    import pickle, sys, os
    d = pickle.load(open(os.path.join(os.environ.get("RESPLAN_DIR", "ResPlan"), "ResPlan.pkl"), "rb"))
    os.makedirs("preview", exist_ok=True)
    for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4):
        im, lb, m = render(d[i], seed=i)
        pal = np.array([[255, 255, 255], [0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 220, 180], [255, 160, 60],
                        [180, 255, 180], [180, 200, 255], [210, 210, 210], [200, 200, 120], [200, 120, 220]], np.uint8)
        cv2.imwrite(f"preview/{i}_img.png", im[..., ::-1])
        cv2.imwrite(f"preview/{i}_lab.png", pal[lb][..., ::-1])
        print(i, im.shape, m)
