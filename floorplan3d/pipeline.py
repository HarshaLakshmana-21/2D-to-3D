"""End-to-end: image -> segmentation + OCR + calibration -> vector model -> outputs."""
import json
import math
import os
import time

import cv2
import numpy as np

from . import __version__
from .segment import segment, default_model_path
from .ocr import read_text, room_type_from_text
from .dimensions import calibrate
from .vectorize import (clean_masks, wall_thickness_px, extract_walls, extract_openings,
                        virtual_walls, extract_rooms, model_room_type, polygon_from_mask,
                        filter_struct, find_wall_gaps, drop_stubs, raster_segments, join_small_gaps, corner_cuts, split_region, closed_region_with_lines)

DEFAULTS = dict(wall_height_m=2.75, door_height_m=2.10, window_sill_m=0.90, window_head_m=2.10,
                floor_thickness_m=0.05, fallback_px_per_m=None)

ROOM_DISPLAY = {"living_kitchen": "Living + Kitchen", "living": "Living Room", "kitchen": "Kitchen",
                "bedroom": "Bedroom", "master_bedroom": "Master Bedroom", "bathroom": "Bathroom",
                "balcony": "Balcony", "storage": "Storage", "stair": "Stairs", "hallway": "Hallway",
                "dining": "Dining", "office": "Office / Study", "garage": "Garage",
                "dressing": "Dressing", "pooja": "Pooja Room", "room": "Room"}


def ft_in(m):
    tot = m / 0.0254
    f = int(tot // 12); i = int(round(tot - 12 * f))
    if i == 12:
        f, i = f + 1, 0
    return f"{f}' {i}\""


def _r(v, n=3):
    return float(round(float(v), n))


def analyze(image_path, out_dir=None, model_path=None, use_ocr=True, scale_px_per_m=None,
            wall_height_m=None, units="both", make_3d=True, verbose=True, tta=True):
    t0 = time.time()
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    cfg = dict(DEFAULTS)
    if wall_height_m:
        cfg["wall_height_m"] = float(wall_height_m)

    # 1. OCR + scale calibration from dimension strings
    texts = read_text(img) if use_ocr else []
    cal = _calibrate_multires(gray, texts) if texts else dict(dims=[], method="none")
    known = scale_px_per_m or (np.mean([cal["px_per_m_x"], cal["px_per_m_y"]]) if cal.get("method") == "dimension_annotations" else None)

    # 2. segmentation (run at the resolution the model was trained for, ~55 px/m, when scale is known)
    f = 1.0
    if known and not (32 <= known <= 85):
        f = float(np.clip(55.0 / known, 0.25, 4.0))
    if abs(f - 1.0) > 0.05:
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        ps = segment(small, model_path, tta=tta)
        prob = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in ps])
    else:
        prob = segment(img, model_path, tta=tta)
    cls, wall, door, win = clean_masks(prob, gray)
    t_px = wall_thickness_px(wall)

    # 3. scale
    if scale_px_per_m:
        sx = sy = float(scale_px_per_m); method = "user_supplied"
    elif cal["method"] == "dimension_annotations":
        sx, sy = cal["px_per_m_x"], cal["px_per_m_y"]; method = cal["method"]
    else:
        # fallback: typical interior door leaf ~0.85 m clear opening
        dm = []
        n, lab, st, _ = cv2.connectedComponentsWithStats(door.astype(np.uint8), 8)
        for i in range(1, n):
            dm.append(max(st[i, 2], st[i, 3]))
        if dm:
            sx = sy = float(np.median(dm)) / 0.85; method = "estimated_from_door_widths (verify!)"
        else:
            sx = sy = t_px / 0.15; method = "estimated_from_wall_thickness (verify!)"

    # 4. vectorize
    spm = (sx + sy) / 2
    struct = wall | door | win
    struct = cv2.morphologyEx(struct.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    struct = filter_struct(struct, spm)
    wall &= struct; door &= struct; win &= struct
    segs = extract_walls(struct, t_px)
    # openings that the model missed: gaps between collinear wall ends (e.g. bi-fold / sliding doors)
    segs = drop_stubs(segs, spm, cls)
    inferred = np.zeros_like(door); bridged = np.zeros_like(door)
    for g in find_wall_gaps(segs, struct, spm):
        if g["gap_px"] < 0.45 * spm:      # tiny break (ink / furniture overlap) -> continuous wall
            bridged[g["y0"]:g["y1"], g["x0"]:g["x1"]] = True
        else:                              # door-sized gap -> opening
            inferred[g["y0"]:g["y1"], g["x0"]:g["x1"]] = True
    if inferred.any() or bridged.any():
        door = door | inferred; struct = struct | inferred | bridged
        segs = drop_stubs(extract_walls(struct, t_px), spm, cls)
    segs = join_small_gaps(segs, 0.45 * spm)
    # keep only structure that belongs to a vector wall (removes plants / furniture blobs)
    struct = (struct & raster_segments(segs, struct.shape, grow=2)) | raster_segments(segs, struct.shape)
    door &= struct; win &= struct
    doors = [o for o in extract_openings(door, segs, "door") if o["wall"] is not None and (o["end_px"] - o["start_px"]) >= 0.3 * spm]
    windows = [o for o in extract_openings(win, segs, "window") if o["wall"] is not None and (o["end_px"] - o["start_px"]) >= 0.25 * spm]
    for o in doors:
        x, y, w, h = o["bbox"]
        o["inferred_from_wall_gap"] = bool(inferred[y:y + h, x:x + w].mean() > 0.5)
    vwalls = virtual_walls(segs, max_gap_px=1.6 * spm)
    rooms_raw, roomlab, vmask = extract_rooms(struct, cls, vwalls, min_area_px=0.9 * sx * sy)

    # 5. room labelling (OCR first, model second)
    name_texts = [t for t in texts if t.get("dim") is None and any(ch.isalpha() for ch in t["text"]) and len(t["text"].strip()) >= 2]

    def assign(rooms_raw, roomlab):
        for r in rooms_raw:
            r["ocr"] = []
        for t in name_texts:
            cx, cy = int(round(t["center"][0])), int(round(t["center"][1]))
            cx = min(max(cx, 0), W - 1); cy = min(max(cy, 0), H - 1)
            lab_id = roomlab[cy, cx]
            if lab_id == 0:  # text sits on a line: look around
                ys, xs = np.mgrid[max(0, cy - 8):min(H, cy + 9), max(0, cx - 8):min(W, cx + 9)]
                ids = roomlab[ys, xs].ravel(); ids = ids[ids > 0]
                lab_id = int(np.bincount(ids).argmax()) if ids.size else 0
            for r in rooms_raw:
                if r["label"] == lab_id:
                    r["ocr"].append(t)
    assign(rooms_raw, roomlab)

    # merge regions split only by a virtual wall when that split is not supported by labels
    _merge_unsupported_splits(rooms_raw, roomlab, vmask, cls)

    # split open-plan regions that carry several different room names (e.g. living | hallway)
    cuts_drawn = []
    new_rooms = []
    next_id = int(roomlab.max()) + 1
    for r in rooms_raw:
        typed = [t for t in r["ocr"] if room_type_from_text(t["text"])]
        if r["exterior"] or len({room_type_from_text(t["text"]) for t in typed}) < 2:
            new_rooms.append(r); continue
        pts = [(min(max(t["center"][0], 0), W - 1), min(max(t["center"][1], 0), H - 1)) for t in typed]
        cuts = corner_cuts(segs, r["mask"], spm)
        lab2, ids, applied = split_region(r["mask"], pts, cuts, 1.0 * sx * sy)
        if not applied:
            new_rooms.append(r); continue
        cuts_drawn += applied
        for k in np.unique(lab2):
            if k == 0:
                continue
            m = (lab2 == k) & r["mask"]
            if m.sum() < 0.9 * sx * sy:
                continue
            roomlab[m] = next_id
            new_rooms.append(dict(label=next_id, mask=m, exterior=False, area_px=int(m.sum()), ocr=[]))
            next_id += 1
    rooms_raw = new_rooms
    assign(rooms_raw, roomlab)

    rooms = []
    for r in rooms_raw:
        mtype, mconf = model_room_type(cls, r["mask"])
        otype = None; oname = None
        for t in r["ocr"]:
            ty = room_type_from_text(t["text"])
            if ty:
                otype, oname = ty, t["text"].strip(); break
        if r["exterior"]:
            # outside the building: keep only balconies / terraces (OCR label or model)
            m2 = None
            seed = None
            if otype == "balcony":
                t = next(t for t in r["ocr"] if room_type_from_text(t["text"]) == "balcony")
                bx0, by0 = t["box"].min(0); bx1, by1 = t["box"].max(0); cx_, cy_ = t["center"]
                seeds = [(cx_, by0 - 4), (cx_, by1 + 4), (bx0 - 4, cy_), (bx1 + 4, cy_), (cx_, by0 - 12), (cx_, by1 + 12)]
            else:
                bal = (cls == 8) & r["mask"]
                if bal.sum() > 1.2 * sx * sy:
                    ys, xs = np.where(bal); seed = (np.median(xs), np.median(ys))
                    seeds = [seed]
            if seed is None and otype != "balcony":
                continue
            m2 = None
            for sd in seeds:
                c = closed_region_with_lines(gray, struct, sd, max_area_px=40 * sx * sy)
                if c is not None and c.sum() > 1.0 * sx * sy and (m2 is None or c.sum() > m2.sum()):
                    m2 = c
            if m2 is None:   # railing not closed: fall back to the model's balcony pixels
                bal = (cls == 8) & r["mask"]
                if bal.sum() < 0.8 * sx * sy:
                    continue
                m2 = cv2.morphologyEx(bal.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
            # balconies are convex in practice: fill notches cut by furniture / plants / door leaves
            cs, _ = cv2.findContours(m2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                hull = np.zeros(m2.shape, np.uint8)
                cv2.fillPoly(hull, [cv2.convexHull(max(cs, key=cv2.contourArea))], 1)
                m2 = (hull > 0) & ~struct
            r = dict(r, mask=m2, area_px=int(m2.sum()))
            otype = otype or "balcony"
        rtype = otype or mtype or "room"
        if rtype == "living" and not otype and r["area_px"] / (sx * sy) < 6 and _elongated(r["mask"]):
            rtype = "hallway"
        src = "ocr_label" if otype else ("segmentation_model" if mtype else "default")
        conf = 0.95 if otype else round(mconf, 2)
        rooms.append(dict(mask=r["mask"], type=rtype, name=oname or ROOM_DISPLAY.get(rtype, rtype.title()),
                          source=src, confidence=conf, model_type=mtype))

    # 6. geometry in metres
    allx = [min(s["x1"], s["x2"]) + 0.5 - s["thickness"] / 2 for s in segs] + [max(s["x1"], s["x2"]) + 0.5 + s["thickness"] / 2 for s in segs]
    ally = [min(s["y1"], s["y2"]) + 0.5 - s["thickness"] / 2 for s in segs] + [max(s["y1"], s["y2"]) + 0.5 + s["thickness"] / 2 for s in segs]
    for r in rooms:
        ys, xs = np.where(r["mask"]); allx += [xs.min(), xs.max() + 1]; ally += [ys.min(), ys.max() + 1]
    ox, oy = (min(allx), min(ally)) if allx else (0.0, 0.0)
    X = lambda px: (px - ox) / sx
    Y = lambda py: (py - oy) / sy

    out_walls = []
    for k, s in enumerate(segs):
        t_m = s["thickness"] / (sx if s["orientation"] == "v" else sy)
        fe = s.get("free_end", {})
        # pixel indices -> continuous coords (pixel i spans [i, i+1)); free ends sit on the pixel edge
        if s["orientation"] == "h":
            y1 = y2 = s["y1"] + 0.5
            x1 = s["x1"] if fe.get(1, True) else s["x1"] + 0.5
            x2 = s["x2"] + 1 if fe.get(2, True) else s["x2"] + 0.5
        else:
            x1 = x2 = s["x1"] + 0.5
            y1 = s["y1"] if fe.get(1, True) else s["y1"] + 0.5
            y2 = s["y2"] + 1 if fe.get(2, True) else s["y2"] + 0.5
        L = math.hypot(X(x2) - X(x1), Y(y2) - Y(y1))
        ext = _is_exterior(s, roomlab, rooms_raw, H, W)
        out_walls.append(dict(id=f"W{k + 1}", start=[_r(X(x1)), _r(Y(y1))], end=[_r(X(x2)), _r(Y(y2))],
                              length_m=_r(L), length_ft_in=ft_in(L), thickness_m=_r(t_m), height_m=cfg["wall_height_m"],
                              orientation="horizontal" if s["orientation"] == "h" else "vertical",
                              type="exterior" if ext else "interior", openings=[],
                              _px=dict(x1=s["x1"], y1=s["y1"], x2=s["x2"], y2=s["y2"], t=s["thickness"])))
    out_open = []
    for kind, items in (("door", doors), ("window", windows)):
        for i, o in enumerate(items):
            s_px = sx if o["orientation"] == "h" else sy
            width = (o["end_px"] - o["start_px"]) / s_px
            cx, cy = o["center"]
            rec = dict(id=f"{'D' if kind == 'door' else 'N'}{i + 1}", type=kind, wall_id=None,
                       center=[_r(X(cx)), _r(Y(cy))], width_m=_r(width), width_ft_in=ft_in(width),
                       height_m=cfg["door_height_m"] if kind == "door" else _r(cfg["window_head_m"] - cfg["window_sill_m"]),
                       sill_height_m=0.0 if kind == "door" else cfg["window_sill_m"],
                       orientation="horizontal" if o["orientation"] == "h" else "vertical",
                       _px=dict(bbox=o["bbox"], a=o["start_px"], b=o["end_px"]))
            if kind == "door":
                rec["inferred_from_wall_gap"] = o.get("inferred_from_wall_gap", False)
            if o["wall"] is not None:
                w = out_walls[o["wall"]]
                rec["wall_id"] = w["id"]
                ws = w["start"]
                a = X(o["start_px"]) if o["orientation"] == "h" else Y(o["start_px"])
                base = ws[0] if o["orientation"] == "h" else ws[1]
                rec["offset_from_wall_start_m"] = _r(a - base)
                w["openings"].append(rec["id"])
            out_open.append(rec)

    out_rooms = []
    eps = max(1.5, 0.35 * t_px)
    cnt = {}
    for r in rooms:
        poly = polygon_from_mask(r["mask"], eps)
        if poly is None or len(poly) < 3:
            continue
        # contour runs through pixel centres; expand by half a pixel to pixel edges
        c = poly.mean(0)
        poly = poly + 0.5 + np.sign(poly - c) * 0.5
        pm = [[_r(X(p[0])), _r(Y(p[1]))] for p in poly]
        area = float(r["mask"].sum()) / (sx * sy)
        xs = [p[0] for p in pm]; ys = [p[1] for p in pm]
        wdt, dpt = max(xs) - min(xs), max(ys) - min(ys)
        per = sum(math.hypot(pm[i][0] - pm[i - 1][0], pm[i][1] - pm[i - 1][1]) for i in range(len(pm)))
        cnt[r["type"]] = cnt.get(r["type"], 0) + 1
        rid = f"R{len(out_rooms) + 1}"
        ys_, xs_ = np.where(r["mask"])
        out_rooms.append(dict(id=rid, name=r["name"], type=r["type"], label_source=r["source"], confidence=r["confidence"],
                              polygon=pm, area_m2=_r(area, 2), area_ft2=_r(area * 10.7639, 1),
                              width_m=_r(wdt), depth_m=_r(dpt), width_ft_in=ft_in(wdt), depth_ft_in=ft_in(dpt),
                              perimeter_m=_r(per), centroid=[_r(X(xs_.mean() + 0.5)), _r(Y(ys_.mean() + 0.5))],
                              floor_elevation_m=0.0, ceiling_height_m=cfg["wall_height_m"],
                              doors=[], _mask=r["mask"]))
    # door -> rooms connectivity
    for o in out_open:
        if o["type"] != "door":
            continue
        x, y, w, h = o["_px"]["bbox"]
        pad = int(max(4, 1.2 * t_px))
        ys, xs = np.mgrid[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        o["connects"] = []
        for rm in out_rooms:
            if rm["_mask"][ys, xs].any():
                rm["doors"].append(o["id"]); o["connects"].append(rm["id"])
        if len(o["connects"]) == 1:
            o["connects"].append("exterior")

    bw = (max(allx) - ox) / sx if allx else 0; bd = (max(ally) - oy) / sy if ally else 0
    result = dict(
        schema="floorplan3d/v1", generator=f"floorplan3d {__version__}", source_image=os.path.basename(image_path),
        units=dict(length="m", area="m2", note="feet-inch strings provided for convenience"),
        coordinate_system=dict(origin="top-left corner of the building footprint", x="right (plan) -> 3D +X",
                               y="down (plan) -> 3D +Z", up="3D +Y (heights)"),
        image=dict(width_px=W, height_px=H, origin_px=[_r(ox, 2), _r(oy, 2)]),
        scale=dict(px_per_m_x=_r(sx, 4), px_per_m_y=_r(sy, 4), px_per_ft_x=_r(sx * 0.3048, 4), px_per_ft_y=_r(sy * 0.3048, 4),
                   method=method, annotations=[dict(text=d["text"], axis=d["axis"], annotated_m=_r(d["metres"]),
                                                    annotated_ft_in=ft_in(d["metres"]), measured_m=_r(d.get("measured_m", 0)),
                                                    error_pct=_r(d.get("error_pct", 0), 2), used=d.get("used", False))
                                               for d in cal.get("dims", [])]),
        defaults=cfg,
        building=dict(width_m=_r(bw), depth_m=_r(bd), width_ft_in=ft_in(bw), depth_ft_in=ft_in(bd),
                      floor_area_m2=_r(sum(r["area_m2"] for r in out_rooms if r["type"] != "balcony"), 2),
                      wall_count=len(out_walls), door_count=sum(o["type"] == "door" for o in out_open),
                      window_count=sum(o["type"] == "window" for o in out_open), room_count=len(out_rooms)),
        walls=out_walls, openings=out_open, rooms=out_rooms,
        texts=[dict(text=t["text"], score=_r(t["score"], 2), vertical=t["vertical"], center_px=[_r(t["center"][0], 1), _r(t["center"][1], 1)],
                    dimension_m=_r(t["dim"]) if t.get("dim") else None) for t in texts],
    )
    if method.startswith("estimated"):
        result["warning"] = ("No readable dimension strings found - scale was estimated (" + method +
                             "). Pass --scale-px-per-ft or --scale-px-per-m for exact measurements.")
    elif cal.get("warning"):
        result["warning"] = cal["warning"]
    result["processing_seconds"] = _r(time.time() - t0, 2)
    outputs = {}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        from .visualize import draw_detection
        det = draw_detection(img, result, prob.argmax(0))
        p_img = os.path.join(out_dir, f"{stem}_detected.png")
        cv2.imencode(".png", det)[1].tofile(p_img)
        outputs["detected_image"] = p_img
        clean = _strip_private(result)
        p_json = os.path.join(out_dir, f"{stem}_3d.json")
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        outputs["json_3d"] = p_json
        if make_3d:
            from .export3d import build_scene
            outputs.update(build_scene(clean, out_dir, stem))
        result["outputs"] = outputs
    if verbose:
        b = result["building"]
        print(f"[floorplan3d] {os.path.basename(image_path)}: {b['room_count']} rooms, {b['wall_count']} walls, "
              f"{b['door_count']} doors, {b['window_count']} windows | scale {sx * 0.3048:.2f}px/ft ({method}) | "
              f"{result['processing_seconds']}s")
    return result


def _calibrate_multires(gray, texts):
    """Dimension-line calibration; small drawings are analysed at 2x so thin
    lines / arrowheads survive. The more consistent result wins."""
    import copy
    best = None
    ups = [1.0, 2.0] if max(gray.shape) < 1400 else [1.0]
    for up in ups:
        g = gray if up == 1 else cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        tt = copy.deepcopy(texts)
        for t in tt:
            t["box"] = t["box"] * up; t["center"] = t["center"] * up; t["w"] *= up; t["h"] *= up
        r = calibrate(g, tt)
        if r.get("method") != "dimension_annotations":
            continue
        for k in ("px_per_m_x", "px_per_m_y"):
            r[k] /= up
        for d in r["dims"]:
            for k in ("p0", "p1", "px", "line_pos", "px_per_m"):
                d[k] /= up
        n_used = sum(d.get("used", False) for d in r["dims"])
        if best is None or n_used > best[0]:
            best = (n_used, r)
    return best[1] if best else dict(dims=[], method="none")


def _strip_private(obj):
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def _elongated(mask):
    ys, xs = np.where(mask)
    w, h = np.ptp(xs) + 1, np.ptp(ys) + 1
    return max(w, h) / max(1, min(w, h)) > 2.2


def _is_exterior(s, roomlab, rooms_raw, H, W):
    """A wall is exterior when (most of) one of its sides faces the outside region."""
    ext_ids = {r["label"] for r in rooms_raw if r["exterior"]}
    off = s["thickness"] / 2 + 3
    hits = 0; n = 0
    for f in np.linspace(0.15, 0.85, 9):
        x = s["x1"] + f * (s["x2"] - s["x1"]); y = s["y1"] + f * (s["y2"] - s["y1"])
        pts = [(x, y - off), (x, y + off)] if s["orientation"] == "h" else [(x - off, y), (x + off, y)]
        hit = False
        for (px, py) in pts:
            xi, yi = int(round(px)), int(round(py))
            if not (0 <= xi < W and 0 <= yi < H) or roomlab[yi, xi] in ext_ids:
                hit = True
        hits += hit; n += 1
    return hits / n >= 0.4


def _merge_unsupported_splits(rooms_raw, roomlab, vmask, cls):
    """Two regions separated only by a virtual wall are merged unless the split
    is supported (each side carries its own text label, or the model sees
    different room types with confidence)."""
    if vmask.sum() == 0:
        return
    by = {r["label"]: r for r in rooms_raw}
    n, vl = cv2.connectedComponents(vmask, 8)
    parent = {r["label"]: r["label"] for r in rooms_raw}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(1, n):
        m = cv2.dilate((vl == i).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ids = [x for x in np.unique(roomlab[m]) if x in by]
        if len(ids) != 2:
            continue
        a, b = by[ids[0]], by[ids[1]]
        if a["exterior"] or b["exterior"]:
            continue
        ta = {room_type_from_text(t["text"]) for t in a["ocr"]} - {None}
        tb = {room_type_from_text(t["text"]) for t in b["ocr"]} - {None}
        if ta and tb:
            continue  # both labelled -> keep split
        ma, ca = model_room_type(cls, a["mask"]); mb, cb = model_room_type(cls, b["mask"])
        if ma != mb and ca > 0.6 and cb > 0.6 and not (ta or tb):
            continue
        if (ta and not tb and mb is not None and cb > 0.7 and mb not in ("living",) and mb != ma):
            continue
        parent[find(a["label"])] = find(b["label"])
    groups = {}
    for r in rooms_raw:
        groups.setdefault(find(r["label"]), []).append(r)
    merged = []
    for g in groups.values():
        if len(g) == 1:
            merged.append(g[0]); continue
        base = dict(g[0]); m = np.zeros_like(g[0]["mask"])
        for r in g:
            m |= r["mask"]
        # include the virtual wall pixels between them
        m2 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        m = m | (m2 & (vmask > 0))
        base.update(mask=m, area_px=int(m.sum()), ocr=sum((r["ocr"] for r in g), []), exterior=any(r["exterior"] for r in g))
        merged.append(base)
    rooms_raw[:] = merged
