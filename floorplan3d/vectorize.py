"""Turn segmentation masks into vector geometry: wall centre-lines with
thickness, door / window openings attached to walls, and room polygons."""
import numpy as np
import cv2

ROOM_CLASS_NAMES = {4: "living", 5: "kitchen", 6: "bedroom", 7: "bathroom",
                    8: "balcony", 9: "storage", 10: "stair"}


def _remove_small(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def _run_lengths(mask, axis):
    """Length of the run of True pixels each pixel belongs to, along axis (1=horizontal)."""
    m = mask if axis == 1 else mask.T
    out = np.zeros(m.shape, np.int32)
    for i in range(m.shape[0]):
        row = m[i]
        if not row.any():
            continue
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
        for a, b in zip(s, e):
            out[i, a:b] = b - a
    return out if axis == 1 else out.T


def clean_masks(prob, gray):
    cls = prob.argmax(0)
    wall = cls == 1; door = cls == 2; win = cls == 3
    # refine solid walls with the actual dark ink (sharper, pixel-accurate edges)
    dark = gray < 110
    near = cv2.dilate(wall.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    solid = dark & near
    solid = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    wall = (wall | solid) & ~door & ~win
    wall = _remove_small(wall, 30)
    door = _remove_small(door, 15)
    win = _remove_small(win, 15)
    return cls, wall, door, win


def wall_thickness_px(wall):
    h = _run_lengths(wall, 1); v = _run_lengths(wall, 0)
    t = np.minimum(h, v)[wall]
    return float(np.median(t)) if t.size else 6.0


def extract_walls(struct, t_med):
    """Manhattan decomposition of the structure mask into wall segments.
    Returns list of dict(x1,y1,x2,y2,thickness,orientation) in pixels (centre-lines)."""
    tmax = max(4.0, 2.6 * t_med)
    h = _run_lengths(struct, 1); v = _run_lengths(struct, 0)
    hm = struct & (v <= tmax) & (h > v)
    vm = struct & (h <= tmax) & (v > h)
    segs = []
    for mask, orient in ((hm, "h"), (vm, "v")):
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, hh, a = st[i]
            ys, xs = np.where(lab == i)
            if orient == "h":
                t = float(np.median(v[ys, xs]))
                L = w
                if L < max(1.2 * t, 4):
                    continue
                yc = float(np.median(ys)) + 0.5 if t % 2 == 0 else float(np.median(ys))
                # centre of the vertical run is more accurate:
                yc = float(np.mean([ys.min(), ys.max()])) if hh <= tmax + 1 else yc
                segs.append(dict(x1=float(x), y1=yc, x2=float(x + w - 1), y2=yc, thickness=t, orientation="h"))
            else:
                t = float(np.median(h[ys, xs]))
                L = hh
                if L < max(1.2 * t, 4):
                    continue
                xc = float(np.mean([xs.min(), xs.max()])) if w <= tmax + 1 else float(np.median(xs))
                segs.append(dict(x1=xc, y1=float(y), x2=xc, y2=float(y + hh - 1), thickness=t, orientation="v"))
    _snap_junctions(segs)
    segs = _merge_collinear(segs)
    _snap_junctions(segs)
    segs = _split_at_junctions(segs)
    return segs


def _split_at_junctions(segs, min_piece=3.0):
    """Split walls where another wall meets them (T / X junctions) so every
    wall piece separates exactly two spaces (needed for exterior/interior typing)."""
    out = []
    for s in segs:
        cuts = []
        for o in segs:
            if o is s or o["orientation"] == s["orientation"]:
                continue
            fe = o.get("free_end", {})
            for e in (1, 2):
                if fe.get(e, True):
                    continue
                if s["orientation"] == "h":
                    if abs(o["y%d" % e] - s["y1"]) < 0.5 and s["x1"] + min_piece < o["x1"] < s["x2"] - min_piece:
                        cuts.append(o["x1"])
                else:
                    if abs(o["x%d" % e] - s["x1"]) < 0.5 and s["y1"] + min_piece < o["y1"] < s["y2"] - min_piece:
                        cuts.append(o["y1"])
        if not cuts:
            out.append(s); continue
        pts = [s["x1"] if s["orientation"] == "h" else s["y1"]] + sorted(set(cuts)) + [s["x2"] if s["orientation"] == "h" else s["y2"]]
        for i in range(len(pts) - 1):
            q = dict(s); fe = dict(s.get("free_end", {}))
            q["free_end"] = {1: fe.get(1, True) if i == 0 else False, 2: fe.get(2, True) if i == len(pts) - 2 else False}
            if s["orientation"] == "h":
                q["x1"], q["x2"] = pts[i], pts[i + 1]
            else:
                q["y1"], q["y2"] = pts[i], pts[i + 1]
            out.append(q)
    return out


def _snap_junctions(segs):
    """Extend wall ends to the centre-line of a perpendicular wall they touch."""
    for s in segs:
        for end in (1, 2):
            best = None
            for o in segs:
                if o is s or o["orientation"] == s["orientation"]:
                    continue
                tol = o["thickness"] / 2 + s["thickness"] + 3
                if s["orientation"] == "h":
                    ex, ey = s["x%d" % end], s["y1"]
                    lo, hi = min(o["y1"], o["y2"]) - s["thickness"], max(o["y1"], o["y2"]) + s["thickness"]
                    d = abs(o["x1"] - ex)
                    if d <= tol and lo <= ey <= hi and (best is None or d < best[0]):
                        best = (d, o["x1"])
                else:
                    ex, ey = s["x1"], s["y%d" % end]
                    lo, hi = min(o["x1"], o["x2"]) - s["thickness"], max(o["x1"], o["x2"]) + s["thickness"]
                    d = abs(o["y1"] - ey)
                    if d <= tol and lo <= ex <= hi and (best is None or d < best[0]):
                        best = (d, o["y1"])
            if best is not None:
                if s["orientation"] == "h":
                    s["x%d" % end] = best[1]
                else:
                    s["y%d" % end] = best[1]
            s.setdefault("free_end", {})[end] = best is None


def _merge_collinear(segs, gap=2):
    out = []
    for orient in ("h", "v"):
        ss = [s for s in segs if s["orientation"] == orient]
        key = (lambda s: (s["y1"], s["x1"])) if orient == "h" else (lambda s: (s["x1"], s["y1"]))
        ss.sort(key=key)
        used = [False] * len(ss)
        for i, s in enumerate(ss):
            if used[i]:
                continue
            cur = dict(s)
            for j in range(i + 1, len(ss)):
                if used[j]:
                    continue
                o = ss[j]
                if orient == "h":
                    if abs(o["y1"] - cur["y1"]) <= max(1.5, 0.35 * cur["thickness"]) and o["x1"] <= cur["x2"] + gap and o["x2"] >= cur["x1"] - gap:
                        cur["x1"], cur["x2"] = min(cur["x1"], o["x1"]), max(cur["x2"], o["x2"]); used[j] = True
                else:
                    if abs(o["x1"] - cur["x1"]) <= max(1.5, 0.35 * cur["thickness"]) and o["y1"] <= cur["y2"] + gap and o["y2"] >= cur["y1"] - gap:
                        cur["y1"], cur["y2"] = min(cur["y1"], o["y1"]), max(cur["y2"], o["y2"]); used[j] = True
            out.append(cur)
    return out


def extract_openings(mask, segs, kind, min_len=4):
    """Openings (doors / windows) as along-wall intervals attached to a wall segment."""
    out = []
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        cx, cy = x + w / 2.0, y + h / 2.0
        best = None
        for k, s in enumerate(segs):
            if s["orientation"] == "h":
                if not (min(s["x1"], s["x2"]) - 3 <= cx <= max(s["x1"], s["x2"]) + 3):
                    continue
                d = abs(cy - s["y1"])
            else:
                if not (min(s["y1"], s["y2"]) - 3 <= cy <= max(s["y1"], s["y2"]) + 3):
                    continue
                d = abs(cx - s["x1"])
            if d <= s["thickness"] + 4 and (best is None or d < best[0]):
                best = (d, k)
        if best is None:
            orient = "h" if w >= h else "v"
            wall_id = None
        else:
            wall_id = best[1]; orient = segs[wall_id]["orientation"]
        if orient == "h":
            a0, a1 = float(x), float(x + w)
        else:
            a0, a1 = float(y), float(y + h)
        if a1 - a0 < min_len:
            continue
        out.append(dict(kind=kind, wall=wall_id, orientation=orient, start_px=a0, end_px=a1,
                        center=(cx, cy), bbox=(int(x), int(y), int(w), int(h))))
    return out


def virtual_walls(segs, max_gap_px):
    """Pairs of collinear free wall ends facing each other (open passages, e.g. a
    room opening onto a hallway). Returns list of line segments in px."""
    ends = []
    for k, s in enumerate(segs):
        fe = s.get("free_end", {})
        for e in (1, 2):
            if fe.get(e, True):
                ends.append((k, e, s["x%d" % e], s["y%d" % e], s["orientation"], s["thickness"]))
    lines = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            a, b = ends[i], ends[j]
            if a[0] == b[0] or a[4] != b[4]:
                continue
            if a[4] == "h" and abs(a[3] - b[3]) <= max(a[5], b[5]):
                gap = abs(a[2] - b[2])
                if 2 < gap <= max_gap_px:
                    lines.append(((a[2], a[3]), (b[2], a[3]), max(a[5], b[5])))
            if a[4] == "v" and abs(a[2] - b[2]) <= max(a[5], b[5]):
                gap = abs(a[3] - b[3])
                if 2 < gap <= max_gap_px:
                    lines.append(((a[2], a[3]), (a[2], b[3]), max(a[5], b[5])))
    return lines


def extract_rooms(struct, cls, virtual, min_area_px):
    """Connected free-space regions bounded by walls/openings/virtual walls."""
    H, W = struct.shape
    barrier = struct.astype(np.uint8).copy()
    vmask = np.zeros_like(barrier)
    for (p, q, t) in virtual:
        cv2.line(vmask, (int(round(p[0])), int(round(p[1]))), (int(round(q[0])), int(round(q[1]))), 1, max(2, int(round(t))))
    free = ((barrier | vmask) == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, 4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
    rooms = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        m = lab == i
        exterior = i in border
        rooms.append(dict(label=i, mask=m, exterior=exterior, area_px=int(st[i, cv2.CC_STAT_AREA])))
    return rooms, lab, vmask


def model_room_type(cls, mask):
    vals = cls[mask]
    vals = vals[vals >= 4]
    if vals.size == 0:
        return None, 0.0
    bc = np.bincount(vals, minlength=11)
    k = int(bc.argmax())
    return ROOM_CLASS_NAMES.get(k), float(bc[k] / bc.sum())


def polygon_from_mask(mask, eps_px):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, eps_px, True)[:, 0, :].astype(np.float64)
    # Manhattan snapping of near-axis edges
    n = len(ap)
    for _ in range(2):
        for i in range(n):
            p, q = ap[i], ap[(i + 1) % n]
            dx, dy = q[0] - p[0], q[1] - p[1]
            if abs(dy) < 0.2 * abs(dx) + 1e-9:
                y = (p[1] + q[1]) / 2; ap[i][1] = y; ap[(i + 1) % n][1] = y
            elif abs(dx) < 0.2 * abs(dy) + 1e-9:
                x = (p[0] + q[0]) / 2; ap[i][0] = x; ap[(i + 1) % n][0] = x
    # pixel-centre convention -> pixel-edge (contours trace pixel centres)
    return ap


def filter_struct(struct, px_per_m):
    """Drop isolated specks (arrowheads, text blobs, plants) that are not part of the building."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(struct.astype(np.uint8), 8)
    if n <= 1:
        return struct
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros(n, bool)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; L = max(st[i, 2], st[i, 3])
        keep[i] = a >= 0.06 * big or L >= 1.2 * px_per_m
    return keep[lab]


def drop_stubs(segs, px_per_m, cls=None):
    """Remove zero-length pieces and short free-standing stubs (usually plants / furniture ink)."""
    out = []
    tmed = float(np.median([s["thickness"] for s in segs])) if segs else 0
    for s in segs:
        L = abs(s["x2"] - s["x1"]) + abs(s["y2"] - s["y1"])
        if L < 1.0 * px_per_m and s["thickness"] > 1.4 * tmed:
            continue   # short blob thicker than the walls (plant, furniture, hatch)
        fe = s.get("free_end", {})
        nfree = sum(1 for e in (1, 2) if fe.get(e, True))
        if L < max(2.0, 0.05 * px_per_m):
            continue
        if nfree == 2 and L < 1.5 * px_per_m:
            continue
        if nfree == 1 and L < 0.35 * px_per_m and L < 2.5 * s["thickness"]:
            continue
        if cls is not None and L < 1.0 * px_per_m:
            # short piece: require the model to actually see a wall there (rejects plants / furniture ink)
            t = s["thickness"] / 2
            x0, x1 = int(min(s["x1"], s["x2"]) - (t if s["orientation"] == "v" else 0)), int(max(s["x1"], s["x2"]) + (t if s["orientation"] == "v" else 0)) + 1
            y0, y1 = int(min(s["y1"], s["y2"]) - (t if s["orientation"] == "h" else 0)), int(max(s["y1"], s["y2"]) + (t if s["orientation"] == "h" else 0)) + 1
            patch = cls[max(0, y0):y1, max(0, x0):x1]
            if patch.size and np.isin(patch, (1, 2, 3)).mean() < 0.5:
                continue
        out.append(s)
    _snap_junctions(out)
    return out


def join_small_gaps(segs, max_gap_px):
    """Vector-space join of collinear walls interrupted by tiny breaks (clutter touching the wall)."""
    segs = _merge_collinear([dict(s) for s in segs], gap=max_gap_px)
    _snap_junctions(segs)
    return _split_at_junctions(segs)


def raster_segments(segs, shape, grow=0):
    m = np.zeros(shape, np.uint8)
    for s in segs:
        t = s["thickness"] / 2 + grow
        if s["orientation"] == "h":
            x0, x1, y0, y1 = min(s["x1"], s["x2"]), max(s["x1"], s["x2"]), s["y1"] - t, s["y1"] + t
        else:
            x0, x1, y0, y1 = s["x1"] - t, s["x1"] + t, min(s["y1"], s["y2"]), max(s["y1"], s["y2"])
        cv2.rectangle(m, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, -1)
    return m > 0


def find_wall_gaps(segs, struct, px_per_m, min_m=0.0, max_m=3.2):
    """Collinear wall ends facing each other with an empty gap => an opening
    (door / passage) whose symbol the model did not mark (e.g. bi-fold doors)."""
    ends = []
    for k, s in enumerate(segs):
        fe = s.get("free_end", {})
        for e in (1, 2):
            if fe.get(e, True):
                ends.append((k, e, s))
    gaps = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            (ka, ea, a), (kb, eb, b) = ends[i], ends[j]
            if ka == kb or a["orientation"] != b["orientation"]:
                continue
            t = max(a["thickness"], b["thickness"])
            if a["orientation"] == "h":
                if abs(a["y1"] - b["y1"]) > max(2.0, 0.5 * t):
                    continue
                pa, pb = a["x%d" % ea], b["x%d" % eb]
                # must face each other: a's end-2 (right) to b's end-1 (left) or vice versa
                if not ((ea == 2 and eb == 1 and pb > pa) or (ea == 1 and eb == 2 and pa > pb)):
                    continue
                lo, hi = sorted((pa, pb)); c = (a["y1"] + b["y1"]) / 2
                y0, y1 = int(round(c - t / 2)), int(round(c + t / 2)) + 1
                x0, x1 = int(lo) + 1, int(hi)
            else:
                if abs(a["x1"] - b["x1"]) > max(2.0, 0.5 * t):
                    continue
                pa, pb = a["y%d" % ea], b["y%d" % eb]
                if not ((ea == 2 and eb == 1 and pb > pa) or (ea == 1 and eb == 2 and pa > pb)):
                    continue
                lo, hi = sorted((pa, pb)); c = (a["x1"] + b["x1"]) / 2
                x0, x1 = int(round(c - t / 2)), int(round(c + t / 2)) + 1
                y0, y1 = int(lo) + 1, int(hi)
            gap = hi - lo
            if not (min_m * px_per_m <= gap <= max_m * px_per_m):
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            if struct[y0:y1, x0:x1].mean() > 0.3 and gap > 0.45 * px_per_m:
                continue
            if gap < 1.5:
                continue
            gaps.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, orientation=a["orientation"], gap_px=gap))
    # keep the shortest non-overlapping gaps
    gaps.sort(key=lambda g: g["gap_px"])
    out = []
    for g in gaps:
        if all(g["x1"] <= o["x0"] or g["x0"] >= o["x1"] or g["y1"] <= o["y0"] or g["y0"] >= o["y1"] for o in out):
            out.append(g)
    return out


def corner_cuts(segs, room_mask, px_per_m, min_m=0.5, max_m=3.2):
    """Candidate straight 'cut' lines between aligned wall end points that run
    through a room region (open-plan boundaries such as living <-> hallway)."""
    pts = []
    for s in segs:
        pts.append((s["x1"], s["y1"], s["thickness"])); pts.append((s["x2"], s["y2"], s["thickness"]))
    H, W = room_mask.shape
    cuts = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (xa, ya, ta), (xb, yb, tb) = pts[i], pts[j]
            t = max(ta, tb)
            if abs(xa - xb) <= t * 0.75 and abs(ya - yb) > 1:
                x = (xa + xb) / 2; p, q = (x, min(ya, yb)), (x, max(ya, yb)); L = abs(ya - yb)
            elif abs(ya - yb) <= t * 0.75 and abs(xa - xb) > 1:
                y = (ya + yb) / 2; p, q = (min(xa, xb), y), (max(xa, xb), y); L = abs(xa - xb)
            else:
                continue
            if not (min_m * px_per_m <= L <= max_m * px_per_m):
                continue
            n = max(2, int(L))
            xs = np.clip(np.round(np.linspace(p[0], q[0], n)).astype(int), 0, W - 1)
            ys = np.clip(np.round(np.linspace(p[1], q[1], n)).astype(int), 0, H - 1)
            inside = room_mask[ys, xs].mean()
            if inside < 0.6:
                continue
            cuts.append((L, p, q, t))
    cuts.sort(key=lambda c: c[0])
    return cuts


def split_region(mask, label_points, cuts, min_area_px):
    """Apply cuts until every label point lies in its own component."""
    m = mask.copy().astype(np.uint8)
    applied = []

    def groups(mm):
        n, lab = cv2.connectedComponents(mm, connectivity=4)
        ids = [lab[int(y), int(x)] for (x, y) in label_points]
        return n, lab, ids
    n, lab, ids = groups(m)
    for (L, p, q, t) in cuts:
        if len(set(ids)) == len(ids) and 0 not in ids:
            break
        trial = m.copy()
        cv2.line(trial, (int(round(p[0])), int(round(p[1]))), (int(round(q[0])), int(round(q[1]))), 0, 2)
        n2, lab2, ids2 = groups(trial)
        if 0 in ids2:
            continue
        if len(set(ids2)) > len(set(ids)):
            areas = [(lab2 == k).sum() for k in set(ids2)]
            if min(areas) >= min_area_px:
                m, n, lab, ids = trial, n2, lab2, ids2
                applied.append((p, q, t))
    return lab, ids, applied


def closed_region_with_lines(gray, struct, seed_xy, max_area_px):
    """Region around seed bounded by walls AND thin drawn lines (e.g. balcony
    railings). Furniture inside is filled. Returns mask or None."""
    lines = cv2.dilate((gray < 200).astype(np.uint8), np.ones((3, 3), np.uint8))
    free = ((lines == 0) & (~struct)).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    x, y = int(seed_xy[0]), int(seed_xy[1])
    H, W = gray.shape
    k = lab[min(max(y, 0), H - 1), min(max(x, 0), W - 1)]
    if k == 0:
        return None
    m = (lab == k).astype(np.uint8)
    if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
        return None
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m); cv2.drawContours(filled, cs, -1, 1, -1)
    filled = cv2.dilate(filled, np.ones((3, 3), np.uint8))   # give back the 1px line dilation
    filled = (filled > 0) & ~struct
    if filled.sum() > max_area_px:
        return None
    return filled
