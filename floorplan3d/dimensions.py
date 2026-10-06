"""Scale calibration from dimension annotations.

For every OCR'd dimension string (e.g. 18' 3") we locate the dimension line
that carries it, find its terminators (arrowheads / ticks / slashes) and
measure the pixel distance between them. pixel/metre is then estimated per
axis with a robust median, and each dimension's residual is reported so the
calibration can be audited."""
import numpy as np
import cv2


def _runs(mask1d):
    """start,end (inclusive) of True runs."""
    m = np.r_[False, mask1d, False].astype(np.int8)
    d = np.diff(m)
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0] - 1
    return list(zip(s, e))


def _thin_lines(binary, horizontal, min_len):
    """Return list of (pos, a, b) of thin straight lines."""
    k = np.ones((4, 4), np.uint8)
    thick = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    thin = binary & ~cv2.dilate(thick, np.ones((3, 3), np.uint8))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1) if horizontal else (1, min_len))
    lines = cv2.morphologyEx(thin, cv2.MORPH_OPEN, ker)
    out = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines, 8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if horizontal and h <= 4:
            out.append((y + h / 2.0, x, x + w - 1))
        elif not horizontal and w <= 4:
            out.append((x + w / 2.0, y, y + h - 1))
    return out


def _terminators(binary, pos, a, b, horizontal, band=5, thick=None):
    """Find terminators (arrow tips / ticks / slashes) along a dimension line.
    Arrowheads are located at their tip (narrow end); symmetric ticks at their centre."""
    p = int(round(pos))
    H, W = binary.shape
    if horizontal:
        strip = binary[max(0, p - band):min(H, p + band + 1), :]
        on = binary[max(0, p - 1):p + 2, :].max(0) > 0
        if thick is not None:
            on &= ~(thick[max(0, p - 1):p + 2, :].max(0) > 0)
        L = W
    else:
        strip = binary[:, max(0, p - band):min(W, p + band + 1)].T
        on = binary[:, max(0, p - 1):p + 2].max(1) > 0
        if thick is not None:
            on &= ~(thick[:, max(0, p - 1):p + 2].max(1) > 0)
        L = H
    mid = min(band, p)
    ext = strip.sum(0).astype(np.int32)
    upx = strip[:max(0, mid - 1)].sum(0) > 0
    dnx = strip[mid + 2:].sum(0) > 0
    lo, hi = int(a), int(b)
    while lo > 0 and on[lo - 1]:
        lo -= 1
    while hi < L - 1 and on[hi + 1]:
        hi += 1
    base = float(np.median(ext[lo:hi + 1])) if hi > lo else 1.0
    cand = (ext >= base + 2)
    rng = np.zeros(L, bool); rng[lo:hi + 1] = True
    cents = [float(lo), float(hi)]
    runs = []
    for s, e in _runs(cand & rng):           # merge blobs separated by tiny gaps
        if runs and s - runs[-1][1] <= 4:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))
    for s, e in runs:
        if e - s > 40:
            continue
        if not (upx[s:e + 1].any() and dnx[s:e + 1].any()):
            continue                      # one-sided => text or clutter, not a terminator
        if s <= lo + 3 or e >= hi - 3:
            continue                      # arrow at the line end: the tip is lo/hi
        cents.append((s + e) / 2.0)       # interior junction (><, ticks, slashes): centre
    cents = sorted(cents)
    merged = []
    for c in cents:
        if merged and c - merged[-1] < 6:
            if merged[-1] in (lo, hi):
                continue
            merged[-1] = (merged[-1] + c) / 2 if c not in (lo, hi) else c
        else:
            merged.append(c)
    return merged, lo, hi


def calibrate(gray, texts, wall_mask=None):
    """Return dict with px_per_m_x, px_per_m_y, dims list."""
    binary = (gray < 200).astype(np.uint8)
    if wall_mask is not None:
        binary = binary & (~(wall_mask > 0)).astype(np.uint8)
    # walls crossing a dimension line: tall bars for horizontal lines, wide bars for vertical
    thick_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((15, 5), np.uint8))
    thick_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 15), np.uint8))
    hl = _thin_lines(binary, True, 25)
    vl = _thin_lines(binary, False, 25)
    dims = []
    for t in texts:
        cands = t.get("dim_cands") or ([(t["dim"], 1.0)] if t.get("dim") else [])
        if not cands:
            continue
        c = t["center"]; horiz = not t["vertical"]
        th = (t["h"] if horiz else t["w"])
        cand = []
        for (pos, a, b) in (hl if horiz else vl):
            along = c[0] if horiz else c[1]
            perp = c[1] if horiz else c[0]
            d = abs(pos - perp)
            if a - 3 <= along <= b + 3 and d < 2.8 * th + 6:
                cand.append((d, pos, a, b))
        if not cand:
            continue
        cand.sort()
        _, pos, a, b = cand[0]
        terms, lo, hi = _terminators(binary, pos, a, b, horiz, thick=thick_v if horiz else thick_h)
        along = c[0] if horiz else c[1]
        left = [x for x in terms if x <= along]
        right = [x for x in terms if x >= along]
        if not left or not right:
            continue
        p0, p1 = max(left), min(right)
        if p1 - p0 < 10:
            continue
        dims.append(dict(text=t["text"], metres=cands[0][0], cands=cands, axis="x" if horiz else "y",
                         line_pos=float(pos), p0=float(p0), p1=float(p1), px=float(p1 - p0),
                         px_per_m=float((p1 - p0) / cands[0][0])))
    res = dict(dims=dims, method="none")
    if not dims:
        return res
    # consensus over all readings: the scale with the largest weighted support wins
    allv = [(i, d["px"] / c, w) for i, d in enumerate(dims) for (c, w) in d["cands"] if c > 0]
    best = None
    for _, v, _ in allv:
        sup = {}
        for i, u, w in allv:
            if abs(u - v) / v < 0.03:
                sup[i] = max(sup.get(i, 0), w)
        score = sum(sup.values())
        if best is None or score > best[0] + 1e-9:
            best = (score, v, sup)
    score, v0, sup = best
    # a single dimension is only trusted if it was read cleanly (explicit units) and is long enough
    if score < 1.0 or (len(sup) == 1 and dims[next(iter(sup))]["px"] < 60):
        res["rejected"] = "dimension strings not reliable enough"
        return res
    for i, d in enumerate(dims):
        c = min((c for c, w in d["cands"]), key=lambda c: abs(d["px"] / c - v0) / v0)
        d["metres"] = c; d["px_per_m"] = d["px"] / c
        d["used"] = abs(d["px_per_m"] - v0) / v0 < 0.03
    # refine with the used readings
    v1 = float(np.median([d["px_per_m"] for d in dims if d["used"]]))
    for d in dims:
        d["used"] = abs(d["px_per_m"] - v1) / v1 < 0.03
    xs = [d["px_per_m"] for d in dims if d["used"] and d["axis"] == "x"]
    ys = [d["px_per_m"] for d in dims if d["used"] and d["axis"] == "y"]
    both = xs + ys
    res["px_per_m_x"] = float(np.median(xs if xs else both))
    res["px_per_m_y"] = float(np.median(ys if ys else both))
    res["method"] = "dimension_annotations"
    res["support"] = f"{sum(d['used'] for d in dims)}/{len(dims)} dimension strings agree"
    for d in dims:
        s_ = res["px_per_m_x"] if d["axis"] == "x" else res["px_per_m_y"]
        d["measured_m"] = d["px"] / s_
        d["error_pct"] = 100.0 * (d["measured_m"] - d["metres"]) / d["metres"]
        d.pop("cands", None)
    return res
