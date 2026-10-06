"""Held-out evaluation on ResPlan test-split renders (never seen in training)."""
import sys, json, glob, os, time
import cv2, numpy as np
sys.path.insert(0, '..')
from floorplan3d.pipeline import analyze
from floorplan3d.segment import segment

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
meta = json.load(open('testset/meta.json'))[:N]
CL = ["bg", "wall", "door", "window", "living", "kitchen", "bedroom", "bathroom", "balcony", "storage", "stair"]
ROOMC = {4: "living", 5: "kitchen", 6: "bedroom", 7: "bathroom", 8: "balcony", 9: "storage", 10: "stair"}
TYPEMAP = {"living_kitchen": "living", "master_bedroom": "bedroom", "hallway": "living", "dining": "living"}
inter = np.zeros(11); uni = np.zeros(11)
scale_err, area_err, type_ok, type_n, open_err, room_recall = [], [], 0, 0, {"door": [], "window": []}, []
open_det = {"door": [0, 0], "window": [0, 0]}
t0 = time.time()
for m in meta:
    f = 'testset/' + m['file']; lab = cv2.imread(f.replace('.png', '_lab.png'), 0)
    r = analyze(f, out_dir=None, verbose=False, tta=True)
    true_s = m['px_per_m']
    # pixel IoU (argmax of model incl. cleaning not applied)
    img = cv2.imread(f); pr = segment(img, tta=True).argmax(0)
    for c in range(11):
        inter[c] += ((pr == c) & (lab == c)).sum(); uni[c] += ((pr == c) | (lab == c)).sum()
    s = (r['scale']['px_per_m_x'] + r['scale']['px_per_m_y']) / 2
    if r['scale']['method'] == 'dimension_annotations':
        scale_err.append(100 * (s - true_s) / true_s)
    # rooms: GT = connected components of room labels (4-connected, per class)
    gt_rooms = []
    for c in range(4, 11):
        n, cc = cv2.connectedComponents((lab == c).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            mk = cc == i
            if mk.sum() > 1.5 * true_s * true_s:
                gt_rooms.append((c, mk))
    for c, mk in gt_rooms:
        best = None
        for rm in r['rooms']:
            iou = (mk & rm['_mask']).sum() / max(1, (mk | rm['_mask']).sum())
            if best is None or iou > best[0]:
                best = (iou, rm)
        found = best is not None and best[0] > 0.5
        room_recall.append(found)
        if found:
            rm = best[1]
            gt_area = mk.sum() / true_s ** 2
            area_err.append(100 * (rm['area_m2'] - gt_area) / gt_area)
            t = TYPEMAP.get(rm['type'], rm['type'])
            type_n += 1; type_ok += (t == ROOMC[c])
    # openings: GT components of door/window; width = long side
    for kind, c in (("door", 2), ("window", 3)):
        n, cc, st, cen = cv2.connectedComponentsWithStats((lab == c).astype(np.uint8), 8)
        dets = [o for o in r['openings'] if o['type'] == kind]
        for i in range(1, n):
            gw = max(st[i, 2], st[i, 3]) / true_s
            if gw < 0.3:
                continue
            open_det[kind][1] += 1
            cx, cy = cen[i]
            best = None
            for o in dets:
                x, y, w, h = o['_px']['bbox']
                d = np.hypot(x + w / 2 - cx, y + h / 2 - cy)
                if d < max(st[i, 2], st[i, 3]) / 2 and (best is None or d < best[0]):
                    best = (d, o)
            if best:
                open_det[kind][0] += 1
                # measured width converted with TRUE scale to isolate detection error from calibration error
                o = best[1]; wpx = o['_px']['b'] - o['_px']['a']
                open_err[kind].append(100 * (wpx / true_s - gw) / gw)
iou = inter / np.maximum(uni, 1)
res = dict(images=len(meta), seconds_per_image=round((time.time() - t0) / len(meta), 2),
           pixel_iou={k: round(float(v), 3) for k, v in zip(CL, iou)}, mIoU=round(float(iou.mean()), 3),
           scale_calibration=dict(n=len(scale_err), median_abs_err_pct=round(float(np.median(np.abs(scale_err))), 2) if scale_err else None,
                                  p90_abs_err_pct=round(float(np.percentile(np.abs(scale_err), 90)), 2) if scale_err else None),
           rooms=dict(recall_iou50=round(float(np.mean(room_recall)), 3), area_median_abs_err_pct=round(float(np.median(np.abs(area_err))), 2),
                      area_p90_abs_err_pct=round(float(np.percentile(np.abs(area_err), 90)), 2), type_accuracy=round(type_ok / max(1, type_n), 3)),
           openings={k: dict(recall=round(open_det[k][0] / max(1, open_det[k][1]), 3),
                             width_median_abs_err_pct=round(float(np.median(np.abs(v))), 2) if v else None) for k, v in open_err.items()})
print(json.dumps(res, indent=2))
json.dump(res, open('eval_results.json', 'w'), indent=2)
