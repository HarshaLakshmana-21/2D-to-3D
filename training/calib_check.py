import sys, json, cv2, numpy as np
sys.path.insert(0, '..')
from floorplan3d.ocr import read_text
from floorplan3d.pipeline import _calibrate_multires
meta = json.load(open('testset/meta.json'))[:60]
errs = []
for m in meta:
    im = cv2.imread('testset/' + m['file']); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    T = read_text(im); r = _calibrate_multires(g, T)
    if r.get('method') != 'dimension_annotations': continue
    s = (r['px_per_m_x'] + r['px_per_m_y']) / 2; e = 100 * (s - m['px_per_m']) / m['px_per_m']
    errs.append(e)
    if abs(e) > 3:
        print(m['file'], round(e, 1), r.get('support'), [(d['text'], round(d['metres'], 3), d['used'], round(d['px'], 1)) for d in r['dims']])
print('n', len(errs), 'median', np.median(np.abs(errs)), 'p90', np.percentile(np.abs(errs), 90))
