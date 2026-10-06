import pickle, sys, os, json, random, cv2, numpy as np
from render import render
part, nparts, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
d = pickle.load(open(os.path.join(os.environ.get('RESPLAN_DIR', 'ResPlan'), 'ResPlan.pkl'),'rb'))
split = json.load(open(os.path.join(os.environ.get('RESPLAN_DIR', 'ResPlan'), 'split.json')))
print(split.keys() if isinstance(split, dict) else type(split), flush=True)
os.makedirs('data/img', exist_ok=True); os.makedirs('data/lab', exist_ok=True)
id2i = {p['id']: k for k, p in enumerate(d)}
tr = [id2i[x] for x in split['train'] if x in id2i]; va = [id2i[x] for x in split['val'] if x in id2i]
NV = 300
for i in range(part, N, nparts):
    idx = va[i % len(va)] if i < NV else tr[(i * 2654435761) % len(tr)]
    try:
        im, lb, m = render(d[idx], max_side=1100, seed=i * 7919 + 13)
    except Exception as e:
        print('err', i, e, flush=True); continue
    cv2.imwrite(f'data/img/{i:05d}.jpg', im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, random.randint(80, 97)])
    cv2.imwrite(f'data/lab/{i:05d}.png', lb)
    if i % 500 == 0: print(i, flush=True)
