import pickle, json, os, random, cv2, numpy as np
from render import render, geoms
d = pickle.load(open(os.path.join(os.environ.get('RESPLAN_DIR', 'ResPlan'), 'ResPlan.pkl'), 'rb'))
split = json.load(open(os.path.join(os.environ.get('RESPLAN_DIR', 'ResPlan'), 'split.json')))
id2i = {p['id']: k for k, p in enumerate(d)}
te = [id2i[x] for x in split['test'] if x in id2i][:120]
os.makedirs('testset', exist_ok=True)
meta = []
for j, idx in enumerate(te):
    random.seed(99000 + j)
    im, lb, m = render(d[idx], max_side=1100, seed=99000 + j)
    cv2.imwrite(f'testset/{j:04d}.png', im[..., ::-1]); cv2.imwrite(f'testset/{j:04d}_lab.png', lb)
    meta.append(dict(file=f'{j:04d}.png', plan_index=idx, plan_id=d[idx]['id'], **m))
json.dump(meta, open('testset/meta.json', 'w'))
print('done', len(meta))
