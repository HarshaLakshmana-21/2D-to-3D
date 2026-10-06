import glob, os, time, random, math, json
import cv2, numpy as np, torch, torch.nn.functional as F
from model import FloorUNet, NUM_CLASSES
torch.set_num_threads(2)
random.seed(0); np.random.seed(0); torch.manual_seed(0)
files = sorted(glob.glob('data/img/*.jpg'))
val = [f for f in files if int(os.path.basename(f)[:5]) < 300]
tr = [f for f in files if int(os.path.basename(f)[:5]) >= 300]
lab = lambda f: f.replace('/img/', '/lab/').replace('.jpg', '.png')
CROP, BS, ITERS = 256, 8, int(os.environ.get('ITERS', 5500))
MEAN = 0.5

def load(f):
    im = cv2.imread(f)[..., ::-1]; lb = cv2.imread(lab(f), 0)
    return im, lb

def aug(im, lb):
    s = random.uniform(0.65, 1.35)
    if abs(s - 1) > 0.05:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        lb = cv2.resize(lb, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    k = random.randint(0, 3)
    im = np.rot90(im, k); lb = np.rot90(lb, k)
    if random.random() < 0.5: im = im[:, ::-1]; lb = lb[:, ::-1]
    H, W = lb.shape
    ph, pw = max(0, CROP - H), max(0, CROP - W)
    if ph or pw:
        im = cv2.copyMakeBorder(np.ascontiguousarray(im), 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        lb = cv2.copyMakeBorder(np.ascontiguousarray(lb), 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        H, W = lb.shape
    # bias crops toward content
    for _ in range(5):
        y = random.randint(0, H - CROP); x = random.randint(0, W - CROP)
        if (lb[y:y + CROP, x:x + CROP] > 0).mean() > 0.15: break
    im = np.ascontiguousarray(im[y:y + CROP, x:x + CROP]).astype(np.float32)
    lb = np.ascontiguousarray(lb[y:y + CROP, x:x + CROP])
    if random.random() < 0.4: im[:] = im.mean(2, keepdims=True)
    im = im * random.uniform(0.8, 1.15) + random.uniform(-20, 20)
    if random.random() < 0.2: im = cv2.GaussianBlur(im, (3, 3), random.uniform(0.3, 1.0))
    if random.random() < 0.2: im = im + np.random.randn(*im.shape).astype(np.float32) * random.uniform(2, 10)
    return np.clip(im, 0, 255) / 255.0 - MEAN, lb

W_CE = torch.tensor([0.5, 1.5, 3.0, 3.0] + [1.0] * (NUM_CLASSES - 4))
def loss_fn(logit, y):
    ce = F.cross_entropy(logit, y, weight=W_CE)
    p = logit.softmax(1)
    dl = 0
    for c in (1, 2, 3):
        t = (y == c).float(); q = p[:, c]
        dl += 1 - (2 * (q * t).sum() + 1) / (q.sum() + t.sum() + 1)
    return ce + dl / 3

def evaluate(m, n=60):
    m.eval(); inter = np.zeros(NUM_CLASSES); uni = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for f in val[:n]:
            im, lb = load(f)
            H, W = lb.shape; H2, W2 = (H + 31) // 32 * 32, (W + 31) // 32 * 32
            x = cv2.copyMakeBorder(np.ascontiguousarray(im), 0, H2 - H, 0, W2 - W, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            x = torch.from_numpy(x.astype(np.float32) / 255 - MEAN).permute(2, 0, 1)[None]
            pr = m(x)[0].argmax(0).numpy()[:H, :W]
            for c in range(NUM_CLASSES):
                inter[c] += ((pr == c) & (lb == c)).sum(); uni[c] += ((pr == c) | (lb == c)).sum()
    m.train()
    iou = inter / np.maximum(uni, 1)
    return iou

m = FloorUNet()
start = 0
if os.path.exists('ckpt.pt') and os.environ.get('RESUME'):
    sd = torch.load('ckpt.pt'); m.load_state_dict(sd['model']); start = sd['it']
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=ITERS, pct_start=0.05)
for _ in range(start): sched.step()
t0 = time.time(); run = 0
for it in range(start, ITERS):
    xs, ys = zip(*[aug(*load(random.choice(tr))) for _ in range(BS)])
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float(); y = torch.from_numpy(np.stack(ys)).long()
    loss = loss_fn(m(x), y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if it > start else loss.item()
    if it % 50 == 0:
        print(f'it {it} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s', flush=True)
    if (it + 1) % 250 == 0:
        torch.save({'model': m.state_dict(), 'it': it + 1}, 'ckpt.pt')
    if (it + 1) % 1000 == 0 or it + 1 == ITERS:
        iou = evaluate(m)
        print('VAL IoU', json.dumps({k: round(float(v), 3) for k, v in zip(['bg','wall','door','window','living','kitchen','bedroom','bathroom','balcony','storage','stair'], iou)}), 'mIoU', round(float(iou.mean()), 3), flush=True)
torch.save({'model': m.state_dict(), 'it': ITERS}, 'final.pt')
print('done', flush=True)
