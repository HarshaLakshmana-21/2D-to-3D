"""Semantic segmentation of floor-plan images with the trained FloorUNet (ONNX)."""
import os
import numpy as np
import cv2

CLASSES = ["background", "wall", "door", "window", "living", "kitchen",
           "bedroom", "bathroom", "balcony", "storage", "stair"]
MEAN = 0.5
_SESS = {}


def default_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "floorplan_seg.onnx")


def _session(path):
    if path not in _SESS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _SESS[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _SESS[path]


def _run(sess, rgb):
    H, W = rgb.shape[:2]
    H2, W2 = (H + 31) // 32 * 32, (W + 31) // 32 * 32
    x = cv2.copyMakeBorder(rgb, 0, H2 - H, 0, W2 - W, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    x = (x.astype(np.float32) / 255.0 - MEAN).transpose(2, 0, 1)[None]
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, :, :H, :W]
    e = np.exp(logits - logits.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)


def segment(img_bgr, model_path=None, tile=768, overlap=128, tta=True):
    """Returns per-class probabilities (C,H,W) float32."""
    sess = _session(model_path or default_model_path())
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    def infer(im):
        h, w = im.shape[:2]
        if max(h, w) <= tile:
            return _run(sess, im)
        out = np.zeros((len(CLASSES), h, w), np.float32); cnt = np.zeros((h, w), np.float32)
        step = tile - overlap
        for y in range(0, max(1, h - overlap), step):
            for x in range(0, max(1, w - overlap), step):
                y0, x0 = min(y, max(0, h - tile)), min(x, max(0, w - tile))
                p = _run(sess, np.ascontiguousarray(im[y0:y0 + tile, x0:x0 + tile]))
                out[:, y0:y0 + p.shape[1], x0:x0 + p.shape[2]] += p
                cnt[y0:y0 + p.shape[1], x0:x0 + p.shape[2]] += 1
        return out / np.maximum(cnt, 1)

    prob = infer(rgb)
    if tta:  # flip test-time augmentation
        prob += infer(np.ascontiguousarray(rgb[:, ::-1]))[:, :, ::-1]
        prob += infer(np.ascontiguousarray(rgb[::-1]))[:, ::-1]
        prob /= 3.0
    return prob.astype(np.float32)
