import sys, torch
from model import FloorUNet
src = sys.argv[1]; dst = sys.argv[2]
m = FloorUNet(); m.load_state_dict(torch.load(src)['model']); m.eval()
x = torch.randn(1, 3, 256, 320)
torch.onnx.export(m, x, dst, input_names=['image'], output_names=['logits'], opset_version=17,
                  dynamic_axes={'image': {0: 'n', 2: 'h', 3: 'w'}, 'logits': {0: 'n', 2: 'h', 3: 'w'}}, dynamo=False)
import onnxruntime as ort, numpy as np
s = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
y = s.run(None, {'image': np.random.randn(1, 3, 320, 480).astype(np.float32)})[0]
with torch.no_grad(): yt = m(torch.from_numpy(np.zeros((1,3,320,480),np.float32))).numpy()
print('onnx ok', y.shape, float(np.abs(s.run(None, {'image': np.zeros((1,3,320,480),np.float32)})[0]-yt).max()))
