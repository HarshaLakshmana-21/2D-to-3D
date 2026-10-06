import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 11


class CBR(nn.Sequential):
    def __init__(self, i, o, k=3, d=1):
        super().__init__(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.a = CBR(i, o); self.b = CBR(o, o)
        self.skip = nn.Conv2d(i, o, 1, bias=False) if i != o else nn.Identity()

    def forward(self, x):
        return self.b(self.a(x)) + self.skip(x)


class Context(nn.Module):
    """ASPP-lite + global pooling for room-type context."""
    def __init__(self, c):
        super().__init__()
        self.b = nn.ModuleList([CBR(c, c // 2, 1), CBR(c, c // 2, 3, 2), CBR(c, c // 2, 3, 4), CBR(c, c // 2, 3, 8)])
        self.out = CBR(c // 2 * 4, c, 1)

    def forward(self, x):
        ys = [b(x) for b in self.b]
        return self.out(torch.cat(ys, 1))


class FloorUNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128, 192), n_classes=NUM_CLASSES):
        super().__init__()
        self.stem = Block(3, ch[0])
        self.downs = nn.ModuleList([Block(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        self.ctx = Context(ch[-1])
        self.ups = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1))])
        self.head = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        for d in self.downs:
            skips.append(x)
            x = d(F.max_pool2d(x, 2))
        x = self.ctx(x)
        for u in self.ups:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[2:], mode="bilinear", align_corners=False)
            x = u(torch.cat([x, s], 1))
        return self.head(x)


if __name__ == "__main__":
    import time
    m = FloorUNet()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
    x = torch.randn(8, 3, 320, 320)
    opt = torch.optim.AdamW(m.parameters())
    for i in range(3):
        t = time.time(); y = m(x); y.mean().backward(); opt.step(); opt.zero_grad(); print(time.time() - t)
