"""
ELA-MPCA step-by-step visualization (English titles, no Chinese glyphs)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. dummy input ----------
torch.manual_seed(42)
img = torch.rand(1, 3, 256, 256)
conv = nn.Conv2d(3, 32, 4, stride=2, padding=1)
X = conv(img)                                    # 1×32×128×128

# ---------- 2. ELA ----------
class ELA(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv_h = nn.Conv1d(C, C, 3, padding=1, groups=C)
        self.conv_w = nn.Conv1d(C, C, 3, padding=1, groups=C)

    def forward(self, x):
        B, C, H, W = x.shape
        gh = x.mean(dim=-1)          # B×C×H
        gw = x.mean(dim=-2)          # B×C×W
        ah = torch.sigmoid(self.conv_h(gh)).unsqueeze(-1)  # B×C×H×1
        aw = torch.sigmoid(self.conv_w(gw)).unsqueeze(-2)  # B×C×1×W
        Ms = ah * aw
        return x * Ms, Ms

ela_layer = ELA(32)
with torch.no_grad():
    X_prime, Ms = ela_layer(X)

# ---------- 3. MPCA (pad + dynamic nWin) ----------
class MPCA(nn.Module):
    def __init__(self, C, window=7):
        super().__init__()
        self.window = window
        self.dw3 = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.dw5 = nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False)
        self.dw7 = nn.Conv2d(C, C, 7, padding=3, groups=C, bias=False)
        self.proj = nn.Conv2d(3 * C, C, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.window
        # multi-scale DWConv
        y = torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)  # B×3C×H×W
        # pad to multiple of window
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, Hp, Wp = y.shape
        # unfold
        y = y.unfold(2, w, w).unfold(3, w, w)              # B×3C×nH×nW×w×w
        nH, nW = y.shape[2], y.shape[3]
        y = y.permute(0, 1, 2, 4, 3, 5).reshape(B, 3*C, nH*nW, w*w)  # 3C×(nH*nW)×49
        y = y.permute(0, 2, 1, 3).reshape(-1, w*w)                    # (B*nH*nW*3C)×49
        # single-head MSA
        attn = torch.softmax(y @ y.T / (w * w), dim=-1)
        y_hat = (attn @ y).view(B, nH, nW, 3*C, w, w)
        # fold back
        y_hat = y_hat.permute(0, 3, 1, 4, 2, 5).reshape(B, 3*C, nH*w, nW*w)[:, :, :H, :W]
        return x + self.proj(y_hat)

mpca_layer = MPCA(32)
with torch.no_grad():
    X_double_prime = mpca_layer(X_prime)

# ---------- 4. visualization ----------
def show(feat, title, cmap='magma', nrow=4):
    B, C = feat.shape[:2]
    feat = feat[0].detach().cpu()
    ncol = int(np.ceil(16 / nrow))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*2.2, nrow*2.2))
    axs = axs.flatten()
    for i in range(16):
        axs[i].imshow(feat[i], cmap=cmap)
        axs[i].set_title(f'ch{i}', fontsize=8)
        axs[i].axis('off')
    plt.suptitle(title, weight='bold')
    plt.tight_layout()
    plt.show()

show(X,          '1. Original X')
show(Ms,         '2. ELA spatial mask Ms')
show(X_prime,    '3. ELA output X\' (background suppressed)')
show(X_double_prime, '4. MPCA refined X\'\' (edge/multi-scale enhanced)')