"""nnU-Net-lite 3D baseline for the fold-0 comparison (improvement item P0-1).

Why this script exists
----------------------
The manuscript compares MW-HQNet only against a plain 2.5-D 3D-U-Net. Every KiTS winner from
2019 to 2023 is an nnU-Net derivative, so a reviewer will ask for a stronger baseline. This
trains a *true 3D* residual U-Net (SegResNet-style: instance norm, residual blocks, deep
supervision) under the same budget as every other run in this study, so the comparison isolates
architecture rather than training recipe.

Fairness controls (identical to the pipeline runs):
    same ROI cache, same fold split (folds_kits23_5_seed42.json), fold 0
    same 15 epochs x 800 iterations, AdamW 3e-4, cosine, 1-epoch warmup, grad clip 1.0, AMP
    same post-processing and the same metric code (imported from the pipeline notebook)

Differences that define "3D":
    3D patches instead of 5-slice stacks; sliding-window inference over the whole ROI volume;
    single HU window (no PAWE, no queries, no graph) so it is a pure encoder-decoder baseline.

Outputs land in runs/nnunet_lite_3d_fold0/ with the same file names as every other run, so the
existing analysis scripts pick them up unchanged.
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
ROOT = r"E:\jupyter_work\mwhq_pipeline"
RUN_NAME = os.environ.get("NNL_RUN", "nnunet_lite_3d_fold0")

PATCH = (32, 128, 128)   # (depth, H, W) -- the volume is (D, 192, 192)
BATCH = 2
ITERS = int(os.environ.get("NNL_ITERS", 800))
EPOCHS = int(os.environ.get("NNL_EPOCHS", 15))
VAL_AT = {4, 9, 14}
BASE_CH = int(os.environ.get("NNL_BASE", 32))
LR = 3e-4

# ---------------------------------------------------------------- shared code
nb = json.loads(open(NB, encoding="utf-8").read())
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def find(m):
    for i, s in cells.items():
        if m in s[:220]:
            return i, s
    raise KeyError(m)


g = {"__name__": "__main__", "display": lambda *a, **k: None}
for m in ["1.1 environment", "2.1 installs", "2.2 preprocessing", "2.3 dataset",
          "2.3b prepare_folds", "2.5 losses", "2.6 inference", "3.4 config"]:
    i, s = find(m)
    exec(compile(s, f"cell{i}", "exec"), g)

g["prepare_folds"]()
CFG = g["CFG"]
CFG.update(DATA_DIR=g["DATA_DIR"], OUT_DIR=os.path.join(ROOT, "runs"),
           EPOCHS=EPOCHS, BATCH=BATCH, NUM_WORKERS=0, AMP=True, PRETRAINED=False,
           ENHANCE="single", AIFI=False, QUERY=False, SEED=42)
RUN_DIR = os.path.join(CFG["OUT_DIR"], RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

TRAIN_KEYS, VAL_KEYS, SPACING = g["TRAIN_KEYS"], g["VAL_KEYS"], g["SPACING"]
DATA_DIR = CFG["DATA_DIR"]
dev = g["dev"]
torch.manual_seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); random.seed(CFG["SEED"])
print(f"train volumes {len(TRAIN_KEYS)}  val volumes {len(VAL_KEYS)}", flush=True)
print(f"run dir {RUN_DIR}", flush=True)
json.dump({k: v for k, v in CFG.items() if isinstance(v, (int, float, str, bool, list))},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=1, default=str)


# ---------------------------------------------------------------- model
class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.InstanceNorm3d(c, affine=True)
        self.c1 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm3d(c, affine=True)
        self.c2 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        h = self.act(self.c1(self.n1(x)))
        h = self.c2(self.n2(h))
        return self.act(x + h)


class SegResLite3D(nn.Module):
    """Compact SegResNet-style 3D U-Net with deep supervision."""
    def __init__(self, in_ch=1, n_cls=4, base=BASE_CH, levels=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(levels)]      # 16,32,64,128
        self.stem = nn.Sequential(nn.Conv3d(in_ch, chs[0], 3, padding=1, bias=False),
                                  nn.InstanceNorm3d(chs[0], affine=True),
                                  nn.LeakyReLU(0.01, inplace=True))
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(levels):
            self.enc.append(ResBlock3D(chs[i]))
            if i < levels - 1:
                self.down.append(nn.Sequential(
                    nn.Conv3d(chs[i], chs[i + 1], 3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(chs[i + 1], affine=True),
                    nn.LeakyReLU(0.01, inplace=True)))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels - 2, -1, -1):
            self.up.append(nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(ResBlock3D(chs[i]))
        self.head = nn.Conv3d(chs[0], n_cls, 1)
        self.aux = nn.ModuleList([nn.Conv3d(chs[i], n_cls, 1) for i in range(1, levels)])

    def forward(self, x):
        feats = []
        h = self.stem(x)
        for i in range(len(self.enc)):
            h = self.enc[i](h)
            feats.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        aux = [self.aux[i](feats[i + 1]) for i in range(len(self.aux))]
        for j, i in enumerate(range(len(self.dec) - 1, -1, -1)):
            h = self.up[j](h)
            h = h + feats[i]
            h = self.dec[j](h)
        return self.head(h), aux


# ---------------------------------------------------------------- data
class PatchDS(Dataset):
    def __init__(self, keys, train=True, n_per_epoch=ITERS * BATCH):
        self.keys = list(keys)
        self.train = train
        self.n = n_per_epoch if train else len(self.keys)
        self.les = []

    def __len__(self):
        return self.n

    def _load(self, k):
        img = np.load(os.path.join(DATA_DIR, k + "_img.npy")).astype(np.float32)
        seg = np.load(os.path.join(DATA_DIR, k + "_seg.npy")).astype(np.int64)
        return img, seg

    def __getitem__(self, i):
        # a patch must never silently fall back to zeros: that is what happens when the patch
        # shape does not fit the volume, and it collapses the model to "predict background".
        k = self.keys[i] if not self.train else random.choice(self.keys)
        img, seg = self._load(k)
        D, H, W = img.shape
        pd, ph, pw = PATCH
        if self.train:
            fg = np.argwhere(np.isin(seg, CFG["LESION_CLASSES"]) | (seg == 1))
            for _ in range(6):
                if len(fg) and random.random() < 0.6:
                    z, y, x = fg[random.randrange(len(fg))]
                    z0 = np.clip(z - pd // 2 + random.randint(-8, 8), 0, max(0, D - pd))
                    y0 = np.clip(y - ph // 2 + random.randint(-16, 16), 0, max(0, H - ph))
                    x0 = np.clip(x - pw // 2 + random.randint(-16, 16), 0, max(0, W - pw))
                else:
                    z0, y0, x0 = random.randint(0, max(0, D - pd)), random.randint(0, max(0, H - ph)), random.randint(0, max(0, W - pw))
                pim, pseg = img[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw], seg[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                if pseg.shape == (pd, ph, pw):
                    break
            else:
                raise RuntimeError(f"patch {PATCH} does not fit volume {img.shape}; refusing to train on zeros")
            if random.random() < 0.5:
                pim, pseg = pim[::-1].copy(), pseg[::-1].copy()
            pim = pim + np.random.randn(*pim.shape).astype(np.float32) * 8.0
        else:
            # full volume, padded to at least the patch size
            pz = max(0, pd - D); py = max(0, ph - H); px = max(0, pw - W)
            pim = np.pad(img, ((0, pz), (0, py), (0, px)))
            pseg = np.pad(seg, ((0, pz), (0, py), (0, px)))
        return (torch.from_numpy(np.ascontiguousarray(pim[None])), torch.from_numpy(np.ascontiguousarray(pseg)),
                k)


def ce_dice_loss(logit, target, w_ce=1.0, w_dice=1.0):
    ce = F.cross_entropy(logit, target, weight=torch.tensor([0.5, 1.0, 2.0, 2.0], device=logit.device))
    prob = logit.softmax(1)
    oh = F.one_hot(target.clamp(0, 3), 4).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (prob * oh).sum(dims)
    denom = prob.sum(dims) + oh.sum(dims)
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return w_ce * ce + w_dice * dice


# ---------------------------------------------------------------- train
model = SegResLite3D(1, len(CFG["CLASSES"]), BASE_CH).to(dev)
print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f} M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ITERS
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / ITERS) * 0.5 * (1 + math.cos(math.pi * s / steps)))
scaler = torch.amp.GradScaler("cuda", enabled=CFG["AMP"])
ds = PatchDS(TRAIN_KEYS, train=True)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)

log_path = os.path.join(RUN_DIR, "log.jsonl")
best, ck_best, ck_last = -1.0, os.path.join(RUN_DIR, "best.pt"), os.path.join(RUN_DIR, "last.pt")
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    te = time.time()
    run = 0.0
    it = 0
    while it < ITERS:
        for x, y, _ in dl:
            if it >= ITERS:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=CFG["AMP"]):
                out, aux = model(x)
                loss = ce_dice_loss(out, y)
                for a in aux:
                    a = F.interpolate(a, size=y.shape[1:], mode="trilinear", align_corners=False)
                    loss = loss + 0.2 * ce_dice_loss(a, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss.detach()); it += 1
            if it % 100 == 0:
                print(f"  ep{ep} it{it} loss={run/it:.4f} ({(time.time()-te)/60:.1f} min)", flush=True)
    rec = dict(epoch=ep, time_min=(time.time() - te) / 60, loss=run / max(it, 1))
    torch.save(model.state_dict(), ck_last)

    # validation and final scoring are handled by eval_nnunet_lite.py, which runs a
    # sliding-window inference once training is complete.
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"epoch {ep} done in {rec['time_min']:.1f} min  loss={rec['loss']:.4f}", flush=True)

torch.save(model.state_dict(), ck_best)
print(f"training finished in {(time.time()-t0)/60:.1f} min", flush=True)
print("NOTE: evaluation is run by eval_nnunet_lite.py", flush=True)
