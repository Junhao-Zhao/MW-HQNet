"""Evaluate the nnU-Net-lite 3D baseline with sliding-window inference.

Reuses the pipeline's own post-processing, metric and summarisation code so the numbers are
directly comparable with every other run in this study.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
ROOT = r"E:\jupyter_work\mwhq_pipeline"
RUN_NAME = os.environ.get("NNL_RUN", "nnunet_lite_3d_fold0")
PATCH = (32, 128, 128)   # (depth, H, W) -- the volume is (D, 192, 192)
BASE_CH = int(os.environ.get("NNL_BASE", 32))

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
CFG.update(DATA_DIR=g["DATA_DIR"], OUT_DIR=os.path.join(ROOT, "runs"), NUM_WORKERS=0, AMP=True)
RUN_DIR = os.path.join(CFG["OUT_DIR"], RUN_NAME)
VAL_KEYS, SPACING, dev = g["VAL_KEYS"], g["SPACING"], g["dev"]
_limit = int(os.environ.get("NNL_VOLS", "0"))
if _limit:
    VAL_KEYS = VAL_KEYS[:_limit]

exec(compile(cells[find("2.4 model")[0]].split("class MWHQNet")[0], "defs", "exec"), g)


# ---- model definition (must match training exactly)
import torch.nn as nn  # noqa: E402


class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.InstanceNorm3d(c, affine=True); self.c1 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm3d(c, affine=True); self.c2 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        h = self.act(self.c1(self.n1(x))); h = self.c2(self.n2(h))
        return self.act(x + h)


class SegResLite3D(nn.Module):
    def __init__(self, in_ch=1, n_cls=4, base=BASE_CH, levels=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(levels)]
        self.stem = nn.Sequential(nn.Conv3d(in_ch, chs[0], 3, padding=1, bias=False),
                                  nn.InstanceNorm3d(chs[0], affine=True),
                                  nn.LeakyReLU(0.01, inplace=True))
        self.enc = nn.ModuleList(); self.down = nn.ModuleList()
        for i in range(levels):
            self.enc.append(ResBlock3D(chs[i]))
            if i < levels - 1:
                self.down.append(nn.Sequential(
                    nn.Conv3d(chs[i], chs[i + 1], 3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(chs[i + 1], affine=True), nn.LeakyReLU(0.01, inplace=True)))
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in range(levels - 2, -1, -1):
            self.up.append(nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(ResBlock3D(chs[i]))
        self.head = nn.Conv3d(chs[0], n_cls, 1)
        self.aux = nn.ModuleList([nn.Conv3d(chs[i], n_cls, 1) for i in range(1, levels)])

    def forward(self, x):
        feats = []; h = self.stem(x)
        for i in range(len(self.enc)):
            h = self.enc[i](h); feats.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        for j, i in enumerate(range(len(self.dec) - 1, -1, -1)):
            h = self.up[j](h); h = h + feats[i]; h = self.dec[j](h)
        return self.head(h)


def starts(n, p, s):
    if n <= p:
        return [0]
    xs = list(range(0, n - p + 1, s))
    if xs[-1] != n - p:
        xs.append(n - p)
    return sorted(set(xs))


@torch.no_grad()
def predict_volume_3d(model, img):
    model.eval()
    D, H, W = img.shape
    pd, ph, pw = PATCH
    pz, py, px = max(0, pd - D), max(0, ph - H), max(0, pw - W)
    x = np.pad(img.astype(np.float32), ((0, pz), (0, py), (0, px)))
    D2, H2, W2 = x.shape
    acc = np.zeros((4, D2, H2, W2), np.float32)
    cnt = np.zeros((1, D2, H2, W2), np.float32)
    for z in starts(D2, pd, pd // 2):
        for y in starts(H2, ph, ph // 2):
            for xx in starts(W2, pw, pw // 2):
                t = torch.from_numpy(x[None, None, z:z + pd, y:y + ph, xx:xx + pw]).to(dev)
                with torch.amp.autocast("cuda", enabled=True):
                    logit = model(t)
                p = torch.softmax(logit.float(), 1)[0].cpu().numpy()
                acc[:, z:z + pd, y:y + ph, xx:xx + pw] += p
                cnt[:, z:z + pd, y:y + ph, xx:xx + pw] += 1
    prob = acc / np.maximum(cnt, 1)
    # postprocess() expects a 4-channel PROBABILITY volume (it calls probs.argmax(0) itself),
    # not a label map -- returning labels here silently produced 2-D "masks".
    return prob[:, :D, :H, :W].astype(np.float32)


model = SegResLite3D(1, len(CFG["CLASSES"]), BASE_CH).to(dev)
ck = os.path.join(RUN_DIR, "best.pt")
model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False))
print(f"loaded {ck}", flush=True)

postprocess, hec_masks, dice = g["postprocess"], g["hec_masks"], g["dice"]
surf_metrics, lesion_detection, summarise = g["surf_metrics"], g["lesion_detection"], g["summarise"]

case_rows, les_rows = [], []
t0 = time.time()
for n, key in enumerate(VAL_KEYS):
    img = np.load(os.path.join(CFG["DATA_DIR"], key + "_img.npy"))
    gt = np.load(os.path.join(CFG["DATA_DIR"], key + "_seg.npy"))
    sp = SPACING[key]
    lab = postprocess(predict_volume_3d(model, img), sp, CFG)
    r = dict(key=key, case=key.rsplit("_s", 1)[0], dose=1.0)
    P, G = hec_masks(lab, CFG), hec_masks(gt, CFG)
    for name in G:
        r[f"dice_{name}"] = dice(P[name], G[name])
        r[f"hd95_{name}"], r[f"nsd_{name}"] = surf_metrics(P[name], G[name], sp)
        r[f"tp_{name}"] = int((P[name] & G[name]).sum())
        r[f"ps_{name}"] = int(P[name].sum())
        r[f"gs_{name}"] = int(G[name].sum())
    lr, fp = lesion_detection(lab, gt, sp, CFG, key)
    r["fp_lesions"] = fp
    case_rows.append(r); les_rows += lr
    if (n + 1) % 20 == 0:
        print(f"  {n+1}/{len(VAL_KEYS)}  {(time.time()-t0)/60:.1f} min", flush=True)

dfc, dfl = pd.DataFrame(case_rows), pd.DataFrame(les_rows)
dfc.to_csv(os.path.join(RUN_DIR, "cases_dose1.0.csv"), index=False)
dfl.to_csv(os.path.join(RUN_DIR, "lesions_dose1.0.csv"), index=False)
summ = summarise(dfc, dfl, CFG)
summary = pd.DataFrame({"dose=1.0": pd.Series(summ)})
summary.to_csv(os.path.join(RUN_DIR, "summary.csv"))
print(f"\nfinished in {(time.time()-t0)/60:.1f} min", flush=True)
print(summary.round(4).to_string(), flush=True)
