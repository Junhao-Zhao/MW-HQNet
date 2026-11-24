"""Rebuild Fig. 3 with a fourth column showing a false-positive case.

Reviewer 1.5 asked for representative true-positive, missed-lesion and false-positive cases.
The previous Fig. 3 had the first two but no false-positive one.

Layout: 4 rows (CT / reference / baseline / full) x 4 columns. Inference is cached to
fig3_cache.npz so that re-rendering does not need the GPU again.

Verified per-slice numbers behind the column labels (baseline -> full overlap with the reference
lesion on the shown slice):
    case_00103_s0  z=31  300 reference voxels   31% -> 90%
    case_00096_s1  z=11  405 reference voxels   66% -> 96%
    case_00025_s0  z=46  7978 reference voxels  99.7% -> 99.7%  (large cyst, both correct)
    case_00208_s1  z=13    0 reference voxels   baseline predicts nothing, full predicts a lesion
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
DATA = r"E:\segmentation research paper\_unzipped\kits23_roi"
RUNS = r"E:\jupyter_work\mwhq_pipeline\runs"
OUT = r"E:\segmentation research paper\change\fig3_v2.png"
CACHEF = r"E:\segmentation research paper\change\fig3_cache.npz"

PANELS = [
    ("case_00103_s0", 31, "TP\n31% \u2192 90%"),
    ("case_00096_s1", 11, "under-covered\n66% \u2192 96%"),
    ("case_00025_s0", 46, "large cyst\nboth correct"),
    ("case_00208_s1", 13, "false positive\nreference empty"),
]


def build_cache():
    import torch
    nb = json.loads(open(NB, encoding="utf-8").read())
    cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}

    def find(m):
        for i, s in cells.items():
            if m in s[:220]:
                return s
        raise KeyError(m)

    g = {"__name__": "__main__", "display": lambda *a, **k: None}
    for m in ["1.1 environment", "2.1 installs", "2.2 preprocessing", "2.3 dataset",
              "2.3b prepare_folds", "2.4 model", "2.5 losses", "2.6 inference", "3.7 config"]:
        exec(compile(find(m), "c", "exec"), g)
    MW, CFG = g["MWHQNet"], g["CFG"]
    dev = torch.device("cuda")
    SP = [2.5, 1.0, 1.0]

    def load(run):
        rc = json.load(open(os.path.join(RUNS, run, "config.json"), encoding="utf-8"))
        ck = torch.load(os.path.join(RUNS, run, "best.pt"), map_location=dev, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        cfg = dict(CFG)
        cfg.update(FOLD=0, NUM_WORKERS=0, SMOKE_TEST=False, D_MODEL=rc["D_MODEL"],
                   AIFI=rc["AIFI"], ENHANCE=rc["ENHANCE"], QUERY=rc["QUERY"])
        m = MW(cfg).to(dev).eval()
        m.load_state_dict(sd)
        return m, cfg

    mb, cb = load("baseline_unet_fold0")
    mf, cf = load("mwhqnet_fixed_fold0")
    out = {}
    for key, z, _ in PANELS:
        img = np.load(os.path.join(DATA, key + "_img.npy"))
        gt = np.load(os.path.join(DATA, key + "_seg.npy"))
        lb = g["postprocess"](g["predict_volume"](mb, img, cb, dose=1.0, tta=True), SP, cb)
        lf = g["postprocess"](g["predict_volume"](mf, img, cf, dose=1.0, tta=True), SP, cf)
        out[key + "_img"] = img[z]
        out[key + "_gt"] = gt[z]
        out[key + "_b"] = lb[z]
        out[key + "_f"] = lf[z]
        print(f"  infer {key} z={z}: ref={int((gt[z] >= 2).sum())} "
              f"base={int((lb[z] >= 2).sum())} full={int((lf[z] >= 2).sum())}", flush=True)
    np.savez_compressed(CACHEF, **out)
    return out


if os.path.exists(CACHEF):
    _z = np.load(CACHEF)
    CACHE = {k: _z[k] for k in _z.files}
    print(f"  loaded cache {CACHEF}")
else:
    CACHE = build_cache()
    print(f"  wrote cache {CACHEF}")

WIN = (-40.0, 160.0)


def norm(sl):
    lo, hi = WIN
    return np.clip((sl.astype(np.float32) - lo) / (hi - lo), 0, 1)


def les(a):
    return (a >= 2).astype(np.float32)


RED, YEL = "#e8232a", "#ffd400"
ROWLAB = ["CT image (window)", "Reference", "Baseline (Dense)", "Full MW-HQNet"]

ncol = len(PANELS)
fig, axes = plt.subplots(4, ncol, figsize=(1.30 * ncol, 4.72), dpi=600)
if ncol == 1:
    axes = axes[:, None]

for j, (key, z, desc) in enumerate(PANELS):
    ct = CACHE[key + "_img"]
    gt = CACHE[key + "_gt"]
    b = CACHE[key + "_b"]
    f = CACHE[key + "_f"]
    for i in range(4):
        ax = axes[i, j]
        ax.imshow(norm(ct), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if i >= 1:
            ax.contour(les(gt), levels=[0.5], colors=RED, linewidths=0.7)
        if i >= 2:
            ax.contour(les(b if i == 2 else f), levels=[0.5], colors=YEL, linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#999")
    case = key.rsplit("_s", 1)[0]
    axes[0, j].set_title(f"{case}\n{desc}", fontsize=4.5, pad=2.2, linespacing=1.25)

for i, lab in enumerate(ROWLAB):
    axes[i, 0].set_ylabel(lab, fontsize=5.0, labelpad=2.5)

handles = [Line2D([], [], color=RED, lw=0.9, label="reference"),
           Line2D([], [], color=YEL, lw=0.9, label="prediction")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=5.0,
           frameon=False, bbox_to_anchor=(0.5, -0.004), handlelength=1.4, columnspacing=1.6)

plt.subplots_adjust(left=0.062, right=0.996, top=0.945, bottom=0.052, wspace=0.028, hspace=0.028)
fig.savefig(OUT, dpi=450, facecolor="white", bbox_inches="tight", pad_inches=0.02)
print(f"  saved -> {OUT}")
