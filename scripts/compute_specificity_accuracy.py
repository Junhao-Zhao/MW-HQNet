"""Fill the metric gaps the comparison papers report.

Specificity and voxel accuracy need the total number of voxels in each ROI, which is not stored
in the CSVs -- but it is just the product of the ground-truth mask shape, which we can read from
the ROI cache without any GPU inference.

    FP = ps - tp        FN = gs - tp        TN = N - ps - gs + tp
    Specificity = TN / (TN + FP)
    Accuracy    = (tp + TN) / N
"""
import os

import numpy as np
import pandas as pd

P = r"E:\jupyter_work\mwhq_pipeline\runs"
ROI = r"E:\segmentation research paper\_unzipped\kits23_roi"
OUT = r"E:\segmentation research paper\change"

RUNS = {
    "Baseline": ["baseline_unet_fold0", "baseline_unet_fold1_fold1", "baseline_unet_fold2_fold2",
                 "baseline_unet_fold3_fold3", "baseline_unet_fold4_fold4"],
    "MW-HQNet": ["mwhqnet_fixed_fold0", "mwhqnet_fixed_fold1_fold1", "mwhqnet_fixed_fold2_fold2",
                 "mwhqnet_fixed_fold3_fold3", "mwhqnet_fixed_fold4_fold4"],
}
CLASSES = ["kidney+masses", "masses", "tumor", "cyst"]

shape_cache = {}


def total_voxels(key):
    if key not in shape_cache:
        shape_cache[key] = int(np.prod(np.load(os.path.join(ROI, key + "_seg.npy"), mmap_mode="r").shape))
    return shape_cache[key]


rows = []
for model, runs in RUNS.items():
    for fold, run in enumerate(runs):
        df = pd.read_csv(os.path.join(P, run, "cases_dose1.0.csv"))
        df["N"] = [total_voxels(k) for k in df["key"]]
        for cls in CLASSES:
            tp = df[f"tp_{cls}"].to_numpy(float)
            ps = df[f"ps_{cls}"].to_numpy(float)
            gs = df[f"gs_{cls}"].to_numpy(float)
            N = df["N"].to_numpy(float)
            FP = ps - tp
            FN = gs - tp
            TN = N - ps - gs + tp
            # micro (pooled)
            sTP, sFP, sFN, sTN = tp.sum(), FP.sum(), FN.sum(), TN.sum()
            spec = sTN / (sTN + sFP) if (sTN + sFP) else np.nan
            acc = (sTP + sTN) / (sTP + sTN + sFP + sFN)
            rows.append(dict(model=model, fold=fold, cls=cls,
                             vox_Specificity=spec, vox_Accuracy=acc,
                             macro_Specificity=float(np.mean(TN / np.maximum(TN + FP, 1))),
                             macro_Accuracy=float(np.mean((tp + TN) / N))))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "specificity_accuracy_per_fold.csv"), index=False)

print("=" * 104)
print("  SPECIFICITY + VOXEL ACCURACY  (micro / macro, five folds)")
print("=" * 104)
for cls in CLASSES:
    print(f"\n  类别 {cls}:")
    print(f"    {'model':10s} {'Spec(micro)':>18s} {'Acc(micro)':>18s} {'Acc(macro)':>18s}")
    for model in RUNS:
        d = df[(df.model == model) & (df.cls == cls)]
        print(f"    {model:10s} "
              f"{d.vox_Specificity.mean():8.6f} +- {d.vox_Specificity.std(ddof=1):.6f} "
              f"{d.vox_Accuracy.mean():8.4f} +- {d.vox_Accuracy.std(ddof=1):.4f} "
              f"{d.macro_Accuracy.mean():8.4f} +- {d.macro_Accuracy.std(ddof=1):.4f}")

print()
print("=" * 104)
print("  vs DAMFF-Net (KiTS23, voxel level): Accuracy 99.01%, Precision 84.33%, Recall 89.05%")
print("=" * 104)
for model in RUNS:
    d = df[(df.model == model) & (df.cls == "tumor")]
    print(f"  {model:10s} tumor voxel accuracy {d.vox_Accuracy.mean()*100:.2f}%  "
          f"(specificity {d.vox_Specificity.mean():.6f})")
print("\nwrote change\\specificity_accuracy_per_fold.csv")
