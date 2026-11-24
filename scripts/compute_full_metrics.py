"""Full five-fold metric suite for the finished runs.

Produces, per model per fold:
  voxel level   : Dice (from summary.csv), IoU = Dice/(2-Dice)
  lesion level  : TP/FP/FN -> Precision / Recall / F1   (from lesions + cases CSVs)
  case level    : TP/FP/FN/TN -> Accuracy / Precision / Recall / F1
then aggregates as mean +- SD and writes CSVs consumed by the markdown documents.
"""
import json
import os

import numpy as np
import pandas as pd

P = r"E:\jupyter_work\mwhq_pipeline\runs"
OUT = r"E:\segmentation research paper\change"

RUNS = {
    "Baseline": ["baseline_unet_fold0", "baseline_unet_fold1_fold1", "baseline_unet_fold2_fold2",
                 "baseline_unet_fold3_fold3", "baseline_unet_fold4_fold4"],
    "MW-HQNet": ["mwhqnet_fixed_fold0", "mwhqnet_fixed_fold1_fold1", "mwhqnet_fixed_fold2_fold2",
                 "mwhqnet_fixed_fold3_fold3", "mwhqnet_fixed_fold4_fold4"],
}
CLASSES = ["kidney+masses", "masses", "tumor", "cyst"]


def summ(run):
    df = pd.read_csv(os.path.join(P, run, "summary.csv"), index_col=0)
    col = "dose=1.0" if "dose=1.0" in df.columns else df.columns[0]
    return df[col]


def pick(idx, prefix):
    return next((k for k in idx if k.startswith(prefix)), None)


def lesion_metrics(run):
    lf = pd.read_csv(os.path.join(P, run, "lesions_dose1.0.csv"))
    cf = pd.read_csv(os.path.join(P, run, "cases_dose1.0.csv"))
    tp = int(lf["hit_cls"].fillna(False).astype(bool).sum())
    fn = len(lf) - tp
    fp = int(cf["fp_lesions"].sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    return dict(les_TP=tp, les_FP=fp, les_FN=fn,
                les_Precision=prec, les_Recall=rec, les_F1=f1)


def case_metrics(run, cls="tumor"):
    cf = pd.read_csv(os.path.join(P, run, "cases_dose1.0.csv"))
    agg = cf.groupby("case")[[f"tp_{cls}", f"ps_{cls}", f"gs_{cls}"]].sum()
    has_gt = agg[f"gs_{cls}"] > 0
    has_pred = agg[f"ps_{cls}"] > 0
    det = agg[f"tp_{cls}"] > 0
    TP = int((has_gt & det).sum()); FN = int((has_gt & ~det).sum())
    FP = int((~has_gt & has_pred).sum()); TN = int((~has_gt & ~has_pred).sum())
    n = TP + FP + FN + TN
    acc = (TP + TN) / n if n else np.nan
    prec = TP / (TP + FP) if TP + FP else np.nan
    rec = TP / (TP + FN) if TP + FN else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    return dict(case_TP=TP, case_FP=FP, case_FN=FN, case_TN=TN,
                case_Accuracy=acc, case_Precision=prec, case_Recall=rec, case_F1=f1)


rows = []
for model, runs in RUNS.items():
    for fold, run in enumerate(runs):
        s = summ(run)
        r = dict(model=model, fold=fold, run=run)
        for c in CLASSES:
            d = float(s[f"Dice {c}"])
            r[f"Dice_{c}"] = d
            r[f"IoU_{c}"] = d / (2 - d) if d else np.nan
        k10 = pick(s.index, "Sens [0.0, 10.0)")
        r["Sens_all"] = float(s["Sens all lesions"])
        r["Sens_lt10"] = float(s[k10])
        r["FP_ROI"] = float(s["FP lesions / kidney ROI"])
        r.update(lesion_metrics(run))
        r.update(case_metrics(run, "tumor"))
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "all_metrics_per_fold.csv"), index=False)

METRICS = [
    ("Dice masses (vox)", "Dice_masses"),
    ("Dice tumor (vox)", "Dice_tumor"),
    ("Dice cyst (vox)", "Dice_cyst"),
    ("IoU masses (vox)", "IoU_masses"),
    ("IoU tumor (vox)", "IoU_tumor"),
    ("IoU cyst (vox)", "IoU_cyst"),
    ("Precision (lesion)", "les_Precision"),
    ("Recall (lesion)", "les_Recall"),
    ("F1 (lesion)", "les_F1"),
    ("Sens all", "Sens_all"),
    ("Sens <10mm", "Sens_lt10"),
    ("FP / ROI", "FP_ROI"),
    ("Accuracy (case)", "case_Accuracy"),
    ("Precision (case)", "case_Precision"),
    ("Recall (case)", "case_Recall"),
    ("F1 (case)", "case_F1"),
]

agg = []
for label, col in METRICS:
    b = df[df.model == "Baseline"][col]
    f = df[df.model == "MW-HQNet"][col]
    agg.append(dict(Metric=label,
                    Baseline_mean=b.mean(), Baseline_sd=b.std(ddof=1),
                    MWHQNet_mean=f.mean(), MWHQNet_sd=f.std(ddof=1),
                    Delta=(f - b).mean(),
                    per_fold_delta=" ".join(f"{x:+.4f}" for x in (f.values - b.values))))
agg = pd.DataFrame(agg)
agg.to_csv(os.path.join(OUT, "five_fold_summary.csv"), index=False)

print("=" * 118)
print("  FIVE-FOLD SUMMARY  (dose=1.0)")
print("=" * 118)
print(f"  {'Metric':22s} {'Baseline':>18s} {'MW-HQNet':>18s} {'Delta':>9s}")
print("  " + "-" * 114)
for _, r in agg.iterrows():
    print(f"  {r.Metric:22s} {r.Baseline_mean:8.4f} +- {r.Baseline_sd:.4f} "
          f"{r.MWHQNet_mean:8.4f} +- {r.MWHQNet_sd:.4f} {r.Delta:+9.4f}")

print()
print("=" * 118)
print("  LESION-LEVEL COUNTS PER FOLD")
print("=" * 118)
for model in ("Baseline", "MW-HQNet"):
    sub = df[df.model == model]
    print(f"  {model}")
    for _, r in sub.iterrows():
        print(f"    fold{int(r.fold)}  GT={int(r.les_TP + r.les_FN):3d}  "
              f"TP={int(r.les_TP):3d}  FP={int(r.les_FP):3d}  FN={int(r.les_FN):3d}  "
              f"P={r.les_Precision:.4f}  R={r.les_Recall:.4f}  F1={r.les_F1:.4f}")

print()
print("=" * 118)
print("  CASE-LEVEL (tumor present?) PER FOLD")
print("=" * 118)
for model in ("Baseline", "MW-HQNet"):
    sub = df[df.model == model]
    print(f"  {model}")
    for _, r in sub.iterrows():
        print(f"    fold{int(r.fold)}  TP={int(r.case_TP):3d} FP={int(r.case_FP):3d} "
              f"FN={int(r.case_FN):3d} TN={int(r.case_TN):3d}  "
              f"Acc={r.case_Accuracy:.4f}  P={r.case_Precision:.4f}  R={r.case_Recall:.4f}")

print("\nwrote change\\all_metrics_per_fold.csv, change\\five_fold_summary.csv")
