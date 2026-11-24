"""Table II (five-fold mean +- SD) and Table III (paired Wilcoxon) from the finished runs.

Reads only summary.csv, so it reproduces whatever the runs actually recorded -- no re-inference.
Fold mapping: fold-0 uses the 15-epoch full model (mwhqnet_fixed_fold0) so that it is matched
to the 15-epoch baseline; the 40-epoch variant is reported separately as a reference row.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

P = r"E:\jupyter_work\mwhq_pipeline\runs"

BASE = ["baseline_unet_fold0", "baseline_unet_fold1_fold1", "baseline_unet_fold2_fold2",
        "baseline_unet_fold3_fold3", "baseline_unet_fold4_fold4"]
FULL = ["mwhqnet_fixed_fold0", "mwhqnet_fixed_fold1_fold1", "mwhqnet_fixed_fold2_fold2",
        "mwhqnet_fixed_fold3_fold3", "mwhqnet_fixed_fold4_fold4"]

KEYS = [
    ("Dice kidney+masses", "Dice kidney+masses"),
    ("Dice masses", "Dice masses"),
    ("Dice tumor", "Dice tumor"),
    ("Dice cyst", "Dice cyst"),
    ("Sens all lesions", "Sens all lesions"),
    ("Sens<10mm", "Sens [0.0, 10.0)"),
    ("FP / ROI", "FP lesions / kidney ROI"),
]


def load(run):
    df = pd.read_csv(os.path.join(P, run, "summary.csv"), index_col=0)
    col = "dose=1.0" if "dose=1.0" in df.columns else df.columns[0]
    v = df[col]
    out = {}
    for label, prefix in KEYS:
        if label == "Sens<10mm":
            k = next((i for i in v.index if i.startswith(prefix)), None)
        else:
            k = prefix if prefix in v.index else None
        out[label] = float(v[k]) if k else np.nan
    return out


base = pd.DataFrame([load(r) for r in BASE], index=[f"fold{i}" for i in range(5)])
full = pd.DataFrame([load(r) for r in FULL], index=[f"fold{i}" for i in range(5)])

print("=" * 108)
print("  TABLE II  -  five-fold results, dose=1.0, mean +- SD")
print("=" * 108)
rows = []
for label, _ in KEYS:
    b, f = base[label], full[label]
    rows.append(dict(
        Metric=label,
        Baseline=f"{b.mean():.4f} +- {b.std(ddof=1):.4f}",
        MW_HQNet=f"{f.mean():.4f} +- {f.std(ddof=1):.4f}",
        Delta=f"{(f - b).mean():+.4f}",
        Better="MW-HQNet" if (f.mean() > b.mean()) != (label == "FP / ROI") else "Baseline",
    ))
t2 = pd.DataFrame(rows)
print(t2.to_string(index=False))

print()
print("=" * 108)
print("  TABLE III  -  paired Wilcoxon signed-rank across 5 folds (two-sided)")
print("=" * 108)
rows = []
for label, _ in KEYS:
    d = (full[label] - base[label]).values
    signs = "".join("+" if x > 0 else ("-" if x < 0 else "0") for x in d)
    try:
        stat, p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = np.nan, np.nan
    rows.append(dict(Metric=label, per_fold=" ".join(f"{x:+.4f}" for x in d),
                     signs=signs, n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                     p_exact=f"{p:.4f}" if p == p else "n/a"))
t3 = pd.DataFrame(rows)
print(t3.to_string(index=False))

print()
print("  note: with n=5 the smallest attainable two-sided p is 0.0625,")
print("        so no metric can reach p < 0.05 before multiple-comparison correction.")

# Holm correction over the six metrics (FP/ROI included)
ps = [(r["Metric"], float(r["p_exact"])) for r in rows if r["p_exact"] not in ("n/a", "nan")]
ps.sort(key=lambda x: x[1])
m = len(ps)
print()
print("=" * 108)
print("  HOLM-BONFERRONI CORRECTION")
print("=" * 108)
print(f"  {'Metric':22s} {'p_raw':>8s} {'rank':>5s} {'threshold':>10s} {'p_holm':>8s}")
for i, (name, p) in enumerate(ps, 1):
    adj = min(1.0, p * (m - i + 1))
    print(f"  {name:22s} {p:8.4f} {i:5d} {0.05/(m-i+1):10.4f} {adj:8.4f}")

base.to_csv(r"E:\segmentation research paper\change\table2_baseline_per_fold.csv")
full.to_csv(r"E:\segmentation research paper\change\table2_mwhqnet_per_fold.csv")
t2.to_csv(r"E:\segmentation research paper\change\table2_mean_sd.csv", index=False)
t3.to_csv(r"E:\segmentation research paper\change\table3_wilcoxon.csv", index=False)
print("\nwrote change\\table2_*.csv and change\\table3_wilcoxon.csv")

print()
print("=" * 108)
print("  REFERENCE: 40-epoch full model (fold 0 only, not part of Table II)")
print("=" * 108)
r40 = load("mwhqnet_full_fold0")
for label, _ in KEYS:
    print(f"  {label:22s} {r40[label]:.4f}")
