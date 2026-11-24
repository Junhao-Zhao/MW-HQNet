"""Paired bootstrap for the remaining headline metrics.

Case-level (Dice masses, Dice tumor, FP/ROI) resample over the 489 pooled cases.
Lesion-level (detection recall, sensitivity <10 mm) resample over pooled lesions,
because that is the unit the metric is defined on.
"""
import os

import numpy as np
import pandas as pd

P = r"E:\jupyter_work\mwhq_pipeline\runs"
OUT = r"E:\segmentation research paper\change"

BASE = ["baseline_unet_fold0", "baseline_unet_fold1_fold1", "baseline_unet_fold2_fold2",
        "baseline_unet_fold3_fold3", "baseline_unet_fold4_fold4"]
FULL = ["mwhqnet_fixed_fold0", "mwhqnet_fixed_fold1_fold1", "mwhqnet_fixed_fold2_fold2",
        "mwhqnet_fixed_fold3_fold3", "mwhqnet_fixed_fold4_fold4"]
rng = np.random.default_rng(42)
N = 1000


def boot(delta, n=N):
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    s = delta[idx].mean(axis=1)
    return s


def report(name, b, f, unit):
    d = f - b
    s = boot(d)
    print(f"\n  {name}   (unit: {unit}, n={len(d)})")
    print(f"    Baseline  {b.mean():.4f}   95% CI [{np.percentile(boot(b),2.5):.4f}, {np.percentile(boot(b),97.5):.4f}]")
    print(f"    MW-HQNet  {f.mean():.4f}   95% CI [{np.percentile(boot(f),2.5):.4f}, {np.percentile(boot(f),97.5):.4f}]")
    print(f"    Δ         {d.mean():+.4f}   95% CI [{np.percentile(s,2.5):+.4f}, {np.percentile(s,97.5):+.4f}]"
          f"   P(Δ>0) = {(s>0).mean():.3f}")


# ---- case-level -------------------------------------------------------------
case_rows = []
for b_run, f_run in zip(BASE, FULL):
    bc = pd.read_csv(os.path.join(P, b_run, "cases_dose1.0.csv"))
    fc = pd.read_csv(os.path.join(P, f_run, "cases_dose1.0.csv"))
    m = bc.merge(fc, on="key", suffixes=("_b", "_f"))
    m["case"] = m["key"].str.rsplit("_s", n=1).str[0]
    agg = m.groupby("case").sum(numeric_only=True)
    for cls in ("masses", "tumor"):
        tp_b, ps_b, gs_b = agg[f"tp_{cls}_b"], agg[f"ps_{cls}_b"], agg[f"gs_{cls}_b"]
        tp_f, ps_f, gs_f = agg[f"tp_{cls}_f"], agg[f"ps_{cls}_f"], agg[f"gs_{cls}_f"]
        agg[f"dice_{cls}_b"] = np.where(ps_b + gs_b > 0, 2 * tp_b / (ps_b + gs_b).clip(lower=1), 1.0)
        agg[f"dice_{cls}_f"] = np.where(ps_f + gs_f > 0, 2 * tp_f / (ps_f + gs_f).clip(lower=1), 1.0)
    agg["fp_b"] = agg["fp_lesions_b"]
    agg["fp_f"] = agg["fp_lesions_f"]
    case_rows.append(agg[["dice_masses_b", "dice_masses_f", "dice_tumor_b", "dice_tumor_f",
                          "fp_b", "fp_f"]])
pool = pd.concat(case_rows)
print("=" * 108)
print("  PAIRED CASE-LEVEL BOOTSTRAP  (489 pooled cases, 1000 resamples)")
print("=" * 108)
for name, cb, cf in (("Dice masses (macro)", "dice_masses_b", "dice_masses_f"),
                     ("Dice tumor (macro)", "dice_tumor_b", "dice_tumor_f"),
                     ("FP lesions per ROI", "fp_b", "fp_f")):
    report(name, pool[cb].to_numpy(), pool[cf].to_numpy(), "case")

# ---- lesion-level -----------------------------------------------------------
les = []
for b_run, f_run in zip(BASE, FULL):
    bl = pd.read_csv(os.path.join(P, b_run, "lesions_dose1.0.csv"))
    fl = pd.read_csv(os.path.join(P, f_run, "lesions_dose1.0.csv"))
    m = bl.merge(fl, on=["key", "cls", "diam_mm"], suffixes=("_b", "_f"))
    les.append(m)
L = pd.concat(les, ignore_index=True)
for c in ("hit_cls_b", "hit_cls_f"):
    L[c] = L[c].fillna(False).astype(float)
print()
print("=" * 108)
print("  PAIRED LESION-LEVEL BOOTSTRAP  (pooled lesions, 1000 resamples)")
print("=" * 108)
report("Lesion detection recall (all sizes)", L["hit_cls_b"].to_numpy(), L["hit_cls_f"].to_numpy(), "lesion")
sub = L[L.diam_mm < 10]
report("Lesion detection recall (<10 mm)", sub["hit_cls_b"].to_numpy(), sub["hit_cls_f"].to_numpy(), "lesion")

L.to_csv(os.path.join(OUT, "pooled_lesions_paired.csv"), index=False)
pool.to_csv(os.path.join(OUT, "pooled_cases_paired.csv"))
print("\nwrote change\\pooled_lesions_paired.csv, change\\pooled_cases_paired.csv")
