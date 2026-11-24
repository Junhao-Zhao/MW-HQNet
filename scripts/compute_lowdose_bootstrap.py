"""Zero-cost additions: low-dose robustness table and case-level bootstrap confidence intervals.

Both are computed from files that already exist -- no GPU, no re-inference.

Bootstrap design: five-fold CV means every case is a validation case exactly once, so the
per-case CSVs from the five folds can be pooled into a single case-level sample (~489 cases).
Both models see the same cases, so the difference is bootstrapped in a paired fashion.
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
DOSES = ["dose=1.0", "dose=0.5", "dose=0.25"]

# ---------------------------------------------------------------- low dose
print("=" * 108)
print("  LOW-DOSE ROBUSTNESS  (five-fold mean +- SD)")
print("=" * 108)
METRICS = [("Dice masses", "Dice masses"), ("Dice tumor", "Dice tumor"),
           ("Dice cyst", "Dice cyst"), ("Sens all", "Sens all lesions"),
           ("FP / ROI", "FP lesions / kidney ROI")]

rows = []
for label, runs in (("Baseline", BASE), ("MW-HQNet", FULL)):
    for run in runs:
        s = pd.read_csv(os.path.join(P, run, "summary.csv"), index_col=0)
        for dose in DOSES:
            if dose not in s.columns:
                continue
            for name, key in METRICS:
                rows.append(dict(model=label, run=run, dose=dose, metric=name, value=float(s.loc[key, dose])))
ld = pd.DataFrame(rows)
ld.to_csv(os.path.join(OUT, "low_dose_per_fold.csv"), index=False)

print(f"  {'metric':14s} {'dose':6s} {'Baseline':>16s} {'MW-HQNet':>16s} {'Δ':>9s}")
print("  " + "-" * 104)
for name, _ in METRICS:
    for dose in DOSES:
        b = ld[(ld.model == "Baseline") & (ld.dose == dose) & (ld.metric == name)].value
        f = ld[(ld.model == "MW-HQNet") & (ld.dose == dose) & (ld.metric == name)].value
        if len(b) == 0 or len(f) == 0:
            continue
        print(f"  {name:14s} {dose:6s} {b.mean():8.4f} +- {b.std(ddof=1):.4f} "
              f"{f.mean():8.4f} +- {f.std(ddof=1):.4f} {f.mean()-b.mean():+9.4f}")
    print()

# ---------------------------------------------------------------- bootstrap
print("=" * 108)
print("  PAIRED CASE-LEVEL BOOTSTRAP  (95% CI, 1000 resamples, pooled over the five folds)")
print("=" * 108)

pairs = []
for b_run, f_run in zip(BASE, FULL):
    bc = pd.read_csv(os.path.join(P, b_run, "cases_dose1.0.csv"))
    fc = pd.read_csv(os.path.join(P, f_run, "cases_dose1.0.csv"))
    m = bc.merge(fc, on="key", suffixes=("_b", "_f"))
    # a case has two ROIs (s0/s1); aggregate to the case so each patient counts once
    agg = m.groupby(m["key"].str.rsplit("_s", n=1).str[0]).agg(
        tp_b=("tp_masses_b", "sum"), ps_b=("ps_masses_b", "sum"), gs_b=("gs_masses_b", "sum"),
        tp_f=("tp_masses_f", "sum"), ps_f=("ps_masses_f", "sum"), gs_f=("gs_masses_f", "sum"))
    agg["dice_b"] = np.where(agg.ps_b + agg.gs_b > 0, 2 * agg.tp_b / (agg.ps_b + agg.gs_b).clip(lower=1), 1.0)
    agg["dice_f"] = np.where(agg.ps_f + agg.gs_f > 0, 2 * agg.tp_f / (agg.ps_f + agg.gs_f).clip(lower=1), 1.0)
    pairs.append(agg)

pool = pd.concat(pairs)
print(f"  pooled cases: {len(pool)}")
rng = np.random.default_rng(42)
N = 1000
idx = rng.integers(0, len(pool), size=(N, len(pool)))

for name, cols in (("Dice masses (macro)", ("dice_b", "dice_f")),):
    b = pool[cols[0]].to_numpy()
    f = pool[cols[1]].to_numpy()
    bs = b[idx].mean(axis=1)
    fs = f[idx].mean(axis=1)
    ds = fs - bs
    print(f"\n  {name}")
    print(f"    Baseline  {b.mean():.4f}   95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")
    print(f"    MW-HQNet  {f.mean():.4f}   95% CI [{np.percentile(fs,2.5):.4f}, {np.percentile(fs,97.5):.4f}]")
    print(f"    Δ         {ds.mean():+.4f}   95% CI [{np.percentile(ds,2.5):+.4f}, {np.percentile(ds,97.5):+.4f}]"
          f"   P(Δ>0) = {(ds>0).mean():.3f}")

    # save
    pd.DataFrame(dict(boot_baseline=bs, boot_mwhqnet=fs, boot_delta=ds)).to_csv(
        os.path.join(OUT, "bootstrap_dice_masses.csv"), index=False)

print("\nwrote change\\low_dose_per_fold.csv, change\\bootstrap_dice_masses.csv")
