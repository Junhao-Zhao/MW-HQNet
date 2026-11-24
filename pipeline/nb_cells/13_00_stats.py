# ============================================================================
# CELL 13 - Paired statistics done properly (reviewer Sec.1.4, Sec.8)
#
# Rules enforced here:
#   * patients, not folds and not lesions, are the resampling unit
#   * the exact Wilcoxon test is used, and the sign pattern is printed next to the p-value so a
#     reader can see immediately why n=5 has a floor of 0.0625
#   * Holm correction is applied only over declared confirmatory metrics
#   * effect sizes and 95% intervals are reported, because a p-value alone cannot carry n=5
# ============================================================================

def exact_wilcoxon_p(d):
    """Exact two-sided Wilcoxon signed-rank p by sign enumeration (no ties assumed)."""
    d = np.asarray(d, dtype=float); d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    total = ranks.sum()
    w = min(ranks[d > 0].sum(), total - ranks[d > 0].sum())
    hits = 0
    for mask in range(1 << n):
        s = sum(ranks[i] for i in range(n) if (mask >> i) & 1)
        if min(s, total - s) <= w + 1e-9:
            hits += 1
    return hits / float(1 << n)


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def bootstrap_ci(x, n=None, seed=42, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    n = n or CFG["BOOTSTRAP_N"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boots = stat(x[idx], axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def patient_dice(df, group):
    """Case-level Dice for one HEC class, pooling a patient's kidney ROIs (one observation each)."""
    a = df.groupby("case")[[f"tp_{group}", f"ps_{group}", f"gs_{group}"]].sum()
    den = (a[f"ps_{group}"] + a[f"gs_{group}"]).replace(0, np.nan)
    return (2 * a[f"tp_{group}"] / den).fillna(1.0)


def paired_stats(exp_a, exp_b, metric, space=None, seed=42):
    """Paired comparison of two finished experiments on exactly the same patients."""
    def load(exp):
        frames = []
        for p in sorted(DIRS["results"].glob(f"{exp}_fold*/cases.csv")):
            d = pd.read_csv(p)
            frames.append(d)
        if not frames:
            return None
        d = pd.concat(frames, ignore_index=True)
        return d[d["space"] == space] if space else d

    A, B = load(exp_a), load(exp_b)
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return None
    if metric in ("Dice masses", "Dice tumor", "Dice cyst", "Dice kidney+masses"):
        g = metric.replace("Dice ", "")
        a = patient_dice(A, g).rename("a"); b = patient_dice(B, g).rename("b")
    elif metric == "FP/ROI":
        a = A.groupby("case")["fp_lesions"].mean().rename("a")
        b = B.groupby("case")["fp_lesions"].mean().rename("b")
    else:
        return None
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    d = (j["b"] - j["a"]).to_numpy()
    lo, hi = bootstrap_ci(d, seed=seed)
    dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
    pos = int((d > 0).sum()); neg = int((d < 0).sum())
    return dict(metric=metric, space=space or "all", exp_a=exp_a, exp_b=exp_b, n_patients=len(d),
                mean_a=round(float(j["a"].mean()), 4), mean_b=round(float(j["b"].mean()), 4),
                mean_delta=round(float(d.mean()), 4),
                ci95_lo=round(lo, 4), ci95_hi=round(hi, 4),
                cohen_dz=round(dz, 3) if np.isfinite(dz) else np.nan,
                rank_biserial=round((pos - neg) / max(pos + neg, 1), 3),
                n_pos=pos, n_neg=neg, p_wilcoxon=round(exact_wilcoxon_p(d), 5))


def lesion_sensitivity_stats(exp, space=None, seed=42):
    """Sensitivity by size bin with a PATIENT-cluster bootstrap interval."""
    frames = [pd.read_csv(p) for p in sorted(DIRS["results"].glob(f"{exp}_fold*/lesions.csv"))]
    if not frames:
        return None
    L = pd.concat(frames, ignore_index=True)
    if space:
        L = L[L["space"] == space]
    gt = L[(L["gt_id"] > 0)].copy()
    gt["bin"] = pd.cut(gt["diam_mm"], CFG["SIZE_BINS_MM"], right=False)
    out = []
    for name, sub in gt.groupby("bin", observed=True):
        pats = sub.groupby("case")["matched"].agg(["sum", "count"])
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(min(CFG["BOOTSTRAP_N"], 4000)):
            s = pats.iloc[rng.integers(0, len(pats), len(pats))]
            boots.append(s["sum"].sum() / max(s["count"].sum(), 1))
        out.append(dict(experiment=exp, bin=str(name), n_lesions=int(len(sub)), n_patients=int(len(pats)),
                        sensitivity=round(float(sub["matched"].mean()), 4),
                        ci95_lo=round(float(np.percentile(boots, 2.5)), 4),
                        ci95_hi=round(float(np.percentile(boots, 97.5)), 4),
                        class_agnostic_anyoverlap=round(float(sub["hit_any_class"].mean()), 4)))
    return pd.DataFrame(out)


def compute_stats(pairs=None):
    """Build results/stats_paired.csv for the comparisons the paper actually claims."""
    if pairs is None:
        pairs = [("B0_baseline", "B1_full"),
                 ("A3_no_graph", "A4_selfattn"),
                 ("A5_full_graph", "A4_selfattn"),
                 ("B0_baseline", "B0_wide")]
    space = "roi" if CFG["TRACK"] == "oracle" else "native"
    metrics = ["Dice masses", "Dice tumor", "Dice cyst", "FP/ROI"]
    rows = []
    for a, b in pairs:
        for m in metrics:
            r = paired_stats(a, b, m, space=space)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        # Holm only across the declared confirmatory family: Dice masses and Sens<10mm of A->B
        conf = df[(df.exp_a == "B0_baseline") & (df.exp_b == "B1_full") & df.metric.isin(["Dice masses"])]
        df["p_holm_confirmatory"] = np.nan
        if len(conf):
            df.loc[conf.index, "p_holm_confirmatory"] = np.round(holm(conf["p_wilcoxon"].to_numpy()), 5)
        df.to_csv(DIRS["results"] / "stats_paired.csv", index=False)
        print(df.to_string(index=False))
    else:
        print("no finished experiments to compare yet - run Cell 11 first")

    sens = lesion_sensitivity_stats("B1_full", space=space)
    if sens is not None:
        sens.to_csv(DIRS["results"] / "sensitivity_bins.csv", index=False)
        print("\nB1 sensitivity by equivalent spherical diameter (patient-cluster CI):")
        print(sens.to_string(index=False))
    return df


print("statistics helpers ready:  compute_stats()  after the runs finish")
print("reading of a result: with 5 folds the exact p-value floor is 0.0625; report the")
print("interval and the effect size as the primary evidence and the p-value as a bound.")

# After a pilot or five-fold sweep, compute the paired statistics automatically.
if ENABLED.get("run_all") and len(list(DIRS["results"].glob("*/summary.json"))) >= 2:
    print("\n" + "=" * 88)
    print("runs finished -> computing paired statistics")
    print("=" * 88, flush=True)
    try:
        compute_stats()
    except Exception as _e:
        print(f"compute_stats() failed: {type(_e).__name__}: {_e}")
        print("run compute_stats() manually once every run is complete")
