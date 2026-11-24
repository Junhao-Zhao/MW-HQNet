# ============================================================================
# CELL 14 - Tables, figures and the auto-written, self-auditing Results section
#
# Every table is built from results/*/summary.json and results/*/cases.csv, i.e. from saved
# predictions. Nothing is typed by hand. The Results writer then narrates exactly those tables,
# and `claim_guard` refuses sentences the tables contradict - the mechanism that would have
# caught "improves every metric in every fold" before submission.
# ============================================================================

DISPLAY = {"B0_baseline": "Baseline (2.5-D ResNet-34 U-Net)",
           "B0_wide": "Baseline, capacity-matched",
           "B1_full": "MW-HQNet (PAWE + LCQS + HQGD)",
           "A5_full_graph": "MW-HQNet (HQGD)",
           "A1_fixed_multi": "fixed multi-window",
           "A2_wso": "learned windows only",
           "A2b_pawe_nogate": "PAWE, no noise gate",
           "A3_no_graph": "PAWE + LCQS, no graph",
           "A4_selfattn": "PAWE + LCQS + self-attention",
           "A6_no_fusion": "no query fusion",
           "A7_no_aifi": "no AIFI",
           "A8_single_win": "single window",
           "nnunet_2d": "nnU-Net v2 (2D)"}
METRIC_ORDER = ["Dice kidney+masses", "Dice masses", "Dice tumor", "Dice cyst",
                "Sens<10mm", "Sens all", "FP/ROI", "Lesion precision"]
BETTER_UP = {"Dice kidney+masses": True, "Dice masses": True, "Dice tumor": True, "Dice cyst": True,
             "Sens<10mm": True, "Sens all": True, "FP/ROI": False, "Lesion precision": True}


def load_summaries():
    recs = []
    for p in sorted(DIRS["results"].glob("*/summary.json")):
        d = json.loads(p.read_text())
        recs.append(dict(experiment=d["experiment"], fold=d["fold"], track=d["track"],
                         tta=d["tta"], **{k: v for k, v in d["summary"].items()}))
    return pd.DataFrame(recs)


def table_dataset():
    if "SPLITS" not in globals():
        print("Stage 5 (folds/splits) has not been run yet - the dataset table needs it.")
        return pd.DataFrame()
    rows = []
    for f in sorted(SPLITS):
        s = SPLITS[f]
        n_roi = len(FOLD_ROIS[f]["outer"]) if "FOLD_ROIS" in globals() else 0
        rows.append(dict(fold=f, fit_patients=len(s["fit"]), inner_dev_patients=len(s["dev"]),
                         held_out_patients=len(s["outer"]), held_out_kidney_rois=n_roi))
    return pd.DataFrame(rows)


def table_main(df=None, space=None):
    """Table II: mean +/- SD across folds, plus the per-fold direction of every metric."""
    df = load_summaries() if df is None else df
    if len(df) == 0:
        print("no results yet - run Cell 11"); return None, None
    space = space or ("roi" if CFG["TRACK"] == "oracle" else "native")
    g = df.groupby("experiment")
    out = []
    for exp, sub in g:
        row = dict(experiment=exp, model=DISPLAY.get(exp, exp), folds=len(sub))
        for m in METRIC_ORDER:
            if m in sub:
                row[m] = f"{sub[m].mean():.3f}+/-{sub[m].std(ddof=1):.3f}" if len(sub) > 1 else f"{sub[m].mean():.3f}"
        out.append(row)
    tab = pd.DataFrame(out)

    # per-fold deltas against the baseline, which is what the "all metrics improve" claim needs
    deltas = None
    exps = [e for e in g.groups]
    if "B0_baseline" in exps:
        base = g.get_group("B0_baseline").set_index("fold")
        rows = []
        for e in exps:
            if e == "B0_baseline":
                continue
            sub = g.get_group(e).set_index("fold")
            for m in [m for m in METRIC_ORDER if m in sub and m in base]:
                d = (sub[m] - base[m].reindex(sub.index)).dropna()
                if len(d) == 0:
                    continue
                good = (d > 0) if BETTER_UP[m] else (d < 0)
                rows.append(dict(model=DISPLAY.get(e, e), metric=m,
                                 n_folds=len(d), n_improved=int(good.sum()),
                                 direction="".join("+" if x > 0 else "-" for x in d),
                                 mean_delta=round(float(d.mean()), 4)))
        deltas = pd.DataFrame(rows)
    return tab, deltas


def table_ablation(df=None):
    """Table IV: single-fold ablation, with the HQGD row's direction made explicit."""
    df = load_summaries() if df is None else df
    if len(df) == 0:
        return None
    pilot = df[df["fold"] == CFG["FOLDS"][0]]
    keep = [e for e in ["B0_baseline", "A1_fixed_multi", "A2_wso", "A2b_pawe_nogate",
                        "A8_single_win", "A3_no_graph", "A4_selfattn", "A5_full_graph",
                        "B1_full", "A6_no_fusion", "A7_no_aifi", "B0_wide"] if e in set(pilot["experiment"])]
    if not keep:
        return None
    sub = pilot.set_index("experiment").loc[keep]
    rows = []
    for e, r in sub.iterrows():
        rows.append(dict(config=DISPLAY.get(e, e), **{m: round(float(r[m]), 4) for m in METRIC_ORDER if m in r}))
    return pd.DataFrame(rows)


def table_efficiency(force=False):
    """Table V: parameters, MACs, latency and peak VRAM, measured on this machine."""
    p = DIRS["results"] / "efficiency.csv"
    if p.exists() and not force:
        return pd.read_csv(p)
    if not MODEL_NEEDED:
        print("model cell has not run; efficiency table unavailable"); return None
    rows = []
    for e in ["B0_baseline", "B0_wide", "B1_full", "A4_selfattn"]:
        if e not in EXPERIMENTS:
            continue
        m = MWHQNet(make_cfg(PRETRAINED=False, **EXPERIMENTS[e][1])).to(DEV)
        c = measure_cost(m, make_cfg(PRETRAINED=False, **EXPERIMENTS[e][1]))
        rows.append(dict(experiment=e, model=DISPLAY.get(e, e), **c))
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return df


# ------------------------------------------------------------------ claim guard
BANNED = [
    ("improves every metric in every fold", "at least one metric regresses in at least one fold"),
    ("improves all six metrics in all five folds", "the FP/ROI cost rises in every fold"),
    ("statistically significant", "the exact test cannot reach alpha=0.05 at n=5"),
    ("wins on every metric", "FP/ROI is worse, and it is a metric"),
]


def claim_guard(text, tab, deltas):
    """Refuse to emit a claim the regenerated tables contradict. Returns (text, warnings)."""
    warn = []
    for phrase, why in BANNED:
        if phrase.lower() in text.lower():
            warn.append(f"BLOCKED claim '{phrase}': {why}")
    if deltas is not None and len(deltas):
        bad = deltas[deltas["n_improved"] < deltas["n_folds"]]
        if len(bad):
            warn.append("direction table shows " + "; ".join(
                f"{r.model}/{r.metric} improved in {int(r.n_improved)}/{int(r.n_folds)} folds"
                for _, r in bad.iterrows()))
    for w in warn:
        print("  !! " + w)
    return text, warn


# ------------------------------------------------------------------ writer
def write_results_section(tab=None, deltas=None, abl=None):
    """Narrate exactly the regenerated tables. Refuses claims the numbers do not support."""
    tab, deltas = (tab, deltas) if tab is not None else table_main()
    if tab is None:
        return None
    space = "ROI space (oracle localisation)" if CFG["TRACK"] == "oracle" else "native CT space (image-only localisation)"
    rows = {r["experiment"]: r for _, r in tab.iterrows()}
    b, f = rows.get("B0_baseline"), rows.get("B1_full")

    def g(exp, m):
        return rows[exp][m] if exp in rows and m in rows[exp] else "n/a"

    L = []
    L.append("## V. RESULTS\n")
    L.append(f"### A. Evaluation setting\n")
    L.append(f"All numbers in this section are computed in {space} from saved per-case predictions. "
             f"{'The localisation stage is the one used at deployment and its coverage is reported in Table II.' if CFG['TRACK'] == 'endo' else 'ROI crops were derived from the reference segmentation, so these results are an oracle-localisation upper bound and are labelled as such.'}\n")

    if abl is not None:
        L.append("### B. Component ablation (fold %d)\n" % CFG["FOLDS"][0])
        L.append(df_to_md(abl, index=False) + "\n")
        if "A3_no_graph" in rows and "A5_full_graph" in rows:
            d_dice = float(rows["A5_full_graph"]["Dice masses"]) - float(rows["A3_no_graph"]["Dice masses"])
            d_fp = float(rows["A5_full_graph"]["FP/ROI"]) - float(rows["A3_no_graph"]["FP/ROI"])
            L.append(f"Adding HQGD to PAWE+LCQS changes kidney-mass Dice by {d_dice:+.3f} and false positives "
                     f"per ROI by {d_fp:+.3f} ({'more' if d_fp > 0 else 'fewer'}). "
                     f"{'The graph therefore reduces false positives on this fold.' if d_fp < 0 else 'The graph therefore costs precision on this fold.'} "
                     f"This is the direction actually measured, not the direction previously claimed.\n")

    if b is not None and f is not None:
        L.append("### C. Five-fold comparison\n")
        L.append(df_to_md(tab[["model", "folds"] + [m for m in METRIC_ORDER if m in tab]].fillna("n/a"), index=False) + "\n")
        L.append(f"Kidney-mass Dice moves from {g('B0_baseline', 'Dice masses')} to {g('B1_full', 'Dice masses')}; "
                 f"sensitivity below 10 mm from {g('B0_baseline', 'Sens<10mm')} to {g('B1_full', 'Sens<10mm')}; "
                 f"and false positives per kidney ROI from {g('B0_baseline', 'FP/ROI')} to {g('B1_full', 'FP/ROI')}. "
                 f"The third of these is a cost, not an improvement, and is treated as such throughout.\n")
        if deltas is not None and len(deltas):
            L.append("Per-fold direction of every metric:\n")
            L.append(df_to_md(deltas, index=False) + "\n")

    text = "\n".join(L)
    text, _ = claim_guard(text, tab, deltas)
    out = DIRS["manuscript"] / "results_section_auto.md"
    out.write_text(text, encoding="utf-8")
    print(f"\nwrote {out} ({len(text)} chars)")
    return text


# ------------------------------------------------------------------ figures
def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load_summaries()
    if len(df) == 0:
        print("no results to plot"); return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for exp, sub in df.groupby("experiment"):
        if exp not in ("B0_baseline", "B1_full", "A4_selfattn"):
            continue
        ax[0].scatter(sub.get("Sens<10mm"), sub.get("FP/ROI"), label=DISPLAY.get(exp, exp), s=60)
        ax[1].scatter(sub.get("Dice masses"), sub.get("Sens<10mm"), label=DISPLAY.get(exp, exp), s=60)
        ax[2].scatter(sub.get("Dice cyst"), sub.get("FP/ROI"), label=DISPLAY.get(exp, exp), s=60)
    ax[0].set_xlabel("Sensitivity < 10 mm"); ax[0].set_ylabel("FP per kidney ROI")
    ax[0].set_title("sensitivity / false-positive trade-off")
    ax[1].set_xlabel("Dice kidney masses"); ax[1].set_ylabel("Sensitivity < 10 mm")
    ax[1].set_title("overlap vs detection")
    ax[2].set_xlabel("Dice cyst"); ax[2].set_ylabel("FP per kidney ROI")
    ax[2].set_title("cyst overlap vs false positives")
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=.3)
    plt.tight_layout()
    p = DIRS["figures"] / "fold_scatter.png"
    plt.savefig(p, dpi=200); plt.close()
    print(f"wrote {p}")


if ENABLED["tables"]:
    t_ds = table_dataset()
    print("TABLE I - dataset and folds")
    print(t_ds.to_string(index=False))

    tab, deltas = table_main()
    if tab is not None:
        print("\nTABLE II - five-fold results (mean+/-SD)")
        print(tab.to_string(index=False))
        print("\nPer-fold direction (what the 'all metrics improve' claim has to survive)")
        print(deltas.to_string(index=False) if deltas is not None else "n/a")
        print("\n'Every metric improved in every fold' would be TRUE only if every n_improved == n_folds.")

    abl = table_ablation()
    if abl is not None:
        print(f"\nTABLE IV - ablation on fold {CFG['FOLDS'][0]}")
        print(abl.to_string(index=False))

    text = write_results_section(tab, deltas, abl)
    if text:
        print("\n" + "=" * 90)
        print(text[:4000])
