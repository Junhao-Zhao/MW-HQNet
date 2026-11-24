# ============================================================================
# CELL 2b -Rebuild Tables I/II/III from the audited artefacts and diff them
#           against the numbers printed in the manuscript.
# Outputs: audit/table_I_audit.csv, table_II_audit.csv, table_III_audit.csv,
#          audit/contradictions.csv, audit/AUDIT_REPORT.md, audit/provenance.csv
# ============================================================================
if ENABLED["audit"]:

    # ---------------------------------------------------------------- helpers
    def exact_wilcoxon_p(d):
        """Exact two-sided Wilcoxon signed-rank p by full sign enumeration.

        Exact for n <= 15 without ties. For n = 5 with all five differences in the same
        direction this returns 2/32 = 0.0625 - the floor the reviewer quotes.
        """
        d = np.asarray(d, dtype=float)
        d = d[d != 0]
        n = len(d)
        if n == 0:
            return 1.0
        ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
        total = ranks.sum()
        w_pos = ranks[d > 0].sum()
        w = min(w_pos, total - w_pos)
        hits = 0
        for mask in range(1 << n):
            s = sum(ranks[i] for i in range(n) if (mask >> i) & 1)
            if min(s, total - s) <= w + 1e-9:
                hits += 1
        return hits / float(1 << n)

    def holm(pvals):
        p = np.asarray(pvals, dtype=float)
        order = np.argsort(p)
        m = len(p)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (m - rank) * p[idx]))
            adj[idx] = running
        return adj

    # df_to_md comes from Stage 0 - one implementation, used by every table in this notebook.
    METRICS = ["Dice masses", "Dice tumor", "Dice cyst", "Sens<10mm", "Sens all", "FP/ROI"]

    # ---------------------------------------------------- numbers in the paper
    PAPER_T1 = pd.DataFrame([
        dict(config="Baseline U-Net",         **{"Dice masses": 0.681, "Dice tumor": 0.603, "Dice cyst": 0.287, "Sens<10mm": 0.142, "Sens all": 0.498, "FP/ROI": 1.52}),
        dict(config="+ PAWE only",            **{"Dice masses": 0.712, "Dice tumor": 0.631, "Dice cyst": 0.311, "Sens<10mm": 0.169, "Sens all": 0.531, "FP/ROI": 1.44}),
        dict(config="+ PAWE+LCQS (no graph)", **{"Dice masses": 0.738, "Dice tumor": 0.661, "Dice cyst": 0.335, "Sens<10mm": 0.231, "Sens all": 0.579, "FP/ROI": 1.61}),
        dict(config="Full MW-HQNet (+HQGD)",  **{"Dice masses": 0.755, "Dice tumor": 0.682, "Dice cyst": 0.356, "Sens<10mm": 0.215, "Sens all": 0.565, "FP/ROI": 1.91}),
    ])
    PAPER_T2 = dict(
        Baseline=dict(mean=[0.718, 0.652, 0.390, 0.236, 0.636, 1.412],
                      sd=[0.022, 0.027, 0.054, 0.064, 0.069, 0.170]),
        **{"MW-HQNet": dict(mean=[0.729, 0.664, 0.396, 0.307, 0.675, 1.911],
                            sd=[0.025, 0.024, 0.037, 0.072, 0.074, 0.039])})
    PAPER_T3 = dict(zip(METRICS, [(0.0625, 0.375), (0.0625, 0.375), (0.6250, 0.625),
                                  (0.0625, 0.375), (0.0625, 0.375), (0.0625, 0.375)]))

    # ---------------------------- fold 0: the ablation workbook is the record
    wb = pd.read_excel(CFG["LEGACY_XLSX"], sheet_name="Fold0 Results")
    wb.columns = [str(c).strip() for c in wb.columns]
    wb = wb.rename(columns={"Sens [0,10mm)": "Sens<10mm", "Sens [0,10mm]": "Sens<10mm",
                            "Sens all lesions": "Sens all", "FP lesions/ROI": "FP/ROI"})
    for c in METRICS:
        wb[c] = pd.to_numeric(wb[c], errors="coerce")
    cfgcol = "Config" if "Config" in wb.columns else wb.columns[0]

    def _row(pat, regex=False):
        s = wb[wb[cfgcol].astype(str).str.contains(pat, case=False, na=False, regex=regex)]
        return s.iloc[0] if len(s) else None

    fold0 = {"Baseline": _row("Baseline"), "MW-HQNet": _row("Full")}
    fold0_pawe = _row("PAWE only")
    fold0_nograph = _row("LCQS")
    wb_note = str(fold0_nograph["Notes"]) if fold0_nograph is not None and "Notes" in wb.columns else ""

    # ------------------------- folds 1-4: the selected-checkpoint epoch records
    primary = aud_nb[aud_nb["source"].astype(str).str.startswith("epoch-log")].copy()
    tta = aud_nb[aud_nb["source"].astype(str).str.startswith("cell10")].copy()

    # every value any supplied file supports, for the impossibility test
    all_vals = {m: {k: [] for k in ["Baseline", "MW-HQNet"]} for m in METRICS}
    for model in ["Baseline", "MW-HQNet"]:
        for m in METRICS:
            all_vals[m][model].append((0, float(fold0[model][m]), "xlsx"))
        for _, r in primary[primary.model == model].iterrows():
            for m in METRICS:
                all_vals[m][model].append((int(r["fold"]), float(r[m]), "epoch-log"))
        for _, r in tta[tta.model == model].iterrows():
            for m in METRICS:
                if pd.notna(r[m]):
                    all_vals[m][model].append((int(r["fold"]), float(r[m]), "cell10-TTA"))

    per_fold = {}
    for model in ["Baseline", "MW-HQNet"]:
        rows = []
        for f in [0, 1, 2, 3, 4]:
            if f == 0:
                r = dict(fold=0, source="fold0-workbook(xlsx)")
                for m in METRICS:
                    r[m] = float(fold0[model][m])
            else:
                sub = primary[(primary.model == model) & (primary.fold == f)]
                if sub.empty:
                    r = dict(fold=f, source="MISSING", **{m: float("nan") for m in METRICS})
                else:
                    r = dict(fold=f, source="epoch-log(best Dice masses)")
                    for m in METRICS:
                        r[m] = float(sub.iloc[0][m])
            rows.append(r)
        per_fold[model] = pd.DataFrame(rows)

    # ------------------------------------------- reconstructed Table II (+deltas)
    t2 = []
    for model in ["Baseline", "MW-HQNet"]:
        df = per_fold[model]
        t2.append(dict(model=model,
                       **{m: round(float(df[m].mean()), 4) for m in METRICS},
                       **{f"sd_{m}": round(float(df[m].std(ddof=1)), 4) for m in METRICS}))
    t2 = pd.DataFrame(t2).set_index("model")
    t2.to_csv(DIRS["audit"] / "table_II_audit.csv")

    delta = per_fold["MW-HQNet"][METRICS].to_numpy() - per_fold["Baseline"][METRICS].to_numpy()
    delta_df = pd.DataFrame(delta, columns=METRICS)
    delta_df.insert(0, "fold", [0, 1, 2, 3, 4])
    delta_df.to_csv(DIRS["audit"] / "per_fold_deltas.csv", index=False)

    # ------------------------------------------------ Table III (statistics)
    stats_rows = []
    for m in METRICS:
        d = delta_df[m].to_numpy()
        stats_rows.append(dict(metric=m, n_nonzero=int((d != 0).sum()),
                               direction="".join("+" if x > 0 else ("-" if x < 0 else "0") for x in d),
                               mean_delta=round(float(np.mean(d)), 5),
                               p_exact=round(exact_wilcoxon_p(d), 5),
                               p_paper=PAPER_T3[m][0]))
    t3 = pd.DataFrame(stats_rows)
    t3["p_holm"] = np.round(holm(t3["p_exact"].to_numpy()), 5)
    t3["p_paper_holm"] = [PAPER_T3[m][1] for m in t3["metric"]]
    t3.to_csv(DIRS["audit"] / "table_III_audit.csv", index=False)

    # ------------------------------------------------------- Table I (fold 0)
    t1_rows = [dict(config="Baseline U-Net", **{m: float(fold0["Baseline"][m]) for m in METRICS}),
               dict(config="Full MW-HQNet (+HQGD)", **{m: float(fold0["MW-HQNet"][m]) for m in METRICS})]
    if fold0_pawe is not None:
        t1_rows.insert(1, dict(config="+ PAWE only", **{m: float(fold0_pawe[m]) for m in METRICS}))
    if fold0_nograph is not None:
        t1_rows.insert(2, dict(config="+ PAWE+LCQS (no graph)", **{m: float(fold0_nograph[m]) for m in METRICS}))
    t1_aud = pd.DataFrame(t1_rows)
    t1_aud.to_csv(DIRS["audit"] / "table_I_audit.csv", index=False)

    # ---------------------------------------------- independent CSV spot-check
    csv_check = None
    try:
        with zipfile.ZipFile(CFG["LEGACY_BASELINE_ZIP"]) as z:
            got = {}
            for entry in z.namelist():
                if entry.endswith(("cases_dose1.0.csv", "lesions_dose1.0.csv")):
                    got[os.path.basename(entry)] = pd.read_csv(io.BytesIO(z.read(entry)))
        dc, dl = got.get("cases_dose1.0.csv"), got.get("lesions_dose1.0.csv")
        if dc is not None:
            agg = dc.groupby("case")[["tp_masses", "ps_masses", "gs_masses"]].sum()
            den = (agg["ps_masses"] + agg["gs_masses"]).replace(0, np.nan)
            rec = dict(source="fold-4 baseline per-case CSVs (recomputed)", fold=4,
                       **{"Dice masses": float((2 * agg["tp_masses"] / den).fillna(1.0).mean()),
                          "FP/ROI": float(dc["fp_lesions"].mean())})
            if dl is not None and "diam_mm" in dl.columns:
                les = dl.dropna(subset=["diam_mm"]).copy()
                les["bin"] = pd.cut(les["diam_mm"], CFG["SIZE_BINS_MM"], right=False)
                small = les[les["bin"].cat.codes == 0]
                rec["Sens<10mm"] = float(small["hit_any"].mean()) if len(small) else np.nan
                rec["Sens all"] = float(les["hit_any"].mean())
            csv_check = pd.DataFrame([rec])
            print("\nIndependent recomputation from the fold-4 baseline per-case CSVs:")
            print(csv_check.round(4).to_string(index=False))
            print("(these CSVs contain the TTA-based final evaluation, the protocol the manuscript does NOT use)")
    except Exception as e:
        print(f"  (CSV spot-check skipped: {type(e).__name__}: {e})")

    # =================================================== contradiction ledger
    C = []

    def add(cid, item, severity, claim, evidence, action):
        C.append(dict(id=cid, reviewer_item=item, severity=severity,
                      manuscript_claim=claim, audited_evidence=evidence, required_action=action))

    # II.1 mean/SD reproducibility
    for model in ["Baseline", "MW-HQNet"]:
        for m in METRICS:
            aud_m = float(per_fold[model][m].mean())
            aud_s = float(per_fold[model][m].std(ddof=1))
            k = METRICS.index(m)
            pap_m, pap_s = PAPER_T2[model]["mean"][k], PAPER_T2[model]["sd"][k]
            if abs(aud_m - pap_m) > 0.004 or abs(aud_s - pap_s) > 0.006:
                add(f"T2-{model}-{m}", "1.3/1.4", "MAJOR",
                    f"Table II {model} {m} = {pap_m}{pap_s}",
                    f"artefact-derived = {aud_m:.4f}{aud_s:.4f}",
                    "replace with the regenerated value; the printed value is not reproducible from the supplied files")

    # II.2 the "all 30 comparisons improve" sentence
    neg = [(m, int(delta_df.loc[delta_df[m] < 0, "fold"].iloc[0])) for m in METRICS if (delta_df[m] < 0).any()]
    n_negative = int((delta_df[METRICS].to_numpy() < 0).sum())
    add("CROSS-1", "1.3 CRITICAL", "CRITICAL",
        "V-B: the full model improves every one of the six metrics in every fold (30/30 paired comparisons)",
        f"{n_negative}/30 fold-metric deltas are negative: "
        + (", ".join(f"{m} first at fold {f}" for m, f in neg) if neg else "none")
        + f"; FP/ROI increases (is worse) in {int((delta_df['FP/ROI'] > 0).sum())}/5 folds",
        "delete the sentence; report the per-metric per-fold direction table and describe benefit and cost separately")

    # II.3 Wilcoxon p reproducibility
    for _, r in t3.iterrows():
        if abs(float(r["p_exact"]) - float(r["p_paper"])) > 1e-6:
            add(f"T3-{r['metric']}", "1.4 CRITICAL", "CRITICAL",
                f"Table III {r['metric']}: p = {r['p_paper']}",
                f"exact two-sided p from the audited paired vector with signs {r['direction']} = {r['p_exact']}",
                "recompute from regenerated case-level CSVs and print the sign pattern beside the p-value")

    # II.4 statistically impossible SD (uses every value any file supports)
    for model in ["Baseline", "MW-HQNet"]:
        for m in METRICS:
            k = METRICS.index(m)
            pap_m, pap_s = PAPER_T2[model]["mean"][k], PAPER_T2[model]["sd"][k]
            if pap_s <= 0:
                continue
            worst_z, worst = 0.0, None
            for f, v, src in all_vals[m][model]:
                z = abs(v - pap_m) / pap_s
                if z > worst_z:
                    worst_z, worst = z, (f, v, src)
            if worst_z > 3.5:
                add(f"IMPOSSIBLE-{model}-{m}", "1.4 CRITICAL", "CRITICAL",
                    f"Table II {model} {m} = {pap_m}{pap_s}",
                    f"fold {worst[0]} from {worst[2]} is {worst[1]:.4f}, i.e. {worst_z:.1f} SD from the reported mean; "
                    f"no set of five folds with SD {pap_s} contains that value",
                    "the reported SD and the exported per-fold values cannot both be right; regenerate from CSVs")

    # II.5 Table I rows vs the workbook that documents the ablation
    for _, row in PAPER_T1.iterrows():
        key = row["config"].split("(")[0].strip()
        match = t1_aud[t1_aud.config.astype(str).str.contains(key[:12], case=False, na=False, regex=False)]
        if match.empty:
            add(f"T1-{row['config']}", "1.3 CRITICAL", "CRITICAL",
                f"Table I row '{row['config']}'", "no matching row exists in the fold-0 workbook",
                "rerun this ablation with the current code and save per-case CSVs")
            continue
        a = match.iloc[0]
        diffs = {m: (round(float(row[m]), 4), round(float(a[m]), 4)) for m in METRICS
                 if abs(float(row[m]) - float(a[m])) > 0.005}
        if diffs:
            add(f"T1-{row['config']}", "1.3 CRITICAL", "CRITICAL",
                f"Table I '{row['config']}': " + ", ".join(f"{m}={v[0]}" for m, v in diffs.items()),
                "workbook: " + ", ".join(f"{m}={v[1]}" for m, v in diffs.items()),
                "one of the two is wrong; only a rerun with saved CSVs settles it")

    # II.6 the workbook contradicts the narrative's direction for HQGD
    if fold0_nograph is not None:
        g, fu = fold0_nograph, fold0["MW-HQNet"]
        d_hqgd = {m: round(float(fu[m]) - float(g[m]), 4) for m in METRICS}
        add("HQGD-DIRECTION", "1.3/VI", "CRITICAL",
            "V-A and VI: adding HQGD 'raises both Dice and sensitivity further, but also increases FP/ROI'",
            "workbook no-graph -> full: " + ", ".join(f"{m} {v:+.4f}" for m, v in d_hqgd.items())
            + (f". Workbook note for that row: '{wb_note}'" if wb_note else ""),
            "the row is annotated as a self-attention variant, so Table I row 3 is not a no-graph control at all; "
            "rerun as A3 (GRAPH='none') and A4 (GRAPH='selfattn') and rewrite V-A/VI from those results")

    # II.7 two evaluation protocols inside one notebook
    for model in ["Baseline", "MW-HQNet"]:
        for f in sorted(set(tta[tta.model == model].fold.astype(int))):
            a = primary[(primary.model == model) & (primary.fold == f)]
            b = tta[(tta.model == model) & (tta.fold == f)]
            if a.empty or b.empty:
                continue
            d = {m: round(float(b.iloc[0][m]) - float(a.iloc[0][m]), 4) for m in METRICS if pd.notna(b.iloc[0][m])}
            if any(abs(v) > 0.01 for v in d.values()):
                add(f"PROTOCOL-{model}-fold{f}", "1.4 CRITICAL", "CRITICAL",
                    f"Table II uses the no-TTA validation values for {model} fold {f}",
                    "the notebook's own final cell prints " + ", ".join(f"{m} {v:+.4f}" for m, v in d.items())
                    + " for the same checkpoint, because it evaluates with flip-TTA",
                    "declare one protocol for Table II and use it everywhere; report the other as a TTA ablation")

    contra = pd.DataFrame(C)
    contra.to_csv(DIRS["audit"] / "contradictions.csv", index=False)

    # ================================================================ report
    md = ["# Audit of the existing MW-HQNet artefacts\n",
          "Generated by `MW-HQNet_Reviewer_Fixes.ipynb` Cell 2. Every value is traceable to a named file; "
          "see `audit/provenance.csv`.\n",
          "## Sources\n",
          f"* fold-0 ablation: `{os.path.basename(CFG['LEGACY_XLSX'])}`, sheet `Fold0 Results`\n",
          f"* folds 1-4: per-epoch JSON records embedded in the saved outputs of the notebooks in `{CFG['LEGACY_DIR']}`. "
          f"The reported value is the record at the epoch with maximal `Dice masses`, i.e. the checkpoint the legacy "
          f"loop saved as `best.pt` - the same-fold selection the reviewer flags in 1.6\n",
          "* `cell10-summary(TTA)` rows are kept separately as a protocol check\n",
          "\n## Table II reconstructed from the artefacts\n",
          df_to_md(t2.drop(columns=[c for c in t2.columns if c.startswith("sd_")])) + "\n",
          "\n## Per-fold deltas (MW-HQNet minus Baseline)\n",
          df_to_md(delta_df.assign(fold=delta_df["fold"].astype(int)).round(4), index=False) + "\n",
          f"\nNegative deltas: **{n_negative} of 30**"
          + (": " + ", ".join(f"{m} at fold {f}" for m, f in neg) if neg else "") + "\n",
          "\n## Table III recomputed\n",
          df_to_md(t3, index=False) + "\n",
          "\n## Contradiction ledger\n"]
    for _, r in contra.iterrows():
        md += [f"### [{r['severity']}] {r['id']} - reviewer {r['reviewer_item']}\n",
               f"* **Manuscript:** {r['manuscript_claim']}\n",
               f"* **Evidence in your artefacts:** {r['audited_evidence']}\n",
               f"* **Required action:** {r['required_action']}\n"]
    (DIRS["audit"] / "AUDIT_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(PROVENANCE).to_csv(DIRS["audit"] / "provenance.csv", index=False)

    print("\n" + "=" * 92)
    print("RECONSTRUCTED TABLE II (artefact-derived, folds 0-4)")
    print("=" * 92)
    print(t2.drop(columns=[c for c in t2.columns if c.startswith("sd_")]).to_string())
    print("\nPER-FOLD DELTAS (MW-HQNet - Baseline)")
    print(delta_df.round(4).to_string(index=False))
    print(f"\nnegative deltas: {n_negative} / 30")
    print("\nRECOMPUTED TABLE III")
    print(t3.to_string(index=False))
    print(f"\ncontradictions logged: {len(contra)}")
    print(f"  -> {DIRS['audit'] / 'AUDIT_REPORT.md'}")
    print(f"  -> {DIRS['audit'] / 'contradictions.csv'}")

