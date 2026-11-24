# ============================================================================
# CELL 2 - Forensic audit of the EXISTING artefacts  (CPU only, < 1 min)
#
# Why this comes first
# --------------------
# The reviewer's Sec.1.3/Sec.1.4 complaints are claims that your tables contradict your text.
# Before spending GPU days, we can settle most of that from the artefacts you already have:
#   * the fold-0 ablation workbook  (MW-HQNet_fold0_results.xlsx)
#   * the per-fold epoch logs embedded in the saved notebook outputs
#   * the fold-4 baseline per-case CSVs inside baseline-fold4-ckpt.zip
# This cell extracts a tidy per-fold, per-metric table from those sources and records
# exactly where each number came from, so every value is traceable to a file.
# ============================================================================
if ENABLED["audit"]:
    import zipfile, re, io

    PROVENANCE = []   # one row per extracted number: where did it come from?

    def _nb_outputs(path):
        """Yield every text output of every code cell of a saved .ipynb."""
        nb = json.load(open(path, encoding="utf-8"))
        for ci, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            for o in cell.get("outputs", []):
                if o.get("output_type") == "stream":
                    yield ci, "".join(o.get("text", []))
                elif o.get("output_type") in ("execute_result", "display_data"):
                    yield ci, "".join(o.get("data", {}).get("text/plain", []))

    def _epoch_records(path):
        """Every per-epoch JSON record printed by the training loop, with its cell index."""
        recs = []
        for ci, txt in _nb_outputs(path):
            for line in txt.splitlines():
                line = line.strip()
                if line.startswith("{") and '"epoch"' in line and '"Dice masses"' in line:
                    try:
                        recs.append((ci, json.loads(line)))
                    except Exception:
                        pass
        return recs

    def _pick_metric(rec, prefix, ci, nbname):
        """First key of rec starting with prefix (metric keys carry the bin/count suffix)."""
        for k, v in rec.items():
            if k.startswith(prefix):
                PROVENANCE.append(dict(notebook=nbname, cell=ci, metric=prefix,
                                       source_key=k, epoch=rec.get("epoch"), value=float(v)))
                return float(v)
        return float("nan")

    # Metric keys as they appear in the epoch logs, shared by both model configurations.
    METRIC_KEYS = {
        "Dice masses": "Dice masses",
        "Dice tumor": "Dice tumor",
        "Dice cyst": "Dice cyst",
        "Sens<10mm": "Sens [0.0, 10.0)",
        "Sens all": "Sens all lesions",
        "FP/ROI": "FP lesions / kidney ROI",
        "Dice kidney+masses": "Dice kidney+masses",
        "NSD masses": "NSD@2mm masses",
        "HD95 masses": "HD95 masses",
    }

    def _harvest(nb_path, fold, model):
        """Best-epoch record = the checkpoint the legacy code actually saved as best.pt.

        The legacy loop saved best.pt whenever 'Dice masses' improved on the SAME validation
        fold it later reported (reviewer Sec.1.6). We therefore recover the *selected* checkpoint,
        i.e. exactly the model whose results went into the paper - leak included.
        """
        recs = _epoch_records(nb_path)
        if not recs:
            return None
        best_ci, best = max(recs, key=lambda t: t[1].get("Dice masses", -1))
        row = dict(fold=fold, model=model, nb=os.path.basename(nb_path),
                   selected_epoch=best.get("epoch"), source="epoch-log(best Dice masses)")
        for name, key in METRIC_KEYS.items():
            row[name] = _pick_metric(best, key, best_ci, os.path.basename(nb_path))
        return row

    def _harvest_summary_table(nb_path, fold, model):
        """Final Cell-10 summary (with TTA), if that cell was executed in the saved notebook.

        That table is printed as `dose=1.0  dose=0.5  dose=0.25` followed by one row per metric,
        so parsing must start *after* the header line and match metric names by prefix because
        the sensitivity rows carry a `(n=...)` suffix.
        """
        for ci, txt in _nb_outputs(nb_path):
            lines = [l for l in txt.splitlines() if l.strip()]
            hdr = next((i for i, l in enumerate(lines) if l.strip().startswith("dose=")), None)
            if hdr is None:
                continue
            picked = {}
            for line in lines[hdr + 1:]:
                m = re.match(r"^(\S.*?)\s{2,}(-?[\d.]+)\s", line)
                if not m:
                    continue
                picked[m.group(1).strip()] = float(m.group(2))
            if not picked:
                continue
            row = dict(fold=fold, model=model, nb=os.path.basename(nb_path),
                       selected_epoch="final", source="cell10-summary(TTA)")
            for name, key in METRIC_KEYS.items():
                val = float("nan")
                for pk, pv in picked.items():
                    if pk.startswith(key):
                        val = pv
                        break
                row[name] = val
            if not np.isnan(row["Dice masses"]):
                return row
        return None

    # -- locate legacy notebooks -------------------------------------------------
    # Windows path ordering is case-insensitive, so the empty template `Step5_Baseline_Fold1.ipynb`
    # can shadow the executed `step5-baseline-fold1.ipynb`. Rank the candidates by how many epoch
    # records they actually contain and keep the executed one.
    legacy = Path(CFG["LEGACY_DIR"])
    cand_mw, cand_bl = {}, {}
    for p in sorted(legacy.glob("*.ipynb")):
        n = p.name.lower()
        m = re.search(r"fold(\d)", n)
        if not m:
            continue
        f = int(m.group(1))
        if n.startswith(("step5-mwhqnet", "step5_mwhqnet")):
            cand_mw.setdefault(f, []).append(p)
        elif n.startswith(("step5-baseline", "step5_baseline")):
            cand_bl.setdefault(f, []).append(p)

    def _choose(cands):
        out = {}
        for f, paths in cands.items():
            ranked = sorted(paths, key=lambda q: -len(_epoch_records(q)))
            if ranked and len(_epoch_records(ranked[0])):
                out[f] = ranked[0]
        return out

    nb_mw, nb_bl = _choose(cand_mw), _choose(cand_bl)
    for tag, c in [("MW-HQNet", cand_mw), ("Baseline", cand_bl)]:
        for f in sorted(c):
            names = [q.name for q in c[f]]
            kept = (nb_mw if tag == "MW-HQNet" else nb_bl).get(f)
            print(f"  {tag:9s} fold {f}: using {kept.name if kept else '(none with outputs)'}"
                  + (f"   [also present: {', '.join(n for n in names if n != (kept.name if kept else ''))}]"
                     if len(names) > 1 else ""))

    rows = []
    for f, p in sorted(nb_mw.items()):
        r = _harvest(p, f, "MW-HQNet")
        if r: rows.append(r)
        r2 = _harvest_summary_table(p, f, "MW-HQNet")
        if r2: rows.append(r2)
    for f, p in sorted(nb_bl.items()):
        r = _harvest(p, f, "Baseline")
        if r: rows.append(r)
        r2 = _harvest_summary_table(p, f, "Baseline")
        if r2: rows.append(r2)

    aud_nb = pd.DataFrame(rows)
    print(f"harvested {len(aud_nb)} rows from {len(nb_mw)} MW-HQNet + {len(nb_bl)} baseline notebooks")
    print(aud_nb.groupby(["model", "source"]).size().to_string())
    display_cols = ["fold", "model", "selected_epoch", "source", "Dice masses", "Dice tumor",
                    "Dice cyst", "Sens<10mm", "Sens all", "FP/ROI"]
    print(aud_nb[display_cols].round(4).to_string(index=False))
    aud_nb.to_csv(DIRS["audit"] / "harvested_notebook_records.csv", index=False)
