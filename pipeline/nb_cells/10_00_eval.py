# ============================================================================
# CELL 10 - Frozen evaluation: outer fold, once, with optional native-space assembly
#
# For each finished run this cell:
#   1. loads best.pt (chosen on inner-dev in Cell 9),
#   2. predicts the outer-fold ROIs with flip-TTA (declared, and the SAME protocol everywhere),
#   3. writes per-case and per-lesion CSVs - the only permitted source for Tables II-V,
#   4. in the endo track, pastes each ROI prediction back onto the native grid, asserts the
#      geometry matches the source CT exactly, and writes a NIfTI for official KiTS scoring.
# ============================================================================
if ENABLED["infer"] and MODEL_NEEDED:
    _ensure("nibabel")
    import nibabel as nib

    TTA_FOR_FINAL = True      # declare once; every reported number uses this setting

    def _geom_for(row):
        g = json.loads(Path(row["geom"]).read_text())
        return g

    def assemble_native(roi_preds, geoms, native_shape):
        """Paste ROI-space label volumes onto the native grid and resolve overlaps by priority."""
        out = np.zeros(native_shape, np.uint8)
        for lab, g in zip(roi_preds, geoms):
            out = np.maximum(out, paste_roi(lab, g, native_shape))
        return out

    def eval_run(exp_id, fold, tta=None, save_nifti=True):
        """Score the outer fold of one finished run. Nothing here influences model selection."""
        tta = TTA_FOR_FINAL if tta is None else tta
        run_dir = DIRS["runs"] / f"{exp_id}_fold{fold}"
        if not (run_dir / "DONE").exists():
            print(f"  {exp_id} fold{fold}: not trained yet")
            return None
        _, ov = EXPERIMENTS[exp_id]
        cfg = make_cfg(**ov); cfg["AMP"] = CFG["AMP"]
        out_dir = DIRS["results"] / f"{exp_id}_fold{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "DONE").exists():
            return out_dir

        model = MWHQNet(cfg).to(DEV)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=DEV, weights_only=False))
        model.eval()

        rows = FOLD_ROIS[fold]["outer"]
        case_rows, les_rows = [], []
        if CFG["TRACK"] == "endo":
            by_case = {}
            for _, r in rows.iterrows():
                by_case.setdefault(r["case"], []).append(r)
            for case, sub in by_case.items():
                labs, geoms = [], []
                for _, r in sub.iterrows():
                    img = np.load(r["img"]); gt = np.load(r["seg"])
                    probs = predict_volume(model, img, cfg, tta=tta)
                    lab = postprocess(probs, r["spacing"], cfg)
                    c, l = score_roi(lab, gt, r["spacing"], cfg, r["key"])
                    c.update(space="roi", fold=fold, experiment=exp_id)
                    case_rows.append(c); les_rows.extend(l)
                    labs.append(lab); geoms.append(_geom_for(r))
                # ---- native reconstruction, geometry asserted ----------------------
                g0 = geoms[0]
                native_shape = tuple(g0["native_shape"])
                nat = assemble_native(labs, geoms, native_shape)
                assert nat.shape == native_shape, "assembled volume is not native-shaped"
                if save_nifti:
                    nat_gt = np.load(DIRS["cache_seg"] / f"{case}.npz")
                    sp_nat = np.asarray(nat_gt["sp"], np.float64)
                    affine = np.diag(list(sp_nat) + [1.0])       # affine of the canonical native grid
                    nii_dir = DIRS["pred"] / f"{exp_id}_fold{fold}"
                    nii_dir.mkdir(parents=True, exist_ok=True)
                    nib.save(nib.Nifti1Image(nat.astype(np.uint8), affine), str(nii_dir / f"{case}.nii.gz"))
                    # native-space scores, computed on the reconstructed volume itself
                    c_nat, l_nat = score_roi(nat, nat_gt["lab"], tuple(sp_nat), cfg, case)
                    c_nat.update(space="native", fold=fold, experiment=exp_id)
                    case_rows.append(c_nat)
                    les_rows.extend(dict(x, space="native") for x in l_nat)
        else:
            for _, r in rows.iterrows():
                img = np.load(r["img"]); gt = np.load(r["seg"])
                probs = predict_volume(model, img, cfg, tta=tta)
                lab = postprocess(probs, r["spacing"], cfg)
                c, l = score_roi(lab, gt, r["spacing"], cfg, r["key"])
                c.update(space="roi_oracle", fold=fold, experiment=exp_id)
                case_rows.append(c); les_rows.extend(l)

        pd.DataFrame(case_rows).to_csv(out_dir / "cases.csv", index=False)
        pd.DataFrame(les_rows).to_csv(out_dir / "lesions.csv", index=False)
        summ = summarise(case_rows, les_rows, cfg)
        json.dump(dict(summary=summ, track=CFG["TRACK"], tta=tta, experiment=exp_id, fold=fold),
                  open(out_dir / "summary.json", "w"), indent=1)
        (out_dir / "DONE").write_text("ok")
        print(f"  {exp_id} fold{fold} [{CFG['TRACK']}]: " +
              "  ".join(f"{k}={v:.4f}" for k, v in summ.items() if isinstance(v, float)))
        return out_dir

    def collect_all():
        """Join every finished run into one long table: experiment x fold x metric."""
        recs = []
        for p in sorted(DIRS["results"].glob("*/summary.json")):
            d = json.loads(p.read_text())
            recs.append(dict(experiment=d["experiment"], fold=d["fold"], track=d["track"], tta=d["tta"],
                             **{f"M[{k}]": v for k, v in d["summary"].items()}))
        return pd.DataFrame(recs)

    print("evaluation driver ready: eval_run(experiment_id, fold) -> results/<exp>_fold<k>/")
    print(f"  final-evaluation protocol: flip-TTA = {TTA_FOR_FINAL} (identical for every model and fold)")
