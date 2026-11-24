# ============================================================================
# CELL 7 - ROI index: the legacy oracle cache, or image-only ROIs built from Cell 5
#
# TRACK="oracle" -> extract and index the legacy GT-cropped cache. It carries no inverse
#                   geometry, so those results stay in ROI space and are labelled oracle.
# TRACK="endo"   -> build image-only ROIs per fold, each with the geometry needed to paste a
#                   prediction back onto the native CT grid.
# ============================================================================
if (ENABLED["roi"] and CFG["TRACK"] == "endo") or CFG["TRACK"] == "oracle":
    import zipfile

    def legacy_roi_root():
        # reuse an extraction we already made, instead of unpacking 2.4 GB a second time
        for cand in [Path(r"E:\segmentation research paper\_unzipped\kits23_roi")]:
            if cand.exists() and len(list(cand.glob("*_meta.json"))) > 0:
                print(f"reusing existing ROI cache: {cand}")
                return cand

        d = DIRS["roi"] / "legacy"
        marker = d / ".extracted"
        if not marker.exists():
            print(f"extracting the legacy ROI cache (~2.4 GB) from "
                  f"{os.path.basename(CFG['LEGACY_ROI_ZIP'])} ...")
            with zipfile.ZipFile(CFG["LEGACY_ROI_ZIP"]) as z:
                z.extractall(d)
            marker.write_text("ok")
        roots = list(d.glob("**/kits23_roi"))
        return roots[0] if roots else d

    def _rows_from_dir(dirpath, space="roi"):
        rows = []
        for mp in sorted(Path(dirpath).glob("*_meta.json")):
            m = json.loads(mp.read_text())
            rows.append(dict(key=m["key"], case=m["case"], side=int(m["key"].rsplit("_s", 1)[1]),
                             img=str(Path(dirpath) / f"{m['key']}_img.npy"),
                             seg=str(Path(dirpath) / f"{m['key']}_seg.npy"),
                             geom=str(Path(dirpath) / f"{m['key']}_geom.json"),
                             spacing=tuple(m["spacing"]), space=space))
        return rows

    def oracle_index():
        return pd.DataFrame(_rows_from_dir(legacy_roi_root(), space="roi"))

    # ----------------------------------------------------------- endo ROI builder
    def _boxes_for(case, localizer):
        """Kidney boxes from the image-only localiser, expressed in native index space."""
        z = np.load(DIRS["cache_ct"] / f"{case}.npz")
        prob = predict_kidney_prob(localizer, z["ct"]).astype(np.float32)
        boxes = boxes_from_prob(prob, z["sp"])
        segz = np.load(DIRS["cache_seg"] / f"{case}.npz")
        sp_nat = np.asarray(segz["sp"], np.float32)
        native_shape = tuple(int(v) for v in segz["lab"].shape)
        ratio = np.asarray(z["sp"], np.float32) / sp_nat
        out = [((np.array(b["lo"]) * ratio).astype(int), (np.array(b["hi"]) * ratio).astype(int)) for b in boxes]
        if not out:                       # image-only fallback: the whole field of view
            out = [(np.zeros(3, int), np.array(native_shape) - 1)]
        return out, sp_nat, native_shape

    def _native_ct(case):
        """Full-resolution HU straight from the kept raw NIfTI (not the 2.5 mm cache)."""
        import nibabel as nib
        p = DIRS["raw"] / "dataset" / case / "imaging.nii.gz"
        img = nib.as_closest_canonical(nib.load(str(p)))
        return np.asanyarray(img.dataobj).astype(np.float32).transpose(2, 1, 0)

    def build_roi(case, localizer, out_dir):
        """Create image-only ROIs for one case plus the label used for training and scoring."""
        boxes, sp_nat, native_shape = _boxes_for(case, localizer)
        ct = _native_ct(case)
        gt = np.load(DIRS["cache_seg"] / f"{case}.npz")["lab"]
        assert ct.shape == gt.shape, f"{case}: CR {ct.shape} vs label {gt.shape}"
        for si, (lo, hi) in enumerate(boxes):
            center = (lo + hi) / 2.0
            ext_mm = (hi - lo + 1) * sp_nat
            roi_mm = max(192.0, float(max(ext_mm[0], ext_mm[1])) + 2 * XY_MARGIN_MM)
            sxy = roi_mm / ROI_PX
            nz = int(np.clip(math.ceil((ext_mm[2] + 2 * Z_MARGIN_MM) / Z_SPACING), 32, 512))
            out_shape = (nz, ROI_PX, ROI_PX)
            out_sp = (Z_SPACING, sxy, sxy)
            key = f"{case}_s{si}"
            im = crop_roi(ct, sp_nat, center, out_shape, out_sp, order=1, cval=-1024.0)
            im = np.clip(np.round(im), -1024, 3071).astype(np.int16)
            sg = crop_roi(gt.astype(np.float32), sp_nat, center, out_shape, out_sp, order=0, cval=0).astype(np.uint8)
            np.save(out_dir / f"{key}_img.npy", im)
            np.save(out_dir / f"{key}_seg.npy", sg)
            json.dump(dict(case=case, key=key, spacing=[Z_SPACING, float(sxy), float(sxy)]),
                      open(out_dir / f"{key}_meta.json", "w"))
            json.dump(dict(case=case, key=key, side=si,
                           center=[float(c) for c in center],
                           sp_native=[float(s) for s in sp_nat],
                           out_shape=[int(v) for v in out_shape],
                           out_sp=[float(s) for s in out_sp],
                           native_shape=[int(v) for v in native_shape]),
                      open(out_dir / f"{key}_geom.json", "w"))
        return True

    def ensure_roi_dir(tag, cases, localizer):
        out_dir = DIRS["roi"] / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [c for c in cases if not list(out_dir.glob(f"{c}_s*_meta.json"))]
        if todo:
            print(f"  building {len(todo)} image-only ROI sets in roi/{tag} ...")
        for i, c in enumerate(todo):
            try:
                build_roi(c, localizer, out_dir)
            except Exception as e:
                print(f"    {c}: FAILED {type(e).__name__}: {e}")
            if (i + 1) % 20 == 0 or i == len(todo) - 1:
                print(f"    {i+1}/{len(todo)}")
        return out_dir

    def endo_index(tag):
        return pd.DataFrame(_rows_from_dir(DIRS["roi"] / tag, space="roi_native"))

    # ------------------------------------------------------------------ build
    if CFG["TRACK"] == "oracle":
        ROI_INDEX = oracle_index()
        print(f"oracle track: {len(ROI_INDEX)} kidney ROIs from {ROI_INDEX['case'].nunique()} cases")
        print("  NOTE: these crops were placed using ground-truth masks. Results from this track are")
        print("        an ORACLE UPPER BOUND and must be labelled as such in the manuscript.")
    else:
        _ensure("nibabel")
        torch = ensure_torch()
        cached = sorted({p.stem for p in DIRS["cache_ct"].glob("*.npz")})
        fj = json.loads(zipfile.ZipFile(CFG["LEGACY_BASELINE_ZIP"]).read(CFG["FOLDS_JSON_ZIP_ENTRY"]))
        FOLD_CASES = {i: v for i, v in enumerate(fj["folds"])}

        def _load_loc(fold):
            st = torch.load(DIRS["localizer"] / f"fold{fold}" / "best.pt", map_location=DEV, weights_only=False)
            m = KidneyLocalizer(LOC["base"]).to(DEV)
            m.load_state_dict(st["state"]); m.eval()
            return m

        ensure_roi_dir("shared", cached, _load_loc(CFG["FOLDS"][0]))
        for fold in CFG["FOLDS"]:
            ensure_roi_dir(f"fold{fold}", [c for c in FOLD_CASES[fold] if c in cached], _load_loc(fold))
        ROI_INDEX = endo_index("shared")
        print(f"endo track: {len(ROI_INDEX)} image-only ROIs in the shared cache; "
              f"held-out ROIs are in roi/fold*/")
