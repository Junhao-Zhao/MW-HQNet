# ============================================================================
# CELL 3 - Raw KiTS23 acquisition and the image-only cache   [TRACK == "endo"]
#
# Reviewer Sec.1.1: your ROIs were cropped from ground-truth masks. Fixing that needs the
# original CT volumes back, because the crop box must be found from the image alone.
#
# huggingface.co is unreachable from mainland China; hf-mirror.com is reachable and serves
# the same repo (Cell 1 already set HF_ENDPOINT). The imaging files are ~42 GB, the
# segmentations only ~0.14 GB. Strategy: stream each case, keep the segmentation at NATIVE
# grid (tiny), keep the CT only after a 2.5 mm downsample (the localizer does not need more),
# and delete the raw file unless KEEP_RAW is set. Final cache: ~7 GB instead of 42 GB.
# ============================================================================
if ENABLED["data"] and CFG["TRACK"] == "endo":
    _ensure("nibabel")
    import nibabel as nib
    from scipy import ndimage as ndi

    HF_REPO = "MedOtter/kits23"          # per-case NIfTI mirror of the public KiTS23 training set
    DS_MM = 2.5                          # isotropic spacing of the cached CT (localizer input)
    KEEP_RAW = False                     # True keeps the 42 GB of raw NIfTI for native-space re-scoring

    def _hf_api_list(repo, repo_type="dataset"):
        """List repo files through the mirror without needing the hf CLI."""
        import urllib.request
        url = f"{os.environ['HF_ENDPOINT']}/api/{repo_type}s/{repo}?blobs=true"
        with urllib.request.urlopen(url, timeout=60) as r:
            meta = json.loads(r.read().decode())
        return meta.get("siblings", [])

    def _download(case, raw_dir):
        from huggingface_hub import hf_hub_download
        out = {}
        for part in ("imaging", "segmentation"):
            out[part] = hf_hub_download(HF_REPO, f"dataset/{case}/{part}.nii.gz", repo_type="dataset",
                                        local_dir=str(raw_dir), token=os.environ.get("HF_TOKEN"))
        return out["imaging"], out["segmentation"]

    def build_case(case, raw_dir):
        """Cache one case: native-grid labels, plus a downsample of the CT for localisation.

        Nothing here reads the label to decide geometry - the downsample grid is fixed by the
        CT header alone, so this function is safe to run before any label is available.
        """
        img_p, seg_p = _download(case, raw_dir)
        img = nib.as_closest_canonical(nib.load(img_p))
        seg = nib.as_closest_canonical(nib.load(seg_p))
        if img.shape != seg.shape:
            raise ValueError(f"{case}: image {img.shape} != label {seg.shape}")

        zooms_xyz = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)  # (sx, sy, sz)
        hu_xyz = np.asanyarray(img.dataobj).astype(np.float32)
        lab_xyz = np.asanyarray(seg.dataobj).astype(np.uint8)

        # internal convention is (Z, Y, X) everywhere, matching the legacy implementation
        hu = hu_xyz.transpose(2, 1, 0)
        lab = lab_xyz.transpose(2, 1, 0)
        sp = zooms_xyz[::-1].copy()                                          # (sz, sy, sx)
        affine = img.affine.astype(np.float64)
        del hu_xyz, lab_xyz

        # fixed, label-free downsample grid
        zoom = (DS_MM / sp[0], DS_MM / sp[1], DS_MM / sp[2])
        ct_ds = ndi.zoom(hu, zoom, order=1, grid_mode=False).astype(np.int16)
        seg_ds = ndi.zoom(lab.astype(np.float32), zoom, order=0, grid_mode=False).astype(np.uint8)
        sp_ds = np.asarray([sp[0] / zoom[0], sp[1] / zoom[1], sp[2] / zoom[2]], dtype=np.float32)

        np.savez_compressed(DIRS["cache_seg"] / f"{case}.npz", lab=lab, sp=sp, affine=affine)
        np.savez_compressed(DIRS["cache_seg"] / f"{case}_ds.npz", lab=seg_ds, sp=sp_ds)
        np.savez_compressed(DIRS["cache_ct"] / f"{case}.npz", ct=ct_ds, sp=sp_ds,
                            native_shape=np.asarray(lab.shape, np.int32))
        if not KEEP_RAW:
            for p in (img_p, seg_p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        vox = float(np.prod(sp))
        return dict(case=case, native_shape=list(lab.shape), native_spacing=sp.round(4).tolist(),
                    ds_shape=list(ct_ds.shape), ds_spacing=sp_ds.round(4).tolist(),
                    kidney_ml=round(float((lab == 1).sum() * vox) / 1000.0, 1),
                    tumor_ml=round(float((lab == 2).sum() * vox) / 1000.0, 1),
                    cyst_ml=round(float((lab == 3).sum() * vox) / 1000.0, 1))

    # ---------------------------------------------------------------- download
    siblings = _hf_api_list(HF_REPO)
    cases = sorted({s["rfilename"].split("/")[1] for s in siblings
                    if s["rfilename"].startswith("dataset/case_") and s["rfilename"].endswith("imaging.nii.gz")})
    if CFG["MAX_CASES_DOWNLOAD"]:
        cases = cases[: CFG["MAX_CASES_DOWNLOAD"]]
    print(f"{len(cases)} cases listed on the mirror; caching at {DS_MM} mm for the localizer")

    manifest_p = DIRS["audit"] / "image_cache_manifest.csv"
    done = set()
    if manifest_p.exists():
        done = set(pd.read_csv(manifest_p)["case"])
    todo = [c for c in cases if c not in done]
    print(f"{len(done)} already cached, {len(todo)} to go")

    def _flush(new):
        if not new:
            return
        old = pd.read_csv(manifest_p) if manifest_p.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(new)], ignore_index=True) \
          .drop_duplicates("case").to_csv(manifest_p, index=False)

    t0 = time.time()
    recs = []
    for i, c in enumerate(todo):
        try:
            recs.append(build_case(c, DIRS["raw"]))
        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] {c}: FAILED {type(e).__name__}: {e}")
            continue
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = (time.time() - t0) / 60
            print(f"  [{i+1}/{len(todo)}] {c}  {el:.1f} min elapsed, "
                  f"~{el / (i + 1) * (len(todo) - i - 1):.0f} min left")
        if (i + 1) % 25 == 0:
            _flush(recs); recs = []
    _flush(recs)

    man = pd.read_csv(manifest_p).drop_duplicates("case")
    print(f"\ncached {len(man)} cases -> {DIRS['cache_ct']}")
    print(man[["case", "native_shape", "ds_shape", "kidney_ml", "tumor_ml", "cyst_ml"]].head().to_string(index=False))
    print(f"median kidney volume {man['kidney_ml'].median():.0f} mL, "
          f"cases without tumour {int((man['tumor_ml'] == 0).sum())}, "
          f"without cyst {int((man['cyst_ml'] == 0).sum())}")
