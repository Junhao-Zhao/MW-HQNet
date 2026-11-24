# ============================================================================
# CELL 6 -Folds and the NESTED split that fixes reviewer 1.6
#
# The legacy loop chose best.pt on the same validation fold it then reported. Here each
# outer fold is split three ways and the roles are enforced, not documented:
#
#   fit        gradient updates, augmentation, label-based sampling
#   inner_dev  checkpoint / threshold / fusion selection ONLY
#   outer_test evaluated exactly once, after the checkpoint is frozen
#
# The outer folds themselves are the ones your old runs used - read from
# runs/folds_kits23_5_seed42.json inside the baseline checkpoint archive - so the new numbers
# are comparable with the old ones fold by fold.
# ============================================================================
if ENABLED["splits"]:
    import zipfile

    fj = json.loads(zipfile.ZipFile(CFG["LEGACY_BASELINE_ZIP"]).read(CFG["FOLDS_JSON_ZIP_ENTRY"]))
    ALL_CASES = sorted(fj["cases"])
    LEGACY_FOLDS = {i: sorted(v) for i, v in enumerate(fj["folds"])}

    # --- the split file must be internally valid before anything is trained -----------
    sets = [set(v) for v in LEGACY_FOLDS.values()]
    assert len(ALL_CASES) == len(set(ALL_CASES)), "duplicate patient ids in the manifest"
    assert set().union(*sets) == set(ALL_CASES), "folds do not cover every case"
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), f"folds {i} and {j} share patients"
    print(f"legacy folds verified: {len(ALL_CASES)} patients, sizes {[len(s) for s in sets]}, "
          f"pairwise overlaps 0")

    SPLITS = {}
    for fold in CFG["FOLDS"]:
        outer_test = list(LEGACY_FOLDS[fold])
        outer_train = [c for c in ALL_CASES if c not in set(outer_test)]
        rng = np.random.default_rng(4200 + fold)
        perm = rng.permutation(len(outer_train))
        n_dev = max(1, int(round(len(outer_train) * CFG["DEV_FRACTION"])))
        dev = [outer_train[i] for i in perm[:n_dev]]
        fit = [outer_train[i] for i in perm[n_dev:]]

        # --- role separation, asserted rather than promised --------------------------
        assert not (set(fit) & set(dev))
        assert not (set(fit) & set(outer_test))
        assert not (set(dev) & set(outer_test))
        assert len(fit) + len(dev) + len(outer_test) == len(ALL_CASES)
        SPLITS[fold] = dict(fit=sorted(fit), dev=sorted(dev), outer=sorted(outer_test))
        json.dump(SPLITS[fold], open(DIRS["splits"] / f"fold{fold}.json", "w"), indent=1)

    print(f"\n{'fold':>4} {'fit':>5} {'dev':>4} {'outer_test':>11}   inner-dev fraction")
    for f, s in SPLITS.items():
        print(f"{f:>4} {len(s['fit']):>5} {len(s['dev']):>4} {len(s['outer']):>11}   "
              f"{len(s['dev']) / (len(s['dev']) + len(s['fit'])):.3f}")

    # ---------------------------------------------------- which ROIs feed which role
    def rois_for(fold, role):
        """ROI rows for a role, using held-out ROIs for the outer fold in the endo track."""
        cases = set(SPLITS[fold][role])
        if CFG["TRACK"] == "oracle":
            df = ROI_INDEX[ROI_INDEX["case"].isin(cases)]
            return df.reset_index(drop=True)
        tag = f"fold{fold}" if role == "outer" else "shared"
        df = endo_index(tag)
        df = df[df["case"].isin(cases)]
        missing = cases - set(df["case"])
        if missing:
            print(f"  fold {fold}/{role}: {len(missing)} cases have no ROI yet (run Cell 4-5)")
        return df.reset_index(drop=True)

    FOLD_ROIS = {f: {r: rois_for(f, r) for r in ("fit", "dev", "outer")} for f in CFG["FOLDS"]}
    print("\nROIs available per fold (volumes = kidney sides):")
    for f in CFG["FOLDS"]:
        print(f"  fold {f}: fit {len(FOLD_ROIS[f]['fit']):>4}  dev {len(FOLD_ROIS[f]['dev']):>4}  "
              f"outer {len(FOLD_ROIS[f]['outer']):>4}")
    if CFG["TRACK"] == "endo":
        miss = [f for f in CFG["FOLDS"] if len(FOLD_ROIS[f]["outer"]) == 0]
        assert not miss, f"no held-out ROIs for folds {miss} - run Cell 4 and Cell 5 first"
