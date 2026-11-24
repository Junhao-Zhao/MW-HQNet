# ============================================================================
# CELL 12 -nnU-Net v2 bridge (reviewer 1.5: "add nnU-Net and a strong 2.5-D baseline")
#
# Run this ONLY in the endo track: nnU-Net must see the original, uncropped CT volumes,
# otherwise the baseline inherits exactly the oracle-localisation problem we are removing.
#
# The cell exports a nnU-Net Dataset401 in the official raw format with the SAME patient folds,
# prints the exact training commands (nnU-Net wants its own process, so it is not launched here),
# and imports the resulting predictions back into this notebook's scorer so the numbers land in
# the same tables as everything else.
# ============================================================================
if ENABLED["nnunet"] and CFG["TRACK"] == "endo" and MODEL_NEEDED:
    import zipfile
    _ensure("nibabel")
    import nibabel as nib
    from scipy import ndimage as ndi

    NN = dict(dataset_id=401, name="KiTS23_MWHQ", nnunet_raw=r"E:\nnUNet_raw",
              nnunet_preprocessed=r"E:\nnUNet_preprocessed", nnunet_results=r"E:\nnUNet_results")

    def export_nnunet_dataset():
        """Write imagesTr/labelsTr at NATIVE spacing + dataset.json + the declared fold split."""
        raw = Path(NN["nnunet_raw"]) / f"Dataset{NN['dataset_id']}_{NN['name']}"
        (raw / "imagesTr").mkdir(parents=True, exist_ok=True)
        (raw / "labelsTr").mkdir(parents=True, exist_ok=True)
        cases = sorted({p.stem for p in DIRS["cache_seg"].glob("*.npz") if "_ds" not in p.stem})
        print(f"exporting {len(cases)} cases at native spacing to {raw}")
        for c in cases:
            src = DIRS["raw"] / "dataset" / c
            if not (src / "imaging.nii.gz").exists():
                print(f"  {c}: raw CT missing (KEEP_RAW was False?) - skipping")
                continue
            img = nib.as_closest_canonical(nib.load(str(src / "imaging.nii.gz")))
            seg = nib.as_closest_canonical(nib.load(str(src / "segmentation.nii.gz")))
            shutil.copy(str(src / "imaging.nii.gz"), raw / "imagesTr" / f"{c}_0000.nii.gz")
            nib.save(nib.Nifti1Image(np.asanyarray(seg.dataobj).astype(np.uint8), img.affine),
                     str(raw / "labelsTr" / f"{c}.nii.gz"))
        json.dump({"channel_names": {"0": "CT"},
                   "labels": {"background": 0, "kidney": 1, "tumor": 2, "cyst": 3},
                   "numTraining": len(cases), "file_ending": ".nii.gz"},
                  open(raw / "dataset.json", "w"), indent=1)
        # the SAME folds as Cell 6, in nnU-Net's plain-text format
        split_dir = raw.parent / f"Dataset{NN['dataset_id']}_{NN['name']}_splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for fold, s in SPLITS.items():
            (split_dir / f"split_fold{fold}_tr.txt").write_text("\n".join(s["fit"] + s["dev"]))
            (split_dir / f"split_fold{fold}_val.txt").write_text("\n".join(s["outer"]))
        for fold in [0, 1, 2, 3, 4]:
            if not (split_dir / f"split_fold{fold}_tr.txt").exists():
                fold_cases = [c for c in cases if c not in set(LEGACY_FOLDS[fold])]
                (split_dir / f"split_fold{fold}_tr.txt").write_text("\n".join(fold_cases))
                (split_dir / f"split_fold{fold}_val.txt").write_text("\n".join(
                    [c for c in LEGACY_FOLDS[fold] if c in cases]))
        print(f"declared splits written to {split_dir}")
        return raw, split_dir

    def nnunet_commands():
        p = f"Dataset{NN['dataset_id']}_{NN['name']}"
        print("\n# ---- run these in a normal terminal, once, in this order -----------------")
        print(f'$env:nnUNet_raw="{NN["nnunet_raw"]}"; '
              f'$env:nnUNet_preprocessed="{NN["nnunet_preprocessed"]}"; '
              f'$env:nnUNet_results="{NN["nnunet_results"]}"')
        print(f"nnUNetv2_plan_and_preprocess -d {NN['dataset_id']} --verify_dataset_integrity")
        print(f"# 2d is the affordable choice on 8 GB; 3d_fullres needs ~10 GB and ~1-2 days/fold")
        for fold in CFG["FOLDS"]:
            print(f"nnUNetv2_train {NN['dataset_id']} 2d {fold} "
                  f"-tr nnUNetTrainer -p nnUNetPlans  "
                  f"# with splits: -tr nnUNetTrainer --npz ... see documentation")
        print(f"nnUNetv2_predict -i <imagesTs> -o {DIRS['pred']}/nnunet_2d -d {NN['dataset_id']} -c 2d -f 0 1 2 3 4")
        print("# ------------------------------------------------------------------------\n")

    def import_nnunet_predictions(pred_dir=None, tag="nnunet_2d"):
        """Score nnU-Net predictions with the SAME metric code, so they share a table."""
        pred_dir = Path(pred_dir) if pred_dir else DIRS["pred"] / tag
        case_rows, les_rows = [], []
        for p in sorted(pred_dir.glob("*.nii.gz")):
            case = p.name.replace(".nii.gz", "")
            gt = np.load(DIRS["cache_seg"] / f"{case}.npz")
            pred = np.asanyarray(nib.load(str(p)).dataobj).transpose(2, 1, 0).astype(np.uint8)
            if pred.shape != gt["lab"].shape:
                print(f"  {case}: shape {pred.shape} != {gt['lab'].shape}, skipped")
                continue
            c, l = score_roi(pred, gt["lab"], tuple(gt["sp"]), make_cfg(), case)
            c.update(space="native", experiment=tag, fold=-1)
            case_rows.append(c)
            les_rows.extend(dict(x, space="native") for x in l)
        out = DIRS["results"] / f"{tag}_fold-1"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(case_rows).to_csv(out / "cases.csv", index=False)
        pd.DataFrame(les_rows).to_csv(out / "lesions.csv", index=False)
        summ = summarise(case_rows, les_rows, make_cfg())
        json.dump(dict(summary=summ, track=CFG["TRACK"], tta=False, experiment=tag, fold=-1),
                  open(out / "summary.json", "w"), indent=1)
        (out / "DONE").write_text("ok")
        print(f"nnU-Net scored with the shared metric code: " +
              "  ".join(f"{k}={v:.4f}" for k, v in summ.items() if isinstance(v, float)))
        return out

    print("nnU-Net bridge ready. Call:")
    print("  export_nnunet_dataset()   # writes Dataset401 in the official raw format")
    print("  nnunet_commands()         # prints the exact commands to run in a terminal")
    print("  import_nnunet_predictions()  # scores them with this notebook's metrics")
    print("\nCost warning: 2d full training is ~6-12 h per fold on an RTX 4060 for 489 cases.")
    print("If that is out of budget, train one fold and report it as a single-fold reference,")
    print("stating that explicitly - do not present it as a five-fold comparison.")
