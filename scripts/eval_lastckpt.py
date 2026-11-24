"""Re-score every run with its FINAL-EPOCH checkpoint (last.pt) instead of best.pt.

Why
---
`train_one()` keeps the checkpoint with the best masses-class Dice on VAL_KEYS, and VAL_KEYS is
the very fold later reported. That is model selection on the evaluation fold (reviewer item 1.6).
The cheapest complete fix is to report the *final-epoch* checkpoint instead: the number of epochs
is fixed in advance, so the choice involves no held-out information at all.

We measured that this changes almost nothing: in 8 of 10 five-fold runs the best checkpoint *is*
the final one, and the mean selection gain is 0.0009 (max 0.0092). So this re-scoring removes the
protocol defect while leaving the numbers essentially intact.

Weights are already on disk, so this is inference only -- no retraining.

Output: runs/<run>/lastep/{cases_dose1.0.csv, lesions_dose1.0.csv, summary.csv}
The original best.pt results are left untouched for comparison.
"""
import json
import os
import time

import pandas as pd

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
P = r"E:\jupyter_work\mwhq_pipeline\runs"

# (run folder, config-cell marker, fold, extra overrides)
RUNS = [
    ("baseline_unet_fold0",              "3.4 config", 0, {}),
    ("baseline_unet_fold1_fold1",        "3.4 config", 1, {}),
    ("baseline_unet_fold2_fold2",        "3.4 config", 2, {}),
    ("baseline_unet_fold3_fold3",        "3.4 config", 3, {}),
    ("baseline_unet_fold4_fold4",        "3.4 config", 4, {}),
    ("mwhqnet_fixed_fold0",              "3.7 config", 0, {}),
    ("mwhqnet_fixed_fold1_fold1",        "3.7 config", 1, {}),
    ("mwhqnet_fixed_fold2_fold2",        "3.7 config", 2, {}),
    ("mwhqnet_fixed_fold3_fold3",        "3.7 config", 3, {}),
    ("mwhqnet_fixed_fold4_fold4",        "3.7 config", 4, {}),
    ("abl_pawe_only_fold0",              "3.8 config", 0, {}),
    ("abl_pawe_lcqs_noGraph_fold0",      "3.9 config", 0, {}),
    ("mwhqnet_full_fold0",               "3.1 config", 0, {}),
    ("a3_nograph_fold0",                 "3.7 config", 0, {"GRAPH": "none"}),
]

ONLY = os.environ.get("LASTCKPT_ONLY", "").strip()

nb = json.loads(open(NB, encoding="utf-8").read())
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def find(marker):
    for i, s in cells.items():
        if s[:220].find(marker) >= 0:
            return i, s
    raise KeyError(marker)


g = {"__name__": "__main__", "display": lambda *a, **k: None}
for m in ["1.1 environment", "2.1 installs", "2.2 preprocessing", "2.3 dataset",
          "2.3b prepare_folds", "2.4 model", "2.5 losses", "2.6 inference"]:
    i, s = find(m)
    exec(compile(s, f"cell{i}", "exec"), g)
print("shared cells loaded", flush=True)

MWHQNet, evaluate, summarise = g["MWHQNet"], g["evaluate"], g["summarise"]
dev = g["dev"]

t_all = time.time()
for run, marker, fold, over in RUNS:
    if ONLY and ONLY not in run:
        continue
    out_dir = os.path.join(P, run, "lastep")
    if os.path.exists(os.path.join(out_dir, "summary.csv")):
        print(f"[skip] {run} (already done)", flush=True)
        continue
    ck = os.path.join(P, run, "last.pt")
    if not os.path.exists(ck):
        print(f"[miss] {run}: no last.pt", flush=True)
        continue

    # fresh config for this run
    i, s = find(marker)
    exec(compile(s, f"cell{i}", "exec"), g)
    CFG = g["CFG"]
    CFG.update(over)          # NOTE: must be positional -- update(over=over) would store a key named "over"
    CFG.update(FOLD=fold, RESUME=False, SMOKE_TEST=False, NUM_WORKERS=0)
    g["prepare_folds"]()
    VAL_KEYS = g["VAL_KEYS"]
    RUN_DIR = os.path.join(P, run)
    g["RUN_DIR"] = RUN_DIR

    t0 = time.time()
    model = MWHQNet(CFG).to(dev)
    _ck = __import__("torch").load(ck, map_location=dev, weights_only=False)
    # last.pt is a checkpoint dict (model/opt/sched/scaler/epoch/best), not a bare state_dict
    _sd = _ck["model"] if isinstance(_ck, dict) and "model" in _ck else _ck
    model.load_state_dict(_sd)
    model.eval()
    dfc, dfl = evaluate(model, VAL_KEYS, CFG, dose=1.0, tta=True, max_vols=None)
    os.makedirs(out_dir, exist_ok=True)
    dfc.to_csv(os.path.join(out_dir, "cases_dose1.0.csv"), index=False)
    dfl.to_csv(os.path.join(out_dir, "lesions_dose1.0.csv"), index=False)
    summ = pd.DataFrame({"dose=1.0": pd.Series(summarise(dfc, dfl, CFG))})
    summ.to_csv(os.path.join(out_dir, "summary.csv"))
    print(f"[done] {run:32s} fold={fold} {(time.time()-t0)/60:5.1f} min  "
          f"Dice masses={summ.loc['Dice masses'].iloc[0]:.4f}", flush=True)

print(f"\nALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)
