"""Capacity-matched comparator for reviewer item 1.5.

The reviewer's objection: "The proposed network is also larger than the baseline
(approximately 25.1 M versus 22.6 M parameters), so the reported gains may reflect capacity
rather than design."

Response: train the full MW-HQNet design with the decoder/query width reduced so the total
parameter count matches the 2.5-D baseline (22.59 M). At D_MODEL = 64 the model has 22.78 M
parameters -- 0.19 M (0.8%) above the baseline, i.e. matched for practical purposes, versus
25.12 M (+11%) for the full model.

If the design (and not the parameter budget) is what produces the sensitivity gain, the
capacity-matched model should still beat the baseline by a similar margin on the five folds.

Everything else -- data, splits, epochs, optimiser, augmentation, metric code -- is identical
to `mwhqnet_fixed_fold*`. Checkpoints are scored with the final-epoch (last.pt) protocol, which
is the protocol used for the re-scored Table II.

Usage:
    python train_matched.py            # all five folds
    MATCH_ONLY=0 python train_matched.py   # single fold (smoke test)
"""
import json
import os
import time

import pandas as pd

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
OUT = r"E:\jupyter_work\mwhq_pipeline\runs"

D_MODEL = int(os.environ.get("MATCH_DMODEL", "64"))
ONLY = os.environ.get("MATCH_ONLY", "").strip()
RESUME = os.environ.get("MATCH_RESUME", "0") == "1"   # set MATCH_RESUME=1 to continue from last.pt
EPOCHS = int(os.environ.get("MATCH_EPOCHS", "0"))
ITERS = int(os.environ.get("MATCH_ITERS", "0"))

nb = json.loads(open(NB, encoding="utf-8").read())
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def find(marker):
    for i, s in cells.items():
        if marker in s[:220]:
            return i, s
    raise KeyError(marker)


g = {"__name__": "__main__", "display": lambda *a, **k: None}
for m in ["1.1 environment", "2.1 installs", "2.2 preprocessing", "2.3 dataset",
          "2.3b prepare_folds", "2.4 model", "2.5 losses", "2.6 inference",
          "2.7 train_one", "2.8 eval_one"]:
    i, s = find(m)
    exec(compile(s, f"cell{i}", "exec"), g)
print("shared cells loaded", flush=True)

t_all = time.time()
for fold in range(5):
    if ONLY and int(ONLY) != fold:
        continue
    run = f"mwhqnet_matched_d{D_MODEL}_fold{fold}"
    out_dir = os.path.join(OUT, run)
    if os.path.exists(os.path.join(out_dir, "lastep", "summary.csv")):
        print(f"[skip] {run} already done", flush=True)
        continue
    _last = os.path.join(out_dir, "last.pt")
    if os.path.exists(_last) and not RESUME:
        print(f"[warn] {run} has a partial last.pt; pass MATCH_RESUME=1 to continue it", flush=True)

    i, s = find("3.7 config")
    exec(compile(s, f"cell{i}", "exec"), g)
    CFG = g["CFG"]
    CFG.update(D_MODEL=D_MODEL, FOLD=fold, RESUME=RESUME, SMOKE_TEST=False, NUM_WORKERS=0,
               RUN_NAME=run)
    if EPOCHS:
        CFG["EPOCHS"] = EPOCHS
    if ITERS:
        CFG["ITERS_PER_EPOCH"] = ITERS

    run_dir = os.path.join(OUT, run)
    os.makedirs(run_dir, exist_ok=True)
    g["RUN_DIR"] = run_dir
    json.dump({k: v for k, v in CFG.items() if isinstance(v, (int, float, str, bool, list, type(None)))},
              open(os.path.join(run_dir, "config.json"), "w"), indent=1, default=str)

    t0 = time.time()
    print(f"\n=== {run}  D_MODEL={D_MODEL}  fold={fold} ===", flush=True)
    g["prepare_folds"]()
    g["train_one"]()
    n_par = sum(p.numel() for p in g["model"].parameters()) / 1e6
    print(f"  params = {n_par:.2f} M", flush=True)
    # score the FINAL-epoch checkpoint only, at dose 1.0, into lastep/
    # (matches the last.pt protocol used for the re-scored Table II; skips the 3-dose sweep)
    import torch
    _sd = torch.load(os.path.join(run_dir, "last.pt"), map_location=g["dev"], weights_only=False)["model"]
    g["model"].load_state_dict(_sd)
    g["model"].eval()
    dfc, dfl = g["evaluate"](g["model"], g["VAL_KEYS"], CFG, dose=1.0, tta=True, max_vols=None)
    ld = os.path.join(run_dir, "lastep")
    os.makedirs(ld, exist_ok=True)
    dfc.to_csv(os.path.join(ld, "cases_dose1.0.csv"), index=False)
    dfl.to_csv(os.path.join(ld, "lesions_dose1.0.csv"), index=False)
    summ = pd.DataFrame({"dose=1.0": pd.Series(g["summarise"](dfc, dfl, CFG))})
    summ.to_csv(os.path.join(ld, "summary.csv"))
    print(f"  Dice masses = {summ.loc['Dice masses'].iloc[0]:.4f}", flush=True)
    print(f"  fold {fold} done in {(time.time()-t0)/60:.1f} min", flush=True)

print(f"\nALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)
