"""Re-run the fold-0 baseline with a different random seed, to measure run-to-run variance.

Why this run exists
-------------------
The paper's Table I baseline (0.681) and Table II baseline (0.718) are inconsistent with each
other, and our reproduced fold-0 baseline (0.7385 dev / 0.7511 TTA) is above both. Before
concluding that the paper's Table I is simply an outlier we must rule out the boring explanation:
that a single fold's Dice simply moves this much between runs.

Note on determinism: in the original pipeline `SliceDS.__init__` creates
`np.random.default_rng()` with no seed, so data augmentation is NOT governed by CFG["SEED"].
A rerun therefore differs in both weight initialisation and the augmentation stream, which is
exactly the variance we want to quantify. The dataset code is left untouched.

Everything else (config, code, data, split, budget, metric) is identical to
`baseline_unet_fold0` except CFG["SEED"] and the output folder name.
"""
import json
import os
import random
import time

import numpy as np
import torch

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
NEW_SEED = int(os.environ.get("SEED_RUN_SEED", "123"))
RUN_NAME = os.environ.get("SEED_RUN_NAME", f"baseline_unet_fold0_seed{NEW_SEED}")

nb = json.loads(open(NB, encoding="utf-8").read())
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def find(m):
    for i, s in cells.items():
        if m in s[:220]:
            return i, s
    raise KeyError(m)


g = {"__name__": "__main__", "display": lambda *a, **k: None}
for m in ["1.1 environment", "2.1 installs", "2.2 preprocessing", "2.3 dataset",
          "2.3b prepare_folds", "2.4 model", "2.5 losses", "2.6 inference",
          "2.7 train_one", "2.8 eval_one", "3.4 config"]:
    i, s = find(m)
    exec(compile(s, f"cell{i}", "exec"), g)
    print(f"  loaded cell {i}: {m}", flush=True)

CFG = g["CFG"]
print("\n=== config before override ===", flush=True)
for k in ("RUN_NAME", "FOLD", "EPOCHS", "ENHANCE", "AIFI", "QUERY", "SEED", "NUM_WORKERS"):
    print(f"  {k:12s} {CFG.get(k)}", flush=True)

CFG.update(SEED=NEW_SEED, RUN_NAME=RUN_NAME, RESUME=False, SMOKE_TEST=False, NUM_WORKERS=0)
_ep, _it = int(os.environ.get("SEED_RUN_EPOCHS", "0")), int(os.environ.get("SEED_RUN_ITERS", "0"))
if _ep:
    CFG["EPOCHS"] = _ep
if _it:
    CFG["ITERS_PER_EPOCH"] = _it

# re-seed AFTER changing CFG["SEED"] -- the config cell already seeded with 42
random.seed(NEW_SEED)
np.random.seed(NEW_SEED)
torch.manual_seed(NEW_SEED)
torch.cuda.manual_seed_all(NEW_SEED)

RUN_DIR = os.path.join(CFG["OUT_DIR"], RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)
g["RUN_DIR"] = RUN_DIR
json.dump({k: v for k, v in CFG.items() if isinstance(v, (int, float, str, bool, list, type(None)))},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=1, default=str)
print(f"\n=== running {RUN_NAME}  SEED={NEW_SEED}  ->  {RUN_DIR} ===\n", flush=True)

t0 = time.time()
g["prepare_folds"]()
g["train_one"]()
if os.environ.get("SEED_RUN_SKIP_EVAL", "0") != "1":
    g["eval_one"]()
print(f"\n=== finished in {(time.time()-t0)/60:.1f} min ===", flush=True)
