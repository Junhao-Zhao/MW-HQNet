"""A3 control: the full model with the graph decoder disabled (GRAPH="none"), fold 0.

Why this run is needed
----------------------
The manuscript's Table I row 3 is labelled "+PAWE+LCQS (no graph)", but that run's configuration
is GRAPH="selfattn" -- a self-attention variant, not a no-graph control. Without a true no-graph
run there is no way to attribute any change to the graph module (HQGD) itself.

This script runs that missing control. It reuses the notebook's own train_one()/eval_one() so the
architecture, loss, schedule, data pipeline and metric code are byte-identical to every other run
in the study -- only CFG["GRAPH"] differs.

GRAPH="none" is already supported by the model: DecLayer skips both the SAGE block and the
self-attention block for any value other than "graph"/"selfattn", so only the cross-attention and
feed-forward paths remain. No model code is modified.

Output: runs/a3_nograph_fold0/  (same file names as every other run)
"""
import json
import os
import time

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
RUN_NAME = os.environ.get("A3_RUN", "a3_nograph_fold0")
GRAPH = os.environ.get("A3_GRAPH", "none")

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
          "2.7 train_one", "2.8 eval_one", "3.7 config"]:
    i, s = find(m)
    exec(compile(s, f"cell{i}", "exec"), g)
    print(f"  loaded cell {i}: {m}", flush=True)

CFG = g["CFG"]
print("\n=== config before override ===", flush=True)
for k in ("RUN_NAME", "FOLD", "EPOCHS", "ENHANCE", "AIFI", "QUERY", "GRAPH", "FUSION_BETA", "NUM_WORKERS"):
    print(f"  {k:12s} {CFG.get(k)}", flush=True)

CFG.update(GRAPH=GRAPH, RUN_NAME=RUN_NAME, RESUME=False, SMOKE_TEST=False, NUM_WORKERS=0)
_ep, _it = int(os.environ.get("A3_EPOCHS", "0")), int(os.environ.get("A3_ITERS", "0"))
if _ep:
    CFG["EPOCHS"] = _ep
if _it:
    CFG["ITERS_PER_EPOCH"] = _it

RUN_DIR = os.path.join(CFG["OUT_DIR"], RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)
g["RUN_DIR"] = RUN_DIR
json.dump({k: v for k, v in CFG.items() if isinstance(v, (int, float, str, bool, list, type(None)))},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=1, default=str)
print(f"\n=== running {RUN_NAME}  GRAPH={CFG['GRAPH']}  ->  {RUN_DIR} ===\n", flush=True)

t0 = time.time()
g["prepare_folds"]()
g["train_one"]()
g["eval_one"]()
print(f"\n=== finished in {(time.time()-t0)/60:.1f} min ===", flush=True)
