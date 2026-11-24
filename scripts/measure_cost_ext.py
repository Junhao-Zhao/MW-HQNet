"""Extend cost_table.json with the 3-D baseline, the capacity-matched model, and per-case timings.

Fixes vs the first attempt:
  * CFG comes from a config cell (3.7), not from the shared cells.
  * the 3-D model class is extracted from train_nnunet_lite.py as source text and exec'd on its own,
    so importing it never starts a training run.
"""
import json
import os
import re
import time

import numpy as np
import torch

OUT = r"E:\segmentation research paper\change\cost_table.json"
NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
NNL = r"E:\segmentation research paper\change\train_nnunet_lite.py"

rows = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else []
if isinstance(rows, dict):          # already extended once
    rows = rows["per_sample"]

nb = json.loads(open(NB, encoding="utf-8").read())
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def find(marker):
    for i, s in cells.items():
        if marker in s[:220]:
            return i, s
    raise KeyError(marker)


g = {"__name__": "__main__", "display": lambda *a, **k: None}
for m in ["1.1 environment", "2.1 installs", "2.2 preprocessing", "2.3 dataset",
          "2.3b prepare_folds", "2.4 model", "2.5 losses", "3.7 config"]:
    i, s = find(m)
    exec(compile(s, f"cell{i}", "exec"), g)
MWHQNet = g["MWHQNet"]
dev = torch.device("cuda")


def bench(model, x, n=10):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(2):
            model(x)
    torch.cuda.synchronize()
    ts = []
    with torch.no_grad():
        for _ in range(n):
            torch.cuda.synchronize(); t0 = time.perf_counter(); model(x); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                model(x)
        fl = fc.get_total_flops() / 1e9
    except Exception:
        fl = None
    return float(np.median(ts)) * 1000, torch.cuda.max_memory_allocated() / 1024 ** 2, fl


NAMES = {r["config"] for r in rows}

# ---------- capacity-matched ----------
label = "MW-HQNet capacity-matched (D_MODEL=64)"
if label not in NAMES:
    CFG = dict(g["CFG"]); CFG.update(D_MODEL=64, NUM_WORKERS=0)
    m = MWHQNet(CFG).to(dev).eval()
    n = sum(p.numel() for p in m.parameters())
    lat, peak, fl = bench(m, torch.randn(1, 5, 192, 192, device=dev))
    rows.append(dict(config=label, params_m=n / 1e6, trainable_m=n / 1e6, flops_g=fl,
                     peak_mib=peak, latency_ms=lat))
    print(f"[ok] {label}: {n/1e6:.2f}M  {fl:.2f}G  {peak:.1f}MiB  {lat:.1f}ms", flush=True)
    del m; torch.cuda.empty_cache()

# ---------- 3-D nnU-Net-lite: pull only the class definitions out of the training script ----------
label3 = "3-D nnU-Net-lite (patch 32x128x128)"
if label3 not in NAMES:
    src = open(NNL, encoding="utf-8-sig").read()
    # exec everything above the training loop: imports, constants (BASE_CH, PATCH), model classes.
    # The file only starts training at the bottom, so this never launches a run.
    end = src.index("class PatchDS") if "class PatchDS" in src else len(src)
    body = src[:end]
    ns = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    exec(body, ns)
    PATCH = tuple(int(x) for x in re.search(r"PATCH\s*=\s*\(([^)]+)\)", src).group(1).split(","))
    net3 = ns["SegResLite3D"]().to(dev).eval()
    n3 = sum(p.numel() for p in net3.parameters())
    lat, peak, fl = bench(net3, torch.randn(1, 1, *PATCH, device=dev))
    rows.append(dict(config=label3, params_m=n3 / 1e6, trainable_m=n3 / 1e6, flops_g=fl,
                     peak_mib=peak, latency_ms=lat))
    print(f"[ok] {label3}: {n3/1e6:.2f}M  {fl:.2f}G  {peak:.1f}MiB  {lat:.1f}ms/patch  patch={PATCH}",
          flush=True)
    del net3; torch.cuda.empty_cache()

# ---------- per-case inference time (derived) ----------
D_TYP = 52
derived = {}
for r in rows:
    if r["config"].startswith("3-D"):
        w = 27   # 32*patch/2 covers ~192x192x48 in 3x3x3 windows
        derived[r["config"]] = dict(per_case_s=r["latency_ms"] * w / 1000.0, forwards=w,
                                    note="sliding window, stride = patch/2")
    else:
        derived[r["config"]] = dict(per_case_s=r["latency_ms"] * D_TYP / 1000.0, forwards=D_TYP,
                                    note=f"one 5-slice sample per axial slice, D~{D_TYP}")

json.dump(dict(per_sample=rows, per_case_derived=derived), open(OUT, "w", encoding="utf-8"), indent=1)

print()
print("=" * 106)
print("  Computational cost -- per-sample measured, per-case DERIVED (not end-to-end measured)")
print("=" * 106)
print(f"  {'configuration':46s} {'params':>8s} {'FLOPs':>8s} {'mem':>9s} {'lat':>9s} {'per case':>9s}")
print("  " + "-" * 102)
for r in rows:
    f = f"{r['flops_g']:.2f}G" if r.get("flops_g") else "n/a"
    d = derived[r["config"]]["per_case_s"]
    print(f"  {r['config'][:46]:46s} {r['params_m']:7.2f}M {f:>8s} {r['peak_mib']:7.1f}MB "
          f"{r['latency_ms']:7.1f}ms {d:8.2f}s")
print(f"\n  saved -> {OUT}")
