"""Measure parameters, FLOPs, peak GPU memory and inference latency for every configuration.

Answers reviewer 1.5's explicit request:
  "Report parameters, FLOPs or approximate computational burden, GPU memory, inference time and
   post-processing settings."

No training involved -- models are instantiated from the same config cells the runs used.
"""
import json
import os
import time

import numpy as np
import torch

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
nb = json.loads(open(NB, encoding="utf-8").read())
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def find(marker):
    for i, s in cells.items():
        if marker in s[:220]:
            return i, s
    raise KeyError(marker)


g = {"__name__": "__main__", "display": lambda *a, **k: None}
for m in ["1.1 environment", "2.1 installs", "2.2 preprocessing", "2.3 dataset",
          "2.3b prepare_folds", "2.4 model", "2.5 losses"]:
    i, s = find(m)
    exec(compile(s, f"cell{i}", "exec"), g)

MWHQNet = g["MWHQNet"]
dev = torch.device("cuda")

CONFIGS = [
    ("2.5-D baseline (ResNet-34, single window)", "3.4 config", {}),
    ("+ PAWE", "3.8 config", {}),
    ("+ PAWE + LCQS, no graph", "3.7 config", {"GRAPH": "none"}),
    ("+ PAWE + LCQS, self-attention", "3.9 config", {}),
    ("Full MW-HQNet (+HQGD graph)", "3.7 config", {}),
]

# a 2.5-D "batch" is 5 slices; the model takes (B, 5, H, W)
H = W = 192
S = 5


def flops_of(model, x):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                model(x)
        return fc.get_total_flops()
    except Exception as e:
        return None


rows = []
for label, marker, over in CONFIGS:
    i, s = find(marker)
    exec(compile(s, f"cell{i}", "exec"), g)
    CFG = g["CFG"]
    CFG.update(over)
    CFG.update(NUM_WORKERS=0, RESUME=False, SMOKE_TEST=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = MWHQNet(CFG).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(1, S, H, W, device=dev)

    # warmup
    with torch.no_grad():
        for _ in range(2):
            model(x)
    torch.cuda.synchronize()

    # latency (median of 10)
    ts = []
    with torch.no_grad():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    lat = float(np.median(ts)) * 1000  # ms

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2  # MiB

    fl = flops_of(model, x)

    rows.append(dict(config=label, params_m=n_par / 1e6, trainable_m=n_tr / 1e6,
                     flops_g=None if fl is None else fl / 1e9,
                     peak_mib=peak, latency_ms=lat))

    print(f"[ok] {label:44s} params={n_par/1e6:6.2f}M  "
          f"flops={('%.2f G' % (fl/1e9)) if fl else 'n/a':>10s}  "
          f"peak={peak:7.1f} MiB  latency={lat:6.1f} ms", flush=True)
    del model
    torch.cuda.empty_cache()

print()
print("=" * 108)
print("  Table: computational cost (2.5-D path, one 5-slice sample, 192x192, batch 1)")
print("=" * 108)
print(f"  {'configuration':44s} {'params':>9s} {'FLOPs':>10s} {'peak GPU':>10s} {'latency':>9s}")
print("  " + "-" * 104)
for r in rows:
    f = f"{r['flops_g']:.2f} G" if r["flops_g"] else "n/a"
    print(f"  {r['config']:44s} {r['params_m']:8.2f}M {f:>10s} {r['peak_mib']:8.1f} MiB {r['latency_ms']:7.1f} ms")

out = r"E:\segmentation research paper\change\cost_table.json"
json.dump(rows, open(out, "w", encoding="utf-8"), indent=1)
print(f"\n  saved -> {out}")
