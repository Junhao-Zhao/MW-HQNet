"""Probe the real forward pass of MW-HQNet and print the shape after every top-level module.

Shapes only, no tensor values.
"""
import json

import torch

NB = r"E:\segmentation research paper\MW-HQNet_Full_Pipeline.ipynb"
nb = json.loads(open(NB, encoding="utf-8").read())
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def find(m):
    for i, s in cells.items():
        if m in s[:220]:
            return s
    raise KeyError(m)


g = {"__name__": "__main__", "display": lambda *a, **k: None}
for m in ["1.1 environment", "2.1 installs", "2.2 preprocessing", "2.3 dataset",
          "2.3b prepare_folds", "2.4 model", "2.5 losses", "2.6 inference", "3.7 config"]:
    exec(compile(find(m), "c", "exec"), g)

CFG, MW = g["CFG"], g["MWHQNet"]
cfg = dict(CFG); cfg.update(FOLD=0, NUM_WORKERS=0, SMOKE_TEST=False, PRETRAINED=False)
net = MW(cfg).cuda().eval()

print("  N_SLICES=%d  D_MODEL=%d  N_QUERIES=%d  N_DEC_LAYERS=%d  KNN=%d  MASK_DIM=%d  BETA=%.2f"
      % (cfg["N_SLICES"], cfg["D_MODEL"], cfg["N_QUERIES"], cfg["N_DEC_LAYERS"],
         cfg["KNN"], cfg["MASK_DIM"], cfg["FUSION_BETA"]))
print("  params %.2f M" % (sum(p.numel() for p in net.parameters()) / 1e6))
print()

order = []
hooks = []
for name, mod in net.named_children():
    def fn(m, i, o, nm=name):
        if isinstance(o, torch.Tensor):
            order.append((nm, tuple(o.shape)))
        elif isinstance(o, (list, tuple)) and o and isinstance(o[0], torch.Tensor):
            order.append((nm, [tuple(t.shape) for t in o]))
    hooks.append(mod.register_forward_hook(fn))

with torch.no_grad():
    out = net(torch.randn(1, cfg["N_SLICES"], 192, 192).cuda(), tf_p=0.0)

print("  module -> output shape")
print("  " + "-" * 70)
for nm, shp in order:
    print(f"    {nm:10s} {shp}")

print("\n  encoder pyramid")
for i, (c, s) in enumerate(zip(net.enc.feature_info.channels(), net.enc.feature_info.reduction())):
    print(f"    e[{i}] {c:4d} ch  stride {s:2d}  {192 // s} x {192 // s}")

print("\n  decoder (Up modules)")
for nm in ("u16", "u8", "u4", "u2", "u1"):
    m = getattr(net, nm)
    attrs = {k: v for k, v in vars(m).items() if isinstance(v, int)}
    print(f"    {nm:4s} {attrs}")

print("\n  heads")
print(f"    dense_head  {net.dense_head.in_channels} -> {net.dense_head.out_channels}")
print(f"    hm_head     48 -> 1")
print(f"    pix         48 -> {cfg['MASK_DIM']}")
print(f"    mem16       128 -> {cfg['D_MODEL']}     mem8 96 -> {cfg['D_MODEL']}")
print(f"    q_in        48+96+{net.pawe.K} -> {cfg['D_MODEL']}")
print(f"    region_proj {net.region_proj.in_features} -> {net.region_proj.out_features}")
print(f"    kidney_proj {net.kidney_proj.in_features} -> {net.kidney_proj.out_features}")
print(f"    cls_head    {net.cls_head.in_features} -> {net.cls_head.out_features}")
print(f"    mask_head   {cfg['D_MODEL']} -> {cfg['MASK_DIM']}")

print("\n  graph")
A = g["build_graph"](out["qp"].float(), cfg["KNN"])
print(f"    nodes = 32 queries + 16 region + 1 kidney = 49")
print(f"    adjacency A {tuple(A.shape)}  (kNN k={cfg['KNN']} over the 32 query positions)")

print("\n  returned")
for k, v in out.items():
    if isinstance(v, torch.Tensor):
        print("    %-6s %s" % (k, tuple(v.shape)))
    elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (tuple, list)):
        print("    %-6s list of %d, each %s / %s" % (k, len(v), v[0][0], v[0][1]))
    else:
        print("    %-6s %s" % (k, type(v).__name__))

for h in hooks:
    h.remove()
