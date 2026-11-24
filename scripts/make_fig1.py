"""Draw the MW-HQNet architecture figure with explicit tensor shapes for Fig. 1.

Two bands: (top) data path from the 5-slice input through PAWE, the shared 2-D ResNet-34 encoder,
AIFI and LCQS; (bottom) the two branches, dense decoder and query-graph decoder, fused at beta=0.5.

Tensor shapes are written as plain text so they render without mathtext.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = r"E:\segmentation research paper\change\fig1_architecture.png"

B = dict(data=("#dbe8f6", "#8fb2d6"), pawe=("#d7ecd0", "#7fb069"),
         enc=("#e2d8ef", "#9b7fc0"), qry=("#ece2f8", "#9b7fc0"),
         gra=("#fbe3c4", "#e0913c"), out=("#cfe8e2", "#3f8f7f"))

fig, ax = plt.subplots(figsize=(11.6, 5.4), dpi=400)
ax.set_xlim(0, 116)
ax.set_ylim(0, 54)
ax.axis("off")
ax.set_aspect("auto")


def box(x, y, w, h, title, shape, kind, fs=7.4, sub=None):
    fc, ec = B[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.30,rounding_size=0.8",
                                fc=fc, ec=ec, lw=0.9, zorder=2))
    ty = y + h / 2 + (1.35 if shape else 0) + (0.9 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="#1b1b1b", zorder=3, linespacing=1.3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.35, sub, ha="center", va="center",
                fontsize=6.5, color="#2b2b2b", zorder=3, linespacing=1.3)
    if shape:
        ax.text(x + w / 2, y + 1.35, shape, ha="center", va="center", fontsize=6.4,
                color="#333333", zorder=3)


def arr(p, q, color="#565656", ls="-", lw=0.95, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 linestyle=ls, shrinkA=1.5, shrinkB=1.5,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ================= band 1 =================
ax.text(1.0, 51.6, "Input and physics-aware enhancement", fontsize=7.6, color="#33506f",
        fontweight="bold", style="italic")

box(1.2, 40.0, 12.4, 8.6, "5 axial slices", "5 x 192 x 192", "data")

box(17.4, 37.6, 20.6, 13.4, "PAWE", "20 x 192 x 192", "pawe",
    sub="4 learnable windows\n+ tone curve\n+ noise-gated residual")
arr((13.7, 44.3), (17.2, 44.3))

# window strip inside band 1
ax.text(41.2, 50.6, "learnable window centres / widths", fontsize=6.0, color="#4d6b45")
for i, yy in enumerate([48.2, 45.6, 43.0, 40.4]):
    ax.add_patch(FancyBboxPatch((41.2, yy - 0.85), 7.0, 1.7,
                                boxstyle="round,pad=0.14,rounding_size=0.4",
                                fc="#f0f8ec", ec=B["pawe"][1], lw=0.55, zorder=2))
    ax.text(44.7, yy, f"W{i+1}", ha="center", va="center", fontsize=6.0, zorder=3)
for yy in [48.2, 45.6, 43.0, 40.4]:
    ax.plot([38.1, 41.0], [44.3, yy], color="#9bb088", lw=0.5, zorder=1)
arr((49.0, 44.3), (52.4, 44.3), color="#9bb088")

# ================= band 2 =================
ax.text(1.0, 34.4, "Shared encoder, global attention, lesion-centre query selection",
        fontsize=7.6, color="#4a3a6b", fontweight="bold", style="italic")

box(1.2, 22.4, 17.2, 10.4, "ResNet-34 encoder", "C4 x 24 x 24", "enc",
    sub="2-D weights, applied\nto each of the 5 slices")
arr((18.6, 27.6), (21.8, 27.6))

box(22.0, 22.4, 16.0, 10.4, "AIFI", "C5 x 6 x 6", "enc",
    sub="transformer block on\nthe coarsest feature map")
arr((38.2, 27.6), (41.4, 27.6))

box(41.6, 22.4, 20.2, 10.4, "LCQS", "32 x 256", "qry",
    sub="score every location,\nkeep the top-K centres,\nstart 32 learned queries")

# heat-map glyph
ax.add_patch(FancyBboxPatch((63.4, 23.4), 9.0, 8.4, boxstyle="round,pad=0.20,rounding_size=0.4",
                            fc="#f6f1fc", ec=B["qry"][1], lw=0.55, zorder=2))
for gx in range(4):
    for gy in range(3):
        a = 0.30 if (gx, gy) not in ((2, 1), (1, 2)) else 0.95
        ax.add_patch(plt.Rectangle((63.95 + gx * 2.05, 24.0 + gy * 2.65), 1.75, 2.35,
                                   fc="#8e6fb8", alpha=a, ec="none", zorder=3))
ax.text(67.9, 22.6, "lesion score map", ha="center", fontsize=5.8, color="#5b4a72")
arr((61.9, 27.6), (63.3, 27.6))

# ================= band 3 =================
ax.text(1.0, 17.4, "Two branches, fused per pixel", fontsize=7.6, color="#8a5a1f",
        fontweight="bold", style="italic")

box(1.2, 5.6, 17.2, 10.0, "U-Net decoder", "4 x 192 x 192", "data",
    sub="dense up-sampling\nbranch")

box(22.0, 4.4, 32.6, 12.8, "HQGD", "32 x 256", "gra",
    sub="hierarchical query-graph decoder\n3 x GraphSAGE + cross-attn\n"
        "queries . regions . kidney")

# three-level graph drawn in the right third of the HQGD box
gx, gy = 49.4, 10.8
for cx, cy, fc, ec in [(gx - 1.5, gy + 3.6, "#f0b8b8", "#c05050"),
                       (gx + 1.5, gy + 3.6, "#f0b8b8", "#c05050"),
                       (gx, gy + 1.5, "#a9c6ec", "#4a76b8"),
                       (gx, gy - 0.6, "#c9b3e0", "#7d5ba6")]:
    ax.add_patch(plt.Circle((cx, cy), 0.62, fc=fc, ec=ec, lw=0.6, zorder=4))
for a, b in [((gx - 1.5, gy + 3.6), (gx, gy + 1.5)), ((gx + 1.5, gy + 3.6), (gx, gy + 1.5)),
             ((gx - 1.5, gy + 3.6), (gx, gy - 0.6)), ((gx + 1.5, gy + 3.6), (gx, gy - 0.6))]:
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#8f8f8f", lw=0.5, zorder=3)

arr((9.8, 22.2), (9.8, 15.9), color="#6f8fb8")
arr((18.6, 27.4), (22.0, 18.0), color="#b9832f", ls=(0, (3.2, 2.0)), rad=0.12)

# ================= band 4 =================
box(57.0, 6.6, 17.0, 10.4, "Fusion gate", "4 x 192 x 192", "out",
    sub="per-pixel blend\nbeta = 0.5")
arr((18.6, 10.6), (56.8, 10.6), color="#6f8fb8")
arr((54.8, 10.8), (56.8, 12.4), color="#b9832f")

box(77.6, 6.6, 15.4, 10.4, "Segmentation head", "4 x 192 x 192", "out",
    sub="1 x 1 conv + softmax")
arr((74.2, 11.8), (77.4, 11.8))

box(96.4, 6.6, 17.2, 10.4, "Prediction", "4 x 192 x 192", "out",
    sub="background . kidney\nmasses . tumour / cyst")
arr((93.2, 11.8), (96.2, 11.8))

ax.text(1.2, 1.4,
        "Training objective:  cross-entropy + Dice over the four classes, a focal heat-map loss on the LCQS "
        "scores, and query classification / mask losses on the HQGD outputs.",
        fontsize=6.2, color="#333333", va="bottom")

plt.subplots_adjust(left=0.006, right=0.994, top=0.985, bottom=0.012)
fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print(f"  saved -> {OUT}")
