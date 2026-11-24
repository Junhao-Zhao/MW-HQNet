"""Emit the corrected draw.io file for Fig. 1, matching the real forward pass.

Fonts are Times New Roman throughout and every text element is 64 px. Because what matters on the
printed page is font size relative to canvas width, the whole geometry is scaled up by S so that
64 px on a 3880 px canvas lands at roughly 8 pt when the figure is placed 17.2 cm wide.

Verified against change/probe_arch.py:
    PAWE                    5x192x192  -> 20x192x192
    ResNet-34 (in_chans=20) 20x192x192 -> 64@96 64@48 128@24 256@12 512@6
    AIFI on e[4]            512@6      -> 192@6
    decoder with 4 skips    192@6      -> 128@12 -> 96@24 -> 64@48 -> 48@96 -> 32@192
    dense head              32@192     -> 4@192
    heat head on p2(48@96)  -> 1@96 -> top-32 centres
    query init              48+96+4    -> 32x192
    memory                  p16 144 tok + p8 576 tok = 720
    graph                   32 queries + 16 region + 1 kidney = 49 nodes
    3 x DecLayer            GraphSAGE(k=6) + cross-attn
    heads                   cls 32x3, mask 32x96x96
    fuse                    0.5 * dense + 0.5 * pq  (fixed, no learned gate)
"""
from xml.sax.saxutils import escape

OUT = r"E:\segmentation research paper\change\fig1_architecture.drawio"

S = 2.31
FONT = "Times New Roman"
F_TITLE = 64
F_DETAIL = 64
F_SHAPE = 64

C = dict(
    data=("#DAE8FC", "#6C8EBF"),
    pawe=("#D5E8D4", "#82B366"),
    enc=("#E1D5E7", "#9673A6"),
    graph=("#FFE6CC", "#D79B00"),
    mem=("#FFF2CC", "#D6B656"),
    out=("#D0E8E4", "#2F7F6F"),
)

N = [
    ("in", 40, 70, 300, 150, "Input", "", "5 x 192 x 192", "data"),
    ("pawe", 380, 70, 300, 150, "PAWE", "", "20 x 192 x 192", "pawe"),
    ("enc", 720, 70, 300, 150, "ResNet-34", "", "20 ch in", "enc"),
    ("aifi", 1060, 70, 300, 150, "AIFI", "", "192 @ 6x6", "enc"),

    ("dec", 40, 290, 640, 160, "U-Net decoder", "4 encoder skips", "32 @ 192x192", "data"),
    ("dense", 720, 290, 300, 160, "1 x 1 conv", "", "4 @ 192x192", "out"),

    ("hm", 40, 520, 250, 160, "Heat map", "", "1 @ 96x96", "enc"),
    ("topk", 320, 520, 240, 160, "Top-32", "peaks", "32 centres", "enc"),
    ("qinit", 590, 520, 250, 160, "Queries", "", "32 x 192", "enc"),
    ("decl", 870, 520, 360, 160, "3 x DecLayer", "GraphSAGE", "49 x 192", "graph"),
    ("heads", 1260, 520, 300, 160, "Heads", "cls / mask", "32 @ 96x96", "graph"),

    ("mem", 700, 740, 340, 150, "Memory", "", "720 tokens", "mem"),

    ("fuse", 1080, 740, 260, 150, "Fuse", "beta 0.5", "4 @ 192x192", "out"),
    ("pred", 1370, 740, 270, 150, "Output", "argmax", "4 @ 192x192", "out"),
]

L = [
    ("l1", 40, 26, 1300, 34, "A . input and encoding", "#33506F"),
    ("l2", 40, 244, 1300, 34, "B . dense decoder", "#33506F"),
    ("l3", 40, 474, 1400, 34, "C . query selection and graph", "#8A5A1F"),
    ("l4", 700, 694, 700, 34, "D . fusion and output", "#2F6F60"),
]

E = [
    ("in", "pawe", None, None),
    ("pawe", "enc", None, None),
    ("enc", "aifi", None, None),
    ("aifi", "dec", (0.5, 1), (0.6, 0)),
    ("enc", "dec", (0.8, 1), (0.8, 0)),
    ("dec", "dense", None, None),
    ("dec", "hm", (0.05, 1), (0.5, 0)),
    ("hm", "topk", None, None),
    ("topk", "qinit", None, None),
    ("qinit", "decl", None, None),
    ("decl", "heads", None, None),
    ("mem", "decl", (0.5, 0), (0.5, 1)),
    ("dense", "fuse", (0.5, 1), (0.3, 0)),
    ("heads", "fuse", (0.4, 1), (0.9, 0)),
    ("fuse", "pred", None, None),
]

PAGE_W = int(1680 * S)
PAGE_H = int(910 * S)


def sc(v):
    return int(round(v * S))


def node(id_, x, y, w, h, title, detail, shape, kind):
    f, s = C[kind]
    parts = ["<b><font style='font-size:%dpx'>%s</font></b>" % (F_TITLE, title)]
    if detail:
        parts.append("<font style='font-size:%dpx'>%s</font>" % (F_DETAIL, detail))
    if shape:
        parts.append("<font style='font-size:%dpx' color='#333333'><i>%s</i></font>"
                     % (F_SHAPE, shape))
    val = "<br>".join(parts)
    style = ("rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
             "fillColor=%s;strokeColor=%s;strokeWidth=3;"
             "align=center;verticalAlign=middle;spacingTop=6;spacingBottom=6;"
             "fontFamily=%s;" % (f, s, FONT))
    return ('        <mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">\n'
            '          <mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/>\n'
            "        </mxCell>\n"
            % (id_, escape(val), style, sc(x), sc(y), sc(w), sc(h)))


def label(id_, x, y, w, h, text, color):
    style = ("text;html=1;align=left;verticalAlign=middle;fontSize=%d;fontStyle=1;"
             "fontFamily=%s;fontColor=%s;" % (F_TITLE, FONT, color))
    return ('        <mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">\n'
            '          <mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/>\n'
            "        </mxCell>\n"
            % (id_, escape(text), style, sc(x), sc(y), sc(w), sc(h)))


def edge(id_, src, tgt, ex, en, dashed=False):
    style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;"
             "strokeWidth=3;strokeColor=#555555;")
    if dashed:
        style += "dashed=1;"
    if ex:
        style += "exitX=%s;exitY=%s;exitDx=0;exitDy=0;" % ex
    if en:
        style += "entryX=%s;entryY=%s;entryDx=0;entryDy=0;" % en
    return ('        <mxCell id="%s" style="%s" edge="1" parent="1" source="%s" target="%s">\n'
            '          <mxGeometry relative="1" as="geometry"/>\n'
            "        </mxCell>\n" % (id_, style, src, tgt))


body = ""
for lb in L:
    body += label(*lb)
for nd in N:
    body += node(*nd)
for k, (s, t, ex, en) in enumerate(E, 1):
    body += edge("e%d" % k, s, t, ex, en, dashed=(s == "enc" and t == "dec"))

xml = '''<mxfile host="app.diagrams.net" agent="dsh" version="24.0.0">
  <diagram id="mwhqnet-arch" name="Fig1 Architecture">
    <mxGraphModel dx="%d" dy="%d" grid="1" gridSize="10" guides="1" tooltips="1"
                  connect="1" arrows="1" fold="1" page="1" pageScale="1"
                  pageWidth="%d" pageHeight="%d" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
%s      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
''' % (PAGE_W, PAGE_H, PAGE_W, PAGE_H, body)

open(OUT, "w", encoding="utf-8", newline="\n").write(xml)
print("  wrote %s" % OUT)
print("  canvas %d x %d px, all text %d px, font %s" % (PAGE_W, PAGE_H, F_TITLE, FONT))
print("  at 17.2 cm wide the text is about %.1f pt" % (F_TITLE / PAGE_W * 17.2 / 2.54 * 72))
print("  %d nodes, %d edges, %d labels, %.1f KB"
      % (len(N), len(E), len(L), len(xml.encode()) / 1024))
