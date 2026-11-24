"""Produce a 64 px version of the methodology figure from the supervisor's draw.io source.

Findings from the analysis:
  * canvas 4500 x 5445, body text 36-42 px  ->  2.6-3.0 pt at 11.5 cm, 3.9-4.2 pt at full width;
  * to reach 7 pt at full width (17.2 cm) the text only has to grow to 64 px (1.6x);
  * at 64 px only 9 of 38 cells overflow, and by 11-83 px each.

So the fix is: every font 64 px, shorten the six long labels, and raise the two thin caption strips
from 75 to 92 px. The page grows by only ~30 px, which at 17.2 cm keeps the figure about 21 cm tall.

Written to MW-HQNet_methodology_v2.drawio; the original is untouched.
"""
import re
import xml.etree.ElementTree as ET

SRC = r"E:\segmentation research paper\change\MW-HQNet_methodology.drawio"
DST = r"E:\segmentation research paper\change\MW-HQNet_methodology_v2.drawio"
NEW_FS = 64

# whole-value overrides: id -> (new value html, new height or None)
OVERRIDE = {
    "b1_problem": ("<b>Problem:</b> small, low-contrast lesions (&lt; 10 mm)", None),
    "b4_out": ("<b>Output:</b> kidney / masses / tumor / cyst", None),
    "b4_leg_lq": ("<b>Query nodes</b>", None),
    "b4_leg_rt": ("<b>Region tokens</b>", None),
    "b4_leg_gk": ("<b>Kidney node</b>", None),
    "b4_gate": ("<b>Fusion gate</b><br>β = 0.5", None),
    "b4_combine": ("<i>Query-level reasoning + dense voxel features</i>", 92),
    "b5_ab": ("<b>Ablation</b>", None),
    "b5_outcome": ("<i>Outcome: does PAWE + LCQS + HQGD improve small-lesion performance?</i>", 92),
}

raw = open(SRC, encoding="utf-8").read()

# 1. every style fontSize -> 64
raw = re.sub(r"fontSize=\d+", "fontSize=%d" % NEW_FS, raw)
# 2. every inline font-size:NNpx -> 64px
raw = re.sub(r"font-size:\s*\d+px", "font-size:%dpx" % NEW_FS, raw)

t = ET.fromstring(raw)
model = t.find(".//mxGraphModel")

changed = []
for c in model.findall(".//mxCell"):
    cid = c.get("id")
    if cid not in OVERRIDE:
        continue
    val, newh = OVERRIDE[cid]
    old = c.get("value")
    c.set("value", val)
    if newh is not None:
        g = c.find("mxGeometry")
        g.set("height", str(newh))
    changed.append((cid, old, val))

out = ET.tostring(t, encoding="unicode")
out = out.replace("/>", " />")           # keep ET from emitting collapsed tags

open(DST, "w", encoding="utf-8", newline="\n").write(out)

print("  wrote %s" % DST)
print("  every font size set to %d px" % NEW_FS)
print("  %d labels rewritten:" % len(changed))
for cid, old, new in changed:
    o = re.sub(r"<[^>]+>", "", old or "").replace("&nbsp;", " ")
    n = re.sub(r"<[^>]+>", "", new).replace("&nbsp;", " ")
    print("    %-12s %s" % (cid, n.strip()[:64]))
    print("    %-12s   was: %s" % ("", o.strip()[:64]))

W, H = 4500, 5445
print("\n  projected size when placed 17.2 cm wide: %.1f x %.1f cm, text %.2f pt"
      % (17.2, 17.2 * H / W, NEW_FS / W * 17.2 / 2.54 * 72))
print("  projected size when placed 11.5 cm wide: %.1f x %.1f cm, text %.2f pt"
      % (11.5, 11.5 * H / W, NEW_FS / W * 11.5 / 2.54 * 72))
