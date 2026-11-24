# ============================================================================
# CELL 1 -Environment, paths, config, GPU probe
# Run this once per kernel. Everything downstream reads CFG / ENABLED from here.
# ============================================================================

#--------------------------------------------------------------------------
# 0. RUN PROFILE - the only line you normally change in this notebook
#--------------------------------------------------------------------------
# The ONE knob you normally change. Everything else follows from it.
#   "audit" -> only the table audit (CPU, under a minute)
#   "pilot" -> audit + 4 runs on fold 0  (B0, B1, A3, A4)  ~4 h on this GPU
#   "full5" -> audit + B0/B1 on all five folds             ~9 h on this GPU
PROFILE = "audit"

_PROFILES = {
    "audit": dict(audit=True,  splits=True,  stats=True,  tables=True,  gates=True,
                  data=False, localizer=False, roi=False, train=False, infer=False,
                  metrics=False, run_all=False, nnunet=False),
    "pilot": dict(audit=True,  splits=True,  stats=True,  tables=True,  gates=True,
                  data=False, localizer=False, roi=False, train=True,  infer=True,
                  metrics=True,  run_all=True,  nnunet=False),
    "full5": dict(audit=True,  splits=True,  stats=True,  tables=True,  gates=True,
                  data=False, localizer=False, roi=False, train=True,  infer=True,
                  metrics=True,  run_all=True,  nnunet=False),
}
if PROFILE not in _PROFILES:
    raise SystemExit(f"PROFILE must be one of {list(_PROFILES)}, got {PROFILE!r}")
ENABLED = dict(_PROFILES[PROFILE])


import os, sys, json, math, time, glob, random, shutil, subprocess, importlib, warnings, hashlib
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.0  Hugging Face mirror -MUST be set before importing huggingface_hub / timm.
#      huggingface.co is unreachable from mainland China; hf-mirror.com is not.
#      If you run this notebook inside Kaggle/Colab, set this to "https://huggingface.co".
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


NO_INSTALL = os.environ.get("MWHQ_NO_INSTALL", "0") == "1"


def _ensure(pkg, pipname=None):
    """Import a package, installing it first only if it is missing and allowed."""
    try:
        return importlib.import_module(pkg)
    except ImportError:
        if NO_INSTALL:
            raise
        print(f"  installing {pipname or pkg} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pipname or pkg], check=True)
        return importlib.import_module(pkg)


# Heavy GPU packages (torch, timm, scikit-learn) are installed in the cells that use them,
# so this first cell stays fast and works on a machine with no CUDA toolkit.
for _p, _n in [("numpy", None), ("pandas", None), ("openpyxl", None)]:
    _ensure(_p, _n)                      # hard requirements of the audit stage

for _p, _n in [("scipy", None), ("nibabel", None), ("matplotlib", None), ("tqdm", None),
               ("cv2", "opencv-python-headless"), ("huggingface_hub", None)]:
    try:
        _ensure(_p, _n)
    except Exception as _e:              # not fatal here; the consuming cell retries
        print(f"  warning: {_p} not installed now ({type(_e).__name__}); retried where used")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1.1  EDIT HERE -the only block you normally need to touch.
# ---------------------------------------------------------------------------
CFG = dict(
    # ---- where things live -------------------------------------------------
    WORK_ROOT=os.environ.get("MWHQ_WORK", r"E:\jupyter_work\reviewer_work"),   # outputs; needs ~60 GB free for the endo track
    LEGACY_DIR=r"E:\segmentation research paper\implementation\done",
    LEGACY_XLSX=r"E:\segmentation research paper\implementation\MWHQNet_fold0_results.xlsx",
    LEGACY_BASELINE_ZIP=r"E:\segmentation research paper\datsets\baseline-fold4-ckpt.zip",
    LEGACY_ROI_ZIP=r"E:\segmentation research paper\datsets\kits23-roi-192.zip",
    FOLDS_JSON_ZIP_ENTRY="runs/folds_kits23_5_seed42.json",   # the exact split used for the old runs

    # ---- which evaluation track -------------------------------------------
    # "oracle" = reuse your existing GT-cropped ROI cache (upper bound, cheap)
    # "endo"   = image-only localisation + native-grid scoring (what the reviewer wants)
    TRACK="oracle",

    # ---- how much to run ---------------------------------------------------
    FOLDS=[0, 1, 2, 3, 4],
    MAX_CASES_DOWNLOAD=None,   # e.g. 60 for a pilot; None = all 489
    DEV_FRACTION=0.15,         # inner-development fraction of the outer-training patients

    # ---- training (matched to your legacy runs so the numbers stay comparable) ----
    EPOCHS=15, ITERS_PER_EPOCH=800, LR=3e-4, WD=1e-4, WARMUP_EPOCHS=1,
    N_SLICES=5, ROI_PX=192, N_QUERIES=32, D_MODEL=192, N_DEC_LAYERS=3,
    MASK_DIM=64, KNN=6, FUSION_BETA=0.5, TF_P0=0.5,
    W_QUERY=0.3, W_HM=0.3, W_TV=1e-3, HM_WARMUP_EPOCHS=3,
    GRAD_CLIP=1.0, SEED=42, VAL_EVERY=3,

    # ---- evaluation --------------------------------------------------------
    TAU_HIGH=0.5, MIN_LESION_MM3=10.0, LESION_IOU_MATCH=0.10,
    NSD_TOL_MM=2.0, SIZE_BINS_MM=[0, 10, 20, 40, 1e9],
    BOOTSTRAP_N=10000,
)


ROOT = Path(CFG["WORK_ROOT"])
DIRS = {k: ROOT / k for k in
        ["audit", "cache_ct", "cache_seg", "raw", "localizer", "roi", "splits",
         "runs", "pred", "results", "figures", "manuscript", "logs"]}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.2  GPU / device probe
# ---------------------------------------------------------------------------
DEV = "cpu"
GPU_NAME, VRAM_GB = "cpu", 0.0
try:
    import torch
    torch.manual_seed(CFG["SEED"])
    torch.cuda.manual_seed_all(CFG["SEED"])
    if torch.cuda.is_available():
        DEV = "cuda"
        GPU_NAME = torch.cuda.get_device_name(0)
        VRAM_GB = torch.cuda.get_device_properties(0).total_mem / 1e9 if hasattr(
            torch.cuda.get_device_properties(0), "total_mem") else torch.cuda.get_device_properties(0).total_memory / 1e9
    amp_ok = DEV == "cuda" and torch.cuda.get_device_capability(0)[0] >= 7
except Exception as e:  # torch not installed yet -> Cell 7 installs the CUDA wheel
    print(f"torch not available yet ({type(e).__name__}); Cell 7 will install it.")
    amp_ok = False

# Batch size scaled to VRAM. Your legacy runs used 12 on a 16 GB T4.
if VRAM_GB >= 15:
    CFG["BATCH"] = 12
elif VRAM_GB >= 11:
    CFG["BATCH"] = 10
elif VRAM_GB >= 7.5:
    CFG["BATCH"] = 8
elif VRAM_GB > 0:
    CFG["BATCH"] = 4
else:
    CFG["BATCH"] = 2
CFG["AMP"] = bool(amp_ok)
CFG["NUM_WORKERS"] = 0 if os.name == "nt" else 4

random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])

# ---------------------------------------------------------------------------
# 1.3  Persist the run configuration (this file is what makes runs traceable)
# ---------------------------------------------------------------------------
def _env_report():
    import platform
    rep = dict(python=sys.version.split()[0], platform=platform.platform(),
               numpy=np.__version__, pandas=pd.__version__, device=DEV, gpu=GPU_NAME,
               vram_gb=round(VRAM_GB, 2), batch=CFG["BATCH"], amp=CFG["AMP"],
               seed=CFG["SEED"], track=CFG["TRACK"])
    for m in ["torch", "timm", "nibabel", "scipy", "sklearn"]:
        try:
            rep[m] = importlib.import_module(m).__version__
        except Exception:
            rep[m] = "not installed"
    return rep

ENV = _env_report()
CFG["_env"] = ENV
with open(ROOT / "config.json", "w") as f:
    json.dump(CFG, f, indent=1, default=str)


# ---------------------------------------------------------------------------
# 1.4  Lazy GPU-package installation.
#      torch is ~2.5 GB and must match your driver, so it is installed by the first cell
#      that needs it, from the CUDA wheel index instead of plain PyPI.
# ---------------------------------------------------------------------------
def ensure_torch():
    try:
        import torch as _t
        return _t
    except ImportError:
        if NO_INSTALL:
            raise
        print("installing torch (CUDA 12.4 wheels) - a few minutes on first run ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "--index-url", "https://download.pytorch.org/whl/cu124",
                        "torch", "torchvision"], check=True)
        import torch as _t
        return _t


def ensure_timm():
    try:
        import timm as _m
        return _m
    except ImportError:
        if NO_INSTALL:
            raise
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
        import timm as _m
        return _m


def df_to_md(df, index=True, floatfmt=".4f"):
    """Minimal markdown table writer, so the notebook does not need `tabulate`.

    Values are read column by column on purpose: `iterrows()` casts every row to one common dtype,
    which would turn an integer fold column into 0.0000.
    """
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for i in range(len(d)):
        cells = []
        for c in d.columns:
            v = d[c].iloc[i]
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else format(float(v), floatfmt))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)

# ---------------------------------------------------------------------------
# 1.5  ENVIRONMENT GUARD - this project runs in segkid, never in base.
#      Enforced in code so the rule cannot be forgotten: JupyterLab will happily
#      let you pick the base kernel, and then nothing imports and every run
#      silently records the wrong environment.
# ---------------------------------------------------------------------------
STRAY_ENV_ALLOWED = os.environ.get("MWHQ_ALLOW_ANY_ENV", "0") == "1"
ENV_IS_SEGKID = "segkid" in sys.executable.replace("\\", "/").lower()
CFG["REQUIRE_SEGKD"] = not STRAY_ENV_ALLOWED
CFG["_env"]["segkid"] = ENV_IS_SEGKID

print("=" * 78)
print(f"interpreter : {sys.executable}")
if ENV_IS_SEGKID:
    print("environment : OK - running inside segkid")
else:
    print("environment : *** WRONG - this is NOT the segkid environment ***")
    print("              In Jupyter: Kernel > Change Kernel > Python (segkid)")
    print("              Or launch with E:\\...\\start_jupyter_lab.bat")
    if CFG["REQUIRE_SEGKD"]:
        raise RuntimeError(
            "Refusing to run outside the segkid environment.\n"
            f"  current interpreter: {sys.executable}\n"
            "  fix: Kernel > Change Kernel > 'Python (segkid)'\n"
            "  or : double-click start_jupyter_lab.bat\n"
            "  (MWHQ_ALLOW_ANY_ENV=1 disables this guard for offline testing only)")

print("=" * 78)
print(f"device      : {DEV}  {GPU_NAME}  ({VRAM_GB:.1f} GB)")
print(f"track       : {CFG['TRACK']}")
print(f"batch / AMP : {CFG['BATCH']} / {CFG['AMP']}")
print(f"work root   : {ROOT}")
print(f"torch       : {ENV.get('torch')}")
print("=" * 78)
for k, v in ENABLED.items():
    print(f"  {'ON ' if v else 'off'}  {k}")
print("=" * 78)
print("Stage 2 (audit) is CPU-only and takes < 1 min. Run it now.")
