# MW-HQNet

Reference implementation and training notebooks for **MW-HQNet: A 2.5D Multi-Window Query-Graph
Network for Small Renal Lesion Segmentation in CT**.

MW-HQNet extends a 2.5-D ResNet-34 U-Net with three components:

| Component | What it does |
|---|---|
| **PAWE** — Physics-Aware Multi-Window Enhancement | learns four Hounsfield-unit windows jointly with the segmentation objective and stacks them with five adjacent slices into a 20-channel input |
| **LCQS** — Lesion-Centre Query Selection | scores the decoder feature and initialises 32 learned queries at the strongest peaks |
| **HQGD** — Hierarchical Query-Graph Decoder | GraphSAGE message passing plus cross-attention over lesion, region and kidney nodes, fused with the dense predictions at a fixed per-pixel weight |

---

## Where the data comes from

**No imaging data are redistributed in this repository.** All experiments use the public **KiTS23**
training cohort: 489 contrast-enhanced abdominal CT volumes with voxel-level kidney, tumour and cyst
annotations.

| What | Where |
|---|---|
| Official challenge record | https://doi.org/10.5281/zenodo.7840134 |
| Per-case NIfTI mirror used by this code | https://huggingface.co/datasets/MedOtter/kits23 |
| Fallback mirror | https://huggingface.co/datasets/Angelou0516/kits23 |
| Pre-processed 192 x 192 ROI cache used by the training notebooks | https://www.kaggle.com/datasets/iqtidarali/kits23-roi-192 |

The notebooks stream the raw cases from the Hugging Face mirror one at a time, crop each kidney to a
192 x 192 ROI at 1.0 x 1.0 x 2.5 mm, and delete the raw volume, so a full preprocessing pass needs
only a few GB of disk. Point `HF_REPO` and `DATA_DIR` in the first cell at your own locations.

The mirror holds 589 case directories (`case_00000` to `case_00588`). **This study uses the 489
official KiTS23 training cases**, selected by the patient-level five-fold split generated with seed
42 in `pipeline/nb_cells/06_00_splits.py`; the remaining directories are not read.

The mirror is distributed under **CC BY-NC-SA 4.0** (non-commercial, share-alike). The data are not
included here, so that licence is not redistributed; please respect it when you download them. When
you use this data, cite the KiTS23 record above as well as the paper.

---

## Environment

    Python 3.10.22
    PyTorch 2.14.1 (CUDA 12.6)
    numpy 2.2.6
    NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB

    pip install -r requirements.txt

---

## `notebooks/` — the runs that produced the reported numbers

| Notebook | Produces |
|---|---|
| `Step1_Preprocess_KiTS23.ipynb` | resampling to 1.0 x 1.0 x 2.5 mm and the 192 x 192 ROI cache |
| `Step2_Baseline_UNet.ipynb` | 2.5-D ResNet-34 U-Net baseline, fold 0 |
| `Step3_MWHQNet_Fixed_v3.ipynb` | full MW-HQNet, fold 0 |
| `Step4a_Ablation_PAWE_only.ipynb` | ablation: + PAWE only (Table 1) |
| `Step4b_Ablation_PAWE_LCQS_noGraph.ipynb` | ablation: + PAWE + LCQS without the graph (Table 1) |
| `step5-baseline-fold1..4.ipynb` | baseline on folds 1 to 4 |
| `step5-mwhqnet-fold1..4.ipynb` | full model on folds 1 to 4 |

Together these give Table 1 (fold-0 ablation) and Table 2 (five-fold cross-validation). Every
notebook is self-contained: it defines the model, trains one configuration on one fold and writes
`summary.csv`. All runs use seed 42 and 15 epochs, exactly as reported.

---

## `scripts/` — the experiments behind the remaining tables and figures

| Script | Produces |
|---|---|
| `train_matched.py` | capacity-matched control at D = 64 (22.78 M parameters), Table 6 |
| `train_nnunet_lite.py`, `eval_nnunet_lite.py` | 3-D SegResNet-style baseline with sliding-window inference, Table 5 |
| `train_a3_nograph.py` | the no-graph control row of Table 1 |
| `train_baseline_seed123.py` | the seed-123 rerun used to quantify run-to-run variance |
| `compute_table2_3.py` | Table 2 (five-fold mean and SD) and Table 3 (paired Wilcoxon) |
| `compute_full_metrics.py` | the voxel / lesion / image metric set, Table 7 |
| `compute_bootstrap_all.py` | paired case- and lesion-level bootstrap intervals |
| `compute_lowdose_bootstrap.py` | low-dose robustness and the case-level bootstrap |
| `compute_specificity_accuracy.py` | the specificity and accuracy rows of Table 7 |
| `measure_cost.py`, `measure_cost_ext.py` | parameters, FLOPs, peak memory, latency, Table 8 |
| `make_fig1.py`, `make_drawio.py` | Figure 1, the architecture |
| `make_fig2_v2.py` | Figure 2, the study design |
| `make_fig3.py` | Figures 3 and 4, the qualitative panels |
| `probe_arch.py` | prints the tensor shape after every stage of the forward pass |

Each script states at the top which claim in the manuscript it supports.

---

## `pipeline/` — the same pipeline as importable modules

`pipeline/nb_cells/` holds the model and training code as individual modules. The architecture lives
in `07_00_model.py`:

    PAWE, AIFI, Up, SAGE, DecLayer, MWHQNet

These are the same classes and hyper-parameters as in the notebooks above, only split out for
readability.

    pipeline/nb_cells/01_00_setup.py    paths, seeds, PROFILE, global configuration
    pipeline/nb_cells/03_00_data.py     streaming from Hugging Face, resampling, ROI crops
    pipeline/nb_cells/04_00_localizer.py image-only 3-D kidney localiser
    pipeline/nb_cells/07_00_model.py    PAWE, AIFI, SAGE, DecLayer, MWHQNet
    pipeline/nb_cells/09_00_train.py    training loop
    pipeline/nb_cells/13_00_stats.py    Wilcoxon and bootstrap
    pipeline/nb_cells/14_00_tables.py   table assembly
    ... and the remaining cells of the pipeline

`pipeline/MW-HQNet_Full_Pipeline.ipynb` chains them into one runnable notebook. Edit the `CFG`
paths, set `PROFILE` in cell 1 and run all:

| `PROFILE` | What runs | Cost |
|---|---|---|
| `"audit"` | dataset audit only | CPU, under a minute |
| `"pilot"` | audit plus the fold-0 ablation | about 4 h on the RTX 4060 |
| `"full5"` | audit plus baseline and full model on all five folds | about 9 h |

`pipeline/Oracle_ROI_Pilot.ipynb` reproduces the ROI-assumption experiment from the Limitations
section: an image-only 3-D kidney localiser trained on 102 cases and evaluated on the 40 held-out
fold-0 cases. `Oracle_ROI_Pilot_executed.ipynb` is the same notebook with its outputs.

---

## Reported results

Five-fold, 489 KiTS23 cases, annotation-derived kidney ROIs:

| Metric | Baseline | MW-HQNet |
|---|---|---|
| Dice kidney+masses | 0.9594 | 0.9576 |
| Dice masses | 0.7298 | 0.7409 |
| Dice tumor | 0.6650 | 0.6747 |
| Dice cyst | 0.4446 | 0.4349 |
| Sensitivity, all lesions | 0.6233 | 0.6824 |
| Sensitivity, lesions < 10 mm | 0.2269 | 0.3327 |
| False positives per ROI | 1.0582 | 1.5105 |

The differences are consistent in direction across all five folds but do not reach conventional
significance at n = 5: the exact two-sided signed-rank test cannot fall below p = 0.0625.

---

## Citation

    Zhao, J.; Ali, I.; Li, J. MW-HQNet: A 2.5D Multi-Window Query-Graph Network for Small Renal
    Lesion Segmentation in CT. Biomedicines 2026, submitted.

If you use the data, please also cite the KiTS23 challenge record:
https://doi.org/10.5281/zenodo.7840134

## Licence

Code released for academic use. The KiTS23 data remain subject to their own licence and are not
included here.
