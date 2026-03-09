# AnomalyD

## Project purpose

Lean, patch-based industrial anomaly detection for MVTec-style data with a frozen backbone and a fully incremental normality model.

## Main architecture (short)

`image -> frozen backbone -> multi-level fused patch features -> running whitening -> fixed random projection -> prototypes -> anomaly score`

- no backbone finetuning
- no PCA
- incremental updates with replay memory and candidate prototypes

## Installation

```bash
pip install -r requirements.txt
```

## Dataset setup

Default run downloads/extracts `metal_nut` into `/content/data/mvtec_ad` if needed.
You can override paths in `PoCConfig` or adapt CLI usage in notebooks/scripts.

## Example commands

### Normal run (lean diagnostics)

```bash
python main.py --backbone-model-name vit_base_patch14_dinov2.lvd142m --enable-diagnostics
```

### Debug run (extended diagnostics)

```bash
python main.py --debug-mode --save-per-sample-report --save-plots --num-visualization-examples 5
```

### Incremental update run (API)

```python
from app.incremental_model import IncrementalADModel

model = IncrementalADModel({"projection_dim": 96, "distance_type": "l2", "num_prototypes": 512})
model.fit_initial(train_patch_features)
model.update_incremental([new_img_patch_feats])
model.consolidate()
model.save_state("incremental_state.pkl")
```

## Important parameters

- `--backbone-model-name`: pretrained timm backbone
- `--feature-size-factor`: controls processing resolution
- `--num-prototypes`: prototype capacity in projected space
- `--distance-type`: `l2`, `cosine`, or `mahalanobis_diag`
- `--enable-diagnostics/--disable-diagnostics`: on/off diagnostics outputs
- `--debug-mode`: enables richer diagnostics

## Output folder structure

Each run writes to:

`/content/project/outputs/<run_id>/`

Subfolders:
- `metrics/`
- `tables/`
- `plots/`
- `visualizations/`

Global comparison table:
- `/content/project/outputs/experiment_results.csv`

## Compare experiments

Use `experiment_results.csv` to compare runs by backbone, transform config, AUROC/AUPRO, and runtime in a single table.

## How to extend the system

- add new distance functions in `PrototypeStore`
- add new replay policies in `ReplayMemoryManager`
- keep hot path lean; add expensive diagnostics only behind `debug_mode`
- document architecture changes in `ARCHITECTURE.md`


## Diagnostics switches

- `--enable-diagnostics` (default): writes lean baseline metrics/tables.
- `--disable-diagnostics`: disables diagnostics files and keeps only console/final minimal output.
- `--debug-mode`: adds extended diagnostics (per-sample report and selected plots) on top of baseline diagnostics.

Normal diagnostics stay lightweight; debug mode is intended for deeper analysis.


## Single-stage vs two-stage commands (optional two-stage)

Fast single-stage baseline:

```bash
python main.py --feature-layer-mode single_last_layer --disable-two-stage-inference
```

Two-stage refinement mode (default):

```bash
python main.py --feature-layer-mode fast_2layer --enable-two-stage-inference --num-refine-rois 3 --image-score-mode hybrid_global_local
```

Use single-stage for strict latency benchmarking; use two-stage when tiny/local defect localization quality matters more.


## Fast CPU examples

A) Fastest default CPU run (single-pass):

```bash
python main.py --backbone-model-name edgenext_small.usi_in1k --feature-layer-mode single_last_layer --disable-two-stage-inference
```

B) `fast_2layer` run:

```bash
python main.py --feature-layer-mode single_last_layer --disable-two-stage-inference
```

C) OpenVINO CPU run:

```bash
python main.py --inference-backend openvino --feature-layer-mode single_last_layer --disable-two-stage-inference
```

D) Optional fast-mode benchmark:

```bash
python main.py --benchmark-fast-modes --max-allowed-pixel-auroc-drop 0.01
```


Tip for faster post-evaluation on CPU: reduce AUPRO integration steps, e.g. `--aupro-num-steps 60` (default is 80).
