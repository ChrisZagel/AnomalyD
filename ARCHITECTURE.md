# ARCHITECTURE

## 1) System overview

Industrial anomaly detection pipeline for MVTec-style data:

`image -> frozen backbone -> multi-level patch features -> running whitening -> fixed random projection -> prototype distances -> anomaly map + image score`

Backbone is always frozen (no finetuning).

## 2) Feature extraction pipeline

- timm backbone (ViT hooks or `features_only` maps)
- multi-level fusion (resize to common grid, L2 normalize, concat)
- flattened patch feature tensor per image

## 3) Transform stage

### Running whitening
- online per-dimension stats (`n_seen`, `mean`, `m2`, `var`)
- numerically stable batch updates
- conservative updates with optional momentum

### Fixed projection
- sparse random projection by default (`projection_dim=96`)
- optional gaussian projection
- matrix initialized once and then frozen

## 4) Prototype model

`PrototypeStore` in projected space:
- means
- diagonal variances
- counts
- meta: status (`stable/candidate/fading`), age, replay hits

Online updates are count-based to avoid abrupt drift.

## 5) Distance scoring

Supported distances:
- `l2` (default)
- `cosine`
- `mahalanobis_diag`

Image score is top-k mean over upsampled patch anomaly map.

## 6) Incremental update flow

`update_incremental(new_batch)`:
1. score with current model
2. acceptance gate (`image_score`, `max_patch`, anomalous area)
3. accepted data -> replay memories + prototype updates
4. candidate creation/promotion for new modes
5. periodic `consolidate()` trigger

## 7) Replay memory design

Three buffers:
- `coverage` (reservoir)
- `boundary` (hard normal cases)
- `recent` (drift adaptation)

Whitening/prototype consolidation must use replay mix, not recent-only.

## 8) Candidate prototype logic

If accepted sample is far from stable prototypes:
- spawn candidate prototype
- promote after repeated confirmations
- age out weak/old candidates

## 9) Diagnostics / logging modes

Two modes:
- **Normal mode**: lightweight summaries and timing only
- **Debug mode**: per-sample report, extra transform/prototype stats, bounded plots

Main classes:
- `RunContext`
- `LeanMetricsLogger`
- `DebugReporter`
- `ExperimentTableWriter`

## 10) Key config parameters

- transform: `transform_type`, `projection_type`, `projection_dim`, `whitening_eps`
- prototypes: `num_prototypes`, `distance_type`
- replay/update: `max_replay_features`, `candidate_promotion_hits`, `update_trigger_images`, `threshold_beta`
- diagnostics: `enable_diagnostics`, `debug_mode`, `save_per_sample_report`, `save_plots`, `num_visualization_examples`


## 11) Lean diagnostics behavior

- With `enable_diagnostics=True`, lightweight baseline outputs are always written (`summary.json`, `timing_summary.json`, `transform_summary.json`, `prototype_summary.json`, `incremental_update_log.csv`, `forgetting_report.csv`).
- With `debug_mode=True`, richer diagnostics (per-sample table + debug plots) are added on top.
- Timing and score-separation metrics reuse values already computed during fit/eval to keep runtime overhead low.


## 12) Single-stage vs two-stage inference

- **Single-stage**: full-image pass only (fast baseline, lower local refinement quality).
- **Two-stage** (default):
  1. Stage A global screening at `feature_size_factor=0.75` using `feature_layer_mode=fast_2layer`.
  2. ROI proposal from coarse anomaly map (top maxima + lightweight overlap suppression).
  3. Stage B selective ROI refinement at `refine_feature_size_factor=1.0`.
  4. Refined ROI scores merged into final pixel map.

This keeps average runtime low (no full-image high-res pass) while improving tiny/thin defect localization.

## 13) Feature layer modes

- `fast_2layer` (default): mid+late features only, lower backbone-side feature processing cost.
- `full_3layer`: previous richer fusion mode for comparison.
