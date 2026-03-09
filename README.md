# MVTec AD `metal_nut` PoC for Google Colab

## Kurze Architekturzusammenfassung

Dieser Stand nutzt jetzt eine **voll inkrementelle, patch-basierte Anomalieerkennung** ohne PCA:

1. **Gefrorener Backbone** (timm, keine Gradientenupdates)
2. **Multi-Level Feature Fusion** (wie bisher)
3. **Running Whitening** (online Mean/Var, batch-stabil)
4. **Fixed Random Projection** (einmal initialisiert, dann fix)
5. **Inkrementelle Prototypen** (stable/candidate/fading)
6. **Replay Memory** (coverage/boundary/recent)
7. **Gated Online Updates + periodische Konsolidierung**

Pipeline:

`multi-level features -> running whitening -> fixed projection -> prototypes -> distance scoring`

---

## Start in Colab

```python
!git clone <DEIN_REPO_URL>
%cd AnomalyD
!pip install -r requirements.txt
!python main.py --backbone-model-name vit_base_patch14_dinov2.lvd142m
```

---

## Wichtige CLI-Parameter

- `--num-prototypes` (Default `512`)
- `--distance-type` (`l2`, `cosine`, `mahalanobis_diag`)
- `--projection-type` (`sparse_random_projection`, `gaussian_random_projection`)
- `--projection-dim` (Default `96`)
- `--projection-seed` (Default `42`)
- `--whitening-eps` (Default `1e-6`)

---

## Neue Kernkomponenten

- `IncrementalWhiteningStats`
- `FixedRandomProjector`
- `IncrementalWhitenedProjection`
- `ReplayMemoryManager`
- `PrototypeStore`
- `IncrementalADModel`

Implementiert in `app/incremental_model.py`.

---

## Beispielablauf (API)

```python
from app.incremental_model import IncrementalADModel

cfg = {
    "projection_type": "sparse_random_projection",
    "projection_dim": 96,
    "distance_type": "l2",
    "num_prototypes": 512,
}

model = IncrementalADModel(cfg)
model.fit_initial(train_patch_features)  # np.ndarray [N, D]

scores = model.predict(test_patch_features)

model.update_incremental([img1_patch_feats, img2_patch_feats])
model.consolidate()

model.save_state("incremental_state.pkl")
loaded = IncrementalADModel.load_state("incremental_state.pkl")
```

---

## Outputs

Unter `/content/project/outputs/`:

- `metrics.json` / `metrics.csv`
- `per_sample_report.csv`
- `per_defect_metrics.csv`
- `models/prototype_model.pkl` (inkl. inkrementellem State)
- `visualizations/*.png`
- `visualizations/final_top5_overlay_gallery.png`
