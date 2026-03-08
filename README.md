# MVTec AD `metal_nut` PoC for Google Colab

## Kurze Architekturzusammenfassung

Dieser PoC implementiert eine **normal-only, patch-basierte Anomalieerkennung** auf **MVTec AD `metal_nut`**:

1. **Gefrorener Backbone:** vortrainiertes timm-Backbone (konfigurierbar über `--backbone-model-name`).
2. **Multi-Level Features:** 3 Ebenen (ViT-Intermediate oder letzte 3 `features_only`-Maps bei CNN/Hybrid-Backbones).
3. **Feature-Fusion:** Ebenen auf gemeinsames Raster, L2-Norm, Concat.
4. **PCA + Prototypen:** PCA-Reduktion und MiniBatchKMeans auf `train/good`.
5. **Inference:** Distanz zu Prototypen -> Pixel-Anomaly-Map + Image-Score.
6. **Evaluation:** Image-AUROC + Pixel-AUROC, plus Profiling-/Diagnose-Outputs.

Die Pipeline läuft auf CPU und GPU in Colab.

---

## Start in Colab

```python
!git clone <DEIN_REPO_URL>
%cd AnomalyD
!pip install -r requirements.txt
!python main.py --backbone-model-name vit_base_patch14_dinov2.lvd142m
```

Beispiele:

```python
!python main.py --backbone-model-name vit_small_patch14_dinov2.lvd142m
!python main.py --backbone-model-name shvit_s4.in1k
!python main.py --backbone-model-name edgenext_small.usi_in1k
```

---

## Unterstützte Backbones / Weights

Folgende Namen werden explizit unterstützt:

- `vit_base_patch14_dinov2.lvd142m`
- `vit_base_patch14_reg4_dinov2.lvd142m`
- `vit_small_patch14_dinov2.lvd142m`
- `shvit_s4.in1k`
- `edgenext_small.usi_in1k`

---

## Wichtige CLI-Parameter

- `--backbone-model-name` (siehe Liste oben)
- `--feature-size-factor` (z. B. `1.0`, `0.75`, `0.5`)
- `--num-prototypes` (z. B. `256`)
- `--pca-dim` (z. B. `128`)
- `--distance-type` (`cosine` oder `l2`)
- `--num-visualization-examples` (Default `5`)

---

## Outputs

Nach erfolgreichem Lauf unter `/content/project/outputs/`:

- `metrics.json` / `metrics.csv`
- `per_sample_report.csv`
- `models/prototype_model.pkl`
- `visualizations/*.png`
- `visualizations/final_top5_overlay_gallery.png`

---

## Hinweis

ADPretrain-Checkpoint-Support wurde entfernt; der PoC arbeitet jetzt ausschließlich mit direkt verfügbaren pretrained timm-Weights über `--backbone-model-name`.
