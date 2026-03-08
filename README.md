# MVTec AD `metal_nut` PoC for Google Colab

## Kurze Architekturzusammenfassung

Dieser PoC implementiert eine **normal-only, patch-basierte Anomalieerkennung** auf **MVTec AD `metal_nut`**:

1. **Gefrorener Backbone:** DINOv2-base (bevorzugt ADPretrain-Checkpoint).
2. **Multi-Level Features:** 3 Transformer-Ebenen (mittel, spät-mittel, spät).
3. **Feature-Fusion:** Ebenen auf ein gemeinsames Token-Raster, L2-Norm, Concat.
4. **Dimensionalitätsreduktion:** PCA (z. B. 128-dim).
5. **Normalitätsmodell:** MiniBatchKMeans-Prototypen auf `train/good`.
6. **Inference:** Minimaldistanz pro Patch zu Prototypen (Cosine/L2) → Anomaly Map + Image Score.
7. **Evaluation:** Image-AUROC + Pixel-AUROC auf `test/*` inkl. MVTec-Masken.

Die Pipeline ist CPU/GPU-kompatibel, ohne Finetuning, mit robusten Checks und Colab-freundlichen Abhängigkeiten.

---

## Vollständiger Code

Der vollständige PoC-Code liegt in:

- `app/metal_nut_poc.py`
- Einstiegspunkt: `main.py`

### Start in Colab

```python
!git clone <DEIN_REPO_URL>
%cd AnomalyD
!pip install -r requirements.txt
!python main.py --allow-backbone-fallback
```

Optional:

```python
!python main.py --feature-size-factor 0.5 --num-prototypes 128 --distance-type l2 --allow-backbone-fallback
```

Für einen **strict ADPretrain-Lauf ohne Fallback** nutze stattdessen:

```python
!python main.py
```

und lege vorher den offiziellen Checkpoint unter `/content/project/checkpoints/adpretrain_dinov2_base.pth` ab.

---

## Colab-Zellen-Reihenfolge (empfohlen)

1. **Setup + Install**
   - `pip install -r requirements.txt`
2. **Imports + Config prüfen**
   - optional Datei `app/metal_nut_poc.py` öffnen
3. **Geräteerkennung**
   - passiert automatisch (`cuda` falls verfügbar, sonst `cpu`)
4. **Download + Extraktion `metal_nut`**
   - automatisch über `ensure_dataset(...)`
5. **Dataset-Checks**
   - automatisch durch Strukturprüfungen
6. **Backbone laden**
   - ADPretrain-Checkpoint aus `/content/project/checkpoints/adpretrain_dinov2_base.pth`
7. **Trainingsfeatures extrahieren**
8. **PCA + Prototypen fitten**
9. **Evaluation auf Testset**
10. **Visualisierungen speichern**
11. **Metriken als JSON/CSV speichern**

---

## ADPretrain-Checkpoint bereitstellen

Standardpfad:

```text
/content/project/checkpoints/adpretrain_dinov2_base.pth
```

Verhalten:

- Wenn der Checkpoint fehlt und `allow_backbone_fallback=False` (Default), bricht der Code mit klarer Fehlermeldung ab.
- Optional kann ein Fallback auf vanilla DINOv2-base aktiviert werden:

```bash
python main.py --allow-backbone-fallback
```

Optional kannst du alternative timm-Pretrained-Weights für den Fallback setzen, z. B.:

```bash
python main.py --allow-backbone-fallback --fallback-model-name vit_base_patch14_reg4_dinov2.lvd142m
```

Dieser Fallback ist bewusst **nicht** stillschweigend aktiv.

Hinweis: Der ADPretrain-Loader akzeptiert auch häufige Checkpoint-Container/Prefixe (z. B. `state_dict`, `model`, `backbone`, `student`, `teacher`, `module`) und mappt kompatible DINOv2-Base-Keys automatisch.

---

## Outputs

Nach erfolgreichem Lauf:

- Modellartefakte: `/content/project/outputs/models/prototype_model.pkl`
- Metriken: `/content/project/outputs/metrics.json` und `metrics.csv`
- Visualisierungen: `/content/project/outputs/visualizations/*.png`

---

## Mögliche spätere Erweiterungen

- AUPRO/PRO-Metrik ergänzen.
- Mehr Kategorien neben `metal_nut` unterstützen.
- Feature-Sampling/Streaming weiter optimieren für sehr große Sets.
- Alternative Prototypenmodelle (z. B. GMM, kNN-Memory) als Vergleich.
