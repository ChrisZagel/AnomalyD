from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tarfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import gaussian_filter, label as cc_label
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from timm import create_model
from tqdm.auto import tqdm

from app.incremental_model import IncrementalADModel
from app.reporting import DebugReporter, ExperimentTableWriter, LeanMetricsLogger, RunContext


SUPPORTED_BACKBONE_MODELS: tuple[str, ...] = (
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_small_patch14_dinov2.lvd142m",
    "shvit_s4.in1k",
    "edgenext_small.usi_in1k",
)


@dataclass
class PoCConfig:
    seed: int = 42
    dataset_root: str = "/content/data/mvtec_ad"
    category: str = "metal_nut"
    archive_url: str = (
        "https://www.mydrive.ch/shares/38536/"
        "3830184030e49fe74747669442f0f283/download/420937637-1629959294/metal_nut.tar.xz"
    )
    archive_path: str = "/content/data/metal_nut.tar.xz"
    extract_root: str = "/content/data/mvtec_ad"
    project_root: str = "/content/project"
    backbone_model_name: str = "edgenext_small.usi_in1k"
    feature_size_factor: float = 0.75
    feature_layer_mode: str = "single_last_layer"
    enable_two_stage_inference: bool = False
    refine_feature_size_factor: float = 1.0
    inference_backend: str = "pytorch"
    run_comparison_suite: bool = False
    run_two_stage_experiment: bool = False
    run_score_ablation: bool = False
    benchmark_fast_modes: bool = False
    max_allowed_pixel_auroc_drop: float = 0.01
    aupro_num_steps: int = 80
    num_refine_rois: int = 3
    roi_crop_size: int = 192
    roi_expand_margin: int = 16
    stage_a_trigger_threshold: float = 2.5
    stage_a_trigger_mode: str = "image_or_patch"
    enable_refine_on_good_margin: float = 0.15
    image_score_mode: str = "hybrid_global_local"
    num_prototypes: int = 512
    transform_type: str = "running_whiten_fixed_proj"
    projection_type: str = "sparse_random_projection"
    projection_dim: int = 96
    projection_seed: int = 42
    whitening_eps: float = 1e-6
    use_whitening: bool = True
    distance_type: str = "l2"
    mahalanobis_alpha: float = 0.7
    mahalanobis_min_var: float = 1e-6
    mahalanobis_eps: float = 1e-12
    topk_percent: float = 1.0
    batch_size: int = 4
    num_workers: int = 2
    use_gaussian_smoothing: bool = True
    gaussian_sigma: float = 2.0
    max_pca_samples: int = 120_000
    max_replay_features: int = 200_000
    coverage_fraction: float = 0.5
    boundary_fraction: float = 0.25
    recent_fraction: float = 0.25
    update_trigger_images: int = 20
    candidate_promotion_hits: int = 5
    threshold_beta: float = 0.8
    prototype_forget_patience: int = 5000
    rollback_enabled: bool = True
    update_accept_quantile: float = 0.99
    max_patch_score_threshold: float = 3.5
    max_anomalous_area_fraction: float = 0.05
    candidate_distance_threshold: float = 3.0
    layers: tuple[int, int, int] = (5, 8, 11)
    enable_diagnostics: bool = True
    debug_mode: bool = False
    save_per_sample_report: bool = False
    save_plots: bool = False
    num_visualization_examples: int = 5


class MVTecMetalNutDataset:
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        if split == "train":
            self.samples = sorted((self.root / "train" / "good").glob("*.png"))
        elif split == "test":
            self.samples = []
            for defect_dir in sorted((self.root / "test").glob("*")):
                for img_path in sorted(defect_dir.glob("*.png")):
                    self.samples.append(img_path)
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        defect_type = img_path.parent.name if self.split == "test" else "good"
        item = {
            "path": str(img_path),
            "image": image,
            "orig_size": image.size[::-1],
            "defect_type": defect_type,
        }

        if self.split == "train":
            item["label"] = 0
            item["mask"] = np.zeros(image.size[::-1], dtype=np.uint8)
            return item

        item["label"] = 0 if defect_type == "good" else 1

        if defect_type == "good":
            item["mask"] = np.zeros(image.size[::-1], dtype=np.uint8)
        else:
            mask_path = (
                self.root
                / "ground_truth"
                / defect_type
                / f"{img_path.stem}_mask.png"
            )
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask file: {mask_path}")
            mask = Image.open(mask_path).convert("L")
            item["mask"] = (np.array(mask) > 0).astype(np.uint8)
        return item


class DINOv2BackboneWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, layers: tuple[int, int, int]) -> None:
        super().__init__()
        self.model = model
        self.layers = layers
        self._hook_outputs: dict[int, torch.Tensor] = {}
        self._handles = []
        for layer_idx in layers:
            handle = self.model.blocks[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _make_hook(self, idx: int):
        def hook(_module, _input, output):
            self._hook_outputs[idx] = output

        return hook

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._hook_outputs = {}
        _ = self.model.forward_features(x)
        token_h = x.shape[-2] // self.model.patch_embed.patch_size[0]
        token_w = x.shape[-1] // self.model.patch_embed.patch_size[1]
        expected_tokens = token_h * token_w
        maps = []
        for idx in self.layers:
            tokens = self._hook_outputs[idx]
            if tokens.dim() != 3:
                raise RuntimeError(f"Unexpected token shape at layer {idx}: {tokens.shape}")

            # Keep only spatial patch tokens.
            # DINOv2 variants may prepend CLS and optional register tokens.
            if tokens.shape[1] > expected_tokens:
                tokens = tokens[:, -expected_tokens:, :]

            if tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"Cannot reshape tokens from layer {idx}. Got {tokens.shape[1]} tokens,"
                    f" expected {expected_tokens}."
                )
            fmap = tokens.reshape(tokens.shape[0], token_h, token_w, tokens.shape[-1]).permute(0, 3, 1, 2)
            maps.append(fmap)
        return maps


class TimmFeaturesBackboneWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.model(x)
        if not isinstance(feats, (list, tuple)):
            raise RuntimeError("features_only model did not return a feature list.")
        if len(feats) < 3:
            raise RuntimeError(
                f"Backbone returned only {len(feats)} feature maps, need at least 3 for multi-level fusion."
            )
        return [f for f in feats[-3:]]


class FeatureExtractor:
    def __init__(self, backbone: torch.nn.Module, device: torch.device, cfg: PoCConfig) -> None:
        self.backbone = backbone.eval().to(device)
        self.device = device
        self.cfg = cfg
        self.inference_backend = cfg.inference_backend
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self._ov_compiled = None
        self._ov_input_name: str | None = None

    def _select_maps(self, maps: list[torch.Tensor], layer_mode: str) -> list[torch.Tensor]:
        if layer_mode == "single_last_layer":
            return [maps[-1]]
        if layer_mode == "fast_2layer" and len(maps) >= 2:
            sel = [max(0, len(maps) - 2), len(maps) - 1]
            return [maps[i] for i in sel]
        if layer_mode == "full_3layer" and len(maps) >= 3:
            return maps[-3:]
        return maps[-1:]

    def _ensure_openvino(self, input_shape: tuple[int, int, int, int]) -> None:
        if self._ov_compiled is not None:
            return
        if not isinstance(self.backbone, TimmFeaturesBackboneWrapper):
            raise RuntimeError("OpenVINO backend currently supports features_only backbones only.")
        try:
            import openvino as ov
        except Exception as exc:
            raise RuntimeError("OpenVINO backend requested but openvino is not installed.") from exc

        class Exportable(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor):
                out = self.model(x)
                return tuple(out)

        export_dir = Path(self.cfg.project_root) / "outputs" / "cache"
        export_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = export_dir / f"backbone_{self.cfg.backbone_model_name.replace('.', '_')}_{input_shape[2]}x{input_shape[3]}.onnx"

        if not onnx_path.exists():
            dummy = torch.randn(input_shape, device=self.device)
            torch.onnx.export(
                Exportable(self.backbone).to(self.device).eval(),
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=[f"feat_{i}" for i in range(3)],
                dynamic_axes=None,
                opset_version=18,
            )

        core = ov.Core()
        model = core.read_model(str(onnx_path))
        self._ov_compiled = core.compile_model(model, "CPU")
        self._ov_input_name = model.inputs[0].get_any_name()

    @torch.no_grad()
    def extract_patch_features(
        self,
        image: Image.Image,
        *,
        feature_size_factor: float | None = None,
        feature_layer_mode: str | None = None,
        return_timing: bool = False,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | tuple[np.ndarray, tuple[int, int], tuple[int, int], dict[str, float]]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        orig_h, orig_w = tensor.shape[-2:]
        fsf = self.cfg.feature_size_factor if feature_size_factor is None else feature_size_factor
        if fsf != 1.0:
            work_h = max(14, int(round(orig_h * fsf / 14) * 14))
            work_w = max(14, int(round(orig_w * fsf / 14) * 14))
            tensor = F.interpolate(tensor, size=(work_h, work_w), mode="bilinear", align_corners=False)
        else:
            work_h, work_w = orig_h, orig_w

        layer_mode = self.cfg.feature_layer_mode if feature_layer_mode is None else feature_layer_mode

        t0 = time.perf_counter()
        if self.inference_backend == "openvino":
            try:
                self._ensure_openvino(tuple(tensor.shape))
                assert self._ov_compiled is not None and self._ov_input_name is not None
                outputs = self._ov_compiled({self._ov_input_name: tensor.detach().cpu().numpy()})
                if isinstance(outputs, dict):
                    ordered_keys = sorted(outputs.keys(), key=lambda k: str(k))
                    maps = [torch.from_numpy(outputs[k]).to(self.device) for k in ordered_keys]
                else:
                    maps = [torch.from_numpy(o).to(self.device) for o in outputs]
            except Exception as exc:
                warnings.warn(
                    f"OpenVINO backend unavailable ({exc}). Falling back to pytorch backend for this run.",
                    RuntimeWarning,
                )
                self.inference_backend = "pytorch"
                maps = self.backbone(tensor)
        else:
            maps = self.backbone(tensor)
        t1 = time.perf_counter()

        maps = self._select_maps(list(maps), layer_mode)
        target_hw = maps[-1].shape[-2:]

        target_hw = maps[-1].shape[-2:]
        norm_maps = []
        for fmap in maps:
            if fmap.shape[-2:] != target_hw:
                fmap = F.interpolate(fmap, size=target_hw, mode="bilinear", align_corners=False)
            fmap = F.normalize(fmap, p=2, dim=1)
            norm_maps.append(fmap)

        fused = torch.cat(norm_maps, dim=1)
        b, c, h, w = fused.shape
        patch_feats = fused.permute(0, 2, 3, 1).reshape(b * h * w, c)
        if return_timing:
            timing = {
                "time_backbone_forward_eval": float(t1 - t0),
                "time_feature_fusion_eval": float(time.perf_counter() - t1),
            }
            return patch_feats.cpu().numpy(), (h, w), (work_h, work_w), timing
        return patch_feats.cpu().numpy(), (h, w), (work_h, work_w)


class PrototypeAnomalyModel:
    def __init__(self, cfg: PoCConfig) -> None:
        self.cfg = cfg
        self.inc_model: IncrementalADModel | None = None
        self.centers: np.ndarray | None = None
        self.mahalanobis_means: np.ndarray | None = None
        self.mahalanobis_vars: np.ndarray | None = None
        self.fit_timing: dict[str, float] = {}

    @staticmethod
    def _resolve_num_clusters(requested: int, n_samples: int) -> int:
        if n_samples <= 0:
            raise ValueError("No training samples available for clustering.")
        return max(1, min(requested, n_samples))

    def _chunked_min_distance(self, x: np.ndarray, centers: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
        mins = []
        if self.cfg.distance_type == "cosine":
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
            for i in range(0, x.shape[0], chunk_size):
                chunk = x[i : i + chunk_size]
                sims = chunk @ centers.T
                mins.append(1.0 - np.max(sims, axis=1))
        elif self.cfg.distance_type == "l2":
            for i in range(0, x.shape[0], chunk_size):
                chunk = x[i : i + chunk_size]
                d2 = np.sum(chunk**2, axis=1, keepdims=True) + np.sum(centers**2, axis=1)[None, :] - 2.0 * chunk @ centers.T
                mins.append(np.sqrt(np.clip(np.min(d2, axis=1), a_min=0.0, a_max=None)))
        elif self.cfg.distance_type == "mahalanobis_diag":
            means = self.mahalanobis_means if self.mahalanobis_means is not None else centers
            vars_ = self.mahalanobis_vars
            if vars_ is None:
                raise RuntimeError("Mahalanobis statistics are not available.")
            eps = float(self.cfg.mahalanobis_eps)
            for i in range(0, x.shape[0], chunk_size):
                chunk = x[i : i + chunk_size]
                diff = chunk[:, None, :] - means[None, :, :]
                d2 = np.sum((diff * diff) / (vars_[None, :, :] + eps), axis=2)
                mins.append(np.min(d2, axis=1))
        else:
            raise ValueError(f"Unsupported distance_type: {self.cfg.distance_type}")
        return np.concatenate(mins, axis=0)

    def _build_inc_config(self) -> dict[str, Any]:
        return {
            "whitening_eps": self.cfg.whitening_eps,
            "projection_dim": self.cfg.projection_dim,
            "projection_type": self.cfg.projection_type,
            "projection_seed": self.cfg.projection_seed,
            "num_prototypes": self.cfg.num_prototypes,
            "distance_type": self.cfg.distance_type,
            "mahalanobis_eps": self.cfg.mahalanobis_eps,
            "candidate_promotion_hits": self.cfg.candidate_promotion_hits,
            "prototype_forget_patience": self.cfg.prototype_forget_patience,
            "max_replay_features": self.cfg.max_replay_features,
            "coverage_fraction": self.cfg.coverage_fraction,
            "boundary_fraction": self.cfg.boundary_fraction,
            "recent_fraction": self.cfg.recent_fraction,
            "update_trigger_images": self.cfg.update_trigger_images,
            "threshold_beta": self.cfg.threshold_beta,
            "update_accept_quantile": self.cfg.update_accept_quantile,
            "max_patch_score_threshold": self.cfg.max_patch_score_threshold,
            "max_anomalous_area_fraction": self.cfg.max_anomalous_area_fraction,
            "candidate_distance_threshold": self.cfg.candidate_distance_threshold,
        }

    def fit(self, dataset: MVTecMetalNutDataset, extractor: FeatureExtractor) -> None:
        train_feats = []
        t_backbone = 0.0
        for sample in tqdm(dataset.samples, desc="Collect features for incremental model"):
            image = Image.open(sample).convert("RGB")
            t0 = time.perf_counter()
            feats, _, _ = extractor.extract_patch_features(image)
            t_backbone += float(time.perf_counter() - t0)
            train_feats.append(feats.astype(np.float32))

        all_feats = np.concatenate(train_feats, axis=0)
        if all_feats.shape[0] > self.cfg.max_pca_samples:
            idx = np.random.choice(all_feats.shape[0], self.cfg.max_pca_samples, replace=False)
            all_feats = all_feats[idx]

        self.inc_model = IncrementalADModel(self._build_inc_config())
        self.inc_model.fit_initial(all_feats)
        self.centers = self.inc_model.prototypes.means
        self.fit_timing = {"time_backbone_forward_train": t_backbone, **self.inc_model.last_fit_timing}

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.inc_model is None:
            raise RuntimeError("Model is not fitted yet.")
        payload = {
            "config": asdict(self.cfg),
            "incremental_state": self.inc_model.state_dict(),
            "centers": self.inc_model.prototypes.means,
        }
        import pickle

        with open(output_dir / "prototype_model.pkl", "wb") as f:
            pickle.dump(payload, f)

    def _compute_image_score(self, anomaly_map: np.ndarray, coarse_score: float, refined_max: float) -> float:
        mode = self.cfg.image_score_mode
        flat = anomaly_map.ravel()
        topk = max(1, int(flat.size * self.cfg.topk_percent / 100.0))
        top_mean = float(np.mean(np.partition(flat, -topk)[-topk:]))
        if mode == "max_patch":
            return float(np.max(flat))
        if mode == "topk_patch_mean":
            return top_mean
        if mode == "top1_percent_mean":
            k = max(1, int(flat.size * 0.01))
            return float(np.mean(np.partition(flat, -k)[-k:]))
        # hybrid_global_local
        return float(0.6 * coarse_score + 0.4 * max(refined_max, top_mean))

    @staticmethod
    def _propose_rois_from_map(
        anomaly_map: np.ndarray,
        num_rois: int,
        crop_size: int,
        margin: int,
    ) -> list[tuple[int, int, int, int]]:
        h, w = anomaly_map.shape
        flat_idx = np.argpartition(anomaly_map.ravel(), -min(num_rois * 8, anomaly_map.size))[-min(num_rois * 8, anomaly_map.size):]
        ys, xs = np.unravel_index(flat_idx, anomaly_map.shape)
        scores = anomaly_map[ys, xs]
        order = np.argsort(scores)[::-1]
        rois: list[tuple[int, int, int, int]] = []
        used_centers: list[tuple[int, int]] = []
        suppress_dist = max(8, crop_size // 3)
        for oi in order:
            cy, cx = int(ys[oi]), int(xs[oi])
            if any((cy - uy) ** 2 + (cx - ux) ** 2 < suppress_dist**2 for uy, ux in used_centers):
                continue
            y1 = max(0, cy - crop_size // 2 - margin)
            y2 = min(h, cy + crop_size // 2 + margin)
            x1 = max(0, cx - crop_size // 2 - margin)
            x2 = min(w, cx + crop_size // 2 + margin)
            if y2 - y1 < 8 or x2 - x1 < 8:
                continue
            rois.append((y1, x1, y2, x2))
            used_centers.append((cy, cx))
            if len(rois) >= num_rois:
                break
        return rois

    def infer_map(
        self,
        image: Image.Image,
        extractor: FeatureExtractor,
        orig_size: tuple[int, int],
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        if self.inc_model is None:
            raise RuntimeError("Model is not fitted yet.")

        t0 = time.perf_counter()
        feats, grid_hw, work_hw, ext_timing = extractor.extract_patch_features(
            image,
            feature_size_factor=self.cfg.feature_size_factor,
            feature_layer_mode=self.cfg.feature_layer_mode,
            return_timing=True,
        )
        t1 = time.perf_counter()
        patch_scores = self.inc_model.predict(feats.astype(np.float32))
        t2 = time.perf_counter()

        token_map = patch_scores.reshape(grid_hw)
        token_map_t = torch.from_numpy(token_map).unsqueeze(0).unsqueeze(0).float()
        work_map = F.interpolate(token_map_t, size=work_hw, mode="bilinear", align_corners=False)
        coarse_map = F.interpolate(work_map, size=orig_size, mode="bilinear", align_corners=False)[0, 0].numpy()

        if self.cfg.use_gaussian_smoothing:
            coarse_map = gaussian_filter(coarse_map, sigma=self.cfg.gaussian_sigma)

        topk = max(1, int(coarse_map.size * self.cfg.topk_percent / 100.0))
        stage_a_image_score = float(np.mean(np.partition(coarse_map.ravel(), -topk)[-topk:]))
        t3 = time.perf_counter()

        trigger_patch = float(np.max(coarse_map)) > self.cfg.stage_a_trigger_threshold
        trigger_image = stage_a_image_score > self.cfg.stage_a_trigger_threshold
        if self.cfg.stage_a_trigger_mode == "image_or_patch":
            stage_a_triggered = trigger_patch or trigger_image
        else:
            stage_a_triggered = trigger_image
        stage_a_triggered = stage_a_triggered or (
            stage_a_image_score > (self.cfg.stage_a_trigger_threshold - self.cfg.enable_refine_on_good_margin)
        )

        final_map = coarse_map.copy()
        refined_max = float(np.max(coarse_map))
        num_rois = 0
        t_roi0 = time.perf_counter()
        rois = self._propose_rois_from_map(
            coarse_map,
            num_rois=self.cfg.num_refine_rois,
            crop_size=self.cfg.roi_crop_size,
            margin=self.cfg.roi_expand_margin,
        ) if (self.cfg.enable_two_stage_inference and stage_a_triggered) else []
        t_roi1 = time.perf_counter()

        stage_b_backbone = 0.0
        stage_b_distance = 0.0
        t_merge = 0.0
        if rois:
            arr = np.array(image)
            for (y1, x1, y2, x2) in rois:
                crop = Image.fromarray(arr[y1:y2, x1:x2])
                tb0 = time.perf_counter()
                c_feats, c_grid, c_work, c_timing = extractor.extract_patch_features(
                    crop,
                    feature_size_factor=self.cfg.refine_feature_size_factor,
                    feature_layer_mode=("full_3layer" if self.cfg.feature_layer_mode == "full_3layer" else "fast_2layer"),
                    return_timing=True,
                )
                tb1 = time.perf_counter()
                c_scores = self.inc_model.predict(c_feats.astype(np.float32))
                tb2 = time.perf_counter()
                c_token = c_scores.reshape(c_grid)
                c_t = torch.from_numpy(c_token).unsqueeze(0).unsqueeze(0).float()
                c_work_map = F.interpolate(c_t, size=c_work, mode="bilinear", align_corners=False)
                c_map = F.interpolate(c_work_map, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)[0, 0].numpy()
                tm0 = time.perf_counter()
                final_map[y1:y2, x1:x2] = np.maximum(final_map[y1:y2, x1:x2], c_map)
                tm1 = time.perf_counter()
                refined_max = max(refined_max, float(np.max(c_map)))
                stage_b_backbone += float(tb1 - tb0)
                stage_b_distance += float(tb2 - tb1)
                t_merge += float(tm1 - tm0)
            num_rois = len(rois)

        final_image_score = self._compute_image_score(final_map, stage_a_image_score, refined_max)
        if self.cfg.use_gaussian_smoothing:
            final_map = gaussian_filter(final_map, sigma=self.cfg.gaussian_sigma)

        timing = {
            "stage_a_backbone_time": float(ext_timing.get("time_backbone_forward_eval", t1 - t0)),
            "stage_b_backbone_time": float(stage_b_backbone),
            "roi_proposal_time": float(t_roi1 - t_roi0),
            "merge_time": float(t_merge),
            "time_backbone_forward_eval": float(ext_timing.get("time_backbone_forward_eval", t1 - t0) + stage_b_backbone),
            "time_feature_fusion_eval": float(ext_timing.get("time_feature_fusion_eval", 0.0)),
            "time_transform_eval": 0.0,
            "time_distance_eval": float((t2 - t1) + stage_b_distance),
            "time_postprocess_eval": float((t3 - t2) + t_merge),
            "time_total_infer": float(time.perf_counter() - t0),
        }
        meta = {
            "stage_a_triggered": int(stage_a_triggered),
            "num_refine_rois": int(num_rois),
            "stage_a_image_score": float(stage_a_image_score),
            "stage_b_refined_max_score": float(refined_max),
            "final_image_score": float(final_image_score),
            **timing,
        }
        return final_map, float(final_image_score), meta

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def _get_pca_seen_samples(pca_obj: Any) -> int:
    n_seen = getattr(pca_obj, "n_samples_seen_", None)
    if n_seen is not None:
        return int(n_seen)
    n_samples = getattr(pca_obj, "n_samples_", None)
    if n_samples is not None:
        return int(n_samples)
    return 0

def setup_project_dirs(cfg: PoCConfig) -> dict[str, Path]:
    project_root = Path(cfg.project_root)
    paths = {
        "project": project_root,
        "src": project_root / "src",
        "checkpoints": project_root / "checkpoints",
        "outputs": project_root / "outputs",
        "cache": project_root / "cache",
        "visualizations": project_root / "outputs" / "visualizations",
        "models": project_root / "outputs" / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _normalize_dataset_layout(extract_root: Path, category: str) -> Path:
    category_dir = extract_root / category
    if category_dir.exists():
        return category_dir

    nested = list(extract_root.glob(f"**/{category}"))
    if nested:
        nested_dir = nested[0]
        category_dir.parent.mkdir(parents=True, exist_ok=True)
        if nested_dir.resolve() != category_dir.resolve():
            if category_dir.exists():
                shutil.rmtree(category_dir)
            shutil.move(str(nested_dir), str(category_dir))
        return category_dir

    raise FileNotFoundError(f"Could not locate extracted category directory for {category} under {extract_root}")


def ensure_dataset(cfg: PoCConfig) -> Path:
    extract_root = Path(cfg.extract_root)
    archive_path = Path(cfg.archive_path)
    target_root = extract_root / cfg.category

    expected = target_root / "train" / "good"
    if expected.exists() and any(expected.glob("*.png")):
        return target_root

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading dataset archive to {archive_path} ...")
        urlretrieve(cfg.archive_url, archive_path)

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, mode="r:xz") as tar:
        for member in tar.getmembers():
            member_path = extract_root / member.name
            if not member_path.resolve().is_relative_to(extract_root.resolve()):
                raise RuntimeError(f"Unsafe archive entry blocked: {member.name}")
        tar.extractall(path=extract_root)

    target_root = _normalize_dataset_layout(extract_root, cfg.category)

    required = [
        target_root / "train" / "good",
        target_root / "test",
        target_root / "ground_truth",
    ]
    for req in required:
        if not req.exists():
            raise FileNotFoundError(f"Dataset structure invalid. Missing: {req}")
    return target_root


def _create_pretrained_model(model_name: str, *, features_only: bool = False) -> torch.nn.Module:
    kwargs: dict[str, Any] = {"pretrained": True}
    if features_only:
        kwargs["features_only"] = True

    # dynamic_img_size is only needed for ViT-style patch models.
    if model_name.startswith("vit_"):
        try:
            return create_model(model_name, dynamic_img_size=True, **kwargs)
        except TypeError:
            return create_model(model_name, **kwargs)

    return create_model(model_name, **kwargs)


def _validate_supported_model_name(model_name: str) -> None:
    if model_name not in SUPPORTED_BACKBONE_MODELS:
        supported = ", ".join(SUPPORTED_BACKBONE_MODELS)
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")


def load_backbone(cfg: PoCConfig, device: torch.device) -> torch.nn.Module:
    _validate_supported_model_name(cfg.backbone_model_name)
    print(f"Loading pretrained backbone: {cfg.backbone_model_name}")

    if cfg.backbone_model_name.startswith("vit_"):
        model = _create_pretrained_model(cfg.backbone_model_name, features_only=False)
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "strict_img_size"):
            model.patch_embed.strict_img_size = False
        wrapped: torch.nn.Module = DINOv2BackboneWrapper(model, cfg.layers)
    else:
        model = _create_pretrained_model(cfg.backbone_model_name, features_only=True)
        wrapped = TimmFeaturesBackboneWrapper(model)

    wrapped.eval().to(device)
    for p in wrapped.parameters():
        p.requires_grad = False
    return wrapped


def evaluate(
    test_ds: MVTecMetalNutDataset,
    extractor: FeatureExtractor,
    proto_model: PrototypeAnomalyModel,
    vis_dir: Path,
    *,
    save_visualizations: bool,
    num_visualization_examples: int,
    debug_mode: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, float]]:
    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    per_sample_rows: list[dict[str, Any]] = []
    per_sample_eval: list[dict[str, Any]] = []

    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    saved_vis = 0
    t_backbone = 0.0
    t_transform = 0.0
    t_distance = 0.0
    t_post = 0.0
    t_fusion = 0.0
    t_total = 0.0
    t_stage_a_backbone = 0.0
    t_stage_b_backbone = 0.0
    t_roi = 0.0
    t_merge = 0.0
    triggered = 0
    refine_rois_total = 0

    for idx in tqdm(range(len(test_ds)), desc="Evaluate test set"):
        sample = test_ds[idx]
        t0 = time.perf_counter()
        anom_map, image_score, meta = proto_model.infer_map(sample["image"], extractor, sample["orig_size"])
        infer_sec = time.perf_counter() - t0

        t_backbone += float(meta.get("time_backbone_forward_eval", 0.0))
        t_transform += float(meta.get("time_transform_eval", 0.0))
        t_distance += float(meta.get("time_distance_eval", 0.0))
        t_post += float(meta.get("time_postprocess_eval", 0.0))
        t_fusion += float(meta.get("time_feature_fusion_eval", 0.0))
        t_total += float(meta.get("time_total_infer", infer_sec))
        t_stage_a_backbone += float(meta.get("stage_a_backbone_time", 0.0))
        t_stage_b_backbone += float(meta.get("stage_b_backbone_time", 0.0))
        t_roi += float(meta.get("roi_proposal_time", 0.0))
        t_merge += float(meta.get("merge_time", 0.0))
        triggered += int(meta.get("stage_a_triggered", 0))
        refine_rois_total += int(meta.get("num_refine_rois", 0))

        image_labels.append(sample["label"])
        image_scores.append(image_score)
        pixel_labels.append(sample["mask"].astype(np.uint8).ravel())
        pixel_scores.append(anom_map.ravel())

        row = {
            "index": idx,
            "path": sample["path"],
            "defect_type": sample["defect_type"],
            "label": sample["label"],
            "image_score": float(image_score),
            "infer_time_sec": float(infer_sec),
            "map_mean": float(anom_map.mean()),
            "map_max": float(anom_map.max()),
            "stage_a_triggered": int(meta.get("stage_a_triggered", 0)),
            "num_refine_rois": int(meta.get("num_refine_rois", 0)),
            "stage_a_image_score": float(meta.get("stage_a_image_score", image_score)),
            "stage_b_refined_max_score": float(meta.get("stage_b_refined_max_score", np.max(anom_map))),
            "final_image_score": float(meta.get("final_image_score", image_score)),
        }
        per_sample_rows.append(row)

        per_sample_eval.append(
            {
                "index": idx,
                "defect_type": sample["defect_type"],
                "label": sample["label"],
                "image_score": float(image_score),
                "mask": sample["mask"].astype(np.uint8),
                "anom_map": anom_map.astype(np.float32),
            }
        )

        if save_visualizations and saved_vis < num_visualization_examples:
            save_visualization(sample, anom_map, vis_dir / f"sample_{idx:03d}.png")
            saved_vis += 1

    image_auc = float(roc_auc_score(image_labels, image_scores))
    pixel_auc = float(roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores)))
    metrics = {
        "image_auroc": image_auc,
        "pixel_auroc": pixel_auc,
        "num_test_images": len(test_ds),
        "mean_infer_time_sec": float(np.mean([r["infer_time_sec"] for r in per_sample_rows])),
        "std_infer_time_sec": float(np.std([r["infer_time_sec"] for r in per_sample_rows])),
        "stage_a_trigger_rate": float(triggered / max(len(test_ds), 1)),
        "mean_num_refine_rois_used": float(refine_rois_total / max(len(test_ds), 1)),
        "mean_stage_b_time": float(t_stage_b_backbone / max(len(test_ds), 1)),
    }
    timing = {
        "time_backbone_forward_eval": t_backbone,
        "time_feature_fusion_eval": t_fusion,
        "time_transform_eval": t_transform,
        "time_distance_eval": t_distance,
        "time_postprocess_eval": t_post,
        "time_total_infer": t_total,
        "stage_a_backbone_time": t_stage_a_backbone,
        "stage_b_backbone_time": t_stage_b_backbone,
        "roi_proposal_time": t_roi,
        "merge_time": t_merge,
    }
    return metrics, per_sample_rows, per_sample_eval, np.array(image_scores, dtype=np.float32), np.concatenate(pixel_scores), timing

def compute_defect_level_aurocs(rows: list[dict[str, Any]]) -> dict[str, float]:
    good_scores = [r["image_score"] for r in rows if r["defect_type"] == "good"]
    out: dict[str, float] = {}
    defect_types = sorted({r["defect_type"] for r in rows if r["defect_type"] != "good"})
    for defect in defect_types:
        defect_scores = [r["image_score"] for r in rows if r["defect_type"] == defect]
        y_true = np.array([0] * len(good_scores) + [1] * len(defect_scores))
        y_score = np.array(good_scores + defect_scores)
        out[f"image_auroc_good_vs_{defect}"] = float(roc_auc_score(y_true, y_score))
    return out


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return float(np.median(y_score))
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def _compute_aupro_for_subset(records: list[dict[str, Any]], max_fpr: float = 0.3, num_steps: int = 200) -> float:
    pos_records = [r for r in records if r["label"] == 1]
    if not pos_records:
        return float("nan")

    all_scores = np.concatenate([r["anom_map"].ravel() for r in records])
    if all_scores.size == 0:
        return float("nan")

    thresholds = np.linspace(float(all_scores.max()), float(all_scores.min()), num_steps)
    fprs: list[float] = []
    pros: list[float] = []

    neg_pixels = sum(int((r["mask"] == 0).sum()) for r in records)
    if neg_pixels == 0:
        return float("nan")

    for thr in thresholds:
        fp = 0
        region_overlaps: list[float] = []
        for r in records:
            pred = (r["anom_map"] >= thr).astype(np.uint8)
            gt = r["mask"]
            fp += int(((pred == 1) & (gt == 0)).sum())

            if r["label"] == 1:
                labeled, n_comp = cc_label(gt)
                for cid in range(1, n_comp + 1):
                    region = labeled == cid
                    denom = int(region.sum())
                    if denom > 0:
                        region_overlaps.append(float(pred[region].sum()) / float(denom))

        fpr = float(fp) / float(neg_pixels)
        pro = float(np.mean(region_overlaps)) if region_overlaps else 0.0
        fprs.append(fpr)
        pros.append(pro)

    fprs_arr = np.array(fprs)
    pros_arr = np.array(pros)
    order = np.argsort(fprs_arr)
    fprs_arr = fprs_arr[order]
    pros_arr = pros_arr[order]

    mask = fprs_arr <= max_fpr
    if not np.any(mask):
        return 0.0

    fprs_clip = np.concatenate([[0.0], fprs_arr[mask], [max_fpr]])
    pros_interp = np.interp(fprs_clip, fprs_arr, pros_arr)
    area = np.trapezoid(pros_interp, fprs_clip)
    return float(area / max_fpr)


def compute_per_defect_metrics(records: list[dict[str, Any]], aupro_num_steps: int = 80) -> list[dict[str, Any]]:
    defect_types = sorted({r["defect_type"] for r in records if r["defect_type"] != "good"})
    good_records = [r for r in records if r["defect_type"] == "good"]
    out: list[dict[str, Any]] = []

    for defect in defect_types:
        defect_records = [r for r in records if r["defect_type"] == defect]
        subset = good_records + defect_records

        # Image-level metrics
        y_true_img = np.array([0] * len(good_records) + [1] * len(defect_records), dtype=np.uint8)
        y_score_img = np.array([r["image_score"] for r in subset], dtype=np.float32)
        img_auroc = float(roc_auc_score(y_true_img, y_score_img))
        img_thr = _best_f1_threshold(y_true_img, y_score_img)
        y_pred_img = (y_score_img >= img_thr).astype(np.uint8)
        img_prec, img_rec, img_f1, _ = precision_recall_fscore_support(
            y_true_img, y_pred_img, average="binary", zero_division=0
        )

        # Pixel-level metrics
        y_true_pix = np.concatenate([r["mask"].ravel() for r in subset]).astype(np.uint8)
        y_score_pix = np.concatenate([r["anom_map"].ravel() for r in subset]).astype(np.float32)
        pix_auroc = float(roc_auc_score(y_true_pix, y_score_pix))
        pix_thr = _best_f1_threshold(y_true_pix, y_score_pix)
        y_pred_pix = (y_score_pix >= pix_thr).astype(np.uint8)
        pix_prec, pix_rec, pix_f1, _ = precision_recall_fscore_support(
            y_true_pix, y_pred_pix, average="binary", zero_division=0
        )

        aupro = _compute_aupro_for_subset(subset, num_steps=aupro_num_steps)

        out.append(
            {
                "defect_type": defect,
                "num_good_images": len(good_records),
                "num_defect_images": len(defect_records),
                "image_auroc": img_auroc,
                "pixel_auroc": pix_auroc,
                "aupro": aupro,
                "image_precision": float(img_prec),
                "image_recall": float(img_rec),
                "image_f1": float(img_f1),
                "pixel_precision": float(pix_prec),
                "pixel_recall": float(pix_rec),
                "pixel_f1": float(pix_f1),
            }
        )
    return out


def save_per_defect_metrics(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out_dir / "per_defect_metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_overlay_gallery(
    rows: list[dict[str, Any]],
    test_ds: MVTecMetalNutDataset,
    extractor: FeatureExtractor,
    proto_model: PrototypeAnomalyModel,
    out_path: Path,
    num_examples: int,
) -> None:
    if not rows or num_examples <= 0:
        return
    top_rows = sorted(rows, key=lambda r: r["image_score"], reverse=True)[:num_examples]
    if len(top_rows) == 0:
        return
    fig, axes = plt.subplots(1, len(top_rows), figsize=(4 * len(top_rows), 4))
    if len(top_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, top_rows):
        sample = test_ds[int(row["index"])]
        anom_map, _, _ = proto_model.infer_map(sample["image"], extractor, sample["orig_size"])
        image_np = np.array(sample["image"]) / 255.0
        norm = (anom_map - anom_map.min()) / (anom_map.max() - anom_map.min() + 1e-8)
        heat = plt.cm.jet(norm)[..., :3]
        overlay = 0.6 * image_np + 0.4 * heat
        ax.imshow(overlay)
        ax.set_title(f"{sample['defect_type']}\nscore={row['image_score']:.3f}")
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_visualization(sample: dict[str, Any], anomaly_map: np.ndarray, out_path: Path) -> None:
    image_np = np.array(sample["image"])
    mask = sample["mask"]

    norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    cmap = plt.cm.jet(norm)[..., :3]
    overlay = (0.65 * (image_np / 255.0) + 0.35 * cmap)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("GT Mask")
    axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title("Anomaly Map")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metrics(metrics: dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def run_poc(cfg: PoCConfig) -> dict[str, float]:
    t_all = time.perf_counter()
    set_seed(cfg.seed)

    ctx = RunContext.start_run(cfg.project_root)
    lean_logger = LeanMetricsLogger(ctx, enable_diagnostics=cfg.enable_diagnostics)
    debug_reporter = DebugReporter(
        ctx,
        enable_diagnostics=cfg.enable_diagnostics,
        debug_mode=cfg.debug_mode,
        save_per_sample_report=cfg.save_per_sample_report,
        save_plots=cfg.save_plots,
    )
    experiment_writer = ExperimentTableWriter(cfg.project_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = ensure_dataset(cfg)
    train_ds = MVTecMetalNutDataset(data_root, split="train")
    test_ds = MVTecMetalNutDataset(data_root, split="test")

    if len(train_ds) == 0:
        raise RuntimeError("No training images found in train/good")
    if len(test_ds) == 0:
        raise RuntimeError("No test images found in test/*")

    backbone = load_backbone(cfg, device)
    extractor = FeatureExtractor(backbone, device, cfg)

    proto_model = PrototypeAnomalyModel(cfg)
    t_train = time.perf_counter()
    original_backend = extractor.inference_backend
    if original_backend == "openvino":
        extractor.inference_backend = "pytorch"
    proto_model.fit(train_ds, extractor)
    extractor.inference_backend = original_backend
    train_time = time.perf_counter() - t_train
    train_backbone = float(proto_model.fit_timing.get("time_backbone_forward_train", 0.0))
    train_transform_t = float(proto_model.fit_timing.get("time_transform_update_train", 0.0))
    train_proto_t = float(proto_model.fit_timing.get("time_prototype_update_train", 0.0))

    train_backbone = float(proto_model.fit_timing.get("time_backbone_forward_train", 0.0))
    train_transform_t = float(proto_model.fit_timing.get("time_transform_update_train", 0.0))
    train_proto_t = float(proto_model.fit_timing.get("time_prototype_update_train", 0.0))

    train_backbone = float(proto_model.fit_timing.get("time_backbone_forward_train", 0.0))
    train_transform_t = float(proto_model.fit_timing.get("time_transform_update_train", 0.0))
    train_proto_t = float(proto_model.fit_timing.get("time_prototype_update_train", 0.0))

    train_backbone = float(proto_model.fit_timing.get("time_backbone_forward_train", 0.0))
    train_transform_t = float(proto_model.fit_timing.get("time_transform_update_train", 0.0))
    train_proto_t = float(proto_model.fit_timing.get("time_prototype_update_train", 0.0))

    proto_model.save(ctx.metrics_dir)

    num_eval_passes_executed = 0
    t_eval = time.perf_counter()
    metrics, per_sample_rows, per_sample_eval, image_scores, pixel_scores, eval_timing = evaluate(
        test_ds,
        extractor,
        proto_model,
        ctx.visualizations_dir,
        save_visualizations=(cfg.enable_diagnostics and cfg.num_visualization_examples > 0),
        num_visualization_examples=cfg.num_visualization_examples,
        debug_mode=cfg.debug_mode,
    )
    num_eval_passes_executed += 1
    eval_time = time.perf_counter() - t_eval
    total_time = time.perf_counter() - t_all

    metrics.update(
        {
            "train_stage_sec": float(train_time),
            "eval_stage_sec": float(eval_time),
            "total_runtime_sec": float(total_time),
            "train_images": len(train_ds),
            "test_images": len(test_ds),
            "num_eval_passes_executed": num_eval_passes_executed,
        }
    )

    if proto_model.centers is not None:
        metrics["effective_num_prototypes"] = int(proto_model.centers.shape[0])

    per_defect_metrics = compute_per_defect_metrics(per_sample_eval, aupro_num_steps=cfg.aupro_num_steps)
    aupro_values = [float(r["aupro"]) for r in per_defect_metrics if not np.isnan(float(r["aupro"]))]
    aupro_mean = float(np.mean(aupro_values)) if aupro_values else float("nan")
    image_f1_values = [float(r["image_f1"]) for r in per_defect_metrics]
    pixel_f1_values = [float(r["pixel_f1"]) for r in per_defect_metrics]

    # optional fast mode benchmark only when requested
    if cfg.enable_diagnostics and cfg.benchmark_fast_modes:
        compare_rows: list[dict[str, Any]] = []
        base_mode = cfg.feature_layer_mode
        base_pixel = float(metrics["pixel_auroc"])
        for mode in ["single_last_layer", "fast_2layer", "full_3layer"]:
            cfg.feature_layer_mode = mode
            c_metrics, _, c_eval, _, _, _ = evaluate(
                test_ds,
                extractor,
                proto_model,
                ctx.visualizations_dir,
                save_visualizations=False,
                num_visualization_examples=0,
                debug_mode=False,
            )
            num_eval_passes_executed += 1
            c_per_def = compute_per_defect_metrics(c_eval, aupro_num_steps=cfg.aupro_num_steps)
            c_aupro = [float(r["aupro"]) for r in c_per_def if not np.isnan(float(r["aupro"]))]
            c_px_f1 = float(np.mean([float(r["pixel_f1"]) for r in c_per_def])) if c_per_def else float("nan")
            compare_rows.append({
                "feature_layer_mode": mode,
                "image_auroc": float(c_metrics["image_auroc"]),
                "pixel_auroc": float(c_metrics["pixel_auroc"]),
                "aupro": float(np.mean(c_aupro)) if c_aupro else float("nan"),
                "pixel_f1": c_px_f1,
                "mean_infer_time_sec": float(c_metrics["mean_infer_time_sec"]),
            })
        cfg.feature_layer_mode = base_mode
        cmp_path = ctx.tables_dir / "feature_layer_mode_comparison.csv"
        with open(cmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
            writer.writeheader()
            writer.writerows(compare_rows)
        best_fast = max(compare_rows[:2], key=lambda r: r["mean_infer_time_sec"] * -1)
        drop = base_pixel - float(best_fast["pixel_auroc"])
        if drop > cfg.max_allowed_pixel_auroc_drop:
            print(
                f"[WARN] Fast-mode benchmark pixel_auroc drop {drop:.4f} exceeds max_allowed_pixel_auroc_drop={cfg.max_allowed_pixel_auroc_drop:.4f}"
            )

    metrics["num_eval_passes_executed"] = num_eval_passes_executed

    if not any([cfg.run_comparison_suite, cfg.run_two_stage_experiment, cfg.run_score_ablation, cfg.benchmark_fast_modes]):
        assert num_eval_passes_executed == 1, "Normal mode must execute exactly one evaluation pass."

    timing_summary = {
        "time_feature_extraction_train": float(train_backbone),
        "time_transform_fit_or_update": float(train_transform_t),
        "time_prototype_fit_or_update": float(train_proto_t),
        "time_backbone_forward_train": float(train_backbone),
        "time_transform_update_train": float(train_transform_t),
        "time_prototype_update_train": float(train_proto_t),
        "time_eval_total": float(eval_time),
        "time_backbone_forward_eval": float(eval_timing.get("time_backbone_forward_eval", 0.0)),
        "time_feature_fusion_eval": float(eval_timing.get("time_feature_fusion_eval", 0.0)),
        "time_transform_eval": float(eval_timing.get("time_transform_eval", 0.0)),
        "time_distance_eval": float(eval_timing.get("time_distance_eval", 0.0)),
        "time_postprocess_eval": float(eval_timing.get("time_postprocess_eval", 0.0)),
        "time_total_infer": float(eval_timing.get("time_total_infer", 0.0)),
        "stage_a_backbone_time": float(eval_timing.get("stage_a_backbone_time", 0.0)),
        "stage_b_backbone_time": float(eval_timing.get("stage_b_backbone_time", 0.0)),
        "roi_proposal_time": float(eval_timing.get("roi_proposal_time", 0.0)),
        "merge_time": float(eval_timing.get("merge_time", 0.0)),
        "mean_infer_time_sec": float(metrics["mean_infer_time_sec"]),
        "num_eval_passes_executed": int(num_eval_passes_executed),
        "inference_backend": cfg.inference_backend,
    }

    summary = {
        "run_id": ctx.run_id,
        "timestamp": ctx.timestamp,
        "backbone_name": cfg.backbone_model_name,
        "transform_type": cfg.transform_type,
        "projection_dim": cfg.projection_dim,
        "feature_size_factor": cfg.feature_size_factor,
        "num_prototypes": cfg.num_prototypes,
        "distance_type": cfg.distance_type,
        "device_type": str(device),
        "num_train_images": len(train_ds),
        "num_test_images": len(test_ds),
        "image_auroc": float(metrics["image_auroc"]),
        "pixel_auroc": float(metrics["pixel_auroc"]),
        "aupro_mean": aupro_mean,
        "image_f1": float(np.mean(image_f1_values)) if image_f1_values else float("nan"),
        "pixel_f1": float(np.mean(pixel_f1_values)) if pixel_f1_values else float("nan"),
        "mean_infer_time_sec": float(metrics["mean_infer_time_sec"]),
        "total_runtime_sec": float(total_time),
        "effective_num_prototypes": int(metrics.get("effective_num_prototypes", 0)),
        "feature_layer_mode": cfg.feature_layer_mode,
        "inference_backend": cfg.inference_backend,
        "num_eval_passes_executed": int(num_eval_passes_executed),
    }

    lean_logger.log_summary_metrics(summary, summary)
    lean_logger.log_timing_summary(timing_summary)
    lean_logger.log_per_defect_metrics(per_defect_metrics)

    experiment_writer.append_row(
        {
            "run_id": ctx.run_id,
            "timestamp": ctx.timestamp,
            "backbone_name": cfg.backbone_model_name,
            "transform_type": cfg.transform_type,
            "projection_dim": cfg.projection_dim,
            "feature_size_factor": cfg.feature_size_factor,
            "num_prototypes": cfg.num_prototypes,
            "distance_type": cfg.distance_type,
            "image_auroc": float(metrics["image_auroc"]),
            "pixel_auroc": float(metrics["pixel_auroc"]),
            "aupro_mean": aupro_mean,
            "mean_infer_time_sec": float(metrics["mean_infer_time_sec"]),
            "total_runtime_sec": float(total_time),
            "device_type": str(device),
        }
    )

    if cfg.enable_diagnostics and cfg.debug_mode and cfg.num_visualization_examples > 0:
        gallery_path = ctx.visualizations_dir / "final_top5_overlay_gallery.png"
        save_overlay_gallery(
            per_sample_rows,
            test_ds,
            extractor,
            proto_model,
            gallery_path,
            num_examples=cfg.num_visualization_examples,
        )

    if cfg.enable_diagnostics and cfg.debug_mode:
        debug_reporter.log_per_sample_table(per_sample_rows)

    print("Final metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Run outputs: {ctx.root_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVTec AD metal_nut PoC (pretrained timm backbones)")
    parser.add_argument("--feature-size-factor", type=float, default=0.75)
    parser.add_argument("--feature-layer-mode", type=str, default="single_last_layer", choices=["single_last_layer", "fast_2layer", "full_3layer"])
    parser.add_argument("--enable-two-stage-inference", dest="enable_two_stage_inference", action="store_true", default=False)
    parser.add_argument("--disable-two-stage-inference", dest="enable_two_stage_inference", action="store_false")
    parser.add_argument("--refine-feature-size-factor", type=float, default=1.0)
    parser.add_argument("--num-refine-rois", type=int, default=3)
    parser.add_argument("--roi-crop-size", type=int, default=192)
    parser.add_argument("--roi-expand-margin", type=int, default=16)
    parser.add_argument("--stage-a-trigger-threshold", type=float, default=2.5)
    parser.add_argument("--image-score-mode", type=str, default="hybrid_global_local", choices=["top1_percent_mean", "topk_patch_mean", "max_patch", "hybrid_global_local"])
    parser.add_argument("--inference-backend", type=str, default="pytorch", choices=["pytorch", "openvino"])
    parser.add_argument("--run-comparison-suite", action="store_true", default=False)
    parser.add_argument("--run-two-stage-experiment", action="store_true", default=False)
    parser.add_argument("--run-score-ablation", action="store_true", default=False)
    parser.add_argument("--benchmark-fast-modes", action="store_true", default=False)
    parser.add_argument("--max-allowed-pixel-auroc-drop", type=float, default=0.01)
    parser.add_argument("--aupro-num-steps", type=int, default=80)
    parser.add_argument("--num-prototypes", type=int, default=512)
    parser.add_argument("--distance-type", type=str, default="l2", choices=["cosine", "l2", "mahalanobis_diag"])
    parser.add_argument("--projection-type", type=str, default="sparse_random_projection", choices=["sparse_random_projection", "gaussian_random_projection"])
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--whitening-eps", type=float, default=1e-6)
    parser.add_argument(
        "--backbone-model-name",
        type=str,
        default="edgenext_small.usi_in1k",
        choices=SUPPORTED_BACKBONE_MODELS,
        help="Pretrained timm backbone model name.",
    )
    parser.add_argument(
        "--num-visualization-examples",
        type=int,
        default=5,
        help="How many final false-color overlay examples to save.",
    )
    parser.add_argument("--mahalanobis-alpha", type=float, default=0.7)
    parser.add_argument("--mahalanobis-min-var", type=float, default=1e-6)
    parser.add_argument("--mahalanobis-eps", type=float, default=1e-12)
    parser.add_argument("--enable-diagnostics", dest="enable_diagnostics", action="store_true", default=True)
    parser.add_argument("--disable-diagnostics", dest="enable_diagnostics", action="store_false")
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--save-per-sample-report", action="store_true", default=False)
    parser.add_argument("--save-plots", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug_mode:
        if not args.save_per_sample_report:
            args.save_per_sample_report = True
        if not args.save_plots:
            args.save_plots = True
    return args


def main() -> None:
    args = parse_args()
    cfg = PoCConfig(
        backbone_model_name=args.backbone_model_name,
        feature_size_factor=args.feature_size_factor,
        feature_layer_mode=args.feature_layer_mode,
        enable_two_stage_inference=args.enable_two_stage_inference,
        refine_feature_size_factor=args.refine_feature_size_factor,
        num_refine_rois=args.num_refine_rois,
        roi_crop_size=args.roi_crop_size,
        roi_expand_margin=args.roi_expand_margin,
        stage_a_trigger_threshold=args.stage_a_trigger_threshold,
        image_score_mode=args.image_score_mode,
        inference_backend=args.inference_backend,
        run_comparison_suite=args.run_comparison_suite,
        run_two_stage_experiment=args.run_two_stage_experiment,
        run_score_ablation=args.run_score_ablation,
        benchmark_fast_modes=args.benchmark_fast_modes,
        max_allowed_pixel_auroc_drop=args.max_allowed_pixel_auroc_drop,
        aupro_num_steps=args.aupro_num_steps,
        num_prototypes=args.num_prototypes,
        distance_type=args.distance_type,
        projection_type=args.projection_type,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        whitening_eps=args.whitening_eps,
        enable_diagnostics=args.enable_diagnostics,
        debug_mode=args.debug_mode,
        save_per_sample_report=args.save_per_sample_report,
        save_plots=args.save_plots,
        num_visualization_examples=args.num_visualization_examples,
        mahalanobis_alpha=args.mahalanobis_alpha,
        mahalanobis_min_var=args.mahalanobis_min_var,
        mahalanobis_eps=args.mahalanobis_eps,
    )
    run_poc(cfg)


if __name__ == "__main__":
    main()
