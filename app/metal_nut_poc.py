from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tarfile
import time
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from timm import create_model
from tqdm.auto import tqdm


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
    backbone_model_name: str = "vit_base_patch14_dinov2.lvd142m"
    feature_size_factor: float = 0.75
    num_prototypes: int = 256
    pca_dim: int = 128
    distance_type: str = "cosine"
    mahalanobis_alpha: float = 0.7
    mahalanobis_min_var: float = 1e-6
    mahalanobis_eps: float = 1e-12
    topk_percent: float = 1.0
    batch_size: int = 4
    num_workers: int = 2
    use_gaussian_smoothing: bool = True
    gaussian_sigma: float = 2.0
    max_pca_samples: int = 120_000
    layers: tuple[int, int, int] = (5, 8, 11)
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
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.no_grad()
    def extract_patch_features(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        orig_h, orig_w = tensor.shape[-2:]
        if self.cfg.feature_size_factor != 1.0:
            work_h = max(14, int(round(orig_h * self.cfg.feature_size_factor / 14) * 14))
            work_w = max(14, int(round(orig_w * self.cfg.feature_size_factor / 14) * 14))
            tensor = F.interpolate(tensor, size=(work_h, work_w), mode="bilinear", align_corners=False)
        else:
            work_h, work_w = orig_h, orig_w

        maps = self.backbone(tensor)
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
        return patch_feats.cpu().numpy(), (h, w), (work_h, work_w)


class PrototypeAnomalyModel:
    def __init__(self, cfg: PoCConfig) -> None:
        self.cfg = cfg
        self.pca: PCA | None = None
        self.centers: np.ndarray | None = None
        self.mahalanobis_means: np.ndarray | None = None
        self.mahalanobis_vars: np.ndarray | None = None
        self.mahalanobis_global_var: np.ndarray | None = None

    @staticmethod
    def _resolve_num_clusters(requested: int, n_samples: int) -> int:
        if n_samples <= 0:
            raise ValueError("No training samples available for clustering.")
        return max(1, min(requested, n_samples))

    @staticmethod
    def _chunked_assign_to_centers(x: np.ndarray, centers: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
        assigns: list[np.ndarray] = []
        c2 = np.sum(centers**2, axis=1)[None, :]
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i : i + chunk_size]
            d2 = np.sum(chunk**2, axis=1, keepdims=True) + c2 - 2.0 * chunk @ centers.T
            assigns.append(np.argmin(d2, axis=1).astype(np.int64))
        return np.concatenate(assigns, axis=0)

    def _fit_mahalanobis_diag_stats(self, train_feats_red: np.ndarray, kmeans_centers: np.ndarray) -> None:
        assigns = self._chunked_assign_to_centers(train_feats_red, kmeans_centers)
        n_clusters = kmeans_centers.shape[0]
        dim = train_feats_red.shape[1]

        counts = np.bincount(assigns, minlength=n_clusters).astype(np.float64)
        sums = np.zeros((n_clusters, dim), dtype=np.float64)
        sums_sq = np.zeros((n_clusters, dim), dtype=np.float64)

        np.add.at(sums, assigns, train_feats_red)
        np.add.at(sums_sq, assigns, train_feats_red * train_feats_red)

        global_var = np.var(train_feats_red, axis=0).astype(np.float32)
        means = kmeans_centers.astype(np.float32).copy()
        local_var = np.zeros((n_clusters, dim), dtype=np.float32)

        non_empty = counts > 0
        if np.any(non_empty):
            means[non_empty] = (sums[non_empty] / counts[non_empty, None]).astype(np.float32)
            ex2 = sums_sq[non_empty] / counts[non_empty, None]
            var = ex2 - np.square(means[non_empty], dtype=np.float64)
            local_var[non_empty] = np.clip(var, a_min=0.0, a_max=None).astype(np.float32)

        if np.any(~non_empty):
            local_var[~non_empty] = global_var

        alpha = float(self.cfg.mahalanobis_alpha)
        reg_var = alpha * local_var + (1.0 - alpha) * global_var[None, :]
        reg_var = np.clip(reg_var, a_min=float(self.cfg.mahalanobis_min_var), a_max=None).astype(np.float32)

        self.mahalanobis_means = means
        self.mahalanobis_vars = reg_var
        self.mahalanobis_global_var = global_var

    def fit(self, dataset: MVTecMetalNutDataset, extractor: FeatureExtractor) -> None:
        train_feature_blocks: list[np.ndarray] = []
        for sample in tqdm(dataset.samples, desc="Collect features for PCA"):
            image = Image.open(sample).convert("RGB")
            feats, _, _ = extractor.extract_patch_features(image)
            train_feature_blocks.append(feats)

        all_train_feats = np.concatenate(train_feature_blocks, axis=0)
        pca_fit_feats = all_train_feats
        if all_train_feats.shape[0] > self.cfg.max_pca_samples:
            idx = np.random.choice(all_train_feats.shape[0], self.cfg.max_pca_samples, replace=False)
            pca_fit_feats = all_train_feats[idx]

        self.pca = PCA(n_components=self.cfg.pca_dim, random_state=self.cfg.seed)
        self.pca.fit(pca_fit_feats)

        reduced_train_feats = [
            self.pca.transform(feats).astype(np.float32)
            for feats in tqdm(train_feature_blocks, desc="Fit prototypes")
        ]

        all_train_red = np.concatenate(reduced_train_feats, axis=0)
        effective_clusters = self._resolve_num_clusters(self.cfg.num_prototypes, all_train_red.shape[0])
        if effective_clusters != self.cfg.num_prototypes:
            print(
                f"[WARN] Reducing num_prototypes from {self.cfg.num_prototypes} to {effective_clusters} "
                f"because only {all_train_red.shape[0]} reduced patch samples are available."
            )

        kmeans = MiniBatchKMeans(
            n_clusters=effective_clusters,
            random_state=self.cfg.seed,
            batch_size=4096,
            n_init="auto",
        )
        kmeans.fit(all_train_red)

        self.centers = kmeans.cluster_centers_.astype(np.float32)

        if self.cfg.distance_type == "mahalanobis_diag":
            self._fit_mahalanobis_diag_stats(all_train_red.astype(np.float32), self.centers)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.pca is None or self.centers is None:
            raise RuntimeError("Model is not fitted yet.")

        import pickle

        payload = {
            "config": asdict(self.cfg),
            "pca": self.pca,
            "centers": self.centers,
            "mahalanobis_means": self.mahalanobis_means,
            "mahalanobis_vars": self.mahalanobis_vars,
            "mahalanobis_global_var": self.mahalanobis_global_var,
        }
        with open(output_dir / "prototype_model.pkl", "wb") as f:
            pickle.dump(payload, f)

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
                d2 = (
                    np.sum(chunk**2, axis=1, keepdims=True)
                    + np.sum(centers**2, axis=1)[None, :]
                    - 2.0 * chunk @ centers.T
                )
                mins.append(np.sqrt(np.clip(np.min(d2, axis=1), a_min=0.0, a_max=None)))
        elif self.cfg.distance_type == "mahalanobis_diag":
            if self.mahalanobis_means is None or self.mahalanobis_vars is None:
                raise RuntimeError("Mahalanobis statistics are not available. Fit model with distance_type='mahalanobis_diag'.")
            means = self.mahalanobis_means
            vars_ = self.mahalanobis_vars
            eps = float(self.cfg.mahalanobis_eps)
            for i in range(0, x.shape[0], chunk_size):
                chunk = x[i : i + chunk_size]
                diff = chunk[:, None, :] - means[None, :, :]
                d2 = np.sum((diff * diff) / (vars_[None, :, :] + eps), axis=2)
                mins.append(np.min(d2, axis=1))
        else:
            raise ValueError(f"Unsupported distance_type: {self.cfg.distance_type}")
        return np.concatenate(mins, axis=0)

    def infer_map(
        self,
        image: Image.Image,
        extractor: FeatureExtractor,
        orig_size: tuple[int, int],
    ) -> tuple[np.ndarray, float]:
        if self.pca is None or self.centers is None:
            raise RuntimeError("Model is not fitted yet.")

        feats, grid_hw, work_hw = extractor.extract_patch_features(image)
        feats_red = self.pca.transform(feats)
        patch_scores = self._chunked_min_distance(feats_red.astype(np.float32), self.centers)

        token_map = patch_scores.reshape(grid_hw)
        token_map_t = torch.from_numpy(token_map).unsqueeze(0).unsqueeze(0).float()
        work_map = F.interpolate(token_map_t, size=work_hw, mode="bilinear", align_corners=False)
        full_map = F.interpolate(work_map, size=orig_size, mode="bilinear", align_corners=False)[0, 0].numpy()

        if self.cfg.use_gaussian_smoothing:
            full_map = gaussian_filter(full_map, sigma=self.cfg.gaussian_sigma)

        topk = max(1, int(full_map.size * self.cfg.topk_percent / 100.0))
        image_score = float(np.mean(np.partition(full_map.ravel(), -topk)[-topk:]))
        return full_map, image_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    out_dir: Path,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    per_sample_rows: list[dict[str, Any]] = []
    per_sample_eval: list[dict[str, Any]] = []

    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    saved_vis = 0
    for idx in tqdm(range(len(test_ds)), desc="Evaluate test set"):
        sample = test_ds[idx]
        t0 = time.perf_counter()
        anom_map, image_score = proto_model.infer_map(sample["image"], extractor, sample["orig_size"])
        infer_sec = time.perf_counter() - t0

        image_labels.append(sample["label"])
        image_scores.append(image_score)
        pixel_labels.append(sample["mask"].astype(np.uint8).ravel())
        pixel_scores.append(anom_map.ravel())
        per_sample_rows.append(
            {
                "index": idx,
                "path": sample["path"],
                "defect_type": sample["defect_type"],
                "label": sample["label"],
                "image_score": float(image_score),
                "infer_time_sec": float(infer_sec),
                "map_mean": float(anom_map.mean()),
                "map_max": float(anom_map.max()),
            }
        )
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

        if saved_vis < 8:
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
    }
    return metrics, per_sample_rows, per_sample_eval


def save_per_sample_report(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "per_sample_report.csv"
    if not rows:
        return
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def compute_per_defect_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

        aupro = _compute_aupro_for_subset(subset)

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
        anom_map, _ = proto_model.infer_map(sample["image"], extractor, sample["orig_size"])
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
    setup_project_dirs(cfg)

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
    proto_model.fit(train_ds, extractor)
    train_time = time.perf_counter() - t_train

    output_dir = Path(cfg.project_root) / "outputs"
    proto_model.save(output_dir / "models")

    t_eval = time.perf_counter()
    metrics, per_sample_rows, per_sample_eval = evaluate(test_ds, extractor, proto_model, output_dir)
    eval_time = time.perf_counter() - t_eval
    total_time = time.perf_counter() - t_all

    metrics.update(
        {
            "train_stage_sec": float(train_time),
            "eval_stage_sec": float(eval_time),
            "total_runtime_sec": float(total_time),
            "train_images": len(train_ds),
            "test_images": len(test_ds),
        }
    )

    if proto_model.pca is not None:
        metrics["pca_explained_variance_ratio_sum"] = float(np.sum(proto_model.pca.explained_variance_ratio_))
    if proto_model.centers is not None:
        metrics["effective_num_prototypes"] = int(proto_model.centers.shape[0])

    metrics.update(compute_defect_level_aurocs(per_sample_rows))
    save_per_sample_report(per_sample_rows, output_dir)

    per_defect_metrics = compute_per_defect_metrics(per_sample_eval)
    save_per_defect_metrics(per_defect_metrics, output_dir)
    for row in per_defect_metrics:
        defect = row["defect_type"]
        metrics[f"pixel_auroc_{defect}"] = float(row["pixel_auroc"])
        metrics[f"aupro_{defect}"] = float(row["aupro"])
        metrics[f"image_precision_{defect}"] = float(row["image_precision"])
        metrics[f"image_recall_{defect}"] = float(row["image_recall"])
        metrics[f"image_f1_{defect}"] = float(row["image_f1"])
        metrics[f"pixel_precision_{defect}"] = float(row["pixel_precision"])
        metrics[f"pixel_recall_{defect}"] = float(row["pixel_recall"])
        metrics[f"pixel_f1_{defect}"] = float(row["pixel_f1"])

    save_metrics(metrics, output_dir)

    gallery_path = output_dir / "visualizations" / "final_top5_overlay_gallery.png"
    save_overlay_gallery(
        per_sample_rows,
        test_ds,
        extractor,
        proto_model,
        gallery_path,
        num_examples=cfg.num_visualization_examples,
    )

    print("Final metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Per-sample report: {output_dir / 'per_sample_report.csv'}")
    print(f"Per-defect metrics: {output_dir / 'per_defect_metrics.csv'}")
    if cfg.num_visualization_examples > 0:
        print(f"Overlay gallery (false-color): {gallery_path}")
    else:
        print("Overlay gallery skipped (--num-visualization-examples <= 0).")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVTec AD metal_nut PoC (pretrained timm backbones)")
    parser.add_argument("--feature-size-factor", type=float, default=0.75)
    parser.add_argument("--num-prototypes", type=int, default=256)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--distance-type", type=str, default="cosine", choices=["cosine", "l2", "mahalanobis_diag"])
    parser.add_argument(
        "--backbone-model-name",
        type=str,
        default="vit_base_patch14_dinov2.lvd142m",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PoCConfig(
        backbone_model_name=args.backbone_model_name,
        feature_size_factor=args.feature_size_factor,
        num_prototypes=args.num_prototypes,
        pca_dim=args.pca_dim,
        distance_type=args.distance_type,
        num_visualization_examples=args.num_visualization_examples,
        mahalanobis_alpha=args.mahalanobis_alpha,
        mahalanobis_min_var=args.mahalanobis_min_var,
        mahalanobis_eps=args.mahalanobis_eps,
    )
    run_poc(cfg)


if __name__ == "__main__":
    main()
