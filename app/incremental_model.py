from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


class IncrementalWhiteningStats:
    def __init__(self, eps: float = 1e-6, momentum: float | None = None) -> None:
        self.eps = float(eps)
        self.momentum = momentum
        self.n_seen: int = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def fit_initial(self, x: np.ndarray) -> None:
        if x.size == 0:
            raise ValueError("Cannot initialize whitening stats with empty input.")
        self.n_seen = int(x.shape[0])
        self.mean = np.mean(x, axis=0).astype(np.float64)
        centered = x.astype(np.float64) - self.mean
        self.m2 = np.sum(centered * centered, axis=0)

    def partial_update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        if self.mean is None or self.m2 is None:
            self.fit_initial(x)
            return

        x = x.astype(np.float64)
        batch_n = int(x.shape[0])
        batch_mean = np.mean(x, axis=0)
        centered = x - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)

        old_n = self.n_seen
        total_n = old_n + batch_n
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * (batch_n / max(total_n, 1))
        new_m2 = self.m2 + batch_m2 + (delta * delta) * (old_n * batch_n / max(total_n, 1))

        if self.momentum is not None:
            m = float(np.clip(self.momentum, 0.0, 1.0))
            new_mean = m * self.mean + (1.0 - m) * new_mean
            new_m2 = m * self.m2 + (1.0 - m) * new_m2

        self.n_seen = total_n
        self.mean = new_mean
        self.m2 = new_m2

    @property
    def var(self) -> np.ndarray:
        if self.mean is None or self.m2 is None:
            raise RuntimeError("Whitening stats not initialized.")
        denom = max(self.n_seen - 1, 1)
        return np.clip(self.m2 / denom, a_min=self.eps, a_max=None)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Whitening stats not initialized.")
        return ((x - self.mean) / np.sqrt(self.var + self.eps)).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "momentum": self.momentum,
            "n_seen": self.n_seen,
            "mean": self.mean,
            "m2": self.m2,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.eps = float(state["eps"])
        self.momentum = state.get("momentum")
        self.n_seen = int(state["n_seen"])
        self.mean = state["mean"]
        self.m2 = state["m2"]


class FixedRandomProjector:
    def __init__(self, projection_dim: int = 96, projection_type: str = "sparse_random_projection", seed: int = 42) -> None:
        self.projection_dim = int(projection_dim)
        self.projection_type = projection_type
        self.seed = int(seed)
        self._projector: SparseRandomProjection | GaussianRandomProjection | None = None

    def fit_initial(self, x: np.ndarray) -> None:
        if self.projection_type == "sparse_random_projection":
            proj = SparseRandomProjection(n_components=self.projection_dim, random_state=self.seed)
        elif self.projection_type == "gaussian_random_projection":
            proj = GaussianRandomProjection(n_components=self.projection_dim, random_state=self.seed)
        else:
            raise ValueError(f"Unsupported projection type: {self.projection_type}")
        proj.fit(x)
        self._projector = proj

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._projector is None:
            raise RuntimeError("Projector not initialized.")
        return self._projector.transform(x).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        if self._projector is None:
            raise RuntimeError("Projector not initialized.")
        return {
            "projection_dim": self.projection_dim,
            "projection_type": self.projection_type,
            "seed": self.seed,
            "projector": self._projector,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.projection_dim = int(state["projection_dim"])
        self.projection_type = str(state["projection_type"])
        self.seed = int(state["seed"])
        self._projector = state["projector"]


class IncrementalWhitenedProjection:
    def __init__(self, eps: float = 1e-6, projection_dim: int = 96, projection_type: str = "sparse_random_projection", seed: int = 42, whitening_momentum: float | None = None) -> None:
        self.whitening = IncrementalWhiteningStats(eps=eps, momentum=whitening_momentum)
        self.projector = FixedRandomProjector(projection_dim=projection_dim, projection_type=projection_type, seed=seed)

    def fit_initial(self, x: np.ndarray) -> None:
        self.whitening.fit_initial(x)
        x_white = self.whitening.transform(x)
        self.projector.fit_initial(x_white)

    def partial_update(self, x: np.ndarray) -> None:
        self.whitening.partial_update(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.projector.transform(self.whitening.transform(x))

    def state_dict(self) -> dict[str, Any]:
        return {
            "whitening": self.whitening.state_dict(),
            "projector": self.projector.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.whitening.load_state_dict(state["whitening"])
        self.projector.load_state_dict(state["projector"])


class ReplayMemoryManager:
    def __init__(self, max_features: int = 200000, coverage_fraction: float = 0.5, boundary_fraction: float = 0.25, recent_fraction: float = 0.25) -> None:
        self.max_features = int(max_features)
        self.coverage_cap = int(max_features * coverage_fraction)
        self.boundary_cap = int(max_features * boundary_fraction)
        self.recent_cap = max(1, int(max_features * recent_fraction))
        self.coverage: list[np.ndarray] = []
        self.boundary: list[tuple[float, np.ndarray]] = []
        self.recent: list[np.ndarray] = []
        self._seen_coverage = 0

    def add(self, feats: np.ndarray, scores: np.ndarray, boundary_quantile: float = 0.9) -> None:
        if feats.size == 0:
            return
        for feat, score in zip(feats, scores):
            self._seen_coverage += 1
            if len(self.coverage) < self.coverage_cap:
                self.coverage.append(feat.copy())
            else:
                j = np.random.randint(0, self._seen_coverage)
                if j < self.coverage_cap:
                    self.coverage[j] = feat.copy()

            self.recent.append(feat.copy())
            if len(self.recent) > self.recent_cap:
                self.recent.pop(0)

            self.boundary.append((float(score), feat.copy()))
        if self.boundary:
            self.boundary.sort(key=lambda x: x[0], reverse=True)
            self.boundary = self.boundary[: self.boundary_cap]

    def sample_for_update(self) -> np.ndarray:
        parts = []
        if self.coverage:
            parts.append(np.stack(self.coverage, axis=0))
        if self.boundary:
            parts.append(np.stack([x[1] for x in self.boundary], axis=0))
        if self.recent:
            parts.append(np.stack(self.recent, axis=0))
        if not parts:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def composition(self) -> dict[str, int]:
        return {"coverage": len(self.coverage), "boundary": len(self.boundary), "recent": len(self.recent)}

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_features": self.max_features,
            "coverage_cap": self.coverage_cap,
            "boundary_cap": self.boundary_cap,
            "recent_cap": self.recent_cap,
            "coverage": self.coverage,
            "boundary": self.boundary,
            "recent": self.recent,
            "seen_coverage": self._seen_coverage,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.max_features = int(state["max_features"])
        self.coverage_cap = int(state["coverage_cap"])
        self.boundary_cap = int(state["boundary_cap"])
        self.recent_cap = int(state["recent_cap"])
        self.coverage = state["coverage"]
        self.boundary = state["boundary"]
        self.recent = state["recent"]
        self._seen_coverage = int(state["seen_coverage"])


@dataclass
class PrototypeMeta:
    status: str
    age: int
    replay_hits: int
    last_update_step: int
    importance: float


class PrototypeStore:
    def __init__(self, num_prototypes: int = 512, distance_type: str = "l2", mahalanobis_eps: float = 1e-12, candidate_promotion_hits: int = 5, prototype_forget_patience: int = 5000) -> None:
        self.num_prototypes = int(num_prototypes)
        self.distance_type = distance_type
        self.mahalanobis_eps = float(mahalanobis_eps)
        self.candidate_promotion_hits = int(candidate_promotion_hits)
        self.prototype_forget_patience = int(prototype_forget_patience)
        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        self.meta: list[PrototypeMeta] = []

    def fit_initial(self, x: np.ndarray, seed: int = 42) -> None:
        n_clusters = max(1, min(self.num_prototypes, x.shape[0]))
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init="auto", batch_size=4096)
        km.fit(x)
        labels = km.predict(x)
        dim = x.shape[1]
        means = km.cluster_centers_.astype(np.float32)
        vars_ = np.zeros((n_clusters, dim), dtype=np.float32)
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
        for k in range(n_clusters):
            cluster = x[labels == k]
            if cluster.size == 0:
                vars_[k] = np.var(x, axis=0) + 1e-6
            else:
                vars_[k] = np.var(cluster, axis=0) + 1e-6
        self.means, self.vars, self.counts = means, vars_, counts
        self.meta = [PrototypeMeta(status="stable", age=0, replay_hits=0, last_update_step=0, importance=float(c)) for c in counts]

    def _distance(self, x: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        if self.distance_type == "cosine":
            xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            mn = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
            return 1.0 - xn @ mn.T
        if self.distance_type == "mahalanobis_diag":
            diff = x[:, None, :] - means[None, :, :]
            return np.sum((diff * diff) / (vars_[None, :, :] + self.mahalanobis_eps), axis=2)
        d2 = np.sum(x**2, axis=1, keepdims=True) + np.sum(means**2, axis=1)[None, :] - 2.0 * x @ means.T
        return np.sqrt(np.clip(d2, 0.0, None))

    def nearest(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.means is None or self.vars is None:
            raise RuntimeError("Prototype store not initialized.")
        d = self._distance(x, self.means, self.vars)
        idx = np.argmin(d, axis=1)
        return d[np.arange(len(idx)), idx], idx

    def update_online(self, x: np.ndarray, distances: np.ndarray, idx: np.ndarray, candidate_threshold: float, step: int) -> int:
        assert self.means is not None and self.vars is not None and self.counts is not None
        promoted = 0
        for i, feat in enumerate(x):
            pid = int(idx[i])
            dist = float(distances[i])
            if self.meta[pid].status == "stable" and dist > candidate_threshold:
                self.means = np.vstack([self.means, feat[None, :]])
                self.vars = np.vstack([self.vars, np.ones_like(feat[None, :], dtype=np.float32)])
                self.counts = np.concatenate([self.counts, np.array([1.0], dtype=np.float32)])
                self.meta.append(PrototypeMeta(status="candidate", age=0, replay_hits=1, last_update_step=step, importance=1.0))
                continue
            c = self.counts[pid] + 1.0
            lr = 1.0 / c
            delta = feat - self.means[pid]
            self.means[pid] += lr * delta
            self.vars[pid] = np.clip((1 - lr) * self.vars[pid] + lr * (delta * delta), 1e-6, None)
            self.counts[pid] = c
            self.meta[pid].age = 0
            self.meta[pid].last_update_step = step
            self.meta[pid].replay_hits += 1
            self.meta[pid].importance = float(self.counts[pid])
            if self.meta[pid].status == "candidate" and self.meta[pid].replay_hits >= self.candidate_promotion_hits:
                self.meta[pid].status = "stable"
                promoted += 1
        for m in self.meta:
            m.age += 1
        return promoted

    def consolidate(self, replay_x: np.ndarray, step: int) -> dict[str, int]:
        removed = 0
        fading = 0
        for i in range(len(self.meta) - 1, -1, -1):
            m = self.meta[i]
            if m.age > self.prototype_forget_patience:
                m.status = "fading"
                fading += 1
            if m.age > self.prototype_forget_patience * 2 and m.replay_hits < 2:
                self.means = np.delete(self.means, i, axis=0)
                self.vars = np.delete(self.vars, i, axis=0)
                self.counts = np.delete(self.counts, i, axis=0)
                self.meta.pop(i)
                removed += 1
        return {"num_fading": fading, "num_removed": removed}

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_prototypes": self.num_prototypes,
            "distance_type": self.distance_type,
            "mahalanobis_eps": self.mahalanobis_eps,
            "candidate_promotion_hits": self.candidate_promotion_hits,
            "prototype_forget_patience": self.prototype_forget_patience,
            "means": self.means,
            "vars": self.vars,
            "counts": self.counts,
            "meta": [m.__dict__ for m in self.meta],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.num_prototypes = int(state["num_prototypes"])
        self.distance_type = state["distance_type"]
        self.mahalanobis_eps = float(state["mahalanobis_eps"])
        self.candidate_promotion_hits = int(state["candidate_promotion_hits"])
        self.prototype_forget_patience = int(state["prototype_forget_patience"])
        self.means = state["means"]
        self.vars = state["vars"]
        self.counts = state["counts"]
        self.meta = [PrototypeMeta(**m) for m in state["meta"]]


class IncrementalADModel:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.transformer = IncrementalWhitenedProjection(
            eps=config.get("whitening_eps", 1e-6),
            projection_dim=config.get("projection_dim", 96),
            projection_type=config.get("projection_type", "sparse_random_projection"),
            seed=config.get("projection_seed", 42),
            whitening_momentum=config.get("whitening_momentum"),
        )
        self.replay = ReplayMemoryManager(
            max_features=config.get("max_replay_features", 200000),
            coverage_fraction=config.get("coverage_fraction", 0.5),
            boundary_fraction=config.get("boundary_fraction", 0.25),
            recent_fraction=config.get("recent_fraction", 0.25),
        )
        self.prototypes = PrototypeStore(
            num_prototypes=config.get("num_prototypes", 512),
            distance_type=config.get("distance_type", "l2"),
            mahalanobis_eps=config.get("mahalanobis_eps", 1e-12),
            candidate_promotion_hits=config.get("candidate_promotion_hits", 5),
            prototype_forget_patience=config.get("prototype_forget_patience", 5000),
        )
        self.model_version = 1
        self.num_seen_images = 0
        self.accepted_images = 0
        self.rejected_images = 0
        self.threshold = float(config.get("update_accept_threshold", 3.0))
        self._checkpoint: dict[str, Any] | None = None
        self.last_fit_timing: dict[str, float] = {}
        self.last_consolidation_stats: dict[str, float] = {}

    def fit_initial(self, x: np.ndarray) -> None:
        import time

        t0 = time.perf_counter()
        self.transformer.fit_initial(x)
        t1 = time.perf_counter()
        z = self.transformer.transform(x)
        t2 = time.perf_counter()
        self.prototypes.fit_initial(z, seed=self.config.get("projection_seed", 42))
        t3 = time.perf_counter()
        d, _ = self.prototypes.nearest(z)
        self.replay.add(x, d)
        self.threshold = float(np.quantile(d, self.config.get("update_accept_quantile", 0.99)))
        t4 = time.perf_counter()
        self.last_fit_timing = {
            "time_transform_update_train": float((t1 - t0) + (t2 - t1)),
            "time_prototype_update_train": float((t3 - t2) + (t4 - t3)),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.transformer.transform(x)
        d, _ = self.prototypes.nearest(z)
        return d

    def update_incremental(self, patch_features_per_image: list[np.ndarray]) -> dict[str, Any]:
        promoted = 0
        accepted_batch: list[np.ndarray] = []
        accepted_scores: list[np.ndarray] = []
        for feats in patch_features_per_image:
            self.num_seen_images += 1
            scores = self.predict(feats)
            image_score = float(np.mean(scores))
            max_patch = float(np.max(scores))
            area = float(np.mean(scores > self.threshold))
            if image_score < self.threshold and max_patch < self.config.get("max_patch_score_threshold", self.threshold * 1.5) and area < self.config.get("max_anomalous_area_fraction", 0.05):
                self.accepted_images += 1
                accepted_batch.append(feats)
                accepted_scores.append(scores)
            else:
                self.rejected_images += 1

        for feats, scores in zip(accepted_batch, accepted_scores):
            self.replay.add(feats, scores)
            z = self.transformer.transform(feats)
            d, idx = self.prototypes.nearest(z)
            promoted += self.prototypes.update_online(
                z, d, idx, candidate_threshold=self.config.get("candidate_distance_threshold", self.threshold), step=self.num_seen_images
            )

        if self.accepted_images > 0 and self.accepted_images % self.config.get("update_trigger_images", 20) == 0:
            self.consolidate()

        return {"accepted": len(accepted_batch), "promoted": promoted, "threshold": self.threshold}

    def consolidate(self) -> dict[str, Any]:
        replay_x = self.replay.sample_for_update()
        if replay_x.size == 0:
            return {"status": "skipped"}
        self._checkpoint = self.state_dict()
        old_mean = self.transformer.whitening.mean.copy() if self.transformer.whitening.mean is not None else None
        old_var = self.transformer.whitening.var.copy() if self.transformer.whitening.mean is not None else None

        self.transformer.partial_update(replay_x)
        z = self.transformer.transform(replay_x)
        d, _ = self.prototypes.nearest(z)
        recent_t = float(np.quantile(d, self.config.get("update_accept_quantile", 0.99)))
        beta = float(self.config.get("threshold_beta", 0.8))
        self.threshold = beta * self.threshold + (1.0 - beta) * recent_t

        stats = self.prototypes.consolidate(z, step=self.num_seen_images)
        self.model_version += 1

        new_mean = self.transformer.whitening.mean
        new_var = self.transformer.whitening.var
        mean_shift = float(np.linalg.norm(new_mean - old_mean)) if old_mean is not None else 0.0
        var_shift = float(np.linalg.norm(new_var - old_var)) if old_var is not None else 0.0
        result = {
            "status": "ok",
            "model_version": self.model_version,
            "whitening_mean_shift": mean_shift,
            "whitening_var_shift": var_shift,
            "replay_sizes": self.replay.composition(),
            "threshold": self.threshold,
            **stats,
        }
        self.last_consolidation_stats = {"whitening_mean_shift": mean_shift, "whitening_var_shift": var_shift}
        if isinstance(self.config, dict):
            self.config["last_consolidation_stats"] = dict(self.last_consolidation_stats)
        return result

    def rollback(self) -> None:
        if self._checkpoint is not None:
            self.load_state_dict(self._checkpoint)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "transformer": self.transformer.state_dict(),
            "replay": self.replay.state_dict(),
            "prototypes": self.prototypes.state_dict(),
            "model_version": self.model_version,
            "num_seen_images": self.num_seen_images,
            "accepted_images": self.accepted_images,
            "rejected_images": self.rejected_images,
            "threshold": self.threshold,
            "last_fit_timing": self.last_fit_timing,
            "last_consolidation_stats": self.last_consolidation_stats,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.config = state["config"]
        self.transformer.load_state_dict(state["transformer"])
        self.replay.load_state_dict(state["replay"])
        self.prototypes.load_state_dict(state["prototypes"])
        self.model_version = int(state["model_version"])
        self.num_seen_images = int(state["num_seen_images"])
        self.accepted_images = int(state["accepted_images"])
        self.rejected_images = int(state["rejected_images"])
        self.threshold = float(state["threshold"])
        self.last_fit_timing = state.get("last_fit_timing", {})
        self.last_consolidation_stats = state.get("last_consolidation_stats", {})

    def save_state(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def load_state(cls, path: str | Path) -> "IncrementalADModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(state["config"])
        model.load_state_dict(state)
        return model
