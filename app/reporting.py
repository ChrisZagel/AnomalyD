from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunContext:
    run_id: str
    timestamp: str
    root_dir: Path
    metrics_dir: Path
    tables_dir: Path
    plots_dir: Path
    visualizations_dir: Path

    @classmethod
    def start_run(cls, project_root: str | Path) -> "RunContext":
        ts = datetime.now(timezone.utc)
        run_id = ts.strftime("%Y%m%d_%H%M%S")
        outputs_root = Path(project_root) / "outputs"
        run_root = outputs_root / run_id
        metrics_dir = run_root / "metrics"
        tables_dir = run_root / "tables"
        plots_dir = run_root / "plots"
        visualizations_dir = run_root / "visualizations"
        for path in (metrics_dir, tables_dir, plots_dir, visualizations_dir):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            timestamp=ts.isoformat(),
            root_dir=run_root,
            metrics_dir=metrics_dir,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            visualizations_dir=visualizations_dir,
        )


class ExperimentTableWriter:
    def __init__(self, project_root: str | Path) -> None:
        self.path = Path(project_root) / "outputs" / "experiment_results.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_row(self, row: dict[str, Any]) -> None:
        required_cols = [
            "run_id",
            "timestamp",
            "backbone_name",
            "transform_type",
            "projection_dim",
            "feature_size_factor",
            "num_prototypes",
            "distance_type",
            "image_auroc",
            "pixel_auroc",
            "aupro_mean",
            "mean_infer_time_sec",
            "total_runtime_sec",
            "device_type",
        ]
        exists = self.path.exists()
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=required_cols)
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in required_cols})


class LeanMetricsLogger:
    def __init__(self, ctx: RunContext, enable_diagnostics: bool) -> None:
        self.ctx = ctx
        self.enable_diagnostics = enable_diagnostics

    def log_summary_metrics(self, summary: dict[str, Any], minimal_csv: dict[str, Any]) -> None:
        if not self.enable_diagnostics:
            with open(self.ctx.root_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            with open(self.ctx.root_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(minimal_csv.keys()))
                writer.writeheader()
                writer.writerow(minimal_csv)
            return

        with open(self.ctx.metrics_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def log_timing_summary(self, timing: dict[str, Any]) -> None:
        if not self.enable_diagnostics:
            return
        with open(self.ctx.metrics_dir / "timing_summary.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)

    def log_per_defect_metrics(self, rows: list[dict[str, Any]]) -> None:
        if not self.enable_diagnostics or not rows:
            return
        out_path = self.ctx.tables_dir / "per_defect_metrics.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


class DebugReporter:
    def __init__(self, ctx: RunContext, enable_diagnostics: bool, debug_mode: bool, save_per_sample_report: bool, save_plots: bool) -> None:
        self.ctx = ctx
        self.enable_diagnostics = enable_diagnostics
        self.debug_mode = debug_mode
        self.save_per_sample_report = save_per_sample_report
        self.save_plots = save_plots

    def log_per_sample_table(self, rows: list[dict[str, Any]]) -> None:
        if not (self.debug_mode and self.save_per_sample_report and rows):
            return
        out_path = self.ctx.tables_dir / "per_sample_report.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def log_prototype_summary(self, summary: dict[str, Any]) -> None:
        if not self.enable_diagnostics:
            return
        with open(self.ctx.metrics_dir / "prototype_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def log_transform_summary(self, summary: dict[str, Any]) -> None:
        if not self.enable_diagnostics:
            return
        with open(self.ctx.metrics_dir / "transform_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def log_incremental_update(self, rows: list[dict[str, Any]]) -> None:
        if not (self.enable_diagnostics and rows):
            return
        out_path = self.ctx.tables_dir / "incremental_update_log.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def log_forgetting_report(self, rows: list[dict[str, Any]]) -> None:
        if not (self.enable_diagnostics and rows):
            return
        out_path = self.ctx.tables_dir / "forgetting_report.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def create_debug_plots(self, per_defect_rows: list[dict[str, Any]], image_scores: np.ndarray, pixel_scores: np.ndarray, timing: dict[str, float]) -> None:
        if not (self.debug_mode and self.save_plots):
            return

        # timing breakdown
        keys = [k for k in timing.keys() if k.startswith("time_")]
        vals = [timing[k] for k in keys]
        if keys:
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(keys))
            ax.bar(x, vals)
            ax.set_ylabel("seconds")
            ax.set_xticks(x)
            ax.set_xticklabels(keys, rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(self.ctx.plots_dir / "timing_breakdown.png", dpi=130)
            plt.close(fig)

        # per-defect overview
        if per_defect_rows:
            defects = [r["defect_type"] for r in per_defect_rows]
            image_auc = [float(r["image_auroc"]) for r in per_defect_rows]
            pixel_auc = [float(r["pixel_auroc"]) for r in per_defect_rows]
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(defects))
            w = 0.35
            ax.bar(x - w / 2, image_auc, w, label="image_auroc")
            ax.bar(x + w / 2, pixel_auc, w, label="pixel_auroc")
            ax.set_xticks(x)
            ax.set_xticklabels(defects, rotation=30, ha="right")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.ctx.plots_dir / "per_defect_metrics_overview.png", dpi=130)
            plt.close(fig)

        if image_scores.size:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(image_scores, bins=40)
            ax.set_title("Image Score Histogram")
            fig.tight_layout()
            fig.savefig(self.ctx.plots_dir / "image_score_hist_global.png", dpi=130)
            plt.close(fig)

        if pixel_scores.size:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(pixel_scores, bins=40)
            ax.set_title("Pixel Score Histogram")
            fig.tight_layout()
            fig.savefig(self.ctx.plots_dir / "pixel_score_hist_global.png", dpi=130)
            plt.close(fig)
