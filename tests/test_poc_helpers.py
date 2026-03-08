from pathlib import Path

import unittest

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


class PrototypeDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if np is None:
            raise unittest.SkipTest("Skipping PoC helper tests due to missing numpy")
        try:
            from app.metal_nut_poc import PoCConfig, PrototypeAnomalyModel
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"Skipping PoC helper tests due to missing dependencies: {exc}")
        cls.PoCConfig = PoCConfig
        cls.PrototypeAnomalyModel = PrototypeAnomalyModel

    def test_chunked_min_distance_cosine_shape(self) -> None:
        cfg = self.PoCConfig(distance_type="cosine")
        model = self.PrototypeAnomalyModel(cfg)
        x = np.random.randn(25, 8).astype(np.float32)
        c = np.random.randn(5, 8).astype(np.float32)
        out = model._chunked_min_distance(x, c, chunk_size=7)
        self.assertEqual(out.shape, (25,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_chunked_min_distance_l2_non_negative(self) -> None:
        cfg = self.PoCConfig(distance_type="l2")
        model = self.PrototypeAnomalyModel(cfg)
        x = np.random.randn(13, 4).astype(np.float32)
        c = np.random.randn(3, 4).astype(np.float32)
        out = model._chunked_min_distance(x, c, chunk_size=5)
        self.assertTrue(np.all(out >= 0))



    def test_chunked_min_distance_mahalanobis_diag_shape(self) -> None:
        cfg = self.PoCConfig(distance_type="mahalanobis_diag")
        model = self.PrototypeAnomalyModel(cfg)
        x = np.random.randn(11, 6).astype(np.float32)
        means = np.random.randn(4, 6).astype(np.float32)
        vars_ = np.abs(np.random.randn(4, 6).astype(np.float32)) + 1e-3
        model.mahalanobis_means = means
        model.mahalanobis_vars = vars_
        out = model._chunked_min_distance(x, means, chunk_size=5)
        self.assertEqual(out.shape, (11,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_resolve_num_clusters_caps_at_samples(self) -> None:
        cfg = self.PoCConfig()
        model = self.PrototypeAnomalyModel(cfg)
        self.assertEqual(model._resolve_num_clusters(256, 81), 81)
        self.assertEqual(model._resolve_num_clusters(64, 81), 64)


    def test_compute_per_defect_metrics_has_required_fields(self) -> None:
        from app.metal_nut_poc import compute_per_defect_metrics

        good = {
            "defect_type": "good",
            "label": 0,
            "image_score": 0.1,
            "mask": np.zeros((4, 4), dtype=np.uint8),
            "anom_map": np.zeros((4, 4), dtype=np.float32),
        }
        defect = {
            "defect_type": "scratch",
            "label": 1,
            "image_score": 0.9,
            "mask": np.pad(np.ones((2, 2), dtype=np.uint8), ((1, 1), (1, 1))),
            "anom_map": np.pad(np.ones((2, 2), dtype=np.float32), ((1, 1), (1, 1))),
        }
        rows = compute_per_defect_metrics([good, defect])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["defect_type"], "scratch")
        self.assertIn("pixel_auroc", rows[0])
        self.assertIn("aupro", rows[0])
        self.assertIn("image_f1", rows[0])
        self.assertIn("pixel_f1", rows[0])


    def test_save_overlay_gallery_ignores_non_positive_examples(self) -> None:
        from app.metal_nut_poc import save_overlay_gallery

        class DummyDS:
            def __getitem__(self, _idx):
                raise AssertionError("Should not be called when num_examples <= 0")

        save_overlay_gallery([], DummyDS(), None, None, Path("/tmp/unused.png"), num_examples=0)

    def test_supported_backbone_models_contains_requested_names(self) -> None:
        from app.metal_nut_poc import SUPPORTED_BACKBONE_MODELS

        required = {
            "vit_base_patch14_dinov2.lvd142m",
            "vit_base_patch14_reg4_dinov2.lvd142m",
            "vit_small_patch14_dinov2.lvd142m",
            "shvit_s4.in1k",
            "edgenext_small.usi_in1k",
        }
        self.assertTrue(required.issubset(set(SUPPORTED_BACKBONE_MODELS)))


if __name__ == "__main__":
    unittest.main()
