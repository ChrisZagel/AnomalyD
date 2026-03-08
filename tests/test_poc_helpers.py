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


    def test_resolve_num_clusters_caps_at_samples(self) -> None:
        cfg = self.PoCConfig()
        model = self.PrototypeAnomalyModel(cfg)
        self.assertEqual(model._resolve_num_clusters(256, 81), 81)
        self.assertEqual(model._resolve_num_clusters(64, 81), 64)

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
