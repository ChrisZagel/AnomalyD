import unittest
from pathlib import Path

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

    def test_missing_checkpoint_message_contains_fallback_hint(self) -> None:
        cfg = self.PoCConfig(checkpoint_path=str(Path("/tmp/does_not_exist.pth")), allow_backbone_fallback=False)
        with self.assertRaises(FileNotFoundError) as ctx:
            from app.metal_nut_poc import ensure_backbone_ready

            ensure_backbone_ready(cfg)
        self.assertIn("--allow-backbone-fallback", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
