import unittest

import numpy as np


class PrototypeDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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


if __name__ == "__main__":
    unittest.main()
