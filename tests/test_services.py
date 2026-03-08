import unittest

from app.services import build_runtime_summary


class RuntimeSummaryTests(unittest.TestCase):
    def test_build_runtime_summary_contains_expected_fields(self) -> None:
        summary = build_runtime_summary("Max").to_dict()

        self.assertEqual(summary["message"], "Hello Max!")
        self.assertIn("timestamp_utc", summary)
        self.assertIn("python_version", summary)
        self.assertIn("platform", summary)


if __name__ == "__main__":
    unittest.main()
