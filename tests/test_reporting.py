import json
import tempfile
import unittest
from pathlib import Path

from object_detection.errors import ConfigurationError
from object_detection.reporting import build_comparison


class ReportingTests(unittest.TestCase):
    def test_comparison_is_generated_from_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "precision": 0.8,
                        "recall": 0.7,
                        "f1": 0.7467,
                        "map50": 0.75,
                        "map50_95": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            result = build_comparison([f"baseline={metrics}"], root / "comparison")
            self.assertTrue(Path(result["csv"]).is_file())
            self.assertTrue(Path(result["json"]).is_file())
            self.assertEqual(result["conditions"][0]["condition"], "baseline")

    def test_missing_metrics_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "missing"):
                build_comparison([f"bad={path}"], Path(temporary) / "out")


if __name__ == "__main__":
    unittest.main()
