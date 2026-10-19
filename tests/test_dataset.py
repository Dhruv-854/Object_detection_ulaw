import csv
import tempfile
import unittest
from pathlib import Path

from object_detection.dataset import audit_dataset, create_robustness_slices, split_by_session


FIELDS = [
    "image_path",
    "session_id",
    "lighting",
    "distance",
    "occlusion",
    "motion_blur",
    "negative",
    "notes",
]


class DatasetTests(unittest.TestCase):
    def _make_dataset(self, root: Path, invalid: bool = False) -> Path:
        (root / "images").mkdir()
        (root / "labels").mkdir()
        rows = []
        for index in range(6):
            image_name = f"image-{index}.jpg"
            (root / "images" / image_name).write_bytes(f"image-{index}".encode())
            label = "8 0.5 0.5 0.2 0.2\n" if invalid and index == 0 else "0 0.5 0.5 0.2 0.2\n"
            (root / "labels" / f"image-{index}.txt").write_text(label, encoding="utf-8")
            rows.append(
                {
                    "image_path": f"images/{image_name}",
                    "session_id": f"session-{index % 3}",
                    "lighting": "normal",
                    "distance": "near",
                    "occlusion": "none",
                    "motion_blur": "false",
                    "negative": "false",
                    "notes": "",
                }
            )
        metadata = root / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return metadata

    def test_pilot_audit_and_grouped_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            metadata = self._make_dataset(root)
            report = audit_dataset(root, metadata, strict_targets=False)
            self.assertEqual(report.error_count, 0)
            self.assertGreater(report.warning_count, 0)
            self.assertEqual(report.image_count, 6)

            output = Path(temporary) / "processed"
            manifest = split_by_session(root, metadata, output)
            assignments = manifest["session_assignments"]
            self.assertEqual(set(assignments.values()), {"train", "val", "test"})
            self.assertTrue((output / "dataset.yaml").is_file())
            self.assertTrue((output / "split-manifest.json").is_file())
            slices = create_robustness_slices(
                root, output / "metadata_with_split.csv", output / "slices"
            )
            self.assertIn("lighting-normal", slices["slices"])
            self.assertTrue(Path(slices["manifest"]).is_file())

    def test_invalid_class_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            metadata = self._make_dataset(root, invalid=True)
            report = audit_dataset(root, metadata, strict_targets=False)
            self.assertGreater(report.error_count, 0)
            self.assertIn("unknown-class", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
