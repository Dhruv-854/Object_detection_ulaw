import tempfile
import unittest
from pathlib import Path

from object_detection.errors import SourceError
from object_detection.sources import parse_source


class SourceTests(unittest.TestCase):
    def test_camera_index(self) -> None:
        source = parse_source("0")
        self.assertEqual(source.kind, "camera")
        self.assertEqual(source.capture_value, 0)

    def test_image_and_video_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "sample.JPG"
            video = root / "sample.mp4"
            image.write_bytes(b"image")
            video.write_bytes(b"video")
            self.assertEqual(parse_source(str(image)).kind, "image")
            self.assertEqual(parse_source(str(video)).kind, "video")

    def test_missing_and_unsupported_sources(self) -> None:
        with self.assertRaisesRegex(SourceError, "does not exist"):
            parse_source("missing.jpg")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.txt"
            path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(SourceError, "Unsupported"):
                parse_source(str(path))


if __name__ == "__main__":
    unittest.main()

