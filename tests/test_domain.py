import unittest

from object_detection.domain import Detection, percentile


class DetectionTests(unittest.TestCase):
    def test_clipping_and_confidence_bounds(self) -> None:
        detection = Detection(0, 0.0, -5, -2, 220, 120, 1, "chair", 1.5)
        clipped = detection.clipped(200, 100)
        self.assertEqual((clipped.x1, clipped.y1, clipped.x2, clipped.y2), (0, 0, 200, 100))
        self.assertEqual(clipped.confidence, 1.0)
        self.assertTrue(clipped.valid_box)

    def test_invalid_box(self) -> None:
        detection = Detection(0, 0.0, 20, 10, 5, 10, 0, "bottle", 0.5)
        self.assertFalse(detection.valid_box)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([], 0.5), 0.0)
        self.assertEqual(percentile([10], 0.5), 10.0)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.95), 3.85)


if __name__ == "__main__":
    unittest.main()

