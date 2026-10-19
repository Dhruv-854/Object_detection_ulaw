import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from object_detection.adapters import (
    ModelSpec,
    UltralyticsDetector,
    YoloV5Detector,
    resolve_model_spec,
)
from object_detection.errors import ModelError


class ModelResolutionTests(unittest.TestCase):
    def test_alias_requires_and_verifies_checksum_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            checkpoint = models / "yolo26n.pt"
            checkpoint.write_bytes(b"fixed checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (models / "model-lock.json").write_text(
                json.dumps({"models": {"yolo26n": {"sha256": digest}}}), encoding="utf-8"
            )
            spec = resolve_model_spec("yolo26n", project_root=root)
            self.assertEqual(spec.family, "ultralytics")
            self.assertEqual(spec.checkpoint, checkpoint.resolve())

            checkpoint.write_bytes(b"replaced")
            with self.assertRaisesRegex(ModelError, "checksum mismatch"):
                resolve_model_spec("yolo26n", project_root=root)

    def test_custom_checkpoint_requires_inferable_or_explicit_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"x")
            with self.assertRaisesRegex(ModelError, "Could not infer"):
                resolve_model_spec(str(checkpoint))
            spec = resolve_model_spec(str(checkpoint), family="yolov5")
            self.assertEqual(spec.family, "yolov5")


class _Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _V5Result:
    xyxy = [_Tensor([[1, 2, 30, 40, 0.75, 0]])]
    names = {0: "bottle"}


class _V5Model:
    names = {0: "bottle"}

    def to(self, device):
        self.device = device

    def __call__(self, frame, size):
        return _V5Result()


class _Hub:
    def __init__(self):
        self.model = _V5Model()

    def load(self, *args, **kwargs):
        return self.model


class _Torch:
    def __init__(self):
        self.hub = _Hub()


class _Boxes:
    xyxy = _Tensor([[5, 6, 50, 60]])
    conf = _Tensor([0.85])
    cls = _Tensor([1])


class _CurrentResult:
    boxes = _Boxes()
    names = {1: "chair"}


class _CurrentModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [_CurrentResult()]


class AdapterNormalizationTests(unittest.TestCase):
    def test_yolov5_output_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "yolov5"
            repo.mkdir()
            (repo / "hubconf.py").write_text("# fake", encoding="utf-8")
            checkpoint = root / "yolov5s.pt"
            checkpoint.write_bytes(b"checkpoint")
            detector = YoloV5Detector(
                ModelSpec("yolov5s", "yolov5", checkpoint),
                "cpu",
                640,
                0.25,
                1000,
                repo,
                torch_module=_Torch(),
            )
            detections = detector.predict(object(), 4, 120.0)
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0].class_name, "bottle")
            self.assertEqual(detections[0].frame_index, 4)
            self.assertEqual(detector._model.max_det, 1000)

    def test_current_output_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "yolo26n.pt"
            checkpoint.write_bytes(b"checkpoint")
            detector = UltralyticsDetector(
                ModelSpec("yolo26n", "ultralytics", checkpoint),
                "cpu",
                640,
                0.25,
                1000,
                yolo_class=_CurrentModel,
            )
            detections = detector.predict(object(), 7, 80.0)
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0].class_name, "chair")
            self.assertAlmostEqual(detections[0].confidence, 0.85)
            self.assertEqual(detector._model.kwargs["max_det"], 1000)


if __name__ == "__main__":
    unittest.main()
