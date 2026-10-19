from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .domain import Detection, sha256_file
from .errors import ModelError


class Detector(Protocol):
    model_id: str
    family: str
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    parameter_count: int | None
    device: str
    input_size: int
    confidence: float
    max_detections: int

    def predict(
        self, frame: Any, frame_index: int, timestamp_ms: float
    ) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    family: str
    checkpoint: Path


def resolve_model_spec(
    model: str,
    family: str | None = None,
    project_root: str | Path | None = None,
) -> ModelSpec:
    root = Path(project_root or Path.cwd()).resolve()
    aliases = {
        "yolov5s": ModelSpec("yolov5s", "yolov5", root / "models/yolov5s-v7.0.pt"),
        "yolo26n": ModelSpec("yolo26n", "ultralytics", root / "models/yolo26n.pt"),
    }
    if model.lower() in aliases:
        spec = aliases[model.lower()]
        is_alias = True
    else:
        checkpoint = Path(model).expanduser().resolve()
        inferred_family = family or _infer_family(checkpoint.name)
        if inferred_family is None:
            raise ModelError(
                "Could not infer the checkpoint family. Add --family yolov5 or "
                "--family ultralytics."
            )
        spec = ModelSpec(checkpoint.stem, inferred_family, checkpoint)
        is_alias = False

    if family and family != spec.family:
        raise ModelError(
            f"Model '{model}' is registered as family '{spec.family}', not '{family}'."
        )
    if spec.family not in {"yolov5", "ultralytics"}:
        raise ModelError("Model family must be 'yolov5' or 'ultralytics'.")
    if not spec.checkpoint.is_file():
        raise ModelError(
            f"Checkpoint not found: {spec.checkpoint}. "
            "Download the pinned checkpoint using the setup instructions in docs/USER_GUIDE.md."
        )
    if is_alias:
        _verify_alias_lock(root, spec)
    return spec


def _infer_family(filename: str) -> str | None:
    lowered = filename.lower()
    if "yolov5" in lowered or lowered.startswith("v5"):
        return "yolov5"
    if "yolo26" in lowered or "yolo11" in lowered or "yolov8" in lowered:
        return "ultralytics"
    return None


def _verify_alias_lock(root: Path, spec: ModelSpec) -> None:
    lock_path = root / "models/model-lock.json"
    if not lock_path.is_file():
        raise ModelError(
            f"Model lock not found: {lock_path}. "
            "Run scripts/bootstrap_models.py before using aliases."
        )
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        entry = payload["models"][spec.model_id]
        expected = str(entry["sha256"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelError(f"Invalid model lock: {lock_path}") from exc
    actual = sha256_file(spec.checkpoint)
    if actual != expected:
        raise ModelError(
            f"Checkpoint checksum mismatch for {spec.model_id}: expected {expected}, got {actual}."
        )


class YoloV5Detector:
    family = "yolov5"

    def __init__(
        self,
        spec: ModelSpec,
        device: str,
        input_size: int,
        confidence: float,
        max_detections: int,
        yolov5_repo: str | Path,
        torch_module: Any | None = None,
    ) -> None:
        repo = Path(yolov5_repo).expanduser().resolve()
        if not (repo / "hubconf.py").is_file():
            raise ModelError(
                f"Pinned YOLOv5 repository not found at {repo}. "
                "Install the v7.0 source as described in docs/USER_GUIDE.md."
            )
        version_file = repo / "VERSION"
        if version_file.is_file() and version_file.read_text(encoding="utf-8").strip() != "v7.0":
            raise ModelError(f"Expected YOLOv5 v7.0 source at {repo}.")
        _verify_yolov5_revision(repo, spec.checkpoint)
        try:
            if torch_module is None:
                import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise ModelError("PyTorch is required to load YOLOv5.") from exc

        try:
            self._model = torch_module.hub.load(
                str(repo),
                "custom",
                path=str(spec.checkpoint),
                source="local",
                autoshape=True,
                verbose=False,
            )
            self._model.to(device)
            self._model.conf = confidence
            self._model.iou = 0.45
            self._model.max_det = max_detections
        except Exception as exc:
            raise ModelError(f"Failed to load YOLOv5 checkpoint: {exc}") from exc

        self.model_id = spec.model_id
        self.checkpoint = spec.checkpoint
        self.checkpoint_sha256 = sha256_file(spec.checkpoint)
        self.checkpoint_size_bytes = spec.checkpoint.stat().st_size
        try:
            self.parameter_count = sum(parameter.numel() for parameter in self._model.parameters())
        except (AttributeError, TypeError):
            self.parameter_count = None
        self.device = device
        self.input_size = input_size
        self.confidence = confidence
        self.max_detections = max_detections

    def predict(self, frame: Any, frame_index: int, timestamp_ms: float) -> list[Detection]:
        try:
            results = self._model(frame, size=self.input_size)
            rows = results.xyxy[0].detach().cpu().tolist()
            names = results.names
        except Exception as exc:
            raise ModelError(f"YOLOv5 inference failed: {exc}") from exc
        return [
            Detection(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                confidence=float(row[4]),
                class_id=int(row[5]),
                class_name=str(names[int(row[5])]),
            )
            for row in rows
        ]


class UltralyticsDetector:
    family = "ultralytics"

    def __init__(
        self,
        spec: ModelSpec,
        device: str,
        input_size: int,
        confidence: float,
        max_detections: int,
        yolo_class: Any | None = None,
    ) -> None:
        try:
            if yolo_class is None:
                from ultralytics import YOLO as yolo_class  # type: ignore[no-redef]
        except ImportError as exc:
            raise ModelError(
                "The pinned ultralytics package is required to load YOLO26."
            ) from exc
        try:
            self._model = yolo_class(str(spec.checkpoint))
        except Exception as exc:
            raise ModelError(f"Failed to load Ultralytics checkpoint: {exc}") from exc

        self.model_id = spec.model_id
        self.checkpoint = spec.checkpoint
        self.checkpoint_sha256 = sha256_file(spec.checkpoint)
        self.checkpoint_size_bytes = spec.checkpoint.stat().st_size
        try:
            self.parameter_count = sum(
                parameter.numel() for parameter in self._model.model.parameters()
            )
        except (AttributeError, TypeError):
            self.parameter_count = None
        self.device = device
        self.input_size = input_size
        self.confidence = confidence
        self.max_detections = max_detections

    def predict(self, frame: Any, frame_index: int, timestamp_ms: float) -> list[Detection]:
        try:
            result = self._model.predict(
                source=frame,
                imgsz=self.input_size,
                conf=self.confidence,
                max_det=self.max_detections,
                device=0 if self.device == "cuda" else self.device,
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is None:
                return []
            coordinates = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            names = result.names
        except Exception as exc:
            raise ModelError(f"Ultralytics inference failed: {exc}") from exc

        detections: list[Detection] = []
        for coordinates_row, confidence, class_id_raw in zip(
            coordinates, confidences, class_ids, strict=True
        ):
            class_id = int(class_id_raw)
            detections.append(
                Detection(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    x1=float(coordinates_row[0]),
                    y1=float(coordinates_row[1]),
                    x2=float(coordinates_row[2]),
                    y2=float(coordinates_row[3]),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=str(names[class_id]),
                )
            )
        return detections


def create_detector(
    spec: ModelSpec,
    device: str,
    input_size: int = 640,
    confidence: float = 0.25,
    max_detections: int = 300,
    yolov5_repo: str | Path = "third_party/yolov5-v7.0",
) -> Detector:
    if not 0.0 <= confidence <= 1.0:
        raise ModelError("Confidence threshold must be between 0 and 1.")
    if input_size <= 0 or input_size % 32 != 0:
        raise ModelError("Input size must be a positive multiple of 32.")
    if not 1 <= max_detections <= 5000:
        raise ModelError("Maximum detections must be between 1 and 5,000.")
    if spec.family == "yolov5":
        return YoloV5Detector(
            spec,
            device,
            input_size,
            confidence,
            max_detections,
            yolov5_repo,
        )
    return UltralyticsDetector(spec, device, input_size, confidence, max_detections)


def _verify_yolov5_revision(repo: Path, checkpoint: Path) -> None:
    lock_path = checkpoint.parent / "model-lock.json"
    if not lock_path.is_file():
        return
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        expected = str(payload["yolov5_source"]["commit"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelError(f"Invalid YOLOv5 source entry in {lock_path}.") from exc
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    actual = completed.stdout.strip()
    if completed.returncode != 0 or actual != expected:
        raise ModelError(
            f"YOLOv5 source revision mismatch: expected {expected}, found {actual or 'unknown'}."
        )
