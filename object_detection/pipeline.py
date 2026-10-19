from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from .adapters import Detector
from .domain import Detection, RunSummary, percentile
from .errors import ConfigurationError, SourceError
from .sources import SourceSpec, parse_source


@dataclass(slots=True)
class RunOptions:
    source: str
    output_root: Path = Path("outputs")
    save_media: bool = True
    display: bool = False
    max_frames: int | None = None
    warmup_frames: int = 0
    write_detections: bool = True

    def validate(self) -> None:
        if self.max_frames is not None and self.max_frames <= 0:
            raise ConfigurationError("max_frames must be greater than zero.")
        if self.warmup_frames < 0:
            raise ConfigurationError("warmup_frames cannot be negative.")


def run_detection(
    detector: Detector,
    options: RunOptions,
    cv2_module: Any | None = None,
    clock: Any = time.perf_counter,
) -> RunSummary:
    options.validate()
    source = parse_source(options.source)
    cv2 = cv2_module or _import_cv2()
    run_id = _run_id(source, detector.model_id)
    run_dir = options.output_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    detections_path = run_dir / "detections.jsonl"
    summary_path = run_dir / "summary.json"

    detection_stream: TextIO | None = None
    if options.write_detections:
        detection_stream = detections_path.open("w", encoding="utf-8")

    try:
        if source.kind == "image":
            statistics = _process_image(
                detector, source, options, run_dir, detection_stream, cv2, clock
            )
        else:
            statistics = _process_stream(
                detector, source, options, run_dir, detection_stream, cv2, clock
            )
    finally:
        if detection_stream is not None:
            detection_stream.close()

    output_paths = statistics.pop("output_paths")
    if options.write_detections:
        output_paths["detections"] = str(detections_path)
    output_paths["summary"] = str(summary_path)
    latencies = statistics.pop("latencies_ms")
    elapsed = float(statistics.pop("elapsed_seconds"))
    frames = int(statistics.pop("frames_processed"))
    summary = RunSummary(
        run_id=run_id,
        source=source.raw,
        source_kind=source.kind,
        model_id=detector.model_id,
        model_family=detector.family,
        checkpoint=str(detector.checkpoint),
        checkpoint_sha256=detector.checkpoint_sha256,
        checkpoint_size_bytes=getattr(detector, "checkpoint_size_bytes", None),
        parameter_count=getattr(detector, "parameter_count", None),
        device=detector.device,
        input_size=detector.input_size,
        confidence_threshold=detector.confidence,
        max_detections=detector.max_detections,
        frames_processed=frames,
        detections=int(statistics.pop("detection_count")),
        elapsed_seconds=elapsed,
        fps_mean=(frames / elapsed) if elapsed > 0 else 0.0,
        latency_ms_mean=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_ms_p50=percentile(latencies, 0.50),
        latency_ms_p95=percentile(latencies, 0.95),
        warmup_frames=options.warmup_frames,
        output_paths=output_paths,
        environment=_environment_versions(cv2),
    )
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary


def _process_image(
    detector: Detector,
    source: SourceSpec,
    options: RunOptions,
    run_dir: Path,
    stream: TextIO | None,
    cv2: Any,
    clock: Any,
) -> dict[str, Any]:
    frame = cv2.imread(str(source.capture_value))
    if frame is None:
        raise SourceError(f"OpenCV could not decode image: {source.capture_value}")
    start = clock()
    inference_start = clock()
    detections = detector.predict(frame, 0, 0.0)
    latency_ms = (clock() - inference_start) * 1000.0
    height, width = frame.shape[:2]
    detections = _clean_detections(detections, width, height, detector.confidence)
    _write_detections(stream, detections)
    output_paths: dict[str, str] = {}
    if options.save_media:
        annotated = annotate_frame(frame, detections, cv2)
        output_path = run_dir / f"annotated{Path(str(source.capture_value)).suffix.lower()}"
        if not cv2.imwrite(str(output_path), annotated):
            raise SourceError(f"Failed to write annotated image: {output_path}")
        output_paths["media"] = str(output_path)
    if options.display:
        cv2.imshow("Object Detection", annotate_frame(frame, detections, cv2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return {
        "frames_processed": 1,
        "detection_count": len(detections),
        "latencies_ms": [latency_ms],
        "elapsed_seconds": clock() - start,
        "output_paths": output_paths,
    }


def _process_stream(
    detector: Detector,
    source: SourceSpec,
    options: RunOptions,
    run_dir: Path,
    stream: TextIO | None,
    cv2: Any,
    clock: Any,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(source.capture_value)
    if not capture.isOpened():
        capture.release()
        raise SourceError(f"Unable to open {source.kind}: {source.raw}")

    writer = None
    output_paths: dict[str, str] = {}
    frames_processed = 0
    detection_count = 0
    latencies_ms: list[float] = []
    total_seen = 0
    measurement_start: float | None = None
    stream_start = clock()
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        if width <= 0 or height <= 0:
            width = height = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                if source.kind == "video" and total_seen == 0:
                    raise SourceError(f"No decodable frames found in video: {source.raw}")
                break
            if width <= 0 or height <= 0:
                height, width = frame.shape[:2]
            timestamp_ms = (
                float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if source.kind == "video"
                else (clock() - stream_start) * 1000.0
            )
            inference_start = clock()
            is_warmup = total_seen < options.warmup_frames
            if not is_warmup and measurement_start is None:
                measurement_start = inference_start
            detections = detector.predict(frame, total_seen, timestamp_ms)
            latency_ms = (clock() - inference_start) * 1000.0
            detections = _clean_detections(detections, width, height, detector.confidence)

            total_seen += 1
            if is_warmup:
                continue
            frames_processed += 1
            detection_count += len(detections)
            latencies_ms.append(latency_ms)
            _write_detections(stream, detections)

            annotated = None
            if writer is not None or options.display:
                annotated = annotate_frame(frame, detections, cv2)
            if options.save_media and writer is None:
                output_path = run_dir / "annotated.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise SourceError(f"Unable to create video output: {output_path}")
                output_paths["media"] = str(output_path)
                if annotated is None:
                    annotated = annotate_frame(frame, detections, cv2)
            if writer is not None:
                writer.write(annotated)
            if options.display:
                cv2.imshow("Object Detection - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if options.max_frames is not None and frames_processed >= options.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if options.display:
            cv2.destroyAllWindows()

    if frames_processed == 0:
        raise SourceError(
            "The source ended before any measured frames were processed. "
            "Reduce --warmup-frames or use a longer source."
        )
    elapsed = clock() - (measurement_start if measurement_start is not None else stream_start)
    return {
        "frames_processed": frames_processed,
        "detection_count": detection_count,
        "latencies_ms": latencies_ms,
        "elapsed_seconds": elapsed,
        "output_paths": output_paths,
    }


def _clean_detections(
    detections: list[Detection], width: int, height: int, threshold: float
) -> list[Detection]:
    cleaned: list[Detection] = []
    for detection in detections:
        clipped = detection.clipped(width, height)
        if clipped.valid_box and clipped.confidence >= threshold:
            cleaned.append(clipped)
    return cleaned


def annotate_frame(frame: Any, detections: list[Detection], cv2: Any) -> Any:
    annotated = frame.copy()
    for detection in detections:
        color = _class_color(detection.class_id)
        start = (int(detection.x1), int(detection.y1))
        end = (int(detection.x2), int(detection.y2))
        cv2.rectangle(annotated, start, end, color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        baseline_y = max(18, int(detection.y1) - 7)
        cv2.putText(
            annotated,
            label,
            (int(detection.x1), baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _class_color(class_id: int) -> tuple[int, int, int]:
    return (
        60 + (class_id * 67) % 196,
        60 + (class_id * 109) % 196,
        60 + (class_id * 149) % 196,
    )


def _write_detections(stream: TextIO | None, detections: list[Detection]) -> None:
    if stream is None:
        return
    for detection in detections:
        stream.write(json.dumps(detection.to_dict(), separators=(",", ":")) + "\n")
    stream.flush()


def _run_id(source: SourceSpec, model_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_source = "".join(character if character.isalnum() else "-" for character in source.name)
    return f"{timestamp}-{safe_source}-{model_id}"


def _environment_versions(cv2: Any) -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": str(getattr(cv2, "__version__", "unknown")),
    }
    try:
        import torch

        versions["torch"] = str(torch.__version__)
    except ImportError:
        versions["torch"] = "not-installed"
    try:
        import ultralytics

        versions["ultralytics"] = str(ultralytics.__version__)
    except ImportError:
        versions["ultralytics"] = "not-installed"
    return versions


def _import_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ConfigurationError(
            "OpenCV is not installed. Install the locked dependencies before running detection."
        ) from exc
