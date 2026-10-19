from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class Detection:
    frame_index: int
    timestamp_ms: float
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float

    def clipped(self, width: int, height: int) -> "Detection":
        """Return a copy bounded to the current frame."""
        return Detection(
            frame_index=self.frame_index,
            timestamp_ms=self.timestamp_ms,
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=max(0.0, min(1.0, self.confidence)),
        )

    @property
    def valid_box(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunSummary:
    run_id: str
    source: str
    source_kind: str
    model_id: str
    model_family: str
    checkpoint: str
    checkpoint_sha256: str | None
    checkpoint_size_bytes: int | None
    parameter_count: int | None
    device: str
    input_size: int
    confidence_threshold: float
    max_detections: int
    frames_processed: int
    detections: int
    elapsed_seconds: float
    fps_mean: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    warmup_frames: int = 0
    output_paths: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
