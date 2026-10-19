from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import SourceError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


@dataclass(frozen=True, slots=True)
class SourceSpec:
    raw: str
    kind: str
    capture_value: str | int
    name: str


def parse_source(raw: str) -> SourceSpec:
    value = str(raw).strip()
    if not value:
        raise SourceError("Source cannot be empty.")
    if value.isdigit():
        index = int(value)
        return SourceSpec(raw=value, kind="camera", capture_value=index, name=f"camera-{index}")

    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise SourceError(f"Source does not exist: {path}")
    if not path.is_file():
        raise SourceError(f"Source must be a file or camera index: {path}")
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return SourceSpec(raw=value, kind="image", capture_value=str(path), name=path.stem)
    if suffix in VIDEO_SUFFIXES:
        return SourceSpec(raw=value, kind="video", capture_value=str(path), name=path.stem)
    raise SourceError(
        f"Unsupported source format '{suffix or '<none>'}'. "
        f"Supported images: {sorted(IMAGE_SUFFIXES)}; videos: {sorted(VIDEO_SUFFIXES)}."
    )
