from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import DatasetError

CLASS_NAMES = ["bottle", "chair", "cup", "laptop", "cell phone"]
REQUIRED_METADATA_COLUMNS = {
    "image_path",
    "session_id",
    "lighting",
    "distance",
    "occlusion",
    "motion_blur",
    "negative",
}
CONTROLLED_VALUES = {
    "lighting": {"normal", "low"},
    "distance": {"near", "far"},
    "occlusion": {"none", "partial"},
    "motion_blur": {"true", "false"},
    "negative": {"true", "false"},
}


@dataclass(frozen=True, slots=True)
class AuditIssue:
    level: str
    code: str
    message: str
    image_path: str | None = None


@dataclass(slots=True)
class AuditReport:
    image_count: int = 0
    session_count: int = 0
    negative_count: int = 0
    class_instances: dict[str, int] = field(default_factory=dict)
    duplicate_groups: int = 0
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.level == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.level == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["error_count"] = self.error_count
        payload["warning_count"] = self.warning_count
        payload["ready"] = self.error_count == 0
        return payload


def audit_dataset(
    root: str | Path,
    metadata_csv: str | Path,
    strict_targets: bool = True,
) -> AuditReport:
    dataset_root = Path(root).expanduser().resolve()
    metadata_path = Path(metadata_csv).expanduser().resolve()
    if not dataset_root.is_dir():
        raise DatasetError(f"Dataset root not found: {dataset_root}")
    rows = _read_metadata(metadata_path)
    report = AuditReport(image_count=len(rows))
    sessions: set[str] = set()
    instance_counts: Counter[int] = Counter()
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for row in rows:
        image_rel = row["image_path"].strip()
        _audit_metadata_values(row, image_rel, report)
        image_path = (dataset_root / image_rel).resolve()
        if dataset_root not in image_path.parents:
            report.issues.append(
                AuditIssue("error", "path-escape", "Image path leaves dataset root.", image_rel)
            )
            continue
        sessions.add(row["session_id"].strip())
        is_negative = _truthy(row["negative"])
        if is_negative:
            report.negative_count += 1
        if not image_path.is_file():
            report.issues.append(
                AuditIssue("error", "missing-image", "Image file is missing.", image_rel)
            )
            continue

        hashes[_sha256(image_path)].append(image_rel)
        label_path = label_path_for_image(dataset_root, image_path)
        if not label_path.is_file():
            if not is_negative:
                report.issues.append(
                    AuditIssue(
                        "error", "missing-label", "Non-negative image has no label file.", image_rel
                    )
                )
            continue
        labels = _audit_label_file(label_path, image_rel, report)
        if is_negative and labels:
            report.issues.append(
                AuditIssue(
                    "warning",
                    "negative-has-labels",
                    "Image is marked negative but contains annotations.",
                    image_rel,
                )
            )
        instance_counts.update(labels)

    duplicate_sets = [paths for paths in hashes.values() if len(paths) > 1]
    report.duplicate_groups = len(duplicate_sets)
    for paths in duplicate_sets:
        report.issues.append(
            AuditIssue(
                "error",
                "duplicate-image",
                f"Byte-identical images detected: {', '.join(sorted(paths))}",
            )
        )
    report.session_count = len(sessions)
    report.class_instances = {
        name: int(instance_counts.get(index, 0)) for index, name in enumerate(CLASS_NAMES)
    }
    _add_target_issues(report, strict_targets)
    return report


def split_by_session(
    root: str | Path,
    metadata_csv: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    rows = _read_metadata(Path(metadata_csv).expanduser().resolve())
    session_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        session_rows[row["session_id"].strip()].append(row)
    if len(session_rows) < 3:
        raise DatasetError("At least three capture sessions are required for grouped splitting.")

    randomizer = random.Random(seed)
    sessions = list(session_rows)
    randomizer.shuffle(sessions)
    sessions.sort(key=lambda session: len(session_rows[session]), reverse=True)
    targets = {"train": len(rows) * 0.70, "val": len(rows) * 0.15, "test": len(rows) * 0.15}
    assignments: dict[str, str] = {}
    counts = {name: 0 for name in targets}
    for index, session in enumerate(sessions):
        if index < 3:
            split = ("train", "val", "test")[index]
        else:
            split = max(targets, key=lambda name: targets[name] - counts[name])
        assignments[session] = split
        counts[split] += len(session_rows[session])

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    split_dir = destination / "splits"
    split_dir.mkdir(exist_ok=True)
    grouped_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        split = assignments[row["session_id"].strip()]
        row_with_split = dict(row)
        row_with_split["split"] = split
        grouped_rows[split].append(row_with_split)

    fieldnames = list(rows[0]) + ["split"]
    split_metadata_path = destination / "metadata_with_split.csv"
    with split_metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val", "test"):
            writer.writerows(grouped_rows[split])

    for split in ("train", "val", "test"):
        lines = [str((dataset_root / row["image_path"]).resolve()) for row in grouped_rows[split]]
        (split_dir / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_lines = [
        f"path: {dataset_root}",
        f"train: {split_dir / 'train.txt'}",
        f"val: {split_dir / 'val.txt'}",
        f"test: {split_dir / 'test.txt'}",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
    ]
    (destination / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    manifest = {
        "seed": seed,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "image_counts": counts,
        "session_assignments": assignments,
        "dataset_yaml": str(destination / "dataset.yaml"),
    }
    (destination / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def create_robustness_slices(
    root: str | Path,
    metadata_with_split_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    rows = _read_metadata(Path(metadata_with_split_csv).expanduser().resolve())
    if "split" not in rows[0]:
        raise DatasetError("Slice metadata must contain the generated 'split' column.")
    test_rows = [row for row in rows if row["split"].strip().lower() == "test"]
    if not test_rows:
        raise DatasetError("No test rows found in split metadata.")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"slices": {}}
    for field in ("lighting", "distance", "occlusion", "motion_blur", "negative"):
        values = sorted({row[field].strip().lower() for row in test_rows})
        for value in values:
            selected = [row for row in test_rows if row[field].strip().lower() == value]
            slug = _slug(f"{field}-{value}")
            list_path = destination / f"{slug}.txt"
            image_paths = [
                str((dataset_root / row["image_path"]).resolve()) for row in selected
            ]
            list_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
            yaml_path = destination / f"{slug}.yaml"
            yaml_lines = [
                f"path: {dataset_root}",
                f"train: {list_path}",
                f"val: {list_path}",
                f"test: {list_path}",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
            ]
            yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
            manifest["slices"][slug] = {
                "field": field,
                "value": value,
                "images": len(selected),
                "dataset_yaml": str(yaml_path),
            }
    manifest_path = destination / "slice-manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def label_path_for_image(root: Path, image_path: Path) -> Path:
    relative = image_path.resolve().relative_to(root.resolve())
    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return root.joinpath(*parts).with_suffix(".txt")
    return root / "labels" / relative.with_suffix(".txt")


def _read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetError(f"Metadata CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_METADATA_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise DatasetError(f"Metadata CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise DatasetError("Metadata CSV contains no image records.")
    return rows


def _audit_label_file(path: Path, image_rel: str, report: AuditReport) -> list[int]:
    class_ids: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            report.issues.append(
                AuditIssue(
                    "error",
                    "invalid-label-columns",
                    f"Label line {line_number} must contain five values.",
                    image_rel,
                )
            )
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            report.issues.append(
                AuditIssue(
                    "error",
                    "invalid-label-number",
                    f"Invalid number on line {line_number}.",
                    image_rel,
                )
            )
            continue
        if class_id not in range(len(CLASS_NAMES)):
            report.issues.append(
                AuditIssue(
                    "error",
                    "unknown-class",
                    f"Class ID {class_id} on line {line_number} is not defined.",
                    image_rel,
                )
            )
            continue
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            report.issues.append(
                AuditIssue(
                    "error",
                    "out-of-range-box",
                    f"Coordinates on line {line_number} must be normalized to [0, 1].",
                    image_rel,
                )
            )
            continue
        if width <= 0.0 or height <= 0.0:
            report.issues.append(
                AuditIssue(
                    "error",
                    "zero-area-box",
                    f"Width and height on line {line_number} must be positive.",
                    image_rel,
                )
            )
            continue
        class_ids.append(class_id)
    return class_ids


def _audit_metadata_values(
    row: dict[str, str], image_rel: str, report: AuditReport
) -> None:
    if not row["session_id"].strip():
        report.issues.append(
            AuditIssue("error", "missing-session", "Capture session cannot be empty.", image_rel)
        )
    for field, allowed in CONTROLLED_VALUES.items():
        value = row[field].strip().lower()
        if value not in allowed:
            report.issues.append(
                AuditIssue(
                    "error",
                    "invalid-metadata",
                    f"{field}='{value}' is invalid; expected one of {sorted(allowed)}.",
                    image_rel,
                )
            )


def _add_target_issues(report: AuditReport, strict: bool) -> None:
    level = "error" if strict else "warning"
    targets = [
        (report.image_count >= 1000, "image-target", "Dataset must contain at least 1,000 images."),
        (report.session_count >= 20, "session-target", "Dataset must cover at least 20 sessions."),
        (report.negative_count >= 100, "negative-target", "Dataset must include 100 negatives."),
    ]
    for passed, code, message in targets:
        if not passed:
            report.issues.append(AuditIssue(level, code, message))
    for class_name, count in report.class_instances.items():
        if count < 150:
            report.issues.append(
                AuditIssue(
                    level,
                    "class-balance",
                    f"Class '{class_name}' has {count}; target is 150+.",
                )
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")
