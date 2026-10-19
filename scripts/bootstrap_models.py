#!/usr/bin/env python3
"""One-time, explicit model acquisition and checksum locking.

Inference never downloads weights or source code implicitly. Download the two
checkpoints from the official Ultralytics release, then register them here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from object_detection.domain import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--yolov5-checkpoint", required=True)
    parser.add_argument("--yolo26-checkpoint", required=True)
    parser.add_argument(
        "--clone-yolov5",
        action="store_true",
        help="Clone the exact YOLOv5 v7.0 tag into third_party/yolov5-v7.0.",
    )
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.clone_yolov5:
        repo = root / "third_party/yolov5-v7.0"
        if repo.exists():
            raise SystemExit(f"Refusing to overwrite existing path: {repo}")
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "v7.0",
                "--depth",
                "1",
                "https://github.com/ultralytics/yolov5.git",
                str(repo),
            ],
            check=True,
        )
        (repo / "VERSION").write_text("v7.0\n", encoding="utf-8")
    else:
        repo = root / "third_party/yolov5-v7.0"
    if not (repo / "hubconf.py").is_file():
        raise SystemExit(
            f"Pinned YOLOv5 source not found at {repo}; add --clone-yolov5 or clone v7.0."
        )
    tag = subprocess.run(
        ["git", "-C", str(repo), "describe", "--tags", "--exact-match"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tag != "v7.0":
        raise SystemExit(f"Expected exact YOLOv5 tag v7.0, found {tag or 'none'}.")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    requested = {
        "yolov5s": (
            Path(args.yolov5_checkpoint).expanduser().resolve(),
            models_dir / "yolov5s-v7.0.pt",
            "Ultralytics YOLOv5 v7.0 checkpoint",
        ),
        "yolo26n": (
            Path(args.yolo26_checkpoint).expanduser().resolve(),
            models_dir / "yolo26n.pt",
            "Ultralytics YOLO26 nano checkpoint",
        ),
    }
    entries: dict[str, dict[str, str]] = {}
    for model_id, (source, destination, description) in requested.items():
        if not source.is_file():
            raise SystemExit(f"Checkpoint not found: {source}")
        if source != destination:
            shutil.copy2(source, destination)
        entries[model_id] = {
            "description": description,
            "path": str(destination.relative_to(root)),
            "sha256": sha256_file(destination),
        }

    lock = {
        "created_utc": datetime.now(UTC).isoformat(),
        "models": entries,
        "yolov5_source": {
            "repository": "https://github.com/ultralytics/yolov5.git",
            "tag": "v7.0",
            "commit": commit,
            "path": "third_party/yolov5-v7.0",
        },
    }
    lock_path = models_dir / "model-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print(lock_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
