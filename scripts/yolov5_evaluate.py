#!/usr/bin/env python3
"""Run pinned YOLOv5 validation and emit a stable metrics JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=["train", "val", "test"], required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    try:
        from val import run
    except ImportError as exc:
        raise SystemExit(f"Could not import pinned YOLOv5 val.py from {repo}: {exc}")

    results, maps, timing = run(
        data=str(Path(args.data).expanduser().resolve()),
        weights=[str(Path(args.weights).expanduser().resolve())],
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        task=args.task,
        save_json=True,
        plots=True,
        project=output.parent,
        name=output.name,
        exist_ok=True,
    )
    precision, recall, map50, map50_95 = (float(value) for value in results[:4])
    payload = {
        "precision": precision,
        "recall": recall,
        "f1": (2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
        "map50": map50,
        "map50_95": map50_95,
        "per_class_map50_95": [float(value) for value in maps],
        "timing_ms": [float(value) for value in timing],
    }
    (output / "yolov5-metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
