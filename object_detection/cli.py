from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import create_detector, resolve_model_spec
from .dataset import audit_dataset, create_robustness_slices, split_by_session
from .devices import select_device
from .errors import ObjectDetectionError
from .experiments import evaluate_model, train_model
from .pipeline import RunOptions, run_detection
from .reporting import build_comparison
from .sources import parse_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="object-detection",
        description="Reproducible YOLOv5/YOLO26 object-detection research application.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Detect objects in an image, video, or camera.")
    _add_model_arguments(detect)
    detect.add_argument(
        "--source", required=True, help="Image/video path or webcam index such as 0."
    )
    detect.add_argument(
        "--output", default="outputs", help="Directory for timestamped run outputs."
    )
    detect.add_argument(
        "--display", action="store_true", help="Display frames in an OpenCV window."
    )
    detect.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save annotated media. Defaults on for files and off for webcams.",
    )
    detect.add_argument("--max-frames", type=int, default=None)

    benchmark = subparsers.add_parser("benchmark", help="Measure end-to-end inference speed.")
    _add_model_arguments(benchmark)
    benchmark.add_argument("--source", required=True, help="Video path or webcam index.")
    benchmark.add_argument("--output", default="runs/benchmark")
    benchmark.add_argument("--warmup-frames", type=int, default=30)
    benchmark.add_argument("--frames", type=int, default=300)

    train = subparsers.add_parser("train", help="Fine-tune a model with the fixed protocol.")
    _add_model_arguments(train, include_threshold=False)
    train.add_argument("--data", required=True, help="YOLO dataset YAML.")
    train.add_argument("--output", default="runs/train")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a frozen checkpoint.")
    _add_model_arguments(evaluate, include_threshold=False)
    evaluate.add_argument("--data", required=True, help="YOLO dataset YAML.")
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--output", default="runs/evaluate")

    audit = subparsers.add_parser("dataset-audit", help="Validate images, labels, and metadata.")
    audit.add_argument("--root", required=True, help="Dataset root containing images and labels.")
    audit.add_argument("--metadata", required=True, help="Dataset metadata CSV.")
    audit.add_argument("--output", default=None, help="Optional JSON report path.")
    audit.add_argument(
        "--pilot", action="store_true", help="Treat final size/session targets as warnings."
    )

    split = subparsers.add_parser("dataset-split", help="Create a leakage-safe session split.")
    split.add_argument("--root", required=True)
    split.add_argument("--metadata", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)

    slices = subparsers.add_parser(
        "dataset-slices", help="Create test manifests for robustness evaluation."
    )
    slices.add_argument("--root", required=True)
    slices.add_argument("--metadata", required=True, help="metadata_with_split.csv")
    slices.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare", help="Build a reproducible metrics table.")
    compare.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Repeat LABEL=/path/to/metrics-summary.json for each model condition.",
    )
    compare.add_argument("--output", default="artifacts/results/comparison")
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser, include_threshold: bool = True) -> None:
    parser.add_argument("--model", required=True, help="yolov5s, yolo26n, or a local checkpoint.")
    parser.add_argument("--family", choices=["yolov5", "ultralytics"], default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--yolov5-repo", default="third_party/yolov5-v7.0")
    if include_threshold:
        parser.add_argument("--conf", type=float, default=0.25)
        parser.add_argument(
            "--max-det",
            type=int,
            default=300,
            help="Maximum objects retained per frame (1-5000).",
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"detect", "benchmark", "train", "evaluate"}:
            device = select_device(args.device)
            spec = resolve_model_spec(args.model, args.family, args.project_root)
        if args.command == "detect":
            source = parse_source(args.source)
            save_media = args.save if args.save is not None else source.kind != "camera"
            detector = create_detector(
                spec,
                device,
                args.imgsz,
                args.conf,
                args.max_det,
                yolov5_repo=args.yolov5_repo,
            )
            summary = run_detection(
                detector,
                RunOptions(
                    source=args.source,
                    output_root=Path(args.output),
                    save_media=save_media,
                    display=args.display,
                    max_frames=args.max_frames,
                ),
            )
            _print_json(summary.to_dict())
        elif args.command == "benchmark":
            if parse_source(args.source).kind == "image":
                raise ObjectDetectionError("Benchmark source must be a video or camera.")
            detector = create_detector(
                spec,
                device,
                args.imgsz,
                args.conf,
                args.max_det,
                yolov5_repo=args.yolov5_repo,
            )
            summary = run_detection(
                detector,
                RunOptions(
                    source=args.source,
                    output_root=Path(args.output),
                    save_media=False,
                    display=False,
                    max_frames=args.frames,
                    warmup_frames=args.warmup_frames,
                    write_detections=False,
                ),
            )
            _print_json(summary.to_dict())
        elif args.command == "train":
            _print_json(
                train_model(spec, args.data, device, args.output, args.yolov5_repo)
            )
        elif args.command == "evaluate":
            _print_json(
                evaluate_model(
                    spec, args.data, args.split, device, args.output, args.yolov5_repo
                )
            )
        elif args.command == "dataset-audit":
            report = audit_dataset(args.root, args.metadata, strict_targets=not args.pilot)
            payload = report.to_dict()
            if args.output:
                output_path = Path(args.output).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _print_json(payload)
            return 2 if report.error_count else 0
        elif args.command == "dataset-split":
            _print_json(split_by_session(args.root, args.metadata, args.output, args.seed))
        elif args.command == "dataset-slices":
            _print_json(create_robustness_slices(args.root, args.metadata, args.output))
        elif args.command == "compare":
            _print_json(build_comparison(args.condition, args.output))
        return 0
    except ObjectDetectionError as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 2


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
