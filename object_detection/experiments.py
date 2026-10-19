from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters import ModelSpec
from .errors import ConfigurationError, ModelError

TRAINING_DEFAULTS = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "seed": 42,
    "patience": 20,
}


def train_model(
    spec: ModelSpec,
    data: str | Path,
    device: str,
    output_root: str | Path = "runs/train",
    yolov5_repo: str | Path = "third_party/yolov5-v7.0",
) -> dict[str, Any]:
    data_path = _validate_dataset_yaml(data)
    run_dir = _new_experiment_dir(output_root, f"{spec.model_id}-seed42")
    _write_environment(run_dir, spec, data_path, device, "train")
    started = time.perf_counter()

    if spec.family == "yolov5":
        repo = _validate_yolov5_repo(yolov5_repo)
        command = [
            sys.executable,
            str(repo / "train.py"),
            "--weights",
            str(spec.checkpoint),
            "--data",
            str(data_path),
            "--epochs",
            str(TRAINING_DEFAULTS["epochs"]),
            "--batch-size",
            str(TRAINING_DEFAULTS["batch"]),
            "--imgsz",
            str(TRAINING_DEFAULTS["imgsz"]),
            "--patience",
            str(TRAINING_DEFAULTS["patience"]),
            "--seed",
            str(TRAINING_DEFAULTS["seed"]),
            "--device",
            "0" if device == "cuda" else device,
            "--project",
            str(run_dir.parent),
            "--name",
            run_dir.name,
            "--exist-ok",
        ]
        _run_checked(command, cwd=repo, log_path=run_dir / "command.log")
    else:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError("Install the locked Ultralytics dependency before training.") from exc
        model = YOLO(str(spec.checkpoint))
        model.train(
            data=str(data_path),
            device=0 if device == "cuda" else device,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            **TRAINING_DEFAULTS,
        )
    result = {
        "status": "completed",
        "run_dir": str(run_dir),
        "training_duration_seconds": time.perf_counter() - started,
        **TRAINING_DEFAULTS,
    }
    (run_dir / "run-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_model(
    spec: ModelSpec,
    data: str | Path,
    split: str,
    device: str,
    output_root: str | Path = "runs/evaluate",
    yolov5_repo: str | Path = "third_party/yolov5-v7.0",
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ConfigurationError("Evaluation split must be train, val, or test.")
    data_path = _validate_dataset_yaml(data)
    run_dir = _new_experiment_dir(output_root, f"{spec.model_id}-{split}")
    _write_environment(run_dir, spec, data_path, device, "evaluate")
    started = time.perf_counter()

    if spec.family == "yolov5":
        repo = _validate_yolov5_repo(yolov5_repo)
        task = split
        wrapper = Path(__file__).resolve().parents[1] / "scripts/yolov5_evaluate.py"
        command = [
            sys.executable,
            str(wrapper),
            "--repo",
            str(repo),
            "--weights",
            str(spec.checkpoint),
            "--data",
            str(data_path),
            "--task",
            task,
            "--imgsz",
            str(TRAINING_DEFAULTS["imgsz"]),
            "--batch-size",
            str(TRAINING_DEFAULTS["batch"]),
            "--device",
            "0" if device == "cuda" else device,
            "--output",
            str(run_dir),
        ]
        _run_checked(command, cwd=repo, log_path=run_dir / "command.log")
        metrics_path = run_dir / "yolov5-metrics.json"
        if not metrics_path.is_file():
            raise ModelError(f"YOLOv5 evaluation did not produce metrics: {metrics_path}")
        metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError(
                "Install the locked Ultralytics dependency before evaluation."
            ) from exc
        model = YOLO(str(spec.checkpoint))
        results = model.val(
            data=str(data_path),
            split=split,
            imgsz=TRAINING_DEFAULTS["imgsz"],
            batch=TRAINING_DEFAULTS["batch"],
            device=0 if device == "cuda" else device,
            plots=True,
            save_json=True,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
        )
        box = results.box
        precision = float(box.mp)
        recall = float(box.mr)
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "per_class_map50_95": [float(value) for value in box.maps],
        }
    result = {
        "status": "completed",
        "run_dir": str(run_dir),
        "split": split,
        "model_id": spec.model_id,
        "evaluation_duration_seconds": time.perf_counter() - started,
        **metrics,
    }
    (run_dir / "metrics-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _validate_dataset_yaml(data: str | Path) -> Path:
    path = Path(data).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError(f"Dataset YAML not found: {path}")
    return path


def _validate_yolov5_repo(repo: str | Path) -> Path:
    path = Path(repo).expanduser().resolve()
    if not (path / "train.py").is_file() or not (path / "val.py").is_file():
        raise ModelError(f"Pinned YOLOv5 v7.0 training source not found at {path}.")
    return path


def _new_experiment_dir(output_root: str | Path, label: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_root).expanduser().resolve() / f"{timestamp}-{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_environment(
    run_dir: Path, spec: ModelSpec, data_path: Path, device: str, action: str
) -> None:
    payload = {
        "action": action,
        "python": sys.version,
        "platform": platform.platform(),
        "model_id": spec.model_id,
        "model_family": spec.family,
        "checkpoint": str(spec.checkpoint),
        "dataset": str(data_path),
        "device": device,
        "defaults": TRAINING_DEFAULTS,
    }
    try:
        import torch

        payload["torch"] = str(torch.__version__)
    except ImportError:
        payload["torch"] = "not-installed"
    try:
        import ultralytics

        payload["ultralytics"] = str(ultralytics.__version__)
    except ImportError:
        payload["ultralytics"] = "not-installed"
    (run_dir / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_checked(command: list[str], cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise ModelError(
            f"Experiment command failed with exit code {completed.returncode}. "
            f"See {log_path}."
        )


def _f1(precision: float, recall: float) -> float:
    return (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
