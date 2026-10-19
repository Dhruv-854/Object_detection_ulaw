# MSc Object Detection

A reproducible Python application and research workflow for comparing the
document-aligned YOLOv5s baseline with the current YOLO26n detector. The same
interface processes images, MP4/video files, and webcam streams, while every run
records detections, timing, checkpoint identity, hardware, and configuration.

## What is implemented

- `detect`, `benchmark`, `train`, `evaluate`, and evidence-table commands.
- Local-only, checksum-locked model loading; inference never downloads weights.
- Separate YOLOv5 NMS and Ultralytics end-to-end model adapters.
- Annotated media, JSONL detections, and reproducibility-focused summary JSON.
- Dataset metadata audit, YOLO-label validation, duplicate detection, and grouped
  70/15/15 splitting by capture session.
- Fixed 640-pixel, 100-epoch, batch-16, seed-42 fine-tuning protocol.
- Automated unit and integration tests that do not need model downloads.
- Ethics, dataset-card, model-card, experiment, user-guide, and report templates.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
python -m object_detection --help
```

Model acquisition is an explicit setup operation. Follow
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) to register both official checkpoints
and create `models/model-lock.json`.

```bash
python -m object_detection detect --source example.jpg --model yolo26n --device auto
python -m object_detection detect --source 0 --model yolov5s --display
python -m object_detection benchmark --source test.mp4 --model yolo26n
```

## Research boundary

The repository contains the application and the full reproducible experiment
workflow. It deliberately contains no invented dataset, trained model, benchmark,
confusion matrix, or claim of achieved accuracy. Those artifacts are produced only
after the ethically approved data collection and frozen-test evaluation described
in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Tests

The core suite uses the standard library and fakes for OpenCV/model behavior:

```bash
python -m unittest discover -s tests -v
```

With development dependencies installed:

```bash
pytest --cov=object_detection
ruff check object_detection tests scripts
```
