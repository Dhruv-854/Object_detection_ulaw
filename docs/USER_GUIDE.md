# User Guide

## 1. Install

Use Python 3.11 in a clean virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate`. PyTorch installation is
platform-specific; if the locked installation cannot select the correct CUDA build,
install the matching build from the official PyTorch selector and record its exact
version in the experiment notes.

## 2. Register fixed models

Obtain `yolov5s.pt` from the official YOLOv5 v7.0 release and `yolo26n.pt` using
the pinned Ultralytics package and official YOLO26 documentation. Keep the files in
a staging directory, then run:

```bash
python scripts/bootstrap_models.py \
  --clone-yolov5 \
  --yolov5-checkpoint /path/to/yolov5s.pt \
  --yolo26-checkpoint /path/to/yolo26n.pt
```

This clones only the `v7.0` YOLOv5 tag, copies the two checkpoints, and writes a
checksum lock. Detection never downloads models automatically. Commit neither the
weights nor `model-lock.json` if the assessment submission excludes large binaries;
archive the lock with the experimental evidence instead.

## 3. Detect objects

Image:

```bash
python -m object_detection detect --source samples/desk.jpg --model yolo26n
```

Video:

```bash
python -m object_detection detect --source samples/test.mp4 --model yolov5s
```

Webcam (frames are not saved by default):

```bash
python -m object_detection detect --source 0 --model yolo26n --display
```

Add `--save` to retain webcam video only when consent and the ethics protocol allow
it. Press `q` to end a displayed stream. Use `--device cpu`, `--device mps`, or
`--device cuda` to override automatic CUDA → MPS → CPU selection.

Each run creates a timestamped directory containing:

- `annotated.*` when media saving is enabled;
- `detections.jsonl`, with one normalized detection per line;
- `summary.json`, with model checksum, device, environment, counts, FPS, and latency.

### Detect more objects in a busy frame

The pretrained models already cover the 80 COCO classes. To retain weaker and
smaller detections, lower the confidence threshold, increase the input size, and
raise the per-frame detection limit:

```bash
python -m object_detection detect \
  --source "/full/path/to/busy-scene.jpg" \
  --model yolo26n \
  --conf 0.10 \
  --imgsz 960 \
  --max-det 1000
```

For webcam use, start with `--conf 0.15 --imgsz 832 --max-det 500`. Larger images
and lower thresholds can find more small objects but reduce FPS and increase false
positives. Keep the default `--conf 0.25 --imgsz 640` for formal evaluation unless
the threshold is selected on validation data.

## 4. Prepare and validate the dataset

Expected layout:

```text
data/custom/
├── images/
│   ├── image-0001.jpg
│   └── ...
├── labels/
│   ├── image-0001.txt
│   └── ...
└── metadata.csv
```

Copy `data/metadata-template.csv` to the dataset root. Use class IDs in this fixed
order: `0 bottle`, `1 chair`, `2 cup`, `3 laptop`, `4 cell phone`.

Audit a pilot dataset without enforcing final quantity targets:

```bash
python -m object_detection dataset-audit \
  --root data/custom --metadata data/custom/metadata.csv --pilot
```

For the final audit, remove `--pilot`; any missing final target becomes an error.
Create the session-grouped split only after the audit is clean:

```bash
python -m object_detection dataset-split \
  --root data/custom \
  --metadata data/custom/metadata.csv \
  --output data/processed
```

Generate test-only robustness manifests after the split is frozen:

```bash
python -m object_detection dataset-slices \
  --root data/custom \
  --metadata data/processed/metadata_with_split.csv \
  --output data/processed/slices
```

Run `evaluate` against each non-negative slice YAML. For the negative-only slice,
use detection JSONL to report false detections per image because AP is undefined
when there are no ground-truth objects.

## 5. Train and evaluate

```bash
python -m object_detection train \
  --model yolov5s --data data/processed/dataset.yaml

python -m object_detection train \
  --model yolo26n --data data/processed/dataset.yaml

python -m object_detection evaluate \
  --model /path/to/frozen/best.pt --family ultralytics \
  --data data/processed/dataset.yaml --split test
```

Use the validation split for confidence-threshold selection. Do not run the test
command until model and threshold choices are frozen.

## 6. Benchmark

```bash
python -m object_detection benchmark \
  --model yolo26n --source samples/test.mp4 \
  --warmup-frames 30 --frames 300
```

Repeat on the same source and hardware at `--imgsz 480` and `--imgsz 640`. Report
both results if the measured 640-pixel run does not achieve 30 FPS.

## 7. Build the four-condition comparison table

After the four frozen evaluations finish, generate the report-ready CSV and its
machine-readable provenance file:

```bash
python -m object_detection compare \
  --condition v5-pretrained=runs/evaluate/v5-pretrained/metrics-summary.json \
  --condition v5-finetuned=runs/evaluate/v5-finetuned/metrics-summary.json \
  --condition yolo26-pretrained=runs/evaluate/yolo26-pretrained/metrics-summary.json \
  --condition yolo26-finetuned=runs/evaluate/yolo26-finetuned/metrics-summary.json
```

Do not manually retype these values into the report; regenerate the comparison
artifact when a source run changes.

## Troubleshooting

- **Checkpoint not found/lock mismatch:** rerun the registration command with the
  intended official files; do not bypass the checksum check.
- **Camera unavailable:** close applications using the camera and verify its index.
- **MPS/CUDA unavailable:** use `--device cpu` or install the correct PyTorch build.
- **Video cannot be written:** verify OpenCV codec support and output permissions.
- **No measured frames:** use a longer benchmark video or fewer warm-up frames.
