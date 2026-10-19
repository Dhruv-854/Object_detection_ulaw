# Experiment Protocol

## Research question

How do document-aligned YOLOv5s and current YOLO26n compare in accuracy, robustness,
and end-to-end speed before and after fine-tuning on the same five-class domain
dataset?

## Controlled design

1. Freeze the session-grouped 70/15/15 split using seed 42.
2. Audit labels and byte-level duplicates before training.
3. Evaluate unmodified COCO-pretrained checkpoints on the custom test distribution.
4. Fine-tune both models at 640 pixels, batch 16, seed 42, for up to 100 epochs with
   patience 20. Use the same data and split; retain model-native training defaults
   and disclose this as part of the system-level comparison.
5. Select each model's confidence threshold on validation data by maximum F1.
6. Freeze checkpoints and thresholds before opening the test results.
7. Save raw predictions so every aggregate table can be regenerated.

The four required conditions are YOLOv5s pretrained, YOLOv5s fine-tuned, YOLO26n
pretrained, and YOLO26n fine-tuned.

## Metrics

- Precision, recall, F1, mAP@0.5, and mAP@0.5:0.95.
- Per-class AP, confusion matrix, and precision-recall curves.
- Model parameters, checkpoint size, training time, and final epoch.
- End-to-end mean FPS and p50/p95 inference latency after 30 warm-up frames over 300
  measured frames.
- Robustness slices: lighting, distance, occlusion, motion blur, and negative scenes.

Report the hardware, operating system, Python, PyTorch, OpenCV, Ultralytics package,
model checksum, input size, and confidence threshold with every result.

## Error analysis

Review false positives, false negatives, localization errors, class confusion,
small/distant objects, overlap, low light, blur, and empty scenes. Include only newly
generated examples from the frozen test split. Treat 30 FPS at 640 pixels as a
measured target, not an assumed achievement; if it is missed, report both 480- and
640-pixel benchmarks without changing the accuracy experiment.

