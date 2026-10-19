# Ten-Week Delivery Plan

This checklist separates implemented infrastructure from work that depends on
ethical approval, original data, training hardware, and measured results.

## Week 1 — governance and reproducibility

- [x] Research question and comparative model strategy defined.
- [x] Ethics and privacy protocol documented.
- [x] Dependency, checkpoint, seed, and environment recording designed.
- [ ] Confirm the university ethical-approval requirement with the supervisor.

## Week 2 — baseline application

- [x] Shared image/video/webcam pipeline and CLI implemented.
- [x] Local-only YOLOv5 adapter, annotated output, JSONL, and run summary implemented.
- [x] Source, configuration, and device errors made actionable.

## Week 3 — current model and tests

- [x] YOLO26 adapter and common detection schema implemented.
- [x] Checksum lock and explicit CUDA → MPS → CPU selection implemented.
- [x] Model-free automated unit/integration suite implemented.
- [ ] Register official checkpoints and run the first hardware-backed smoke tests.

## Week 4 — pilot dataset

- [x] Five-class schema, metadata template, label audit, duplicate detection, and
  grouped splitter implemented.
- [ ] Collect a small privacy-reviewed pilot across at least three sessions.
- [ ] Resolve annotation-rule disagreements before scaling collection.

## Week 5 — final dataset

- [ ] Collect and review 1,000 images across at least 20 sessions.
- [ ] Include 100 negatives and meet the 150-instance minimum for every class.
- [ ] Pass the strict audit and freeze the seed-42 split manifest.

## Week 6 — training

- [x] Fixed training commands and environment capture implemented.
- [ ] Fine-tune YOLOv5s and YOLO26n; retain complete logs and best checkpoints.

## Week 7 — frozen evaluation

- [x] Machine-readable evaluation summaries and comparison-table generator implemented.
- [ ] Select thresholds on validation data, freeze decisions, then evaluate test once.

## Week 8 — performance and errors

- [x] 30-warm-up/300-frame benchmark with p50/p95 latency implemented.
- [ ] Run CPU and accelerator benchmarks at 480 and 640 pixels.
- [ ] Complete robustness slices and qualitative error review from original test data.

## Week 9 — packaging

- [x] User guide, dataset/model cards, ethics protocol, experiment protocol, and
  report outline created.
- [ ] Repeat installation on a clean Python 3.11 environment and archive evidence.
- [ ] Complete final dataset and model cards with verified values.

## Week 10 — assessment evidence

- [ ] Generate every table, curve, confusion matrix, and screenshot from frozen runs.
- [ ] Complete the technical report and trace each empirical claim to a run artifact.
- [ ] Rehearse image, video, webcam, error-handling, and reproducibility demonstrations.

