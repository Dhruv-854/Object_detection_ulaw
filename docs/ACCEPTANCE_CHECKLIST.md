# Acceptance Checklist

## Software

- [ ] Both checksum-locked official checkpoints load without network access.
- [ ] Image detection writes annotated media, JSONL, and summary JSON.
- [ ] MP4 detection preserves frame order and writes playable output.
- [ ] Webcam inference displays live results and saves no frames by default.
- [ ] Missing/corrupt sources, cameras, models, and invalid devices fail clearly.
- [ ] Automated tests pass in the submitted environment.

## Data and experiments

- [ ] Strict audit has zero errors; final counts are copied from the audit JSON.
- [ ] No capture session or byte-identical image crosses splits.
- [ ] Thresholds are selected on validation data before the test split is opened.
- [ ] All four model conditions have frozen metrics summaries.
- [ ] CPU and available-accelerator benchmarks use 30 warm-up and 300 measured frames.
- [ ] Any missed 30-FPS target is reported with both 480- and 640-pixel results.

## Ethics and documentation

- [ ] Ethical approval/decision is documented before identifiable-person data use.
- [ ] Dataset contains no unreviewed personal or sensitive information.
- [ ] Licences, permissions, retention, bias, and limitations are documented.
- [ ] User guide succeeds from a clean Python 3.11 environment.
- [ ] Dataset card, two fine-tuned model cards, report, and raw result archive are complete.
- [ ] No screenshot, metric, authorship, or first-person claim is copied from the
  supplied reference report.

