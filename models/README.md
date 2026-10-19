# Model checkpoints

Checkpoints are intentionally excluded from Git. Register locally obtained
official `yolov5s` v7.0 and `yolo26n` checkpoints using
`scripts/bootstrap_models.py`. The command copies the files to this directory and
creates an ignored `model-lock.json` containing their SHA-256 checksums.

Runtime model aliases refuse to load if that lock is absent or a checksum differs.
This prevents silent use of a replaced or newly downloaded checkpoint.

