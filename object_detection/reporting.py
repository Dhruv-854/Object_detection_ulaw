from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

REQUIRED_METRICS = ("precision", "recall", "f1", "map50", "map50_95")


def build_comparison(
    conditions: list[str], output_dir: str | Path = "artifacts/results/comparison"
) -> dict[str, Any]:
    if not conditions:
        raise ConfigurationError("Provide at least one LABEL=metrics-summary.json condition.")
    rows: list[dict[str, Any]] = []
    labels: set[str] = set()
    for condition in conditions:
        if "=" not in condition:
            raise ConfigurationError(f"Invalid condition '{condition}'; expected LABEL=PATH.")
        label, path_raw = condition.split("=", 1)
        label = label.strip()
        if not label or label in labels:
            raise ConfigurationError(f"Condition label must be non-empty and unique: '{label}'.")
        labels.add(label)
        path = Path(path_raw).expanduser().resolve()
        if not path.is_file():
            raise ConfigurationError(f"Metrics file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid metrics JSON: {path}") from exc
        missing = [metric for metric in REQUIRED_METRICS if metric not in payload]
        if missing:
            raise ConfigurationError(f"Metrics file {path} is missing {missing}.")
        rows.append(
            {
                "condition": label,
                **{metric: float(payload[metric]) for metric in REQUIRED_METRICS},
                "source": str(path),
            }
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "model-comparison.csv"
    json_path = destination / "model-comparison.json"
    fields = ("condition", *REQUIRED_METRICS, "source")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {"conditions": rows, "csv": str(csv_path), "json": str(json_path)}
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

