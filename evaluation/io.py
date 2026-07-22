"""JSONL prediction and YOLO ground-truth I/O."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Box, PredictionRecord


def _clamp_roundoff(value: float, maximum: float, tolerance: float = 1e-3) -> float:
    if -tolerance <= value < 0:
        return 0.0
    if maximum < value <= maximum + tolerance:
        return maximum
    return value


def read_prediction_jsonl(path: Path) -> list[PredictionRecord]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(PredictionRecord.from_dict(value))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid prediction at {path}:{line_number}: {error}"
                ) from error
    return records


def append_prediction_jsonl(path: Path, record: PredictionRecord) -> None:
    """Append one validated record and flush it for interruption safety."""
    record.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        destination.flush()


def load_yolo_ground_truth(
    records: list[PredictionRecord],
    dataset_dir: Path,
    split: str = "test",
    class_names: dict[int, str] | None = None,
) -> dict[str, tuple[Box, ...]]:
    """Load normalized YOLO labels in each record's pixel coordinate system."""
    class_names = class_names or {0: "person"}
    ground_truth = {}
    for record in records:
        label_path = dataset_dir / "labels" / split / f"{record.image_id}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"ground-truth label not found: {label_path}")
        boxes = []
        for line_number, line in enumerate(
            label_path.read_text().splitlines(), start=1
        ):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"malformed YOLO label at {label_path}:{line_number}")
            class_id = int(fields[0])
            if class_id not in class_names:
                raise ValueError(
                    f"unknown class ID {class_id} at {label_path}:{line_number}"
                )
            x_center, y_center, width, height = map(float, fields[1:])
            x1 = (x_center - width / 2) * record.image_width
            y1 = (y_center - height / 2) * record.image_height
            x2 = (x_center + width / 2) * record.image_width
            y2 = (y_center + height / 2) * record.image_height
            box = Box(
                x1=_clamp_roundoff(x1, record.image_width),
                y1=_clamp_roundoff(y1, record.image_height),
                x2=_clamp_roundoff(x2, record.image_width),
                y2=_clamp_roundoff(y2, record.image_height),
                label=class_names[class_id],
            )
            box.validate(record.image_width, record.image_height)
            boxes.append(box)
        ground_truth[record.image_id] = tuple(boxes)
    return ground_truth
