"""Convert Ultralytics Results objects to the unified prediction schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import Box, PredictionRecord


def _tensor_values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def yolo_result_to_record(
    result: Any,
    *,
    run_id: str,
    modality: str,
    model_id: str,
    model_revision: str,
    metadata: dict[str, Any],
    image_id: str | None = None,
) -> PredictionRecord:
    height, width = map(int, result.orig_shape)
    xyxy = _tensor_values(result.boxes.xyxy)
    confidence = _tensor_values(result.boxes.conf)
    class_ids = _tensor_values(result.boxes.cls)
    if not (len(xyxy) == len(confidence) == len(class_ids)):
        raise ValueError("Ultralytics result arrays have inconsistent lengths")
    boxes = tuple(
        Box(
            x1=float(coordinates[0]),
            y1=float(coordinates[1]),
            x2=float(coordinates[2]),
            y2=float(coordinates[3]),
            label=str(result.names[int(class_id)]),
            confidence=float(score),
        )
        for coordinates, score, class_id in zip(xyxy, confidence, class_ids)
    )
    speed = getattr(result, "speed", {}) or {}
    record = PredictionRecord(
        run_id=run_id,
        image_id=image_id or Path(result.path).stem,
        modality=modality,
        model_id=model_id,
        model_revision=model_revision,
        image_width=width,
        image_height=height,
        boxes=boxes,
        status="ok",
        latency_ms=sum(float(value) for value in speed.values()),
        metadata={**metadata, "ultralytics_speed_ms": dict(speed)},
    )
    record.validate()
    return record
