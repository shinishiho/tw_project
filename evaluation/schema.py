"""Versioned, model-independent prediction records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PredictionStatus = Literal["ok", "no_output", "malformed", "error"]
Modality = Literal["visible", "infrared"]


@dataclass(frozen=True)
class Box:
    """One pixel-coordinate XYXY box in the source image coordinate system."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "person"
    confidence: float | None = None

    def validate(self, image_width: int, image_height: int) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(isinstance(value, (int, float)) for value in coordinates):
            raise TypeError(f"box coordinates must be numeric: {coordinates}")
        if not (0 <= self.x1 < self.x2 <= image_width):
            raise ValueError(f"invalid horizontal box coordinates: {coordinates}")
        if not (0 <= self.y1 < self.y2 <= image_height):
            raise ValueError(f"invalid vertical box coordinates: {coordinates}")
        if not self.label:
            raise ValueError("box label cannot be empty")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1]: {self.confidence}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Box:
        return cls(
            x1=float(value["x1"]),
            y1=float(value["y1"]),
            x2=float(value["x2"]),
            y2=float(value["y2"]),
            label=str(value.get("label", "person")),
            confidence=(
                None if value.get("confidence") is None else float(value["confidence"])
            ),
        )


@dataclass(frozen=True)
class PredictionRecord:
    """One resumable JSONL record for one model, image, and modality."""

    run_id: str
    image_id: str
    modality: Modality
    model_id: str
    model_revision: str
    image_width: int
    image_height: int
    boxes: tuple[Box, ...] = ()
    status: PredictionStatus = "ok"
    raw_output: str | None = None
    prompt: str | None = None
    latency_ms: float | None = None
    parser_diagnostics: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        for field_name in ("run_id", "image_id", "model_id", "model_revision"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty")
        if self.modality not in ("visible", "infrared"):
            raise ValueError(f"unsupported modality: {self.modality}")
        if self.status not in ("ok", "no_output", "malformed", "error"):
            raise ValueError(f"unsupported prediction status: {self.status}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency cannot be negative")
        if self.status == "no_output" and self.boxes:
            raise ValueError("a no_output record cannot contain boxes")
        for box in self.boxes:
            box.validate(self.image_width, self.image_height)
        for name, count in self.parser_diagnostics.items():
            if not isinstance(name, str) or not isinstance(count, int) or count < 0:
                raise ValueError(f"invalid parser diagnostic: {name}={count}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PredictionRecord:
        record = cls(
            schema_version=int(value.get("schema_version", 1)),
            run_id=str(value["run_id"]),
            image_id=str(value["image_id"]),
            modality=value["modality"],
            model_id=str(value["model_id"]),
            model_revision=str(value["model_revision"]),
            image_width=int(value["image_width"]),
            image_height=int(value["image_height"]),
            boxes=tuple(Box.from_dict(item) for item in value.get("boxes", [])),
            status=value.get("status", "ok"),
            raw_output=value.get("raw_output"),
            prompt=value.get("prompt"),
            latency_ms=(
                None if value.get("latency_ms") is None else float(value["latency_ms"])
            ),
            parser_diagnostics={
                str(name): int(count)
                for name, count in value.get("parser_diagnostics", {}).items()
            },
            metadata=dict(value.get("metadata", {})),
        )
        record.validate()
        return record
