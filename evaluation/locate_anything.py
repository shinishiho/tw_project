"""Parse LocateAnything's official normalized structured-box output."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .schema import Box


BOX_PATTERN = re.compile(
    r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
)
BOX_SEGMENT_PATTERN = re.compile(r"<box>.*?</box>", re.DOTALL)
NO_OBJECT_PATTERN = re.compile(r"<box>\s*none\s*</box>", re.IGNORECASE)
NORMALIZED_MAX = 1000


@dataclass(frozen=True)
class LocateAnythingParseResult:
    boxes: tuple[Box, ...]
    status: str
    diagnostics: dict[str, int]


def parse_locate_anything_boxes(
    answer: str,
    image_width: int,
    image_height: int,
    label: str = "person",
) -> LocateAnythingParseResult:
    """Parse boxes while retaining duplicates and counting every failure mode.

    NVIDIA's model card defines integer coordinates normalized to [0, 1000].
    Malformed, out-of-range, and degenerate candidates are counted and omitted.
    Duplicate valid boxes are deliberately retained for later error analysis.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    diagnostics: Counter[str] = Counter()
    segments = BOX_SEGMENT_PATTERN.findall(answer)
    diagnostics["box_segments"] = len(segments)
    diagnostics["unclosed_box_tags"] = max(
        0, answer.count("<box>") - answer.count("</box>")
    )
    diagnostics["orphan_closing_tags"] = max(
        0, answer.count("</box>") - answer.count("<box>")
    )

    boxes = []
    normalized_seen: Counter[tuple[int, int, int, int]] = Counter()
    for segment in segments:
        if NO_OBJECT_PATTERN.fullmatch(segment):
            diagnostics["no_object_segments"] += 1
            continue
        match = BOX_PATTERN.fullmatch(segment)
        if match is None:
            diagnostics["malformed_segments"] += 1
            continue
        normalized = tuple(int(match.group(name)) for name in ("x1", "y1", "x2", "y2"))
        if not all(0 <= value <= NORMALIZED_MAX for value in normalized):
            diagnostics["out_of_range_boxes"] += 1
            continue
        x1, y1, x2, y2 = normalized
        if x1 >= x2 or y1 >= y2:
            diagnostics["degenerate_boxes"] += 1
            continue
        normalized_seen[normalized] += 1
        boxes.append(
            Box(
                x1=x1 / NORMALIZED_MAX * image_width,
                y1=y1 / NORMALIZED_MAX * image_height,
                x2=x2 / NORMALIZED_MAX * image_width,
                y2=y2 / NORMALIZED_MAX * image_height,
                label=label,
            )
        )

    diagnostics["valid_boxes"] = len(boxes)
    diagnostics["duplicate_boxes"] = sum(
        count - 1 for count in normalized_seen.values() if count > 1
    )
    malformed_count = sum(
        diagnostics[key]
        for key in (
            "malformed_segments",
            "out_of_range_boxes",
            "degenerate_boxes",
            "unclosed_box_tags",
            "orphan_closing_tags",
        )
    )
    if malformed_count:
        status = "malformed"
    elif not boxes:
        status = "no_output"
    else:
        status = "ok"
    return LocateAnythingParseResult(tuple(boxes), status, dict(diagnostics))
