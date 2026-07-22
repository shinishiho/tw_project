"""Confidence-ranked person AP with COCO crowd-ignore behavior."""

from __future__ import annotations

import numpy as np

from .matching import intersection_over_prediction, intersection_over_union
from .schema import Box, PredictionRecord


def average_precision(
    predictions: list[PredictionRecord],
    ground_truth: dict[str, tuple[Box, ...]],
    ignored: dict[str, tuple[Box, ...]],
    iou_threshold: float,
) -> float:
    """Compute 101-point interpolated AP for one IoU threshold."""
    by_id = {record.image_id: record for record in predictions}
    if len(by_id) != len(predictions) or set(by_id) != set(ground_truth):
        raise ValueError(
            "AP predictions and ground truth must have identical unique IDs"
        )
    total_truth = sum(len(boxes) for boxes in ground_truth.values())
    if not total_truth:
        raise ValueError("AP requires at least one non-crowd ground-truth box")
    ranked = sorted(
        (
            (float(box.confidence), record.image_id, index, box)
            for record in predictions
            for index, box in enumerate(record.boxes)
            if box.confidence is not None
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    matched: dict[str, set[int]] = {image_id: set() for image_id in ground_truth}
    true_positive = []
    false_positive = []
    for _, image_id, _, prediction in ranked:
        candidates = [
            (intersection_over_union(prediction, truth), index)
            for index, truth in enumerate(ground_truth[image_id])
            if index not in matched[image_id] and prediction.label == truth.label
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            true_positive.append(1)
            false_positive.append(0)
            continue
        if any(
            prediction.label == region.label
            and intersection_over_prediction(prediction, region) >= iou_threshold
            for region in ignored.get(image_id, ())
        ):
            continue
        true_positive.append(0)
        false_positive.append(1)
    if not true_positive:
        return 0.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / total_truth
    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp, dtype=np.float64),
        where=(cumulative_tp + cumulative_fp) != 0,
    )
    values = [
        float(np.max(precision[recall >= level])) if np.any(recall >= level) else 0.0
        for level in np.linspace(0.0, 1.0, 101)
    ]
    return float(np.mean(values))


def coco_style_ap(
    predictions: list[PredictionRecord],
    ground_truth: dict[str, tuple[Box, ...]],
    ignored: dict[str, tuple[Box, ...]],
) -> dict[str, float]:
    thresholds = [round(0.50 + index * 0.05, 2) for index in range(10)]
    values = {
        threshold: average_precision(
            predictions, ground_truth, ignored, iou_threshold=threshold
        )
        for threshold in thresholds
    }
    return {
        "ap50": values[0.50],
        "ap75": values[0.75],
        "map50_95": float(np.mean(list(values.values()))),
        "method": "101-point interpolation over IoU 0.50:0.05:0.95 with COCO crowd ignore",
    }
