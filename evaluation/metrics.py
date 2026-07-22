"""Confidence-independent metrics shared by YOLO and LocateAnything."""

from __future__ import annotations

from dataclasses import dataclass

from .matching import ImageMatches, intersection_over_union, match_boxes
from .schema import Box, PredictionRecord


@dataclass(frozen=True)
class EvaluationSummary:
    image_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    false_positives_per_image: float
    false_negatives_per_image: float
    duplicate_box_rate: float
    malformed_output_rate: float
    no_output_rate: float
    error_output_rate: float
    assignments: dict[str, ImageMatches]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _duplicate_count(boxes: tuple[Box, ...], threshold: float = 0.95) -> int:
    duplicates = 0
    for index, box in enumerate(boxes):
        if any(
            box.label == earlier.label
            and intersection_over_union(box, earlier) >= threshold
            for earlier in boxes[:index]
        ):
            duplicates += 1
    return duplicates


def evaluate_records(
    predictions: list[PredictionRecord],
    ground_truth_by_image: dict[str, tuple[Box, ...]],
    iou_threshold: float,
) -> EvaluationSummary:
    """Evaluate one record per image and preserve per-image assignments."""
    if len({record.image_id for record in predictions}) != len(predictions):
        raise ValueError("prediction records contain duplicate image IDs")
    prediction_ids = {record.image_id for record in predictions}
    if prediction_ids != set(ground_truth_by_image):
        missing = sorted(set(ground_truth_by_image) - prediction_ids)[:5]
        extra = sorted(prediction_ids - set(ground_truth_by_image))[:5]
        raise ValueError(
            f"prediction/ground-truth stem mismatch; missing={missing}, extra={extra}"
        )

    assignments = {}
    matched_ious = []
    true_positives = false_positives = false_negatives = duplicates = 0
    malformed = no_output = errors = total_predictions = 0
    for record in predictions:
        record.validate()
        truth = ground_truth_by_image[record.image_id]
        for box in truth:
            box.validate(record.image_width, record.image_height)
        image_matches = match_boxes(record.boxes, truth, iou_threshold)
        assignments[record.image_id] = image_matches
        true_positives += len(image_matches.matches)
        false_positives += len(image_matches.false_positive_indices)
        false_negatives += len(image_matches.false_negative_indices)
        matched_ious.extend(item.iou for item in image_matches.matches)
        duplicates += _duplicate_count(record.boxes)
        total_predictions += len(record.boxes)
        malformed += record.status == "malformed"
        no_output += record.status == "no_output"
        errors += record.status == "error"

    image_count = len(predictions)
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    return EvaluationSummary(
        image_count=image_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=_safe_divide(2 * precision * recall, precision + recall),
        mean_matched_iou=_safe_divide(sum(matched_ious), len(matched_ious)),
        false_positives_per_image=_safe_divide(false_positives, image_count),
        false_negatives_per_image=_safe_divide(false_negatives, image_count),
        duplicate_box_rate=_safe_divide(duplicates, total_predictions),
        malformed_output_rate=_safe_divide(malformed, image_count),
        no_output_rate=_safe_divide(no_output, image_count),
        error_output_rate=_safe_divide(errors, image_count),
        assignments=assignments,
    )
