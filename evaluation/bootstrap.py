"""Paired image-level bootstrap utilities for fixed-output detection metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matching import (
    intersection_over_union,
    match_boxes_with_ignored_regions,
)
from .schema import Box, PredictionRecord


METRIC_NAMES = (
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "false_positives_per_image",
    "false_negatives_per_image",
    "duplicate_box_rate",
    "malformed_output_rate",
    "no_output_rate",
    "error_output_rate",
)


@dataclass(frozen=True)
class ImageStatistics:
    """Sufficient per-image statistics for aggregate and bootstrap metrics."""

    image_ids: tuple[str, ...]
    true_positives: np.ndarray
    false_positives: np.ndarray
    false_negatives: np.ndarray
    matched_iou_sum: np.ndarray
    matched_iou_count: np.ndarray
    duplicate_boxes: np.ndarray
    predicted_boxes: np.ndarray
    malformed_outputs: np.ndarray
    no_outputs: np.ndarray
    error_outputs: np.ndarray


def _duplicate_count(boxes: tuple[Box, ...], threshold: float = 0.95) -> int:
    return sum(
        any(
            box.label == earlier.label
            and intersection_over_union(box, earlier) >= threshold
            for earlier in boxes[:index]
        )
        for index, box in enumerate(boxes)
    )


def build_image_statistics(
    predictions: list[PredictionRecord],
    ground_truth_by_image: dict[str, tuple[Box, ...]],
    iou_threshold: float,
    ignored_by_image: dict[str, tuple[Box, ...]] | None = None,
) -> ImageStatistics:
    """Match each image once and retain sufficient statistics in ID order."""
    by_id = {record.image_id: record for record in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("prediction records contain duplicate image IDs")
    image_ids = tuple(sorted(ground_truth_by_image))
    if set(by_id) != set(image_ids):
        raise ValueError("prediction and ground-truth image IDs do not match")

    values: dict[str, list[float | int]] = {
        "true_positives": [],
        "false_positives": [],
        "false_negatives": [],
        "matched_iou_sum": [],
        "matched_iou_count": [],
        "duplicate_boxes": [],
        "predicted_boxes": [],
        "malformed_outputs": [],
        "no_outputs": [],
        "error_outputs": [],
    }
    for image_id in image_ids:
        record = by_id[image_id]
        record.validate()
        ignored_regions = (ignored_by_image or {}).get(image_id, ())
        matches = match_boxes_with_ignored_regions(
            record.boxes,
            ground_truth_by_image[image_id],
            ignored_regions,
            iou_threshold,
        )
        values["true_positives"].append(len(matches.matches))
        values["false_positives"].append(len(matches.false_positive_indices))
        values["false_negatives"].append(len(matches.false_negative_indices))
        values["matched_iou_sum"].append(sum(item.iou for item in matches.matches))
        values["matched_iou_count"].append(len(matches.matches))
        scored_boxes = tuple(
            box
            for index, box in enumerate(record.boxes)
            if index not in set(matches.ignored_prediction_indices)
        )
        values["duplicate_boxes"].append(_duplicate_count(scored_boxes))
        values["predicted_boxes"].append(len(scored_boxes))
        values["malformed_outputs"].append(record.status == "malformed")
        values["no_outputs"].append(record.status == "no_output")
        values["error_outputs"].append(record.status == "error")

    return ImageStatistics(
        image_ids=image_ids,
        true_positives=np.asarray(values["true_positives"], dtype=np.int64),
        false_positives=np.asarray(values["false_positives"], dtype=np.int64),
        false_negatives=np.asarray(values["false_negatives"], dtype=np.int64),
        matched_iou_sum=np.asarray(values["matched_iou_sum"], dtype=np.float64),
        matched_iou_count=np.asarray(values["matched_iou_count"], dtype=np.int64),
        duplicate_boxes=np.asarray(values["duplicate_boxes"], dtype=np.int64),
        predicted_boxes=np.asarray(values["predicted_boxes"], dtype=np.int64),
        malformed_outputs=np.asarray(values["malformed_outputs"], dtype=np.int64),
        no_outputs=np.asarray(values["no_outputs"], dtype=np.int64),
        error_outputs=np.asarray(values["error_outputs"], dtype=np.int64),
    )


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def _metrics_from_indices(
    statistics: ImageStatistics, indices: np.ndarray
) -> dict[str, np.ndarray]:
    if indices.ndim == 1:
        indices = indices[np.newaxis, :]
    image_count = np.full(indices.shape[0], indices.shape[1], dtype=np.float64)
    totals = {
        name: getattr(statistics, name)[indices].sum(axis=1)
        for name in (
            "true_positives",
            "false_positives",
            "false_negatives",
            "matched_iou_sum",
            "matched_iou_count",
            "duplicate_boxes",
            "predicted_boxes",
            "malformed_outputs",
            "no_outputs",
            "error_outputs",
        )
    }
    precision = _divide(
        totals["true_positives"],
        totals["true_positives"] + totals["false_positives"],
    )
    recall = _divide(
        totals["true_positives"],
        totals["true_positives"] + totals["false_negatives"],
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
        "mean_matched_iou": _divide(
            totals["matched_iou_sum"], totals["matched_iou_count"]
        ),
        "false_positives_per_image": totals["false_positives"] / image_count,
        "false_negatives_per_image": totals["false_negatives"] / image_count,
        "duplicate_box_rate": _divide(
            totals["duplicate_boxes"], totals["predicted_boxes"]
        ),
        "malformed_output_rate": totals["malformed_outputs"] / image_count,
        "no_output_rate": totals["no_outputs"] / image_count,
        "error_output_rate": totals["error_outputs"] / image_count,
    }


def aggregate_metrics(statistics: ImageStatistics) -> dict[str, float]:
    """Compute point estimates over all images."""
    indices = np.arange(len(statistics.image_ids))
    return {
        name: float(values[0])
        for name, values in _metrics_from_indices(statistics, indices).items()
    }


def subset_image_statistics(
    statistics: ImageStatistics, image_ids: set[str] | list[str] | tuple[str, ...]
) -> ImageStatistics:
    """Select a non-empty image-ID subset while preserving deterministic order."""
    requested = set(image_ids)
    if not requested:
        raise ValueError("image statistics subset cannot be empty")
    positions_by_id = {
        image_id: index for index, image_id in enumerate(statistics.image_ids)
    }
    missing = requested - set(positions_by_id)
    if missing:
        raise ValueError(
            f"unknown image IDs in statistics subset: {sorted(missing)[:5]}"
        )
    ordered_ids = tuple(sorted(requested))
    positions = np.asarray(
        [positions_by_id[image_id] for image_id in ordered_ids], dtype=np.int64
    )
    return ImageStatistics(
        image_ids=ordered_ids,
        true_positives=statistics.true_positives[positions],
        false_positives=statistics.false_positives[positions],
        false_negatives=statistics.false_negatives[positions],
        matched_iou_sum=statistics.matched_iou_sum[positions],
        matched_iou_count=statistics.matched_iou_count[positions],
        duplicate_boxes=statistics.duplicate_boxes[positions],
        predicted_boxes=statistics.predicted_boxes[positions],
        malformed_outputs=statistics.malformed_outputs[positions],
        no_outputs=statistics.no_outputs[positions],
        error_outputs=statistics.error_outputs[positions],
    )


def bootstrap_intervals(
    statistics: ImageStatistics,
    *,
    replicates: int = 2_000,
    seed: int = 20260721,
    confidence: float = 0.95,
    chunk_size: int = 256,
    groups_by_image: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Return percentile intervals from paired image resampling."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if groups_by_image is not None:
        return _clustered_bootstrap_intervals(
            statistics,
            groups_by_image=groups_by_image,
            replicates=replicates,
            seed=seed,
            confidence=confidence,
        )
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRIC_NAMES}
    image_count = len(statistics.image_ids)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        indices = rng.integers(0, image_count, size=(size, image_count))
        for name, values in _metrics_from_indices(statistics, indices).items():
            samples[name].append(values)
    alpha = (1 - confidence) / 2
    point = aggregate_metrics(statistics)
    return {
        name: {
            "estimate": point[name],
            "low": float(np.quantile(np.concatenate(samples[name]), alpha)),
            "high": float(np.quantile(np.concatenate(samples[name]), 1 - alpha)),
        }
        for name in METRIC_NAMES
    }


def paired_bootstrap_differences(
    left: ImageStatistics,
    right: ImageStatistics,
    *,
    replicates: int = 2_000,
    seed: int = 20260721,
    confidence: float = 0.95,
    chunk_size: int = 256,
    groups_by_image: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Return left-minus-right metric differences using identical resamples."""
    if left.image_ids != right.image_ids:
        raise ValueError(
            "paired bootstrap inputs must have identical ordered image IDs"
        )
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if groups_by_image is not None:
        return _clustered_paired_differences(
            left,
            right,
            groups_by_image=groups_by_image,
            replicates=replicates,
            seed=seed,
            confidence=confidence,
        )
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRIC_NAMES}
    image_count = len(left.image_ids)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        indices = rng.integers(0, image_count, size=(size, image_count))
        left_metrics = _metrics_from_indices(left, indices)
        right_metrics = _metrics_from_indices(right, indices)
        for name in METRIC_NAMES:
            samples[name].append(left_metrics[name] - right_metrics[name])
    alpha = (1 - confidence) / 2
    left_point = aggregate_metrics(left)
    right_point = aggregate_metrics(right)
    return {
        name: {
            "estimate": left_point[name] - right_point[name],
            "low": float(np.quantile(np.concatenate(samples[name]), alpha)),
            "high": float(np.quantile(np.concatenate(samples[name]), 1 - alpha)),
        }
        for name in METRIC_NAMES
    }


def _group_positions(
    statistics: ImageStatistics, groups_by_image: dict[str, str]
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    if set(groups_by_image) != set(statistics.image_ids):
        missing = sorted(set(statistics.image_ids) - set(groups_by_image))[:5]
        extra = sorted(set(groups_by_image) - set(statistics.image_ids))[:5]
        raise ValueError(f"group/image mismatch; missing={missing}, extra={extra}")
    groups = tuple(sorted(set(groups_by_image.values())))
    positions = {
        group: np.asarray(
            [
                index
                for index, image_id in enumerate(statistics.image_ids)
                if groups_by_image[image_id] == group
            ],
            dtype=np.int64,
        )
        for group in groups
    }
    return groups, positions


def _cluster_indices(
    rng: np.random.Generator,
    groups: tuple[str, ...],
    positions: dict[str, np.ndarray],
) -> np.ndarray:
    sampled = rng.choice(groups, size=len(groups), replace=True)
    return np.concatenate([positions[str(group)] for group in sampled])


def _clustered_bootstrap_intervals(
    statistics: ImageStatistics,
    *,
    groups_by_image: dict[str, str],
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, dict[str, float]]:
    groups, positions = _group_positions(statistics, groups_by_image)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRIC_NAMES}
    for _ in range(replicates):
        values = _metrics_from_indices(
            statistics, _cluster_indices(rng, groups, positions)
        )
        for name in METRIC_NAMES:
            samples[name].append(float(values[name][0]))
    alpha = (1 - confidence) / 2
    point = aggregate_metrics(statistics)
    return {
        name: {
            "estimate": point[name],
            "low": float(np.quantile(samples[name], alpha)),
            "high": float(np.quantile(samples[name], 1 - alpha)),
        }
        for name in METRIC_NAMES
    }


def _clustered_paired_differences(
    left: ImageStatistics,
    right: ImageStatistics,
    *,
    groups_by_image: dict[str, str],
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, dict[str, float]]:
    if left.image_ids != right.image_ids:
        raise ValueError(
            "paired bootstrap inputs must have identical ordered image IDs"
        )
    groups, positions = _group_positions(left, groups_by_image)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRIC_NAMES}
    for _ in range(replicates):
        indices = _cluster_indices(rng, groups, positions)
        left_values = _metrics_from_indices(left, indices)
        right_values = _metrics_from_indices(right, indices)
        for name in METRIC_NAMES:
            samples[name].append(float(left_values[name][0] - right_values[name][0]))
    alpha = (1 - confidence) / 2
    left_point = aggregate_metrics(left)
    right_point = aggregate_metrics(right)
    return {
        name: {
            "estimate": left_point[name] - right_point[name],
            "low": float(np.quantile(samples[name], alpha)),
            "high": float(np.quantile(samples[name], 1 - alpha)),
        }
        for name in METRIC_NAMES
    }


def independent_paired_difference_of_differences(
    first_left: ImageStatistics,
    first_right: ImageStatistics,
    second_left: ImageStatistics,
    second_right: ImageStatistics,
    *,
    first_groups: dict[str, str] | None = None,
    second_groups: dict[str, str] | None = None,
    metric: str = "f1",
    replicates: int = 2_000,
    seed: int = 20260721,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap (left-right) in one dataset minus that in another dataset."""
    if metric not in METRIC_NAMES:
        raise ValueError(f"unknown metric: {metric}")
    if first_left.image_ids != first_right.image_ids:
        raise ValueError("first paired inputs have different image IDs")
    if second_left.image_ids != second_right.image_ids:
        raise ValueError("second paired inputs have different image IDs")
    rng = np.random.default_rng(seed)

    def setup(statistics, groups):
        if groups is None:
            ids = tuple(str(index) for index in range(len(statistics.image_ids)))
            return ids, {item: np.asarray([int(item)], dtype=np.int64) for item in ids}
        return _group_positions(statistics, groups)

    first_names, first_positions = setup(first_left, first_groups)
    second_names, second_positions = setup(second_left, second_groups)
    samples = []
    for _ in range(replicates):
        first_indices = _cluster_indices(rng, first_names, first_positions)
        second_indices = _cluster_indices(rng, second_names, second_positions)
        first_delta = (
            _metrics_from_indices(first_left, first_indices)[metric][0]
            - _metrics_from_indices(first_right, first_indices)[metric][0]
        )
        second_delta = (
            _metrics_from_indices(second_left, second_indices)[metric][0]
            - _metrics_from_indices(second_right, second_indices)[metric][0]
        )
        samples.append(float(first_delta - second_delta))
    first_point = (
        aggregate_metrics(first_left)[metric] - aggregate_metrics(first_right)[metric]
    )
    second_point = (
        aggregate_metrics(second_left)[metric] - aggregate_metrics(second_right)[metric]
    )
    alpha = (1 - confidence) / 2
    return {
        "estimate": first_point - second_point,
        "low": float(np.quantile(samples, alpha)),
        "high": float(np.quantile(samples, 1 - alpha)),
    }
