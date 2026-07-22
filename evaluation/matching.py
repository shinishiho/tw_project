"""Deterministic maximum-cardinality, maximum-IoU box matching."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Box


@dataclass(frozen=True)
class BoxMatch:
    prediction_index: int
    ground_truth_index: int
    iou: float


@dataclass(frozen=True)
class ImageMatches:
    matches: tuple[BoxMatch, ...]
    false_positive_indices: tuple[int, ...]
    false_negative_indices: tuple[int, ...]


@dataclass
class _Edge:
    target: int
    reverse: int
    capacity: int
    cost: float
    prediction_index: int | None = None
    ground_truth_index: int | None = None
    iou: float | None = None


def intersection_over_union(left: Box, right: Box) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    return intersection / (left_area + right_area - intersection)


def _add_edge(
    graph: list[list[_Edge]],
    source: int,
    target: int,
    cost: float,
    prediction_index: int | None = None,
    ground_truth_index: int | None = None,
    iou: float | None = None,
) -> None:
    forward = _Edge(
        target,
        len(graph[target]),
        1,
        cost,
        prediction_index,
        ground_truth_index,
        iou,
    )
    reverse = _Edge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(reverse)


def match_boxes(
    predictions: tuple[Box, ...] | list[Box],
    ground_truth: tuple[Box, ...] | list[Box],
    iou_threshold: float,
) -> ImageMatches:
    """Match within class, maximizing TP count first and total IoU second."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be in [0, 1]")
    prediction_count = len(predictions)
    ground_truth_count = len(ground_truth)
    source = 0
    prediction_offset = 1
    ground_truth_offset = prediction_offset + prediction_count
    sink = ground_truth_offset + ground_truth_count
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]

    for prediction_index in range(prediction_count):
        _add_edge(graph, source, prediction_offset + prediction_index, 0.0)
    for ground_truth_index in range(ground_truth_count):
        _add_edge(graph, ground_truth_offset + ground_truth_index, sink, 0.0)
    for prediction_index, prediction in enumerate(predictions):
        for ground_truth_index, truth in enumerate(ground_truth):
            if prediction.label != truth.label:
                continue
            iou = intersection_over_union(prediction, truth)
            if iou >= iou_threshold:
                _add_edge(
                    graph,
                    prediction_offset + prediction_index,
                    ground_truth_offset + ground_truth_index,
                    -iou,
                    prediction_index,
                    ground_truth_index,
                    iou,
                )

    # Successive shortest augmenting paths. Bellman-Ford permits negative
    # residual costs and lets later paths repair an earlier assignment.
    while True:
        distance = [float("inf")] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] == float("inf"):
                    continue
                for edge_index, edge in enumerate(edges):
                    candidate = distance[node] + edge.cost
                    if edge.capacity and candidate < distance[edge.target] - 1e-12:
                        distance[edge.target] = candidate
                        previous[edge.target] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            prior_node, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[prior_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = prior_node

    matches = []
    for prediction_index in range(prediction_count):
        node = prediction_offset + prediction_index
        for edge in graph[node]:
            if (
                edge.prediction_index is not None
                and edge.ground_truth_index is not None
                and edge.capacity == 0
            ):
                matches.append(
                    BoxMatch(
                        edge.prediction_index, edge.ground_truth_index, edge.iou or 0.0
                    )
                )
    matches.sort(key=lambda item: (item.prediction_index, item.ground_truth_index))
    matched_predictions = {item.prediction_index for item in matches}
    matched_truth = {item.ground_truth_index for item in matches}
    return ImageMatches(
        tuple(matches),
        tuple(
            index
            for index in range(prediction_count)
            if index not in matched_predictions
        ),
        tuple(
            index for index in range(ground_truth_count) if index not in matched_truth
        ),
    )
