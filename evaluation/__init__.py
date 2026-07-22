"""Shared prediction parsing and evaluation primitives for LLVIP experiments."""

from .matching import BoxMatch, ImageMatches, match_boxes
from .metrics import EvaluationSummary, evaluate_records
from .schema import Box, PredictionRecord

__all__ = [
    "Box",
    "BoxMatch",
    "EvaluationSummary",
    "ImageMatches",
    "PredictionRecord",
    "evaluate_records",
    "match_boxes",
]
