"""Shared prediction parsing and evaluation primitives for detector experiments."""

from .matching import BoxMatch, ImageMatches, match_boxes
from .schema import Box, PredictionRecord

__all__ = [
    "Box",
    "BoxMatch",
    "ImageMatches",
    "PredictionRecord",
    "match_boxes",
]
