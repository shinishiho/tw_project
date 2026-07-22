"""FLIR manifest ground-truth and identity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import Box, PredictionRecord


DATASET_ID = "FLIR_ADAS_v2"
DATASET_RELEASE = "expanded-2022"
DATASET_SPLIT = "official-validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_flir_manifest(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[Box, ...]],
    dict[str, tuple[Box, ...]],
]:
    manifest = json.loads(path.read_text())
    manifest["_file_sha256"] = sha256(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported FLIR manifest schema")
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(f"unexpected FLIR dataset ID: {manifest.get('dataset_id')}")
    records = manifest.get("records", [])
    ids = [str(record["image_id"]) for record in records]
    if len(ids) != len(set(ids)) or len(ids) != manifest.get("image_count"):
        raise ValueError("FLIR manifest image count or IDs are inconsistent")
    ground_truth: dict[str, tuple[Box, ...]] = {}
    ignored: dict[str, tuple[Box, ...]] = {}
    for record in records:
        image_id = str(record["image_id"])
        width = int(record["width"])
        height = int(record["height"])
        ground_truth[image_id] = tuple(
            _box_from_manifest(value, width, height)
            for value in record.get("person_boxes_xyxy", [])
        )
        ignored[image_id] = tuple(
            _box_from_manifest(value, width, height)
            for value in record.get("ignored_person_boxes_xyxy", [])
        )
    return manifest, ground_truth, ignored


def _box_from_manifest(value: list[float], width: int, height: int) -> Box:
    if len(value) != 4:
        raise ValueError(f"invalid manifest box: {value}")
    box = Box(*map(float, value))
    box.validate(width, height)
    return box


def groups_from_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        str(record["image_id"]): str(record["group_id"])
        for record in manifest["records"]
    }


def validate_flir_prediction_identity(
    records: list[PredictionRecord],
    manifest: dict[str, Any],
) -> None:
    expected_ids = {str(record["image_id"]) for record in manifest["records"]}
    actual_ids = [record.image_id for record in records]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("FLIR predictions do not exactly match the locked manifest")
    manifest_hash = manifest.get("_file_sha256")
    for record in records:
        metadata = record.metadata
        expected = {
            "dataset_id": DATASET_ID,
            "dataset_release": DATASET_RELEASE,
            "dataset_split": DATASET_SPLIT,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"unexpected {key} for {record.image_id}: {metadata.get(key)!r}"
                )
        if manifest_hash and metadata.get("dataset_manifest_sha256") != manifest_hash:
            raise ValueError(f"manifest hash mismatch for {record.image_id}")


def validate_resumable_flir_predictions(
    records: list[PredictionRecord],
    manifest: dict[str, Any],
    *,
    run_id: str,
    model_id: str,
    model_revision: str,
    expected_ids: set[str],
) -> set[str]:
    """Validate a possibly partial prediction file before resuming inference."""
    completed: set[str] = set()
    manifest_hash = manifest.get("_file_sha256")
    source_hash = manifest.get("source", {}).get("sha256")
    expected_metadata = {
        "dataset_id": DATASET_ID,
        "dataset_release": DATASET_RELEASE,
        "dataset_split": DATASET_SPLIT,
        "dataset_source_sha256": source_hash,
        "dataset_manifest_sha256": manifest_hash,
    }
    for record in records:
        record.validate()
        identity = (
            record.run_id,
            record.modality,
            record.model_id,
            record.model_revision,
        )
        expected_identity = (run_id, "infrared", model_id, model_revision)
        if identity != expected_identity:
            raise ValueError(f"unexpected resumable prediction identity: {identity}")
        if record.image_id in completed:
            raise ValueError(f"duplicate resumable image ID: {record.image_id}")
        if record.image_id not in expected_ids:
            raise ValueError(
                f"resumable image ID is outside the split: {record.image_id}"
            )
        for key, value in expected_metadata.items():
            if value is not None and record.metadata.get(key) != value:
                raise ValueError(
                    f"unexpected resumable {key} for {record.image_id}: "
                    f"{record.metadata.get(key)!r}"
                )
        completed.add(record.image_id)
    return completed
