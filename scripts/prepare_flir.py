#!/usr/bin/env python3
"""Prepare the official expanded FLIR ADAS validation set for external evaluation.

The Teledyne download is registration-gated, so this command intentionally
accepts a user-supplied ZIP file or extracted directory and never downloads it.
Only the official validation COCO annotations and 8-bit thermal JPEGs are
materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "FLIR_ADAS_v2"
DATASET_RELEASE = "expanded-2022"
DATASET_SPLIT = "official-validation"
DATASET_DIR_NAME = "FLIR-ADAS-v2-YOLO-infrared"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / DATASET_DIR_NAME
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "FLIR-ADAS-v2-val-v1.json"
DEFAULT_PILOT = PROJECT_ROOT / "manifests" / "FLIR-ADAS-v2-pilot-100-v1.json"
DEFAULT_PAYLOAD = PROJECT_ROOT / "manifests" / "FLIR-ADAS-v2-payload-v1.json"
DEFAULT_TAR = PROJECT_ROOT / "artifacts" / "FLIR-ADAS-v2-val-infrared.tar"
PILOT_SIZE = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_member(name: str) -> str:
    normalized = PurePosixPath(name)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return normalized.as_posix()


class DatasetSource:
    """Read-only view over either an extracted directory or a ZIP archive."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        if self.path.is_dir():
            self.kind = "directory"
            self._zip: zipfile.ZipFile | None = None
            self.members = tuple(
                sorted(
                    child.relative_to(self.path).as_posix()
                    for child in self.path.rglob("*")
                    if child.is_file()
                )
            )
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.kind = "zip"
            self._zip = zipfile.ZipFile(self.path)
            self.members = tuple(
                sorted(
                    _safe_member(item.filename)
                    for item in self._zip.infolist()
                    if not item.is_dir()
                )
            )
        else:
            raise ValueError("--source must be an extracted directory or ZIP archive")
        if len(self.members) != len(set(self.members)):
            raise ValueError("dataset source contains duplicate member paths")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def open(self, member: str) -> BinaryIO:
        member = _safe_member(member)
        if member not in self.members:
            raise FileNotFoundError(f"source member not found: {member}")
        if self._zip is not None:
            return self._zip.open(member)
        return (self.path / member).open("rb")

    def read_bytes(self, member: str) -> bytes:
        with self.open(member) as source:
            return source.read()

    def source_sha256(self) -> str:
        if self.kind == "zip":
            return sha256_file(self.path)
        digest = hashlib.sha256()
        for member in self.members:
            digest.update(member.encode())
            digest.update(b"\0")
            with self.open(member) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class CocoCandidate:
    member: str
    value: dict[str, Any]
    score: int


def _looks_like_coco(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(value.get(key), list)
        for key in ("images", "annotations", "categories")
    )


def _annotation_score(member: str) -> int:
    name = member.casefold()
    score = 0
    score += 20 if re.search(r"(^|[/_.-])val(idation)?([/_.-]|$)", name) else 0
    score += 8 if "thermal" in name else 0
    score += 4 if "coco" in name else 0
    score += 2 if "annotation" in name else 0
    score -= 30 if "rgb" in name or "visible" in name else 0
    score -= 10 if "video" in name else 0
    score -= 20 if re.search(r"(^|[/_.-])train([/_.-]|$)", name) else 0
    return score


def discover_coco_annotation(
    source: DatasetSource, explicit: str | None = None
) -> CocoCandidate:
    if explicit:
        candidates = [explicit]
    else:
        candidates = [
            member
            for member in source.members
            if member.casefold().endswith(".json")
            and any(
                token in member.casefold()
                for token in ("val", "thermal", "coco", "annotation")
            )
        ]
    parsed = []
    for member in candidates:
        try:
            value = json.loads(source.read_bytes(member))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if _looks_like_coco(value):
            parsed.append(CocoCandidate(member, value, _annotation_score(member)))
    if not parsed:
        raise ValueError(
            "no COCO validation annotations found; pass --annotations MEMBER"
        )
    parsed.sort(key=lambda item: (-item.score, item.member))
    if len(parsed) > 1 and parsed[0].score == parsed[1].score:
        names = [item.member for item in parsed if item.score == parsed[0].score]
        raise ValueError(
            "ambiguous COCO validation annotations; pass --annotations: "
            + ", ".join(names[:5])
        )
    selected = parsed[0]
    if not explicit and selected.score <= 0:
        raise ValueError(
            f"could not identify an official thermal validation JSON: {selected.member}"
        )
    return selected


def discover_training_file_names(
    source: DatasetSource, selected_member: str
) -> tuple[set[str], list[str]]:
    """Find official train COCO files for a conservative split-overlap audit."""
    file_names: set[str] = set()
    members = []
    for member in source.members:
        name = member.casefold()
        if (
            member != selected_member
            and member.casefold().endswith(".json")
            and re.search(r"(^|[/_.-])train([/_.-]|$)", name)
            and "rgb" not in name
            and "visible" not in name
            and "video" not in name
        ):
            try:
                value = json.loads(source.read_bytes(member))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if _looks_like_coco(value):
                members.append(member)
                file_names.update(str(image["file_name"]) for image in value["images"])
    return file_names, sorted(members)


def _image_score(member: str, explicit_prefix: str | None) -> int:
    name = member.casefold()
    directory = PurePosixPath(name).parent.as_posix()
    if explicit_prefix and not name.startswith(
        explicit_prefix.casefold().rstrip("/") + "/"
    ):
        return -10_000
    score = 0
    score += 20 if "thermal" in directory else 0
    score += 6 if "val" in directory else 0
    score += 2 if "jpeg" in name or "jpg" in name else 0
    score -= 40 if "rgb" in directory or "visible" in directory else 0
    score -= 10 if "video" in directory else 0
    return score


def resolve_image_member(
    file_name: str,
    members: tuple[str, ...],
    *,
    images_prefix: str | None,
) -> str:
    requested = _safe_member(file_name).casefold()
    candidates = [
        member
        for member in members
        if member.casefold() == requested or member.casefold().endswith("/" + requested)
    ]
    candidates = [
        item for item in candidates if Path(item).suffix.casefold() in {".jpg", ".jpeg"}
    ]
    ranked = sorted(
        ((_image_score(item, images_prefix), item) for item in candidates),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] < 0:
        raise FileNotFoundError(f"8-bit thermal JPEG not found for {file_name}")
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        raise ValueError(
            f"ambiguous image for {file_name}: {ranked[0][1]}, {ranked[1][1]}"
        )
    return ranked[0][1]


def _group_id(image: dict[str, Any], file_name: str, image_id: str) -> tuple[str, str]:
    metadata = image.get("extra_info", {})
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("video_id", "sequence_id", "sequence", "clip_id"):
        value = image.get(key, metadata.get(key))
        if value not in (None, ""):
            return f"{key}:{value}", f"COCO image field {key}"
    path = PurePosixPath(file_name)
    if len(path.parts) > 1 and path.parent.name.casefold() not in {
        "val",
        "validation",
        "thermal",
        "images",
        "data",
    }:
        return f"directory:{path.parent.as_posix()}", "source image directory"
    prefix = re.sub(r"(?:[_-]?\d{4,})$", "", path.stem)
    if prefix and prefix != path.stem:
        return f"filename:{prefix}", "filename without trailing frame number"
    return f"image:{image_id}", "image-level fallback"


def _day_night(image: dict[str, Any]) -> str | None:
    metadata = image.get("extra_info", {})
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("day_night", "timeofday", "time_of_day", "lighting", "hours"):
        value = image.get(key, metadata.get(key))
        if value not in (None, ""):
            return str(value).strip().casefold()
    return None


def _crowd_bin(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 2:
        return str(count)
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


def _box_size(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def _clip_coco_box(
    bbox: Any, width: int, height: int
) -> tuple[list[float] | None, dict[str, Any] | None]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, {"bbox": bbox, "reason": "bbox must be [x, y, width, height]"}
    try:
        x, y, box_width, box_height = map(float, bbox)
    except (TypeError, ValueError):
        return None, {"bbox": bbox, "reason": "bbox values must be numeric"}
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return None, {"bbox": bbox, "reason": "bbox values must be finite"}
    original = [x, y, x + box_width, y + box_height]
    clipped = [
        max(0.0, min(float(width), original[0])),
        max(0.0, min(float(height), original[1])),
        max(0.0, min(float(width), original[2])),
        max(0.0, min(float(height), original[3])),
    ]
    if (
        box_width <= 0
        or box_height <= 0
        or clipped[0] >= clipped[2]
        or clipped[1] >= clipped[3]
    ):
        return None, {"bbox": bbox, "reason": "non-positive area after clipping"}
    if clipped != original:
        return clipped, {
            "bbox": bbox,
            "clipped_xyxy": clipped,
            "reason": "clipped to image bounds",
        }
    return clipped, None


def _write_if_same_or_force(path: Path, data: bytes, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data and not force:
        raise ValueError(f"existing output differs: {path}; use --force")
    if not path.exists() or path.read_bytes() != data:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _copy_member(
    source: DatasetSource, member: str, destination: Path, force: bool
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = destination.with_name(f".{destination.name}.tmp")
    with source.open(member) as input_file, temporary.open("wb") as output_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
            output_file.write(chunk)
    source_hash = digest.hexdigest()
    if destination.exists():
        if sha256_file(destination) != source_hash and not force:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"existing image differs: {destination}; use --force")
        if sha256_file(destination) == source_hash:
            temporary.unlink()
            return source_hash
    temporary.replace(destination)
    return source_hash


def _thermal_intensity(path: Path) -> float:
    with Image.open(path) as image:
        if image.format != "JPEG" or image.mode not in {"L", "RGB"}:
            raise ValueError(
                f"expected an 8-bit thermal JPEG, got {image.format}/{image.mode}: {path}"
            )
        grayscale = image.convert("L").resize((64, 64))
        return float(ImageStat.Stat(grayscale).mean[0])


def _yolo_label(boxes: list[list[float]], width: int, height: int) -> bytes:
    lines = []
    for x1, y1, x2, y2 in boxes:
        lines.append(
            "0 "
            f"{((x1 + x2) / 2) / width:.8f} "
            f"{((y1 + y2) / 2) / height:.8f} "
            f"{(x2 - x1) / width:.8f} "
            f"{(y2 - y1) / height:.8f}\n"
        )
    return "".join(lines).encode()


def _select_pilot(records: list[dict[str, Any]], size: int) -> list[str]:
    if len(records) < size:
        raise ValueError(f"validation set has only {len(records)} images; need {size}")
    buckets: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        size_counts = record["ground_truth_box_size_counts"]
        dominant_size = max(
            ("small", "medium", "large"),
            key=lambda item: (size_counts[item], item),
        )
        if not sum(size_counts.values()):
            dominant_size = "none"
        key = (
            record["day_night"] or "unknown",
            record["crowd_bin"],
            dominant_size,
            record["intensity_quintile"],
        )
        buckets[key].append(record)
    for key, values in list(buckets.items()):
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in values:
            by_group[str(record["group_id"])].append(record)
        for group_records in by_group.values():
            group_records.sort(key=lambda item: item["image_id"])
        interleaved = []
        while any(by_group.values()):
            for group in sorted(by_group):
                if by_group[group]:
                    interleaved.append(by_group[group].pop(0))
        buckets[key] = interleaved
    selected: list[str] = []
    keys = sorted(buckets)
    while len(selected) < size:
        progressed = False
        for key in keys:
            if buckets[key]:
                selected.append(str(buckets[key].pop(0)["image_id"]))
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            raise RuntimeError("pilot selection exhausted unexpectedly")
    return sorted(selected)


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def build_deterministic_tar(
    output: Path,
    dataset_root: Path,
    manifest_path: Path,
    pilot_path: Path,
    force: bool,
) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(dataset_root.rglob("*")):
            archive.add(
                path,
                arcname=f"{dataset_root.name}/{path.relative_to(dataset_root)}",
                recursive=False,
                filter=_tar_filter,
            )
        archive.add(
            manifest_path,
            arcname=manifest_path.name,
            recursive=False,
            filter=_tar_filter,
        )
        archive.add(
            pilot_path,
            arcname=pilot_path.name,
            recursive=False,
            filter=_tar_filter,
        )
    digest = sha256_file(temporary)
    if output.exists() and sha256_file(output) == digest:
        temporary.unlink()
        return digest
    if output.exists() and not force:
        temporary.unlink()
        raise ValueError(f"existing tar differs: {output}; use --force")
    temporary.replace(output)
    return digest


def prepare(
    *,
    source_path: Path,
    output_root: Path,
    manifest_path: Path,
    pilot_path: Path,
    payload_path: Path,
    tar_path: Path,
    annotations: str | None,
    images_prefix: str | None,
    force: bool,
) -> dict[str, Any]:
    source = DatasetSource(source_path)
    try:
        candidate = discover_coco_annotation(source, annotations)
        coco = candidate.value
        train_file_names, train_annotation_members = discover_training_file_names(
            source, candidate.member
        )
        validation_file_names = {str(image["file_name"]) for image in coco["images"]}
        overlap = validation_file_names & train_file_names
        if overlap:
            raise ValueError(
                f"official train/validation image overlap: {sorted(overlap)[:5]}"
            )
        categories = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
        person_ids = [
            category_id
            for category_id, name in categories.items()
            if name.strip().casefold() == "person"
        ]
        if len(person_ids) != 1:
            raise ValueError(
                f"expected exactly one exact person category, found {person_ids}"
            )
        person_id = person_ids[0]
        images = {str(item["id"]): item for item in coco["images"]}
        if len(images) != len(coco["images"]):
            raise ValueError("COCO validation annotations contain duplicate image IDs")
        annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        annotation_ids: set[str] = set()
        for annotation in coco["annotations"]:
            annotation_id = str(annotation["id"])
            if annotation_id in annotation_ids:
                raise ValueError(f"duplicate COCO annotation ID: {annotation_id}")
            annotation_ids.add(annotation_id)
            image_id = str(annotation["image_id"])
            if image_id not in images:
                raise ValueError(f"annotation references unknown image ID: {image_id}")
            annotations_by_image[image_id].append(annotation)

        image_directory = output_root / "images" / "val"
        label_directory = output_root / "labels" / "val"
        records: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        group_methods = Counter()
        for image_id, image in sorted(images.items(), key=lambda item: item[0]):
            width = int(image["width"])
            height = int(image["height"])
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid COCO dimensions for image {image_id}")
            file_name = str(image["file_name"])
            member = resolve_image_member(
                file_name, source.members, images_prefix=images_prefix
            )
            destination = image_directory / f"{image_id}.jpg"
            image_hash = _copy_member(source, member, destination, force)
            with Image.open(destination) as decoded:
                if decoded.size != (width, height):
                    raise ValueError(
                        f"dimension mismatch for {image_id}: COCO={width}x{height}, image={decoded.size}"
                    )
            boxes: list[list[float]] = []
            ignored_boxes: list[list[float]] = []
            for annotation in annotations_by_image.get(image_id, []):
                if int(annotation["category_id"]) != person_id:
                    continue
                box, audit = _clip_coco_box(annotation.get("bbox"), width, height)
                if audit is not None:
                    audits.append(
                        {
                            "annotation_id": str(annotation["id"]),
                            "image_id": image_id,
                            **audit,
                        }
                    )
                if box is None:
                    continue
                if int(annotation.get("iscrowd", 0)):
                    ignored_boxes.append(box)
                else:
                    boxes.append(box)
            _write_if_same_or_force(
                label_directory / f"{image_id}.txt",
                _yolo_label(boxes, width, height),
                force,
            )
            size_counts = Counter(
                _box_size((box[2] - box[0]) * (box[3] - box[1])) for box in boxes
            )
            group_id, group_method = _group_id(image, file_name, image_id)
            group_methods[group_method] += 1
            records.append(
                {
                    "image_id": image_id,
                    "source_image_id": image["id"],
                    "source_file_name": file_name,
                    "source_member": member,
                    "prepared_file_name": f"{image_id}.jpg",
                    "image_sha256": image_hash,
                    "width": width,
                    "height": height,
                    "group_id": group_id,
                    "group_method": group_method,
                    "day_night": _day_night(image),
                    "person_boxes_xyxy": boxes,
                    "ignored_person_boxes_xyxy": ignored_boxes,
                    "person_count": len(boxes),
                    "ignored_person_count": len(ignored_boxes),
                    "crowd_bin": _crowd_bin(len(boxes)),
                    "ground_truth_box_size_counts": {
                        name: size_counts[name] for name in ("small", "medium", "large")
                    },
                    "thermal_intensity": round(_thermal_intensity(destination), 6),
                }
            )

        for inferred_method in (
            "source image directory",
            "filename without trailing frame number",
        ):
            inferred_groups = {
                record["group_id"]
                for record in records
                if record["group_method"] == inferred_method
            }
            if len(inferred_groups) == 1:
                for record in records:
                    if record["group_method"] == inferred_method:
                        group_methods[inferred_method] -= 1
                        record["group_id"] = f"image:{record['image_id']}"
                        record["group_method"] = "image-level fallback"
                        group_methods["image-level fallback"] += 1
                if not group_methods[inferred_method]:
                    del group_methods[inferred_method]

        intensity_order = sorted(
            records, key=lambda item: (item["thermal_intensity"], item["image_id"])
        )
        for rank, record in enumerate(intensity_order):
            record["intensity_quintile"] = min(4, rank * 5 // len(records))
        records.sort(key=lambda item: item["image_id"])

        source_hash = source.source_sha256()
        annotation_bytes = json.dumps(
            coco, sort_keys=True, separators=(",", ":")
        ).encode()
        manifest = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "dataset_release": DATASET_RELEASE,
            "dataset_split": DATASET_SPLIT,
            "image_representation": "8-bit AGC thermal JPEG",
            "source": {
                "kind": source.kind,
                "name": source.path.name,
                "sha256": source_hash,
                "annotation_member": candidate.member,
                "annotation_canonical_sha256": sha256_bytes(annotation_bytes),
                "train_annotation_members_audited": train_annotation_members,
                "train_validation_overlap_count": len(overlap),
                "inventory": {
                    "member_count": len(source.members),
                    "bytes": source.path.stat().st_size
                    if source.kind == "zip"
                    else None,
                    "extensions": dict(
                        sorted(
                            Counter(
                                Path(member).suffix.casefold()
                                for member in source.members
                            ).items()
                        )
                    ),
                },
            },
            "source_categories": {
                str(category_id): name
                for category_id, name in sorted(categories.items())
            },
            "category_policy": {
                "included": "exact case-insensitive category name person",
                "person_category_id": person_id,
                "ignored_categories": {
                    str(category_id): name
                    for category_id, name in sorted(categories.items())
                    if category_id != person_id
                },
                "iscrowd": "preserved as ignored regions with COCO crowd overlap semantics",
            },
            "image_count": len(records),
            "person_box_count": sum(record["person_count"] for record in records),
            "ignored_person_box_count": sum(
                record["ignored_person_count"] for record in records
            ),
            "person_negative_image_count": sum(
                not record["person_count"] for record in records
            ),
            "group_method_counts": dict(sorted(group_methods.items())),
            "annotation_audit": {
                "changed_or_dropped_count": len(audits),
                "records": audits,
            },
            "uses_model_results": False,
            "records": records,
        }
        serialized = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        _write_if_same_or_force(manifest_path, serialized, force)
        manifest_hash = sha256_file(manifest_path)

        pilot_ids = _select_pilot(records, min(PILOT_SIZE, len(records)))
        pilot = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "dataset_release": DATASET_RELEASE,
            "dataset_split": DATASET_SPLIT,
            "source_manifest": _display_path(manifest_path),
            "source_manifest_sha256": manifest_hash,
            "uses_model_results": False,
            "selection_method": "deterministic sequence-interleaved round-robin over day/night, crowd, dominant person size, and thermal intensity strata",
            "size": len(pilot_ids),
            "image_ids": pilot_ids,
        }
        _write_if_same_or_force(
            pilot_path,
            json.dumps(pilot, indent=2, sort_keys=True).encode() + b"\n",
            force,
        )
        data_yaml = (
            f"path: {output_root.resolve()}\n"
            "train: images/val\n"
            "val: images/val\n"
            "test: images/val\n"
            "names:\n  0: person\n"
        ).encode()
        _write_if_same_or_force(output_root / "data.yaml", data_yaml, force)
        tar_hash = build_deterministic_tar(
            tar_path, output_root, manifest_path, pilot_path, force
        )
        payload = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "dataset_release": DATASET_RELEASE,
            "dataset_split": DATASET_SPLIT,
            "archive_name": tar_path.name,
            "archive_sha256": tar_hash,
            "manifest_name": manifest_path.name,
            "manifest_sha256": manifest_hash,
            "pilot_manifest_name": pilot_path.name,
            "pilot_manifest_sha256": sha256_file(pilot_path),
            "dataset_directory": output_root.name,
            "image_count": len(records),
        }
        _write_if_same_or_force(
            payload_path,
            json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n",
            force,
        )
        return {
            "images": len(records),
            "person_boxes": manifest["person_box_count"],
            "ignored_person_boxes": manifest["ignored_person_box_count"],
            "person_negative_images": manifest["person_negative_image_count"],
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "pilot": str(pilot_path),
            "payload": str(payload_path),
            "tar": str(tar_path),
            "tar_sha256": tar_hash,
        }
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        help="COCO JSON member/path relative to --source when auto-discovery is ambiguous",
    )
    parser.add_argument(
        "--images",
        help="thermal JPEG member/path prefix when image discovery is ambiguous",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--payload-manifest", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--tar", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare(
        source_path=args.source,
        output_root=args.output_root,
        manifest_path=args.manifest,
        pilot_path=args.pilot_manifest,
        payload_path=args.payload_manifest,
        tar_path=args.tar,
        annotations=args.annotations,
        images_prefix=args.images,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
