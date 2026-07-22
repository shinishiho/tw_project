"""Prepare LLVIP for leakage-free YOLO training and paired evaluation.

The official training split is divided by capture sequence (the first two
digits of each image ID). The official test split is never used as validation.

Outputs:
    datasets/LLVIP-splits-v1.json
    datasets/LLVIP-YOLO-{visible,infrared}/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt
        data.yaml

The image files are hard-linked by default, so preparing both modalities does
not duplicate the 3.9 GiB extracted dataset. Copying is used automatically
when hard links are unavailable.

Usage:
    uv run scripts/prepare_dataset.py
    uv run scripts/prepare_dataset.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ARCHIVE_MD5 = "e64affb4b0b50e1772ff6f67da873bf6"
EXPECTED_SPLIT_COUNTS = {"train": 12_025, "test": 3_463}
MANIFEST_NAME = "LLVIP-splits-v1.json"
MODALITIES = ("visible", "infrared")
OUTPUT_SPLITS = ("train", "val", "test")
SCHEMA_VERSION = 1
VAL_FRACTION = 0.20


@dataclass(frozen=True)
class Annotation:
    image_id: str
    width: int
    height: int
    boxes: tuple[tuple[str, float, float, float, float], ...]
    dropped_boxes: tuple[tuple[int, str, float, float, float, float, str], ...]


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the published file
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(zip_path: Path, raw_dir: Path) -> None:
    """Verify and extract the archive if the complete raw tree is absent."""
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"{zip_path} not found; run scripts/download_dataset.py first"
        )
    archive_md5 = file_md5(zip_path)
    if archive_md5 != ARCHIVE_MD5:
        raise ValueError(
            f"archive MD5 mismatch: expected {ARCHIVE_MD5}, got {archive_md5}"
        )

    expected = [raw_dir / "Annotations"] + [raw_dir / m for m in MODALITIES]
    if all(path.is_dir() for path in expected):
        print(f"Verified archive MD5; raw dataset exists at {raw_dir}")
        return

    print(f"Extracting verified archive to {raw_dir.parent}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        destination = raw_dir.parent.resolve()
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError(f"unsafe path in archive: {member.filename}")
        archive.extractall(destination)


def image_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.jpg")}


def sequence_id(image_id: str) -> str:
    if len(image_id) != 6 or not image_id.isdigit():
        raise ValueError(f"unexpected LLVIP image ID: {image_id!r}")
    return image_id[:2]


def choose_validation_groups(group_counts: Counter[str]) -> tuple[str, ...]:
    """Choose whole groups nearest to 20%, with deterministic tie-breaking."""
    total = sum(group_counts.values())
    target_images = round(total * VAL_FRACTION)
    target_groups = round(len(group_counts) * VAL_FRACTION)

    subsets_by_sum: dict[int, tuple[str, ...]] = {0: ()}
    for group in sorted(group_counts):
        additions: dict[int, tuple[str, ...]] = {}
        for subtotal, subset in list(subsets_by_sum.items()):
            new_total = subtotal + group_counts[group]
            candidate = subset + (group,)
            current = subsets_by_sum.get(new_total) or additions.get(new_total)
            if current is None or candidate < current:
                additions[new_total] = candidate
        subsets_by_sum.update(additions)

    _, selected = min(
        subsets_by_sum.items(),
        key=lambda item: (
            abs(item[0] - target_images),
            abs(len(item[1]) - target_groups),
            item[1],
        ),
    )
    if not selected or len(selected) == len(group_counts):
        raise ValueError("validation group selection produced an empty split")
    return selected


def parse_annotation(path: Path) -> Annotation:
    root = ET.parse(path).getroot()
    image_id = Path(root.findtext("filename", default="")).stem
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing image size in {path}")
    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {path}: {width}x{height}")

    boxes = []
    dropped_boxes = []
    for object_index, obj in enumerate(root.findall("object")):
        label = obj.findtext("name", default="").strip()
        bounds = obj.find("bndbox")
        if not label or bounds is None:
            raise ValueError(f"invalid object in {path}")
        xmin = float(bounds.findtext("xmin", default="nan"))
        ymin = float(bounds.findtext("ymin", default="nan"))
        xmax = float(bounds.findtext("xmax", default="nan"))
        ymax = float(bounds.findtext("ymax", default="nan"))
        coordinates = (xmin, ymin, xmax, ymax)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"non-finite box in {path}: {coordinates}")
        if not (0 <= xmin <= width and 0 <= xmax <= width):
            raise ValueError(f"out-of-bounds horizontal box in {path}: {coordinates}")
        if not (0 <= ymin <= height and 0 <= ymax <= height):
            raise ValueError(f"out-of-bounds vertical box in {path}: {coordinates}")
        if xmin >= xmax or ymin >= ymax:
            reasons = []
            if xmin >= xmax:
                reasons.append("non-positive width")
            if ymin >= ymax:
                reasons.append("non-positive height")
            dropped_boxes.append(
                (object_index, label, xmin, ymin, xmax, ymax, ", ".join(reasons))
            )
            continue
        boxes.append((label, xmin, ymin, xmax, ymax))
    return Annotation(image_id, width, height, tuple(boxes), tuple(dropped_boxes))


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions without adding an image-library dependency."""
    with path.open("rb") as image:
        if image.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG image: {path}")
        while True:
            marker_start = image.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = image.read(1)
            while marker == b"\xff":
                marker = image.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_raw = image.read(2)
            if len(length_raw) != 2:
                break
            length = struct.unpack(">H", length_raw)[0]
            if marker and marker[0] in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                payload = image.read(5)
                if len(payload) != 5:
                    break
                height, width = struct.unpack(">HH", payload[1:])
                return width, height
            image.seek(length - 2, os.SEEK_CUR)
    raise ValueError(f"could not read JPEG dimensions: {path}")


def inspect_raw_dataset(
    raw_dir: Path,
) -> tuple[dict[str, Annotation], dict[str, str], tuple[str, ...]]:
    ids_by_source: dict[str, set[str]] = {}
    for source_split in ("train", "test"):
        paired_ids = []
        for modality in MODALITIES:
            ids = image_ids(raw_dir / modality / source_split)
            if len(ids) != EXPECTED_SPLIT_COUNTS[source_split]:
                raise ValueError(
                    f"expected {EXPECTED_SPLIT_COUNTS[source_split]} "
                    f"{modality}/{source_split} images, found {len(ids)}"
                )
            paired_ids.append(ids)
        if paired_ids[0] != paired_ids[1]:
            raise ValueError(f"visible/infrared stem mismatch in {source_split}")
        ids_by_source[source_split] = paired_ids[0]

    if ids_by_source["train"] & ids_by_source["test"]:
        raise ValueError("official train and test image IDs overlap")

    annotation_paths = sorted((raw_dir / "Annotations").glob("*.xml"))
    annotations = {
        annotation.image_id: annotation
        for annotation in (parse_annotation(path) for path in annotation_paths)
    }
    all_ids = ids_by_source["train"] | ids_by_source["test"]
    if set(annotations) != all_ids:
        missing = sorted(all_ids - set(annotations))[:5]
        extra = sorted(set(annotations) - all_ids)[:5]
        raise ValueError(f"annotation stem mismatch; missing={missing}, extra={extra}")

    train_group_counts = Counter(sequence_id(item) for item in ids_by_source["train"])
    test_groups = {sequence_id(item) for item in ids_by_source["test"]}
    if set(train_group_counts) & test_groups:
        raise ValueError("a capture-sequence prefix crosses official splits")
    validation_groups = choose_validation_groups(train_group_counts)

    split_by_id = {}
    for image_id in sorted(ids_by_source["train"]):
        split_by_id[image_id] = (
            "val" if sequence_id(image_id) in validation_groups else "train"
        )
    split_by_id.update({image_id: "test" for image_id in ids_by_source["test"]})
    return annotations, split_by_id, validation_groups


def write_manifest(
    path: Path,
    annotations: dict[str, Annotation],
    split_by_id: dict[str, str],
    validation_groups: tuple[str, ...],
) -> None:
    records = [
        {
            "image_id": image_id,
            "group_id": sequence_id(image_id),
            "source_split": "test" if split == "test" else "train",
            "split": split,
        }
        for image_id, split in sorted(split_by_id.items())
    ]
    invalid_boxes = [
        {
            "image_id": annotation.image_id,
            "object_index": box[0],
            "label": box[1],
            "box_xyxy": list(box[2:6]),
            "reason": box[6],
        }
        for annotation in annotations.values()
        for box in annotation.dropped_boxes
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "LLVIP",
        "archive_md5": ARCHIVE_MD5,
        "group_key": "first two digits of the six-digit image ID",
        "validation_fraction_target": VAL_FRACTION,
        "validation_groups": list(validation_groups),
        "counts": dict(sorted(Counter(split_by_id.values()).items())),
        "annotation_validation": {
            "source_box_count": sum(
                len(annotation.boxes) + len(annotation.dropped_boxes)
                for annotation in annotations.values()
            ),
            "valid_box_count": sum(
                len(annotation.boxes) for annotation in annotations.values()
            ),
            "invalid_box_count": len(invalid_boxes),
            "empty_after_filter_image_ids": sorted(
                annotation.image_id
                for annotation in annotations.values()
                if not annotation.boxes
            ),
            "invalid_boxes": invalid_boxes,
        },
        "records": records,
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def yolo_line(
    box: tuple[str, float, float, float, float],
    annotation: Annotation,
    label_to_id: dict[str, int],
) -> str:
    label, xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / (2 * annotation.width)
    y_center = (ymin + ymax) / (2 * annotation.height)
    width = (xmax - xmin) / annotation.width
    height = (ymax - ymin) / annotation.height
    values = (x_center, y_center, width, height)
    if not all(0 <= value <= 1 for value in values) or width <= 0 or height <= 0:
        raise ValueError(f"invalid normalized box for {annotation.image_id}: {values}")
    return f"{label_to_id[label]} " + " ".join(f"{value:.8f}" for value in values)


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def build_modality(
    annotations: dict[str, Annotation],
    split_by_id: dict[str, str],
    raw_dir: Path,
    out_dir: Path,
    modality: str,
    label_to_id: dict[str, int],
) -> tuple[Path, Counter[str]]:
    final_dir = out_dir / f"LLVIP-YOLO-{modality}"
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}-", dir=out_dir))
    link_modes: Counter[str] = Counter()
    try:
        for split in OUTPUT_SPLITS:
            (temp_dir / "images" / split).mkdir(parents=True)
            (temp_dir / "labels" / split).mkdir(parents=True)

        for image_id, split in sorted(split_by_id.items()):
            source_split = "test" if split == "test" else "train"
            source = raw_dir / modality / source_split / f"{image_id}.jpg"
            destination = temp_dir / "images" / split / source.name
            link_modes[link_or_copy(source, destination)] += 1

            annotation = annotations[image_id]
            actual_dimensions = jpeg_dimensions(source)
            if actual_dimensions != (annotation.width, annotation.height):
                raise ValueError(
                    f"dimension mismatch for {source}: XML has "
                    f"{annotation.width}x{annotation.height}, JPEG has "
                    f"{actual_dimensions[0]}x{actual_dimensions[1]}"
                )
            lines = [
                yolo_line(box, annotation, label_to_id) for box in annotation.boxes
            ]
            label_path = temp_dir / "labels" / split / f"{image_id}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        names = "\n".join(
            f"  {class_id}: {label}"
            for label, class_id in sorted(label_to_id.items(), key=lambda item: item[1])
        )
        (temp_dir / "data.yaml").write_text(
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"names:\n{names}\n",
            encoding="utf-8",
        )

        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)
        return final_dir, link_modes
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def validate_output(
    dataset_dir: Path,
    annotations: dict[str, Annotation],
    split_by_id: dict[str, str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for split in OUTPUT_SPLITS:
        expected_ids = {
            item for item, item_split in split_by_id.items() if item_split == split
        }
        image_stems = image_ids(dataset_dir / "images" / split)
        label_stems = {
            path.stem for path in (dataset_dir / "labels" / split).glob("*.txt")
        }
        if image_stems != expected_ids or label_stems != expected_ids:
            raise ValueError(f"output stem mismatch in {dataset_dir.name}/{split}")
        counts[split] = len(expected_ids)

        for image_id in sorted(expected_ids):
            lines = (
                (dataset_dir / "labels" / split / f"{image_id}.txt")
                .read_text()
                .splitlines()
            )
            if len(lines) != len(annotations[image_id].boxes):
                raise ValueError(f"box-count mismatch for {image_id}")
            for line in lines:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"malformed YOLO label for {image_id}: {line}")
                class_id = int(fields[0])
                values = tuple(float(value) for value in fields[1:])
                if class_id < 0 or not all(0 <= value <= 1 for value in values):
                    raise ValueError(f"out-of-bounds YOLO label for {image_id}: {line}")
                if values[2] <= 0 or values[3] <= 0:
                    raise ValueError(f"empty YOLO box for {image_id}: {line}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("LLVIP.zip"))
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/LLVIP"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--force", action="store_true", help="rebuild generated outputs"
    )
    args = parser.parse_args()

    extract_archive(args.zip, args.raw_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotations, split_by_id, validation_groups = inspect_raw_dataset(args.raw_dir)
    labels = sorted(
        {box[0] for annotation in annotations.values() for box in annotation.boxes}
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    if not label_to_id:
        raise ValueError("dataset contains no labeled objects")

    manifest_path = args.out_dir / MANIFEST_NAME
    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.out_dir, prefix=f".{MANIFEST_NAME}-", delete=False
    ) as manifest_temp:
        temp_manifest_path = Path(manifest_temp.name)
    write_manifest(temp_manifest_path, annotations, split_by_id, validation_groups)
    if manifest_path.exists() and not args.force:
        if manifest_path.read_bytes() != temp_manifest_path.read_bytes():
            temp_manifest_path.unlink()
            raise ValueError(f"existing manifest differs: {manifest_path}; use --force")
        temp_manifest_path.unlink()
    else:
        temp_manifest_path.replace(manifest_path)

    print(f"Validation sequence groups: {', '.join(validation_groups)}")
    print(f"Manifest SHA256: {file_sha256(manifest_path)}")
    print(f"Split counts: {dict(sorted(Counter(split_by_id.values()).items()))}")
    empty_annotations = sum(not annotation.boxes for annotation in annotations.values())
    invalid_boxes = sum(
        len(annotation.dropped_boxes) for annotation in annotations.values()
    )
    print(
        f"Invalid source boxes excluded: {invalid_boxes}; "
        f"empty annotations after filtering: {empty_annotations}"
    )

    for modality in MODALITIES:
        dataset_dir = args.out_dir / f"LLVIP-YOLO-{modality}"
        if args.force or not dataset_dir.exists():
            dataset_dir, link_modes = build_modality(
                annotations,
                split_by_id,
                args.raw_dir,
                args.out_dir,
                modality,
                label_to_id,
            )
            print(f"Built {dataset_dir} using {dict(link_modes)}")
        counts = validate_output(dataset_dir, annotations, split_by_id)
        print(f"Validated {dataset_dir}: {dict(counts)}")

    print("Dataset preparation complete")


if __name__ == "__main__":
    main()
