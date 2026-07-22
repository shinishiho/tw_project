"""Build dataset-only strata for the locked LLVIP paired test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageStat


SMALL_AREA = 32**2
MEDIUM_AREA = 96**2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_brightness(path: Path) -> float:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((64, 64))
        return float(ImageStat.Stat(grayscale).mean[0])


def crowd_bin(count: int) -> str:
    if count <= 2:
        return str(count)
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


def box_size(area: float) -> str:
    if area < SMALL_AREA:
        return "small"
    if area < MEDIUM_AREA:
        return "medium"
    return "large"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("datasets/LLVIP-splits-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/LLVIP-test-attributes-v1.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_manifest = json.loads(args.split_manifest.read_text())
    image_ids = sorted(
        record["image_id"]
        for record in split_manifest["records"]
        if record["split"] == "test"
    )
    visible_root = args.dataset_root / "LLVIP-YOLO-visible"
    attributes = []
    for image_id in image_ids:
        image_path = visible_root / "images" / "test" / f"{image_id}.jpg"
        label_path = visible_root / "labels" / "test" / f"{image_id}.txt"
        with Image.open(image_path) as image:
            width, height = image.size
        size_counts = Counter()
        for line in label_path.read_text().splitlines():
            _, _, _, normalized_width, normalized_height = line.split()
            area = float(normalized_width) * width * float(normalized_height) * height
            size_counts[box_size(area)] += 1
        crowd_count = sum(size_counts.values())
        attributes.append(
            {
                "image_id": image_id,
                "group_id": image_id[:2],
                "visible_brightness": round(visible_brightness(image_path), 6),
                "crowd_count": crowd_count,
                "crowd_bin": crowd_bin(crowd_count),
                "ground_truth_box_size_counts": {
                    name: size_counts[name] for name in ("small", "medium", "large")
                },
            }
        )

    brightness_order = sorted(
        attributes,
        key=lambda record: (record["visible_brightness"], record["image_id"]),
    )
    for rank, record in enumerate(brightness_order):
        record["brightness_quintile"] = min(4, rank * 5 // len(attributes))
    attributes.sort(key=lambda record: record["image_id"])

    output = {
        "schema_version": 1,
        "dataset": "LLVIP",
        "source_split": "official test",
        "source_split_manifest": str(args.split_manifest),
        "source_split_manifest_sha256": sha256(args.split_manifest),
        "uses_model_results": False,
        "brightness_method": "mean visible-image grayscale intensity after 64x64 resize; rank quintiles 0=darkest through 4=brightest",
        "crowd_bins": {
            "1": [1, 1],
            "2": [2, 2],
            "3-4": [3, 4],
            "5-8": [5, 8],
            "9+": [9, None],
        },
        "box_size_method": "COCO native-pixel area thresholds: small < 32^2, medium < 96^2, large >= 96^2",
        "size": len(attributes),
        "attribute_counts": {
            "brightness_quintile": dict(
                sorted(
                    Counter(str(r["brightness_quintile"]) for r in attributes).items()
                )
            ),
            "crowd_bin": dict(
                sorted(Counter(r["crowd_bin"] for r in attributes).items())
            ),
            "ground_truth_box_size": dict(
                sum(
                    (Counter(r["ground_truth_box_size_counts"]) for r in attributes),
                    Counter(),
                )
            ),
        },
        "records": attributes,
    }
    serialized = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and not args.force:
        if args.output.read_text() != serialized:
            raise ValueError(f"existing attributes differ: {args.output}; use --force")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(f"Wrote {len(attributes)} locked test attributes to {args.output}")
    print(f"Attribute manifest SHA256: {sha256(args.output)}")
    print(json.dumps(output["attribute_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
