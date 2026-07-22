"""Select a deterministic, dataset-only 100-pair LLVIP pilot sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proportional_allocation(
    weights: dict[object, int], total: int
) -> dict[object, int]:
    """Hamilton allocation with deterministic key-based tie breaking."""
    weight_total = sum(weights.values())
    exact = {key: total * weight / weight_total for key, weight in weights.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    ranked = sorted(
        weights,
        key=lambda key: (-(exact[key] - allocation[key]), str(key)),
    )
    for key in ranked[:remaining]:
        allocation[key] += 1
    return allocation


def evenly_spaced(items: list[str], count: int) -> list[str]:
    if count == 0:
        return []
    if count > len(items):
        raise ValueError("cannot sample more items than a stratum contains")
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("systematic sample produced duplicate positions")
    return [items[index] for index in indices]


def crowd_bin(count: int) -> str:
    if count <= 2:
        return str(count)
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


def visible_brightness(path: Path) -> float:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((64, 64))
        return float(ImageStat.Stat(grayscale).mean[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("datasets/LLVIP-splits-v1.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("manifests/LLVIP-pilot-100-v1.json")
    )
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_manifest = json.loads(args.split_manifest.read_text())
    test_ids = sorted(
        record["image_id"]
        for record in split_manifest["records"]
        if record["split"] == "test"
    )
    if args.size <= 0 or args.size > len(test_ids):
        raise ValueError(f"sample size must be in [1, {len(test_ids)}]")

    visible_root = args.dataset_root / "LLVIP-YOLO-visible"
    attributes = {}
    for image_id in test_ids:
        label_path = visible_root / "labels" / "test" / f"{image_id}.txt"
        image_path = visible_root / "images" / "test" / f"{image_id}.jpg"
        attributes[image_id] = {
            "group_id": image_id[:2],
            "crowd_count": len(label_path.read_text().splitlines()),
            "visible_brightness": visible_brightness(image_path),
        }

    brightness_order = sorted(
        test_ids,
        key=lambda image_id: (attributes[image_id]["visible_brightness"], image_id),
    )
    for rank, image_id in enumerate(brightness_order):
        attributes[image_id]["brightness_quintile"] = min(4, rank * 5 // len(test_ids))
        attributes[image_id]["crowd_bin"] = crowd_bin(
            attributes[image_id]["crowd_count"]
        )

    ids_by_group = defaultdict(list)
    for image_id in test_ids:
        ids_by_group[attributes[image_id]["group_id"]].append(image_id)
    group_allocation = proportional_allocation(
        {group: len(ids) for group, ids in ids_by_group.items()}, args.size
    )

    selected = []
    stratum_allocations = {}
    for group in sorted(ids_by_group):
        strata = defaultdict(list)
        for image_id in ids_by_group[group]:
            item = attributes[image_id]
            key = (item["crowd_bin"], item["brightness_quintile"])
            strata[key].append(image_id)
        allocation = proportional_allocation(
            {key: len(ids) for key, ids in strata.items()}, group_allocation[group]
        )
        stratum_allocations[group] = {
            f"crowd={key[0]},brightness_q={key[1]}": count
            for key, count in sorted(allocation.items())
            if count
        }
        for key in sorted(strata):
            selected.extend(evenly_spaced(sorted(strata[key]), allocation[key]))

    if len(selected) != args.size or len(set(selected)) != args.size:
        raise ValueError("pilot selection did not produce the requested unique count")
    selected.sort()
    records = [
        {
            "image_id": image_id,
            "group_id": attributes[image_id]["group_id"],
            "crowd_count": attributes[image_id]["crowd_count"],
            "crowd_bin": attributes[image_id]["crowd_bin"],
            "visible_brightness": round(attributes[image_id]["visible_brightness"], 6),
            "brightness_quintile": attributes[image_id]["brightness_quintile"],
        }
        for image_id in selected
    ]
    output = {
        "schema_version": 1,
        "dataset": "LLVIP",
        "source_split_manifest": str(args.split_manifest),
        "source_split_manifest_sha256": sha256(args.split_manifest),
        "source_split": "official test",
        "selection_uses_model_results": False,
        "method": (
            "Hamilton allocation by capture-sequence group, then by visible-brightness "
            "quintile and ground-truth crowd bin; evenly spaced image IDs per stratum"
        ),
        "size": args.size,
        "group_allocation": dict(sorted(group_allocation.items())),
        "stratum_allocation": stratum_allocations,
        "sample_attribute_counts": {
            "group_id": dict(
                sorted(Counter(item["group_id"] for item in records).items())
            ),
            "crowd_bin": dict(
                sorted(Counter(item["crowd_bin"] for item in records).items())
            ),
            "brightness_quintile": dict(
                sorted(
                    Counter(
                        str(item["brightness_quintile"]) for item in records
                    ).items()
                )
            ),
        },
        "records": records,
    }
    serialized = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and not args.force:
        if args.output.read_text() != serialized:
            raise ValueError(f"existing sample differs: {args.output}; use --force")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(f"Selected {args.size} pairs in {args.output}")
    print(f"Pilot manifest SHA256: {sha256(args.output)}")
    print(json.dumps(output["sample_attribute_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
