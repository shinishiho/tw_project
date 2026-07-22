#!/usr/bin/env python3
"""Validate, analyze, and plot FLIR ADAS external-domain predictions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.ap import coco_style_ap  # noqa: E402
from evaluation.bootstrap import (  # noqa: E402
    bootstrap_intervals,
    build_image_statistics,
    independent_paired_difference_of_differences,
    paired_bootstrap_differences,
    subset_image_statistics,
)
from evaluation.flir import (  # noqa: E402
    DATASET_ID,
    DATASET_RELEASE,
    DATASET_SPLIT,
    groups_from_manifest,
    load_flir_manifest,
)
from evaluation.io import load_yolo_ground_truth, read_prediction_jsonl  # noqa: E402
from evaluation.matching import match_boxes_with_ignored_regions  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = {
    "yolo_pretrained_infrared": "yolo_pretrained_infrared.jsonl",
    "yolo_finetuned_infrared": "yolo_finetuned_infrared.jsonl",
    "locate_anything_infrared": "locate_anything_infrared.jsonl",
}
AP_FILES = {
    "yolo_pretrained_infrared": "yolo_pretrained_infrared_ap.jsonl",
    "yolo_finetuned_infrared": "yolo_finetuned_infrared_ap.jsonl",
}
COMPARISONS = {
    "finetuned_minus_locate_anything": (
        "yolo_finetuned_infrared",
        "locate_anything_infrared",
    ),
    "finetuned_minus_pretrained": (
        "yolo_finetuned_infrared",
        "yolo_pretrained_infrared",
    ),
    "locate_anything_minus_pretrained": (
        "locate_anything_infrared",
        "yolo_pretrained_infrared",
    ),
}
L40S_USD_PER_SECOND = 0.000542


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _validate_run(records, expected_ids: set[str], manifest_hash: str) -> None:
    ids = [record.image_id for record in records]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("prediction records do not exactly match selected FLIR IDs")
    identities = {
        (record.run_id, record.modality, record.model_id, record.model_revision)
        for record in records
    }
    if len(identities) != 1:
        raise ValueError("prediction file contains multiple run identities")
    for record in records:
        expected = {
            "dataset_id": DATASET_ID,
            "dataset_release": DATASET_RELEASE,
            "dataset_split": DATASET_SPLIT,
            "dataset_manifest_sha256": manifest_hash,
        }
        for key, value in expected.items():
            if record.metadata.get(key) != value:
                raise ValueError(f"prediction {record.image_id} has wrong {key}")
    if any(record.status == "error" for record in records):
        raise ValueError(
            "runtime-error prediction records must be repaired before reporting"
        )


def _size_recall(records, truth, ignored, threshold: float = 0.50) -> dict:
    totals = Counter()
    matched = Counter()
    by_id = {record.image_id: record for record in records}
    for image_id, boxes in truth.items():
        assignment = match_boxes_with_ignored_regions(
            by_id[image_id].boxes, boxes, ignored[image_id], threshold
        )
        matched_truth = {item.ground_truth_index for item in assignment.matches}
        for index, box in enumerate(boxes):
            area = (box.x2 - box.x1) * (box.y2 - box.y1)
            size = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            totals[size] += 1
            matched[size] += index in matched_truth
    return {
        size: {
            "ground_truth_boxes": totals[size],
            "matched_boxes": matched[size],
            "recall": matched[size] / totals[size] if totals[size] else None,
        }
        for size in ("small", "medium", "large")
    }


def _strata(manifest: dict, statistics_by_run: dict) -> dict:
    records = {str(item["image_id"]): item for item in manifest["records"]}
    definitions = {
        "intensity_quintile": [str(value) for value in range(5)],
        "crowd_bin": ["0", "1", "2", "3-4", "5-8", "9+"],
    }
    day_night = sorted(
        {str(item["day_night"]) for item in records.values() if item["day_night"]}
    )
    if day_night:
        definitions["day_night"] = day_night + ["unknown"]
    result = {}
    for attribute, values in definitions.items():
        result[attribute] = {}
        for value in values:
            selected = {
                image_id
                for image_id, record in records.items()
                if (
                    "unknown"
                    if record[attribute] is None or record[attribute] == ""
                    else str(record[attribute])
                )
                == value
            }
            if not selected:
                continue
            result[attribute][value] = {
                run_name: bootstrap_intervals(
                    subset_image_statistics(statistics, selected),
                    replicates=500,
                    seed=20260721,
                )["f1"]
                for run_name, statistics in statistics_by_run.items()
            }
    return result


def _qualitative_selection(records_by_run, truth, ignored, manifest) -> dict:
    assignments = {
        run_name: {
            image_id: match_boxes_with_ignored_regions(
                record.boxes, truth[image_id], ignored[image_id], 0.50
            )
            for image_id, record in ((item.image_id, item) for item in records)
        }
        for run_name, records in records_by_run.items()
    }

    def utility(assignment) -> int:
        return (
            2 * len(assignment.matches)
            - len(assignment.false_positive_indices)
            - len(assignment.false_negative_indices)
        )

    categories = {
        "finetuned_over_locate_anything": (
            "yolo_finetuned_infrared",
            "locate_anything_infrared",
        ),
        "locate_anything_over_finetuned": (
            "locate_anything_infrared",
            "yolo_finetuned_infrared",
        ),
        "finetuned_over_pretrained": (
            "yolo_finetuned_infrared",
            "yolo_pretrained_infrared",
        ),
    }
    chosen = {}
    used = set()
    for category, (left, right) in categories.items():
        ranked = sorted(
            (
                (
                    utility(assignments[left][image_id])
                    - utility(assignments[right][image_id]),
                    image_id,
                )
                for image_id in truth
            ),
            key=lambda item: (-item[0], item[1]),
        )
        ids = [
            image_id for score, image_id in ranked if score > 0 and image_id not in used
        ][:3]
        chosen[category] = ids
        used.update(ids)
    person_negative = {
        str(item["image_id"])
        for item in manifest["records"]
        if not item["person_count"]
    }
    ranked_negative = sorted(
        (
            (
                max(
                    len(assignments[run][image_id].false_positive_indices)
                    for run in records_by_run
                ),
                image_id,
            )
            for image_id in person_negative
        ),
        key=lambda item: (-item[0], item[1]),
    )
    chosen["person_negative_false_positives"] = [
        image_id
        for count, image_id in ranked_negative
        if count > 0 and image_id not in used
    ][:3]
    return {
        "selection_method": "fixed ranking by per-image TP/FP/FN utility at IoU 0.50; no manual cherry-picking",
        "categories": chosen,
        "image_ids": sorted({item for values in chosen.values() for item in values}),
    }


def _render_qualitative(
    selection: dict,
    records_by_run: dict,
    truth: dict,
    ignored: dict,
    dataset_root: Path,
    output_dir: Path,
) -> None:
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    by_run = {
        run_name: {record.image_id: record for record in records}
        for run_name, records in records_by_run.items()
    }
    for image_id in selection["image_ids"]:
        panels = []
        for run_name in RUN_FILES:
            record = by_run[run_name][image_id]
            with Image.open(
                dataset_root / "images" / "val" / f"{image_id}.jpg"
            ).convert("RGB") as source:
                image = source.copy()
            draw = ImageDraw.Draw(image)
            assignment = match_boxes_with_ignored_regions(
                record.boxes, truth[image_id], ignored[image_id], 0.50
            )
            for box in truth[image_id]:
                draw.rectangle(
                    (box.x1, box.y1, box.x2, box.y2), outline="#f59e0b", width=3
                )
            for box in ignored[image_id]:
                draw.rectangle(
                    (box.x1, box.y1, box.x2, box.y2), outline="#a855f7", width=3
                )
            matched_predictions = {item.prediction_index for item in assignment.matches}
            for index, box in enumerate(record.boxes):
                color = (
                    "#38bdf8"
                    if index in matched_predictions
                    else "#6b7280"
                    if index in assignment.ignored_prediction_indices
                    else "#ef4444"
                )
                draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)
            draw.rectangle((0, 0, image.width, 24), fill="black")
            draw.text((6, 6), run_name, fill="white")
            panels.append(image)
        canvas = Image.new(
            "RGB", (sum(item.width for item in panels), panels[0].height)
        )
        offset = 0
        for panel in panels:
            canvas.paste(panel, (offset, 0))
            offset += panel.width
        canvas.save(output_dir / f"{image_id}.jpg", quality=92)


def _plot(summary: dict, figures: Path) -> None:
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    names = list(RUN_FILES)
    values = [summary["runs"][name]["metrics"]["iou_0.50"]["f1"] for name in names]
    estimates = [item["estimate"] for item in values]
    errors = [
        [item["estimate"] - item["low"] for item in values],
        [item["high"] - item["estimate"] for item in values],
    ]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        names,
        estimates,
        yerr=errors,
        capsize=4,
        color=["#64748b", "#0ea5e9", "#f59e0b"],
    )
    axis.set_ylabel("F1 at IoU 0.50")
    axis.set_ylim(0, 1)
    axis.set_title("FLIR ADAS official validation: external-domain thermal detection")
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(figures / "flir-primary-f1.png", dpi=180)
    plt.close(figure)

    comparisons = summary["paired_differences_left_minus_right"]
    labels = list(comparisons)
    intervals = [comparisons[name]["iou_0.50"]["f1"] for name in labels]
    estimates = [item["estimate"] for item in intervals]
    errors = [
        [item["estimate"] - item["low"] for item in intervals],
        [item["high"] - item["estimate"] for item in intervals],
    ]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.errorbar(estimates, labels, xerr=errors, fmt="o", capsize=4, color="#0f766e")
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Paired F1 difference at IoU 0.50")
    axis.set_title("FLIR model differences with clustered bootstrap intervals")
    figure.tight_layout()
    figure.savefig(figures / "flir-paired-f1-differences.png", dpi=180)
    plt.close(figure)


def _llvip_specialization(flir_stats, args) -> dict | None:
    paths = {
        "finetuned": args.llvip_artifacts / "yolo_finetuned_infrared.jsonl",
        "pretrained": args.llvip_artifacts / "yolo_pretrained_infrared.jsonl",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    llvip = {name: read_prediction_jsonl(path) for name, path in paths.items()}
    truth = load_yolo_ground_truth(llvip["finetuned"], args.llvip_dataset, split="test")
    llvip_stats = {
        name: build_image_statistics(records, truth, 0.50)
        for name, records in llvip.items()
    }
    split = json.loads(args.llvip_split_manifest.read_text())
    groups = {
        str(record["image_id"]): str(record["group_id"])
        for record in split["records"]
        if record["split"] == "test"
    }
    return {
        "definition": "(fine-tuned - pretrained on FLIR) - (fine-tuned - pretrained on LLVIP)",
        "f1_iou_0.50": independent_paired_difference_of_differences(
            flir_stats["yolo_finetuned_infrared"],
            flir_stats["yolo_pretrained_infrared"],
            llvip_stats["finetuned"],
            llvip_stats["pretrained"],
            first_groups=args.flir_groups,
            second_groups=groups,
            replicates=args.replicates,
            seed=args.seed + 99,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("pilot", "full"), default="full")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/FLIR-ADAS-v2-val-v1.json")
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=Path("manifests/FLIR-ADAS-v2-pilot-100-v1.json"),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/FLIR-ADAS-v2-YOLO-infrared")
    )
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument(
        "--qualitative-dir", type=Path, default=Path("artifacts/flir/qualitative")
    )
    parser.add_argument("--llvip-artifacts", type=Path, default=Path("artifacts/full"))
    parser.add_argument(
        "--llvip-dataset", type=Path, default=Path("datasets/LLVIP-YOLO-infrared")
    )
    parser.add_argument(
        "--llvip-split-manifest",
        type=Path,
        default=Path("datasets/LLVIP-splits-v1.json"),
    )
    args = parser.parse_args()
    args.artifact_dir = args.artifact_dir or Path("artifacts/flir") / args.sample
    args.json_output = args.json_output or args.artifact_dir / "flir_summary.json"
    args.figures = args.figures or (
        Path("reports/figures")
        if args.sample == "full"
        else args.artifact_dir / "figures"
    )

    manifest, all_truth, all_ignored = load_flir_manifest(args.manifest)
    selected_ids = {str(record["image_id"]) for record in manifest["records"]}
    if args.sample == "pilot":
        pilot = json.loads(args.pilot_manifest.read_text())
        if pilot["source_manifest_sha256"] != manifest["_file_sha256"]:
            raise ValueError("pilot and FLIR manifest hashes differ")
        selected_ids = set(map(str, pilot["image_ids"]))
    records_by_run = {
        name: read_prediction_jsonl(args.artifact_dir / filename)
        for name, filename in RUN_FILES.items()
    }
    for records in records_by_run.values():
        _validate_run(records, selected_ids, manifest["_file_sha256"])
    truth = {image_id: all_truth[image_id] for image_id in selected_ids}
    ignored = {image_id: all_ignored[image_id] for image_id in selected_ids}
    groups = {
        image_id: group_id
        for image_id, group_id in groups_from_manifest(manifest).items()
        if image_id in selected_ids
    }
    args.flir_groups = groups
    grouped_records = sum(
        count
        for method, count in manifest["group_method_counts"].items()
        if method != "image-level fallback"
    )
    bootstrap_method = (
        "paired sequence-group percentile bootstrap"
        if grouped_records
        else "paired image-level percentile bootstrap (no sequence metadata recovered)"
    )
    statistics_by_run = {}
    runs = {}
    for run_name, records in records_by_run.items():
        statistics_by_run[run_name] = build_image_statistics(
            records, truth, 0.50, ignored
        )
        threshold_metrics = {}
        for threshold in (0.50, 0.75):
            statistics_for_threshold = build_image_statistics(
                records, truth, threshold, ignored
            )
            threshold_metrics[f"iou_{threshold:.2f}"] = bootstrap_intervals(
                statistics_for_threshold,
                replicates=args.replicates,
                seed=args.seed + int(threshold * 100),
                groups_by_image=groups,
            )
        latencies = [
            record.latency_ms for record in records if record.latency_ms is not None
        ]
        gpu_seconds = sum(latencies) / 1000
        runs[run_name] = {
            "identity": {
                "run_id": records[0].run_id,
                "model_id": records[0].model_id,
                "model_revision": records[0].model_revision,
            },
            "records": len(records),
            "status_counts": dict(
                sorted(Counter(record.status for record in records).items())
            ),
            "metrics": threshold_metrics,
            "ground_truth_box_size_recall_iou_0.50": _size_recall(
                records, truth, ignored
            ),
            "efficiency": {
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "warm_batch1_images_per_second": len(records) / gpu_seconds,
                "warm_batch1_cost_per_1000_usd": gpu_seconds
                / len(records)
                * 1000
                * L40S_USD_PER_SECOND,
                "peak_gpu_memory_gib": max(
                    int(record.metadata.get("peak_gpu_memory_bytes", 0))
                    for record in records
                )
                / 1024**3,
            },
        }

    comparisons = {
        name: {
            f"iou_{threshold:.2f}": paired_bootstrap_differences(
                build_image_statistics(records_by_run[left], truth, threshold, ignored),
                build_image_statistics(
                    records_by_run[right], truth, threshold, ignored
                ),
                replicates=args.replicates,
                seed=args.seed + index * 10 + int(threshold * 100),
                groups_by_image=groups,
            )
            for threshold in (0.50, 0.75)
        }
        for index, (name, (left, right)) in enumerate(COMPARISONS.items())
    }
    ap = {}
    if args.sample == "full":
        for run_name, filename in AP_FILES.items():
            records = read_prediction_jsonl(args.artifact_dir / filename)
            _validate_run(records, selected_ids, manifest["_file_sha256"])
            ap[run_name] = coco_style_ap(records, truth, ignored)
            runs[run_name]["secondary_yolo_ap"] = ap[run_name]

    specialization = (
        _llvip_specialization(statistics_by_run, args)
        if args.sample == "full"
        else None
    )
    selected_manifest = {
        **manifest,
        "records": [
            record
            for record in manifest["records"]
            if str(record["image_id"]) in selected_ids
        ],
    }
    strata = _strata(selected_manifest, statistics_by_run)
    qualitative = _qualitative_selection(
        records_by_run, truth, ignored, selected_manifest
    )
    args.qualitative_dir.mkdir(parents=True, exist_ok=True)
    (args.qualitative_dir / f"{args.sample}_selection.json").write_text(
        json.dumps(qualitative, indent=2, sort_keys=True) + "\n"
    )
    _render_qualitative(
        qualitative,
        records_by_run,
        truth,
        ignored,
        args.dataset_root,
        args.qualitative_dir / args.sample,
    )
    result = {
        "schema_version": 1,
        "dataset": {
            "id": DATASET_ID,
            "release": DATASET_RELEASE,
            "split": DATASET_SPLIT,
            "manifest_sha256": manifest["_file_sha256"],
            "images": len(selected_ids),
            "person_boxes": sum(len(boxes) for boxes in truth.values()),
            "ignored_person_boxes": sum(len(boxes) for boxes in ignored.values()),
            "person_negative_images": sum(not boxes for boxes in truth.values()),
            "group_count": len(set(groups.values())),
            "grouping": manifest["group_method_counts"],
        },
        "bootstrap": {
            "method": bootstrap_method,
            "replicates": args.replicates,
            "seed": args.seed,
        },
        "runs": runs,
        "paired_differences_left_minus_right": comparisons,
        "strata_iou_0.50": strata,
        "cross_domain_specialization": specialization,
        "qualitative_selection": qualitative,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _plot(result, args.figures)

    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
