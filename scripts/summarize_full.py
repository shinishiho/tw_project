"""Validate and summarize the six locked full-test prediction files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bootstrap import (  # noqa: E402
    aggregate_metrics,
    bootstrap_intervals,
    build_image_statistics,
    paired_bootstrap_differences,
    subset_image_statistics,
)
from evaluation.io import load_yolo_ground_truth, read_prediction_jsonl  # noqa: E402
from evaluation.matching import match_boxes  # noqa: E402


EXPECTED_IMAGES = 3_463
L40S_USD_PER_SECOND = 0.000542
RUN_FILES = {
    "yolo_pretrained_visible": "yolo_pretrained_visible.jsonl",
    "yolo_pretrained_infrared": "yolo_pretrained_infrared.jsonl",
    "yolo_finetuned_visible": "yolo_finetuned_visible.jsonl",
    "yolo_finetuned_infrared": "yolo_finetuned_infrared.jsonl",
    "locate_anything_visible": "locate_anything_visible.jsonl",
    "locate_anything_infrared": "locate_anything_infrared.jsonl",
}
AP_FILES = {
    "yolo_pretrained_visible": "yolo_ap_pretrained_visible.json",
    "yolo_pretrained_infrared": "yolo_ap_pretrained_infrared.json",
    "yolo_finetuned_visible": "yolo_ap_finetuned_visible.json",
    "yolo_finetuned_infrared": "yolo_ap_finetuned_infrared.json",
}
COMPARISONS = {
    "visible_headline_yolo_pretrained_minus_locate_anything": (
        "yolo_pretrained_visible",
        "locate_anything_visible",
    ),
    "thermal_headline_yolo_finetuned_minus_locate_anything": (
        "yolo_finetuned_infrared",
        "locate_anything_infrared",
    ),
    "pretrained_modality_visible_minus_infrared": (
        "yolo_pretrained_visible",
        "yolo_pretrained_infrared",
    ),
    "thermal_supervision_finetuned_minus_pretrained": (
        "yolo_finetuned_infrared",
        "yolo_pretrained_infrared",
    ),
    "finetuned_modality_infrared_minus_visible": (
        "yolo_finetuned_infrared",
        "yolo_finetuned_visible",
    ),
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _fmt_interval(value: dict[str, float]) -> str:
    return f"{value['estimate']:.3f} [{value['low']:.3f}, {value['high']:.3f}]"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_size_recall(records, ground_truth, iou_threshold: float) -> dict[str, dict]:
    totals = Counter()
    matched = Counter()
    records_by_id = {record.image_id: record for record in records}
    for image_id, truth in ground_truth.items():
        matches = match_boxes(records_by_id[image_id].boxes, truth, iou_threshold)
        matched_truth = {item.ground_truth_index for item in matches.matches}
        for index, box in enumerate(truth):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--ap-dir", type=Path, default=Path("artifacts/full/ap"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("manifests/LLVIP-test-attributes-v1.json"),
    )
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/full/full_summary.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/RESULTS.md"))
    args = parser.parse_args()

    attribute_manifest = json.loads(args.attributes.read_text())
    attribute_records = {
        record["image_id"]: record for record in attribute_manifest["records"]
    }
    if len(attribute_records) != EXPECTED_IMAGES:
        raise ValueError("test attribute manifest does not contain 3,463 unique IDs")
    attribute_groups = {
        "brightness_quintile": {
            str(value): {
                image_id
                for image_id, record in attribute_records.items()
                if str(record["brightness_quintile"]) == str(value)
            }
            for value in range(5)
        },
        "crowd_bin": {
            value: {
                image_id
                for image_id, record in attribute_records.items()
                if record["crowd_bin"] == value
            }
            for value in ("1", "2", "3-4", "5-8", "9+")
        },
    }
    ap_results = {
        run_name: json.loads((args.ap_dir / filename).read_text())
        for run_name, filename in AP_FILES.items()
    }

    records_by_run = {}
    locked_ids: set[str] | None = None
    for run_name, filename in RUN_FILES.items():
        path = args.artifact_dir / filename
        records = read_prediction_jsonl(path)
        image_ids = [record.image_id for record in records]
        run_keys = {
            (
                record.run_id,
                record.modality,
                record.model_id,
                record.model_revision,
            )
            for record in records
        }
        if len(records) != EXPECTED_IMAGES:
            raise ValueError(f"expected {EXPECTED_IMAGES} records in {path}")
        if len(set(image_ids)) != EXPECTED_IMAGES:
            raise ValueError(f"duplicate image IDs in {path}")
        if len(run_keys) != 1:
            raise ValueError(f"multiple run identities in {path}")
        if locked_ids is None:
            locked_ids = set(image_ids)
        elif set(image_ids) != locked_ids:
            raise ValueError(f"paired image IDs differ in {path}")
        records_by_run[run_name] = records

    statistics_by_run = {}
    runs = {}
    for run_index, (run_name, records) in enumerate(records_by_run.items()):
        modality = records[0].modality
        ground_truth = load_yolo_ground_truth(
            records, args.dataset_root / f"LLVIP-YOLO-{modality}"
        )
        threshold_results = {}
        for threshold_index, threshold in enumerate((0.50, 0.75)):
            image_statistics = build_image_statistics(records, ground_truth, threshold)
            statistics_by_run[(run_name, threshold)] = image_statistics
            threshold_results[f"iou_{threshold:.2f}"] = bootstrap_intervals(
                image_statistics,
                replicates=args.replicates,
                seed=args.seed + run_index * 10 + threshold_index,
            )
        primary_statistics = statistics_by_run[(run_name, 0.50)]
        strata = {
            attribute: {
                value: {
                    "images": len(image_ids),
                    "metrics": aggregate_metrics(
                        subset_image_statistics(primary_statistics, image_ids)
                    ),
                }
                for value, image_ids in groups.items()
            }
            for attribute, groups in attribute_groups.items()
        }
        latencies = [
            record.latency_ms for record in records if record.latency_ms is not None
        ]
        if len(latencies) != EXPECTED_IMAGES:
            raise ValueError(f"missing batch-1 latency in {run_name}")
        warm_gpu_seconds = sum(latencies) / 1000
        peak_memory = max(
            int(record.metadata["peak_gpu_memory_bytes"]) for record in records
        )
        diagnostics = Counter()
        for record in records:
            diagnostics.update(record.parser_diagnostics)
        identity = next(
            iter(
                {
                    (
                        record.run_id,
                        record.modality,
                        record.model_id,
                        record.model_revision,
                    )
                    for record in records
                }
            )
        )
        runs[run_name] = {
            "run_id": identity[0],
            "modality": identity[1],
            "model_id": identity[2],
            "model_revision": identity[3],
            "records": len(records),
            "status_counts": dict(sorted(Counter(r.status for r in records).items())),
            "parser_diagnostics": dict(sorted(diagnostics.items())),
            "box_count": sum(len(record.boxes) for record in records),
            "metrics": threshold_results,
            "strata_iou_0.50": strata,
            "ground_truth_box_size_recall_iou_0.50": _box_size_recall(
                records, ground_truth, 0.50
            ),
            "efficiency": {
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "warm_batch1_gpu_seconds": warm_gpu_seconds,
                "warm_batch1_images_per_second": len(records) / warm_gpu_seconds,
                "warm_batch1_cost_per_1000_usd": (
                    warm_gpu_seconds / len(records) * 1000 * L40S_USD_PER_SECOND
                ),
                "peak_gpu_memory_gib": peak_memory / 1024**3,
            },
        }
        if run_name in ap_results:
            ap = ap_results[run_name]
            if (
                ap["records"] != EXPECTED_IMAGES
                or ap["model_revision"] != identity[3]
                or ap["modality"] != modality
            ):
                raise ValueError(f"YOLO AP artifact identity mismatch for {run_name}")
            runs[run_name]["secondary_yolo_ap"] = {
                "ap50": ap["map50"],
                "ap75": ap["map75"],
                "map50_95": ap["map50_95"],
                "settings": ap["settings"],
                "artifact": str(args.ap_dir / AP_FILES[run_name]),
                "artifact_sha256": _sha256(args.ap_dir / AP_FILES[run_name]),
            }

    comparisons = {}
    for comparison_index, (name, (left, right)) in enumerate(COMPARISONS.items()):
        comparisons[name] = {}
        for threshold_index, threshold in enumerate((0.50, 0.75)):
            comparisons[name][f"iou_{threshold:.2f}"] = paired_bootstrap_differences(
                statistics_by_run[(left, threshold)],
                statistics_by_run[(right, threshold)],
                replicates=args.replicates,
                seed=args.seed + 10_000 + comparison_index * 10 + threshold_index,
            )

    result = {
        "schema_version": 1,
        "test_images": EXPECTED_IMAGES,
        "bootstrap": {
            "method": "paired image-level percentile bootstrap",
            "replicates": args.replicates,
            "seed": args.seed,
            "confidence": 0.95,
        },
        "test_attributes": {
            "path": str(args.attributes),
            "sha256": _sha256(args.attributes),
            "uses_model_results": attribute_manifest["uses_model_results"],
        },
        "runs": runs,
        "paired_differences_left_minus_right": comparisons,
        "cost_rate": {
            "gpu": "L40S",
            "usd_per_second": L40S_USD_PER_SECOND,
            "date": "2026-07-22",
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Locked Full-Test Results",
        "",
        f"All six runs contain exactly {EXPECTED_IMAGES:,} paired official test IDs. "
        f"Intervals are 95% paired image-level percentile bootstrap intervals "
        f"({args.replicates:,} replicates; seed {args.seed}).",
        "",
        "## Primary confidence-independent metrics at IoU 0.50",
        "",
        "| Run | P [95% CI] | R [95% CI] | F1 [95% CI] | Mean matched IoU | FP/image | FN/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, run in runs.items():
        metrics = run["metrics"]["iou_0.50"]
        lines.append(
            f"| {name} | {_fmt_interval(metrics['precision'])} | "
            f"{_fmt_interval(metrics['recall'])} | {_fmt_interval(metrics['f1'])} | "
            f"{_fmt_interval(metrics['mean_matched_iou'])} | "
            f"{_fmt_interval(metrics['false_positives_per_image'])} | "
            f"{_fmt_interval(metrics['false_negatives_per_image'])} |"
        )
    lines.extend(
        [
            "",
            "![Primary F1 estimates with paired bootstrap intervals]"
            "(figures/primary-f1.png)",
            "",
            "### Output audit",
            "",
            "| Run | Status counts | Duplicate-box rate | Malformed rate | No-output rate | Error rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, run in runs.items():
        metrics = run["metrics"]["iou_0.50"]
        status_counts = ", ".join(
            f"{status}={count}" for status, count in run["status_counts"].items()
        )
        lines.append(
            f"| {name} | {status_counts} | "
            f"{metrics['duplicate_box_rate']['estimate']:.4f} | "
            f"{metrics['malformed_output_rate']['estimate']:.4f} | "
            f"{metrics['no_output_rate']['estimate']:.4f} | "
            f"{metrics['error_output_rate']['estimate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset-only strata at IoU 0.50",
            "",
            "Brightness quintiles use visible-image intensity for both paired modalities "
            "(Q0 darkest, Q4 brightest). Values are F1 point estimates.",
            "",
            "| Run | Q0 | Q1 | Q2 | Q3 | Q4 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, run in runs.items():
        brightness = run["strata_iou_0.50"]["brightness_quintile"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{brightness[str(q)]['metrics']['f1']:.3f}" for q in range(5))
            + " |"
        )
    lines.extend(
        [
            "",
            "| Run | 1 person | 2 people | 3-4 | 5-8 | 9+ |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, run in runs.items():
        crowd = run["strata_iou_0.50"]["crowd_bin"]
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{crowd[value]['metrics']['f1']:.3f}"
                for value in ("1", "2", "3-4", "5-8", "9+")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Object-size values are ground-truth recall using COCO native-pixel area "
            "thresholds. LLVIP's locked test labels contain no small boxes under this definition.",
            "",
            "| Run | Medium recall | Large recall |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, run in runs.items():
        size_recall = run["ground_truth_box_size_recall_iou_0.50"]
        lines.append(
            f"| {name} | {size_recall['medium']['recall']:.3f} | "
            f"{size_recall['large']['recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "![F1 across brightness and crowd strata]"
            "(figures/stratified-f1.png)",
            "",
            "## Paired F1 differences (left minus right)",
            "",
            "| Comparison | IoU 0.50 | IoU 0.75 |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, differences in comparisons.items():
        lines.append(
            f"| {name} | {_fmt_interval(differences['iou_0.50']['f1'])} | "
            f"{_fmt_interval(differences['iou_0.75']['f1'])} |"
        )
    lines.extend(
        [
            "",
            "![Paired F1 differences with confidence intervals]"
            "(figures/paired-f1-differences.png)",
            "",
            "## Secondary YOLO confidence-sweep metrics",
            "",
            "| Run | AP50 | AP75 | mAP50-95 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in AP_FILES:
        ap = runs[name]["secondary_yolo_ap"]
        lines.append(
            f"| {name} | {ap['ap50']:.3f} | {ap['ap75']:.3f} | {ap['map50_95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Warm batch-1 efficiency",
            "",
            "| Run | Median ms | p95 ms | Images/s | Peak GiB | Cost/1,000 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, run in runs.items():
        efficiency = run["efficiency"]
        lines.append(
            f"| {name} | {efficiency['median_latency_ms']:.1f} | "
            f"{efficiency['p95_latency_ms']:.1f} | "
            f"{efficiency['warm_batch1_images_per_second']:.3f} | "
            f"{efficiency['peak_gpu_memory_gib']:.2f} | "
            f"${efficiency['warm_batch1_cost_per_1000_usd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Accuracy and warm batch-1 latency]"
            "(figures/accuracy-latency.png)",
            "",
            "Efficiency uses per-record warm batch-1 latency and Modal's July 22, "
            "2026 L40S GPU rate of $0.000542/second. It excludes cold start, host "
            "CPU/memory, storage, and optimized batch throughput.",
            "",
            "YOLO AP metrics come from a separate confidence-0.001 sweep and are not "
            "inferred from the confidence-0.25 primary records.",
            "",
            "Fixed disagreement selection and inspected overlays are documented in "
            "`reports/QUALITATIVE.md`.",
            "",
            "Current pricing: https://modal.com/pricing",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(f"Wrote {args.json_output} and {args.report}")


if __name__ == "__main__":
    main()
