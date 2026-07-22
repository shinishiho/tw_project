"""Summarize the six locked 100-pair pilot prediction files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_yolo_ground_truth, read_prediction_jsonl
from evaluation.metrics import evaluate_records


L40S_USD_PER_SECOND = 0.000542
FULL_TEST_IMAGES = 3_463


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/modal-pilot")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/modal-pilot/pilot_summary.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/PILOT.md"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.artifact_dir.glob("*_100.jsonl")):
        records = read_prediction_jsonl(path)
        if len(records) != 100:
            raise ValueError(f"expected 100 records in {path}, found {len(records)}")
        modality = records[0].modality
        ground_truth = load_yolo_ground_truth(
            records, args.dataset_root / f"LLVIP-YOLO-{modality}"
        )
        metric_50 = evaluate_records(records, ground_truth, 0.50)
        metric_75 = evaluate_records(records, ground_truth, 0.75)
        latencies = sorted(record.latency_ms or 0 for record in records)
        warm_gpu_seconds = sum(latencies) / 1000
        projected_seconds = warm_gpu_seconds / len(records) * FULL_TEST_IMAGES
        statuses = Counter(record.status for record in records)
        rows.append(
            {
                "run": path.stem,
                "model_id": records[0].model_id,
                "model_revision": records[0].model_revision,
                "modality": modality,
                "records": len(records),
                "status_counts": dict(sorted(statuses.items())),
                "box_count": sum(len(record.boxes) for record in records),
                "precision_50": metric_50.precision,
                "recall_50": metric_50.recall,
                "f1_50": metric_50.f1,
                "f1_75": metric_75.f1,
                "mean_matched_iou_50": metric_50.mean_matched_iou,
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": latencies[math.ceil(0.95 * len(latencies)) - 1],
                "peak_gpu_memory_gib": max(
                    record.metadata["peak_gpu_memory_bytes"] for record in records
                )
                / 1024**3,
                "warm_gpu_seconds_100": warm_gpu_seconds,
                "projected_full_gpu_seconds": projected_seconds,
                "projected_full_gpu_cost_usd": projected_seconds * L40S_USD_PER_SECOND,
            }
        )

    if len(rows) != 6:
        raise ValueError(f"expected six pilot combinations, found {len(rows)}")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Locked 100-Pair Pilot",
        "",
        "Generated from `manifests/LLVIP-pilot-100-v1.json`. This is a dataset-only, "
        "sequence/brightness/crowd-stratified pilot; it is not final test evidence.",
        "",
        "| Model state | Modality | P@.50 | R@.50 | F1@.50 | F1@.75 | Median ms | p95 ms | Peak GiB | Projected full cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {modality} | {precision:.3f} | {recall:.3f} | {f1_50:.3f} | "
            "{f1_75:.3f} | {median:.1f} | {p95:.1f} | {memory:.2f} | ${cost:.2f} |".format(
                model=row["model_id"],
                modality=row["modality"],
                precision=row["precision_50"],
                recall=row["recall_50"],
                f1_50=row["f1_50"],
                f1_75=row["f1_75"],
                median=row["median_latency_ms"],
                p95=row["p95_latency_ms"],
                memory=row["peak_gpu_memory_gib"],
                cost=row["projected_full_gpu_cost_usd"],
            )
        )
    lines.extend(
        [
            "",
            "Cost uses Modal's July 22, 2026 L40S rate of $0.000542/second and warm "
            "batch-1 model time. It excludes container startup, model download, CPU, "
            "memory, storage, and any future batch-throughput optimization.",
            "",
            "Two LocateAnything records returned the explicit no-object token; there "
            "were no malformed or error records after parser normalization.",
            "",
            "The `old-test-leaked-best.pt` row is pipeline-only context. That checkpoint "
            "used the official test split for selection and cannot support final claims.",
            "",
            "A three-seed probe on the first ten pilot pairs found identical raw output "
            "for 7/10 images in each modality. F1@.50 ranged from 0.898 to 0.917 on "
            "visible and stayed at 0.939 on infrared, so LocateAnything sampling is not "
            "strictly deterministic even with otherwise fixed settings.",
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
