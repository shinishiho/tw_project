"""Evaluate unified prediction JSONL against prepared LLVIP YOLO labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_yolo_ground_truth, read_prediction_jsonl
from evaluation.metrics import evaluate_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--iou",
        type=float,
        nargs="+",
        default=(0.50, 0.75),
        help="one or more IoU thresholds (default: 0.50 0.75)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_prediction_jsonl(args.predictions)
    if not records:
        raise ValueError(f"no prediction records found in {args.predictions}")
    run_keys = {
        (record.run_id, record.model_id, record.model_revision, record.modality)
        for record in records
    }
    if len(run_keys) != 1:
        raise ValueError(
            "a prediction file must contain exactly one run/model/revision/modality"
        )
    ground_truth = load_yolo_ground_truth(records, args.dataset_dir, args.split)
    summaries = {
        f"iou_{threshold:.2f}": asdict(
            evaluate_records(records, ground_truth, iou_threshold=threshold)
        )
        for threshold in args.iou
    }
    result = {
        "schema_version": 1,
        "predictions": str(args.predictions),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "run": dict(
            zip(
                ("run_id", "model_id", "model_revision", "modality"),
                next(iter(run_keys)),
            )
        ),
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote traceable evaluation to {args.output}")


if __name__ == "__main__":
    main()
