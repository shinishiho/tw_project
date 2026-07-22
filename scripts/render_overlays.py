"""Render traceable ground-truth and prediction assignments on LLVIP images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_yolo_ground_truth, read_prediction_jsonl
from evaluation.metrics import evaluate_records


COLORS = {
    "matched_ground_truth": "#22c55e",
    "matched_prediction": "#38bdf8",
    "false_positive": "#ef4444",
    "false_negative": "#f59e0b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required; run `uv pip install -r requirements.txt`"
        ) from error

    records = read_prediction_jsonl(args.predictions)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("no prediction records selected")
    ground_truth = load_yolo_ground_truth(records, args.dataset_dir, args.split)
    summary = evaluate_records(records, ground_truth, args.iou)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        image_path = args.dataset_dir / "images" / args.split / f"{record.image_id}.jpg"
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            assignment = summary.assignments[record.image_id]
            truth = ground_truth[record.image_id]
            for match in assignment.matches:
                ground_truth_box = truth[match.ground_truth_index]
                prediction_box = record.boxes[match.prediction_index]
                draw.rectangle(
                    (
                        ground_truth_box.x1,
                        ground_truth_box.y1,
                        ground_truth_box.x2,
                        ground_truth_box.y2,
                    ),
                    outline=COLORS["matched_ground_truth"],
                    width=5,
                )
                draw.rectangle(
                    (
                        prediction_box.x1,
                        prediction_box.y1,
                        prediction_box.x2,
                        prediction_box.y2,
                    ),
                    outline=COLORS["matched_prediction"],
                    width=3,
                )
                draw.text(
                    (prediction_box.x1, max(0, prediction_box.y1 - 12)),
                    f"TP IoU={match.iou:.2f}",
                    fill=COLORS["matched_prediction"],
                    stroke_width=2,
                    stroke_fill="black",
                )
            for index in assignment.false_positive_indices:
                box = record.boxes[index]
                draw.rectangle(
                    (box.x1, box.y1, box.x2, box.y2),
                    outline=COLORS["false_positive"],
                    width=4,
                )
                draw.text(
                    (box.x1, max(0, box.y1 - 12)),
                    "FP",
                    fill=COLORS["false_positive"],
                    stroke_width=2,
                    stroke_fill="black",
                )
            for index in assignment.false_negative_indices:
                box = truth[index]
                draw.rectangle(
                    (box.x1, box.y1, box.x2, box.y2),
                    outline=COLORS["false_negative"],
                    width=4,
                )
                draw.text(
                    (box.x1, max(0, box.y1 - 12)),
                    "FN",
                    fill=COLORS["false_negative"],
                    stroke_width=2,
                    stroke_fill="black",
                )

            legend = "GT matched=green  prediction matched=blue  FP=red  FN=orange"
            draw.rectangle((0, 0, 560, 22), fill="black")
            draw.text((6, 5), legend, fill="white")
            image.save(args.output_dir / f"{record.image_id}.jpg", quality=92)
    print(f"Rendered {len(records)} traceable overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
