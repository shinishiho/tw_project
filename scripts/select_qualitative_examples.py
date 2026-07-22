"""Select deterministic paired disagreement examples from locked full-test runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_yolo_ground_truth, read_prediction_jsonl  # noqa: E402
from evaluation.matching import match_boxes  # noqa: E402


COMPARISONS = {
    "visible_headline": (
        "visible",
        "yolo_pretrained_visible.jsonl",
        "locate_anything_visible.jsonl",
    ),
    "thermal_headline": (
        "infrared",
        "yolo_finetuned_infrared.jsonl",
        "locate_anything_infrared.jsonl",
    ),
}

COLORS = {
    "matched_ground_truth": "#22c55e",
    "matched_prediction": "#38bdf8",
    "false_positive": "#ef4444",
    "false_negative": "#f59e0b",
}


def _image_metrics(record, truth) -> dict[str, float | int]:
    matches = match_boxes(record.boxes, truth, 0.50)
    true_positives = len(matches.matches)
    false_positives = len(matches.false_positive_indices)
    false_negatives = len(matches.false_negative_indices)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "f1": f1,
    }


def _take_unique(
    candidates: list[tuple[float, str]], count: int, selected: set[str]
) -> list[str]:
    chosen = []
    for _, image_id in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if image_id in selected:
            continue
        chosen.append(image_id)
        selected.add(image_id)
        if len(chosen) == count:
            break
    return chosen


def _render_overlays(records, ground_truth, dataset_dir: Path, output_dir: Path) -> None:
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        image_path = dataset_dir / "images" / "test" / f"{record.image_id}.jpg"
        matches = match_boxes(record.boxes, ground_truth[record.image_id], 0.50)
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            truth = ground_truth[record.image_id]
            for match in matches.matches:
                truth_box = truth[match.ground_truth_index]
                prediction_box = record.boxes[match.prediction_index]
                draw.rectangle(
                    (truth_box.x1, truth_box.y1, truth_box.x2, truth_box.y2),
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
            for index in matches.false_positive_indices:
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
            for index in matches.false_negative_indices:
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
            draw.rectangle((0, 0, 560, 22), fill="black")
            draw.text(
                (6, 5),
                "GT matched=green  prediction matched=blue  FP=red  FN=orange",
                fill="white",
            )
            image.save(output_dir / f"{record.image_id}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/full/qualitative")
    )
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()
    if args.per_category <= 0:
        raise ValueError("per-category must be positive")

    output = {
        "schema_version": 1,
        "selection_uses_model_results": True,
        "iou_threshold": 0.50,
        "method": (
            "deterministic extremes by per-image F1 difference, both-model F1, "
            "shared misses, and LocateAnything parser diagnostics"
        ),
        "comparisons": {},
    }
    subset_records = {}
    ground_truth_by_file = {}
    dataset_dir_by_file = {}
    for comparison_name, (modality, yolo_file, locate_file) in COMPARISONS.items():
        yolo_records = read_prediction_jsonl(args.artifact_dir / yolo_file)
        locate_records = read_prediction_jsonl(args.artifact_dir / locate_file)
        yolo_by_id = {record.image_id: record for record in yolo_records}
        locate_by_id = {record.image_id: record for record in locate_records}
        if set(yolo_by_id) != set(locate_by_id):
            raise ValueError(f"paired IDs differ for {comparison_name}")
        ground_truth = load_yolo_ground_truth(
            yolo_records, args.dataset_root / f"LLVIP-YOLO-{modality}"
        )
        measurements = {}
        categories: dict[str, list[tuple[float, str]]] = {
            "yolo_advantage": [],
            "locate_anything_advantage": [],
            "both_strong": [],
            "both_miss": [],
            "locate_anything_duplicates": [],
            "locate_anything_output_failure": [],
        }
        for image_id in sorted(yolo_by_id):
            yolo = yolo_by_id[image_id]
            locate = locate_by_id[image_id]
            yolo_metrics = _image_metrics(yolo, ground_truth[image_id])
            locate_metrics = _image_metrics(locate, ground_truth[image_id])
            measurements[image_id] = {
                "yolo": yolo_metrics,
                "locate_anything": locate_metrics,
                "locate_anything_status": locate.status,
                "locate_anything_parser_diagnostics": locate.parser_diagnostics,
            }
            difference = yolo_metrics["f1"] - locate_metrics["f1"]
            if difference > 0:
                categories["yolo_advantage"].append((difference, image_id))
            elif difference < 0:
                categories["locate_anything_advantage"].append((-difference, image_id))
            categories["both_strong"].append(
                (min(yolo_metrics["f1"], locate_metrics["f1"]), image_id)
            )
            if (
                yolo_metrics["true_positives"] == 0
                and locate_metrics["true_positives"] == 0
            ):
                categories["both_miss"].append(
                    (yolo_metrics["false_negatives"], image_id)
                )
            duplicate_count = locate.parser_diagnostics.get("duplicate_boxes", 0)
            if duplicate_count:
                categories["locate_anything_duplicates"].append(
                    (duplicate_count, image_id)
                )
            if locate.status in ("malformed", "no_output", "error"):
                categories["locate_anything_output_failure"].append(
                    (locate_metrics["false_negatives"], image_id)
                )

        selected: set[str] = set()
        chosen_categories = {
            category: _take_unique(candidates, args.per_category, selected)
            for category, candidates in categories.items()
        }
        output["comparisons"][comparison_name] = {
            "modality": modality,
            "yolo_predictions": yolo_file,
            "locate_anything_predictions": locate_file,
            "categories": chosen_categories,
            "records": {
                image_id: measurements[image_id] for image_id in sorted(selected)
            },
        }
        subset_records[yolo_file] = [
            yolo_by_id[image_id] for image_id in sorted(selected)
        ]
        subset_records[locate_file] = [
            locate_by_id[image_id] for image_id in sorted(selected)
        ]
        for filename in (yolo_file, locate_file):
            ground_truth_by_file[filename] = ground_truth
            dataset_dir_by_file[filename] = (
                args.dataset_root / f"LLVIP-YOLO-{modality}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for filename, records in subset_records.items():
        path = args.output_dir / filename
        path.write_text(
            "".join(
                json.dumps(record.to_dict(), sort_keys=True) + "\n"
                for record in records
            )
        )
        _render_overlays(
            records,
            ground_truth_by_file[filename],
            dataset_dir_by_file[filename],
            args.output_dir / "overlays" / Path(filename).stem,
        )
    print(f"Wrote qualitative selection, subset JSONL, and overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
