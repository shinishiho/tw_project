from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.bootstrap import (
    aggregate_metrics,
    bootstrap_intervals,
    build_image_statistics,
    paired_bootstrap_differences,
    subset_image_statistics,
)
from evaluation.io import (
    append_prediction_jsonl,
    load_yolo_ground_truth,
    read_prediction_jsonl,
)
from evaluation.locate_anything import parse_locate_anything_boxes
from evaluation.matching import match_boxes
from evaluation.metrics import evaluate_records
from evaluation.schema import Box, PredictionRecord
from evaluation.yolo import yolo_result_to_record


class LocateAnythingParserTests(unittest.TestCase):
    def test_parses_normalized_boxes_and_retains_duplicates(self) -> None:
        answer = (
            "person <box><100><200><500><800></box> "
            "person <box><100><200><500><800></box>"
        )
        parsed = parse_locate_anything_boxes(answer, 1280, 1024)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(len(parsed.boxes), 2)
        self.assertEqual(parsed.boxes[0], Box(128, 204.8, 640, 819.2))
        self.assertEqual(parsed.diagnostics["duplicate_boxes"], 1)

    def test_counts_malformed_out_of_range_and_degenerate_boxes(self) -> None:
        answer = (
            "<box><10><20><30></box> "
            "<box><0><0><1001><50></box> "
            "<box><20><20><20><30></box> <box>"
        )
        parsed = parse_locate_anything_boxes(answer, 100, 100)
        self.assertEqual(parsed.status, "malformed")
        self.assertEqual(parsed.boxes, ())
        self.assertEqual(parsed.diagnostics["malformed_segments"], 1)
        self.assertEqual(parsed.diagnostics["out_of_range_boxes"], 1)
        self.assertEqual(parsed.diagnostics["degenerate_boxes"], 1)
        self.assertEqual(parsed.diagnostics["unclosed_box_tags"], 1)

    def test_plain_text_without_boxes_is_no_output(self) -> None:
        parsed = parse_locate_anything_boxes("No people found.", 100, 100)
        self.assertEqual(parsed.status, "no_output")

    def test_explicit_no_object_token_is_not_malformed(self) -> None:
        parsed = parse_locate_anything_boxes(
            "<ref>person</ref><box>None</box><|im_end|>", 100, 100
        )
        self.assertEqual(parsed.status, "no_output")
        self.assertEqual(parsed.diagnostics["no_object_segments"], 1)
        self.assertNotIn("malformed_segments", parsed.diagnostics)


class MatchingTests(unittest.TestCase):
    def test_maximum_cardinality_repairs_greedy_choice(self) -> None:
        truth = [Box(0, 0, 10, 10), Box(8, 0, 18, 10)]
        predictions = [Box(2, 0, 16, 10), Box(0, 0, 10, 10)]
        result = match_boxes(predictions, truth, iou_threshold=0.5)
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(
            {
                (item.prediction_index, item.ground_truth_index)
                for item in result.matches
            },
            {(0, 1), (1, 0)},
        )

    def test_class_labels_do_not_cross_match(self) -> None:
        result = match_boxes(
            [Box(0, 0, 10, 10, label="car")],
            [Box(0, 0, 10, 10, label="person")],
            iou_threshold=0.5,
        )
        self.assertEqual(result.false_positive_indices, (0,))
        self.assertEqual(result.false_negative_indices, (0,))


class MetricTests(unittest.TestCase):
    def test_aggregates_traceable_confidence_independent_metrics(self) -> None:
        record = PredictionRecord(
            run_id="run-1",
            image_id="010001",
            modality="infrared",
            model_id="nvidia/LocateAnything-3B",
            model_revision="abc123",
            image_width=100,
            image_height=100,
            boxes=(Box(0, 0, 10, 10), Box(0, 0, 10, 10), Box(50, 50, 60, 60)),
            status="malformed",
        )
        summary = evaluate_records(
            [record], {"010001": (Box(0, 0, 10, 10),)}, iou_threshold=0.5
        )
        self.assertEqual(summary.true_positives, 1)
        self.assertEqual(summary.false_positives, 2)
        self.assertEqual(summary.false_negatives, 0)
        self.assertAlmostEqual(summary.precision, 1 / 3)
        self.assertEqual(summary.recall, 1.0)
        self.assertAlmostEqual(summary.duplicate_box_rate, 1 / 3)
        self.assertEqual(summary.malformed_output_rate, 1.0)
        self.assertEqual(summary.error_output_rate, 0.0)
        self.assertEqual(len(summary.assignments["010001"].matches), 1)

    def test_paired_bootstrap_is_deterministic_and_preserves_pairing(self) -> None:
        predictions = [
            PredictionRecord(
                run_id="run-1",
                image_id="010001",
                modality="infrared",
                model_id="model",
                model_revision="revision",
                image_width=100,
                image_height=100,
                boxes=(Box(0, 0, 10, 10),),
            ),
            PredictionRecord(
                run_id="run-1",
                image_id="010002",
                modality="infrared",
                model_id="model",
                model_revision="revision",
                image_width=100,
                image_height=100,
            ),
        ]
        truth = {
            "010001": (Box(0, 0, 10, 10),),
            "010002": (Box(0, 0, 10, 10),),
        }
        statistics = build_image_statistics(predictions, truth, 0.5)
        self.assertEqual(aggregate_metrics(statistics)["recall"], 0.5)
        subset = subset_image_statistics(statistics, {"010001"})
        self.assertEqual(aggregate_metrics(subset)["recall"], 1.0)
        first = bootstrap_intervals(statistics, replicates=100, seed=7)
        second = bootstrap_intervals(statistics, replicates=100, seed=7)
        self.assertEqual(first, second)
        differences = paired_bootstrap_differences(
            statistics, statistics, replicates=100, seed=7
        )
        self.assertEqual(differences["f1"], {"estimate": 0.0, "low": 0.0, "high": 0.0})


class PredictionIoTests(unittest.TestCase):
    def test_jsonl_round_trip_and_yolo_ground_truth_conversion(self) -> None:
        record = PredictionRecord(
            run_id="run-1",
            image_id="010001",
            modality="visible",
            model_id="yolo26n.pt",
            model_revision="sha256:abc",
            image_width=200,
            image_height=100,
            boxes=(Box(50, 25, 150, 75, confidence=0.9),),
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions_path = root / "predictions.jsonl"
            label_dir = root / "dataset" / "labels" / "test"
            label_dir.mkdir(parents=True)
            (label_dir / "010001.txt").write_text(
                "0 0.50000000 0.50000000 0.50000000 0.50000000\n"
            )
            append_prediction_jsonl(predictions_path, record)
            loaded = read_prediction_jsonl(predictions_path)
            self.assertEqual(loaded, [record])
            truth = load_yolo_ground_truth(loaded, root / "dataset")
            self.assertEqual(truth["010001"], (Box(50, 25, 150, 75),))

    def test_yolo_boundary_roundoff_is_clamped(self) -> None:
        record = PredictionRecord(
            run_id="run-1",
            image_id="010002",
            modality="infrared",
            model_id="model",
            model_revision="revision",
            image_width=1280,
            image_height=1024,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            label_dir = root / "labels" / "test"
            label_dir.mkdir(parents=True)
            (label_dir / "010002.txt").write_text(
                "0 0.97460938 0.61523438 0.05078125 0.25585938\n"
            )
            truth = load_yolo_ground_truth([record], root)
            self.assertEqual(truth["010002"][0].x2, 1280)


class YoloAdapterTests(unittest.TestCase):
    def test_converts_an_ultralytics_style_result(self) -> None:
        class Values:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class Boxes:
            xyxy = Values([[10, 20, 30, 40]])
            conf = Values([0.75])
            cls = Values([0])

        class Result:
            orig_shape = (100, 200)
            path = "/dataset/010001.jpg"
            boxes = Boxes()
            names = {0: "person"}
            speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}

        record = yolo_result_to_record(
            Result(),
            run_id="run",
            modality="visible",
            model_id="yolo26n.pt",
            model_revision="sha256:abc",
            metadata={"head": "one-to-one"},
            image_id="official-id",
        )
        self.assertEqual(record.boxes, (Box(10, 20, 30, 40, confidence=0.75),))
        self.assertEqual(record.latency_ms, 3.5)
        self.assertEqual(record.image_id, "official-id")


if __name__ == "__main__":
    unittest.main()
