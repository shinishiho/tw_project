from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from evaluation.ap import coco_style_ap
from evaluation.bootstrap import (
    bootstrap_intervals,
    build_image_statistics,
    independent_paired_difference_of_differences,
    paired_bootstrap_differences,
)
from evaluation.flir import (
    load_flir_manifest,
    validate_flir_prediction_identity,
    validate_resumable_flir_predictions,
)
from evaluation.matching import match_boxes_with_ignored_regions
from evaluation.schema import Box, PredictionRecord
from scripts.prepare_flir import (
    DatasetSource,
    discover_coco_annotation,
    prepare,
    resolve_image_member,
)


def prediction(
    image_id: str,
    boxes: tuple[Box, ...],
    *,
    manifest_hash: str = "manifest",
) -> PredictionRecord:
    return PredictionRecord(
        run_id="flir-test",
        image_id=image_id,
        modality="infrared",
        model_id="model",
        model_revision="revision",
        image_width=20,
        image_height=16,
        boxes=boxes,
        metadata={
            "dataset_id": "FLIR_ADAS_v2",
            "dataset_release": "expanded-2022",
            "dataset_split": "official-validation",
            "dataset_manifest_sha256": manifest_hash,
        },
    )


class CrowdMatchingTests(unittest.TestCase):
    def test_unmatched_detection_inside_crowd_is_ignored(self) -> None:
        result = match_boxes_with_ignored_regions(
            [Box(1, 1, 5, 5), Box(10, 10, 14, 14)],
            [],
            [Box(0, 0, 8, 8)],
            0.50,
        )
        self.assertEqual(result.ignored_prediction_indices, (0,))
        self.assertEqual(result.false_positive_indices, (1,))

    def test_crowd_detections_are_removed_from_primary_statistics(self) -> None:
        records = [prediction("1", (Box(1, 1, 5, 5),))]
        statistics = build_image_statistics(
            records,
            {"1": ()},
            0.50,
            ignored_by_image={"1": (Box(0, 0, 8, 8),)},
        )
        self.assertEqual(statistics.false_positives.tolist(), [0])
        self.assertEqual(statistics.predicted_boxes.tolist(), [0])


class GroupedBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = {
            "1": (Box(0, 0, 5, 5),),
            "2": (Box(0, 0, 5, 5),),
            "3": (Box(0, 0, 5, 5),),
            "4": (Box(0, 0, 5, 5),),
        }
        self.left = build_image_statistics(
            [
                prediction("1", (Box(0, 0, 5, 5),)),
                prediction("2", (Box(0, 0, 5, 5),)),
                prediction("3", ()),
                prediction("4", ()),
            ],
            self.truth,
            0.50,
        )
        self.right = build_image_statistics(
            [prediction(image_id, ()) for image_id in self.truth],
            self.truth,
            0.50,
        )
        self.groups = {"1": "a", "2": "a", "3": "b", "4": "b"}

    def test_grouped_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_intervals(
            self.left, replicates=50, seed=7, groups_by_image=self.groups
        )
        second = bootstrap_intervals(
            self.left, replicates=50, seed=7, groups_by_image=self.groups
        )
        self.assertEqual(first, second)
        paired = paired_bootstrap_differences(
            self.left,
            self.right,
            replicates=50,
            seed=7,
            groups_by_image=self.groups,
        )
        self.assertGreater(paired["f1"]["estimate"], 0)

    def test_independent_difference_of_differences(self) -> None:
        result = independent_paired_difference_of_differences(
            self.left,
            self.right,
            self.right,
            self.right,
            first_groups=self.groups,
            second_groups=self.groups,
            replicates=50,
            seed=9,
        )
        self.assertGreater(result["estimate"], 0)


class AveragePrecisionTests(unittest.TestCase):
    def test_ranked_ap_and_crowd_ignore(self) -> None:
        records = [
            prediction(
                "1",
                (
                    Box(0, 0, 5, 5, confidence=0.9),
                    Box(10, 10, 14, 14, confidence=0.8),
                ),
            )
        ]
        result = coco_style_ap(
            records,
            {"1": (Box(0, 0, 5, 5),)},
            {"1": (Box(9, 9, 15, 15),)},
        )
        self.assertEqual(result["ap50"], 1.0)
        self.assertEqual(result["map50_95"], 1.0)


class FlirPreparationTests(unittest.TestCase):
    def test_random_rgb_letters_in_filename_do_not_change_modality(self) -> None:
        member = resolve_image_member(
            "data/video-frame-rGb5bC9.jpg",
            ("images_thermal_val/data/video-frame-rGb5bC9.jpg",),
            images_prefix=None,
        )
        self.assertEqual(member, "images_thermal_val/data/video-frame-rGb5bC9.jpg")

    def _fixture(self, root: Path) -> Path:
        source = root / "source"
        (source / "annotations").mkdir(parents=True)
        (source / "thermal" / "val").mkdir(parents=True)
        for name, value in (("a.jpg", 32), ("b.jpg", 128), ("c.jpg", 224)):
            Image.new("L", (20, 16), value).save(source / "thermal" / "val" / name)
        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": "a.jpg",
                    "width": 20,
                    "height": 16,
                    "sequence_id": "s1",
                    "day_night": "night",
                },
                {
                    "id": 2,
                    "file_name": "b.jpg",
                    "width": 20,
                    "height": 16,
                    "sequence_id": "s1",
                    "day_night": "night",
                },
                {
                    "id": 3,
                    "file_name": "c.jpg",
                    "width": 20,
                    "height": 16,
                    "sequence_id": "s2",
                    "day_night": "day",
                },
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "bike"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [-1, 1, 6, 6],
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 2, 8, 8],
                    "iscrowd": 1,
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 2,
                    "bbox": [1, 1, 4, 4],
                    "iscrowd": 0,
                },
                {
                    "id": 4,
                    "image_id": 3,
                    "category_id": 1,
                    "bbox": [2, 2, 0, 4],
                    "iscrowd": 0,
                },
            ],
        }
        (source / "annotations" / "thermal_val_coco.json").write_text(json.dumps(coco))
        train = {**coco, "images": [], "annotations": []}
        (source / "annotations" / "thermal_train_coco.json").write_text(
            json.dumps(train)
        )
        return source

    def test_preparation_is_idempotent_and_preserves_policy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._fixture(root)
            arguments = {
                "source_path": source,
                "output_root": root / "dataset",
                "manifest_path": root / "manifest.json",
                "pilot_path": root / "pilot.json",
                "payload_path": root / "payload.json",
                "tar_path": root / "payload.tar",
                "annotations": None,
                "images_prefix": None,
                "force": False,
            }
            first = prepare(**arguments)
            first_tar = Path(first["tar"]).read_bytes()
            second = prepare(**arguments)
            self.assertEqual(first, second)
            self.assertEqual(first_tar, Path(second["tar"]).read_bytes())
            manifest, truth, ignored = load_flir_manifest(root / "manifest.json")
            self.assertEqual(manifest["image_count"], 3)
            self.assertEqual(manifest["person_negative_image_count"], 2)
            self.assertEqual(truth["1"], (Box(0, 1, 5, 7),))
            self.assertEqual(ignored["1"], (Box(10, 2, 18, 10),))
            self.assertEqual(
                (root / "dataset" / "labels" / "val" / "2.txt").read_text(), ""
            )
            self.assertEqual(
                manifest["annotation_audit"]["changed_or_dropped_count"], 2
            )
            self.assertEqual(json.loads((root / "pilot.json").read_text())["size"], 3)

            records = [
                prediction(image_id, (), manifest_hash=manifest["_file_sha256"])
                for image_id in truth
            ]
            validate_flir_prediction_identity(records, manifest)

    def test_resumable_predictions_require_locked_identity(self) -> None:
        manifest = {
            "_file_sha256": "manifest",
            "source": {"sha256": "source"},
        }
        record = prediction("1", ())
        record.metadata["dataset_source_sha256"] = "source"
        completed = validate_resumable_flir_predictions(
            [record],
            manifest,
            run_id="flir-test",
            model_id="model",
            model_revision="revision",
            expected_ids={"1", "2"},
        )
        self.assertEqual(completed, {"1"})

        wrong_manifest = prediction("1", ())
        wrong_manifest.metadata["dataset_source_sha256"] = "source"
        wrong_manifest.metadata["dataset_manifest_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "dataset_manifest_sha256"):
            validate_resumable_flir_predictions(
                [wrong_manifest],
                manifest,
                run_id="flir-test",
                model_id="model",
                model_revision="revision",
                expected_ids={"1"},
            )

    def test_ambiguous_annotations_fail(self) -> None:
        with TemporaryDirectory() as temporary:
            source_path = self._fixture(Path(temporary))
            original = source_path / "annotations" / "thermal_val_coco.json"
            (source_path / "annotations" / "thermal_val_coco-copy.json").write_bytes(
                original.read_bytes()
            )
            source = DatasetSource(source_path)
            try:
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    discover_coco_annotation(source)
            finally:
                source.close()

    def test_ambiguous_thermal_image_roots_fail(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = self._fixture(root)
            duplicate = source_path / "thermal" / "validation-copy" / "a.jpg"
            duplicate.parent.mkdir(parents=True)
            duplicate.write_bytes(
                (source_path / "thermal" / "val" / "a.jpg").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "ambiguous image"):
                prepare(
                    source_path=source_path,
                    output_root=root / "dataset",
                    manifest_path=root / "manifest.json",
                    pilot_path=root / "pilot.json",
                    payload_path=root / "payload.json",
                    tar_path=root / "payload.tar",
                    annotations=None,
                    images_prefix=None,
                    force=False,
                )

    def test_train_validation_overlap_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = self._fixture(root)
            train_path = source_path / "annotations" / "thermal_train_coco.json"
            train = json.loads(train_path.read_text())
            train["images"] = [
                {"id": 99, "file_name": "a.jpg", "width": 20, "height": 16}
            ]
            train_path.write_text(json.dumps(train))
            with self.assertRaisesRegex(ValueError, "train/validation image overlap"):
                prepare(
                    source_path=source_path,
                    output_root=root / "dataset",
                    manifest_path=root / "manifest.json",
                    pilot_path=root / "pilot.json",
                    payload_path=root / "payload.json",
                    tar_path=root / "payload.tar",
                    annotations=None,
                    images_prefix=None,
                    force=False,
                )

    def test_unsafe_zip_member_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.jpg", b"bad")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                DatasetSource(archive)


if __name__ == "__main__":
    unittest.main()
