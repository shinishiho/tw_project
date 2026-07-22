"""Run the six locked full-test model/modality combinations on Modal.

The paired test set is uploaded once as a verified tar archive and extracted
inside each warm container. Prediction JSONL files live on the artifact Volume
and are validated before resuming, so interrupted jobs do not duplicate work.

Examples:
    uvx modal run modal_apps/full_inference.py --target yolo
    uvx modal run modal_apps/full_inference.py --target locate
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.locate_anything import parse_locate_anything_boxes  # noqa: E402
from evaluation.io import append_prediction_jsonl  # noqa: E402
from evaluation.schema import PredictionRecord  # noqa: E402
from evaluation.yolo import yolo_result_to_record  # noqa: E402


TEST_ARCHIVE_NAME = "LLVIP-test-paired.tar"
TEST_ARCHIVE_SHA256 = "8b4db30cc40279cf04105cdf1859d6961a55182afe072617d409ccc77ec1ba6b"
SPLIT_MANIFEST_SHA256 = (
    "05facc1b82630ec515cfdb0df16617f1c6390fc5af009b4c090a8343e78b33ef"
)
EXPECTED_TEST_IMAGES = 3463
ULTRALYTICS_VERSION = "8.4.102"
PRETRAINED_MODEL = "/models/yolo26n.pt"
PRETRAINED_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
FINETUNED_MODEL = "/artifacts/training/yolo26n-thermal-e50-seed20260721/weights/best.pt"
FINETUNED_SHA256 = "66ba7bf3c07ea894e96767cc184d2f060d1baa0f8aaa3f6912a9600ddbdf0eed"
LOCATE_MODEL_ID = "nvidia/LocateAnything-3B"
LOCATE_MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
LOCATE_PROMPT = (
    "Locate all the instances that matches the following description: person."
)
BASE_SEED = 20260721

app = modal.App("llvip-full-inference")
dataset_volume = modal.Volume.from_name("llvip-experiment-data")
artifact_volume = modal.Volume.from_name("llvip-experiment-artifacts")
model_cache = modal.Volume.from_name("llvip-huggingface-cache")

yolo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(f"ultralytics=={ULTRALYTICS_VERSION}")
    .run_commands(
        "mkdir -p /models",
        "cd /models && python -c \"from ultralytics import YOLO; YOLO('yolo26n.pt')\"",
    )
    .add_local_python_source("evaluation")
)

locate_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.57.1",
        "tokenizers==0.22.0",
        "accelerate==1.5.2",
        "peft==0.12.0",
        "timm>=1.0.11",
        "numpy==1.25.0",
        "pillow==11.1.0",
        "opencv-python-headless==4.11.0.86",
        "decord==0.6.0",
        "lmdb==1.7.5",
    )
    .env({"HF_HOME": "/cache/huggingface"})
    .add_local_python_source("evaluation", "inference")
)

utility_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "evaluation"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _extract_and_validate_test_archive() -> tuple[Path, list[str]]:
    import shutil
    import tarfile

    archive_path = Path("/data") / TEST_ARCHIVE_NAME
    archive_hash = _sha256(archive_path)
    if archive_hash != TEST_ARCHIVE_SHA256:
        raise ValueError(
            f"test archive mismatch: expected {TEST_ARCHIVE_SHA256}, got {archive_hash}"
        )
    destination = Path("/tmp/llvip-full-test")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    with tarfile.open(archive_path, "r") as archive:
        members = [
            member
            for member in archive.getmembers()
            if not any(part.startswith("._") for part in Path(member.name).parts)
        ]
        for member in members:
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError(f"unsafe test archive path: {member.name}")
        archive.extractall(  # noqa: S202 - verified immutable archive
            destination, members=members
        )

    manifest_path = destination / "LLVIP-splits-v1.json"
    if _sha256(manifest_path) != SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest hash mismatch inside test archive")
    manifest = json.loads(manifest_path.read_text())
    image_ids = sorted(
        record["image_id"]
        for record in manifest["records"]
        if record["split"] == "test"
    )
    if len(image_ids) != EXPECTED_TEST_IMAGES or len(set(image_ids)) != len(image_ids):
        raise ValueError("locked test manifest does not contain 3,463 unique IDs")
    for modality in ("visible", "infrared"):
        image_directory = destination / f"LLVIP-YOLO-{modality}" / "images" / "test"
        actual_ids = {path.stem for path in image_directory.glob("*.jpg")}
        if actual_ids != set(image_ids):
            raise ValueError(f"{modality} test images do not match locked manifest")
    return destination, image_ids


def _read_completed(
    path: Path,
    *,
    run_id: str,
    modality: str,
    model_id: str,
    model_revision: str,
) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = PredictionRecord.from_dict(json.loads(line))
        expected = (run_id, modality, model_id, model_revision)
        actual = (
            record.run_id,
            record.modality,
            record.model_id,
            record.model_revision,
        )
        if actual != expected:
            raise ValueError(f"unexpected record at {path}:{line_number}: {actual}")
        if record.image_id in completed:
            raise ValueError(f"duplicate image ID at {path}:{line_number}")
        completed.add(record.image_id)
    return completed


@app.cls(
    image=yolo_image,
    gpu="L40S",
    cpu=8,
    memory=16 * 1024,
    timeout=2 * 60 * 60,
    max_containers=1,
    scaledown_window=10 * 60,
    volumes={
        "/data": dataset_volume.with_mount_options(read_only=True),
        "/artifacts": artifact_volume,
    },
)
class FullYoloEvaluator:
    @modal.enter()
    def load(self) -> None:
        import torch
        import ultralytics
        from ultralytics import YOLO

        self.torch = torch
        self.ultralytics_version = ultralytics.__version__
        self.app_source_sha256 = _sha256(Path(__file__))
        self.dataset_root, self.image_ids = _extract_and_validate_test_archive()
        if _sha256(Path(PRETRAINED_MODEL)) != PRETRAINED_SHA256:
            raise ValueError("pretrained YOLO checkpoint hash mismatch")
        if _sha256(Path(FINETUNED_MODEL)) != FINETUNED_SHA256:
            raise ValueError("fine-tuned YOLO checkpoint hash mismatch")
        self.models = {
            "pretrained": YOLO(PRETRAINED_MODEL),
            "finetuned": YOLO(FINETUNED_MODEL),
        }

    @modal.method()
    def run(self, state: str, modality: str) -> str:
        from PIL import Image

        if state not in self.models:
            raise ValueError(f"unsupported YOLO state: {state}")
        if modality not in ("visible", "infrared"):
            raise ValueError(f"unsupported modality: {modality}")
        model_id = "yolo26n.pt" if state == "pretrained" else "yolo26n-thermal-best.pt"
        model_revision = (
            PRETRAINED_SHA256 if state == "pretrained" else FINETUNED_SHA256
        )
        run_id = f"phase5-yolo-{state}-{modality}-full"
        output_path = (
            Path("/artifacts/predictions/full") / f"yolo_{state}_{modality}.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed = _read_completed(
            output_path,
            run_id=run_id,
            modality=modality,
            model_id=model_id,
            model_revision=model_revision,
        )
        if not completed <= set(self.image_ids):
            raise ValueError(
                "existing YOLO output contains IDs outside locked test set"
            )
        pending = [image_id for image_id in self.image_ids if image_id not in completed]
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        errors = 0
        model = self.models[state]
        for index, image_id in enumerate(pending, start=1):
            image_path = (
                self.dataset_root
                / f"LLVIP-YOLO-{modality}"
                / "images"
                / "test"
                / f"{image_id}.jpg"
            )
            self.torch.cuda.reset_peak_memory_stats()
            try:
                result = model.predict(
                    source=image_path,
                    imgsz=640,
                    conf=0.25,
                    classes=[0],
                    max_det=300,
                    device=0,
                    end2end=True,
                    verbose=False,
                )[0]
                record = yolo_result_to_record(
                    result,
                    run_id=run_id,
                    modality=modality,
                    model_id=model_id,
                    model_revision=model_revision,
                    image_id=image_id,
                    metadata={
                        "model_state": state,
                        "head": "one-to-one",
                        "confidence": 0.25,
                        "nms_iou": None,
                        "image_size": 640,
                        "batch_size": 1,
                        "max_detections": 300,
                        "gpu": "L40S",
                        "torch_version": str(self.torch.__version__),
                        "ultralytics_version": self.ultralytics_version,
                        "peak_gpu_memory_bytes": int(
                            self.torch.cuda.max_memory_allocated()
                        ),
                        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
                        "test_archive_sha256": TEST_ARCHIVE_SHA256,
                    },
                )
            except Exception as error:  # keep the full run auditable and resumable
                errors += 1
                with Image.open(image_path) as image:
                    width, height = image.size
                record = PredictionRecord(
                    run_id=run_id,
                    image_id=image_id,
                    modality=modality,
                    model_id=model_id,
                    model_revision=model_revision,
                    image_width=width,
                    image_height=height,
                    status="error",
                    raw_output=f"{type(error).__name__}: {error}",
                    metadata={"model_state": state},
                )
            append_prediction_jsonl(output_path, record)
            if index % 50 == 0:
                artifact_volume.commit()
                print(
                    f"{state}/{modality}: {len(completed) + index}/{len(self.image_ids)}"
                )
        summary = {
            "state": state,
            "modality": modality,
            "records": len(completed) + len(pending),
            "new_records": len(pending),
            "errors": errors,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "output": str(output_path),
            "app_source_sha256": self.app_source_sha256,
        }
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        artifact_volume.commit()
        return json.dumps(summary, sort_keys=True)

    @modal.method()
    def validate_ap(self, state: str, modality: str) -> str:
        """Run YOLO's confidence sweep for secondary AP metrics."""
        import yaml

        if state not in self.models:
            raise ValueError(f"unsupported YOLO state: {state}")
        if modality not in ("visible", "infrared"):
            raise ValueError(f"unsupported modality: {modality}")
        dataset_directory = self.dataset_root / f"LLVIP-YOLO-{modality}"
        labels_directory = dataset_directory / "labels"
        if not labels_directory.exists():
            labels_directory.symlink_to(
                self.dataset_root / "LLVIP-YOLO-infrared" / "labels",
                target_is_directory=True,
            )
        data_path = Path(f"/tmp/llvip-full-{modality}.yaml")
        data_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(dataset_directory),
                    "train": "images/test",
                    "val": "images/test",
                    "test": "images/test",
                    "names": {0: "person"},
                },
                sort_keys=False,
            )
        )
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        metrics = self.models[state].val(
            data=data_path,
            split="test",
            imgsz=640,
            batch=64,
            conf=0.001,
            iou=0.7,
            max_det=300,
            device=0,
            end2end=True,
            plots=False,
            save_json=False,
            project="/tmp/yolo-ap",
            name=f"{state}-{modality}",
            exist_ok=True,
        )
        model_revision = (
            PRETRAINED_SHA256 if state == "pretrained" else FINETUNED_SHA256
        )
        summary = {
            "run_id": f"phase6-yolo-ap-{state}-{modality}-full",
            "state": state,
            "modality": modality,
            "model_revision": model_revision,
            "records": len(self.image_ids),
            "map50": float(metrics.box.map50),
            "map75": float(metrics.box.map75),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
            "settings": {
                "image_size": 640,
                "batch_size": 64,
                "confidence": 0.001,
                "nms_iou": 0.7,
                "max_detections": 300,
                "head": "one-to-one",
            },
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "app_source_sha256": self.app_source_sha256,
            "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
            "test_archive_sha256": TEST_ARCHIVE_SHA256,
        }
        output_path = (
            Path("/artifacts/evaluation/full") / f"yolo_ap_{state}_{modality}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        artifact_volume.commit()
        return json.dumps(summary, sort_keys=True)


@app.cls(
    image=locate_image,
    gpu="L40S",
    cpu=4,
    memory=16 * 1024,
    timeout=12 * 60 * 60,
    max_containers=1,
    scaledown_window=10 * 60,
    volumes={
        "/data": dataset_volume.with_mount_options(read_only=True),
        "/artifacts": artifact_volume,
        "/cache": model_cache,
    },
)
class FullLocateAnythingEvaluator:
    @modal.enter()
    def load(self) -> None:
        import torch
        import transformers

        from inference.locate_anything_worker import LocateAnythingWorker

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.app_source_sha256 = _sha256(Path(__file__))
        self.dataset_root, self.image_ids = _extract_and_validate_test_archive()
        self.worker = LocateAnythingWorker(
            LOCATE_MODEL_ID, LOCATE_MODEL_REVISION, "cuda"
        )

    @modal.method()
    def run(
        self,
        modality: str,
        base_seed: int = BASE_SEED,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> str:
        import numpy
        from PIL import Image

        if modality not in ("visible", "infrared"):
            raise ValueError(f"unsupported modality: {modality}")
        if shard_count <= 0 or not 0 <= shard_index < shard_count:
            raise ValueError("shard index must be in [0, shard count)")
        run_id = f"phase5-locate-anything-{modality}-full-seed{base_seed}"
        canonical_path = (
            Path("/artifacts/predictions/full") / f"locate_anything_{modality}.jsonl"
        )
        output_path = (
            canonical_path
            if shard_count == 1
            else canonical_path.with_name(
                f"locate_anything_{modality}.shard-{shard_index:02d}"
                f"-of-{shard_count:02d}.jsonl"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_completed = _read_completed(
            canonical_path,
            run_id=run_id,
            modality=modality,
            model_id=LOCATE_MODEL_ID,
            model_revision=LOCATE_MODEL_REVISION,
        )
        shard_completed = (
            set()
            if output_path == canonical_path
            else _read_completed(
                output_path,
                run_id=run_id,
                modality=modality,
                model_id=LOCATE_MODEL_ID,
                model_revision=LOCATE_MODEL_REVISION,
            )
        )
        if base_completed & shard_completed:
            raise ValueError("canonical and shard outputs contain duplicate IDs")
        completed = base_completed | shard_completed
        if not completed <= set(self.image_ids):
            raise ValueError("existing grounding output contains IDs outside test set")
        pending = [
            image_id
            for position, image_id in enumerate(self.image_ids)
            if position % shard_count == shard_index and image_id not in completed
        ]
        expected_shard_ids = {
            image_id
            for position, image_id in enumerate(self.image_ids)
            if position % shard_count == shard_index
        }
        if not shard_completed <= expected_shard_ids:
            raise ValueError("existing grounding shard contains IDs assigned elsewhere")
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        errors = 0
        for index, image_id in enumerate(pending, start=1):
            image_path = (
                self.dataset_root
                / f"LLVIP-YOLO-{modality}"
                / "images"
                / "test"
                / f"{image_id}.jpg"
            )
            image_seed = (base_seed + int(image_id)) % (2**31)
            random.seed(image_seed)
            numpy.random.seed(image_seed)
            self.torch.manual_seed(image_seed)
            self.torch.cuda.manual_seed_all(image_seed)
            self.torch.cuda.reset_peak_memory_stats()
            try:
                with Image.open(image_path).convert("RGB") as image:
                    width, height = image.size
                    inference_started = time.perf_counter()
                    response = self.worker.predict(
                        image,
                        LOCATE_PROMPT,
                        generation_mode="hybrid",
                        max_new_tokens=8192,
                        temperature=0.7,
                    )
                    latency_ms = (time.perf_counter() - inference_started) * 1000
                answer = str(response["answer"])
                parsed = parse_locate_anything_boxes(answer, width, height)
                record = PredictionRecord(
                    run_id=run_id,
                    image_id=image_id,
                    modality=modality,
                    model_id=LOCATE_MODEL_ID,
                    model_revision=LOCATE_MODEL_REVISION,
                    image_width=width,
                    image_height=height,
                    boxes=parsed.boxes,
                    status=parsed.status,
                    raw_output=answer,
                    prompt=LOCATE_PROMPT,
                    latency_ms=latency_ms,
                    parser_diagnostics=parsed.diagnostics,
                    metadata={
                        "generation_mode": "hybrid",
                        "max_new_tokens": 8192,
                        "temperature": 0.7,
                        "do_sample": True,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "base_seed": base_seed,
                        "image_seed": image_seed,
                        "batch_size": 1,
                        "gpu": "L40S",
                        "torch_version": str(self.torch.__version__),
                        "transformers_version": self.transformers_version,
                        "peak_gpu_memory_bytes": int(
                            self.torch.cuda.max_memory_allocated()
                        ),
                        "worker_stats": _json_safe(response.get("stats")),
                        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
                        "test_archive_sha256": TEST_ARCHIVE_SHA256,
                        "app_source_sha256": self.app_source_sha256,
                    },
                )
            except Exception as error:  # keep failures explicit in the JSONL
                errors += 1
                with Image.open(image_path) as image:
                    width, height = image.size
                record = PredictionRecord(
                    run_id=run_id,
                    image_id=image_id,
                    modality=modality,
                    model_id=LOCATE_MODEL_ID,
                    model_revision=LOCATE_MODEL_REVISION,
                    image_width=width,
                    image_height=height,
                    status="error",
                    raw_output=f"{type(error).__name__}: {error}",
                    prompt=LOCATE_PROMPT,
                    metadata={"base_seed": base_seed, "image_seed": image_seed},
                )
            append_prediction_jsonl(output_path, record)
            if index % 25 == 0:
                artifact_volume.commit()
                print(
                    f"locate/{modality} shard {shard_index + 1}/{shard_count}: "
                    f"{len(shard_completed) + index}/{len(expected_shard_ids)}"
                )
        summary = {
            "modality": modality,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "base_records": len(base_completed),
            "shard_records": len(shard_completed) + len(pending),
            "new_records": len(pending),
            "errors": errors,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "output": str(output_path),
            "app_source_sha256": self.app_source_sha256,
        }
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        artifact_volume.commit()
        return json.dumps(summary, sort_keys=True)


@app.function(
    image=utility_image,
    cpu=1,
    memory=1024,
    timeout=10 * 60,
    volumes={
        "/data": dataset_volume.with_mount_options(read_only=True),
        "/artifacts": artifact_volume,
    },
)
def finalize_locate_predictions(
    modality: str, base_seed: int = BASE_SEED, shard_count: int = 4
) -> str:
    """Validate disjoint shards and atomically create one canonical JSONL."""
    import shutil

    if modality not in ("visible", "infrared"):
        raise ValueError(f"unsupported modality: {modality}")
    if shard_count <= 1:
        raise ValueError("finalization requires at least two shards")
    manifest_path = Path("/data/LLVIP-splits-v1.json")
    if _sha256(manifest_path) != SPLIT_MANIFEST_SHA256:
        raise ValueError("remote split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_ids = {
        record["image_id"]
        for record in manifest["records"]
        if record["split"] == "test"
    }
    if len(expected_ids) != EXPECTED_TEST_IMAGES:
        raise ValueError("remote split manifest does not contain 3,463 test IDs")

    run_id = f"phase5-locate-anything-{modality}-full-seed{base_seed}"
    canonical_path = (
        Path("/artifacts/predictions/full") / f"locate_anything_{modality}.jsonl"
    )
    canonical_ids = _read_completed(
        canonical_path,
        run_id=run_id,
        modality=modality,
        model_id=LOCATE_MODEL_ID,
        model_revision=LOCATE_MODEL_REVISION,
    )
    existing_summary_path = canonical_path.with_suffix(".summary.json")
    if canonical_ids == expected_ids and existing_summary_path.is_file():
        return json.dumps(json.loads(existing_summary_path.read_text()), sort_keys=True)
    paths = [canonical_path] + [
        canonical_path.with_name(
            f"locate_anything_{modality}.shard-{index:02d}-of-{shard_count:02d}.jsonl"
        )
        for index in range(shard_count)
    ]
    records_by_id: dict[str, PredictionRecord] = {}
    source_counts = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"prediction shard not found: {path}")
        count = 0
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            record = PredictionRecord.from_dict(json.loads(line))
            identity = (
                record.run_id,
                record.modality,
                record.model_id,
                record.model_revision,
            )
            expected_identity = (
                run_id,
                modality,
                LOCATE_MODEL_ID,
                LOCATE_MODEL_REVISION,
            )
            if identity != expected_identity:
                raise ValueError(f"unexpected record at {path}:{line_number}")
            if record.image_id in records_by_id:
                raise ValueError(f"duplicate image ID across shards: {record.image_id}")
            records_by_id[record.image_id] = record
            count += 1
        source_counts[str(path)] = count
    if set(records_by_id) != expected_ids:
        missing = sorted(expected_ids - set(records_by_id))[:5]
        extra = sorted(set(records_by_id) - expected_ids)[:5]
        raise ValueError(f"merged test ID mismatch; missing={missing}, extra={extra}")

    backup_path = canonical_path.with_name(
        f"locate_anything_{modality}.pre-shard-partial.jsonl"
    )
    if not backup_path.exists():
        shutil.copy2(canonical_path, backup_path)
    temporary_path = canonical_path.with_suffix(".merged.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for image_id in sorted(records_by_id):
            output.write(
                json.dumps(records_by_id[image_id].to_dict(), sort_keys=True) + "\n"
            )
    temporary_path.replace(canonical_path)
    statuses = Counter(record.status for record in records_by_id.values())
    source_hashes = sorted(
        {
            str(record.metadata.get("app_source_sha256", "unrecorded"))
            for record in records_by_id.values()
        }
    )
    summary = {
        "modality": modality,
        "records": len(records_by_id),
        "unique_image_ids": len(records_by_id),
        "status_counts": dict(sorted(statuses.items())),
        "source_counts_before_merge": source_counts,
        "inference_app_source_sha256_values": source_hashes,
        "finalizer_app_source_sha256": _sha256(Path(__file__)),
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "test_archive_sha256": TEST_ARCHIVE_SHA256,
        "finalized_at": datetime.now(UTC).isoformat(),
        "output": str(canonical_path),
        "partial_backup": str(backup_path),
    }
    summary_path = canonical_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    return json.dumps(summary, sort_keys=True)


@app.local_entrypoint()
def main(target: str = "yolo", shard_index: int = 0, shard_count: int = 1) -> None:
    if target not in (
        "yolo",
        "yolo-ap",
        "locate",
        "locate-visible",
        "locate-infrared",
        "finalize-visible",
        "finalize-infrared",
        "all",
    ):
        raise ValueError(
            "target must be yolo, yolo-ap, locate, locate-visible, "
            "locate-infrared, finalize-visible, finalize-infrared, or all"
        )
    summaries = []
    if target in ("yolo", "all"):
        evaluator = FullYoloEvaluator()
        for state in ("pretrained", "finetuned"):
            for modality in ("visible", "infrared"):
                summaries.append(json.loads(evaluator.run.remote(state, modality)))
    if target == "yolo-ap":
        evaluator = FullYoloEvaluator()
        for state in ("pretrained", "finetuned"):
            for modality in ("visible", "infrared"):
                summaries.append(
                    json.loads(evaluator.validate_ap.remote(state, modality))
                )
    if target in ("locate", "locate-visible", "locate-infrared", "all"):
        evaluator = FullLocateAnythingEvaluator()
        modalities = {
            "locate-visible": ("visible",),
            "locate-infrared": ("infrared",),
        }.get(target, ("visible", "infrared"))
        for modality in modalities:
            summaries.append(
                json.loads(
                    evaluator.run.remote(modality, BASE_SEED, shard_index, shard_count)
                )
            )
    if target in ("finalize-visible", "finalize-infrared"):
        modality = target.removeprefix("finalize-")
        summaries.append(
            json.loads(
                finalize_locate_predictions.remote(modality, BASE_SEED, shard_count)
            )
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
