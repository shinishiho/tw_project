"""Run the frozen three-model FLIR ADAS external-domain benchmark on Modal.

Before running, upload the three generated payload files to the data volume:

    uvx modal volume put flir-experiment-data \
      artifacts/FLIR-ADAS-v2-val-infrared.tar /
    uvx modal volume put flir-experiment-data \
      manifests/FLIR-ADAS-v2-payload-v1.json /

Examples:
    uvx modal run modal_apps/flir_inference.py --target pilot
    uvx modal run modal_apps/flir_inference.py --target yolo
    uvx modal run modal_apps/flir_inference.py --target yolo-ap
    uvx modal run modal_apps/flir_inference.py --target locate --shard-count 4
    uvx modal run modal_apps/flir_inference.py --target finalize-locate --shard-count 4
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

from evaluation.flir import (  # noqa: E402
    DATASET_ID,
    DATASET_RELEASE,
    DATASET_SPLIT,
    validate_resumable_flir_predictions,
)
from evaluation.io import append_prediction_jsonl  # noqa: E402
from evaluation.locate_anything import parse_locate_anything_boxes  # noqa: E402
from evaluation.schema import PredictionRecord  # noqa: E402
from evaluation.yolo import yolo_result_to_record  # noqa: E402


PAYLOAD_NAME = "FLIR-ADAS-v2-payload-v1.json"
ULTRALYTICS_VERSION = "8.4.102"
PRETRAINED_MODEL = "/models/yolo26n.pt"
PRETRAINED_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
FINETUNED_MODEL = (
    "/llvip-artifacts/training/yolo26n-thermal-e50-seed20260721/weights/best.pt"
)
FINETUNED_SHA256 = "66ba7bf3c07ea894e96767cc184d2f060d1baa0f8aaa3f6912a9600ddbdf0eed"
LOCATE_MODEL_ID = "nvidia/LocateAnything-3B"
LOCATE_MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
LOCATE_PROMPT = (
    "Locate all the instances that matches the following description: person."
)
BASE_SEED = 20260721

app = modal.App("flir-adas-external-domain")
dataset_volume = modal.Volume.from_name("flir-experiment-data", create_if_missing=True)
artifact_volume = modal.Volume.from_name(
    "flir-experiment-artifacts", create_if_missing=True
)
llvip_artifact_volume = modal.Volume.from_name("llvip-experiment-artifacts")
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


def _extract_and_validate_payload() -> tuple[dict, dict, Path]:
    import shutil
    import tarfile

    payload_path = Path("/data") / PAYLOAD_NAME
    if not payload_path.is_file():
        raise FileNotFoundError(f"payload manifest not found: {payload_path}")
    payload = json.loads(payload_path.read_text())
    expected_identity = (DATASET_ID, DATASET_RELEASE, DATASET_SPLIT)
    actual_identity = (
        payload.get("dataset_id"),
        payload.get("dataset_release"),
        payload.get("dataset_split"),
    )
    if actual_identity != expected_identity:
        raise ValueError(f"unexpected payload identity: {actual_identity}")
    archive_path = Path("/data") / payload["archive_name"]
    if _sha256(archive_path) != payload["archive_sha256"]:
        raise ValueError("FLIR validation archive hash mismatch")
    destination = Path("/tmp/flir-adas-v2")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError(f"unsafe validation archive member: {member.name}")
        archive.extractall(destination)  # noqa: S202 - every member checked above
    manifest_path = destination / payload["manifest_name"]
    pilot_path = destination / payload["pilot_manifest_name"]
    if _sha256(manifest_path) != payload["manifest_sha256"]:
        raise ValueError("FLIR manifest hash mismatch")
    if _sha256(pilot_path) != payload["pilot_manifest_sha256"]:
        raise ValueError("FLIR pilot manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    image_ids = [str(record["image_id"]) for record in manifest["records"]]
    if len(image_ids) != payload["image_count"] or len(image_ids) != len(
        set(image_ids)
    ):
        raise ValueError("FLIR payload image count or IDs are inconsistent")
    if set(pilot["image_ids"]) - set(image_ids) or pilot["size"] != 100:
        raise ValueError("FLIR pilot IDs are invalid")
    dataset_root = destination / payload["dataset_directory"]
    for image_id in image_ids:
        if not (dataset_root / "images" / "val" / f"{image_id}.jpg").is_file():
            raise FileNotFoundError(f"payload image missing: {image_id}")
        if not (dataset_root / "labels" / "val" / f"{image_id}.txt").is_file():
            raise FileNotFoundError(f"payload label missing: {image_id}")
    manifest["_file_sha256"] = payload["manifest_sha256"]
    return manifest, pilot, dataset_root


def _metadata_base(manifest: dict, app_source_sha256: str) -> dict:
    return {
        "dataset_id": DATASET_ID,
        "dataset_release": DATASET_RELEASE,
        "dataset_split": DATASET_SPLIT,
        "image_representation": "8-bit AGC thermal JPEG",
        "dataset_source_sha256": manifest["source"]["sha256"],
        "dataset_manifest_sha256": manifest["_file_sha256"],
        "app_source_sha256": app_source_sha256,
    }


def _selected_ids(manifest: dict, pilot: dict, sample: str) -> list[str]:
    if sample == "pilot":
        return sorted(map(str, pilot["image_ids"]))
    if sample == "full":
        return sorted(str(record["image_id"]) for record in manifest["records"])
    raise ValueError("sample must be pilot or full")


def _read_completed(
    path: Path,
    *,
    manifest: dict,
    run_id: str,
    model_id: str,
    model_revision: str,
    expected_ids: set[str],
) -> set[str]:
    if not path.exists():
        return set()
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(PredictionRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid prediction at {path}:{line_number}") from error
    return validate_resumable_flir_predictions(
        records,
        manifest,
        run_id=run_id,
        model_id=model_id,
        model_revision=model_revision,
        expected_ids=expected_ids,
    )


@app.cls(
    image=yolo_image,
    gpu="L40S",
    cpu=8,
    memory=16 * 1024,
    timeout=3 * 60 * 60,
    max_containers=1,
    scaledown_window=10 * 60,
    volumes={
        "/data": dataset_volume.with_mount_options(read_only=True),
        "/artifacts": artifact_volume,
        "/llvip-artifacts": llvip_artifact_volume.with_mount_options(read_only=True),
    },
)
class FlirYoloEvaluator:
    @modal.enter()
    def load(self) -> None:
        import torch
        import ultralytics
        from ultralytics import YOLO

        self.torch = torch
        self.ultralytics_version = ultralytics.__version__
        self.app_source_sha256 = _sha256(Path(__file__))
        self.manifest, self.pilot, self.dataset_root = _extract_and_validate_payload()
        if _sha256(Path(PRETRAINED_MODEL)) != PRETRAINED_SHA256:
            raise ValueError("pretrained YOLO checkpoint hash mismatch")
        if _sha256(Path(FINETUNED_MODEL)) != FINETUNED_SHA256:
            raise ValueError("fine-tuned YOLO checkpoint hash mismatch")
        self.models = {
            "pretrained": YOLO(PRETRAINED_MODEL),
            "finetuned": YOLO(FINETUNED_MODEL),
        }

    @modal.method()
    def run(self, state: str, sample: str = "full", confidence: float = 0.25) -> str:
        from PIL import Image

        if state not in self.models:
            raise ValueError(f"unsupported YOLO state: {state}")
        if confidence not in (0.25, 0.001):
            raise ValueError("YOLO confidence must be the locked 0.25 or AP 0.001")
        ids = _selected_ids(self.manifest, self.pilot, sample)
        purpose = "ap" if confidence == 0.001 else "primary"
        model_id = "yolo26n.pt" if state == "pretrained" else "yolo26n-thermal-best.pt"
        revision = PRETRAINED_SHA256 if state == "pretrained" else FINETUNED_SHA256
        run_id = f"flir-v2-{sample}-yolo-{state}-{purpose}"
        suffix = "_ap" if purpose == "ap" else ""
        output_path = (
            Path("/artifacts/predictions")
            / sample
            / f"yolo_{state}_infrared{suffix}.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed = _read_completed(
            output_path,
            manifest=self.manifest,
            run_id=run_id,
            model_id=model_id,
            model_revision=revision,
            expected_ids=set(ids),
        )
        pending = [image_id for image_id in ids if image_id not in completed]
        base_metadata = _metadata_base(self.manifest, self.app_source_sha256)
        errors = 0
        started = time.perf_counter()
        for index, image_id in enumerate(pending, start=1):
            image_path = self.dataset_root / "images" / "val" / f"{image_id}.jpg"
            self.torch.cuda.reset_peak_memory_stats()
            try:
                result = self.models[state].predict(
                    source=image_path,
                    imgsz=640,
                    conf=confidence,
                    classes=[0],
                    max_det=300,
                    device=0,
                    end2end=True,
                    verbose=False,
                )[0]
                record = yolo_result_to_record(
                    result,
                    run_id=run_id,
                    modality="infrared",
                    model_id=model_id,
                    model_revision=revision,
                    image_id=image_id,
                    metadata={
                        **base_metadata,
                        "model_state": state,
                        "purpose": purpose,
                        "head": "one-to-one",
                        "confidence": confidence,
                        "image_size": 640,
                        "batch_size": 1,
                        "max_detections": 300,
                        "gpu": "L40S",
                        "torch_version": str(self.torch.__version__),
                        "ultralytics_version": self.ultralytics_version,
                        "peak_gpu_memory_bytes": int(
                            self.torch.cuda.max_memory_allocated()
                        ),
                    },
                )
            except Exception as error:
                errors += 1
                with Image.open(image_path) as image:
                    width, height = image.size
                record = PredictionRecord(
                    run_id=run_id,
                    image_id=image_id,
                    modality="infrared",
                    model_id=model_id,
                    model_revision=revision,
                    image_width=width,
                    image_height=height,
                    status="error",
                    raw_output=f"{type(error).__name__}: {error}",
                    metadata={
                        **base_metadata,
                        "model_state": state,
                        "purpose": purpose,
                    },
                )
            append_prediction_jsonl(output_path, record)
            if index % 50 == 0:
                artifact_volume.commit()
                print(
                    f"{sample}/{state}/{purpose}: {len(completed) + index}/{len(ids)}"
                )
        summary = {
            "sample": sample,
            "state": state,
            "purpose": purpose,
            "records": len(completed) + len(pending),
            "new_records": len(pending),
            "errors": errors,
            "elapsed_seconds": time.perf_counter() - started,
            "output": str(output_path),
        }
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
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
class FlirLocateEvaluator:
    @modal.enter()
    def load(self) -> None:
        import torch
        import transformers

        from inference.locate_anything_worker import LocateAnythingWorker

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.app_source_sha256 = _sha256(Path(__file__))
        self.manifest, self.pilot, self.dataset_root = _extract_and_validate_payload()
        self.worker = LocateAnythingWorker(
            LOCATE_MODEL_ID, LOCATE_MODEL_REVISION, "cuda"
        )

    @modal.method()
    def run(
        self,
        sample: str = "full",
        base_seed: int = BASE_SEED,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> str:
        import numpy
        from PIL import Image

        if shard_count <= 0 or not 0 <= shard_index < shard_count:
            raise ValueError("shard index must be in [0, shard count)")
        ids = _selected_ids(self.manifest, self.pilot, sample)
        run_id = f"flir-v2-{sample}-locate-anything-seed{base_seed}"
        canonical = (
            Path("/artifacts/predictions") / sample / "locate_anything_infrared.jsonl"
        )
        output_path = (
            canonical
            if shard_count == 1
            else canonical.with_name(
                f"locate_anything_infrared.shard-{shard_index:02d}"
                f"-of-{shard_count:02d}.jsonl"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_completed = _read_completed(
            canonical,
            manifest=self.manifest,
            run_id=run_id,
            model_id=LOCATE_MODEL_ID,
            model_revision=LOCATE_MODEL_REVISION,
            expected_ids=set(ids),
        )
        shard_completed = (
            set()
            if output_path == canonical
            else _read_completed(
                output_path,
                manifest=self.manifest,
                run_id=run_id,
                model_id=LOCATE_MODEL_ID,
                model_revision=LOCATE_MODEL_REVISION,
                expected_ids=set(ids),
            )
        )
        if base_completed & shard_completed:
            raise ValueError("canonical and shard outputs contain duplicate IDs")
        assigned = {
            image_id
            for position, image_id in enumerate(ids)
            if position % shard_count == shard_index
        }
        if not shard_completed <= assigned:
            raise ValueError("existing LocateAnything shard contains foreign IDs")
        pending = sorted(assigned - base_completed - shard_completed)
        base_metadata = _metadata_base(self.manifest, self.app_source_sha256)
        errors = 0
        started = time.perf_counter()
        for index, image_id in enumerate(pending, start=1):
            image_path = self.dataset_root / "images" / "val" / f"{image_id}.jpg"
            seed_component = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16)
            image_seed = (base_seed + seed_component) % (2**31)
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
                    modality="infrared",
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
                        **base_metadata,
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
                    },
                )
            except Exception as error:
                errors += 1
                with Image.open(image_path) as image:
                    width, height = image.size
                record = PredictionRecord(
                    run_id=run_id,
                    image_id=image_id,
                    modality="infrared",
                    model_id=LOCATE_MODEL_ID,
                    model_revision=LOCATE_MODEL_REVISION,
                    image_width=width,
                    image_height=height,
                    status="error",
                    raw_output=f"{type(error).__name__}: {error}",
                    prompt=LOCATE_PROMPT,
                    metadata={
                        **base_metadata,
                        "base_seed": base_seed,
                        "image_seed": image_seed,
                    },
                )
            append_prediction_jsonl(output_path, record)
            if index % 25 == 0:
                artifact_volume.commit()
                print(
                    f"{sample}/locate shard {shard_index + 1}/{shard_count}: "
                    f"{len(shard_completed) + index}/{len(assigned)}"
                )
        summary = {
            "sample": sample,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "base_records": len(base_completed),
            "shard_records": len(shard_completed) + len(pending),
            "new_records": len(pending),
            "errors": errors,
            "elapsed_seconds": time.perf_counter() - started,
            "output": str(output_path),
        }
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
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
    sample: str = "full", base_seed: int = BASE_SEED, shard_count: int = 4
) -> str:
    import shutil

    if shard_count <= 1:
        raise ValueError("finalization requires at least two shards")
    manifest, pilot, _ = _extract_and_validate_payload()
    ids = _selected_ids(manifest, pilot, sample)
    expected_ids = set(ids)
    run_id = f"flir-v2-{sample}-locate-anything-seed{base_seed}"
    canonical = (
        Path("/artifacts/predictions") / sample / "locate_anything_infrared.jsonl"
    )
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical_ids = _read_completed(
        canonical,
        manifest=manifest,
        run_id=run_id,
        model_id=LOCATE_MODEL_ID,
        model_revision=LOCATE_MODEL_REVISION,
        expected_ids=expected_ids,
    )
    existing_summary = canonical.with_suffix(".summary.json")
    if canonical_ids == expected_ids and existing_summary.is_file():
        records = [
            PredictionRecord.from_dict(json.loads(line))
            for line in canonical.read_text().splitlines()
            if line.strip()
        ]
        if any(record.status == "error" for record in records):
            raise ValueError("canonical LocateAnything output contains runtime errors")
        return json.dumps(json.loads(existing_summary.read_text()), sort_keys=True)
    shard_paths = [
        canonical.with_name(
            f"locate_anything_infrared.shard-{index:02d}-of-{shard_count:02d}.jsonl"
        )
        for index in range(shard_count)
    ]
    paths = ([canonical] if canonical.is_file() else []) + shard_paths
    records_by_id: dict[str, PredictionRecord] = {}
    source_counts = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"prediction shard not found: {path}")
        count = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = PredictionRecord.from_dict(json.loads(line))
            validate_resumable_flir_predictions(
                [record],
                manifest,
                run_id=run_id,
                model_id=LOCATE_MODEL_ID,
                model_revision=LOCATE_MODEL_REVISION,
                expected_ids=expected_ids,
            )
            if record.image_id in records_by_id:
                raise ValueError(f"duplicate image ID across shards: {record.image_id}")
            records_by_id[record.image_id] = record
            count += 1
        source_counts[str(path)] = count
    if set(records_by_id) != expected_ids:
        missing = sorted(expected_ids - set(records_by_id))[:5]
        extra = sorted(set(records_by_id) - expected_ids)[:5]
        raise ValueError(f"merged FLIR ID mismatch; missing={missing}, extra={extra}")
    if any(record.status == "error" for record in records_by_id.values()):
        raise ValueError("cannot finalize LocateAnything output with runtime errors")
    if canonical.exists():
        backup = canonical.with_name("locate_anything_infrared.pre-shard-partial.jsonl")
        if not backup.exists():
            shutil.copy2(canonical, backup)
    temporary = canonical.with_suffix(".merged.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for image_id in sorted(records_by_id):
            output.write(
                json.dumps(records_by_id[image_id].to_dict(), sort_keys=True) + "\n"
            )
    temporary.replace(canonical)
    summary = {
        "sample": sample,
        "records": len(records_by_id),
        "status_counts": dict(
            sorted(Counter(record.status for record in records_by_id.values()).items())
        ),
        "source_counts_before_merge": source_counts,
        "finalized_at": datetime.now(UTC).isoformat(),
        "output": str(canonical),
    }
    canonical.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    artifact_volume.commit()
    return json.dumps(summary, sort_keys=True)


@app.local_entrypoint()
def main(target: str = "pilot", shard_index: int = 0, shard_count: int = 1) -> None:
    if target not in {"pilot", "yolo", "yolo-ap", "locate", "finalize-locate"}:
        raise ValueError(
            "target must be pilot, yolo, yolo-ap, locate, or finalize-locate"
        )
    summaries = []
    if target == "pilot":
        yolo = FlirYoloEvaluator()
        for state in ("pretrained", "finetuned"):
            summaries.append(json.loads(yolo.run.remote(state, "pilot", 0.25)))
        locate = FlirLocateEvaluator()
        summaries.append(json.loads(locate.run.remote("pilot", BASE_SEED, 0, 1)))
    elif target in {"yolo", "yolo-ap"}:
        confidence = 0.25 if target == "yolo" else 0.001
        evaluator = FlirYoloEvaluator()
        for state in ("pretrained", "finetuned"):
            summaries.append(
                json.loads(evaluator.run.remote(state, "full", confidence))
            )
    elif target == "locate":
        summaries.append(
            json.loads(
                FlirLocateEvaluator().run.remote(
                    "full", BASE_SEED, shard_index, shard_count
                )
            )
        )
    else:
        summaries.append(
            json.loads(
                finalize_locate_predictions.remote("full", BASE_SEED, shard_count)
            )
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
