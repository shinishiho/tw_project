"""Run the paired LocateAnything Phase 2 smoke test on one warm Modal GPU."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from io import BytesIO
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.locate_anything import parse_locate_anything_boxes  # noqa: E402
from evaluation.schema import PredictionRecord  # noqa: E402


MODEL_ID = "nvidia/LocateAnything-3B"
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
PROMPT = "Locate all the instances that matches the following description: person."
BASE_SEED = 20260721

app = modal.App("llvip-locate-anything-smoke")
model_cache = modal.Volume.from_name("llvip-huggingface-cache", create_if_missing=True)

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


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


@app.cls(
    image=locate_image,
    gpu="L40S",
    timeout=60 * 60,
    scaledown_window=5 * 60,
    volumes={"/cache": model_cache},
)
class LocateAnythingModel:
    @modal.enter()
    def load(self) -> None:
        import torch
        import transformers

        from inference.locate_anything_worker import LocateAnythingWorker

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.worker = LocateAnythingWorker(MODEL_ID, MODEL_REVISION, "cuda")
        model_cache.commit()

    @modal.method()
    def infer(
        self,
        image_bytes: bytes,
        image_id: str,
        modality: str,
        run_id: str,
        sample_manifest_sha256: str | None,
        base_seed: int,
    ) -> str:
        import numpy
        from PIL import Image

        image_seed = (base_seed + int(image_id)) % (2**31)
        random.seed(image_seed)
        numpy.random.seed(image_seed)
        self.torch.manual_seed(image_seed)
        self.torch.cuda.manual_seed_all(image_seed)
        self.torch.cuda.reset_peak_memory_stats()

        with Image.open(BytesIO(image_bytes)).convert("RGB") as image:
            width, height = image.size
            started = time.perf_counter()
            response = self.worker.predict(
                image,
                PROMPT,
                generation_mode="hybrid",
                max_new_tokens=8192,
                temperature=0.7,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            answer = str(response["answer"])
            parsed = parse_locate_anything_boxes(answer, width, height)

        record = PredictionRecord(
            run_id=run_id,
            image_id=image_id,
            modality=modality,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            image_width=width,
            image_height=height,
            boxes=parsed.boxes,
            status=parsed.status,
            raw_output=answer,
            prompt=PROMPT,
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
                "gpu": "L40S",
                "torch_version": str(self.torch.__version__),
                "transformers_version": self.transformers_version,
                "peak_gpu_memory_bytes": int(self.torch.cuda.max_memory_allocated()),
                "worker_stats": _json_safe(response.get("stats")),
                "sample_manifest_sha256": sample_manifest_sha256,
            },
        )
        return json.dumps(record.to_dict(), sort_keys=True)


def _read_completed(path: Path, run_id: str, modality: str) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = PredictionRecord.from_dict(json.loads(line))
        expected = (run_id, modality, MODEL_ID, MODEL_REVISION)
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


@app.local_entrypoint()
def main(
    dataset_root: str = "datasets",
    output_dir: str = "artifacts/modal-smoke",
    limit: int = 20,
    sample_manifest: str = "",
    base_seed: int = BASE_SEED,
) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")
    dataset_path = Path(dataset_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    ids_by_modality = {
        modality: {
            path.stem
            for path in (
                dataset_path / f"LLVIP-YOLO-{modality}" / "images" / "test"
            ).glob("*.jpg")
        }
        for modality in ("visible", "infrared")
    }
    if ids_by_modality["visible"] != ids_by_modality["infrared"]:
        raise ValueError("visible and infrared test IDs differ")
    sample_manifest_sha256 = None
    if sample_manifest:
        manifest_path = Path(sample_manifest)
        manifest = json.loads(manifest_path.read_text())
        image_ids = [record["image_id"] for record in manifest["records"]]
        sample_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if len(image_ids) < limit:
            raise ValueError(
                f"manifest contains only {len(image_ids)} IDs but --limit is {limit}"
            )
        image_ids = image_ids[:limit]
    else:
        image_ids = sorted(ids_by_modality["visible"])[:limit]
    if len(image_ids) != limit:
        raise ValueError(f"requested {limit} pairs but found {len(image_ids)}")
    if not set(image_ids) <= ids_by_modality["visible"]:
        raise ValueError("sample manifest contains IDs outside the locked test split")

    model = LocateAnythingModel()
    phase = "phase3" if sample_manifest else "phase2"
    seed_suffix = "" if base_seed == BASE_SEED else f"-seed-{base_seed}"
    for modality in ("visible", "infrared"):
        run_id = f"{phase}-locate-anything-{modality}-{limit}{seed_suffix}"
        output_path = destination / f"locate_anything_{modality}_{limit}.jsonl"
        completed = _read_completed(output_path, run_id, modality)
        pending = [image_id for image_id in image_ids if image_id not in completed]
        print(f"{modality}: {len(completed)} complete, {len(pending)} pending")
        for image_id in pending:
            image_path = (
                dataset_path
                / f"LLVIP-YOLO-{modality}"
                / "images"
                / "test"
                / f"{image_id}.jpg"
            )
            record = json.loads(
                model.infer.remote(
                    image_path.read_bytes(),
                    image_id,
                    modality,
                    run_id,
                    sample_manifest_sha256,
                    base_seed,
                )
            )
            with output_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
            print(
                f"{modality}/{image_id}: {record['status']}, "
                f"{len(record['boxes'])} boxes, {record['latency_ms']:.0f} ms"
            )
