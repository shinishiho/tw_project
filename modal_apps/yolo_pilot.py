"""Run both YOLO states on the locked 100-pair Modal pilot sample."""

from __future__ import annotations

import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_CHECKPOINT = (
    PROJECT_ROOT.parent / "new_code/runs/detect/llvip_yolo26n/weights/best.pt"
)
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.schema import PredictionRecord  # noqa: E402
from evaluation.yolo import yolo_result_to_record  # noqa: E402


app = modal.App("llvip-yolo-pilot")

yolo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.102")
    .add_local_file(OLD_CHECKPOINT, "/models/old_test_leaked_best.pt")
    .add_local_python_source("evaluation")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_inference(
    model: object,
    torch: object,
    image_bytes: bytes,
    image_id: str,
    modality: str,
    run_id: str,
    model_id: str,
    model_revision: str,
    ultralytics_version: str,
    sample_manifest_sha256: str,
    pilot_only_test_leaked_checkpoint: bool,
) -> str:
    from PIL import Image

    torch.cuda.reset_peak_memory_stats()
    with Image.open(BytesIO(image_bytes)).convert("RGB") as image:
        result = model.predict(
            source=image,
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
            "head": "one-to-one",
            "confidence": 0.25,
            "nms_iou": None,
            "image_size": 640,
            "batch_size": 1,
            "max_detections": 300,
            "gpu": "L40S",
            "torch_version": str(torch.__version__),
            "ultralytics_version": ultralytics_version,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "sample_manifest_sha256": sample_manifest_sha256,
            "pilot_only_test_leaked_checkpoint": pilot_only_test_leaked_checkpoint,
        },
    )
    return json.dumps(record.to_dict(), sort_keys=True)


@app.cls(image=yolo_image, gpu="L40S", timeout=30 * 60, scaledown_window=5 * 60)
class PretrainedYolo:
    @modal.enter()
    def load(self) -> None:
        import torch
        import ultralytics
        from ultralytics import YOLO

        self.torch = torch
        self.model = YOLO("yolo26n.pt")
        checkpoint = Path(self.model.ckpt_path)
        self.revision = f"sha256:{_sha256(checkpoint)}"
        self.version = ultralytics.__version__

    @modal.method()
    def infer(
        self,
        image_bytes: bytes,
        image_id: str,
        modality: str,
        run_id: str,
        sample_manifest_sha256: str,
    ) -> str:
        return _run_inference(
            self.model,
            self.torch,
            image_bytes,
            image_id,
            modality,
            run_id,
            "yolo26n.pt",
            self.revision,
            self.version,
            sample_manifest_sha256,
            False,
        )


@app.cls(image=yolo_image, gpu="L40S", timeout=30 * 60, scaledown_window=5 * 60)
class OldFineTunedYolo:
    @modal.enter()
    def load(self) -> None:
        import torch
        import ultralytics
        from ultralytics import YOLO

        self.torch = torch
        checkpoint = Path("/models/old_test_leaked_best.pt")
        self.model = YOLO(checkpoint)
        self.revision = f"sha256:{_sha256(checkpoint)}"
        self.version = ultralytics.__version__

    @modal.method()
    def infer(
        self,
        image_bytes: bytes,
        image_id: str,
        modality: str,
        run_id: str,
        sample_manifest_sha256: str,
    ) -> str:
        return _run_inference(
            self.model,
            self.torch,
            image_bytes,
            image_id,
            modality,
            run_id,
            "old-test-leaked-best.pt",
            self.revision,
            self.version,
            sample_manifest_sha256,
            True,
        )


def _read_completed(path: Path, run_id: str, modality: str, model_id: str) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = PredictionRecord.from_dict(json.loads(line))
        if (record.run_id, record.modality, record.model_id) != (
            run_id,
            modality,
            model_id,
        ):
            raise ValueError(f"unexpected record at {path}:{line_number}")
        if record.image_id in completed:
            raise ValueError(f"duplicate image ID at {path}:{line_number}")
        completed.add(record.image_id)
    return completed


@app.local_entrypoint()
def main(
    sample_manifest: str = "manifests/LLVIP-pilot-100-v1.json",
    dataset_root: str = "datasets",
    output_dir: str = "artifacts/modal-pilot",
) -> None:
    manifest_path = Path(sample_manifest)
    manifest = json.loads(manifest_path.read_text())
    image_ids = [record["image_id"] for record in manifest["records"]]
    if len(image_ids) != 100 or len(set(image_ids)) != 100:
        raise ValueError("pilot manifest must contain 100 unique image IDs")
    sample_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    dataset_path = Path(dataset_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    model_states = (
        ("pretrained", "yolo26n.pt", PretrainedYolo()),
        ("old_finetuned", "old-test-leaked-best.pt", OldFineTunedYolo()),
    )
    for state, model_id, remote_model in model_states:
        for modality in ("visible", "infrared"):
            run_id = f"phase3-{state}-yolo-{modality}-100"
            output_path = destination / f"yolo_{state}_{modality}_100.jsonl"
            completed = _read_completed(output_path, run_id, modality, model_id)
            pending = [image_id for image_id in image_ids if image_id not in completed]
            print(
                f"{state}/{modality}: {len(completed)} complete, {len(pending)} pending"
            )
            for image_id in pending:
                image_path = (
                    dataset_path
                    / f"LLVIP-YOLO-{modality}"
                    / "images"
                    / "test"
                    / f"{image_id}.jpg"
                )
                record = json.loads(
                    remote_model.infer.remote(
                        image_path.read_bytes(),
                        image_id,
                        modality,
                        run_id,
                        sample_hash,
                    )
                )
                with output_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
            print(f"completed {state}/{modality}: {len(image_ids)} records")
