"""Train the clean thermal YOLO26n checkpoint on Modal.

The dataset Volume is mounted read-only. A single-file train/validation archive
is extracted to ephemeral storage so uploading and startup avoid thousands of
small remote file operations while Ultralytics retains a writable local tree.

Examples:
    uvx modal run modal_apps/yolo_train.py --epochs 1 --run-id smoke-e1
    uvx modal run modal_apps/yolo_train.py --epochs 50 --run-id main-e50
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal


ULTRALYTICS_VERSION = "8.4.102"
PRETRAINED_MODEL = "/models/yolo26n.pt"
DATASET_VOLUME_NAME = "llvip-experiment-data"
ARTIFACT_VOLUME_NAME = "llvip-experiment-artifacts"
DATASET_DIRECTORY = "LLVIP-YOLO-infrared"
DATASET_ARCHIVE_NAME = "LLVIP-YOLO-infrared-trainval.tar"
DATASET_ARCHIVE_SHA256 = (
    "2916455d4e9afa6c0c5d74db3785a6a2f8adc9b304ea797fc3699095fe6a3c44"
)
MANIFEST_NAME = "LLVIP-splits-v1.json"
MANIFEST_SHA256 = "05facc1b82630ec515cfdb0df16617f1c6390fc5af009b4c090a8343e78b33ef"
DEFAULT_SEED = 20260721

app = modal.App("llvip-yolo-clean-training")
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)

yolo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(f"ultralytics=={ULTRALYTICS_VERSION}")
    .run_commands(
        "mkdir -p /models",
        "cd /models && python -c \"from ultralytics import YOLO; YOLO('yolo26n.pt')\"",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _json_value(value: object) -> object:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _prepare_ephemeral_dataset() -> Path:
    """Extract the verified train/validation archive to writable local storage."""
    import shutil
    import tarfile

    destination = Path("/tmp") / DATASET_DIRECTORY
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive_path = Path("/data") / DATASET_ARCHIVE_NAME
    archive_hash = _sha256(archive_path)
    if archive_hash != DATASET_ARCHIVE_SHA256:
        raise ValueError(
            f"dataset archive mismatch: expected {DATASET_ARCHIVE_SHA256}, "
            f"got {archive_hash}"
        )
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError(f"unsafe path in dataset archive: {member.name}")
        archive.extractall(destination)  # noqa: S202 - every path was checked above
    training_yaml = "\n".join(
        line
        for line in (destination / "data.yaml").read_text().splitlines()
        if not line.startswith("test:")
    )
    (destination / "data.yaml").write_text(training_yaml + "\n")
    return destination / "data.yaml"


@app.function(
    image=yolo_image,
    gpu="A10",
    cpu=8,
    memory=24 * 1024,
    timeout=24 * 60 * 60,
    max_containers=1,
    volumes={
        "/data": dataset_volume.with_mount_options(read_only=True),
        "/artifacts": artifact_volume,
    },
)
def train(
    run_id: str,
    epochs: int,
    seed: int = DEFAULT_SEED,
    batch: int = 64,
    resume: bool = False,
    source_state: dict[str, object] | None = None,
) -> str:
    import importlib.metadata
    import os
    import platform
    import shutil

    import torch
    import ultralytics
    from ultralytics import YOLO

    allowed_run_id_characters = "-_.abcdefghijklmnopqrstuvwxyz0123456789"
    if not run_id or any(
        character not in allowed_run_id_characters for character in run_id
    ):
        raise ValueError(
            "run_id must use lowercase letters, digits, dots, dashes, or underscores"
        )
    if epochs <= 0 or batch <= 0:
        raise ValueError("epochs and batch must be positive")

    manifest_path = Path("/data") / MANIFEST_NAME
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != MANIFEST_SHA256:
        raise ValueError(
            f"split manifest mismatch: expected {MANIFEST_SHA256}, got {manifest_hash}"
        )

    data_yaml = _prepare_ephemeral_dataset()
    run_dir = Path("/artifacts/training") / run_id
    last_checkpoint = run_dir / "weights" / "last.pt"
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"cannot resume without {last_checkpoint}")
    elif run_dir.exists():
        raise FileExistsError(f"immutable run already exists: {run_dir}")
    else:
        run_dir.mkdir(parents=True)

    shutil.copy2(manifest_path, run_dir / MANIFEST_NAME)
    shutil.copy2(data_yaml, run_dir / "data.yaml")
    started_at = _timestamp()
    pretrained_hash = _sha256(Path(PRETRAINED_MODEL))
    requested = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at,
        "resume": resume,
        "dataset": DATASET_DIRECTORY,
        "dataset_archive": DATASET_ARCHIVE_NAME,
        "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
        "dataset_yaml_sha256": _sha256(data_yaml),
        "split_manifest_sha256": manifest_hash,
        "initial_model": "yolo26n.pt",
        "initial_checkpoint_sha256": pretrained_hash,
        "selection_data": "validation split only",
        "selection_fitness": "Ultralytics default detection fitness",
        "test_split_used_during_training": False,
        "modal_gpu_request": "A10",
        "epochs": epochs,
        "batch": batch,
        "image_size": 640,
        "image_cache": "disk",
        "detection_head": "Ultralytics default end-to-end head",
        "seed": seed,
        "source_state": source_state or {},
    }
    (run_dir / "requested_run.json").write_text(
        json.dumps(requested, indent=2, sort_keys=True) + "\n"
    )
    artifact_volume.commit()

    os.environ["WANDB_DISABLED"] = "true"
    started = time.perf_counter()
    if resume:
        model = YOLO(last_checkpoint)
        results = model.train(resume=True, device=0, workers=8)
    else:
        model = YOLO(PRETRAINED_MODEL)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=batch,
            optimizer="MuSGD",
            lr0=0.0054,
            lrf=0.0495,
            momentum=0.947,
            weight_decay=0.00064,
            warmup_epochs=0.98,
            project=run_dir.parent,
            name=run_id,
            exist_ok=True,
            device=0,
            workers=8,
            cache="disk",
            patience=20,
            seed=seed,
            deterministic=True,
            close_mosaic=15,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.2,
            degrees=0.0,
            translate=0.1,
            scale=0.4,
            shear=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.2,
            mixup=0.0,
            erasing=0.2,
            plots=True,
            verbose=True,
        )
    elapsed_seconds = time.perf_counter() - started

    package_versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    summary = {
        **requested,
        "finished_at": _timestamp(),
        "elapsed_seconds": elapsed_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "ultralytics_version": ultralytics.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": dict(sorted(package_versions.items())),
        "metrics": {
            key: _json_value(value)
            for key, value in (getattr(results, "results_dict", {}) or {}).items()
        },
        "best_checkpoint_sha256": _sha256(run_dir / "weights" / "best.pt"),
        "last_checkpoint_sha256": _sha256(last_checkpoint),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    artifact_volume.commit()
    return json.dumps(summary, sort_keys=True)


def _local_source_state() -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short", "--untracked-files=normal"),
        "training_app_sha256": _sha256(Path(__file__)),
    }


@app.local_entrypoint()
def main(
    run_id: str,
    epochs: int = 1,
    seed: int = DEFAULT_SEED,
    batch: int = 64,
    resume: bool = False,
) -> None:
    summary = train.remote(
        run_id=run_id,
        epochs=epochs,
        seed=seed,
        batch=batch,
        resume=resume,
        source_state=_local_source_state(),
    )
    parsed = json.loads(summary)
    print(json.dumps(parsed, indent=2, sort_keys=True))
