"""Run resumable YOLO inference and write unified prediction JSONL."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import append_prediction_jsonl, read_prediction_jsonl
from evaluation.yolo import yolo_result_to_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model", required=True, help="checkpoint path or Ultralytics model ID"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=("visible", "infrared"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--head", choices=("one-to-one", "one-to-many"), default="one-to-one"
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="deterministic first-N smoke test")
    args = parser.parse_args()

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed; run "
            "`uv pip install -r requirements-yolo.txt`"
        ) from error

    if not 0 <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1:
        raise ValueError("confidence and NMS IoU must be in [0, 1]")
    image_paths = sorted((args.dataset_dir / "images" / args.split).glob("*.jpg"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError("no input images found")
    expected_ids = {path.stem for path in image_paths}

    model = YOLO(args.model)
    checkpoint_path = Path(args.model)
    if not checkpoint_path.is_file():
        checkpoint_path = Path(model.ckpt_path)
    model_revision = f"sha256:{sha256(checkpoint_path)}"
    expected_key = (args.run_id, args.model, model_revision, args.modality)

    completed = set()
    if args.output.exists():
        prior = read_prediction_jsonl(args.output)
        if len({record.image_id for record in prior}) != len(prior):
            raise ValueError("existing output contains duplicate image IDs")
        for record in prior:
            key = (
                record.run_id,
                record.model_id,
                record.model_revision,
                record.modality,
            )
            if key != expected_key:
                raise ValueError(f"existing output belongs to another run: {key}")
            completed.add(record.image_id)
    if completed - expected_ids:
        raise ValueError(
            "existing output contains images outside this invocation: "
            f"{sorted(completed - expected_ids)[:5]}"
        )
    pending = [path for path in image_paths if path.stem not in completed]
    print(f"YOLO run {args.run_id}: {len(completed)} complete, {len(pending)} pending")
    if not pending:
        return

    end_to_end = args.head == "one-to-one"
    settings = {
        "head": args.head,
        "confidence": args.confidence,
        "nms_iou": None if end_to_end else args.nms_iou,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "max_detections": args.max_detections,
        "device": args.device,
        "ultralytics_version": ultralytics.__version__,
    }
    predict_options = {
        "source": [str(path) for path in pending],
        "stream": True,
        "batch": args.batch_size,
        "imgsz": args.image_size,
        "conf": args.confidence,
        "max_det": args.max_detections,
        "classes": [0],
        "device": args.device,
        "end2end": end_to_end,
        "verbose": False,
    }
    if not end_to_end:
        predict_options["iou"] = args.nms_iou
    result_count = 0
    for result_count, (result, image_path) in enumerate(
        zip(model.predict(**predict_options), pending, strict=True), start=1
    ):
        record = yolo_result_to_record(
            result,
            run_id=args.run_id,
            modality=args.modality,
            model_id=args.model,
            model_revision=model_revision,
            metadata=settings,
            image_id=image_path.stem,
        )
        append_prediction_jsonl(args.output, record)
        completed.add(record.image_id)
    if result_count != len(pending):
        raise ValueError(
            f"Ultralytics yielded {result_count} results for {len(pending)} inputs"
        )

    if completed != expected_ids:
        missing = sorted(expected_ids - completed)[:5]
        extra = sorted(completed - expected_ids)[:5]
        raise ValueError(f"incomplete YOLO output; missing={missing}, extra={extra}")
    print(f"Completed {len(completed)} predictions in {args.output}")


if __name__ == "__main__":
    main()
