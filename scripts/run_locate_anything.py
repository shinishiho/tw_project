"""Run resumable LocateAnything inference into the unified JSONL schema."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import append_prediction_jsonl, read_prediction_jsonl
from evaluation.locate_anything import parse_locate_anything_boxes
from evaluation.schema import PredictionRecord
from inference.locate_anything_worker import LocateAnythingWorker


MODEL_ID = "nvidia/LocateAnything-3B"
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
PROMPT = "Locate all the instances that matches the following description: person."


def set_image_seed(base_seed: int, image_id: str) -> int:
    import numpy
    import torch

    image_seed = (base_seed + int(image_id)) % (2**31)
    random.seed(image_seed)
    numpy.random.seed(image_seed)
    torch.manual_seed(image_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(image_seed)
    return image_seed


def json_safe(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=("visible", "infrared"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--generation-mode", choices=("fast", "hybrid", "slow"), default="hybrid"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from PIL import Image
    import torch
    import transformers

    image_paths = sorted((args.dataset_dir / "images" / args.split).glob("*.jpg"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError("no input images found")
    expected_ids = {path.stem for path in image_paths}
    expected_key = (args.run_id, args.model_id, args.revision, args.modality)

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
    print(
        f"LocateAnything run {args.run_id}: "
        f"{len(completed)} complete, {len(pending)} pending"
    )
    if not pending:
        return

    worker = LocateAnythingWorker(args.model_id, args.revision, args.device)
    common_metadata = {
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "base_seed": args.seed,
        "device": args.device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    for image_path in pending:
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            image_seed = set_image_seed(args.seed, image_path.stem)
            started = time.perf_counter()
            try:
                response = worker.predict(
                    image,
                    PROMPT,
                    generation_mode=args.generation_mode,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                answer = str(response["answer"])
                parsed = parse_locate_anything_boxes(answer, width, height)
                metadata = {
                    **common_metadata,
                    "image_seed": image_seed,
                    "worker_stats": json_safe(response.get("stats")),
                }
                record = PredictionRecord(
                    run_id=args.run_id,
                    image_id=image_path.stem,
                    modality=args.modality,
                    model_id=args.model_id,
                    model_revision=args.revision,
                    image_width=width,
                    image_height=height,
                    boxes=parsed.boxes,
                    status=parsed.status,
                    raw_output=answer,
                    prompt=PROMPT,
                    latency_ms=latency_ms,
                    parser_diagnostics=parsed.diagnostics,
                    metadata=metadata,
                )
            except Exception as error:
                latency_ms = (time.perf_counter() - started) * 1000
                record = PredictionRecord(
                    run_id=args.run_id,
                    image_id=image_path.stem,
                    modality=args.modality,
                    model_id=args.model_id,
                    model_revision=args.revision,
                    image_width=width,
                    image_height=height,
                    status="error",
                    prompt=PROMPT,
                    latency_ms=latency_ms,
                    metadata={
                        **common_metadata,
                        "image_seed": image_seed,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            append_prediction_jsonl(args.output, record)
            completed.add(record.image_id)

    if completed != expected_ids:
        raise ValueError(
            f"incomplete output; missing={sorted(expected_ids - completed)[:5]}"
        )
    print(f"Completed {len(completed)} predictions in {args.output}")


if __name__ == "__main__":
    main()
