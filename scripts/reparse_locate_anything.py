"""Reparse preserved LocateAnything raw outputs with the current parser."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import read_prediction_jsonl
from evaluation.locate_anything import parse_locate_anything_boxes
from evaluation.schema import PredictionRecord


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.paths:
        records = read_prediction_jsonl(path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}-",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            for record in records:
                if record.raw_output is None:
                    raise ValueError(f"{path} contains a record without raw output")
                parsed = parse_locate_anything_boxes(
                    record.raw_output, record.image_width, record.image_height
                )
                value = record.to_dict()
                value["boxes"] = [asdict(box) for box in parsed.boxes]
                value["status"] = parsed.status
                value["parser_diagnostics"] = parsed.diagnostics
                updated = PredictionRecord.from_dict(value)
                destination.write(json.dumps(updated.to_dict(), sort_keys=True) + "\n")
                destination.flush()
        temporary_path.replace(path)
        print(f"Reparsed {len(records)} records in {path}")


if __name__ == "__main__":
    main()
