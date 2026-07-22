# Thermal pedestrian detection evaluation

This repository compares a compact supervised detector, YOLO26n, with the
general-purpose LocateAnything-3B grounding model for person detection in
visible and thermal imagery.

The completed academic evaluation—including training provenance, LLVIP and
FLIR results, qualitative errors, limitations, and pilot appendices—is in
[reports/EVALUATION.md](reports/EVALUATION.md).

## What is evaluated

- **LLVIP:** all 3,463 paired official test IDs in both visible and infrared
  modalities.
- **FLIR ADAS v2:** 1,144 official-validation 8-bit AGC thermal images as a
  frozen-model external-domain test.
- **Models:** pretrained YOLO26n, a clean LLVIP-thermal-fine-tuned YOLO26n, and
  `nvidia/LocateAnything-3B`.

The primary comparison uses the exact boxes emitted at the locked operating
point: YOLO confidence 0.25 and LocateAnything hybrid generation. Precision,
recall, F1, matched IoU, FP/FN rates, output failures, and paired bootstrap
intervals are reported at IoU 0.50 and 0.75. Separate YOLO confidence sweeps
provide secondary AP metrics.

## Local setup

The local environment is only for dataset preparation, analysis, figures, and
tests. GPU model environments are pinned inside the Modal applications.

```console
uv venv
uv pip install -r requirements.txt
uv run -m unittest discover -s tests -v
```

## Prepare the datasets

Download and prepare LLVIP:

```console
uv run scripts/download_dataset.py
uv run scripts/prepare_dataset.py
```

Preparation preserves the source modalities, builds leakage-free
sequence-grouped train/validation splits, and keeps the official paired test set
locked. The clean split contains 9,620 thermal training images, 2,405 validation
images, and 3,463 test pairs.

Download the expanded FLIR ADAS v2 archive through the
[official Teledyne FLIR registration page](https://oem.flir.com/en-gb/solutions/automotive/adas-dataset-form/)
and prepare only its official-validation 8-bit AGC thermal images:

```console
uv run scripts/prepare_flir.py --source FLIR_ADAS_v2.zip
```

If archive discovery is ambiguous, the command stops and lists candidates;
resolve them explicitly with `--annotations` and `--images`. FLIR preparation
preserves person-negative images and COCO crowd-ignore semantics.

## Run the locked Modal evaluation

The Modal applications verify the exact archive, split-manifest, model revision,
and record identity before resuming a run. Upload the already prepared locked
payloads to the volume names declared in each application.

Train the clean single-seed thermal checkpoint:

```console
uvx modal run modal_apps/yolo_train.py \
  --run-id yolo26n-thermal-e50-seed20260721 --epochs 50 \
  --seed 20260721 --batch 64
```

Run the six LLVIP model/modality combinations and the separate YOLO AP sweeps:

```fish
uvx modal run modal_apps/full_inference.py --target yolo
uvx modal run modal_apps/full_inference.py --target yolo-ap
for modality in visible infrared
    for shard in 0 1 2 3
        uvx modal run modal_apps/full_inference.py \
            --target locate-$modality --shard-index $shard --shard-count 4 &
    end
end
wait
uvx modal run modal_apps/full_inference.py \
    --target finalize-visible --shard-count 4
uvx modal run modal_apps/full_inference.py \
    --target finalize-infrared --shard-count 4
```

Run the frozen models on FLIR:

```fish
uvx modal run modal_apps/flir_inference.py --target yolo
uvx modal run modal_apps/flir_inference.py --target yolo-ap
for shard in 0 1 2 3
    uvx modal run modal_apps/flir_inference.py \
        --target locate --shard-index $shard --shard-count 4 &
end
wait
uvx modal run modal_apps/flir_inference.py \
    --target finalize-locate --shard-count 4
```

LocateAnything shards are deterministic and disjoint. The examples use fish
shell and run all four indices before finalization.

## Analyze downloaded artifacts

Place the canonical LLVIP prediction JSONL files under `artifacts/full/` and
the FLIR files under `artifacts/flir/full/`, then run:

```console
uv run scripts/build_test_attributes.py
uv run scripts/summarize_full.py
uv run scripts/plot_full_results.py
uv run scripts/select_qualitative_examples.py
uv run scripts/summarize_flir.py --sample full
```

The summarizers validate identity and completeness before writing
machine-readable JSON. LLVIP plots go to `reports/figures/`; the qualitative
selector also writes its fixed subset JSONL and TP/FP/FN overlays. FLIR pilot
validation remains available with:

```console
uv run scripts/summarize_flir.py --sample pilot
```

Pilot figures are written beneath `artifacts/flir/pilot/`, so they cannot
overwrite the final publication figures.

## Repository layout

- `evaluation/`: prediction schema, parsing, matching, AP, and bootstrap logic.
- `modal_apps/`: clean training and locked full-evaluation applications.
- `scripts/`: dataset preparation and final analysis entry points.
- `manifests/`: locked LLVIP attributes and LLVIP/FLIR pilot or payload
  provenance.
- `reports/EVALUATION.md`: reviewed consolidated evaluation.
- `reports/figures/`: the six publication figures.
- `artifacts/` and `datasets/`: ignored local/remote outputs and prepared data.
