# LLVIP Generalist Grounding vs. Object Detection Roadmap

## Objective

Compare a generalist vision-language grounding model with a conventional object
detector on low-light visible and thermal pedestrian detection.

The two headline comparisons are:

1. On visible LLVIP images, compare pretrained YOLO26n with
   `nvidia/LocateAnything-3B`.
2. On thermal LLVIP images, compare a thermal-fine-tuned YOLO26n with
   `nvidia/LocateAnything-3B`.

Additional control runs will separate the effects of image modality and
fine-tuning. Accuracy, robustness, latency, throughput, and GPU cost will be
reported separately.

## Current Assets and Limitations

The repository already contains:

- The LLVIP archive, whose MD5 is
  `e64affb4b0b50e1772ff6f67da873bf6`.
- 12,025 training pairs and 3,463 test pairs with matching visible and thermal
  filenames.
- Dataset acquisition and Pascal VOC to YOLO conversion code in `main.ipynb`.
- YOLO26n training, inference, and evaluation code in `main.ipynb`.
- A YOLO26n pedestrian checkpoint at
  `runs/detect/llvip_yolo26n/weights/best.pt`.
- Results and artifacts from the earlier Qwen-based detection post-filter
  experiment.

The rebooted checkout now preserves the extracted source directories as
`visible` and `infrared`. `scripts/prepare_dataset.py` creates separate
hard-linked YOLO trees, a deterministic sequence-grouped validation split, and
a versioned paired manifest without renaming source data.

The existing YOLO checkpoint is suitable for smoke tests and demonstrations,
but not as the final experimental checkpoint. Its training configuration used
the official LLVIP test split as `val` during training and selected the best
checkpoint using metrics from that split. This leaks test information into
model selection. A clean checkpoint must therefore be trained using a new
validation subset taken only from the official training data.

## Experimental Design

Run every model state on both modalities:

| Model state | Visible test | Thermal test | Purpose |
| --- | ---: | ---: | --- |
| Pretrained YOLO26n | Yes | Yes | Specialist zero-shot baseline and thermal domain shift |
| LocateAnything-3B | Yes | Yes | Generalist grounding baseline and thermal robustness |
| Thermal-fine-tuned YOLO26n | Yes | Yes | Benefit and cross-modality cost of thermal specialization |

This produces five useful comparisons:

1. Pretrained YOLO26n vs. LocateAnything on visible images.
2. Fine-tuned YOLO26n vs. LocateAnything on thermal images.
3. Pretrained YOLO26n on visible vs. thermal images: modality shift.
4. Pretrained vs. fine-tuned YOLO26n on thermal images: value of supervision.
5. Fine-tuned YOLO26n on thermal vs. visible images: specialization cost.

### Locked Test Set

- Preserve the official 3,463-pair test set untouched until final evaluation.
- Use exactly the same paired image IDs for visible and thermal evaluation.
- Never use test metrics for checkpoint selection, early stopping, confidence
  calibration, prompt selection, or hyperparameter tuning.

### Training and Validation Split

- Divide the official 12,025-pair training set into training and validation.
- Split by acquisition scene or sequence rather than by random individual
  frames, because temporally or spatially related frames may be near
  duplicates.
- Investigate whether filename prefixes reliably identify acquisition groups.
  If they do, use a deterministic group-aware split and save it as a manifest.
- Apply one split manifest to both modalities so paired visible and thermal
  images never cross split boundaries.

### Reproducibility

Record the following for every run:

- Git commit and dirty-worktree state.
- Dataset archive hash and split-manifest hash.
- Model identifier, exact model revision, and checkpoint hash.
- Package versions and Modal image definition.
- GPU type, batch size, image size, precision, and inference settings.
- Prompt, generation mode, random seed, and parser version.
- Start/end timestamps and measured GPU-seconds.

Use immutable run IDs and write predictions incrementally in a resumable JSONL
format.

## Model Protocols

### YOLO26n

- Use the official COCO-pretrained `yolo26n.pt` as the pretrained baseline and
  as the initialization for thermal fine-tuning.
- Train only on thermal images from the new training split.
- Select the final checkpoint only using the new validation split.
- Keep image size, detection head, NMS settings, maximum detections, and
  confidence handling identical between pretrained and fine-tuned inference.
- Explicitly choose either the default end-to-end head or the one-to-many head;
  do not allow the two YOLO states to inherit different defaults.
- Run with one fixed seed for the minimum viable experiment. For a stronger
  result, train three seeds and report mean and standard deviation.

Report two YOLO operating points when useful:

1. A predefined confidence threshold such as 0.25.
2. A separately labelled threshold selected on the validation split.

### LocateAnything-3B

- Use the exact model identifier `nvidia/LocateAnything-3B`.
- Pin the Hugging Face revision because the model loads remote implementation
  code with `trust_remote_code=True`.
- Use the official single-category detection prompt for `person`:
  `Locate all the instances that matches the following description: person.`
- Use `generation_mode="hybrid"` for the main experiment.
- Convert both visible and thermal inputs to RGB without pseudo-coloring the
  thermal data.
- Use deterministic decoding if the implementation supports it. Otherwise,
  fix the full-run seed and repeat a fixed 200-image subset using three seeds
  to quantify generation instability.
- Count truncated, malformed, duplicate, out-of-range, and empty outputs rather
  than silently discarding them.
- Preserve the raw generated response alongside parsed boxes.

Pseudo-colored thermal input, alternate prompts, fast generation, and slow
generation are possible ablations, not part of the primary comparison.

## Evaluation

### Box Matching

- Use one-to-one matching between predicted and ground-truth boxes within each
  image and class.
- Process predictions consistently and ensure each ground-truth box can be
  matched at most once.
- Store TP, FP, and FN assignments so every aggregate number can be traced back
  to individual images and boxes.

### Primary Cross-Model Metrics

LocateAnything returns boxes without detector confidence scores. Conventional
average precision is therefore not directly comparable to YOLO AP. Assigning
every LocateAnything box the same score would create an arbitrary ranking and
must not be used as the main comparison.

Use confidence-independent metrics:

- Precision, recall, and F1 at IoU 0.50.
- Precision, recall, and F1 at IoU 0.75.
- Mean IoU of matched true positives.
- False positives and false negatives per image.
- Duplicate-box rate.
- Malformed-output and no-output rates.
- Paired bootstrap confidence intervals over test image pairs.

### Secondary Metrics

For YOLO, additionally report:

- AP50.
- AP75.
- mAP50-95.
- Precision-recall curves and threshold sensitivity.

For all models, report:

- Warm batch-1 latency, including preprocessing, inference, and output parsing.
- Median and p95 latency.
- Optimized batched throughput in a separate measurement.
- Peak GPU memory.
- GPU-seconds and estimated cost per 1,000 images.

Exclude model download and cold container startup from warm inference latency,
but report them separately as operational overhead.

### Statistical and Qualitative Analysis

- Compute paired bootstrap intervals by resampling paired test image IDs and
  recomputing aggregate metrics.
- Report differences between models with confidence intervals, not only point
  estimates.
- Stratify results by pedestrian box size, visible-image brightness, and crowd
  count.
- Inspect a fixed sample of disagreements and categorize localization errors,
  missed pedestrians, background false positives, duplicate detections, and
  malformed grounding output.
- Include identical paired visible/thermal examples so modality effects can be
  inspected directly.

## Modal Architecture

Use separate Modal images for YOLO and LocateAnything. LocateAnything pins a
specific Transformers/Numpy stack that should not be forced into the existing
Python 3.13 project environment.

Use a Modal Volume for:

- The verified LLVIP archive and extracted dataset.
- Hugging Face model cache.
- YOLO checkpoints and training runs.
- Prediction JSONL files.
- Split manifests, configuration snapshots, and evaluation outputs.

Load each model once per warm container. Batch inputs within the container and
limit container concurrency during the pilot to avoid paying for duplicate
model loads.

Recommended initial hardware:

- LocateAnything pilot and inference: one L40S, with A100 40 GB as a fallback.
- YOLO training: begin with A10 or L40S and increase only if the measured
  training time justifies it.
- YOLO inference: use the same GPU as LocateAnything for controlled latency
  comparison, then optionally measure it on cheaper hardware separately.

As of July 21, 2026, Modal's published base GPU rates are approximately:

| GPU | Cost per GPU-hour |
| --- | ---: |
| A10 | $1.10 |
| L40S | $1.95 |
| A100 40 GB | $2.10 |
| H100 | $3.95 |

Treat these values as time-sensitive. Record the current rate when the
experiment is executed.

## Implementation Roadmap

### Phase 1: Rebuild the Data Layer

**Status (July 21, 2026): complete.** The selected validation sequences are
01, 03, 11, 12, and 18, giving 9,620 train / 2,405 validation / 3,463 locked
test pairs. The manifest SHA256 is
`05facc1b82630ec515cfdb0df16617f1c6390fc5af009b4c090a8343e78b33ef`.
Five zero-width source boxes are recorded and excluded; two images consequently
have empty labels. A forced second preparation produced the identical generated
artifact hash.

- Move acquisition and conversion logic out of the notebook into standalone
  commands.
- Preserve source directories as `visible` and `infrared`.
- Make extraction and conversion idempotent.
- Generate modality-specific YOLO dataset YAML files without renaming source
  data.
- Generate and validate the group-aware train/validation/test manifest.
- Verify pair counts, stem equality, image dimensions, label bounds, and empty
  annotations.

**Exit condition:** A clean checkout can prepare the dataset twice with the
same outputs and hashes, without errors or destructive renaming.

### Phase 2: Build a Unified Inference and Evaluation Harness

**Status (July 22, 2026): complete.** The prediction record, resumable JSONL
I/O, LocateAnything parser diagnostics, maximum-cardinality/maximum-IoU
one-to-one matcher, primary aggregate metrics, traceable evaluation command,
YOLO adapter, pinned LocateAnything worker, and overlays are implemented and
covered by hand-checkable tests. LocateAnything is pinned to Hugging Face
revision `c32291ca5e996f5a7a485845b4f57a233936bba0`. The old YOLO checkpoint
and LocateAnything both completed paired GPU smoke tests with hand-inspected
overlays, satisfying the phase exit condition.

- Define one prediction schema for YOLO and LocateAnything.
- Implement LocateAnything output parsing and validation.
- Implement one-to-one box matching and all primary metrics.
- Add visual overlays for ground truth, matched predictions, false positives,
  and false negatives.
- Run 20-50 paired images locally or on a small Modal job.
- Use the existing YOLO checkpoint only to validate this pipeline.

**Exit condition:** All model adapters produce valid, traceable predictions
and the evaluator reproduces hand-checked examples.

### Phase 3: Run a 100-Pair Modal Pilot

**Status (July 22, 2026): go/no-go exit condition complete.** The locked sample
is `manifests/LLVIP-pilot-100-v1.json` (SHA256
`c44c17e36bb8a629efc75a65e1325064b54ddd0036b1bc9fe45bd0c8cec6117c`).
All six model/modality combinations completed on L40S. LocateAnything used
13.92 GiB and about 1.77 s/image at batch 1; its two-modality full-test warm
GPU projection is about $6.77 at Modal's July 22 rate. The three-seed probe
confirmed modest sampling variation. `reports/PILOT.md` contains the full
table. Optimized batch throughput remains a separate measurement before the
full run, but the remote pipeline and spending estimate are now validated.

- Select a deterministic, representative 100-pair sample without using model
  results.
- Run all three model states on both modalities.
- Measure accuracy sanity checks, memory, latency, throughput, output failure
  rate, and GPU cost.
- Test LocateAnything batch sizes and deterministic decoding behavior.
- Extrapolate the full-run cost from measured GPU-seconds.

**Exit condition:** The complete pipeline works remotely and a defensible
full-run cost estimate is available before spending the remaining credits.

### Phase 4: Train the Clean YOLO Checkpoint

**Status (July 22, 2026): complete.** The single-seed run
`yolo26n-thermal-e50-seed20260721` completed 50 epochs on A10 in 2,859.2
seconds. The selected checkpoint used only the sequence-grouped validation
split and revalidated at 0.9503 mAP50 / 0.5879 mAP50-95. Its SHA256 is
`66ba7bf3c07ea894e96767cc184d2f060d1baa0f8aaa3f6912a9600ddbdf0eed`.
Best/last weights, settings, logs, environment, manifests, and hashes are in
the `llvip-experiment-artifacts` Modal Volume and summarized in
`reports/TRAINING.md`.

- Fine-tune `yolo26n.pt` on the thermal training split.
- Use only the validation split for early stopping and checkpoint selection.
- Save best and last checkpoints, full training logs, arguments, environment,
  and hashes to the Modal Volume.
- Run one seed for the minimum viable experiment or three for the stronger
  experiment.

**Exit condition:** A clean checkpoint is selected without test leakage and is
fully reproducible from saved configuration and split manifests.

### Phase 5: Lock and Run the Full Experiment

**Status (July 22, 2026): complete.** All six model/modality configurations
contain exactly 3,463 unique prediction records for the locked official test
IDs. The paired test archive SHA256 is
`8b4db30cc40279cf04105cdf1859d6961a55182afe072617d409ccc77ec1ba6b`.
Four YOLO files contain only `ok` records. LocateAnything visible contains
3,423 `ok`, 32 `no_output`, and 8 `malformed` records; infrared contains 3,414
`ok`, 37 `no_output`, and 12 `malformed` records. Neither modality has an
inference `error`. Sharded grounding runs were merged only after exact-ID and
identity validation; canonical partials and all shards remain preserved in the
Modal artifact Volume.

- Freeze all model revisions, prompts, settings, thresholds, and checkpoint
  hashes.
- Run the six model/modality combinations on all 3,463 official test pairs.
- Resume safely after interruption without duplicating records.
- Validate record counts and parsing failures before evaluation.

**Exit condition:** Every expected image has one auditable prediction record
per model/modality configuration.

### Phase 6: Analyze and Report

**Status (July 22, 2026): complete for the primary experiment.** At IoU 0.50,
LocateAnything beats pretrained YOLO on visible images by 0.091 F1 (95% paired
bootstrap interval 0.083 to 0.099 in LocateAnything's favor). Fine-tuned YOLO
beats LocateAnything on thermal images by 0.042 F1 (95% interval 0.035 to
0.048). `reports/RESULTS.md` contains the six-run table, 2,000-replicate paired
intervals, YOLO AP, output audit, efficiency, and dataset-only strata.
`reports/QUALITATIVE.md` documents the fixed disagreement taxonomy and inspected
overlays. Optimized LocateAnything batch throughput remains explicitly separate
from the completed warm batch-1 primary experiment.

- Produce the primary comparison table and paired confidence intervals.
- Add YOLO-specific AP metrics separately.
- Produce latency, throughput, memory, and cost tables.
- Analyze modality shift, fine-tuning gain, and specialization cost.
- Create paired qualitative examples and a fixed failure taxonomy.
- Document limitations, including LocateAnything's lack of confidence scores,
  possible stochastic generation, resolution differences, and unknown thermal
  representation in its pretraining data.

**Exit condition:** All headline claims are supported by locked test results,
uncertainty estimates, and traceable artifacts.

## Expected Deliverables

- Idempotent dataset preparation command.
- Versioned paired split manifest.
- Modal applications for YOLO and LocateAnything.
- Clean thermal-fine-tuned YOLO26n checkpoint.
- Raw, resumable prediction JSONL files.
- Unified matching and metrics implementation with tests.
- Quantitative accuracy and efficiency tables.
- Qualitative paired error-analysis figures.
- Reproducibility record containing hashes, versions, settings, and costs.

## Licensing and References

Both LLVIP and LocateAnything are released for non-commercial research use.
Retain their license notices and attribution in any published code or report.

- [LLVIP paper](https://arxiv.org/html/2108.10831)
- [LLVIP official repository](https://github.com/bupt-ai-cz/LLVIP)
- [LocateAnything-3B model card](https://huggingface.co/nvidia/LocateAnything-3B)
- [NVIDIA Eagle repository](https://github.com/NVlabs/Eagle)
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26)
- [Modal GPU documentation](https://modal.com/docs/guide/gpu)
- [Modal Volumes documentation](https://modal.com/docs/guide/volumes)
- [Modal pricing](https://modal.com/pricing)
