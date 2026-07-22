# Fine-tuned Object Detection vs. General-purpose Grounding Vision-enabled LLM for Human Detection task on thermal images

## Research Question

- Can a general-purpose vision-enabled LLM beat a fine-tuned object detection
  model on domain-shifted detection task?
- Up-front fine-tuning cost and downstream inference performance tradeoff.

## Evaluation Setup

### Dataset and Models

- **Dataset**: [LLVIP: A Visible-infrared Paired Dataset for Low-light Vision](https://github.com/bupt-ai-cz/LLVIP)
- **Object Detection model**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **Vision-language Grounding model**: [LocateAnything](https://research.nvidia.com/labs/lpr/locate-anything)

### Environment

The code is designed to run on [Modal](https://modal.com) sandboxes with GPU
acceleration.

For local dataset preparation and evaluator tests:

```console
uv venv
uv pip install -r requirements.txt
uv run scripts/download_dataset.py
uv run scripts/prepare_dataset.py
uv run -m unittest discover -s tests -v
```

Preparation preserves the source `visible` and `infrared` directories and
creates hard-linked YOLO trees under `datasets/LLVIP-YOLO-*`. The official
12,025 training pairs are split by capture-sequence prefix into 9,620 training
and 2,405 validation pairs. All 3,463 official test pairs remain locked for
final evaluation. The versioned split and source-annotation audit are stored in
`datasets/LLVIP-splits-v1.json`.

Phase 2 uses one JSONL prediction schema for every model. Once an adapter has
produced records, evaluate them at both primary IoU thresholds with:

```console
uv run scripts/evaluate_predictions.py \
  --predictions artifacts/predictions.jsonl \
  --dataset-dir datasets/LLVIP-YOLO-infrared \
  --output artifacts/metrics.json
```

Render TP/FP/FN assignments for hand-checking with:

```console
uv run scripts/render_overlays.py \
  --predictions artifacts/predictions.jsonl \
  --dataset-dir datasets/LLVIP-YOLO-infrared \
  --output-dir artifacts/overlays
```

The inference entry points are resumable and append one validated record per
image. YOLO26 defaults explicitly to its one-to-one end-to-end head. The
LocateAnything runner pins the model revision recorded in `AI-ROADMAP.md`, uses
the official person prompt and hybrid generation, converts every image to RGB,
and preserves both raw text and parser diagnostics:

```console
uv pip install -r requirements-yolo.txt
uv run scripts/run_yolo.py --help

# In the separate NVIDIA/CUDA environment described by
# requirements-locate-anything.txt:
uv run scripts/run_locate_anything.py --help
```

### Full Modal experiment

The remote data Volume uses archives rather than tens of thousands of separate
uploads. The locked paired-test payload is one 787 MB file,
`LLVIP-test-paired.tar`, with SHA256
`8b4db30cc40279cf04105cdf1859d6961a55182afe072617d409ccc77ec1ba6b`.
The Modal runner verifies that hash, filters macOS AppleDouble metadata while
extracting, verifies the split-manifest hash, and requires the exact 3,463
paired stems before loading a model.

Run the four YOLO configurations and their separate AP sweep with:

```console
uvx modal run modal_apps/full_inference.py --target yolo
uvx modal run modal_apps/full_inference.py --target yolo-ap
```

LocateAnything supports one canonical resumable run or deterministic disjoint
shards. The following fish-shell form runs four shards per modality in parallel:

```fish
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

The CPU-only finalizer refuses missing, duplicate, out-of-split, or mismatched
records before atomically creating each canonical JSONL; it preserves the
pre-shard partial file as a backup. After downloading the six canonical files
into `artifacts/full` and four YOLO AP summaries into `artifacts/full/ap`,
generate the locked report with:

```console
uv run scripts/build_test_attributes.py
uv run scripts/summarize_full.py
uv run scripts/plot_full_results.py
uv run scripts/select_qualitative_examples.py
```

The plotting script reads `artifacts/full/full_summary.json` and writes the
report figures under `reports/figures`.

### Methodology

The Ultralytics YOLO26 model was pretrained on COCO dataset [cite], while
LocateAnything was trained on a multi-domain dataset [cite]. In both cases,
it is not clearly indicated that there are thermal images in the datasets,
so it is assumed that the base models are designed for RGB image use cases.

The experiment runs the models through both modalities offered by LLVIP:

| Model              | RGB set | Thermal set |
| ------------------ | ------: | ----------- |
| Pretrained YOLO26n |     Yes | Yes         |
| LocateAnything-3B  |     Yes | Yes         |
| Fine-tuned YOLO26n |     Yes | Yes         |

## References

```tex
@misc{jia2023llvipvisibleinfraredpaireddataset,
title={LLVIP: A Visible-infrared Paired Dataset for Low-light Vision},
author={Xinyu Jia and Chuang Zhu and Minzhen Li and Wenqi Tang and Shengjie Liu and Wenli Zhou},
year={2023},
eprint={2108.10831},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2108.10831},
}
@article{Jocher_Ultralytics_YOLO26_Unified_2026,
author = {Jocher, Glenn and Qiu, Jing and Liu, Mengyu and Lyu, Shuai and Akyon, Fatih Cagatay and Kalfaoglu, Muhammet Esat},
doi = {10.48550/arXiv.2606.03748},
title = {{Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models}},
url = {https://arxiv.org/abs/2606.03748},
year = {2026}
}
@article{wang2025locateanything,
title   = {LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding},
author  = {Shihao Wang and Shilong Liu and Yuanguo Kuang and Xinyu Wei and Yangzhou Liu and Zhiqi Li and Yunze Man and Guo Chen and Andrew Tao and Guilin Liu and Jan Kautz and Lei Zhang and Zhiding Yu},
journal = {arXiv preprint arXiv:2605.27365},
year    = {2026},
}
```
