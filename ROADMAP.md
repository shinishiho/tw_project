# Roadmap

## Dataset acquisition and preprocess

- [x] Download and checksum the dataset
- [x] Prepare group-aware train/validation splits and a locked paired test split

## Unified inference and evaluation harness

- [x] Define the resumable model-independent prediction JSONL schema
- [x] Parse and audit LocateAnything structured box output
- [x] Implement traceable one-to-one matching and primary metrics
- [x] Add resumable YOLO and pinned LocateAnything inference adapters
- [x] Add visual overlays and run a 20-pair YOLO pipeline smoke test
- [x] Run both adapters on paired images on an NVIDIA GPU and inspect overlays

## 100-pair Modal pilot

- [x] Lock a dataset-only sequence/brightness/crowd-stratified sample
- [x] Run all three pilot model states on both modalities on L40S
- [x] Measure accuracy, batch-1 latency, memory, failures, and projected cost
- [x] Probe LocateAnything sampling stability with three seeds
- [ ] Benchmark optimized LocateAnything batch throughput separately

## Run YOLO26n finetuning

- [x] Finetune YOLO26n on LLVIP thermal set

## Evaluation

- [x] Run pretrained YOLO26n checkpoint on LLVIP RGB and thermal set
- [x] Run LocateAnything on LLVIP RGB and thermal set
- [x] Run finetuned YOLO26n checkpoint on LLVIP RGB and thermal set

## Analysis and reporting

- [x] Validate six canonical prediction files against all 3,463 locked paired IDs
- [x] Compute confidence-independent metrics and 2,000-replicate paired intervals
- [x] Run separate YOLO AP sweeps
- [x] Report latency, memory, GPU cost, brightness, crowd, and object-size strata
- [x] Select and inspect a fixed qualitative disagreement taxonomy
