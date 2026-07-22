# Clean YOLO26n Thermal Training

## Outcome

The clean single-seed YOLO26n run completed all 50 epochs on an NVIDIA A10.
Training used only 9,620 thermal training images; checkpoint selection and
early stopping used only the 2,405-image validation split. The official LLVIP
test split was not present in the runtime dataset YAML.

The selected `best.pt` was revalidated on the validation split with:

| Metric | Value |
| --- | ---: |
| Precision | 0.9321 |
| Recall | 0.9167 |
| mAP50 | 0.9503 |
| mAP50-95 | 0.5879 |

The per-epoch CSV selected epoch 44 with mAP50-95 0.5877. Ultralytics' final
revalidation of that saved checkpoint produced the 0.5879 value above.

## Reproducibility

| Field | Value |
| --- | --- |
| Run ID | `yolo26n-thermal-e50-seed20260721` |
| Seed | `20260721` |
| Ultralytics | `8.4.102` |
| PyTorch | `2.13.0+cu130` |
| GPU | NVIDIA A10 |
| Image size | 640 |
| Batch size | 64 |
| Image cache | deterministic disk cache |
| Initial checkpoint SHA256 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Best checkpoint SHA256 | `66ba7bf3c07ea894e96767cc184d2f060d1baa0f8aaa3f6912a9600ddbdf0eed` |
| Last checkpoint SHA256 | `8642c2cc271805bca13d0892bc33e1d62c9680a1784f65cffda7e9bb67943aee` |
| Split manifest SHA256 | `05facc1b82630ec515cfdb0df16617f1c6390fc5af009b4c090a8343e78b33ef` |
| Train/validation archive SHA256 | `2916455d4e9afa6c0c5d74db3785a6a2f8adc9b304ea797fc3699095fe6a3c44` |
| Training app SHA256 | `5b5f6ad439bf7139540c95b09c5b886f4305a84d509c891359a6eb610cd71bed` |

The initial repository has no Git commit and all project files remain
untracked; this state is recorded in the run summary rather than represented
as a nonexistent revision.

The durable run is stored in Modal Volume `llvip-experiment-artifacts` at
`training/yolo26n-thermal-e50-seed20260721/`. It includes `best.pt`, `last.pt`,
`args.yaml`, `results.csv`, plots, the exact split manifest, the training-only
dataset YAML, requested settings, package versions, and checkpoint hashes.

## Runtime and Cost

Measured training-function time was 2,859.2 seconds (47 minutes 39 seconds),
with 8.90 GiB peak allocated GPU memory. At Modal's July 22, 2026 published
rates for one A10, eight CPU cores, and 24 GiB memory, that interval is about
$1.33. This estimate excludes the short container-start and pre-training
archive-verification interval. Current rates are available on the
[Modal pricing page](https://modal.com/pricing).

## Training Settings

The run initialized from official `yolo26n.pt` and used the default
end-to-end detection head. It retained the earlier experiment's deliberate
thermal hyperparameters: 50 epochs, MuSGD, `lr0=0.0054`, `lrf=0.0495`,
`momentum=0.947`, `weight_decay=0.00064`, `warmup_epochs=0.98`, patience 20,
and the recorded low-mosaic thermal augmentations. Best-checkpoint fitness was
Ultralytics' default detection fitness, evaluated only on the validation split.
