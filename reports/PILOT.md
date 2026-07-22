# Locked 100-Pair Pilot

Generated from `manifests/LLVIP-pilot-100-v1.json`. This is a dataset-only, sequence/brightness/crowd-stratified pilot; it is not final test evidence.

| Model state | Modality | P@.50 | R@.50 | F1@.50 | F1@.75 | Median ms | p95 ms | Peak GiB | Projected full cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nvidia/LocateAnything-3B | infrared | 0.955 | 0.834 | 0.890 | 0.737 | 1768.8 | 1955.2 | 13.92 | $3.37 |
| nvidia/LocateAnything-3B | visible | 0.895 | 0.821 | 0.856 | 0.579 | 1767.1 | 2065.1 | 13.92 | $3.40 |
| old-test-leaked-best.pt | infrared | 0.954 | 0.908 | 0.931 | 0.765 | 7.4 | 7.6 | 0.06 | $0.01 |
| old-test-leaked-best.pt | visible | 0.513 | 0.087 | 0.149 | 0.022 | 7.4 | 7.6 | 0.06 | $0.02 |
| yolo26n.pt | infrared | 0.880 | 0.576 | 0.697 | 0.538 | 7.6 | 8.2 | 0.06 | $0.01 |
| yolo26n.pt | visible | 0.902 | 0.646 | 0.753 | 0.499 | 7.4 | 8.1 | 0.06 | $0.02 |

Cost uses Modal's July 22, 2026 L40S rate of $0.000542/second and warm batch-1 model time. It excludes container startup, model download, CPU, memory, storage, and any future batch-throughput optimization.

Two LocateAnything records returned the explicit no-object token; there were no malformed or error records after parser normalization.

The `old-test-leaked-best.pt` row is pipeline-only context. That checkpoint used the official test split for selection and cannot support final claims.

A three-seed probe on the first ten pilot pairs found identical raw output for 7/10 images in each modality. F1@.50 ranged from 0.898 to 0.917 on visible and stayed at 0.939 on infrared, so LocateAnything sampling is not strictly deterministic even with otherwise fixed settings.

Current pricing: https://modal.com/pricing
