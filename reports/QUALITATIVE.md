# Fixed Qualitative Error Analysis

`scripts/select_qualitative_examples.py` deterministically selected three
non-overlapping examples per available category for each headline comparison at
IoU 0.50. Selection uses locked model results and is intended for error
interpretation, not prevalence estimation. The machine-readable selection is
`artifacts/full/qualitative/selection.json`; its subset JSONL files and 60
TP/FP/FN overlays are under `artifacts/full/qualitative/overlays/`.

## Taxonomy

| Category | Meaning |
| --- | --- |
| YOLO advantage | Largest positive per-image F1 difference for the headline YOLO state |
| LocateAnything advantage | Largest positive per-image F1 difference for LocateAnything |
| Both strong | Highest minimum F1 across both models |
| Both miss | Neither model matches a ground-truth box |
| LocateAnything output failure | `no_output`, `malformed`, or `error` parser status |
| LocateAnything duplicates | Repeated normalized boxes in one generated response |

The selector found three examples in every category except duplicates; no
duplicate-output examples remained after non-overlapping category selection.

## Inspected Examples

### Visible image `190051`: LocateAnything advantage

Pretrained YOLO misses the single partially occluded pedestrian near the upper
edge (one FN). LocateAnything returns one matched box with no FP or FN, giving
per-image F1 1.0 versus 0.0. This is consistent with the aggregate visible
recall advantage, not merely extra false-positive tolerance.

- YOLO overlay: `artifacts/full/qualitative/overlays/yolo_pretrained_visible/190051.jpg`
- LocateAnything overlay: `artifacts/full/qualitative/overlays/locate_anything_visible/190051.jpg`

### Thermal image `190099`: fine-tuned YOLO advantage

The person is a narrow, partially clipped target at the far-right image edge.
Fine-tuned YOLO matches it with no FP or FN. LocateAnything emits the explicit
`<box>None</box>` response, producing one FN and per-image F1 0.0.

- YOLO overlay: `artifacts/full/qualitative/overlays/yolo_finetuned_infrared/190099.jpg`
- LocateAnything overlay: `artifacts/full/qualitative/overlays/locate_anything_infrared/190099.jpg`

### Visible image `240275`: shared dark-scene failure

Both methods miss all four heavily obscured pedestrians. LocateAnything also
generates a reversed/degenerate normalized box
`<box><878><541><158><748></box>`, which the parser records as malformed rather
than silently repairing. This example combines a genuinely difficult scene with
an auditable structured-generation failure.

### Thermal image `190698`: grounding no-output failure

LocateAnything emits the explicit no-object token and misses all three people.
Fine-tuned YOLO matches two of the three (per-image F1 0.8); the remaining miss
is a small/clipped edge target. This illustrates why no-output rate is reported
separately from ordinary localization and classification errors.

## Interpretation

The inspected pairs support the quantitative pattern: LocateAnything's visible
advantage often comes from recovering difficult or partially occluded people,
while supervised thermal YOLO more consistently detects narrow edge targets and
crowded thermal pedestrians. LocateAnything failures include both valid
no-object generations and a small number of malformed coordinates. These
examples are fixed extremes and should not be treated as a random sample.
