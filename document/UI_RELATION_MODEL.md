# Unified UI defect model on LocateAnything-3B

This implementation covers the shared-model/SFT stage. It keeps one MoonViT,
one projector, one Qwen2.5 decoder and the existing six-token PBD box protocol.
It does not instantiate five visual experts and is not a conventional MoE.
Zoom/Crop policy learning and RL are intentionally left to the tool-policy
stage; the visual cache, query attention, coarse boxes and defectness output are
exposed for that stage.

## Task mapping

| defect type id | task | relation family id | relation query |
| ---: | --- | ---: | --- |
| 0 | text overflow | 0 | boundary |
| 1 | cropped element | 0 | boundary |
| 2 | overlapping elements | 1 | pairwise |
| 3 | abnormal text ellipsis | 2 | text |
| 4 | missing content | 3 | presence |

Text overflow and cropping share boundary reasoning, but retain separate defect
embeddings. Every relation family has eight Evidence/Context slot pairs.

## Model path

`MoonVitEncoder` captures only three configured layers (defaults for 27 layers:
5, 15 and 26). The final MoonViT stream still passes through the unchanged
patch merger and `mlp1` projector. The three unmerged maps are independently
normalized/projected to 256 dimensions, then fused with a learnable
relation-specific softmax gate. Initial early/middle/final weights are:

| family | early | middle | final |
| --- | ---: | ---: | ---: |
| boundary | 0.50 | 0.35 | 0.15 |
| pairwise | 0.15 | 0.40 | 0.45 |
| text | 0.40 | 0.25 | 0.35 |
| presence | 0.10 | 0.20 | 0.70 |

Evidence and Context queries cross-attend directly to the fused patch map. For
each slot the relation state is:

```text
r = MLP(z_e, z_c, z_e - z_c, z_e * z_c)
relation_token = sigmoid(gate_logit) * family_adapter(r)
```

The gate is deliberately not reported as final detection confidence.
`p_defect` is the conservative maximum slot defectness. BBox quality and future
tool confidence remain separate outputs.

No detail patch or relation token is added to the Qwen sequence. Gated relation
residuals are injected only into states whose input token is `<box>`: the
semantic/negative residual supports the `none` versus coordinate decision, and
the box residual supports coordinate prediction. The original six-token box
format and 8192 context remain unchanged.

## Weak supervision and loss

The data loader parses only the current screenshot, task text, positive/negative
answer and normalized bbox. For the first `min(number_of_boxes, 8)` slots:

- Evidence target: patch centers inside the GT bbox.
- Context target: patch centers in the centered 3x enlarged bbox, excluding the
  original bbox.
- Gate target: 1 for an assigned box slot; 0 for unused slots.
- Negative sample: all related slot targets are 0.

Tiny boxes that contain no patch center use the closest patch as a stable weak
target. The total SFT objective is language/PBD loss plus class-balanced focal
gate loss plus 0.1 times the Evidence/Context attention loss.

Formal sampling fixes every task at 17,604 records per effective epoch and uses
a 1:2 positive/negative ratio with replacement. This is 88,020 records total.
It explicitly oversamples the 742 text-overflow positives while retaining twice
as many negatives to prioritize false-positive reduction. Epoch ordering evenly
interleaves positive/negative examples inside each class and round-robins all
five classes before the bounded packing buffer.

## Unified interface

Training `forward(..., return_ui_defect_outputs=True)` returns
`UIDefectModelOutput`. Exported inference checkpoints support
`generate(..., return_ui_defect_interface=True)` and
`get_last_ui_defect_interface()`. The stable keys are:

```text
relation_tokens
relation_family
p_defect
coarse_boxes
query_attention
box_anchor_hidden
coordinate_logits
global_visual_cache
```

`global_visual_cache` contains the unchanged merged visual features, three
detail maps and `image_grid_hws`. It is returned only when explicitly requested
in training, so ordinary SFT does not retain a large extra output graph.

## Parameter budget and checks

With MoonViT hidden size 1152, relation size 256, adapter bottleneck 64 and Qwen
hidden size 2048, the complete Detail Pyramid, Query Bank, four adapters/gates
and two PBD projections add exactly 2,624,790 parameters. This is about 0.088%
of a 3B model and comfortably below the 5% ceiling.

Run in the LocateAnything environment:

```bash
python -m unittest tests.test_ui_defect_data tests.test_relation_modules
python scripts/check_ui_relation_budget.py /path/to/checkpoint
DRY_RUN=1 bash shell/run_locany_ui_defect.sh a800
```

The local smoke data validates plumbing only. Image F1/BBox F1 targets require
formal SFT and the fixed 7,775-record evaluation set; the code does not claim
those metrics before an actual training run.
