# LocateAnything UI CPT

This pipeline continues training `nvidia/LocateAnything-3B` on the UI v4.1
caption, action, grounding, defect, OCR, referring, and VQA mixture.  It reuses
the same model loader, image processor, online packing, DeepSpeed ZeRO-2,
stateful dataloader resume, and `checkpoint-<global_step>` lifecycle as the
current LocateAnything UI5 pipeline.

## Data format

The raw files are streamed into ten task-family JSONL files.  Every output row
uses LocateAnything's native structure:

```json
{
  "conversations": [
    {"from": "human", "value": "<image>..."},
    {"from": "gpt", "value": "..."}
  ],
  "image": "/absolute/or/root-relative/image.jpg",
  "cpt_task": "single_grounding"
}
```

Grounding answers use normalized `[0,1000]` coordinates:

```text
<ref>目标文字</ref><box><x1><y1><x2><y2></box>
```

Qwen-VL `<|object_ref_start|>/<|box_start|>` markers and structured
`objects.bbox`/JSON `bbox_2d` targets are converted to this grammar.  Caption,
action prediction, VQA, and region-function descriptions retain their natural
language answers.

## Balanced streaming

The recipe writes `sampling_weight: 1.0` for every task family.  The online
packer therefore samples each of the ten task families with probability 10%,
independent of file size.  It does not truncate the 608K single-grounding set
to the 5,667-example defect-set size, and it does not materialize repeated
copies of small files.  Each task iterator still visits its own records and
reshuffles when exhausted.

## H20: prepare all data

Run this once on the HL cluster:

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied-CPT

python scripts/prepare_locany_cpt.py \
  --source-root /mnt/bn/intelligent-service-arnold-hl/dataset/gui/gui_base/sample/raw_data_v4.1_hl_norm1k/raw_data_v4.1_hl \
  --output-dir /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data/locany_cpt_v4 \
  --recipe-name locany_cpt_train.json
```

The converter writes rejected row locations and reasons to `rejected.jsonl`.
It fails when the rejected rate exceeds 0.1%.

## A100: portable four-card smoke data

On a node that can read the HL source, export eight rows per task and copy the
referenced images into a small self-contained directory:

```bash
python scripts/prepare_locany_cpt.py \
  --source-root /mnt/bn/intelligent-service-arnold-hl/dataset/gui/gui_base/sample/raw_data_v4.1_hl_norm1k/raw_data_v4.1_hl \
  --output-dir /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data/locany_cpt_v4_smoke \
  --recipe-name locany_cpt_smoke.json \
  --max-records-per-task 8 \
  --copy-images
```

Copy the complete `locany_cpt_v4_smoke` directory to:

```text
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/locany_cpt_v4_smoke
```

Because copied image paths are root-relative, no JSONL rewriting is needed
after the directory moves.

## Four-card Merlin submissions

A100-profile smoke test on YG (the current YG Arnold resource enum is
`A800_SXM_40GB`, four cards, SDPA):

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied-CPT
mlx job submitv2 --path locany_cpt_v4_a100x4_smoke_merlin.yaml
```

The default smoke run performs two optimizer steps and saves
`checkpoint-2`.

A100-profile formal training on YG:

The complete prepared directory must exist at
`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/locany_cpt_v4`
before submission.

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied-CPT
mlx job submitv2 --path locany_cpt_v4_a100x4_formal_merlin.yaml
```

H20 Magi formal training on HL:

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied-CPT
mlx job submitv2 --path locany_cpt_v4_h20x4_formal_merlin.yaml
```

Formal defaults are full-parameter training, four GPUs, gradient accumulation
2, learning rate `5e-6`, 20,000 optimizer steps, A100-profile `7268/12800`
and H20 `8192/25600` per-sample/per-rank packed-token limits, and a checkpoint
approximately every 12 hours. Trainer checkpoint directories always include
the current optimizer global step, for example `checkpoint-4372`.

All defaults can be overridden without editing code:

```bash
RUN_NAME=locany-3b-ui-cpt-v4-h20x4-run2 \
MAX_STEPS=30000 \
LEARNING_RATE=3e-6 \
SAVE_EVERY_N_HOURS=12 \
bash shell/run_locany_cpt.sh h20 formal
```

The direct `bash shell/run_locany_cpt.sh ...` commands are retained for an
already allocated interactive worker.  Normal smoke and formal runs should be
submitted through the Merlin YAML files above.  All three YAMLs use
`shell/run_locany_cpt_merlin.sh` for cache isolation, four-GPU preflight,
logging, and exit-code propagation, then delegate training parameters to
`shell/run_locany_cpt.sh`.

An interrupted run resumes from the newest complete checkpoint in the same
`OUTPUT_DIR`, including optimizer, scheduler, random state, and packed
dataloader state.
