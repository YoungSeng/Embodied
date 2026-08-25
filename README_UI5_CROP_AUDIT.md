# UI5 detector crop audit (CPT disabled)

This branch is based on `locany-cpt-v1@c06f1479a11b0175579994b880466b57bba50a87`.
It reuses the updated Detail Pyramid, relation/gate, BF16 fixes, and UI5
training/evaluation infrastructure, but this workflow does **not** load CPT data,
expose a CPT training entrypoint, start training, or infer that good crop coverage
guarantees a model improvement.

The external `ui-region-parser` must remain a separate sibling checkout at
`06eaebf8eb4ea01e61b690f2ff972bf614915918`. The audit runner imports only its
detector classes and geometry functions. It does not use the parser's
basename-indexed annotation/image discovery and does not enter its rendering
loop during full detection, so the fixed checkout remains clean.

## One-time checkout

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied
git fetch origin
git worktree add ../Embodied-ui5-det-crop \
  -b locany-ui5-det-crop-v1 \
  origin/locany-cpt-v1

cd ..
git clone https://github.com/YoungSeng/ui-region-parser.git
git -C ui-region-parser checkout 06eaebf8eb4ea01e61b690f2ff972bf614915918
```

Before running, install the existing project/parser requirements and place the
models at the parser defaults (or pass the model overrides):

- `weights/PP-OCRv5_server_det_infer/`
- `weights/icon_detect_v3/model.pt`

## Cluster launch commands

Use one timestamped output directory for an immutable detector run. All five
paths are CLI-controlled; no source root is taken from a hand-edited global.

```bash
COMMON_ARGS=(
  --source-dir /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data
  --locany-data-dir data/ui_defect_locany
  --parser-root ../ui-region-parser
  --output-dir work_dirs/ui5_crop_audit_20260825
  --gpus 0,1,2,3
  --workers-per-gpu 1
  --resume
)

# 1. Task-aware manifest, content deduplication, overlap audit, 500–1000 image shards.
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage prepare

# 2. Four persistent Paddle workers; shard JSONL + completion marker per shard.
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage text

# 3. Paddle has exited; now four persistent Torch/OmniParser workers.
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage icon

# 4. Strict image-id/count/dimension merge.
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage merge

# 5. CPU-only A/B/C geometry, raw crops, preview labels, metrics and Excel.
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage crop-audit
```

`--stage all` executes the same stages in order. Detection results live only in
`detections/text`, `detections/icon`, and `detections/merged`; changing crop
parameters never overwrites or reruns them. A shard is skipped under `--resume`
only when its output and marker have the exact expected count and image-id set.
Writes use a temporary file followed by atomic rename.

The default is one model process per GPU and 2–4 image-loader threads per process.
Two processes per GPU are rejected unless `--allow-two-processes-per-gpu` is also
passed after a 2,000-image benchmark has shown sustained GPU utilization below
40%, stable memory below 12 GB, and an actual throughput gain.

Use separate detector directories for the process-count benchmark so neither run
can be mistaken for the formal detector output:

```bash
# Prepare the same stable 2,000-image subset in two output directories.
for N in 1 2; do
  bash shell/run_ui5_crop_audit.sh \
    --source-dir /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data \
    --locany-data-dir data/ui_defect_locany \
    --parser-root ../ui-region-parser \
    --output-dir "work_dirs/ui5_crop_benchmark_${N}proc" \
    --stage prepare --max-unique-images 2000
done

bash shell/run_ui5_crop_audit.sh \
  --source-dir /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data \
  --locany-data-dir data/ui_defect_locany --parser-root ../ui-region-parser \
  --output-dir work_dirs/ui5_crop_benchmark_1proc --stage text \
  --gpus 0,1,2,3 --workers-per-gpu 1 --resume

# Run only after nvidia-smi monitoring satisfies the utilization/memory gate.
bash shell/run_ui5_crop_audit.sh \
  --source-dir /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data \
  --locany-data-dir data/ui_defect_locany --parser-root ../ui-region-parser \
  --output-dir work_dirs/ui5_crop_benchmark_2proc --stage text \
  --gpus 0,1,2,3 --workers-per-gpu 2 \
  --allow-two-processes-per-gpu --resume
```

Compare `detections/text/stage_summary.json` throughput. The icon stage writes
the same summary layout. Formal runs must omit `--max-unique-images`.

## Invariants

- Detection is once per byte-unique original image, independent of task and GT.
- GT remains separate per `image_id × task`; basename is warning-only.
- `ui_content_missing` gets one lightly trimmed near-full crop because of its
  task identity. The same image uses regional crops for other tasks.
- Expansion boxes are geometric only. Saved crops are ordinary unannotated
  rectangular pixels from the source image; no masks, boxes, or background edits.
- Dense detections may form one near-full connected component. There is no
  whitespace/long-edge post-split.
- Any crop that partially intersects a GT is `training_eligible=false`; it is not
  a negative. At most one truly non-intersecting hard negative is retained for
  each image/task. This stage produces preview JSONL only.
- Train/val identity remains content-based; crop filenames are never randomly
  re-split. The overlap report surfaces any existing content leakage.

## Output layout

```text
work_dirs/ui5_crop_audit_20260825/
  manifest/
    unique_images.jsonl
    task_samples.jsonl
    shards/shard_*.jsonl
    overlap/source_overlap.json
  detections/
    detector_config.json
    text/shard_*.jsonl
    icon/shard_*.jsonl
    merged/detections.jsonl
  crop_audit/
    config_A/
    config_B/
    config_C/
    summary.json
    statistics.csv
    task_aware_manifest.jsonl
    gt_failures.jsonl
    ui5_crop_audit.xlsx
```

The workbook has exactly five decision sheets: `summary`, `task_overlap`,
`image_detail`, `gt_failures`, and `config_compare`. Overview images are separate
from raw crop files and are linked from the workbook.

Do not start the full-image/full+crop/full+crop-at-inference experiment until the
report is reviewed. The suggested minimum gate is at least 99% combined GT-box
containment across the four local tasks, at least 98% for each local task, and
zero detector boxes cut by crop boundaries. `ui_content_missing` is evaluated
separately as a near-full-image policy. Failed coverage must be addressed by
link/context geometry and detector behavior, never by reading GT to add a crop.
