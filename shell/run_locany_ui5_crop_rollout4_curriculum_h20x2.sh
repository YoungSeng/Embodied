#!/usr/bin/env bash
set -Eeuo pipefail

# Formal H20x2 continuation loop:
# train 200 optimizer steps -> stop torchrun -> five single-GPU UI5 workers
# -> merge/score -> best-only preservation -> resume from resume/latest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=shell/bash_error_report.sh
source "${SCRIPT_DIR}/bash_error_report.sh"

WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}"
ENV_DIR="${ENV_DIR:-${WORKSPACE}/conda_envs/LocateAnything}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
SUGGESTED_RUN_NAME="locany-ui5-crop-rollout4-curriculum-hard114-h20x2-sdpa7268-v1"
RUN_NAME="${RUN_NAME:-}"
[[ -n "${RUN_NAME}" ]] || locany_die 18 \
  "RUN_NAME must be explicit so each formal run has an intentional OUTPUT_DIR; suggested: ${SUGGESTED_RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/gui_models/Embodied-ui5-det-crop/${RUN_NAME}}"

MODEL_PATH="${MODEL_PATH:-${WORKSPACE}/gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000}"
BASE_MODEL="${BASE_MODEL:-${WORKSPACE}/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
if [[ -z "${PROCESSOR_PATH}" ]]; then
  for candidate in \
    "${BASE_MODEL}" \
    "${WORKSPACE}/cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0"; do
    if [[ -d "${candidate}" ]]; then
      PROCESSOR_PATH="${candidate}"
      break
    fi
  done
fi

ROLLOUT_BUNDLE_ROOT="${ROLLOUT_BUNDLE_ROOT:-${WORKSPACE}/gui_data/ui5_train_rollout_bundle_v1}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${WORKSPACE}/gui_rollouts/ui5-train-rollout8-h20x2-v6-20260904}"
# complete8.jsonl retains prompt/GT and the technical-error evidence that the
# compact sample_difficulty projection intentionally omits.  The recipe builder
# also cross-checks the compact sibling when one is supplied explicitly.
FROZEN_SELECTION="${FROZEN_SELECTION:-}"
[[ -n "${FROZEN_SELECTION}" ]] || locany_die 19 \
  "FROZEN_SELECTION must name one immutable selection produced by merge_ui5_rollout_selections.py"
ROLLOUT_DIFFICULTY="${FROZEN_SELECTION}/complete8.jsonl"
CURRICULUM_SOURCE_RECIPE="${CURRICULUM_SOURCE_RECIPE:-}"
CURRICULUM_REUSE_CROPS_FROM="${CURRICULUM_REUSE_CROPS_FROM:-}"
CURRICULUM_DATA_DIR="${CURRICULUM_DATA_DIR:-${OUTPUT_DIR}/curriculum_data}"
CURRICULUM_PROGRESS_INTERVAL_SECONDS="${CURRICULUM_PROGRESS_INTERVAL_SECONDS:-10}"
META_PATH="${CURRICULUM_DATA_DIR}/ui5_crop_rollout4_curriculum.json"
HARD_GROUPS_JSONL="${CURRICULUM_DATA_DIR}/hard_groups.jsonl"

EVAL_INPUT_DIR="${EVAL_INPUT_DIR:-${WORKSPACE}/data}"
EVAL_SCAN_NAME="${EVAL_SCAN_NAME:-horizontal_scan_v5_raw_detector_edge_aligned}"
EVAL_DETECTOR_CACHE="${EVAL_DETECTOR_CACHE:-${WORKSPACE}/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5}"
EVAL_DETECTOR_MANIFEST="${EVAL_DETECTOR_MANIFEST:-${EVAL_DETECTOR_CACHE}/${EVAL_SCAN_NAME}/detector_scan_crops.jsonl}"
SCORER_SCRIPT="${SCORER_SCRIPT:-${PROJECT_ROOT}/qwen3vl_merge_and_score_fixed_5tasks.py}"

CURRICULUM_MODE="${CURRICULUM_MODE:-scheduled}"
TOTAL_STEPS="${TOTAL_STEPS:-1200}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-200}"
ROLLING_CHECKPOINT_DIR="${ROLLING_CHECKPOINT_DIR:-resume/latest}"
CHECKPOINT_SAVE_POLICY="${CHECKPOINT_SAVE_POLICY:-best_only}"
UI5_GPU0_WORKERS="${UI5_GPU0_WORKERS:-2}"
UI5_GPU1_WORKERS="${UI5_GPU1_WORKERS:-3}"
UI5_EVAL_HEARTBEAT_SECONDS="${UI5_EVAL_HEARTBEAT_SECONDS:-30}"
HARD_RATIOS="${HARD_RATIOS:-0.60,0.45,0.30}"
ANCHOR_RATIOS="${ANCHOR_RATIOS:-0.25,0.35,0.30}"
GLOBAL_REPLAY_RATIOS="${GLOBAL_REPLAY_RATIOS:-0.15,0.20,0.40}"
LLM_LRS="${LLM_LRS:-1e-6,7e-7,5e-7}"
SEED="${SEED:-42}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-7268}"
MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-7268}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-7268}"

if [[ "${ROLLING_CHECKPOINT_DIR}" = /* ]]; then
  ROLLING_CHECKPOINT_PATH="${ROLLING_CHECKPOINT_DIR}"
else
  ROLLING_CHECKPOINT_PATH="${OUTPUT_DIR}/${ROLLING_CHECKPOINT_DIR}"
fi

require_equal() {
  local name="$1" actual="$2" expected="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    locany_die 20 "Formal curriculum requires ${name}=${expected}; got ${actual}"
  fi
}

# This entrypoint intentionally has one formal profile. Failing here is safer
# than completing an expensive run with a subtly different curriculum.
require_equal CURRICULUM_MODE "${CURRICULUM_MODE}" scheduled
require_equal TOTAL_STEPS "${TOTAL_STEPS}" 1200
require_equal EVAL_INTERVAL_STEPS "${EVAL_INTERVAL_STEPS}" 200
require_equal ROLLING_CHECKPOINT_DIR "${ROLLING_CHECKPOINT_DIR}" resume/latest
require_equal CHECKPOINT_SAVE_POLICY "${CHECKPOINT_SAVE_POLICY}" best_only
require_equal UI5_GPU0_WORKERS "${UI5_GPU0_WORKERS}" 2
require_equal UI5_GPU1_WORKERS "${UI5_GPU1_WORKERS}" 3
require_equal HARD_RATIOS "${HARD_RATIOS}" 0.60,0.45,0.30
require_equal ANCHOR_RATIOS "${ANCHOR_RATIOS}" 0.25,0.35,0.30
require_equal GLOBAL_REPLAY_RATIOS "${GLOBAL_REPLAY_RATIOS}" 0.15,0.20,0.40
require_equal LLM_LRS "${LLM_LRS}" 1e-6,7e-7,5e-7
require_equal SEED "${SEED}" 42
require_equal CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}" 0,1
require_equal NNODES "${NNODES}" 1
require_equal NODE_RANK "${NODE_RANK}" 0
require_equal ATTN_IMPLEMENTATION "${ATTN_IMPLEMENTATION}" sdpa
require_equal MAX_SEQ_LENGTH "${MAX_SEQ_LENGTH}" 7268
require_equal MAX_NUM_TOKENS_PER_SAMPLE "${MAX_NUM_TOKENS_PER_SAMPLE}" 7268
require_equal MAX_NUM_TOKENS "${MAX_NUM_TOKENS}" 7268

[[ -x "${PYTHON_BIN}" ]] || locany_die 21 "Python is missing: ${PYTHON_BIN}"
[[ -d "${MODEL_PATH}" ]] || locany_die 22 "Original crop checkpoint is missing: ${MODEL_PATH}"
[[ -d "${PROCESSOR_PATH}" ]] || locany_die 23 "Processor/tokenizer snapshot is missing: ${PROCESSOR_PATH:-<unset>}"
[[ -d "${ROLLOUT_BUNDLE_ROOT}" ]] || locany_die 24 "Rollout bundle is missing: ${ROLLOUT_BUNDLE_ROOT}"
[[ -d "${FROZEN_SELECTION}" ]] || locany_die 25 "Frozen selection is missing: ${FROZEN_SELECTION}"
[[ -s "${ROLLOUT_DIFFICULTY}" ]] || locany_die 25 "Rollout difficulty file is missing: ${ROLLOUT_DIFFICULTY}"
[[ -d "${EVAL_INPUT_DIR}" ]] || locany_die 26 "UI5 full-evaluation input is missing: ${EVAL_INPUT_DIR}"
[[ -s "${EVAL_DETECTOR_MANIFEST}" ]] || locany_die 27 "GT-free detector crop manifest is missing: ${EVAL_DETECTOR_MANIFEST}"
[[ -s "${SCORER_SCRIPT}" ]] || locany_die 28 "Canonical evaluator is missing: ${SCORER_SCRIPT}"

# The formal hard-set cardinality is data, not a launcher constant.  The
# resolver verifies the immutable publication and its exact file inventory
# before returning the count recorded in summary.json.
if [[ -v EXPECTED_HARD_GROUPS && -n "${EXPECTED_HARD_GROUPS}" ]]; then
  echo "[WARN] inherited EXPECTED_HARD_GROUPS is ignored; deriving from ${FROZEN_SELECTION}/summary.json"
fi
readonly EXPECTED_HARD_GROUPS="$(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/ui5_frozen_selection.py" \
    --frozen-selection "${FROZEN_SELECTION}" \
    --field formal_crop_hard_groups
)"
[[ "${EXPECTED_HARD_GROUPS}" =~ ^[1-9][0-9]*$ ]] || \
  locany_die 29 "Frozen selection returned an invalid hard-group count: ${EXPECTED_HARD_GROUPS}"
echo "[FROZEN SELECTION] root=${FROZEN_SELECTION} formal_crop_hard_groups=${EXPECTED_HARD_GROUPS}"

mkdir -p \
  "${OUTPUT_DIR}/logs" \
  "${OUTPUT_DIR}/diagnostics" \
  "${OUTPUT_DIR}/evaluation" \
  "$(dirname "${ROLLING_CHECKPOINT_PATH}")"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PROJECT_ROOT WORKSPACE ENV_DIR MODEL_PATH BASE_MODEL PROCESSOR_PATH
export OUTPUT_DIR RUN_NAME META_PATH CUDA_VISIBLE_DEVICES
export NNODES NODE_RANK
export CURRICULUM_MODE TOTAL_STEPS EVAL_INTERVAL_STEPS ROLLING_CHECKPOINT_DIR
export CHECKPOINT_SAVE_POLICY UI5_GPU0_WORKERS UI5_GPU1_WORKERS
export UI5_EVAL_HEARTBEAT_SECONDS
export HARD_RATIOS ANCHOR_RATIOS GLOBAL_REPLAY_RATIOS LLM_LRS
export FROZEN_SELECTION ROLLOUT_DIFFICULTY EXPECTED_HARD_GROUPS SEED
export MAX_STEPS="${TOTAL_STEPS}"
export SAVE_STEPS="${EVAL_INTERVAL_STEPS}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export SAVE_EVERY_N_HOURS=0
export LOCANY_ENABLE_MILESTONE_COPIES=0
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export WARMUP_STEPS=0
export GPUS=2
export GPU_COUNT=2
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export ATTN_IMPLEMENTATION MAX_SEQ_LENGTH MAX_NUM_TOKENS_PER_SAMPLE MAX_NUM_TOKENS
export CHECK_MAGI_IMPORT=0
# Crop images are already materialized and digest-bound by the curriculum
# builder.  Keep the legacy on-the-fly crop-audit switch disabled (it would
# reject the intentional full-image replay dataset), but report the actual
# mixed training view correctly.
export UI5_USE_DETECTION_CROPS=0
export UI5_CROP_TRAIN_MODE=full_plus_crop
export UI5_UI_SAMPLING_MODE=fixed_ratio
export BALANCE_UI_DEFECTS=False
export OVERWRITE_OUTPUT_DIR=False
export EVAL_FAIL_POLICY=stop
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PIPELINE_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG="${OUTPUT_DIR}/logs/curriculum-${PIPELINE_STAMP}-${BASHPID}.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "===== UI5 crop rollout4 curriculum (H20x2) ====="
printf '%-30s %s\n' \
  "MODEL_PATH" "${MODEL_PATH}" \
  "PROCESSOR_PATH" "${PROCESSOR_PATH}" \
  "OUTPUT_DIR" "${OUTPUT_DIR}" \
  "ROLLING_CHECKPOINT" "${ROLLING_CHECKPOINT_PATH}" \
  "ROLLOUT_DIFFICULTY" "${ROLLOUT_DIFFICULTY}" \
  "FROZEN_SELECTION" "${FROZEN_SELECTION}" \
  "EXPECTED_HARD_GROUPS" "${EXPECTED_HARD_GROUPS}" \
  "CURRICULUM_PROGRESS_SECONDS" "${CURRICULUM_PROGRESS_INTERVAL_SECONDS}" \
  "CURRICULUM_PROGRESS_JSON" "${CURRICULUM_DATA_DIR}/progress/build_progress.json" \
  "CURRICULUM_REUSE_CROPS_FROM" "${CURRICULUM_REUSE_CROPS_FROM:-<none>}" \
  "ATTN_IMPLEMENTATION" "${ATTN_IMPLEMENTATION}" \
  "TOKEN_LIMITS" "${MAX_SEQ_LENGTH}/${MAX_NUM_TOKENS_PER_SAMPLE}/${MAX_NUM_TOKENS}" \
  "EVAL_DETECTOR_MANIFEST" "${EVAL_DETECTOR_MANIFEST}" \
  "CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES}" \
  "NNODES" "${NNODES}" \
  "NODE_RANK" "${NODE_RANK}"

# Reject copied manifests with stale source-machine aliases before any model
# load. The standalone JSONL contains all coordinates needed by inference;
# detector-cache sidecars are not required for this content/coverage check.
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/relocate_ui5_eval_detector_manifest.py" \
  --manifest "${EVAL_DETECTOR_MANIFEST}" \
  --input-dir "${EVAL_INPUT_DIR}"

recipe_command=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/build_ui5_curriculum_recipe.py"
  --rollout-difficulty "${ROLLOUT_DIFFICULTY}"
  --rollout-bundle-root "${ROLLOUT_BUNDLE_ROOT}"
  --output-dir "${CURRICULUM_DATA_DIR}"
  --expected-hard-groups "${EXPECTED_HARD_GROUPS}"
  --seed "${SEED}"
  --progress-interval-seconds "${CURRICULUM_PROGRESS_INTERVAL_SECONDS}"
)
if [[ -n "${CURRICULUM_SOURCE_RECIPE}" ]]; then
  [[ -s "${CURRICULUM_SOURCE_RECIPE}" ]] || \
    locany_die 29 "CURRICULUM_SOURCE_RECIPE is missing: ${CURRICULUM_SOURCE_RECIPE}"
  recipe_command+=(--base-recipe "${CURRICULUM_SOURCE_RECIPE}")
fi
if [[ -n "${CURRICULUM_REUSE_CROPS_FROM}" ]]; then
  recipe_command+=(--reuse-crops-from "${CURRICULUM_REUSE_CROPS_FROM}")
fi
"${recipe_command[@]}"
[[ -s "${META_PATH}" && -s "${HARD_GROUPS_JSONL}" ]] || \
  locany_die 30 "Curriculum recipe publication is incomplete"
"${PYTHON_BIN}" - "${CURRICULUM_DATA_DIR}/curriculum_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pools = state["pools"]
policy = state.get("training_view_policy", {})
if policy.get("tile_selection_uses_gt") is not False:
    raise SystemExit("curriculum crop geometry is not explicitly GT-free")
print(
    "[CURRICULUM TRAIN VIEWS] "
    f"hard_groups={state['hard_groups']} "
    f"hard_tile_records={pools['hard']['crop_training_records']} "
    f"hard_content_missing_global={pools['hard']['content_missing_global_records']} "
    f"anchor_groups={state['matched_anchor_groups']} "
    f"anchor_tile_records={pools['matched_anchor']['crop_training_records']} "
    f"anchor_content_missing_global={pools['matched_anchor']['content_missing_global_records']} "
    f"replay_crop_records={pools['global_replay']['crop_training_records']} "
    f"replay_full_image_records={pools['global_replay']['retention_full_image_records']} "
    f"crop_assets={len(state['crop_assets'])}",
    flush=True,
)
PY

"${PYTHON_BIN}" -c \
  'import openpyxl; assert tuple(map(int, openpyxl.__version__.split(".")[:2])) >= (3, 1)' \
  || locany_die 31 "openpyxl>=3.1 is required for the formal Excel artifact"

checkpoint_step() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "trainer_state.json"
value = json.loads(path.read_text(encoding="utf-8"))
step = value.get("global_step")
if isinstance(step, bool) or not isinstance(step, int) or step < 0:
    raise SystemExit(f"invalid global_step in {path}: {step!r}")
print(step)
PY
}

evaluation_recorded() {
  "${PYTHON_BIN}" - \
    "${OUTPUT_DIR}/checkpoints.json" \
    "${OUTPUT_DIR}/diagnostics/ui5_crop_rollout4_curriculum_evaluation.xlsx" \
    "$1" <<'PY'
from collections import Counter
import json
import sys
from pathlib import Path

from eaglevl.train.ui5_curriculum_artifacts import CHECKPOINT_COLUMNS, SHEET_ORDER
from openpyxl import load_workbook

path, workbook_path, step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if not path.is_file() or not workbook_path.is_file():
    raise SystemExit(1)
state = json.loads(path.read_text(encoding="utf-8"))
evaluations = state.get("evaluations", [])
expected_steps = [int(row["step"]) for row in evaluations]
if step not in expected_steps:
    raise SystemExit(1)
try:
    workbook = load_workbook(workbook_path, read_only=True, data_only=False)
    try:
        if tuple(workbook.sheetnames) != SHEET_ORDER:
            raise ValueError("workbook sheet set is stale")
        checkpoint_rows = list(workbook["checkpoints"].iter_rows(values_only=True))
        if not checkpoint_rows or tuple(checkpoint_rows[0]) != CHECKPOINT_COLUMNS:
            raise ValueError("checkpoints sheet schema is stale")
        workbook_steps = [int(row[0]) for row in checkpoint_rows[1:]]
        overall_steps = [
            int(row[0])
            for row in list(
                workbook["ui5_overall"].iter_rows(min_row=2, values_only=True)
            )
        ]
        task_steps = [
            int(row[0])
            for row in list(
                workbook["ui5_by_task"].iter_rows(min_row=2, values_only=True)
            )
        ]
        if workbook_steps != expected_steps or overall_steps != expected_steps:
            raise ValueError("workbook evaluation steps are stale")
        if Counter(task_steps) != Counter({value: 10 for value in expected_steps}):
            raise ValueError("ui5_by_task sheet is incomplete")
    finally:
        workbook.close()
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

evaluation_seconds() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads((Path(sys.argv[1]) / "evaluation_status.json").read_text(encoding="utf-8"))
if status.get("success") is not True or status.get("status") != "completed":
    raise SystemExit("evaluation status is not completed/success")
print(float(status["evaluation_seconds"]))
PY
}

evaluate_and_register() {
  local step="$1" candidate="$2" resume_from="$3"

  # Trainer checkpoints contain the full weights/config but do not
  # necessarily carry the Python files referenced by config.auto_map.  Patch
  # only inference metadata/code (never weights), then validate every trained
  # Relation/Gate/PBD group before any of the five workers sees the candidate.
  patch_command=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/patch_locany_checkpoint.py"
    --base-model "${PROCESSOR_PATH}"
    --checkpoint "${candidate}"
    --project-root "${PROJECT_ROOT}"
    --force
    --validate-relation-weights
  )
  echo "[EVAL CHECKPOINT PATCH] step=${step} candidate=${candidate}"
  "${patch_command[@]}"

  local step_tag eval_dir metrics_json overwrite=0
  printf -v step_tag '%06d' "${step}"
  eval_dir="${OUTPUT_DIR}/evaluation/step-${step_tag}"
  metrics_json="${eval_dir}/ui5_metrics.json"
  # A completed worker directory is not itself a durable evaluation.  If the
  # process died before checkpoints.json + Excel were committed, rerun the node
  # from the current candidate instead of trusting stale status/metrics from a
  # possibly different model or evaluation identity.
  if [[ -d "${eval_dir}" ]]; then
    overwrite=1
  fi
  eval_command=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_ui5_curriculum_evaluation.py"
    --checkpoint "${candidate}"
    --processor-path "${PROCESSOR_PATH}"
    --input-dir "${EVAL_INPUT_DIR}"
      --output-dir "${eval_dir}"
      --total-steps "${TOTAL_STEPS}"
    --hard-groups-jsonl "${HARD_GROUPS_JSONL}"
    --rollout-bundle-root "${ROLLOUT_BUNDLE_ROOT}"
    --curriculum-manifest "${CURRICULUM_DATA_DIR}/curriculum_manifest.json"
    --frozen-selection "${FROZEN_SELECTION}"
    --expected-hard-groups "${EXPECTED_HARD_GROUPS}"
    --step "${step}"
    --seed "${SEED}"
    --gpu-devices "${CUDA_VISIBLE_DEVICES}"
    --gpu0-workers "${UI5_GPU0_WORKERS}"
      --gpu1-workers "${UI5_GPU1_WORKERS}"
      --heartbeat-seconds "${UI5_EVAL_HEARTBEAT_SECONDS}"
    --python "${PYTHON_BIN}"
    --project-root "${PROJECT_ROOT}"
    --worker-script "${PROJECT_ROOT}/scripts/inference_ui_defect_locany.py"
    --scorer-script "${SCORER_SCRIPT}"
    --dtype bf16
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    --vision-attn-implementation flash_attention_2
    --generation-mode hybrid
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-4096}"
    --n-future-tokens "${EVAL_N_FUTURE_TOKENS:-6}"
    --temperature "${EVAL_TEMPERATURE:-0.7}"
    --top-p "${EVAL_TOP_P:-0.9}"
    --top-k "${EVAL_TOP_K:-0}"
    --repetition-penalty "${EVAL_REPETITION_PENALTY:-1.1}"
    --relation-gate-mode "${RELATION_GATE_MODE:-observe}"
    --inference-crop-mode detector_scan
    --detector-crop-manifest "${EVAL_DETECTOR_MANIFEST}"
    --tile-max-count "${EVAL_TILE_MAX_COUNT:-10}"
    --tile-target-long-side "${EVAL_TILE_TARGET_LONG_SIDE:-1600}"
    --tile-overlap-ratio "${EVAL_TILE_OVERLAP_RATIO:-0.10}"
    --tile-nms-iou "${EVAL_TILE_NMS_IOU:-0.50}"
    --evaluator-iou-threshold "${EVAL_IOU_THRESHOLD:-0.10}"
  )
  if evaluation_recorded "${step}"; then
    eval_command+=(--verify-existing-identity)
    echo "[EVAL REUSE CHECK] step=${step} verifying checkpoint/curriculum/selection/eval identity"
    "${eval_command[@]}"
    echo "[EVAL REUSE] step=${step} is durable and identity-exact"
    return 0
  fi
  if (( overwrite == 1 )); then
    eval_command+=(--overwrite)
  fi
  echo "[EVAL START] step=${step} candidate=${candidate}"
  "${eval_command[@]}"

  [[ -s "${metrics_json}" ]] || \
    locany_die 40 "UI5 metrics missing after step ${step}: ${metrics_json}"
  diagnostics_command=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_ui5_curriculum_diagnostics.py"
    --step "${step}"
    --evaluation-dir "${eval_dir}"
    --curriculum-dir "${CURRICULUM_DATA_DIR}"
    --total-steps "${TOTAL_STEPS}"
  )
  if (( step > 0 )); then
    diagnostics_command+=(--trainer-state "${candidate}/trainer_state.json")
  fi
  "${diagnostics_command[@]}"

  local duration diagnostics_dir
  duration="$(evaluation_seconds "${eval_dir}")"
  diagnostics_dir="${eval_dir}/diagnostics"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/update_ui5_curriculum_artifacts.py" \
    --run-dir "${OUTPUT_DIR}" \
    --step "${step}" \
    --metrics-json "${metrics_json}" \
    --candidate-checkpoint "${candidate}" \
    --resume-from "${resume_from}" \
    --evaluation-seconds "${duration}" \
    --expected-ranks 2 \
    --total-steps "${TOTAL_STEPS}" \
    --eval-interval-steps "${EVAL_INTERVAL_STEPS}" \
    --train-curve-json "${diagnostics_dir}/train_curve.json" \
    --hard-transition-json "${diagnostics_dir}/hard_transition.json" \
    --anchor-retention-json "${diagnostics_dir}/anchor_retention.json"
  echo "[EVAL DURABLE] step=${step} Excel/checkpoints.json updated"
}

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" recover \
  --destination "${ROLLING_CHECKPOINT_PATH}" \
  --expected-ranks 2 \
  --expected-step-delta "${EVAL_INTERVAL_STEPS}" \
  --strict

current_step=0
if [[ -d "${ROLLING_CHECKPOINT_PATH}" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" validate \
    --checkpoint "${ROLLING_CHECKPOINT_PATH}" --mode resume \
    --expected-ranks 2 --strict
  current_step="$(checkpoint_step "${ROLLING_CHECKPOINT_PATH}")"
fi

# Recover the one legal crash window: the segment checkpoint was completed but
# the process stopped before it was promoted to resume/latest. More than one
# transient directory is ambiguous and is never guessed through.
transient_candidates=()
for candidate in "${OUTPUT_DIR}"/checkpoint-*; do
  [[ -d "${candidate}" ]] || continue
  transient_candidates+=("${candidate}")
done
if (( ${#transient_candidates[@]} > 1 )); then
  locany_die 34 \
    "Multiple transient checkpoints require manual audit: ${transient_candidates[*]}"
elif (( ${#transient_candidates[@]} == 1 )); then
  transient_candidate="${transient_candidates[0]}"
  transient_step="$(checkpoint_step "${transient_candidate}")"
  expected_transient_step=$((current_step + EVAL_INTERVAL_STEPS))
  require_equal transient_global_step "${transient_step}" "${expected_transient_step}"
  echo "[RECOVER] promoting completed transient checkpoint step=${transient_step}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" promote \
    --source "${transient_candidate}" \
    --destination "${ROLLING_CHECKPOINT_PATH}" \
    --expected-ranks 2 \
    --strict \
    --move-source
  current_step="${transient_step}"
fi
if [[ ! "${current_step}" =~ ^[0-9]+$ ]] || \
    (( current_step < 0 || current_step > TOTAL_STEPS || current_step % EVAL_INTERVAL_STEPS != 0 )); then
  locany_die 32 "Invalid rolling checkpoint step: ${current_step}"
fi
durable_latest_step="$("${PYTHON_BIN}" - "${OUTPUT_DIR}/checkpoints.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    rows = json.loads(path.read_text(encoding="utf-8")).get("evaluations", [])
    print(max((int(row["step"]) for row in rows), default=0))
PY
)"
if (( durable_latest_step > current_step )); then
  locany_die 35 \
    "checkpoints.json is ahead of resume/latest: durable=${durable_latest_step}, rolling=${current_step}"
fi

# The original crop checkpoint is always the step-0 baseline, including when a
# previously interrupted run already has a nonzero resume/latest.
evaluate_and_register 0 "${MODEL_PATH}" ""

if (( current_step > 0 )); then
  previous_step=$((current_step - EVAL_INTERVAL_STEPS))
  recovered_resume_from="${ROLLING_CHECKPOINT_PATH}@step-${previous_step}"
  if (( previous_step == 0 )); then
    recovered_resume_from="${MODEL_PATH}"
  fi
  evaluate_and_register \
    "${current_step}" "${ROLLING_CHECKPOINT_PATH}" \
    "${recovered_resume_from}"
fi

while (( current_step < TOTAL_STEPS )); do
  next_step=$((current_step + EVAL_INTERVAL_STEPS))
  if (( next_step > TOTAL_STEPS )); then
    next_step="${TOTAL_STEPS}"
  fi
  export LOCANY_SEGMENT_MODE=1
  export LOCANY_STOP_AFTER_STEP="${next_step}"
  export CURRICULUM_START_STEP="${current_step}"
  segment_resume="${MODEL_PATH}"
  if (( current_step > 0 )); then
    export RESUME_FROM_CHECKPOINT="${ROLLING_CHECKPOINT_PATH}"
    segment_resume="${ROLLING_CHECKPOINT_PATH}@step-${current_step}"
  else
    unset RESUME_FROM_CHECKPOINT
  fi

  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/report_ui5_training_segment.py" \
    --event start \
    --start-step "${current_step}" \
    --target-step "${next_step}" \
    --total-steps "${TOTAL_STEPS}"
  bash "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"
  transient_checkpoint="${OUTPUT_DIR}/checkpoint-${next_step}"
  [[ -d "${transient_checkpoint}" ]] || \
    locany_die 33 "Training did not publish checkpoint-${next_step}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" promote \
    --source "${transient_checkpoint}" \
    --destination "${ROLLING_CHECKPOINT_PATH}" \
    --expected-ranks 2 \
    --strict \
    --move-source
  require_equal rolling_global_step \
    "$(checkpoint_step "${ROLLING_CHECKPOINT_PATH}")" "${next_step}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/report_ui5_training_segment.py" \
    --event complete \
    --start-step "${current_step}" \
    --target-step "${next_step}" \
    --total-steps "${TOTAL_STEPS}" \
    --checkpoint "${ROLLING_CHECKPOINT_PATH}"

  # torchrun has exited at this point, so no DDP process survives into the five
  # independent inference workers.
  evaluate_and_register \
    "${next_step}" "${ROLLING_CHECKPOINT_PATH}" "${segment_resume}"
  current_step="${next_step}"
done

unset LOCANY_SEGMENT_MODE LOCANY_STOP_AFTER_STEP RESUME_FROM_CHECKPOINT

# A training process reaching step 1200 is not the terminal condition: the
# final five-task evaluation, best-only copy, checkpoints.json and workbook
# must all be durable first.  Publish a distinct pipeline marker only after
# every required node passes the same JSON/Excel consistency guard used for
# crash recovery above.
for completed_step in 0 200 400 600 800 1000 1200; do
  evaluation_recorded "${completed_step}" || \
    locany_die 41 "Pipeline completion audit failed at evaluation step ${completed_step}"
done
"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${ROLLING_CHECKPOINT_PATH}" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from eaglevl.train.ui5_checkpoint_utils import validate_checkpoint

output_dir = Path(sys.argv[1]).resolve()
rolling = Path(sys.argv[2]).resolve()
state_path = output_dir / "checkpoints.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
expected_steps = [0, 200, 400, 600, 800, 1000, 1200]
actual_steps = [int(row["step"]) for row in state.get("evaluations", [])]
if actual_steps != expected_steps:
    raise SystemExit(
        f"cannot publish completion marker: evaluation steps={actual_steps}, "
        f"expected={expected_steps}"
    )
formal_root = (output_dir / "checkpoints").resolve()
expected_formal_paths = set()
for row in state["evaluations"]:
    step = int(row["step"])
    if not row.get("checkpoint_preserved"):
        if step > 0 and row.get("checkpoint_path"):
            raise SystemExit(
                "cannot publish completion marker: an unpreserved step has a "
                f"formal checkpoint path at step {step}"
            )
        continue
    checkpoint_path = Path(str(row.get("checkpoint_path") or "")).resolve()
    if (
        checkpoint_path.parent != formal_root
        or checkpoint_path.name != f"step-{step:06d}"
    ):
        raise SystemExit(
            "cannot publish completion marker: preserved checkpoint path is not "
            f"the canonical formal path for step {step}: {checkpoint_path}"
        )
    report = validate_checkpoint(
        checkpoint_path,
        mode="resume",
        expected_ranks=2,
        strict=True,
        require_completion_marker=True,
    )
    if not report.get("valid") or int(
        report.get("details", {}).get("global_step", -1)
    ) != step:
        raise SystemExit(
            "cannot publish completion marker: preserved checkpoint is not a "
            f"complete step-{step} resume point: {report.get('errors', [])}"
        )
    expected_formal_paths.add(checkpoint_path)

actual_formal_paths = set()
if formal_root.exists():
    for child in formal_root.iterdir():
        if not child.is_dir():
            raise SystemExit(
                "cannot publish completion marker: unexpected file in formal "
                f"checkpoint directory: {child}"
            )
        actual_formal_paths.add(child.resolve())
if actual_formal_paths != expected_formal_paths:
    raise SystemExit(
        "cannot publish completion marker: formal checkpoint directory does not "
        "exactly match improved evaluation steps; "
        f"expected={sorted(map(str, expected_formal_paths))}, "
        f"actual={sorted(map(str, actual_formal_paths))}"
    )

for best_name, metric in (
    ("best_image", "image_macro_f1"),
    ("best_bbox", "bbox_macro_f1"),
    ("best_joint", "joint_score"),
):
    selected = max(
        state["evaluations"],
        key=lambda row: (float(row[metric]), -int(row["step"])),
    )
    expected_best = {
        "step": int(selected["step"]),
        "score": float(selected[metric]),
        "checkpoint_preserved": bool(selected["checkpoint_preserved"]),
        "checkpoint_path": str(selected.get("checkpoint_path") or ""),
    }
    if state.get(best_name) != expected_best:
        raise SystemExit(
            f"cannot publish completion marker: {best_name} is stale or invalid"
        )
payload = {
    "schema_version": 1,
    "status": "completed",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "total_optimizer_steps": 1200,
    "evaluation_steps": expected_steps,
    "rolling_checkpoint": str(rolling),
    "checkpoints_json": str(state_path),
    "workbook": str(
        output_dir
        / "diagnostics"
        / "ui5_crop_rollout4_curriculum_evaluation.xlsx"
    ),
    "best_image": state.get("best_image"),
    "best_bbox": state.get("best_bbox"),
    "best_joint": state.get("best_joint"),
}
destination = output_dir / "pipeline_complete.json"
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{destination.name}.tmp-", dir=output_dir
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, destination)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
PY
echo "[CURRICULUM COMPLETE] step=${current_step} output=${OUTPUT_DIR}"
echo "[CURRICULUM COMPLETE] rolling resume=${ROLLING_CHECKPOINT_PATH}"
echo "[CURRICULUM COMPLETE] formal best-only checkpoints=${OUTPUT_DIR}/checkpoints"
echo "[CURRICULUM COMPLETE] marker=${OUTPUT_DIR}/pipeline_complete.json"
