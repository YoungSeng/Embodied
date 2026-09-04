#!/usr/bin/env bash
set -Eeuo pipefail

# CPU-only preflight for the formal H20x2 curriculum launcher.  This script
# never starts torchrun, inference workers, or model loading.  It is safe to
# run on an allocated H20 node while the GPUs remain available to other jobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=shell/bash_error_report.sh
source "${SCRIPT_DIR}/bash_error_report.sh"

WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}"
ENV_DIR="${ENV_DIR:-${WORKSPACE}/conda_envs/LocateAnything}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
RUN_NAME="${RUN_NAME:-locany-ui5-crop-rollout4-curriculum-h20x2-20260904}"
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
ROLLOUT_DIFFICULTY="${ROLLOUT_DIFFICULTY:-${ROLLOUT_ROOT}/selection/complete8.jsonl}"
CURRICULUM_SOURCE_RECIPE="${CURRICULUM_SOURCE_RECIPE:-}"
EVAL_INPUT_DIR="${EVAL_INPUT_DIR:-${WORKSPACE}/data}"
EVAL_SCAN_NAME="${EVAL_SCAN_NAME:-horizontal_scan_v5_raw_detector_edge_aligned}"
EVAL_DETECTOR_CACHE="${EVAL_DETECTOR_CACHE:-${WORKSPACE}/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5}"
EVAL_DETECTOR_MANIFEST="${EVAL_DETECTOR_MANIFEST:-${EVAL_DETECTOR_CACHE}/${EVAL_SCAN_NAME}/detector_scan_crops.jsonl}"
SCORER_SCRIPT="${SCORER_SCRIPT:-${PROJECT_ROOT}/qwen3vl_merge_and_score_fixed_5tasks.py}"
EXPECTED_HARD_GROUPS="${EXPECTED_HARD_GROUPS:-72}"
SEED="${SEED:-42}"
ROLLING_CHECKPOINT_DIR="${ROLLING_CHECKPOINT_DIR:-resume/latest}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

[[ "${NNODES}" == 1 ]] || \
  locany_die 2 "Formal H20x2 launch requires NNODES=1; got ${NNODES}"
[[ "${NODE_RANK}" == 0 ]] || \
  locany_die 2 "Formal H20x2 launch requires NODE_RANK=0; got ${NODE_RANK}"

if [[ "${ROLLING_CHECKPOINT_DIR}" = /* ]]; then
  ROLLING_CHECKPOINT_PATH="${ROLLING_CHECKPOINT_DIR}"
else
  ROLLING_CHECKPOINT_PATH="${OUTPUT_DIR}/${ROLLING_CHECKPOINT_DIR}"
fi

PREFLIGHT_MODE="${PREFLIGHT_MODE:-fast}"
KEEP_PREFLIGHT_WORKDIR="${KEEP_PREFLIGHT_WORKDIR:-0}"
while (( $# > 0 )); do
  case "$1" in
    --fast)
      PREFLIGHT_MODE=fast
      ;;
    --full)
      PREFLIGHT_MODE=full
      ;;
    --keep-work-dir)
      KEEP_PREFLIGHT_WORKDIR=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: preflight_locany_ui5_crop_rollout4_curriculum_h20x2.sh [--fast|--full] [--keep-work-dir]

  --fast           Run static checks, input/recipe/checkpoint checks, and the
                   lightweight curriculum suite (default).
  --full           Also run the related UI5 regression suite.
  --keep-work-dir  Keep the generated dry-run curriculum for inspection.

All formal launcher paths may be overridden with the same environment
variables used by run_locany_ui5_crop_rollout4_curriculum_h20x2.sh.
EOF
      exit 0
      ;;
    *)
      locany_die 2 "Unknown preflight argument: $1"
      ;;
  esac
  shift
done
case "${PREFLIGHT_MODE,,}" in
  fast|full)
    PREFLIGHT_MODE="${PREFLIGHT_MODE,,}"
    ;;
  *)
    locany_die 2 "PREFLIGHT_MODE must be fast or full; got ${PREFLIGHT_MODE}"
    ;;
esac
if [[ "${KEEP_PREFLIGHT_WORKDIR}" != 0 && "${KEEP_PREFLIGHT_WORKDIR}" != 1 ]]; then
  locany_die 2 "KEEP_PREFLIGHT_WORKDIR must be 0 or 1"
fi

# This is intentionally unconditional.  In particular, an inherited Slurm or
# interactive-shell CUDA_VISIBLE_DEVICES value cannot leak into any check.
export CUDA_VISIBLE_DEVICES=""
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PROJECT_ROOT WORKSPACE ENV_DIR PYTHON_BIN MODEL_PATH BASE_MODEL PROCESSOR_PATH
export OUTPUT_DIR ROLLOUT_BUNDLE_ROOT ROLLOUT_DIFFICULTY CURRICULUM_SOURCE_RECIPE
export EVAL_INPUT_DIR EVAL_DETECTOR_MANIFEST SCORER_SCRIPT EXPECTED_HARD_GROUPS SEED
export ROLLING_CHECKPOINT_PATH
export NNODES NODE_RANK

PREFLIGHT_TMP_ROOT="${PREFLIGHT_TMP_ROOT:-${TMPDIR:-/tmp}}"
[[ -d "${PREFLIGHT_TMP_ROOT}" ]] || locany_die 3 "Temporary root is missing: ${PREFLIGHT_TMP_ROOT}"
PREFLIGHT_WORK_DIR="$(mktemp -d "${PREFLIGHT_TMP_ROOT%/}/ui5-curriculum-preflight.XXXXXX")"
DRY_RUN_CURRICULUM_DIR="${PREFLIGHT_WORK_DIR}/curriculum_data"
PREFLIGHT_STARTED_SECONDS=${SECONDS}
PREFLIGHT_PASSED=0

preflight_finish() {
  local exit_code=$?
  local elapsed=$((SECONDS - PREFLIGHT_STARTED_SECONDS))
  trap - EXIT
  set +e
  if [[ "${KEEP_PREFLIGHT_WORKDIR}" == 1 ]]; then
    echo "[PREFLIGHT INFO] kept_work_dir=${PREFLIGHT_WORK_DIR}"
  elif [[ "${PREFLIGHT_WORK_DIR}" == "${PREFLIGHT_TMP_ROOT%/}"/ui5-curriculum-preflight.* ]]; then
    rm -rf -- "${PREFLIGHT_WORK_DIR}"
  else
    echo "[PREFLIGHT WARN] refusing to remove unexpected work dir: ${PREFLIGHT_WORK_DIR}" >&2
  fi
  if (( exit_code == 0 )); then
    echo "[PREFLIGHT PASS] checks=${PREFLIGHT_PASSED} mode=${PREFLIGHT_MODE} elapsed=${elapsed}s gpu_count=0"
  else
    echo "[PREFLIGHT FAIL] checks_passed=${PREFLIGHT_PASSED} mode=${PREFLIGHT_MODE} elapsed=${elapsed}s exit=${exit_code}" >&2
  fi
  exit "${exit_code}"
}
trap preflight_finish EXIT

run_check() {
  local label="$1"
  shift
  local started=${SECONDS}
  echo
  echo "[CHECK START] ${label}"
  if "$@"; then
    local elapsed=$((SECONDS - started))
    PREFLIGHT_PASSED=$((PREFLIGHT_PASSED + 1))
    echo "[CHECK PASS] ${label} elapsed=${elapsed}s"
  else
    local exit_code=$?
    local elapsed=$((SECONDS - started))
    echo "[CHECK FAIL] ${label} elapsed=${elapsed}s exit=${exit_code}" >&2
    return "${exit_code}"
  fi
}

check_python_dependencies() {
  "${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.metadata
import importlib.util
import os
import sys
from packaging.version import Version

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES is not empty")

versions = {"python": sys.version.split()[0]}
for name in ("numpy", "scipy", "openpyxl", "PIL", "torch", "transformers", "accelerate"):
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, "__version__", "present"))
openpyxl_version = tuple(
    int(part) for part in versions["openpyxl"].split(".")[:2]
)
if openpyxl_version < (3, 1):
    raise SystemExit(
        f"openpyxl>=3.1 is required, found {versions['openpyxl']}"
    )
for name in ("deepspeed", "magi_attention", "flash_attn"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"required module is missing: {name}")
    versions[name] = "present"
versions["deepspeed"] = importlib.metadata.version("deepspeed")
if Version(versions["transformers"]) != Version("4.57.1"):
    raise SystemExit(f"transformers must be 4.57.1; found {versions['transformers']}")
if Version(versions["accelerate"]) != Version("1.5.2"):
    raise SystemExit(f"accelerate must be 1.5.2; found {versions['accelerate']}")
if Version(versions["deepspeed"]) != Version("0.15.4"):
    raise SystemExit(f"deepspeed must be 0.15.4; found {versions['deepspeed']}")
if Version(versions["openpyxl"]) < Version("3.1"):
    raise SystemExit(f"openpyxl must be >=3.1; found {versions['openpyxl']}")

import torch
if torch.cuda.is_available() or torch.cuda.device_count() != 0:
    raise SystemExit(
        "GPU isolation failed: torch sees CUDA devices despite CUDA_VISIBLE_DEVICES=''"
    )
print("dependency_versions=" + ", ".join(f"{key}:{value}" for key, value in versions.items()))
print("cuda_visible_devices=<empty> torch_cuda_device_count=0")
PY
}

check_formal_inputs() {
  "${PYTHON_BIN}" - \
    "${MODEL_PATH}" "${PROCESSOR_PATH}" "${ROLLOUT_BUNDLE_ROOT}" \
    "${ROLLOUT_DIFFICULTY}" "${EVAL_INPUT_DIR}" \
    "${EVAL_DETECTOR_MANIFEST}" "${SCORER_SCRIPT}" \
    "${CURRICULUM_SOURCE_RECIPE}" <<'PY'
import json
import sys
from pathlib import Path

(
    model,
    processor,
    bundle,
    difficulty,
    eval_input,
    detector_manifest,
    scorer,
    source_recipe,
) = (Path(value).expanduser() if value else None for value in sys.argv[1:])

required_dirs = {
    "MODEL_PATH": model,
    "PROCESSOR_PATH": processor,
    "ROLLOUT_BUNDLE_ROOT": bundle,
    "EVAL_INPUT_DIR": eval_input,
}
for label, path in required_dirs.items():
    if path is None or not path.is_dir():
        raise SystemExit(f"{label} directory is missing: {path}")

required_files = {
    "ROLLOUT_DIFFICULTY": difficulty,
    "EVAL_DETECTOR_MANIFEST": detector_manifest,
    "SCORER_SCRIPT": scorer,
    "bundle_manifest": bundle / "bundle_manifest.json",
    "bundle_crop_samples": bundle / "manifest" / "crop_samples.jsonl",
    "bundle_base_scan_plans": bundle / "base_scan_plans.json",
}
if source_recipe is not None:
    required_files["CURRICULUM_SOURCE_RECIPE"] = source_recipe
for label, path in required_files.items():
    if path is None or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"{label} file is missing or empty: {path}")

task_files = (
    "test_ui_occlusion_wcnt_no_figma.jsonl",
    "test_ui_cropping_wcnt_no_figma.jsonl",
    "test_ui_text_overflow_wcnt_no_figma.jsonl",
    "test_ui_text_ellipsis_wcnt_no_figma.jsonl",
    "test_ui_content_missing_wcnt_no_figma.jsonl",
)
for filename in task_files:
    path = eval_input / filename
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"UI5 evaluation input is missing or empty: {path}")

processor_assets = (
    processor / "preprocessor_config.json",
    processor / "processor_config.json",
)
if not any(path.is_file() and path.stat().st_size > 0 for path in processor_assets):
    raise SystemExit(f"processor snapshot has no processor config: {processor}")
if not (processor / "tokenizer_config.json").is_file():
    raise SystemExit(f"processor snapshot has no tokenizer_config.json: {processor}")

for label, path in (
    ("difficulty", difficulty),
    ("detector manifest", detector_manifest),
):
    with path.open("r", encoding="utf-8") as handle:
        first = next((line for line in handle if line.strip()), "")
    if not first:
        raise SystemExit(f"{label} has no JSONL records: {path}")
    if not isinstance(json.loads(first), dict):
        raise SystemExit(f"{label} first JSONL record is not an object: {path}")

bundle_state = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
if not isinstance(bundle_state, dict) or bundle_state.get("complete") is not True:
    raise SystemExit("rollout bundle manifest is not complete")
declared_bundle_files = bundle_state.get("files")
for relative in ("manifest/crop_samples.jsonl", "base_scan_plans.json"):
    if not isinstance(declared_bundle_files, dict) or relative not in declared_bundle_files:
        raise SystemExit(f"rollout bundle manifest does not declare {relative}")
print(f"formal_inputs=ok eval_task_files={len(task_files)}")
for label, path in required_dirs.items():
    print(f"{label}={path.resolve()}")
for label, path in required_files.items():
    print(f"{label}={path.resolve()}")
PY
}

check_python_syntax() {
  "${PYTHON_BIN}" - "${PROJECT_ROOT}" <<'PY'
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1])
relative_paths = (
    "eaglevl/train/locany_finetune_magi_stream.py",
    "eaglevl/train/ui5_checkpoint_utils.py",
    "eaglevl/train/ui5_curriculum.py",
    "eaglevl/train/ui5_curriculum_artifacts.py",
    "scripts/build_ui5_curriculum_recipe.py",
    "scripts/inference_ui_defect_locany.py",
    "scripts/locany_ui5_checkpoint.py",
    "scripts/merge_ui5_rollout_selections.py",
    "scripts/report_ui5_training_segment.py",
    "scripts/run_ui5_curriculum_evaluation.py",
    "scripts/summarize_ui5_curriculum_diagnostics.py",
    "scripts/update_ui5_curriculum_artifacts.py",
    "tests/test_ui5_curriculum.py",
    "tests/test_ui5_curriculum_artifacts.py",
    "tests/test_ui5_curriculum_diagnostics.py",
    "tests/test_ui5_curriculum_evaluation.py",
    "tests/test_ui5_curriculum_pipeline.py",
    "tests/test_ui5_curriculum_recipe.py",
    "tests/test_ui5_curriculum_status.py",
    "tests/test_ui5_rollout_selection_merge.py",
)
for relative in relative_paths:
    path = root / relative
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"python_ast_files={len(relative_paths)}")
PY
}

check_bash_syntax() {
  local path
  local paths=(
    "${PROJECT_ROOT}/shell/run_locany_ui5_crop_rollout4_curriculum_h20x2.sh"
    "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"
    "${PROJECT_ROOT}/shell/preflight_locany_ui5_crop_rollout4_curriculum_h20x2.sh"
  )
  for path in "${paths[@]}"; do
    bash -n "${path}"
  done
  echo "bash_syntax_files=${#paths[@]}"
}

build_recipe_dry_run() {
  local command=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ui5_curriculum_recipe.py"
    --rollout-difficulty "${ROLLOUT_DIFFICULTY}"
    --rollout-bundle-root "${ROLLOUT_BUNDLE_ROOT}"
    --output-dir "${DRY_RUN_CURRICULUM_DIR}"
    --expected-hard-groups "${EXPECTED_HARD_GROUPS}"
    --seed "${SEED}"
  )
  if [[ -n "${CURRICULUM_SOURCE_RECIPE}" ]]; then
    command+=(--base-recipe "${CURRICULUM_SOURCE_RECIPE}")
  fi
  "${command[@]}"

  "${PYTHON_BIN}" - "${DRY_RUN_CURRICULUM_DIR}" "${EXPECTED_HARD_GROUPS}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_hard = int(sys.argv[2])
manifest = json.loads((root / "curriculum_manifest.json").read_text(encoding="utf-8"))
success = json.loads((root / "_SUCCESS.json").read_text(encoding="utf-8"))
if success.get("complete") is not True:
    raise SystemExit("dry-run curriculum success marker is incomplete")
if success.get("identity_digest") != manifest.get("identity_digest"):
    raise SystemExit("dry-run curriculum identity mismatch")
if manifest.get("hard_groups") != expected_hard:
    raise SystemExit(
        f"dry-run hard-group mismatch: expected={expected_hard}, "
        f"observed={manifest.get('hard_groups')}"
    )
if manifest.get("matched_anchor_groups") != expected_hard:
    raise SystemExit("dry-run matched-anchor count does not equal hard-group count")
if set(manifest.get("pools", {})) != {"hard", "matched_anchor", "global_replay"}:
    raise SystemExit("dry-run curriculum does not contain exactly three pools")
expected_policy = {
    "hard": "all_gt_free_detector_scan_base_tiles",
    "matched_anchor": "all_gt_free_detector_scan_base_tiles",
    "content_missing": "full_image_global_view",
    "global_replay": "full_image_retention",
    "tile_selection_uses_gt": False,
    "partial_gt_allowed": False,
}
if manifest.get("training_view_policy") != expected_policy:
    raise SystemExit(
        f"dry-run curriculum training-view policy differs: "
        f"{manifest.get('training_view_policy')!r}"
    )
for filename in success.get("files", {}):
    path = root / filename
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"dry-run artifact is missing or empty: {path}")

records = {}
for pool in ("hard", "matched_anchor", "global_replay"):
    records[pool] = [
        json.loads(line)
        for line in (root / f"{pool}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
for pool in ("hard", "matched_anchor"):
    by_sample = {}
    for row in records[pool]:
        by_sample.setdefault(str(row["_ui5_sample_id"]), []).append(row)
        task = str(row["_ui5_task"]).removeprefix("ui_")
        kind = row.get("_ui5_record_kind")
        if task == "content_missing":
            if kind != "global_view" or row.get("_ui5_crop_source") != "content_missing_global":
                raise SystemExit(f"{pool} content_missing is not a global view")
        elif (
            kind != "crop"
            or row.get("_ui5_crop_source") != "gt_free_detector_scan_base_tile"
            or row.get("_ui5_gt_used_for_geometry") is not False
            or row.get("_ui5_partial_gt_indices") != []
        ):
            raise SystemExit(f"{pool} region record is not a verified GT-free crop")
    for sample_id, sample_rows in by_sample.items():
        task = str(sample_rows[0]["_ui5_task"]).removeprefix("ui_")
        expected_records = 1 if task == "content_missing" else int(
            sample_rows[0]["_ui5_base_tile_count"]
        )
        if len(sample_rows) != expected_records:
            raise SystemExit(
                f"{pool} sample {sample_id} omitted a base tile: "
                f"expected={expected_records}, observed={len(sample_rows)}"
            )
if any(
    row.get("_ui5_record_kind") != "full_image"
    or row.get("_ui5_retention_view") is not True
    or row.get("_ui5_crop_source") != "global_replay_retention"
    for row in records["global_replay"]
):
    raise SystemExit("global replay is not an all-full-image retention pool")

asset_rows = [
    json.loads(line)
    for line in (root / "crop_assets.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
crop_records = [
    row
    for pool in ("hard", "matched_anchor")
    for row in records[pool]
    if row.get("_ui5_record_kind") == "crop"
]
if len(asset_rows) != len(crop_records) or len(asset_rows) != len(manifest["crop_assets"]):
    raise SystemExit("crop asset/record counts differ")
asset_paths = {str(row["relative_path"]) for row in asset_rows}
record_paths = {
    str(Path(row["image"]).resolve().relative_to(root.resolve())).replace("\\", "/")
    for row in crop_records
}
if asset_paths != record_paths:
    raise SystemExit("crop records do not reference exactly the asset inventory")
for relative, metadata in success["files"].items():
    path = root / relative
    if not isinstance(metadata, dict):
        raise SystemExit(f"success inventory lacks hash/bytes metadata: {relative}")
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != metadata.get("bytes") or digest != metadata.get("sha256"):
        raise SystemExit(f"success inventory differs: {relative}")
print(
    "recipe_dry_run=ok "
    f"hard_groups={manifest['hard_groups']} "
    f"matched_anchor_groups={manifest['matched_anchor_groups']} "
    f"base_sample_groups={manifest['base_sample_groups']} "
    f"hard_tile_records={manifest['pools']['hard']['crop_training_records']} "
    f"anchor_tile_records={manifest['pools']['matched_anchor']['crop_training_records']} "
    f"crop_assets={len(asset_rows)}"
)
for pool, state in sorted(manifest["pools"].items()):
    print(
        f"pool={pool} training_records={state['training_records']} "
        f"sample_groups={state['sample_groups']}"
    )
PY
}

check_eval_checkpoint_contract() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" validate \
    --checkpoint "${MODEL_PATH}" --mode eval
}

check_resume_checkpoint_contract() {
  if [[ -d "${ROLLING_CHECKPOINT_PATH}" ]]; then
    echo "resume_checkpoint=${ROLLING_CHECKPOINT_PATH}"
    # Strict validation deserializes only RNG/dataloader state with
    # map_location='cpu'; model/optimizer weights are never deserialized.
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" validate \
      --checkpoint "${ROLLING_CHECKPOINT_PATH}" --mode resume \
      --expected-ranks 2 --strict
  else
    echo "resume_checkpoint=<absent:fresh-start>"
    echo "resume_contract=covered_by_lightweight_tests"
  fi
}

run_lightweight_tests() {
  "${PYTHON_BIN}" -B -m unittest \
    tests.test_ui5_curriculum \
    tests.test_ui5_curriculum_artifacts \
    tests.test_ui5_curriculum_evaluation \
    tests.test_ui5_curriculum_recipe \
    tests.test_ui5_curriculum_diagnostics \
    tests.test_ui5_curriculum_pipeline \
    tests.test_ui5_curriculum_status \
    tests.test_ui5_rollout_selection_merge
}

run_related_regressions() {
  "${PYTHON_BIN}" -B -m unittest \
    tests.test_ui5_excel_logger \
    tests.test_ui5_pipeline \
    tests.test_ui5_train_rollouts \
    tests.test_ui5_eval_detector_scan \
    tests.test_ui5_eval_detector_scan_v5 \
    tests.test_ui5_croponly_training \
    tests.test_ui5_sampling_coverage \
    tests.test_ui5_tiled_evaluation
}

echo "===== UI5 curriculum CPU-only preflight ====="
printf '%-30s %s\n' \
  "PREFLIGHT_MODE" "${PREFLIGHT_MODE}" \
  "PROJECT_ROOT" "${PROJECT_ROOT}" \
  "PYTHON_BIN" "${PYTHON_BIN}" \
  "MODEL_PATH" "${MODEL_PATH}" \
  "OUTPUT_DIR" "${OUTPUT_DIR}" \
  "ROLLING_CHECKPOINT" "${ROLLING_CHECKPOINT_PATH}" \
  "NNODES" "${NNODES}" \
  "NODE_RANK" "${NODE_RANK}" \
  "CUDA_VISIBLE_DEVICES" "<empty>" \
  "DRY_RUN_OUTPUT" "${DRY_RUN_CURRICULUM_DIR}"

run_check "Python executable" test -x "${PYTHON_BIN}"
run_check "Python dependencies and zero visible GPUs" check_python_dependencies
run_check "formal input paths and lightweight JSON readability" check_formal_inputs
run_check "Bash static syntax" check_bash_syntax
run_check "Python static syntax (AST, no imports)" check_python_syntax
run_check "rollout bundle/difficulty integrity and curriculum recipe dry-run" build_recipe_dry_run
run_check "source crop checkpoint evaluation contract (metadata only)" check_eval_checkpoint_contract
run_check "rolling checkpoint resume contract (CPU state only)" check_resume_checkpoint_contract
run_check "lightweight UI5 curriculum suite" run_lightweight_tests
if [[ "${PREFLIGHT_MODE}" == full ]]; then
  run_check "related UI5 regression suite" run_related_regressions
fi
