from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "shell" / "run_locany_ui5_crop_rollout4_curriculum_h20x2.sh"
PREFLIGHT = ROOT / "shell" / "preflight_locany_ui5_crop_rollout4_curriculum_h20x2.sh"
TRAINER = ROOT / "eaglevl" / "train" / "locany_finetune_magi_stream.py"


class CurriculumPipelineContractTests(unittest.TestCase):
    def test_formal_launcher_contains_the_complete_200_step_loop(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        for contract in (
            'SUGGESTED_RUN_NAME="locany-ui5-crop-rollout4-curriculum-hard114-h20x2-sdpa7268-v1"',
            'RUN_NAME="${RUN_NAME:-}"',
            'TOTAL_STEPS="${TOTAL_STEPS:-1200}"',
            'EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-200}"',
            'ROLLING_CHECKPOINT_DIR="${ROLLING_CHECKPOINT_DIR:-resume/latest}"',
            'CHECKPOINT_SAVE_POLICY="${CHECKPOINT_SAVE_POLICY:-best_only}"',
            'UI5_GPU0_WORKERS="${UI5_GPU0_WORKERS:-2}"',
            'UI5_GPU1_WORKERS="${UI5_GPU1_WORKERS:-3}"',
            'UI5_EVAL_HEARTBEAT_SECONDS="${UI5_EVAL_HEARTBEAT_SECONDS:-30}"',
            'HARD_RATIOS="${HARD_RATIOS:-0.60,0.45,0.30}"',
            'ANCHOR_RATIOS="${ANCHOR_RATIOS:-0.25,0.35,0.30}"',
            'GLOBAL_REPLAY_RATIOS="${GLOBAL_REPLAY_RATIOS:-0.15,0.20,0.40}"',
            'LLM_LRS="${LLM_LRS:-1e-6,7e-7,5e-7}"',
            'SEED="${SEED:-42}"',
            'FROZEN_SELECTION="${FROZEN_SELECTION:-}"',
            'ROLLOUT_DIFFICULTY="${FROZEN_SELECTION}/complete8.jsonl"',
            'ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"',
            'MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-7268}"',
            'MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-7268}"',
            'MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-7268}"',
            'NNODES="${NNODES:-1}"',
            'NODE_RANK="${NODE_RANK:-0}"',
            'require_equal NNODES "${NNODES}" 1',
            'require_equal NODE_RANK "${NODE_RANK}" 0',
            'export NNODES NODE_RANK',
        ):
            self.assertIn(contract, source)
        self.assertIn("evaluate_and_register 0", source)
        self.assertIn("RUN_NAME must be explicit", source)
        self.assertNotIn(
            "locany-ui5-crop-rollout4-curriculum-h20x2-20260904", source
        )
        self.assertIn("scripts/ui5_frozen_selection.py", source)
        self.assertIn("--field formal_crop_hard_groups", source)
        self.assertNotIn("EXPECTED_HARD_GROUPS:-", source)
        self.assertIn("while (( current_step < TOTAL_STEPS )); do", source)
        self.assertIn('export LOCANY_STOP_AFTER_STEP="${next_step}"', source)
        self.assertIn('bash "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"', source)
        self.assertIn("run_ui5_curriculum_evaluation.py", source)
        self.assertIn("update_ui5_curriculum_artifacts.py", source)
        self.assertIn("report_ui5_training_segment.py", source)
        self.assertIn('--heartbeat-seconds "${UI5_EVAL_HEARTBEAT_SECONDS}"', source)
        self.assertIn("scripts/patch_locany_checkpoint.py", source)
        self.assertIn("--validate-relation-weights", source)
        self.assertIn('--curriculum-manifest "${CURRICULUM_DATA_DIR}/curriculum_manifest.json"', source)
        self.assertIn('--frozen-selection "${FROZEN_SELECTION}"', source)
        self.assertIn("--verify-existing-identity", source)
        self.assertIn("--move-source", source)
        self.assertIn("--strict", source)
        self.assertIn("export SAVE_EVERY_N_HOURS=0", source)
        self.assertIn("export LOCANY_ENABLE_MILESTONE_COPIES=0", source)

        patch = source.index("scripts/patch_locany_checkpoint.py")
        evaluate = source.index("scripts/run_ui5_curriculum_evaluation.py", patch)
        self.assertLess(patch, evaluate)

        artifact_call = source.index("scripts/update_ui5_curriculum_artifacts.py")
        artifact_end = source.index("[EVAL DURABLE]", artifact_call)
        artifact_source = source[artifact_call:artifact_end]
        self.assertIn("--expected-ranks 2", artifact_source)
        for argument in (
            '--train-curve-json "${diagnostics_dir}/train_curve.json"',
            '--hard-transition-json "${diagnostics_dir}/hard_transition.json"',
            '--anchor-retention-json "${diagnostics_dir}/anchor_retention.json"',
        ):
            self.assertIn(argument, artifact_source)

    def test_rank_zero_training_progress_is_structured_and_uses_actual_pool_draws(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn("def _reduce_curriculum_pool_draw_counts", source)
        self.assertIn("curriculum_pool_draw_counts(", source)
        self.assertIn("dist.all_reduce(count_tensor", source)
        for field in (
            "curriculum_hard_samples",
            "curriculum_anchor_samples",
            "curriculum_global_replay_samples",
        ):
            self.assertIn(field, source)
        self.assertIn('step % 100 == 0', source)
        self.assertIn('"event": "train_progress"', source)
        self.assertIn('"[CURRICULUM STATUS] %s"', source)
        self.assertIn('if self.is_world_process_zero():', source)

        preflight = PREFLIGHT.read_text(encoding="utf-8")
        self.assertIn(
            'SUGGESTED_RUN_NAME="locany-ui5-crop-rollout4-curriculum-hard114-h20x2-sdpa7268-v1"',
            preflight,
        )
        self.assertIn('RUN_NAME="${RUN_NAME:-}"', preflight)
        self.assertIn("RUN_NAME must be explicit", preflight)
        self.assertNotIn(
            "locany-ui5-crop-rollout4-curriculum-h20x2-20260904", preflight
        )
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', preflight)
        self.assertIn('NNODES="${NNODES:-1}"', preflight)
        self.assertIn('NODE_RANK="${NODE_RANK:-0}"', preflight)
        self.assertIn('requires NNODES=1', preflight)
        self.assertIn('requires NODE_RANK=0', preflight)
        self.assertIn('export NNODES NODE_RANK', preflight)
        self.assertIn("scripts/ui5_frozen_selection.py", preflight)
        self.assertIn("immutable frozen selection and derived hard-group count", preflight)
        self.assertNotIn("magi_attention\", \"flash_attn", preflight)
        self.assertIn(
            "all_gt_free_detector_scan_base_tiles_except_content_missing_full_image",
            preflight,
        )
        self.assertIn(
            'for pool in ("hard", "matched_anchor", "global_replay"):',
            preflight,
        )
        self.assertNotIn("global replay is not an all-full-image retention pool", preflight)
        self.assertIn('PREFLIGHT_MODE="${PREFLIGHT_MODE:-fast}"', preflight)
        self.assertIn("--full", preflight)
        self.assertIn("build_ui5_curriculum_recipe.py", preflight)
        self.assertIn("merge_ui5_rollout_selections.py", preflight)
        self.assertIn("tests.test_ui5_rollout_selection_merge", preflight)
        self.assertIn('--checkpoint "${MODEL_PATH}" --mode eval', preflight)
        self.assertIn('--checkpoint "${ROLLING_CHECKPOINT_PATH}" --mode resume', preflight)
        self.assertIn("--expected-ranks 2 --strict", preflight)
        self.assertIn("lightweight UI5 curriculum suite", preflight)
        self.assertIn("related UI5 regression suite", preflight)
        self.assertIn('bash -n "${path}"', preflight)
        self.assertIn("[PREFLIGHT PASS]", preflight)
        self.assertIn("[PREFLIGHT FAIL]", preflight)
        self.assertNotIn("torchrun \\", preflight)
        self.assertNotIn(
            '"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_ui5_curriculum_evaluation.py"',
            preflight,
        )
        self.assertNotIn('bash "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"', preflight)

    def test_scheduled_stream_samples_groups_and_persists_exact_v8_state(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        for contract in (
            "curriculum_group_sampling=scheduled_curriculum",
            "CurriculumGroupCycle(group_views)",
            "dataset.curriculum_draw_length",
            "dataset_sampler_draws[ds_idx] += 1",
            "deferred_locations.pop_for_dataset(ds_idx)",
            "deferred_locations=deferred_locations.to_list()",
            "'version': 8",
            '"dataloader_state_version": 8',
        ):
            self.assertIn(contract, source)
        self.assertIn("exact curriculum group/view failed", source)
        self.assertNotIn("discarded pending current_batch", source)

    def test_durable_eval_reuse_requires_matching_json_and_workbook(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        start = source.index("evaluation_recorded()")
        end = source.index("evaluation_seconds()", start)
        guard = source[start:end]
        self.assertIn("checkpoints.json", guard)
        self.assertIn("ui5_crop_rollout4_curriculum_evaluation.xlsx", guard)
        self.assertIn("workbook_steps != expected_steps", guard)
        self.assertIn("overall_steps != expected_steps", guard)
        self.assertIn("Counter(task_steps)", guard)

    def test_training_and_evaluation_process_boundaries_are_separate(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        train = source.index('bash "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"')
        promote = source.index("scripts/locany_ui5_checkpoint.py\" promote", train)
        evaluate = source.index("evaluate_and_register", promote)
        self.assertLess(train, promote)
        self.assertLess(promote, evaluate)
        self.assertNotIn("torchrun \\", source[evaluate:])

    def test_incomplete_artifact_commit_reruns_evaluation_instead_of_reusing_status(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        start = source.index("evaluate_and_register()")
        end = source.index("current_step=0", start)
        body = source[start:end]
        self.assertNotIn("[EVAL RECOVER]", body)
        self.assertIn('if [[ -d "${eval_dir}" ]]; then', body)
        self.assertIn("eval_command+=(--overwrite)", body)
        self.assertIn('"${eval_command[@]}"', body)

    def test_step_200_recovery_records_the_original_model_as_resume_source(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('if (( previous_step == 0 )); then', source)
        self.assertIn('recovered_resume_from="${MODEL_PATH}"', source)
        self.assertIn('"${recovered_resume_from}"', source)

    def test_completion_marker_is_published_only_after_all_seven_evaluations(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        loop = "for completed_step in 0 200 400 600 800 1000 1200; do"
        marker = 'destination = output_dir / "pipeline_complete.json"'
        self.assertIn(loop, source)
        self.assertIn(marker, source)
        self.assertLess(source.index(loop), source.index(marker))
        self.assertIn(
            "actual_steps != expected_steps",
            source[source.index(loop) : source.index(marker) + len(marker)],
        )


if __name__ == "__main__":
    unittest.main()
