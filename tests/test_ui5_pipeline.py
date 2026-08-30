from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collect_ui5_metrics
import check_locany_environment
import check_ui5_train_eval_content_overlap
import check_ui5_validation_early_stop
import locany_ui5_checkpoint
import locany_ui5_common
import patch_locany_checkpoint
import preflight_locany_runtime
import prepare_ui5_validation_eval_input
import run_locany_ui5_local_debug
import score_ui5_frozen_gate
import submit_locany_ui5
from eaglevl.train.ui5_excel_logger import build_eval_rows
from eaglevl.train.ui5_checkpoint_utils import atomic_save_with_fsync


class RuntimeConfigTests(unittest.TestCase):
    def test_content_leak_gate_rejects_validation_test_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation"
            test = root / "test"
            validation.mkdir()
            test.mkdir()
            for task in locany_ui5_common.TASKS:
                (validation / f"ui_{task}_val.jsonl").write_text("{}\n", encoding="utf-8")
                (test / locany_ui5_common.TASK_JSONL[task]).write_text("{}\n", encoding="utf-8")
            train = root / "unique_images.jsonl"
            train.write_text(json.dumps({"content_id": "train-only"}) + "\n", encoding="utf-8")
            output = root / "overlap.json"
            args = Namespace(
                train_unique_manifest=train,
                validation_data_dir=validation,
                test_data_dir=test,
                output=output,
            )
            with mock.patch.object(
                check_ui5_train_eval_content_overlap,
                "_content_ids",
                side_effect=[({"eval-shared"}, 5), ({"eval-shared"}, 5)],
            ):
                with self.assertRaisesRegex(RuntimeError, "validation-test=1"):
                    check_ui5_train_eval_content_overlap.build(args)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["passes"])
            self.assertEqual(report["validation_test_content_overlap_count"], 1)

    def test_local_debug_starts_same_pipeline_without_evaluation(self) -> None:
        args = run_locany_ui5_local_debug.parse_args(
            [
                "--machine",
                "a800",
                "--gpus",
                "4",
                "--cuda-devices",
                "0,1,2,3",
                "--max-steps",
                "20",
                "--run-name",
                "local-smoke",
                "--project-root",
                str(PROJECT_ROOT),
            ]
        )
        env = run_locany_ui5_local_debug.build_environment(args, base_env={})
        command = run_locany_ui5_local_debug.build_command(args)
        self.assertEqual(env["ENABLE_EVAL"], "0")
        self.assertEqual(env["EVAL_AT_START"], "0")
        self.assertEqual(env["MAX_NUM_TOKENS"], "12800")
        self.assertEqual(env["GRADIENT_ACCUMULATION_STEPS"], "2")
        self.assertEqual(env["MAX_STEPS"], "20")
        self.assertEqual(env["SAVE_STEPS"], "20")
        self.assertEqual(env["RUN_NAME"], "local-smoke")
        self.assertEqual(
            Path(command[-1]), PROJECT_ROOT / "shell" / "run_locany_ui5_pipeline.sh"
        )

    def test_local_eight_gpu_debug_keeps_formal_accumulation_schedule(self) -> None:
        args = run_locany_ui5_local_debug.parse_args(
            ["--gpus", "8", "--project-root", str(PROJECT_ROOT)]
        )
        env = run_locany_ui5_local_debug.build_environment(args, base_env={})
        self.assertEqual(env["MAX_NUM_TOKENS"], "25600")
        self.assertEqual(env["GRADIENT_ACCUMULATION_STEPS"], "1")

    def test_a800_defaults_are_gpu_count_specific(self) -> None:
        common = {"MACHINE_TYPE": "a800", "VERSION": "v4"}
        four = locany_ui5_common.resolve_runtime_config(
            {**common, "GPU_COUNT": "4", "CUDA_DEVICES": "0,1,2,3"}
        )
        eight = locany_ui5_common.resolve_runtime_config(
            {
                **common,
                "GPU_COUNT": "8",
                "CUDA_DEVICES": "0,1,2,3,4,5,6,7",
            }
        )
        self.assertEqual(four["MAX_NUM_TOKENS"], 12800)
        self.assertEqual(eight["MAX_NUM_TOKENS"], 25600)
        self.assertEqual(four["GRADIENT_ACCUMULATION_STEPS"], 2)
        self.assertEqual(eight["GRADIENT_ACCUMULATION_STEPS"], 1)
        self.assertEqual(
            locany_ui5_common.machine_resource_config("a800")["cpu"], 58
        )
        self.assertEqual(
            locany_ui5_common.machine_resource_config("a800")["group_id"], 1602
        )
        self.assertNotEqual(four["OUTPUT_DIR"], eight["OUTPUT_DIR"])
        locany_ui5_common.assert_gpu_mode_consistency(four, eight)
        self.assertEqual(four["MAX_NUM_TOKENS_SCOPE"], "per_rank_packed_batch")
        self.assertEqual(
            four["TRAINING_DATA_SOURCE_DIR"],
            "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/"
            "code/Eagle/Embodied/data/ui_defect_locany_v3",
        )

    def test_gpu_parity_check_rejects_training_hyperparameter_drift(self) -> None:
        four = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "4",
                "CUDA_DEVICES": "0,1,2,3",
            }
        )
        eight = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "8",
                "CUDA_DEVICES": "0,1,2,3,4,5,6,7",
            }
        )
        eight["LEARNING_RATE"] = "3e-5"
        with self.assertRaisesRegex(ValueError, "LEARNING_RATE"):
            locany_ui5_common.assert_gpu_mode_consistency(four, eight)

    def test_gpu_parity_check_rejects_wrong_accumulation_schedule(self) -> None:
        four = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "4",
                "CUDA_DEVICES": "0,1,2,3",
                "GRADIENT_ACCUMULATION_STEPS": "1",
            }
        )
        eight = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "8",
                "CUDA_DEVICES": "0,1,2,3,4,5,6,7",
            }
        )
        with self.assertRaisesRegex(ValueError, "original 2/1"):
            locany_ui5_common.assert_gpu_mode_consistency(four, eight)

    def test_explicit_max_num_tokens_wins(self) -> None:
        config = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "4",
                "CUDA_DEVICES": "0,1,2,3",
                "MAX_NUM_TOKENS": "25600",
            }
        )
        self.assertEqual(config["MAX_NUM_TOKENS"], 25600)

    def test_h20_machine_paths_and_attention(self) -> None:
        config = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "h20",
                "GPU_COUNT": "4",
                "CUDA_DEVICES": "0,1,2,3",
            }
        )
        self.assertEqual(config["ATTN_IMPLEMENTATION"], "magi")
        self.assertEqual(config["MAX_NUM_TOKENS"], 12800)
        self.assertEqual(config["GRADIENT_ACCUMULATION_STEPS"], 2)
        self.assertIn("intelligent-service-arnold-hl", config["WORKSPACE"])
        self.assertTrue(config["PROJECT_ROOT"].startswith("/mnt/"))

    def test_aiai_locate_resource_group_renders_group_and_queue(self) -> None:
        resource = locany_ui5_common.machine_resource_config(
            "a800", resource_group="aiai_locate"
        )
        self.assertEqual(resource["group_id"], 2146)
        self.assertEqual(
            resource["display_name"], "ies_aiai_experience/AIAI_locate"
        )
        self.assertEqual(
            resource["queue_name"],
            "compute-3302-yg-cloudnative-ai-aiai.locate-guarantee",
        )

        args = submit_locany_ui5.parse_args(
            [
                "--machine",
                "a800",
                "--gpus",
                "4",
                "--resource-group",
                "aiai_locate",
                "--render-only",
            ]
        )
        rendered, runtime = submit_locany_ui5.render_job(args)
        self.assertIn("        - 2146", rendered)
        self.assertIn(
            "queueName: compute-3302-yg-cloudnative-ai-aiai.locate-guarantee",
            rendered,
        )
        self.assertIn("name: 'locany-ui5-v4-a800x4-aiai-locate'", rendered)
        self.assertEqual(runtime["RESOURCE_GROUP"], "aiai_locate")
        self.assertEqual(runtime["RESOURCE_GROUP_ID"], 2146)

    def test_unknown_resource_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown resource group"):
            locany_ui5_common.machine_resource_config(
                "a800", resource_group="does_not_exist"
            )

    def test_rendered_yaml_uses_posix_cluster_paths(self) -> None:
        args = Namespace(
            machine="a800",
            gpus=4,
            cuda_devices=None,
            eval_gpu_devices=None,
            max_num_tokens=12800,
            max_seq_length=None,
            max_num_tokens_per_sample=None,
            max_steps=2,
            save_steps=4000,
            eval_interval_steps=1000,
            eval_max_images_per_task=10,
            warmup_steps=0,
            learning_rate="2e-5",
            version="v4",
            data_version="v3",
            run_name=None,
            scorer_root=None,
            training_data_source_dir=None,
            training_data_dir=None,
            eval_checkpoint=None,
            eval_step=None,
            eval_skip_patch=False,
            eval_fail_policy="stop",
            enable_eval=False,
            eval_at_start=True,
            config=locany_ui5_common.DEFAULT_CONFIG_PATH,
            template=submit_locany_ui5.TEMPLATE_PATH,
        )
        rendered, runtime = submit_locany_ui5.render_job(args)
        self.assertIn("mnt: /mnt/bn/intelligent-service-yg/", rendered)
        self.assertIn('MAX_NUM_TOKENS: "12800"', rendered)
        self.assertIn('EVAL_MAX_IMAGES_PER_TASK: "10"', rendered)
        self.assertIn('INSTALL_SYSTEM_RUNTIME_DEPS: "1"', rendered)
        self.assertNotIn("@@", rendered)
        self.assertEqual(runtime["ENABLE_EVAL"], 0)
        self.assertEqual(runtime["INSTALL_SYSTEM_RUNTIME_DEPS"], 1)

    def test_formal_runtime_dependency_install_can_be_disabled(self) -> None:
        args = submit_locany_ui5.parse_args(
            [
                "--machine",
                "a800",
                "--gpus",
                "4",
                "--no-install-system-runtime-deps",
                "--render-only",
            ]
        )
        rendered, runtime = submit_locany_ui5.render_job(args)
        self.assertIn('INSTALL_SYSTEM_RUNTIME_DEPS: "0"', rendered)
        self.assertEqual(runtime["INSTALL_SYSTEM_RUNTIME_DEPS"], 0)

    def test_validation_early_stop_defaults_off_and_is_explicitly_switchable(self) -> None:
        default_args = submit_locany_ui5.parse_args(
            ["--machine", "a800", "--gpus", "4", "--render-only"]
        )
        default_env = submit_locany_ui5.build_submission_environment(default_args)
        self.assertFalse(default_args.validation_early_stop)
        self.assertEqual(default_env["EVAL_VALIDATION_EARLY_STOP"], "0")

        enabled_args = submit_locany_ui5.parse_args(
            [
                "--machine",
                "a800",
                "--gpus",
                "4",
                "--validation-early-stop",
                "--render-only",
            ]
        )
        rendered, runtime = submit_locany_ui5.render_job(enabled_args)
        self.assertTrue(enabled_args.validation_early_stop)
        self.assertEqual(runtime["EVAL_VALIDATION_EARLY_STOP"], 1)
        self.assertIn('EVAL_VALIDATION_EARLY_STOP: "1"', rendered)

        disabled_args = submit_locany_ui5.parse_args(
            [
                "--machine",
                "a800",
                "--gpus",
                "4",
                "--no-validation-early-stop",
                "--render-only",
            ]
        )
        disabled_env = submit_locany_ui5.build_submission_environment(disabled_args)
        self.assertFalse(disabled_args.validation_early_stop)
        self.assertEqual(disabled_env["EVAL_VALIDATION_EARLY_STOP"], "0")

        pipeline = (PROJECT_ROOT / "shell" / "run_locany_ui5_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '[[ "${EVAL_VALIDATION_EARLY_STOP:-0}" == "1"', pipeline
        )
        self.assertIn('&& "${EVAL_DATA_SPLIT}" == "validation"', pipeline)

    def test_runtime_config_defaults_validation_early_stop_off(self) -> None:
        default = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "4",
                "CUDA_DEVICES": "0,1,2,3",
            }
        )
        enabled = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "4",
                "CUDA_DEVICES": "0,1,2,3",
                "EVAL_VALIDATION_EARLY_STOP": "1",
            }
        )
        self.assertEqual(default["EVAL_VALIDATION_EARLY_STOP"], 0)
        self.assertEqual(enabled["EVAL_VALIDATION_EARLY_STOP"], 1)

    def test_runtime_config_accepts_source_balanced_rotating_sampling(self) -> None:
        config = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "4",
                "CUDA_DEVICES": "0,1,2,3",
                "UI5_CROP_TRAIN_MODE": "crop_only",
                "UI5_UI_SAMPLING_MODE": "task_source_balanced_rotating",
                "UI_NEGATIVE_TO_POSITIVE_RATIO": "2.0",
            }
        )
        self.assertEqual(
            config["UI5_UI_SAMPLING_MODE"],
            "task_source_balanced_rotating",
        )
        self.assertEqual(config["UI_NEGATIVE_TO_POSITIVE_RATIO"], 2.0)

    def test_runtime_config_rejects_nonpositive_source_balanced_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            locany_ui5_common.resolve_runtime_config(
                {
                    "MACHINE_TYPE": "a800",
                    "GPU_COUNT": "4",
                    "CUDA_DEVICES": "0,1,2,3",
                    "UI5_CROP_TRAIN_MODE": "crop_only",
                    "UI5_UI_SAMPLING_MODE": "task_source_balanced_rotating",
                    "UI_NEGATIVE_TO_POSITIVE_RATIO": "0",
                }
            )

    def test_gpu_parity_rejects_validation_early_stop_drift(self) -> None:
        four = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "4",
                "CUDA_DEVICES": "0,1,2,3",
                "EVAL_VALIDATION_EARLY_STOP": "0",
            }
        )
        eight = locany_ui5_common.resolve_runtime_config(
            {
                "MACHINE_TYPE": "a800",
                "GPU_COUNT": "8",
                "CUDA_DEVICES": "0,1,2,3,4,5,6,7",
                "EVAL_VALIDATION_EARLY_STOP": "1",
            }
        )
        with self.assertRaisesRegex(ValueError, "EVAL_VALIDATION_EARLY_STOP"):
            locany_ui5_common.assert_gpu_mode_consistency(four, eight)

    def test_preflight_uses_distinct_libgl_exit_code(self) -> None:
        self.assertEqual(preflight_locany_runtime.EXIT_LIBGL_MISSING, 42)

    def test_preflight_libgl_error_has_distinct_code_and_fix_hint(self) -> None:
        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("libGL.so.1: cannot open shared object file")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(
                preflight_locany_runtime,
                "parse_args",
                return_value=Namespace(processor_path=Path("model"), skip_processor=False),
            ),
            mock.patch("builtins.__import__", side_effect=fake_import),
            mock.patch("builtins.print") as printer,
        ):
            code = preflight_locany_runtime.main()
        self.assertEqual(code, preflight_locany_runtime.EXIT_LIBGL_MISSING)
        rendered = " ".join(str(call.args[0]) for call in printer.call_args_list if call.args)
        self.assertIn("libgl1", rendered)
        self.assertIn("libglib2.0-0", rendered)

    def test_preflight_reports_processor_path_and_preserves_failure_type(self) -> None:
        cv2 = ModuleType("cv2")
        cv2.__version__ = "test"
        cv2.__file__ = "cv2.so"
        transformers = ModuleType("transformers")

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                raise ValueError(f"bad local processor: {path}; {kwargs}")

        transformers.AutoProcessor = FakeProcessor
        with tempfile.TemporaryDirectory() as temporary:
            processor_path = Path(temporary)
            with (
                mock.patch.object(
                    preflight_locany_runtime,
                    "parse_args",
                    return_value=Namespace(
                        processor_path=processor_path, skip_processor=False
                    ),
                ),
                mock.patch.dict(sys.modules, {"cv2": cv2, "transformers": transformers}),
                mock.patch("builtins.print") as printer,
            ):
                code = preflight_locany_runtime.main()
        self.assertEqual(code, preflight_locany_runtime.EXIT_PROCESSOR_FAILED)
        rendered = " ".join(str(call.args[0]) for call in printer.call_args_list if call.args)
        self.assertIn(str(processor_path.resolve()), rendered)
        self.assertIn("ValueError", rendered)

    def test_environment_pre_post_fingerprint_change_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = {
                "schema_version": 1,
                "fingerprint_sha256": "a",
                "stable": {
                    "packages": {"torch": "1", "transformers": "1", "deepspeed": "1"}
                },
            }
            changed = {
                **first,
                "fingerprint_sha256": "b",
            }
            with (
                mock.patch.object(
                    check_locany_environment,
                    "parse_args",
                    return_value=Namespace(output_dir=output, phase="pre", allow_change=False),
                ),
                mock.patch.object(check_locany_environment, "collect_environment", return_value=first),
            ):
                self.assertEqual(check_locany_environment.main(), 0)
            with (
                mock.patch.object(
                    check_locany_environment,
                    "parse_args",
                    return_value=Namespace(output_dir=output, phase="post", allow_change=False),
                ),
                mock.patch.object(check_locany_environment, "collect_environment", return_value=changed),
            ):
                self.assertEqual(check_locany_environment.main(), 46)

    def test_h20_render_uses_h20_resources(self) -> None:
        args = Namespace(
            machine="h20",
            gpus=4,
            cuda_devices=None,
            eval_gpu_devices=None,
            max_num_tokens=None,
            max_seq_length=None,
            max_num_tokens_per_sample=None,
            max_steps=16000,
            save_steps=4000,
            eval_interval_steps=1000,
            warmup_steps=500,
            learning_rate="2e-5",
            version="v4",
            data_version="v3",
            run_name=None,
            scorer_root=None,
            training_data_source_dir=None,
            training_data_dir=None,
            eval_checkpoint=None,
            eval_step=None,
            eval_skip_patch=False,
            eval_fail_policy="stop",
            enable_eval=True,
            eval_at_start=True,
            config=locany_ui5_common.DEFAULT_CONFIG_PATH,
            template=submit_locany_ui5.TEMPLATE_PATH,
        )
        rendered, runtime = submit_locany_ui5.render_job(args)
        self.assertIn("gpuv: NVIDIA_H20", rendered)
        self.assertIn("queueName: compute-329-hl", rendered)
        self.assertEqual(runtime["ATTN_IMPLEMENTATION"], "magi")


class CheckpointTests(unittest.TestCase):
    def make_eval_checkpoint(self, root: Path, step: int) -> Path:
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        return checkpoint

    def test_training_args_atomic_publish_rejects_zero_byte_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "training_args.bin"

            def empty_save(_obj, path):
                Path(path).touch()

            with self.assertRaisesRegex(RuntimeError, "empty file"):
                atomic_save_with_fsync(empty_save, object(), target)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob(".training_args.bin.tmp-*")), [])

    def test_training_args_atomic_publish_replaces_only_after_nonempty_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "training_args.bin"
            target.write_bytes(b"old")

            def save(_obj, path):
                Path(path).write_bytes(b"new-training-args")
                return "saved"

            self.assertEqual(atomic_save_with_fsync(save, object(), target), "saved")
            self.assertEqual(target.read_bytes(), b"new-training-args")
            self.assertEqual(list(target.parent.glob(".training_args.bin.tmp-*")), [])

    def test_sharded_model_requires_every_indexed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-1000"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"one")
            self.assertFalse(
                locany_ui5_checkpoint.validate_checkpoint(
                    checkpoint, mode="eval"
                )["valid"]
            )
            (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"two")
            self.assertTrue(
                locany_ui5_checkpoint.validate_checkpoint(
                    checkpoint, mode="eval"
                )["valid"]
            )

    def make_resume_checkpoint(self, root: Path, step: int) -> Path:
        checkpoint = self.make_eval_checkpoint(root, step)
        (checkpoint / "training_args.bin").write_bytes(b"training-arguments")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        state_dir = checkpoint / f"global_step{step}"
        state_dir.mkdir()
        (state_dir / "mp_rank_00_model_states.pt").write_bytes(b"state")
        (state_dir / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(
            b"optimizer"
        )
        return checkpoint

    def test_zero_byte_training_args_is_eval_only_not_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.make_eval_checkpoint(Path(temporary), 1000)
            (checkpoint / "training_args.bin").touch()
            report = locany_ui5_checkpoint.validate_checkpoint(
                checkpoint, mode="resume"
            )
            self.assertFalse(report["valid"])
            self.assertIn("missing or empty training_args.bin", report["errors"])
            self.assertTrue(
                locany_ui5_checkpoint.validate_checkpoint(
                    checkpoint, mode="eval"
                )["valid"]
            )

    def test_deepspeed_optimizer_state_is_required_not_just_model_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.make_eval_checkpoint(Path(temporary), 1000)
            (checkpoint / "training_args.bin").write_bytes(b"args")
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 1000}), encoding="utf-8"
            )
            state_dir = checkpoint / "global_step1000"
            state_dir.mkdir()
            (state_dir / "mp_rank_00_model_states.pt").write_bytes(b"state")
            report = locany_ui5_checkpoint.validate_checkpoint(
                checkpoint, mode="resume"
            )
            self.assertFalse(report["valid"])
            self.assertIn(
                "missing optimizer/DeepSpeed optimizer state", report["errors"]
            )

    def test_checkpoint_zero_is_not_a_training_resume_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint_zero = self.make_eval_checkpoint(output, 0)

            self.assertEqual(
                locany_ui5_checkpoint.list_training_checkpoints(output), []
            )
            self.assertTrue(checkpoint_zero.is_dir())

            broken_nonzero = self.make_eval_checkpoint(output, 1000)
            candidates = locany_ui5_checkpoint.list_training_checkpoints(output)
            self.assertEqual(
                [(step, path.name) for step, path in candidates],
                [(1000, "checkpoint-1000")],
            )
            self.assertFalse(
                locany_ui5_checkpoint.validate_checkpoint(
                    broken_nonzero, mode="resume"
                )["valid"]
            )

    def test_training_candidates_cli_excludes_checkpoint_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.make_eval_checkpoint(output, 0)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "locany_ui5_checkpoint.py"),
                    "training-candidates",
                    "--output-dir",
                    str(output),
                    "--field",
                    "count",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "0")

    def test_latest_resume_never_falls_back_past_newer_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.make_resume_checkpoint(output, 1000)
            broken = self.make_eval_checkpoint(output, 2000)
            (broken / "training_args.bin").touch()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "locany_ui5_checkpoint.py"),
                    "latest",
                    "--output-dir",
                    str(output),
                    "--require-resume",
                    "--field",
                    "step",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "0")
            self.assertTrue(broken.is_dir(), "validator must never delete user data")

    def test_patch_is_idempotent_and_force_refreshes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            (base / "modeling_locateanything.py").write_text(
                "BASE = 1\n", encoding="utf-8"
            )
            project = root / "project"
            inference = project / "eaglevl" / "utils" / "locany"
            inference.mkdir(parents=True)
            (inference / "modeling_locateanything.py").write_text(
                "PROJECT = 1\n", encoding="utf-8"
            )
            relation_dir = project / "eaglevl" / "model" / "locany"
            relation_dir.mkdir(parents=True)
            (relation_dir / "relation_modules.py").write_text(
                "RELATION = 1\n", encoding="utf-8"
            )
            checkpoint = self.make_eval_checkpoint(root, 1000)

            first = patch_locany_checkpoint.patch_checkpoint(
                base_model=base,
                checkpoint=checkpoint,
                project_root=project,
            )
            second = patch_locany_checkpoint.patch_checkpoint(
                base_model=base,
                checkpoint=checkpoint,
                project_root=project,
            )
            self.assertIn("modeling_locateanything.py", first["copied"])
            self.assertIn("modeling_locateanything.py", second["skipped"])
            self.assertIn("relation_modules.py", first["copied"])

            (inference / "modeling_locateanything.py").write_text(
                "PROJECT = 2\n", encoding="utf-8"
            )
            forced = patch_locany_checkpoint.patch_checkpoint(
                base_model=base,
                checkpoint=checkpoint,
                project_root=project,
                force=True,
            )
            self.assertIn("modeling_locateanything.py", forced["copied"])
            self.assertEqual(
                (checkpoint / "modeling_locateanything.py").read_text(encoding="utf-8"),
                "PROJECT = 2\n",
            )

    def test_relation_weight_validation_covers_relation_gate_and_pbd(self) -> None:
        keys = {
            f"model.{group}weight"
            for group in patch_locany_checkpoint.REQUIRED_RELATION_WEIGHT_GROUPS
        }
        report = patch_locany_checkpoint.validate_relation_weight_keys(keys)
        self.assertTrue(report["valid"], report)
        keys = {key for key in keys if "gate_heads" not in key}
        report = patch_locany_checkpoint.validate_relation_weight_keys(keys)
        self.assertFalse(report["valid"])
        self.assertIn(
            "relation_pyramid.gate_heads.", report["missing_groups"]
        )

    def test_pbd_checkpoint_config_validation_requires_saved_selector_ids(self) -> None:
        report = patch_locany_checkpoint.validate_pbd_config(
            {
                "box_start_token_id": 151668,
                "text_config": {
                    "block_size": 6,
                    "text_mask_token_id": 151666,
                },
            }
        )
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["block_size"], 6)
        invalid = patch_locany_checkpoint.validate_pbd_config(
            {"box_start_token_id": 151668, "text_config": {}}
        )
        self.assertFalse(invalid["valid"])
        self.assertIn("text_config.block_size", invalid["missing"])

    def test_cleanup_keeps_latest_and_formal_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for step in (1000, 2000, 3000, 4000):
                self.make_resume_checkpoint(output, step)
            latest = output / "checkpoint-4000"
            self.assertTrue(
                locany_ui5_checkpoint.validate_checkpoint(latest, mode="resume")[
                    "valid"
                ]
            )
            for step, path in locany_ui5_checkpoint.list_checkpoints(output):
                if step != 4000 and step % 4000 != 0:
                    locany_ui5_checkpoint.safe_remove_checkpoint(path, output)
            self.assertEqual(
                [step for step, _ in locany_ui5_checkpoint.list_checkpoints(output)],
                [4000],
            )


class HistoryTests(unittest.TestCase):
    def test_validation_early_stop_requires_two_points_without_either_macro_improving(self):
        rows = [
            {
                "step": 1000,
                "evaluation_status": "success",
                "evaluation_split": "validation",
                "image_macro_f1": 0.40,
                "bbox_macro_f1": 0.30,
            },
            {
                "step": 2000,
                "evaluation_status": "success",
                "evaluation_split": "validation",
                "image_macro_f1": 0.39,
                "bbox_macro_f1": 0.29,
            },
            {
                "step": 3000,
                "evaluation_status": "success",
                "evaluation_split": "validation",
                "image_macro_f1": 0.38,
                "bbox_macro_f1": 0.28,
            },
        ]
        result = check_ui5_validation_early_stop.evaluate(
            rows, patience=2, min_delta=0.0
        )
        self.assertTrue(result["should_stop"])
        rows[-1]["bbox_macro_f1"] = 0.31
        result = check_ui5_validation_early_stop.evaluate(
            rows, patience=2, min_delta=0.0
        )
        self.assertFalse(result["should_stop"])

    def test_collect_metrics_direct_script_uses_checkout_eaglevl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "collect_ui5_metrics.py"),
                    "--help",
                ],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("{record,has-success,convert-report}", completed.stdout)

    def test_tiled_gate_aggregation_keeps_image_probability(self) -> None:
        diagnostics = locany_ui5_common.aggregate_tiled_gate_diagnostics(
            [
                {
                    "available": True,
                    "p_defect": 0.27,
                    "would_pass": False,
                    "gate_filtered": True,
                },
                {
                    "available": True,
                    "p_defect": 0.81,
                    "would_pass": True,
                    "gate_filtered": False,
                },
            ],
            crop_mode="detector_scan",
        )
        self.assertEqual(diagnostics["p_defect"], 0.81)
        self.assertEqual(diagnostics["p_defect_aggregation"], "max_tile")
        self.assertEqual(diagnostics["p_defect_tile_count"], 2)
        self.assertTrue(diagnostics["would_pass"])
        self.assertFalse(diagnostics["gate_filtered"])

    def test_collect_gate_metrics_recovers_legacy_detector_scan_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_dir = root / "gt"
            scorer_root = root / "scorer"
            prediction_dir = root / "predictions"
            gt_dir.mkdir()
            scorer_root.mkdir()
            image = root / "images" / "same.jpg"
            source = gt_dir / locany_ui5_common.TASK_JSONL["occlusion"]
            source.write_text(
                json.dumps({"images": str(image), "positive": True}) + "\n",
                encoding="utf-8",
            )
            (scorer_root / "qwen3vl_merge_and_score_fixed_5tasks.py").write_text(
                textwrap.dedent(
                    """
                    def get_gt_payload(sample):
                        return sample.get("positive", False)

                    def extract_bboxes_for_issue(payload, issue):
                        return [[0, 0, 1, 1]] if payload else []
                    """
                ),
                encoding="utf-8",
            )
            gate_dir = prediction_dir / "occlusion" / "gate"
            gate_dir.mkdir(parents=True)
            (gate_dir / "same.json").write_text(
                json.dumps(
                    {
                        "image_path": str(image),
                        "prediction_status": "defect",
                        "p_defect": None,
                        "would_pass": True,
                        "tile_gates": [
                            {"available": True, "p_defect": 0.24},
                            {"available": True, "p_defect": 0.76},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            metrics = collect_ui5_metrics.collect_gate_metrics(
                prediction_dir,
                gt_dir,
                scorer_root,
            )

        self.assertEqual(metrics["occlusion"]["samples"], 1)
        self.assertEqual(metrics["occlusion"]["p_defect_pos"], 0.76)
        self.assertEqual(metrics["occlusion"]["legacy_tile_gate_recovered"], 1)
        self.assertEqual(
            metrics["occlusion"]["_sweep_samples"][0]["p_defect"], 0.76
        )

    def test_gate_sweep_zero_is_raw_and_selects_without_regeneration(self) -> None:
        metrics = {
            task: {"_sweep_samples": []}
            for task in locany_ui5_common.TASKS
        }
        metrics["occlusion"]["_sweep_samples"] = [
            {"label": True, "raw_positive": True, "p_defect": 0.30},
            {"label": False, "raw_positive": True, "p_defect": 0.10},
            {"label": False, "raw_positive": False, "p_defect": 0.90},
        ]
        sweep = collect_ui5_metrics.build_gate_threshold_sweep(metrics)
        raw = sweep["tasks"]["occlusion"]["raw"]
        selected = sweep["tasks"]["occlusion"]["selected"]
        self.assertEqual(raw["threshold"], 0.0)
        self.assertEqual(raw["predicted_positive"], 2)
        self.assertEqual(raw["tp"], 1)
        self.assertEqual(raw["fp"], 1)
        self.assertGreaterEqual(selected["f1"], raw["f1"])
        self.assertEqual(len(sweep["tasks"]["occlusion"]["sweep"]), 61)

    def test_frozen_gate_publishes_filtered_prediction_tree_without_mutating_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "raw"
            gated = root / "gated"
            for index, task in enumerate(locany_ui5_common.TASKS):
                task_dir = predictions / task
                gate_dir = task_dir / "gate"
                gate_dir.mkdir(parents=True)
                payload = [{"bbox_2d": [1, 2, 3, 4], "label": "x"}]
                (task_dir / "sample.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                (gate_dir / "sample.json").write_text(
                    json.dumps({"p_defect": 0.2 + 0.1 * index}), encoding="utf-8"
                )
            thresholds = {task: 0.5 for task in locany_ui5_common.TASKS}
            counts = score_ui5_frozen_gate._publish_gated_predictions(
                predictions, gated, thresholds
            )
            self.assertEqual(
                json.loads((predictions / "occlusion" / "sample.json").read_text()),
                [{"bbox_2d": [1, 2, 3, 4], "label": "x"}],
            )
            self.assertEqual(
                json.loads((gated / "occlusion" / "sample.json").read_text()), []
            )
            self.assertEqual(counts["content_missing"]["kept_by_frozen_gate"], 1)

    def test_bbox_gated_metrics_are_taken_from_genuine_rescore(self) -> None:
        metrics = {
            "tasks": {
                task: {
                    "image": {"precision": 0.4, "recall": 0.5, "f1": 0.44, "tp": 4, "fp": 6, "fn": 4, "tn": 2},
                    "bbox": {"precision": 0.3, "recall": 0.6, "f1": 0.4, "tp": 3, "fp": 7, "fn": 2},
                }
                for task in locany_ui5_common.TASKS
            },
            "macro": {
                "image": {"precision": 0.4, "recall": 0.5, "f1": 0.44},
                "bbox": {"precision": 0.3, "recall": 0.6, "f1": 0.4},
            },
        }
        gates = {
            task: {
                "selected_gate_threshold": 0.5,
                "gated_precision": 0.9,
                "gated_recall": 0.8,
                "gated_f1": 0.85,
                "gated_predicted_positive": 9,
                "gated_metrics_by_granularity": {
                    "image": {"precision": 0.7, "recall": 0.5, "f1": 0.58, "tp": 4, "fp": 2},
                    "bbox": {"precision": 0.25, "recall": 0.4, "f1": 0.31, "tp": 2, "fp": 6},
                },
            }
            for task in locany_ui5_common.TASKS
        }
        rows = build_eval_rows(step=1000, checkpoint="ckpt", metrics=metrics, gate_metrics=gates)
        bbox = next(
            row
            for row in rows
            if row["task"] == "element_overlap" and row["granularity"] == "bbox"
        )
        image = next(
            row
            for row in rows
            if row["task"] == "element_overlap" and row["granularity"] == "image"
        )
        self.assertEqual(bbox["gated_f1"], 0.31)
        self.assertEqual(image["gated_f1"], 0.58)
        self.assertNotEqual(bbox["gated_f1"], image["gated_f1"])

    def test_validation_staging_reports_content_unique_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "staged"
            source.mkdir()
            image = root / "same.png"
            image.write_bytes(b"same-image-content")
            for task in locany_ui5_common.TASKS:
                (source / f"ui_{task}_val.jsonl").write_text(
                    json.dumps({"images": [str(image)]}) + "\n", encoding="utf-8"
                )
            summary = prepare_ui5_validation_eval_input.build(
                Namespace(source_dir=source, output_dir=output)
            )
            self.assertEqual(summary["total_records"], 5)
            self.assertEqual(summary["content_unique_images"], 1)
            self.assertEqual(summary["expected_unique_images"], 1)
            for filename in locany_ui5_common.TASK_JSONL.values():
                self.assertTrue((output / filename).is_file())

    def test_legacy_markdown_report_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "all_tasks_evaluation.txt"
            sections = []
            for title in ("Bbox", "Image"):
                rows = [
                    f"| {issue} | 0.5000 | 0.6000 | 0.5500 |"
                    for issue in locany_ui5_common.TASK_ISSUE_NAMES.values()
                ]
                rows.append("| 五类平均 | 0.5000 | 0.6000 | 0.5500 |")
                sections.append(
                    f">>> {title} 粒度\n"
                    "| task | prec | recall | f1 |\n"
                    "|---|---:|---:|---:|\n"
                    + "\n".join(rows)
                )
            report.write_text("\n".join(sections), encoding="utf-8")
            converted = collect_ui5_metrics.parse_markdown_report(report)
            self.assertEqual(converted["macro"]["image"]["f1"], 0.55)
            self.assertEqual(set(converted["tasks"]), set(locany_ui5_common.TASKS))

    def test_history_upserts_by_step_and_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric_path = root / "metrics.json"
            metric_path.write_text(
                json.dumps(
                    {
                        "macro": {
                            "image": {"precision": 0.6, "recall": 0.7, "f1": 0.65},
                            "bbox": {"precision": 0.5, "recall": 0.6, "f1": 0.55},
                        },
                        "tasks": {
                            "occlusion": {
                                "image": {"precision": 0.4, "recall": 0.5, "f1": 0.44},
                                "bbox": {"precision": 0.3, "recall": 0.4, "f1": 0.34},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                step=1000,
                machine_type="a800",
                gpu_count=4,
                max_num_tokens=12800,
                max_num_tokens_scope="per_rank_packed_batch",
                checkpoint=root / "checkpoint-1000",
                metrics_json=metric_path,
                start_time="start",
                end_time="end",
                status="success",
                prediction_dir=root / "pred",
                evaluation_run_dir=root / "eval",
                error="",
            )
            row = collect_ui5_metrics.build_row(args)
            collect_ui5_metrics.write_history(root, [row])
            history = collect_ui5_metrics.load_history(root / "evaluation_history.json")
            self.assertEqual(history[0]["macro_f1"], 0.65)
            self.assertEqual(history[0]["occlusion_bbox_f1"], 0.34)
            self.assertTrue((root / "evaluation_history.csv").is_file())


class ParallelInferenceTests(unittest.TestCase):
    def test_five_tasks_run_through_four_worker_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            processor = root / "processor"
            input_dir = root / "data"
            output_dir = root / "predictions"
            checkpoint.mkdir()
            processor.mkdir()
            input_dir.mkdir()
            for task, filename in locany_ui5_common.TASK_JSONL.items():
                (input_dir / filename).write_text(
                    json.dumps({"images": [f"{task}.png"]}) + "\n",
                    encoding="utf-8",
                )

            fake_inference = root / "fake_inference.py"
            fake_inference.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument('--output-dir', required=True)
                    parser.add_argument('--summary-path', required=True)
                    parser.add_argument('--tasks', nargs='+', required=True)
                    args, _ = parser.parse_known_args()
                    for task in args.tasks:
                        task_dir = Path(args.output_dir) / task
                        task_dir.mkdir(parents=True, exist_ok=True)
                        (task_dir / 'sample.json').write_text('[]\\n', encoding='utf-8')
                    summary = Path(args.summary_path)
                    summary.parent.mkdir(parents=True, exist_ok=True)
                    summary.write_text(json.dumps({'tasks': args.tasks}), encoding='utf-8')
                    """
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "run_ui5_parallel_inference.py"),
                "--checkpoint",
                str(checkpoint),
                "--processor-path",
                str(processor),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--gpu-devices",
                "0,1,2,3",
                "--attn-implementation",
                "sdpa",
                "--inference-script",
                str(fake_inference),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            status = json.loads(
                (output_dir / "parallel_inference_status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(status["success"])
            self.assertEqual(set(status["tasks"]), set(locany_ui5_common.TASKS))
            for task in locany_ui5_common.TASKS:
                self.assertTrue((output_dir / task / "sample.json").is_file())

    def test_model_load_preflight_failure_is_echoed_and_stops_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            processor = root / "processor"
            input_dir = root / "data"
            output_dir = root / "predictions"
            checkpoint.mkdir()
            processor.mkdir()
            input_dir.mkdir()
            for task, filename in locany_ui5_common.TASK_JSONL.items():
                (input_dir / filename).write_text(
                    json.dumps({"images": [f"{task}.png"]}) + "\n",
                    encoding="utf-8",
                )

            fake_inference = root / "failing_inference.py"
            fake_inference.write_text(
                textwrap.dedent(
                    """
                    import sys

                    if '--load-only' in sys.argv:
                        print('MODEL_LOAD_ROOT_CAUSE_FOR_TEST', file=sys.stderr)
                        raise SystemExit(17)
                    raise AssertionError('task worker must not start after failed preflight')
                    """
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "run_ui5_parallel_inference.py"),
                "--checkpoint",
                str(checkpoint),
                "--processor-path",
                str(processor),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--gpu-devices",
                "0,1,2,3",
                "--attn-implementation",
                "sdpa",
                "--inference-script",
                str(fake_inference),
                "--model-load-preflight",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            combined_output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 1, combined_output)
            self.assertIn("MODEL_LOAD_ROOT_CAUSE_FOR_TEST", combined_output)
            status = json.loads(
                (output_dir / "parallel_inference_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["model_load_preflight"]["return_code"], 17)
            self.assertIn(
                "MODEL_LOAD_ROOT_CAUSE_FOR_TEST",
                status["model_load_preflight"]["log_tail"],
            )
            self.assertEqual(status["tasks"], {})


if __name__ == "__main__":
    unittest.main()
