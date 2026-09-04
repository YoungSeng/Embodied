from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import aggregate_ui5_train_rollouts as aggregate
import preflight_ui5_train_rollouts as preflight
import prepare_ui5_train_rollout_bundle as prepare
import render_ui5_train_rollout_gallery as gallery
import run_ui5_train_rollout_worker as worker
import snapshot_ui5_train_rollouts as snapshot
import summarize_ui5_rollout_oom as oom_summary
from run_ui5_train_rollout_worker import (
    FORMAL_SEEDS,
    MAX_NUM_TOKENS_PER_SAMPLE,
    MAX_SEQ_LENGTH,
    OOMRetryFailure,
    PROCESSOR_IN_TOKEN_LIMIT,
    ROLLOUT_MAX_NEW_TOKENS,
    TRAINING_MAX_NUM_TOKENS,
    fixed_interleaved_samples,
    install_generation_token_budget,
    load_module,
    parse_args as parse_worker_args,
    prediction_with_oom_retry,
    score_prediction,
    validate_run_args,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class UI5TrainRolloutTest(unittest.TestCase):
    @staticmethod
    def write_checkpoint(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for filename in preflight.REQUIRED_CHECKPOINT_CODE:
            (path / filename).write_text("CHECKPOINT_SOURCE = True\n", encoding="utf-8")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "locateanything",
                    "auto_map": {
                        "AutoConfig": (
                            "configuration_locateanything.LocateAnythingConfig"
                        ),
                        "AutoModel": "modeling_locateanything.LocateAnythingModel",
                    },
                }
            ),
            encoding="utf-8",
        )
        (path / "model.safetensors").write_bytes(b"synthetic-nonempty-shard")

    def test_formal_worker_validation_and_launcher_contract(self) -> None:
        worker_args = parse_worker_args(
            [
                "--output-root",
                "output",
                "--model-id",
                "m31",
                "--checkpoint",
                "checkpoint",
                "--processor-path",
                "processor",
                "--bundle-root",
                "bundle",
                "--rollout-ids",
                "0",
                "--seeds",
                "20260903",
                "--physical-gpu",
                "0",
                "--gpu-model-processes",
                "4",
            ]
        )
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            validate_run_args(worker_args)
            worker_args.gpu_model_processes = 2
            with self.assertRaisesRegex(ValueError, "four model processes per GPU"):
                validate_run_args(worker_args)
            worker_args.gpu_model_processes = 4
            worker_args.max_new_tokens = 7268
            with self.assertRaisesRegex(ValueError, "token configuration mismatch"):
                validate_run_args(worker_args)

        launcher = (
            PROJECT_ROOT / "shell" / "run_ui5_train_rollouts_h20x2.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("launch_worker m31"), 1)
        self.assertEqual(launcher.count("launch_worker crop"), 1)
        self.assertEqual(launcher.count("for rollout_id in 0 1 2 3; do"), 2)
        self.assertIn(
            'launch_worker m31 0 "${rollout_id}" "${SEEDS[rollout_id]}"',
            launcher,
        )
        self.assertIn(
            'launch_worker crop 1 "${rollout_id}" "${SEEDS[rollout_id]}"',
            launcher,
        )
        self.assertEqual(tuple(FORMAL_SEEDS.values()), (20260903, 20260917, 20260931, 20260947))
        self.assertIn("--gpu-model-processes 4", launcher)
        self.assertIn('HF_MODULES_CACHE="${hf_modules_cache}"', launcher)
        self.assertIn('PYTHONPYCACHEPREFIX="${python_pycache}"', launcher)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", launcher)
        self.assertIn("runtime_cache/hf_modules/${worker_key}", launcher)
        self.assertIn("runtime_cache/pycache/${worker_key}", launcher)
        self.assertNotIn(
            "/home/tiger/.cache/huggingface/modules/transformers_modules/"
            "checkpoint_hyphen_12000",
            launcher,
        )
        for argument in (
            "--max-seq-length 7268",
            "--max-num-tokens-per-sample 7268",
            "--training-max-num-tokens 12800",
            "--processor-in-token-limit 25600",
            "--max-new-tokens 512",
            "--n-future-tokens 6",
            "--attn-implementation sdpa",
            "--vision-attn-implementation flash_attention_2",
            "--temperature 0.7",
            "--top-p 0.9",
            "--top-k 0",
            "--repetition-penalty 1.1",
        ):
            self.assertIn(argument, launcher)
        for backend_status in (
            "text_config=sdpa",
            "vision_config=flash_attention_2",
            "vision_first_layer=flash_attention_2",
            "vision_blocks=27/27",
        ):
            self.assertIn(backend_status, launcher)
        self.assertIn("physical_processes=8", launcher)
        self.assertIn("unique=8", launcher)
        self.assertIn("Embodied-rollout8-h20x2-v6", launcher)
        self.assertIn("ui5-train-rollout8-h20x2-v6-20260904", launcher)
        self.assertNotIn("ui5-train-rollout8-h20x2-v5-20260904", launcher)

    def test_formal_model_load_gate_rejects_ok_log_for_dead_pid(self) -> None:
        launcher = (
            PROJECT_ROOT / "shell" / "run_ui5_train_rollouts_h20x2.sh"
        ).read_text(encoding="utf-8")
        marker = "<<'PY'\n"
        start = launcher.index(marker) + len(marker)
        gate = launcher[start : launcher.index("\nPY\nthen", start)]

        dead_pid = 99_999_999
        with self.assertRaises(OSError):
            os.kill(dead_pid, 0)
        ownership = tuple(
            (model, rollout_id)
            for model in ("m31", "crop")
            for rollout_id in range(4)
        )
        required_attention = (
            "text_config=sdpa vision_config=flash_attention_2 "
            "vision_first_layer=flash_attention_2 vision_blocks=27/27"
        )
        def exercise_gate(pids: list[int]) -> tuple[subprocess.CompletedProcess[str], dict, bool]:
            self.assertEqual(len(pids), 8)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "diagnostics").mkdir()
                arguments: list[str] = [str(root)]
                for index, ((model, rollout_id), pid) in enumerate(
                    zip(ownership, pids)
                ):
                    gpu = 0 if model == "m31" else 1
                    log_path = root / f"worker-{index}.log"
                    log_path.write_text(
                        f"[MODEL_LOAD_OK] model={model} gpu={gpu} pid={pid} "
                        f"rollouts={rollout_id} {required_attention}\n",
                        encoding="utf-8",
                    )
                    arguments.extend(
                        (str(pid), model, str(rollout_id), str(log_path))
                    )
                result = subprocess.run(
                    [sys.executable, "-", *arguments],
                    input=gate,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                status_name = (
                    "formal_run_valid.json"
                    if result.returncode == 0
                    else "formal_run_invalid.json"
                )
                payload = json.loads(
                    (root / "diagnostics" / status_name).read_text(encoding="utf-8")
                )
                marker_exists = (
                    root / "diagnostics" / "_MODEL_LOADS_OK"
                ).is_file()
                return result, payload, marker_exists

        children = [
            subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(7)
        ]
        try:
            live_pids = [os.getpid(), *(process.pid for process in children)]
            result, invalid, marker_exists = exercise_gate(
                [*live_pids[:7], dead_pid]
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(invalid["valid"])
            self.assertEqual(len(invalid["workers"]), 8)
            self.assertTrue(all(row["alive"] for row in invalid["workers"][:7]))
            self.assertFalse(invalid["workers"][7]["alive"])
            self.assertFalse(invalid["workers"][7]["validated"])
            self.assertFalse(marker_exists)

            result, duplicate, marker_exists = exercise_gate(
                [live_pids[0]] * 8
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(duplicate["valid"])
            self.assertEqual(duplicate["unique_pid_count"], 1)
            self.assertFalse(marker_exists)

            result, valid, marker_exists = exercise_gate(live_pids)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(valid["valid"])
            self.assertEqual(valid["unique_pid_count"], 8)
            self.assertTrue(all(row["alive"] for row in valid["workers"]))
            self.assertTrue(marker_exists)
        finally:
            for process in children:
                process.terminate()
            for process in children:
                process.wait(timeout=10)

    def test_model_load_barrier_precedes_resume_completion_and_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            marker = output_root / "diagnostics" / "_MODEL_LOADS_OK"

            def release_marker(_: float) -> None:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("released\n", encoding="utf-8")

            with mock.patch.object(
                worker.time, "sleep", side_effect=release_marker
            ) as sleep_mock, mock.patch("builtins.print") as print_mock:
                report = worker.wait_for_model_load_barrier(
                    output_root,
                    model_id="m31",
                    physical_gpu=0,
                    rollout_ids=(0,),
                )
            sleep_mock.assert_called_once_with(1.0)
            self.assertEqual(report["marker"], str(marker))
            messages = [str(call.args[0]) for call in print_mock.call_args_list]
            self.assertTrue(any("[MODEL_LOAD_BARRIER_WAIT]" in row for row in messages))
            self.assertTrue(
                any("[MODEL_LOAD_BARRIER_RELEASE]" in row for row in messages)
            )

        worker_source = (
            PROJECT_ROOT / "scripts" / "run_ui5_train_rollout_worker.py"
        ).read_text(encoding="utf-8")
        load_ok = worker_source.index('"[MODEL_LOAD_OK] "')
        barrier = worker_source.index("wait_for_model_load_barrier(", load_ok)
        resume_completion = worker_source.index("already_complete =", barrier)
        sample_inference = worker_source.index(
            'load_stage = "sample_major_inference"', barrier
        )
        self.assertLess(load_ok, barrier)
        self.assertLess(barrier, resume_completion)
        self.assertLess(barrier, sample_inference)

    def test_attention_audit_resume_and_output_preflight(self) -> None:
        flash = "flash_attention_2"
        blocks = [SimpleNamespace(attn_implementation=flash) for _ in range(27)]
        inferencer = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    text_config=SimpleNamespace(_attn_implementation="sdpa"),
                    vision_config=SimpleNamespace(_attn_implementation=flash),
                ),
                vision_model=SimpleNamespace(
                    encoder=SimpleNamespace(blocks=blocks)
                ),
            )
        )
        attention_args = SimpleNamespace(
            attn_implementation="sdpa",
            vision_attn_implementation=flash,
        )
        report = worker.verify_loaded_attention_backends(inferencer, attention_args)
        self.assertEqual(report["text_config"], "sdpa")
        self.assertEqual(report["vision_config"], flash)
        self.assertEqual(report["vision_first_layer"], flash)
        self.assertEqual(report["vision_blocks"], "27/27")
        blocks[-1].attn_implementation = "sdpa"
        with self.assertRaisesRegex(RuntimeError, "vision_blocks=26/27"):
            worker.verify_loaded_attention_backends(inferencer, attention_args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "output" / "raw" / "m31" / "rollout_0"
            checkpoint = root / "checkpoint"
            processor = root / "processor"
            generation = {"mode": "hybrid", "vision_attn_implementation": flash}
            sample = {
                "record_id": "record_0",
                "sample_id": "sample_0",
                "task": "occlusion",
                "source_image_id": "image_0",
                "image_relpath": "images/image_0.png",
            }
            prior = {
                "schema_version": worker.SCHEMA_VERSION,
                "model_id": "m31",
                "rollout_id": 0,
                "seed": FORMAL_SEEDS[0],
                "checkpoint": str(checkpoint),
                "processor_path": str(processor),
                "generation_config": generation,
                "git_commit": "model-head",
                "baseline_git_commit": "5d7a313",
                "worker_git_commit": "worker-head",
                **sample,
                "inference_success": True,
                "runtime_error": None,
                "parse_status": "ok",
                "contains_crop_parse_error": False,
                "oom_events": 1,
                "oom_recovered": True,
                "oom_final_failure": False,
            }
            write_jsonl(raw_dir / "part-00000.jsonl", [prior])
            completed, counters = worker.resume_route_state(
                raw_dir,
                model_id="m31",
                rollout_id=0,
                seed=FORMAL_SEEDS[0],
                checkpoint=checkpoint,
                processor_path=processor,
                generation=generation,
                git_commit="model-head",
                baseline_git_commit="5d7a313",
                worker_git_commit="worker-head",
                samples_by_record={"record_0": sample},
            )
            self.assertEqual(completed, {"record_0"})
            self.assertEqual(counters["attempted"], 1)
            self.assertEqual(counters["inference_success"], 1)
            self.assertEqual(counters["oom_recovered_samples"], 1)
            writer = worker.PartWriter(raw_dir, part_size=10)
            writer.write({"new": True})
            writer.close()
            self.assertTrue((raw_dir / "part-00001.jsonl").is_file())

            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                worker.resume_route_state(
                    raw_dir,
                    model_id="m31",
                    rollout_id=0,
                    seed=FORMAL_SEEDS[0],
                    checkpoint=checkpoint,
                    processor_path=processor,
                    generation={"mode": "changed"},
                    git_commit="model-head",
                    baseline_git_commit="5d7a313",
                    worker_git_commit="worker-head",
                    samples_by_record={"record_0": sample},
                )

            output = root / "resumable_output"
            (output / "raw").mkdir(parents=True)
            output_report = preflight.check_output_root(output)
            self.assertTrue(output_report["complete"])
            self.assertFalse(output_report["fresh"])
            self.assertTrue(output_report["resume"])
            (output / "unrelated.payload").write_text("no", encoding="utf-8")
            rejected = preflight.check_output_root(output)
            self.assertFalse(rejected["complete"])
            self.assertIn("unrelated.payload", rejected["unexpected_entries"])

    def test_resume_repairs_only_unterminated_highest_eof_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            processor = root / "processor"
            diagnostics = root / "diagnostics"
            generation = {"mode": "hybrid"}
            sample = {
                "record_id": "record_0",
                "sample_id": "sample_0",
                "task": "occlusion",
                "source_image_id": "image_0",
                "image_relpath": "images/image_0.png",
            }
            prior = {
                "schema_version": worker.SCHEMA_VERSION,
                "model_id": "m31",
                "rollout_id": 0,
                "seed": FORMAL_SEEDS[0],
                "checkpoint": str(checkpoint),
                "processor_path": str(processor),
                "generation_config": generation,
                "git_commit": "model-head",
                "baseline_git_commit": "5d7a313",
                "worker_git_commit": "worker-head",
                **sample,
                "inference_success": True,
                "runtime_error": None,
                "parse_status": "ok",
                "oom_events": 0,
            }
            valid_raw = (json.dumps(prior) + "\n").encode("utf-8")
            raw_dir = root / "raw"
            raw_dir.mkdir()
            highest = raw_dir / "part-00000.jsonl"
            highest.write_bytes(valid_raw + b'{"schema_version":5')
            resume_kwargs = {
                "model_id": "m31",
                "rollout_id": 0,
                "seed": FORMAL_SEEDS[0],
                "checkpoint": checkpoint,
                "processor_path": processor,
                "generation": generation,
                "git_commit": "model-head",
                "baseline_git_commit": "5d7a313",
                "worker_git_commit": "worker-head",
                "samples_by_record": {"record_0": sample},
                "recovery_diagnostics_dir": diagnostics,
            }
            completed, counters = worker.resume_route_state(raw_dir, **resume_kwargs)
            self.assertEqual(completed, {"record_0"})
            self.assertEqual(counters["attempted"], 1)
            self.assertEqual(highest.read_bytes(), valid_raw)

            valid_unterminated_dir = root / "valid_unterminated"
            valid_unterminated_dir.mkdir()
            valid_unterminated = valid_unterminated_dir / "part-00000.jsonl"
            valid_unterminated.write_bytes(valid_raw.rstrip(b"\n"))
            normalized, _ = worker.resume_route_state(
                valid_unterminated_dir, **resume_kwargs
            )
            self.assertEqual(normalized, {"record_0"})
            self.assertEqual(valid_unterminated.read_bytes(), valid_raw)

            progress_path = root / "progress.jsonl"
            progress_row = {
                "model_id": "m31",
                "rollout_id": 0,
                "seed": FORMAL_SEEDS[0],
                "total": 2,
                "attempted": 1,
                "elapsed_seconds": 5.0,
                "inference_elapsed_seconds": 4.0,
            }
            valid_progress = (json.dumps(progress_row) + "\n").encode("utf-8")
            progress_path.write_bytes(valid_progress + b'{"attempted":')
            progress = worker.ProgressWriter(
                progress_path,
                "m31",
                0,
                FORMAL_SEEDS[0],
                2,
                resume_attempted=1,
                recovery_diagnostics_dir=diagnostics,
            )
            progress.close()
            self.assertEqual(progress_path.read_bytes(), valid_progress)
            events = [json.loads(path.read_text(encoding="utf-8")) for path in diagnostics.glob("*.json")]
            self.assertEqual(len(events), 3)
            self.assertEqual(
                sum(
                    event["action"]
                    == "truncate_incomplete_unterminated_eof_fragment"
                    for event in events
                ),
                2,
            )
            self.assertEqual(
                sum(
                    event["action"]
                    == "append_missing_newline_after_valid_eof_record"
                    for event in events
                ),
                1,
            )
            truncated = [event for event in events if event["removed_bytes"] > 0]
            self.assertEqual(len(truncated), 2)

            nonhighest = root / "nonhighest"
            nonhighest.mkdir()
            (nonhighest / "part-00000.jsonl").write_bytes(
                valid_raw + b'{"schema_version":5'
            )
            (nonhighest / "part-00001.jsonl").write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "not recoverable|invalid JSONL"):
                worker.resume_route_state(nonhighest, **resume_kwargs)

            terminated_bad = root / "terminated_bad"
            terminated_bad.mkdir()
            (terminated_bad / "part-00000.jsonl").write_bytes(
                valid_raw + b'{"schema_version":5\n'
            )
            with self.assertRaisesRegex(RuntimeError, "invalid JSONL"):
                worker.resume_route_state(terminated_bad, **resume_kwargs)

            bad_progress = root / "bad_progress.jsonl"
            bad_progress.write_bytes(valid_progress + b'{"attempted":\n')
            with self.assertRaisesRegex(RuntimeError, "invalid JSONL"):
                worker.ProgressWriter(
                    bad_progress,
                    "m31",
                    0,
                    FORMAL_SEEDS[0],
                    2,
                    resume_attempted=1,
                    recovery_diagnostics_dir=diagnostics,
                )

    def test_checkpoint_source_syntax_and_exact_copy_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid"
            self.write_checkpoint(valid)
            report = preflight.check_checkpoint(valid)
            self.assertTrue(report["complete"])
            self.assertTrue(report["checkpoint_code_complete"])
            self.assertEqual(
                [item["filename"] for item in report["checkpoint_code"]],
                list(preflight.REQUIRED_CHECKPOINT_CODE),
            )
            self.assertTrue(all(item["syntax_ok"] for item in report["checkpoint_code"]))

            empty = root / "empty"
            shutil.copytree(valid, empty)
            (empty / "relation_modules.py").write_bytes(b"")
            empty_report = preflight.check_checkpoint(empty)
            self.assertFalse(empty_report["complete"])
            empty_source = next(
                row
                for row in empty_report["checkpoint_code"]
                if row["filename"] == "relation_modules.py"
            )
            self.assertTrue(empty_source["exists"])
            self.assertFalse(empty_source["nonzero"])
            self.assertIn("required checkpoint source is empty", "\n".join(empty_report["errors"]))

            broken = root / "broken"
            shutil.copytree(valid, broken)
            (broken / "modeling_locateanything.py").write_text(
                "def broken(:\n    pass\n", encoding="utf-8"
            )
            broken_report = preflight.check_checkpoint(broken)
            broken_source = next(
                row
                for row in broken_report["checkpoint_code"]
                if row["filename"] == "modeling_locateanything.py"
            )
            self.assertFalse(broken_report["complete"])
            self.assertEqual(broken_source["syntax_error"]["python_type"], "SyntaxError")
            self.assertIsNotNone(broken_source["syntax_error"]["line"])
            commands = preflight.copy_commands(
                broken_report,
                report,
                [{"complete": True}],
                Path(preflight.H20_BUNDLE),
            )
            self.assertNotIn(
                f"{preflight.A800_M31_SOURCE}/modeling_locateanything.py",
                commands,
            )
            self.assertIn("existing config/code is not overwritten", commands)

            missing_m31 = root / "missing_m31_source"
            shutil.copytree(valid, missing_m31)
            (missing_m31 / "relation_modules.py").unlink()
            m31_report = preflight.check_checkpoint(missing_m31)
            missing_crop = root / "missing_crop_source"
            shutil.copytree(valid, missing_crop)
            (missing_crop / "configuration_locateanything.py").unlink()
            crop_report = preflight.check_checkpoint(missing_crop)
            commands = preflight.copy_commands(
                m31_report,
                crop_report,
                [{"complete": True}],
                Path(preflight.H20_BUNDLE),
            )
            self.assertIn(
                f"{preflight.A800_M31_SOURCE}/relation_modules.py", commands
            )
            self.assertIn(
                f"{preflight.A800_CROP_SOURCE}/configuration_locateanything.py",
                commands,
            )
            self.assertNotIn(
                f"{preflight.A800_CROP_SOURCE}/relation_modules.py", commands
            )
            self.assertNotIn(
                f"{preflight.A800_M31_SOURCE}/configuration_locateanything.py",
                commands,
            )

    def test_oom_retry_recovery_and_final_failure(self) -> None:
        class FakeOOM(RuntimeError):
            pass

        class FakeCuda:
            OutOfMemoryError = FakeOOM

            def __init__(self) -> None:
                self.empty_cache_calls = 0

            @staticmethod
            def mem_get_info():
                return 8 * 1024**3, 80 * 1024**3

            @staticmethod
            def memory_allocated():
                return 10 * 1024**3

            @staticmethod
            def memory_reserved():
                return 12 * 1024**3

            def empty_cache(self):
                self.empty_cache_calls += 1

        fake_torch = SimpleNamespace(cuda=FakeCuda())
        inferencer = SimpleNamespace(
            active_rollout_context={"stale": True},
            last_rollout_token_usage={"stale": True},
        )
        calls: list[int] = []

        def recovers(context):
            calls.append(20260903)
            context.update(
                {
                    "stage": "crop_generate",
                    "crop_id": "crop_2",
                    "crop_index": 2,
                    "crop_xyxy": [0, 100, 200, 300],
                    "input_tokens": 7000,
                    "tile_count": 4,
                    "tile_size": {"width": 200, "height": 200},
                    "memory_before_oom": {"allocated_gib": 10.0},
                }
            )
            if len(calls) == 1:
                raise FakeOOM("CUDA out of memory on first attempt")
            return {"pred_global": [[1, 2, 3, 4]]}

        recovered = prediction_with_oom_retry(
            recovers, torch=fake_torch, inferencer=inferencer
        )
        self.assertEqual(calls, [20260903, 20260903])
        self.assertTrue(recovered["oom_recovered"])
        self.assertEqual(recovered["oom_events"], 1)
        self.assertEqual(fake_torch.cuda.empty_cache_calls, 1)
        retry = recovered["oom_retry"]
        self.assertEqual(retry["first_attempt"]["context"]["crop_id"], "crop_2")
        self.assertEqual(retry["first_attempt"]["context"]["input_tokens"], 7000)
        self.assertEqual(retry["first_attempt"]["exception"]["python_type"], "FakeOOM")
        self.assertIn("first attempt", retry["first_attempt"]["exception"]["traceback"])
        self.assertEqual(retry["retry_attempt"]["status"], "success")

        failed_calls = 0

        def always_oom(context):
            nonlocal failed_calls
            failed_calls += 1
            context.update(
                {
                    "stage": "full_image_generate",
                    "crop_id": "full_image",
                    "crop_index": None,
                    "crop_xyxy": [0, 0, 100, 100],
                    "input_tokens": 7100,
                    "tile_count": 1,
                    "tile_size": {"width": 100, "height": 100},
                    "memory_before_oom": {"reserved_gib": 12.0},
                }
            )
            raise FakeOOM(f"CUDA out of memory attempt {failed_calls}")

        with self.assertRaises(OOMRetryFailure) as caught:
            prediction_with_oom_retry(
                always_oom, torch=fake_torch, inferencer=inferencer
            )
        diagnostics = caught.exception.oom_diagnostics
        self.assertEqual(failed_calls, 2)
        self.assertEqual(diagnostics["oom_events"], 2)
        self.assertTrue(diagnostics["oom_final_failure"])
        self.assertEqual(diagnostics["first_attempt"]["context"]["input_tokens"], 7100)
        self.assertEqual(diagnostics["retry_attempt"]["status"], "oom")
        self.assertIn("attempt 2", diagnostics["retry_attempt"]["exception"]["message"])
        self.assertEqual(fake_torch.cuda.empty_cache_calls, 3)
        self.assertIsNone(inferencer.active_rollout_context)
        self.assertIsNone(inferencer.last_rollout_token_usage)

    def test_no_jq_oom_summary_deduplicates_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            common = {
                "record_id": "same_record",
                "sample_id": "same_sample",
                "source_image_id": "same_image",
                "task": "cropping",
                "image_relpath": "images/same.png",
            }
            write_jsonl(
                output / "raw" / "m31" / "rollout_0" / "part-00000.jsonl",
                [
                    {
                        **common,
                        "model_id": "m31",
                        "rollout_id": 0,
                        "oom_events": 1,
                        "oom_recovered": True,
                        "oom_final_failure": False,
                        "oom_retry": {"retry_attempt": {"status": "success"}},
                        "runtime_error": None,
                    }
                ],
            )
            write_jsonl(
                output / "raw" / "crop" / "rollout_3" / "part-00000.jsonl",
                [
                    {
                        **common,
                        "model_id": "crop",
                        "rollout_id": 3,
                        "oom_events": 2,
                        "oom_recovered": False,
                        "oom_final_failure": True,
                        "oom_retry": {"retry_attempt": {"status": "oom"}},
                        "runtime_error": {"type": "CUDA_OOM"},
                    },
                    {
                        **common,
                        "record_id": "no_oom",
                        "model_id": "crop",
                        "rollout_id": 3,
                        "oom_events": 0,
                        "oom_recovered": False,
                        "oom_final_failure": False,
                    },
                ],
            )
            result = oom_summary.run(output)
            self.assertEqual(result["raw_records_scanned"], 3)
            self.assertEqual(result["oom_total_count"], 3)
            self.assertEqual(result["oom_recovered_count"], 1)
            self.assertEqual(result["oom_final_failed_count"], 1)
            self.assertEqual(result["oom_affected_rollout_records"], 2)
            self.assertEqual(result["unique_oom_record_ids"], 1)
            self.assertEqual(result["unique_oom_samples"][0]["oom_events"], 3)
            self.assertTrue((output / "reports" / "oom_summary.json").is_file())
            affected = snapshot.read_jsonl(
                output / "reports" / "oom_affected_rollouts.jsonl"
            )
            unique = snapshot.read_jsonl(output / "selection" / "oom_samples.jsonl")
            self.assertEqual(len(affected), 2)
            self.assertEqual(len(unique), 1)

    def test_progress_snapshot_keeps_failed_physical_worker_and_invalidates_eta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            completed = {
                "status": "completed",
                "attempted": 5,
                "total": 5,
                "throughput_attempted_per_second": 2.0,
            }
            running = {
                "status": "running",
                "attempted": 2,
                "total": 5,
                "throughput_attempted_per_second": 1.0,
            }
            write_jsonl(
                output / "progress" / "m31" / "rollout_0.jsonl",
                [{**completed, "status": "running", "attempted": 3}, completed],
            )
            write_jsonl(
                output / "progress" / "m31" / "rollout_1.jsonl",
                [running],
            )
            write_jsonl(
                output / "progress" / "crop" / "rollout_0.jsonl",
                [
                    {**running, "attempted": 0},
                    {**running, "attempted": 1},
                    running,
                ],
            )
            args = parse_worker_args(
                [
                    "--mode",
                    "progress-snapshot",
                    "--output-root",
                    str(output),
                    "--expected-workers",
                    "8",
                    "--physical-worker",
                    "m31,0,111,0",
                    "--physical-worker",
                    "m31,0,112,1",
                    "--physical-worker",
                    "m31,0,113,2",
                    "--physical-worker",
                    "m31,0,114,3",
                    "--physical-worker",
                    "crop,1,221,0",
                    "--physical-worker",
                    "crop,1,222,1",
                    "--physical-worker",
                    "crop,1,223,2",
                    "--physical-worker",
                    "crop,1,224,3",
                ]
            )
            with mock.patch.object(worker, "pid_alive", side_effect=lambda pid: pid != 112):
                self.assertEqual(worker.progress_snapshot(args), 0)

            snapshots = snapshot.read_jsonl(output / "progress" / "total_eta.jsonl")
            self.assertEqual(len(snapshots), 1)
            status = snapshots[0]
            self.assertEqual(status["physical_processes_expected"], 8)
            self.assertEqual(status["physical_processes_seen"], 8)
            self.assertTrue(status["physical_pids_unique"])
            self.assertEqual(len(status["physical_processes"]), 8)
            self.assertEqual(status["logical_rollouts_expected"], 8)
            self.assertEqual(len(status["logical_rollouts"]), 8)
            self.assertEqual(status["progress_files_discovered"], 3)
            self.assertEqual(status["progress_history_rows_total"], 6)
            self.assertEqual(status["workers_seen"], 3)

            by_pid = {row["pid"]: row for row in status["physical_processes"]}
            self.assertEqual(
                set(by_pid), {111, 112, 113, 114, 221, 222, 223, 224}
            )
            self.assertEqual(by_pid[112]["status"], "failed")
            self.assertFalse(by_pid[112]["alive"])
            self.assertEqual(by_pid[111]["status"], "alive")
            self.assertTrue(by_pid[111]["alive"])
            self.assertEqual(by_pid[221]["status"], "alive")
            self.assertTrue(by_pid[221]["alive"])
            self.assertEqual(
                by_pid[112]["logical_rollout_statuses"],
                {"1": "failed"},
            )
            self.assertEqual(by_pid[221]["logical_rollout_statuses"]["0"], "running")
            self.assertEqual(status["failed_physical_processes"], 1)
            self.assertEqual(status["failed_logical_rollouts"], 1)
            self.assertFalse(status["eta_valid"])
            self.assertEqual(
                status["eta_unavailable_reason"],
                "failed physical worker or logical rollout",
            )
            self.assertIsNone(status["total_remaining_seconds"])
            self.assertIsNone(status["total_estimated_completion"])

    def test_progress_snapshot_uses_maximum_of_eight_parallel_etas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            route_states = {
                ("m31", 0): (2, 2.0),
                ("m31", 1): (5, 1.0),
                ("m31", 2): (9, 0.5),
                ("m31", 3): (10, 2.0),
                ("crop", 0): (1, 1.0),
                ("crop", 1): (6, 2.0),
                ("crop", 2): (4, 3.0),
                ("crop", 3): (8, 0.5),
            }
            for (model, rollout_id), (attempted, rate) in route_states.items():
                write_jsonl(
                    output / "progress" / model / f"rollout_{rollout_id}.jsonl",
                    [
                        {
                            "status": "completed" if attempted == 10 else "running",
                            "attempted": attempted,
                            "total": 10,
                            "throughput_attempted_per_second": rate,
                        }
                    ],
                )
            mappings = [
                f"m31,0,{111 + rollout_id},{rollout_id}"
                for rollout_id in range(4)
            ] + [
                f"crop,1,{221 + rollout_id},{rollout_id}"
                for rollout_id in range(4)
            ]
            argv = [
                "--mode",
                "progress-snapshot",
                "--output-root",
                str(output),
                "--expected-workers",
                "8",
            ]
            for mapping in mappings:
                argv.extend(("--physical-worker", mapping))
            args = parse_worker_args(argv)
            with mock.patch.object(worker, "pid_alive", return_value=True):
                self.assertEqual(worker.progress_snapshot(args), 0)

            status = snapshot.read_jsonl(output / "progress" / "total_eta.jsonl")[-1]
            # crop rollout 0 is the straggler: (10 - 1) / 1.0 = 9 seconds.
            self.assertTrue(status["eta_valid"])
            self.assertEqual(status["total_remaining_seconds"], 9.0)
            self.assertIn("maximum remaining time", status["eta_basis"])
            per_process = [
                row["remaining_seconds"] for row in status["physical_processes"]
            ]
            self.assertEqual(status["total_remaining_seconds"], max(per_process))
            self.assertNotEqual(
                status["total_remaining_seconds"], sum(per_process)
            )

    def test_fixed_interleave_and_effective_generation_budget(self) -> None:
        tasks = (
            "occlusion",
            "cropping",
            "text_overflow",
            "text_ellipsis",
            "content_missing",
        )
        unordered = [
            {
                "record_id": f"{task}_{polarity}",
                "sample_id": f"{task}_{polarity}",
                "task": task,
                "positive": polarity == "positive",
            }
            for polarity in ("negative", "positive")
            for task in reversed(tasks)
        ]
        ordered = fixed_interleaved_samples(unordered)
        self.assertEqual(
            [row["record_id"] for row in ordered],
            [
                f"{task}_{polarity}"
                for task in tasks
                for polarity in ("positive", "negative")
            ],
        )

        class FakeModel:
            def generate(self, **inputs):
                return inputs["max_new_tokens"]

        inferencer = SimpleNamespace(
            processor=SimpleNamespace(in_token_limit=PROCESSOR_IN_TOKEN_LIMIT),
            model=FakeModel(),
        )
        args = SimpleNamespace(
            processor_in_token_limit=PROCESSOR_IN_TOKEN_LIMIT,
            max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        install_generation_token_budget(inferencer, args)
        self.assertEqual(
            inferencer.model.generate(input_ids=np.zeros((1, 100), dtype=np.int64)),
            512,
        )
        self.assertEqual(
            inferencer.model.generate(input_ids=np.zeros((1, 7000), dtype=np.int64)),
            268,
        )
        self.assertEqual(
            inferencer.last_rollout_token_usage["input_plus_generation_limit"],
            MAX_SEQ_LENGTH,
        )
        with self.assertRaisesRegex(RuntimeError, "input exceeds MAX_SEQ_LENGTH"):
            inferencer.model.generate(
                input_ids=np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int64)
            )
        self.assertEqual(MAX_NUM_TOKENS_PER_SAMPLE, 7268)
        self.assertEqual(TRAINING_MAX_NUM_TOKENS, 12800)

    def build_bundle(self, root: Path) -> Path:
        full = root / "full"
        audit = root / "audit"
        crop = audit / "crop"
        output = root / "bundle"
        full.mkdir(parents=True)
        image_path = root / "source.png"
        Image.new("RGB", (100, 100), "white").save(image_path)
        image_id = "img_test"
        labels = {
            "occlusion": "overlapping elements",
            "cropping": "cropped element",
            "text_overflow": "text overflow",
            "text_ellipsis": "abnormal text ellipsis",
            "content_missing": "missing content",
        }
        task_samples = []
        task_aware = []
        for task, label in labels.items():
            pixel_gt = (
                [10, 10, 30, 30]
                if task == "text_ellipsis"
                else [10, 40, 30, 60]
            )
            normalized_gt = (
                [100, 100, 300, 300]
                if task == "text_ellipsis"
                else [100, 400, 300, 600]
            )
            source_path = full / f"ui_{task}_train.jsonl"
            original = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Locate all the instances that match the following description: {label}.",
                    },
                    {
                        "from": "gpt",
                        "value": (
                            f"<ref>{label}</ref><box>"
                            f"<{normalized_gt[0]}><{normalized_gt[1]}>"
                            f"<{normalized_gt[2]}><{normalized_gt[3]}></box>"
                        ),
                    },
                ],
                "image": str(image_path),
            }
            write_jsonl(source_path, [original])
            sample_id = f"sample_{task}"
            task_samples.append(
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "task": f"ui_{task}",
                    "width": 100,
                    "height": 100,
                    "gt_boxes": [pixel_gt],
                    "gt_boxes_1000": [normalized_gt],
                    "source_records": [
                        {"source_file": str(source_path), "line_no": 1}
                    ],
                    "same_task_polarity_conflict": False,
                }
            )
            if task != "text_overflow":
                task_aware.append(
                    {
                        "sample_id": sample_id,
                        "image_id": image_id,
                        "task": f"ui_{task}",
                        "base_tiles": [[0, 0, 100, 50], [0, 50, 100, 100]],
                        "final_tiles": [[0, 0, 100, 100]],
                        "removed_gt_crossing_seams": [50],
                    }
                )
        write_jsonl(
            audit / "manifest" / "unique_images.jsonl",
            [
                {
                    "image_id": image_id,
                    "content_id": "bytes",
                    "image_path": str(image_path),
                    "canonical_paths": [str(image_path)],
                    "basename": image_path.name,
                    "width": 100,
                    "height": 100,
                    "tasks": [f"ui_{task}" for task in labels],
                }
            ],
        )
        write_jsonl(
            audit / "manifest" / "task_samples.jsonl",
            task_samples,
        )
        crop.mkdir(parents=True)
        write_jsonl(
            crop.parent / "excluded_training_samples.jsonl",
            [
                {
                    "sample_id": "sample_text_overflow",
                    "image_id": image_id,
                    "task": "ui_text_overflow",
                    "reason": "annotation_error",
                }
            ],
        )
        (crop / "base_scan_plans.json").write_text(
            json.dumps(
                {
                    image_id: {
                        "tiles": [[0, 0, 100, 50], [0, 50, 100, 100]],
                        "horizontal_seams": [50],
                    }
                }
            ),
            encoding="utf-8",
        )
        # These training-only fields exist in the source audit but must not
        # reach the portable rollout plan.
        write_jsonl(
            crop / "task_aware_manifest.jsonl",
            task_aware,
        )
        detector_path = audit / "detections" / "merged" / "detections.jsonl"
        write_jsonl(detector_path, [{"image_id": image_id}])
        digest = hashlib.blake2b(detector_path.read_bytes(), digest_size=16).hexdigest()
        (crop / "summary.json").write_text(
            json.dumps(
                {"input_state": {"detections_digest": "blake2b128:" + digest}}
            ),
            encoding="utf-8",
        )
        # Simulate the real failure mode: image copying completed, then the
        # sample pass stopped and left a partial output directory behind.
        (output / "images").mkdir(parents=True)
        (output / "manifest").mkdir(parents=True)
        shutil.copy2(image_path, output / "images" / f"{image_id}.png")
        summary = prepare.build(
            SimpleNamespace(
                full_data=full,
                audit_root=audit,
                crop_root=crop,
                output_dir=output,
            )
        )
        self.assertEqual(summary["pipeline_coverage_failures"], 3)
        self.assertEqual(summary["registered_annotation_exclusions"], 1)
        samples = prepare.read_jsonl(output / "manifest" / "task_samples.jsonl")
        excluded = next(row for row in samples if row["task"] == "text_overflow")
        self.assertTrue(excluded["annotation_anomaly"])
        self.assertFalse(excluded["grpo_eligible"])
        portable = json.loads((output / "base_scan_plans.json").read_text())
        self.assertIn("base_tiles", portable[image_id])
        self.assertNotIn("final_tiles", portable[image_id])
        self.assertFalse(any((output / "images").glob("*__y*.png")))
        return output

    def test_bundle_preflight_scoring_aggregate_and_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.build_bundle(root)
            summary, status = preflight.run(
                SimpleNamespace(
                    bundle_root=bundle,
                    diagnostics_dir=root / "diagnostics",
                    m31_checkpoint=root / "missing_m31",
                    crop_checkpoint=root / "missing_crop",
                    processor_candidate=[root / "missing_processor"],
                    m31_repo=PROJECT_ROOT,
                    crop_repo=PROJECT_ROOT,
                    require_runtime=False,
                )
            )
            self.assertEqual(status, 0)
            self.assertTrue(summary["bundle"]["complete"])
            self.assertTrue((root / "diagnostics" / "nastk_copy_commands.sh").is_file())

            # The Codex desktop's small CPU runtime omits SciPy.  Install a
            # tiny exact assignment stub only in this unit-test process; the
            # production worker still imports the formal scorer's SciPy
            # linear_sum_assignment and the H20 preflight requires SciPy.
            try:
                import scipy.optimize  # noqa: F401
            except ImportError:
                scipy_module = types.ModuleType("scipy")
                optimize_module = types.ModuleType("scipy.optimize")

                def linear_sum_assignment(cost: np.ndarray):
                    rows, cols = cost.shape
                    if rows <= cols:
                        best = min(
                            itertools.permutations(range(cols), rows),
                            key=lambda chosen: sum(cost[row, col] for row, col in enumerate(chosen)),
                        )
                        return np.arange(rows), np.array(best)
                    chosen_rows = min(
                        itertools.permutations(range(rows), cols),
                        key=lambda chosen: sum(cost[row, col] for col, row in enumerate(chosen)),
                    )
                    return np.array(chosen_rows), np.arange(cols)

                optimize_module.linear_sum_assignment = linear_sum_assignment
                scipy_module.optimize = optimize_module
                sys.modules["scipy"] = scipy_module
                sys.modules["scipy.optimize"] = optimize_module

            scorer = load_module(
                PROJECT_ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py",
                "test_ui5_formal_scorer",
            )
            wrong = score_prediction(
                scorer,
                [[10, 10, 30, 30]],
                [[70, 70, 90, 90]],
                "defect",
                0.1,
                (100, 100),
            )
            self.assertEqual(wrong["image_confusion"], "TP")
            self.assertEqual(wrong["error_type"], "LOC_WRONG")
            self.assertEqual((wrong["TP_box"], wrong["FP_box"], wrong["FN_box"]), (0, 1, 1))

            output = root / "rollouts"
            output.mkdir()
            samples = fixed_interleaved_samples(
                prepare.read_jsonl(bundle / "manifest" / "task_samples.jsonl")
            )
            first_snapshot, first_summary = snapshot.create_snapshot(
                output,
                bundle,
                kind="hourly",
                scheduled_hour=3,
                started_at_epoch=0.0,
                created_at_epoch=3 * 3600 + 1,
            )
            self.assertTrue(first_snapshot.name.startswith("hour_003_"))
            self.assertEqual(
                first_summary["difficulty_counts"],
                {
                    "easy": 0,
                    "medium": 0,
                    "hard": 0,
                },
            )
            self.assertEqual(first_summary["complete8_samples"], 0)
            self.assertEqual(first_summary["file_counts"]["complete8"], 0)
            self.assertEqual(
                first_summary["file_counts"]["incomplete_or_technical_error"], 5
            )
            self.assertEqual(first_summary["file_counts"]["delta_since_previous"], 0)
            self.assertTrue((first_snapshot / "snapshot_statistics.xlsx").is_file())
            self.assertTrue((first_snapshot / "visualizations" / "index.html").is_file())
            for model in ("m31", "crop"):
                for rollout in range(4):
                    raw_rows = []
                    for sample in samples:
                        task = sample["task"]
                        if task == "cropping":
                            correct = True
                        elif task == "occlusion":
                            correct = False
                        elif task == "content_missing":
                            correct = model == "m31"
                        else:
                            correct = rollout < (2 if model == "m31" else 1)
                        pred = sample["gt_global"] if correct else [[70, 70, 90, 90]]
                        score = score_prediction(
                            scorer,
                            sample["gt_global"],
                            pred,
                            "defect",
                            0.1,
                            (100, 100),
                        )
                        crop_outputs = []
                        if model == "crop":
                            crop_outputs = [
                                {
                                    "crop_id": crop_id,
                                    "crop_xyxy": (
                                        [0, 0, 100, 50]
                                        if index == 0
                                        else [0, 50, 100, 100]
                                    ),
                                    "gt_local": [],
                                    "raw_output": "<box>none</box>",
                                    "parse_status": "ok",
                                    "exact_correct": True,
                                }
                                for index, crop_id in enumerate(sample["crop_ids"])
                            ]
                        raw = {
                            "model_id": model,
                            "checkpoint": f"/{model}/checkpoint",
                            "git_commit": "deadbeef",
                            "baseline_git_commit": (
                                "5d7a313" if model == "m31" else "945ce39"
                            ),
                            "rollout_id": rollout,
                            "seed": 100 + rollout,
                            "generation_config": {"mode": "hybrid", "do_sample": True},
                            "record_id": sample["record_id"],
                            "sample_id": sample["sample_id"],
                            "source_image_id": sample["source_image_id"],
                            "image_id": sample["source_image_id"],
                            "image_relpath": sample["image_relpath"],
                            "image_size": {"width": 100, "height": 100},
                            "task": sample["task"],
                            "source_records": sample["source_records"],
                            "original_training_record": sample[
                                "original_training_record"
                            ],
                            "prompt": sample["prompt"],
                            "gt_global": sample["gt_global"],
                            "gt_local": sample["gt_global"],
                            "pred_local": pred,
                            "pred_global": pred,
                            "raw_output": "<box>synthetic</box>",
                            "parse_status": "defect",
                            "latency_seconds": 1.0,
                            "crop_outputs": crop_outputs,
                            "pipeline_coverage_failure": bool(
                                sample["pipeline_coverage_failure"]
                            ),
                            "annotation_anomaly": bool(sample["annotation_anomaly"]),
                            "coordinate_transform_anomaly": False,
                            "inference_success": True,
                            "runtime_error": None,
                            **score,
                        }
                        if (
                            model == "crop"
                            and rollout == 0
                            and sample["task"] == "text_ellipsis"
                        ):
                            raw.update(
                                {
                                    "oom_events": 1,
                                    "oom_recovered": True,
                                    "oom_final_failure": False,
                                    "oom_retry": {
                                        "first_attempt": {
                                            "status": "oom",
                                            "context": {
                                                "stage": "crop_generate",
                                                "crop_id": sample["crop_ids"][0],
                                                "crop_index": 0,
                                                "input_tokens": 7000,
                                            },
                                        },
                                        "retry_attempt": {"status": "success"},
                                    },
                                }
                            )
                        if (
                            model == "m31"
                            and rollout == 3
                            and sample["task"] == "text_overflow"
                        ):
                            raw.update(
                                {
                                    "pred_global": None,
                                    "parse_status": "not_attempted",
                                    "matched_pairs": [],
                                    "TP_box": None,
                                    "FP_box": None,
                                    "FN_box": None,
                                    "image_confusion": None,
                                    "error_type": "RUNTIME_ERROR",
                                    "exact_correct": None,
                                    "inference_success": False,
                                    "oom_events": 2,
                                    "oom_recovered": False,
                                    "oom_final_failure": True,
                                    "oom_retry": {
                                        "first_attempt": {
                                            "status": "oom",
                                            "context": {
                                                "stage": "full_image_generate",
                                                "crop_id": "full_image",
                                                "crop_index": None,
                                                "crop_xyxy": [0, 0, 100, 100],
                                                "input_tokens": 7100,
                                                "tile_count": 1,
                                                "tile_size": {
                                                    "width": 100,
                                                    "height": 100,
                                                },
                                            },
                                        },
                                        "retry_attempt": {"status": "oom"},
                                    },
                                    "runtime_error": {
                                        "type": "CUDA_OOM",
                                        "python_type": "RuntimeError",
                                        "message": "synthetic OOM",
                                        "traceback": "synthetic",
                                    },
                                }
                            )
                        raw_rows.append(raw)
                    write_jsonl(
                        output / "raw" / model / f"rollout_{rollout}" / "part-00000.jsonl",
                        raw_rows,
                    )
                    write_jsonl(
                        output / "progress" / model / f"rollout_{rollout}.jsonl",
                        [
                            {
                                "status": "completed",
                                "attempted": 5,
                                "inference_success": (
                                    4 if model == "m31" and rollout == 3 else 5
                                ),
                                "runtime_error": (
                                    1 if model == "m31" and rollout == 3 else 0
                                ),
                                "parse_error": 0,
                                "total": 5,
                                "elapsed_seconds": 1.0,
                                "throughput_samples_per_second": 1.0,
                                "remaining_seconds": 0.0,
                                "estimated_completion": None,
                                "gpu_memory": {"allocated_gib": 7.0},
                            }
                        ],
                    )
            second_snapshot, second_summary = snapshot.create_snapshot(
                output,
                bundle,
                kind="hourly",
                scheduled_hour=6,
                started_at_epoch=0.0,
                created_at_epoch=6 * 3600 + 1,
            )
            self.assertTrue(second_snapshot.name.startswith("hour_006_"))
            self.assertEqual(
                second_summary["difficulty_counts"],
                {
                    "easy": 1,
                    "medium": 2,
                    "hard": 1,
                },
            )
            self.assertEqual(second_summary["complete8_samples"], 4)
            self.assertEqual(second_summary["file_counts"]["complete8"], 4)
            self.assertEqual(
                second_summary["file_counts"]["incomplete_or_technical_error"], 1
            )
            self.assertEqual(second_summary["file_counts"]["delta_since_previous"], 4)
            self.assertEqual(second_summary["grpo_ready_counts"], {"m31": 1, "crop": 1})
            self.assertEqual(second_summary["error_counts"]["runtime_errors"], 1)
            self.assertEqual(second_summary["error_counts"]["oom_events"], 2)
            self.assertEqual(len(second_summary["correct_count_4"]), 60)
            self.assertEqual(len(second_summary["correct_count_8"]), 54)
            self.assertEqual(len(second_summary["cumulative_metrics"]), 70)
            for metric_row in second_summary["cumulative_metrics"]:
                self.assertTrue(
                    {
                        "TP",
                        "TN",
                        "FP",
                        "FN",
                        "precision",
                        "recall",
                        "f1",
                        "accuracy",
                        "specificity",
                    }.issubset(metric_row)
                )
            medium_rows = snapshot.read_jsonl(second_snapshot / "medium.jsonl")
            no_reward_variance = next(
                row for row in medium_rows if row["task"] == "content_missing"
            )
            self.assertEqual(no_reward_variance["m31_correct_count"], 4)
            self.assertEqual(no_reward_variance["crop_correct_count"], 0)
            self.assertFalse(no_reward_variance["grpo_ready_m31"])
            self.assertFalse(no_reward_variance["grpo_ready_crop"])
            self.assertTrue(no_reward_variance["m31_complete4"])
            self.assertTrue(no_reward_variance["crop_complete4"])
            self.assertTrue(no_reward_variance["cross_model_complete8"])
            self.assertEqual(len(no_reward_variance["rollouts"]["m31"]), 4)
            self.assertEqual(len(no_reward_variance["rollouts"]["crop"]), 4)
            for rollout_payload in (
                no_reward_variance["rollouts"]["m31"]
                + no_reward_variance["rollouts"]["crop"]
            ):
                self.assertTrue(
                    {
                        "raw_output",
                        "reward",
                        "pred_global",
                        "gt_global",
                        "parse_status",
                        "runtime_error",
                    }.issubset(rollout_payload)
                )
            incomplete = snapshot.read_jsonl(
                second_snapshot / "incomplete_or_technical_error.jsonl"
            )[0]
            self.assertFalse(incomplete["m31_complete4"])
            self.assertTrue(incomplete["crop_complete4"])
            self.assertFalse(incomplete["cross_model_complete8"])
            self.assertFalse(incomplete["grpo_ready_crop"])
            self.assertIsNone(incomplete["difficulty"])
            self.assertEqual(incomplete["runtime_error_count"], 1)
            grpo_crop = snapshot.read_jsonl(second_snapshot / "grpo_crop_ready.jsonl")
            self.assertEqual(len(grpo_crop), 1)
            self.assertTrue(all(row["group_size"] == 4 for row in grpo_crop))
            self.assertTrue(all(row["cross_model_group"] is False for row in grpo_crop))
            self.assertTrue(all(len(set(row["rewards_exact"])) == 2 for row in grpo_crop))
            for filename in (
                "errors/runtime_errors.jsonl",
                "errors/parse_errors.jsonl",
                "errors/oom_events.jsonl",
                "errors/model_load_errors.jsonl",
            ):
                self.assertTrue((second_snapshot / filename).is_file())
            manifest = json.loads(
                (second_snapshot / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["append_only"])
            self.assertTrue(manifest["atomic_publish"])
            self.assertEqual(manifest["success_marker"], "_SUCCESS")
            self.assertTrue((second_snapshot / "_SUCCESS").is_file())
            self.assertFalse(
                any(
                    path.name.startswith(f".{second_snapshot.name}.tmp-")
                    for path in second_snapshot.parent.iterdir()
                )
            )
            for item in manifest["files"]:
                artifact = second_snapshot / item["path"]
                self.assertTrue(artifact.is_file())
                self.assertEqual(snapshot.sha256_file(artifact), item["sha256"])
            third_snapshot, third_summary = snapshot.create_snapshot(
                output,
                bundle,
                kind="hourly",
                scheduled_hour=9,
                started_at_epoch=0.0,
                created_at_epoch=9 * 3600 + 1,
            )
            self.assertTrue(third_snapshot.name.startswith("hour_009_"))
            self.assertEqual(third_summary["file_counts"]["delta_since_previous"], 0)
            final_snapshot, final_summary = snapshot.create_snapshot(
                output,
                bundle,
                kind="final",
                scheduled_hour=None,
                started_at_epoch=0.0,
                export_selection_dir=output / "selection",
                created_at_epoch=12 * 3600 + 1,
            )
            self.assertTrue(final_snapshot.name.startswith("final_"))
            self.assertEqual(final_summary["file_counts"]["delta_since_previous"], 0)
            self.assertFalse((first_snapshot / "cross_model_complete8.jsonl").exists())
            self.assertTrue((second_snapshot / "complete8.jsonl").is_file())
            self.assertTrue((second_snapshot / "sample_difficulty.jsonl").is_file())
            self.assertTrue((second_snapshot / "sample_difficulty.csv").is_file())
            self.assertTrue((second_snapshot / "confusion_metrics.jsonl").is_file())
            self.assertTrue((second_snapshot / "confusion_metrics.csv").is_file())
            self.assertTrue((second_snapshot / "correct_count_8.jsonl").is_file())
            self.assertTrue((second_snapshot / "correct_count_8.csv").is_file())
            self.assertEqual(
                set(second_summary["correct_count_distribution_0to8"]),
                {str(value) for value in range(9)},
            )
            launcher = (PROJECT_ROOT / "shell" / "run_ui5_train_rollouts_h20x2.sh").read_text(
                encoding="utf-8"
            )
            self.assertIn("SNAPSHOT_INTERVAL_SECONDS=10800", launcher)
            self.assertIn("NEXT_SNAPSHOT_HOUR=$((NEXT_SNAPSHOT_HOUR + 3))", launcher)
            self.assertNotIn("SNAPSHOT_SAMPLE", launcher)
            self.assertEqual(launcher.count("launch_worker m31"), 1)
            self.assertEqual(launcher.count("launch_worker crop"), 1)
            self.assertEqual(launcher.count("for rollout_id in 0 1 2 3; do"), 2)
            self.assertIn(
                'launch_worker m31 0 "${rollout_id}" "${SEEDS[rollout_id]}"',
                launcher,
            )
            self.assertIn(
                'launch_worker crop 1 "${rollout_id}" "${SEEDS[rollout_id]}"',
                launcher,
            )
            self.assertIn("--max-seq-length 7268", launcher)
            self.assertIn("--max-new-tokens 512", launcher)
            analysis = aggregate.run(
                SimpleNamespace(output_root=output, bundle_root=bundle, repo_root=PROJECT_ROOT)
            )
            self.assertEqual(analysis["common_image_task_intersection"], 4)
            m31_r3 = next(
                row
                for row in analysis["execution_counts"]
                if row["model_id"] == "m31"
                and row["rollout_id"] == 3
                and row["scope"] == "micro"
            )
            self.assertEqual(m31_r3["runtime_error"], 1)
            self.assertEqual(m31_r3["inference_success"], 4)
            self.assertTrue(all(row["complete"] for row in analysis["raw_alignment"]))
            self.assertEqual(analysis["difficulty_file_counts"]["complete8"], 4)
            self.assertEqual(analysis["difficulty_file_counts"]["easy"], 1)
            self.assertEqual(analysis["difficulty_file_counts"]["medium"], 2)
            self.assertEqual(analysis["difficulty_file_counts"]["hard"], 1)
            self.assertEqual(
                analysis["difficulty_file_counts"]["incomplete_or_technical_error"],
                1,
            )
            self.assertEqual(analysis["difficulty_file_counts"]["grpo_m31_ready"], 1)
            self.assertEqual(analysis["difficulty_file_counts"]["grpo_crop_ready"], 1)
            self.assertFalse((output / "selection" / "cross_model_complete8.jsonl").exists())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "reports" / "ui5_train_rollout_analysis.xlsx").is_file())
            run_config = json.loads(
                (output / "run_config.snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_config["generation"]["max_seq_length"], 7268)
            self.assertEqual(run_config["generation"]["max_new_tokens"], 512)
            self.assertEqual(
                run_config["generation"]["training_max_num_tokens_record_only"],
                12800,
            )
            self.assertEqual(
                run_config["sample_order"]["policy"],
                "sample_major_fixed_task_polarity_round_robin_v1",
            )
            self.assertEqual(
                run_config["execution_architecture"]["physical_processes_total"],
                8,
            )
            self.assertEqual(
                run_config["execution_architecture"]["physical_processes_per_gpu"],
                4,
            )
            self.assertEqual(
                run_config["execution_architecture"]["rollouts_per_physical_process"],
                1,
            )
            self.assertEqual(
                run_config["execution_architecture"]["global_eta_reduction"],
                "maximum_estimated_completion_across_8_processes",
            )
            self.assertEqual(
                run_config["generation"]["vision_attention"],
                "flash_attention_2",
            )
            sync_script = (
                PROJECT_ROOT / "shell" / "sync_ui5_rollout_results.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("nastk cp -c=32", sync_script)
            self.assertIn('if [[ $# -ne 2 ]]', sync_script)
            self.assertIn('"${OUTPUT_ROOT}" "${DESTINATION}"', sync_script)
            rendered = gallery.render(
                SimpleNamespace(output_root=output, bundle_root=bundle, panel_long_side=160)
            )
            self.assertGreater(rendered["rendered"], 0)
            self.assertTrue((output / "visualizations" / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
