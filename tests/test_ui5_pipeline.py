from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collect_ui5_metrics
import locany_ui5_checkpoint
import locany_ui5_common
import patch_locany_checkpoint
import submit_locany_ui5


class RuntimeConfigTests(unittest.TestCase):
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
        self.assertEqual(four["MAX_NUM_TOKENS_SCOPE"], "per_rank_packed_batch")
        self.assertEqual(
            four["TRAINING_DATA_SOURCE_DIR"],
            "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/"
            "code/Eagle/Embodied/data/ui_defect_locany_v3",
        )

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
        self.assertIn("intelligent-service-arnold-hl", config["WORKSPACE"])
        self.assertTrue(config["PROJECT_ROOT"].startswith("/mnt/"))

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
        self.assertNotIn("@@", rendered)
        self.assertEqual(runtime["ENABLE_EVAL"], 0)

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

    def make_resume_checkpoint(self, root: Path, step: int) -> Path:
        checkpoint = self.make_eval_checkpoint(root, step)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        state_dir = checkpoint / f"global_step{step}"
        state_dir.mkdir()
        (state_dir / "mp_rank_00_model_states.pt").write_bytes(b"state")
        return checkpoint

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
