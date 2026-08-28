from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import locany_ui5_common
import run_locany_cpt_ui5_checkpoint_sweep as sweep


def canonical_metrics(f1: float = 0.5) -> dict:
    tasks = {}
    for task in locany_ui5_common.TASKS:
        tasks[task] = {
            "issue_name": locany_ui5_common.TASK_ISSUE_NAMES[task],
            "image": {
                "precision": f1,
                "recall": f1,
                "f1": f1,
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 1,
                "accuracy": 0.5,
            },
            "bbox": {
                "precision": f1,
                "recall": f1,
                "f1": f1,
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "count_accuracy": 0.5,
            },
        }
    return {
        "schema_version": 1,
        "tasks": tasks,
        "macro": {
            "image": {"precision": f1, "recall": f1, "f1": f1},
            "bbox": {"precision": f1, "recall": f1, "f1": f1},
        },
    }


class CPTUI5CheckpointSweepTests(unittest.TestCase):
    def test_empty_input_dir_infers_workspace_data_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            run_dir = workspace / "gui_models" / "legacy-cpt"
            test_dir = workspace / "data"
            run_dir.mkdir(parents=True)
            test_dir.mkdir()
            for filename in locany_ui5_common.TASK_JSONL.values():
                (test_dir / filename).write_text("{}\n", encoding="utf-8")

            resolved, source = sweep.resolve_ui5_input_dir("", run_dir)

            self.assertEqual(resolved, test_dir.resolve())
            self.assertEqual(source, "inferred-from-run-dir")

    def test_two_gpu_checkpoint_sweep_scores_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            processor = root / "processor"
            input_dir = root / "test_data"
            output_root = root / "sweep"
            run_dir.mkdir()
            processor.mkdir()
            input_dir.mkdir()
            for step in (516, 1033):
                (run_dir / f"checkpoint-{step}").mkdir()
            for task, filename in locany_ui5_common.TASK_JSONL.items():
                (input_dir / filename).write_text(
                    json.dumps({"images": [str(root / f"{task}.png")]}) + "\n",
                    encoding="utf-8",
                )

            fake_inference = root / "fake_inference.py"
            fake_inference.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    import os
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument('--checkpoint', required=True)
                    parser.add_argument('--output-dir', required=True)
                    parser.add_argument('--summary-path', required=True)
                    args, _ = parser.parse_known_args()
                    output = Path(args.output_dir)
                    tasks = ('occlusion', 'cropping', 'text_overflow', 'text_ellipsis', 'content_missing')
                    for task in tasks:
                        task_dir = output / task
                        task_dir.mkdir(parents=True, exist_ok=True)
                        (task_dir / 'sample.json').write_text('[]\\n', encoding='utf-8')
                    (output / '_fake_gpu.txt').write_text(os.environ.get('CUDA_VISIBLE_DEVICES', ''), encoding='utf-8')
                    Path(args.summary_path).write_text(
                        json.dumps({'checkpoint': args.checkpoint, 'totals': {'inference_error': 0}}),
                        encoding='utf-8',
                    )
                    """
                ),
                encoding="utf-8",
            )
            fake_scorer = root / "fake_scorer.py"
            fake_scorer.write_text(
                textwrap.dedent(
                    f"""
                    import argparse
                    import json
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument('--output_root', required=True)
                    parser.add_argument('--run_name', required=True)
                    args, _ = parser.parse_known_args()
                    output = Path(args.output_root) / args.run_name
                    output.mkdir(parents=True)
                    metrics = {canonical_metrics()!r}
                    (output / 'all_tasks_evaluation.json').write_text(
                        json.dumps(metrics), encoding='utf-8'
                    )
                    """
                ),
                encoding="utf-8",
            )

            command = [
                sys.executable,
                str(SCRIPTS_DIR / "run_locany_cpt_ui5_checkpoint_sweep.py"),
                "--run-dir",
                str(run_dir),
                "--processor-path",
                str(processor),
                "--input-dir",
                str(input_dir),
                "--output-root",
                str(output_root),
                "--steps",
                "516",
                "1033",
                "--gpu-devices",
                "0,1",
                "--python",
                sys.executable,
                "--inference-script",
                str(fake_inference),
                "--scorer-script",
                str(fake_scorer),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            status = json.loads((output_root / "sweep_status.json").read_text(encoding="utf-8"))
            self.assertTrue(status["success"])
            self.assertEqual(set(status["checkpoints"]), {"516", "1033"})
            self.assertTrue(all(row["status"] == "success" for row in status["checkpoints"].values()))
            for step in (516, 1033):
                prediction = output_root / "predictions" / f"checkpoint-{step}"
                self.assertIn((prediction / "_fake_gpu.txt").read_text(encoding="utf-8"), {"0", "1"})
                self.assertTrue((output_root / "metrics" / f"checkpoint-{step}" / "iou-0p1.json").is_file())

            comparison = json.loads(
                (output_root / "comparison" / "checkpoint_comparison_iou-0p1.json").read_text(encoding="utf-8")
            )
            self.assertEqual([model["step"] for model in comparison["models"]], [516, 1033])
            self.assertEqual(comparison["models"][0]["micro"]["image"]["tp"], 5)
            self.assertEqual(comparison["models"][0]["micro"]["bbox"]["f1"], 0.5)
            self.assertTrue(
                (output_root / "comparison" / "checkpoint_comparison_iou-0p1_image.csv").is_file()
            )
            self.assertTrue(
                (output_root / "comparison" / "checkpoint_comparison_iou-0p1_bbox.csv").is_file()
            )

    def test_aggregate_only_can_append_an_external_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            processor = root / "processor"
            input_dir = root / "test_data"
            output_root = root / "sweep"
            checkpoint = run_dir / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            processor.mkdir()
            input_dir.mkdir()
            for filename in locany_ui5_common.TASK_JSONL.values():
                (input_dir / filename).write_text("{}\n", encoding="utf-8")
            metric_dir = output_root / "metrics" / "checkpoint-10"
            metric_dir.mkdir(parents=True)
            (metric_dir / "iou-0p1.json").write_text(
                json.dumps(canonical_metrics(0.4)), encoding="utf-8"
            )
            external = root / "other_model.json"
            external.write_text(json.dumps(canonical_metrics(0.7)), encoding="utf-8")

            command = [
                sys.executable,
                str(SCRIPTS_DIR / "run_locany_cpt_ui5_checkpoint_sweep.py"),
                "--run-dir",
                str(run_dir),
                "--processor-path",
                str(processor),
                "--input-dir",
                str(input_dir),
                "--output-root",
                str(output_root),
                "--stage",
                "aggregate-only",
                "--external-metrics",
                f"qwen3vl={external}",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            comparison = json.loads(
                (output_root / "comparison" / "checkpoint_comparison_iou-0p1.json").read_text(encoding="utf-8")
            )
            self.assertEqual([row["label"] for row in comparison["models"]], ["cpt-step-10", "qwen3vl"])
            self.assertEqual(comparison["models"][1]["macro"]["bbox"]["f1"], 0.7)

    def test_legacy_cpt_command_disables_new_relation_paths(self) -> None:
        args = type(
            "Args",
            (),
            {
                "python": "python",
                "inference_script": Path("infer.py"),
                "processor_path": Path("processor"),
                "input_dir": Path("input"),
                "dtype": "bf16",
                "attn_implementation": "sdpa",
                "vision_attn_implementation": "flash_attention_2",
                "generation_mode": "hybrid",
                "max_new_tokens": 4096,
                "n_future_tokens": 6,
                "seed": 1,
                "save_raw_answer": True,
                "save_visualization": False,
                "enable_ui_relation": False,
                "enable_pbd": False,
                "greedy": False,
                "max_images_per_task": 0,
                "overwrite": False,
            },
        )()
        command = sweep.build_inference_command(args, Path("checkpoint-1"), Path("out"), "0")
        self.assertIn("--no-enable-ui-relation", command)
        self.assertIn("--no-enable-pbd", command)
        self.assertIn("--save-raw-answer", command)
        self.assertNotIn("--save-visualization", command)


if __name__ == "__main__":
    unittest.main()
