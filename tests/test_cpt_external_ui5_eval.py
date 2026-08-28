from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.locany_ui5_common import TASK_JSONL, TASKS
from scripts.run_locany_cpt_external_ui5_eval import (
    base_bootstrap_command,
    cpt_metric_shape,
    write_eval_rows,
)


def canonical_metrics() -> dict:
    tasks = {}
    for index, task in enumerate(TASKS, start=1):
        tasks[task] = {
            "image": {
                "tp": index,
                "fp": 1,
                "fn": 2,
                "tn": 10,
                "precision": index / (index + 1),
                "recall": index / (index + 2),
                "f1": 0.5,
                "accuracy": 0.8,
            },
            "bbox": {
                "tp": index + 1,
                "fp": 2,
                "fn": 3,
                "precision": 0.6,
                "recall": 0.5,
                "f1": 6 / 11,
                "count_accuracy": 0.7,
            },
        }
    return {
        "tasks": tasks,
        "macro": {
            "image": {"precision": 0.7, "recall": 0.6, "f1": 0.65},
            "bbox": {"precision": 0.6, "recall": 0.5, "f1": 6 / 11},
        },
    }


class CPTExternalUI5EvalTest(unittest.TestCase):
    def test_end_to_end_writes_two_thresholds_for_base_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            input_dir = root / "data"
            base = root / "base"
            checkpoint = root / "checkpoint-10"
            for path in (run_dir / "diagnostics", input_dir, base, checkpoint):
                path.mkdir(parents=True)
            for filename in TASK_JSONL.values():
                (input_dir / filename).write_text("{}\n", encoding="utf-8")

            fake_inference = root / "fake_inference.py"
            fake_inference.write_text(
                textwrap.dedent(
                    f"""
                    import argparse
                    import json
                    from pathlib import Path
                    parser = argparse.ArgumentParser(allow_abbrev=False)
                    parser.add_argument('--output-dir', required=True)
                    args, _ = parser.parse_known_args()
                    output = Path(args.output_dir)
                    output.mkdir(parents=True, exist_ok=True)
                    for task in {list(TASKS)!r}:
                        (output / task).mkdir(parents=True, exist_ok=True)
                    (output / '_summary.json').write_text(
                        json.dumps({{'totals': {{'inference_error': 0}}}}),
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
                    parser = argparse.ArgumentParser(allow_abbrev=False)
                    parser.add_argument('--output_root', required=True)
                    parser.add_argument('--run_name', required=True)
                    args, _ = parser.parse_known_args()
                    output = Path(args.output_root) / args.run_name
                    output.mkdir(parents=True, exist_ok=False)
                    (output / 'all_tasks_evaluation.json').write_text(
                        json.dumps({canonical_metrics()!r}), encoding='utf-8'
                    )
                    """
                ),
                encoding="utf-8",
            )
            script = Path(__file__).resolve().parents[1] / "scripts/run_locany_cpt_external_ui5_eval.py"

            def command(model: Path, step: int) -> list[str]:
                return [
                    sys.executable,
                    str(script),
                    "--checkpoint", str(model),
                    "--checkpoint-step", str(step),
                    "--base-model", str(base),
                    "--run-dir", str(run_dir),
                    "--input-dir", str(input_dir),
                    "--python", sys.executable,
                    "--inference-script", str(fake_inference),
                    "--scorer-script", str(fake_scorer),
                    "--no-build-excel",
                ]

            base_run = subprocess.run(command(base, 0), capture_output=True, text=True)
            self.assertEqual(base_run.returncode, 0, base_run.stdout + base_run.stderr)
            checkpoint_run = subprocess.run(
                command(checkpoint, 10), capture_output=True, text=True
            )
            self.assertEqual(
                checkpoint_run.returncode,
                0,
                checkpoint_run.stdout + checkpoint_run.stderr,
            )
            rows = [
                json.loads(line)
                for line in (run_dir / "diagnostics/cpt_eval_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["step"] for row in rows}, {0, 10})
            self.assertEqual({row["iou_threshold"] for row in rows}, {0.1})
            self.assertTrue(all(row["split"] == "external_ui5" for row in rows))
            self.assertTrue(
                all(row["eligible_for_best_checkpoint"] is False for row in rows)
            )
            self.assertIn("EXTERNAL_UI5_EVAL=COMPLETED", checkpoint_run.stdout)

    def test_bootstrap_command_preserves_external_protocol(self) -> None:
        args = SimpleNamespace(
            python="python",
            base_model=Path("/models/base"),
            processor_path=None,
            run_dir=Path("/runs/formal"),
            input_dir=Path("/workspace/data"),
            inference_script=Path("/repo/infer.py"),
            scorer_script=Path("/repo/score.py"),
            metrics_jsonl=Path("/runs/formal/diagnostics/cpt_eval_metrics.jsonl"),
            device="cuda:0",
            dtype="bf16",
            attn_implementation="sdpa",
            vision_attn_implementation="flash_attention_2",
            generation_mode="hybrid",
            max_new_tokens=4096,
            n_future_tokens=6,
            seed=20260826,
            max_images_per_task=0,
            iou_thresholds=(0.1,),
        )
        command = base_bootstrap_command(args, force=False)
        self.assertIn("--checkpoint-step", command)
        self.assertEqual(command[command.index("--checkpoint-step") + 1], "0")
        self.assertEqual(command[-2:], ["0.1", "--no-build-excel"])
        self.assertNotIn("--force", command)

    def test_canonical_metrics_project_to_five_class_cpt_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(canonical_metrics()), encoding="utf-8")
            shaped = cpt_metric_shape(path, canonical_metrics(), 0.1)
        self.assertEqual(set(shaped["per_class"]), set(TASKS))
        self.assertEqual(shaped["iou_threshold"], 0.1)
        self.assertAlmostEqual(shaped["bbox_macro"]["f1"], 6 / 11)
        self.assertGreater(shaped["image_micro"]["tp"], 0)
        self.assertGreater(shaped["bbox_micro"]["fn"], 0)

    def test_eval_jsonl_replaces_same_evaluation_id_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpt_eval_metrics.jsonl"
            path.write_text(
                json.dumps({"evaluation_id": "heldout", "step": 1, "task": "vqa"})
                + "\n"
                + json.dumps(
                    {
                        "evaluation_id": "obsolete-external-05",
                        "step": 1,
                        "split": "external_ui5",
                        "task": "ui_defect_external",
                        "iou_threshold": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            row = {
                "evaluation_id": "external",
                "step": 2,
                "split": "external_ui5",
                "task": "ui_defect_external",
                "iou_threshold": 0.1,
                "primary_metric": 0.4,
            }
            write_eval_rows(path, [row])
            write_eval_rows(path, [{**row, "primary_metric": 0.6}])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {value.get("iou_threshold") for value in rows if value.get("split") == "external_ui5"},
            {0.1},
        )
        external = next(value for value in rows if value["evaluation_id"] == "external")
        self.assertEqual(external["primary_metric"], 0.6)


if __name__ == "__main__":
    unittest.main()
