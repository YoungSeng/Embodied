from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from eaglevl.train.cpt_eval_metrics import UI_DEFECT_CLASSES
from eaglevl.train.cpt_observability import CPT_TASKS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_locany_cpt_initial_eval.py"


def summary_fixture() -> dict:
    per_task = {
        task: {
            "eval_main_token_ce": 1.0,
            "eval_main_loss_tokens": 10,
            "inference_error_count": 0,
            "primary_metric": 0.5,
        }
        for task in CPT_TASKS
    }
    per_task["ui_defect"].update(
        {
            "per_class": {
                label: {"image": {"f1": 0.5}, "bbox": {"f1": 0.5}}
                for label in UI_DEFECT_CLASSES
            },
            "defect_image_macro_f1": 0.5,
            "defect_image_micro_f1": 0.5,
            "iou_threshold": 0.1,
            "defect_bbox_macro_f1": 0.5,
            "defect_bbox_micro_f1": 0.5,
        }
    )
    for task in ("all_ui_elements", "single_grounding", "ocr"):
        per_task[task]["iou_threshold"] = 0.1
    aggregate = {
        "per_task": per_task,
        "heldout_task_macro_primary": 0.5,
    }
    return {
        "schema_version": 2,
        "step": 0,
        "split": "heldout",
        "teacher_forced": True,
        "iou_threshold": 0.1,
        "task_counts": {task: 10 for task in CPT_TASKS},
        "manifest_id": "manifest",
        "evaluation_protocol_id": "protocol",
        "base": aggregate,
        "checkpoint_metrics": aggregate,
    }


class CPTInitialEvalTest(unittest.TestCase):
    def test_initial_eval_runs_once_validates_and_writes_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            data_dir = root / "data"
            external_data_dir = root / "external_data"
            base_model = root / "base"
            for path in (
                run_dir / "diagnostics",
                data_dir,
                external_data_dir,
                base_model,
            ):
                path.mkdir(parents=True)
            calls = root / "calls.txt"
            evaluator = root / "fake_evaluator.py"
            evaluator.write_text(
                textwrap.dedent(
                    f"""
                    import argparse
                    import json
                    from pathlib import Path

                    parser = argparse.ArgumentParser(allow_abbrev=False)
                    parser.add_argument('--output-dir', required=True)
                    parser.add_argument('--checkpoint-step', type=int, required=True)
                    parser.add_argument('--iou-threshold', type=float, required=True)
                    args, _ = parser.parse_known_args()
                    assert args.checkpoint_step == 0
                    assert args.iou_threshold == 0.1
                    output = Path(args.output_dir)
                    output.mkdir(parents=True, exist_ok=True)
                    calls = Path({str(calls)!r})
                    previous = calls.read_text(encoding='utf-8') if calls.exists() else ''
                    calls.write_text(previous + 'called\\n', encoding='utf-8')
                    (output / 'summary.json').write_text(
                        json.dumps({summary_fixture()!r}), encoding='utf-8'
                    )
                    """
                ),
                encoding="utf-8",
            )
            external_calls = root / "external_calls.txt"
            external_evaluator = root / "fake_external_evaluator.py"
            external_evaluator.write_text(
                textwrap.dedent(
                    f"""
                    import argparse
                    import json
                    from pathlib import Path

                    parser = argparse.ArgumentParser(allow_abbrev=False)
                    parser.add_argument('--run-dir', required=True)
                    parser.add_argument('--checkpoint-step', type=int, required=True)
                    args, _ = parser.parse_known_args()
                    assert args.checkpoint_step == 0
                    output = Path(args.run_dir) / 'eval_external_ui5/checkpoint-0'
                    output.mkdir(parents=True, exist_ok=True)
                    calls = Path({str(external_calls)!r})
                    previous = calls.read_text(encoding='utf-8') if calls.exists() else ''
                    calls.write_text(previous + 'called\\n', encoding='utf-8')
                    (output / 'summary.json').write_text(
                        json.dumps({{
                            'split': 'external_ui5',
                            'step': 0,
                            'metrics': {{'iou-0p1': {{}}}},
                        }}),
                        encoding='utf-8',
                    )
                    """
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(SCRIPT),
                "--run-dir",
                str(run_dir),
                "--data-dir",
                str(data_dir),
                "--base-model",
                str(base_model),
                "--external-ui5-data-dir",
                str(external_data_dir),
                "--python",
                sys.executable,
                "--evaluator",
                str(evaluator),
                "--external-evaluator",
                str(external_evaluator),
            ]

            first = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            marker_path = run_dir / "eval/checkpoint-0/initial_eval_complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["identity"]["step"], 0)
            self.assertIn("INITIAL_CPT_EVAL=COMPLETED", first.stdout)

            second = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("INITIAL_CPT_EVAL=SKIPPED_VALID_COMPLETION", second.stdout)
            self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), ["called"])
            self.assertEqual(
                external_calls.read_text(encoding="utf-8").splitlines(), ["called"]
            )


if __name__ == "__main__":
    unittest.main()
