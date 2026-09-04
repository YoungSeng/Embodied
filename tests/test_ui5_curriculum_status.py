from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_ui5_training_segment as training_status  # noqa: E402
import update_ui5_curriculum_artifacts as artifact_status  # noqa: E402


class TrainingSegmentStatusTests(unittest.TestCase):
    def test_complete_segment_prints_actual_metrics_and_cumulative_draws(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "resume" / "latest"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "global_step": 200,
                        "log_history": [
                            {
                                "step": 200,
                                "loss": 1.25,
                                "loss_lm": 1.1,
                                "grad_norm": 2.5,
                                "learning_rate": 1.0e-6,
                                "curriculum_hard_samples": 600,
                                "curriculum_anchor_samples": 250,
                                "curriculum_global_replay_samples": 150,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                event="complete",
                start_step=0,
                target_step=200,
                total_steps=1200,
                checkpoint=checkpoint,
            )
            status = training_status.build_status(args)
            self.assertEqual(status["training"]["loss_total"], 1.25)
            self.assertEqual(
                status["training"]["pool_samples_cumulative"],
                {"hard": 600, "anchor": 250, "global_replay": 150},
            )
            output = io.StringIO()
            with redirect_stdout(output):
                training_status.print_status(status)
            structured = next(
                line.split(" ", 2)[2]
                for line in output.getvalue().splitlines()
                if line.startswith("[CURRICULUM STATUS] ")
            )
            self.assertEqual(json.loads(structured), status)

    def test_start_segment_explicitly_prints_na_for_unavailable_measurements(self) -> None:
        args = SimpleNamespace(
            event="start",
            start_step=400,
            target_step=600,
            total_steps=1200,
            checkpoint=None,
        )
        status = training_status.build_status(args)
        output = io.StringIO()
        with redirect_stdout(output):
            training_status.print_status(status)
        self.assertEqual(status["phase"], 2)
        self.assertIn("loss=N/A", output.getvalue())
        self.assertIn("pool_cumulative=hard:N/A,anchor:N/A,global_replay:N/A", output.getvalue())


class RegisteredEvaluationStatusTests(unittest.TestCase):
    def test_status_contains_post_decision_best_records_and_next_action(self) -> None:
        tasks = {
            f"ui_{name}": {
                "image": {"f1": 0.6},
                "bbox": {"f1": 0.4},
            }
            for name in (
                "occlusion",
                "cropping",
                "text_overflow",
                "text_ellipsis",
                "content_missing",
            )
        }
        result = {
            "step": 200,
            "candidate_checkpoint": "resume/latest",
            "evaluation_seconds": 17.25,
            "metrics": {
                "tasks": tasks,
                "macro": {"image": {"f1": 0.6}, "bbox": {"f1": 0.4}},
                "micro": {"image": {"f1": 0.55}, "bbox": {"f1": 0.35}},
                "overall": {"joint_score": 0.5},
            },
            "improved_image": True,
            "improved_bbox": False,
            "improved_joint": True,
            "checkpoint_preserved": True,
            "checkpoint_path": "checkpoints/step-000200",
            "best_image": {
                "step": 200,
                "score": 0.6,
                "checkpoint_path": "checkpoints/step-000200",
            },
            "best_bbox": {
                "step": 0,
                "score": 0.5,
                "checkpoint_path": "",
            },
            "best_joint": {
                "step": 200,
                "score": 0.5,
                "checkpoint_path": "checkpoints/step-000200",
            },
            "checkpoints_json": "checkpoints.json",
            "workbook": "diagnostics.xlsx",
        }
        status = artifact_status.build_curriculum_status(
            result,
            train_curve_rows=[
                {
                    "step": 200,
                    "learning_rate": 1.0e-6,
                    "loss_total": 1.2,
                    "loss_lm": 1.1,
                    "grad_norm": 2.0,
                    "hard_samples": 60,
                    "anchor_samples": 25,
                    "global_replay_samples": 15,
                }
            ],
            total_steps=1200,
            eval_interval_steps=200,
        )
        self.assertEqual(status["checkpoint"]["improved_joint"], True)
        self.assertEqual(status["best"]["image"]["step"], 200)
        self.assertEqual(status["next_action"], "train_step_200_to_400")
        self.assertEqual(
            status["training"]["learning_rate_semantics"], "next_optimizer_step"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            artifact_status.print_curriculum_status(status)
        rendered = output.getvalue()
        self.assertEqual(rendered.count("[UI5 TASK METRICS]"), 5)
        self.assertIn("[CHECKPOINT DECISION]", rendered)
        self.assertIn("[BEST CHECKPOINTS]", rendered)
        structured = next(
            line.split(" ", 2)[2]
            for line in rendered.splitlines()
            if line.startswith("[CURRICULUM STATUS] ")
        )
        self.assertEqual(json.loads(structured), status)

        boundary_result = {**result, "step": 400}
        boundary = artifact_status.build_curriculum_status(
            boundary_result,
            train_curve_rows=[
                {
                    "step": 400,
                    # HF reports the scheduler value for the next optimizer
                    # update after completing step 400.
                    "learning_rate": 7.0e-7,
                }
            ],
            total_steps=1200,
            eval_interval_steps=200,
        )
        self.assertEqual(boundary["phase"], 1)
        self.assertEqual(boundary["curriculum_target"]["llm_lr"], 1.0e-6)
        self.assertEqual(boundary["next_curriculum_target"]["phase"], 2)
        self.assertEqual(boundary["next_curriculum_target"]["llm_lr"], 7.0e-7)
        boundary_output = io.StringIO()
        with redirect_stdout(boundary_output):
            artifact_status.print_curriculum_status(boundary)
        self.assertIn("lr_next=7e-07 next_target_lr=7e-07", boundary_output.getvalue())


if __name__ == "__main__":
    unittest.main()
