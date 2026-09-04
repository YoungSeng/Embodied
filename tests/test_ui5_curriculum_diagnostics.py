from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_ui5_curriculum_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("ui5_curriculum_diagnostics", SCRIPT)
assert SPEC and SPEC.loader
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class CurriculumDiagnosticsTests(unittest.TestCase):
    def test_exports_curve_hard_transitions_and_anchor_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            curriculum = root / "curriculum"
            for task in diagnostics.TASKS:
                for rollout in range(4):
                    rows = []
                    if task == "occlusion":
                        rows = [
                            {
                                "record_id": "hard-1",
                                "exact_correct": rollout < 2,
                                "parse_status": "ok",
                                "runtime_error": None,
                            }
                        ]
                    write_jsonl(
                        evaluation
                        / f"ui_{task}"
                        / "rollout4"
                        / f"rollout_{rollout}.jsonl",
                        rows,
                    )
            write_jsonl(
                curriculum / "matched_anchor_groups.jsonl",
                [
                    {
                        "sample_id": "anchor-1",
                        "task": "occlusion",
                    }
                ],
            )
            write_jsonl(
                curriculum / "matched_anchor.jsonl",
                [
                    {
                        "_ui5_sample_id": "anchor-1",
                        "_ui5_task": "occlusion",
                    }
                ],
            )
            trainer_state = root / "trainer_state.json"
            trainer_state.write_text(
                json.dumps(
                    {
                        "global_step": 400,
                        "log_history": [
                            {"step": 200, "loss": 1.2, "learning_rate": 1e-6},
                            {"step": 400, "loss": 1.0, "learning_rate": 1e-6},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            outputs = diagnostics.run(
                Namespace(
                    step=400,
                    evaluation_dir=evaluation,
                    curriculum_dir=curriculum,
                    trainer_state=trainer_state,
                    total_steps=1200,
                    output_dir=None,
                )
            )
            curve = json.loads(Path(outputs["train_curve"]).read_text())[
                "train_curve"
            ]
            self.assertEqual([row["step"] for row in curve], [200, 400])
            self.assertEqual(curve[-1]["phase"], 1)
            transitions = json.loads(
                Path(outputs["hard_transition"]).read_text()
            )["hard_transition"]
            recovered = next(
                row
                for row in transitions
                if row["task"] == "ui_occlusion"
                and row["transition"] == "0/4 -> 2/4"
            )
            self.assertEqual(recovered["samples"], 1)
            anchors = json.loads(
                Path(outputs["anchor_retention"]).read_text()
            )["anchor_retention"]
            occlusion = next(row for row in anchors if row["task"] == "ui_occlusion")
            self.assertTrue(occlusion["retained"])
            self.assertEqual(occlusion["current_score"], 1.0)

    def test_phase_switches_after_global_step_400(self) -> None:
        phase400, ratios400 = diagnostics._phase_for_step(400, 1200)
        phase401, ratios401 = diagnostics._phase_for_step(401, 1200)
        phase801, ratios801 = diagnostics._phase_for_step(801, 1200)
        self.assertEqual((phase400, phase401, phase801), (0, 1, 2))
        self.assertEqual(ratios400[:3], (0.60, 0.25, 0.15))
        self.assertEqual(ratios401[:3], (0.45, 0.35, 0.20))
        self.assertEqual(ratios801[:3], (0.30, 0.30, 0.40))

    def test_parse_error_is_incorrect_but_runtime_error_is_fatal(self) -> None:
        self.assertFalse(
            diagnostics._correct(
                {
                    "record_id": "bad-model-output",
                    "exact_correct": False,
                    "parse_status": "parse_error",
                    "runtime_error": None,
                }
            )
        )
        with self.assertRaisesRegex(RuntimeError, "runtime error"):
            diagnostics._correct(
                {
                    "record_id": "failed-execution",
                    "exact_correct": False,
                    "parse_status": "parse_error",
                    "runtime_error": {"type": "CUDA_OOM"},
                }
            )

    def test_anchor_retention_uses_id_intersection_not_only_equal_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            curriculum = Path(temporary)
            write_jsonl(
                curriculum / "matched_anchor_groups.jsonl",
                [{"sample_id": "expected-anchor", "task": "occlusion"}],
            )
            write_jsonl(
                curriculum / "matched_anchor.jsonl",
                [
                    {
                        "_ui5_sample_id": "different-anchor",
                        "_ui5_task": "occlusion",
                    }
                ],
            )

            rows = diagnostics.anchor_retention_rows(
                step=200, curriculum_dir=curriculum
            )
            occlusion = next(row for row in rows if row["task"] == "ui_occlusion")
            self.assertEqual(occlusion["samples"], 1)
            self.assertEqual(occlusion["current_score"], 0.0)
            self.assertEqual(occlusion["retained"], False)

    def test_anchor_retention_rejects_duplicate_ids_and_task_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            curriculum = Path(temporary)
            write_jsonl(
                curriculum / "matched_anchor_groups.jsonl",
                [
                    {"sample_id": "duplicate", "task": "occlusion"},
                    {"sample_id": "duplicate", "task": "cropping"},
                ],
            )
            write_jsonl(curriculum / "matched_anchor.jsonl", [])
            with self.assertRaisesRegex(ValueError, "duplicate matched-anchor"):
                diagnostics.anchor_retention_rows(step=200, curriculum_dir=curriculum)

        with tempfile.TemporaryDirectory() as temporary:
            curriculum = Path(temporary)
            write_jsonl(
                curriculum / "matched_anchor_groups.jsonl",
                [{"sample_id": "anchor", "task": "occlusion"}],
            )
            write_jsonl(
                curriculum / "matched_anchor.jsonl",
                [
                    {
                        "_ui5_sample_id": "anchor",
                        "_ui5_task": "cropping",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "conflicts with its expected"):
                diagnostics.anchor_retention_rows(step=200, curriculum_dir=curriculum)

        with tempfile.TemporaryDirectory() as temporary:
            curriculum = Path(temporary)
            write_jsonl(
                curriculum / "matched_anchor_groups.jsonl",
                [{"sample_id": "", "task": "occlusion"}],
            )
            write_jsonl(curriculum / "matched_anchor.jsonl", [])
            with self.assertRaisesRegex(ValueError, "lacks a stable sample_id"):
                diagnostics.anchor_retention_rows(step=200, curriculum_dir=curriculum)

        with tempfile.TemporaryDirectory() as temporary:
            curriculum = Path(temporary)
            write_jsonl(
                curriculum / "matched_anchor_groups.jsonl",
                [{"sample_id": "anchor", "task": "occlusion"}],
            )
            write_jsonl(
                curriculum / "matched_anchor.jsonl",
                [
                    {"_ui5_sample_id": "extra", "_ui5_task": "cropping"},
                    {"_ui5_sample_id": "extra", "_ui5_task": "text_ellipsis"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "belongs to multiple tasks"):
                diagnostics.anchor_retention_rows(step=200, curriculum_dir=curriculum)


if __name__ == "__main__":
    unittest.main()
