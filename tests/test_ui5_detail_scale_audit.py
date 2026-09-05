"""CPU-only regressions for loaded/fresh/resumed Detail Pyramid diagnostics."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "eaglevl/model/locany/ui_relation_setup.py"
SPEC = importlib.util.spec_from_file_location("ui5_scale_audit_setup", SETUP)
assert SPEC and SPEC.loader
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)
TRAINER = ROOT / "eaglevl/train/locany_finetune_magi_stream.py"


class DetailScaleAuditTests(unittest.TestCase):
    def weights(self):
        # The actual nonuniform learned values reported by both H20 ranks.
        return torch.tensor([
            [0.32633867859840393, 0.32990342378616333, 0.34375789761543274],
            [0.3249208927154541, 0.3322710692882538, 0.3428080379962921],
            [0.32457470893859863, 0.3316842019557953, 0.34374111890792847],
            [0.32721683382987976, 0.32889875769615173, 0.3438844382762909],
        ], dtype=torch.float32, requires_grad=True)

    def audit(self, values, *, step=0, state="complete", resume=False):
        return setup.audit_ui5_detail_scale_weights(
            values, global_step=step, load_state=state, resuming_from_checkpoint=resume,
        )

    def test_loaded_crop_checkpoint_at_new_run_step_zero_is_accepted_without_mutation(self):
        weights = self.weights()
        before = weights.detach().clone()
        rng = torch.get_rng_state().clone()
        report = self.audit(weights)
        self.assertTrue(report["scale_weights_valid"])
        self.assertFalse(report["initial_thirds_required"])
        self.assertEqual(report["ui_relation_load_state"], "complete")
        self.assertGreater(report["scale_weight_max_deviation_from_thirds"], 0.01)
        torch.testing.assert_close(weights.detach(), before, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(weights.grad)
        weights.sum().backward()
        torch.testing.assert_close(weights.grad, torch.ones_like(weights))

    def test_fresh_initialization_still_requires_thirds(self):
        for state in ("all_missing", "new_model"):
            with self.subTest(state=state):
                report = self.audit(torch.full((4, 3), 1.0 / 3.0), state=state)
                self.assertTrue(report["initial_thirds_required"])
                with self.assertRaisesRegex(RuntimeError, "Freshly initialized.*not thirds"):
                    self.audit(self.weights(), state=state)

    def test_initialization_constraint_expires_after_an_optimizer_update(self):
        self.assertFalse(self.audit(self.weights(), state="all_missing", step=1)["initial_thirds_required"])

    def test_exact_resume_overrides_base_initialization_including_checkpoint_zero(self):
        for step in (0, 200, 400, 600, 800, 1000, 1200):
            for state in ("complete", "all_missing", "new_model"):
                with self.subTest(step=step, state=state):
                    report = self.audit(self.weights(), step=step, state=state, resume=True)
                    self.assertFalse(report["initial_thirds_required"])
                    self.assertTrue(report["resuming_from_checkpoint"])

    def test_invalid_weights_are_rejected_for_loaded_and_resumed_models(self):
        cases = (
            (torch.tensor([[float("nan"), 0.3, 0.7]]), FloatingPointError, "non-finite"),
            (torch.tensor([[float("inf"), 0.3, 0.7]]), FloatingPointError, "non-finite"),
            (torch.tensor([[-0.1, 0.4, 0.7]]), RuntimeError, "\\[0, 1\\]"),
            (torch.tensor([[1.1, -0.1, 0.0]]), RuntimeError, "\\[0, 1\\]"),
            (torch.tensor([[0.2, 0.2, 0.2]]), RuntimeError, "sum to one"),
            (torch.ones(1, 2), RuntimeError, "shape"),
            (torch.empty(0, 3), RuntimeError, "shape"),
        )
        for resume in (False, True):
            for values, error_type, message in cases:
                with self.subTest(resume=resume, values=values):
                    with self.assertRaisesRegex(error_type, message):
                        self.audit(values, resume=resume)

    def test_loading_complete_ui_weights_does_not_reinitialize_them(self):
        model = mock.Mock()
        model.named_parameters.return_value = iter([
            ("relation_pyramid.scale_logits", self.weights()),
            ("relation_pbd.box_scale", torch.tensor(0.1)),
        ])
        model.validate_ui_relation_parameters.return_value = {"valid": True}
        report = setup.initialize_or_validate_ui_relation(
            model, {"missing_keys": []}, seed=42, all_missing_reason="test",
        )
        self.assertEqual(report["state"], "complete")
        model.initialize_ui_relation_modules.assert_not_called()
        model.validate_ui_relation_parameters.assert_called_once_with()

    def test_loading_all_missing_ui_parameters_is_the_only_base_checkpoint_fresh_case(self):
        model = mock.Mock()
        names = ["relation_pyramid.scale_logits", "relation_pbd.box_scale"]
        model.named_parameters.return_value = iter((name, object()) for name in names)
        model.initialize_ui_relation_modules.return_value = {"initialized": True}
        report = setup.initialize_or_validate_ui_relation(
            model, {"missing_keys": names}, seed=42, all_missing_reason="all-missing-test",
        )
        self.assertEqual(report["state"], "all_missing")
        model.initialize_ui_relation_modules.assert_called_once_with(42, "all-missing-test")

    def test_partial_checkpoint_still_fails_before_training(self):
        model = mock.Mock()
        model.named_parameters.return_value = iter([
            ("relation_pyramid.scale_logits", object()), ("relation_pbd.box_scale", object()),
        ])
        with self.assertRaisesRegex(RuntimeError, "Partial UI"):
            setup.initialize_or_validate_ui_relation(
                model, {"missing_keys": ["relation_pbd.box_scale"]}, seed=42, all_missing_reason="test",
            )
        model.initialize_ui_relation_modules.assert_not_called()

    def test_actual_trainer_first_batch_writes_loaded_weight_audit_for_both_ranks(self):
        # Execute the production diagnostic method, excluding heavyweight model
        # imports and Trainer initialization. No fake tensor implementation.
        tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
        trainer_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                             and node.name == "StreamPackingMTPTrainer")
        method = next(node for node in trainer_class.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_capture_ui5_batch")
        import numpy as np
        for rank in (0, 1):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as temporary:
                namespace = {
                    "torch": torch, "np": np, "os": os, "osp": os.path, "json": json,
                    "get_rank": lambda: rank, "logger": mock.Mock(),
                    "audit_ui5_detail_scale_weights": setup.audit_ui5_detail_scale_weights,
                }
                exec(compile(ast.Module(body=[method], type_ignores=[]), str(TRAINER), "exec"), namespace)
                trainer = SimpleNamespace(
                    model=SimpleNamespace(config=SimpleNamespace()),
                    state=SimpleNamespace(global_step=0), args=SimpleNamespace(output_dir=temporary),
                    _ui5_real_data_audit_logged=False, _ui5_relation_load_state="complete",
                    _ui5_scale_audit_resuming=False, _add_ui5_scalar=mock.Mock(),
                    _add_ui5_weighted_scalar=mock.Mock(),
                    _tensor_float=lambda value: float(value) if value is not None else None,
                )
                outputs = SimpleNamespace(
                    detail_feature_norm=torch.ones(3), detail_feature_abs_max=torch.ones(3),
                    detail_saturation_fraction=torch.zeros(3), detail_norm_ratio=torch.tensor(1.0),
                    detail_layer_weights=self.weights(),
                )
                namespace["_capture_ui5_batch"](trainer, outputs, {})
                report = json.loads((Path(temporary) / "diagnostics" /
                                     f"first_real_batch_audit_rank{rank}.json").read_text())
                self.assertFalse(report["initial_thirds_required"])
                self.assertEqual(report["scale_weights"], self.weights().detach().tolist())
                self.assertEqual(report["rank"], rank)
                self.assertTrue(trainer._ui5_real_data_audit_logged)
                namespace["logger"].warning.assert_called_once()

    def test_training_wires_process_load_report_and_resume_before_train_call(self):
        tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
        constructors = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "CustomTrainer"]
        self.assertEqual(len(constructors), 1)
        load_state = next(kw.value for kw in constructors[0].keywords if kw.arg == "ui_relation_load_state")
        self.assertEqual(ast.unparse(load_state), "ui_load_report['state']")
        context = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(ast.unparse(target) == "trainer._ui5_scale_audit_resuming" for target in node.targets)]
        self.assertEqual(len(context), 1)
        self.assertEqual(ast.unparse(context[0].value), "checkpoint is not None")
        train_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and ast.unparse(node.func) == "trainer.train"]
        self.assertLess(context[0].lineno, train_calls[0].lineno)


if __name__ == "__main__":
    unittest.main()
