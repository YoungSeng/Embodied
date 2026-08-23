from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eaglevl.model.locany.relation_modules import (
    RELATION_FAMILIES,
    DEFAULT_UI_DETAIL_LAYERS,
    UI_RELATION_PROMPT_SPECS,
    RelationConditionedDetailPyramid,
    RelationToPBD,
    class_balanced_focal_loss,
    match_ui_relation_prompt,
    pbd_active_delta_norm,
    pbd_prediction_positions,
    passes_relation_gate,
    relation_gate_output_override,
)
from eaglevl.model.locany.ui_relation_setup import (
    configure_ui5_model_config,
    initialize_or_validate_ui_relation,
)


class UIRelationPipelineTest(unittest.TestCase):
    BOX = 101
    MASK = 102
    BLOCK_SIZE = 6

    def test_training_and_checkpoint0_share_ui5_config_builder(self):
        config = SimpleNamespace(
            text_config=SimpleNamespace(),
            vision_config=SimpleNamespace(),
            relation_gate_thresholds={"text_overflow": 0.23},
        )
        configure_ui5_model_config(
            config,
            attn_implementation="sdpa",
            image_token_index=11,
            block_size=6,
            causal_attn=False,
            text_mask_token_id=12,
            null_token_id=13,
            box_start_token_id=14,
            box_end_token_id=15,
            coord_start_token_id=16,
            coord_end_token_id=17,
            ref_start_token_id=18,
            ref_end_token_id=19,
            none_token_id=20,
            enable_ui_relation=True,
            relation_detail_hidden_size=256,
            relation_num_slots=8,
            relation_adapter_bottleneck=64,
            relation_detail_layers=(5, 15, 26),
            relation_gate_loss_weight=1.0,
            relation_slot_gate_loss_weight=0.1,
            relation_attention_loss_weight=0.1,
            relation_gate_threshold=0.5,
            relation_focal_beta=0.999,
            relation_focal_gamma=2.0,
        )
        self.assertEqual(config.text_config.block_size, 6)
        self.assertFalse(config.text_config.causal_attn)
        self.assertEqual(config.text_config.text_mask_token_id, 12)
        self.assertEqual(config.relation_detail_layers, [5, 15, 26])
        self.assertEqual(config.relation_gate_mode, "observe")
        self.assertEqual(config.relation_gate_thresholds["text_overflow"], 0.23)

    def test_partial_ui_checkpoint_is_rejected(self):
        class FakeModel:
            def named_parameters(self):
                return iter(
                    (
                        ("relation_pyramid.scale_logits", object()),
                        ("relation_pbd.box_scale", object()),
                    )
                )

            def initialize_ui_relation_modules(self, seed, reason):
                return {"seed": seed, "reason": reason}

            def validate_ui_relation_parameters(self):
                return {"valid": True}

        with self.assertRaisesRegex(RuntimeError, "Partial UI"):
            initialize_or_validate_ui_relation(
                FakeModel(),
                {"missing_keys": ["relation_pbd.box_scale"]},
                seed=42,
                all_missing_reason="test",
            )

    def test_detail_pyramid_uses_fixed_moonvit_layers(self):
        self.assertEqual(DEFAULT_UI_DETAIL_LAYERS, (5, 15, 26))

    def make_pyramid(self):
        torch.manual_seed(17)
        module = RelationConditionedDetailPyramid(
            vision_hidden_size=12,
            detail_hidden_size=8,
            num_slots=2,
            adapter_bottleneck=4,
        )
        features = tuple(
            torch.randn(16, 12, requires_grad=True) for _ in range(3)
        )
        output = module(
            features,
            torch.tensor([[4, 4]]),
            torch.tensor([0]),
            torch.tensor([0]),
            image_flags=torch.tensor([1]),
            target_boxes=torch.tensor([[[100.0, 100.0, 500.0, 500.0]]]),
            target_box_mask=torch.tensor([[True]]),
        )
        return module, features, output

    def test_five_fixed_prompts_have_no_default_route(self):
        expected = {
            "text_overflow": ("boundary", 0),
            "cropping": ("boundary", 1),
            "occlusion": ("pairwise", 2),
            "text_ellipsis": ("text", 3),
            "content_missing": ("presence", 4),
        }
        self.assertEqual(len(UI_RELATION_PROMPT_SPECS), 5)
        for spec in UI_RELATION_PROMPT_SPECS:
            matched = match_ui_relation_prompt(spec.prompt)
            self.assertIsNotNone(matched, spec.prompt)
            self.assertEqual(matched.task_name, spec.task_name)
            family_name, defect_type = expected[spec.task_name]
            self.assertEqual(RELATION_FAMILIES[matched.relation_family], family_name)
            self.assertEqual(matched.defect_type, defect_type)

    def test_training_and_patched_inference_relation_shapes_match(self):
        training, features, training_output = self.make_pyramid()
        inference = RelationConditionedDetailPyramid(
            vision_hidden_size=12,
            detail_hidden_size=8,
            num_slots=2,
            adapter_bottleneck=4,
        )
        incompatible = inference.load_state_dict(training.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        inference_output = inference(
            tuple(feature.detach().clone() for feature in features),
            torch.tensor([[4, 4]]),
            torch.tensor([0]),
            torch.tensor([0]),
            image_flags=torch.tensor([1]),
        )
        self.assertEqual(
            training_output.relation_tokens.shape,
            inference_output.relation_tokens.shape,
        )
        self.assertEqual(
            training_output.relation_summary.shape,
            inference_output.relation_summary.shape,
        )
        self.assertEqual(training_output.image_gate_logits.shape, (1,))
        self.assertEqual(training_output.slot_gate_logits.shape, (1, 2))
        self.assertEqual(training_output.p_defect.shape, (1,))
        self.assertIsNotNone(training_output.image_gate_loss)
        self.assertIsNotNone(training_output.slot_gate_loss)

    def test_detail_scale_weights_start_as_exact_thirds(self):
        module, _, output = self.make_pyramid()
        torch.testing.assert_close(
            module.scale_logits,
            torch.zeros_like(module.scale_logits),
        )
        torch.testing.assert_close(
            output.scale_weights.sum(dim=-1),
            torch.ones(output.scale_weights.shape[0]),
        )
        torch.testing.assert_close(
            output.scale_weights,
            torch.full_like(output.scale_weights, 1.0 / 3.0),
        )
        for head in (*module.gate_heads, *module.image_gate_heads):
            torch.testing.assert_close(
                head[-1].bias,
                torch.full_like(head[-1].bias, -2.0),
            )

    def test_image_focal_loss_does_not_broadcast_batch_to_square(self):
        logits = torch.tensor([-2.0, 1.0], requires_grad=True)
        targets = torch.tensor([0.0, 1.0])
        defect_type = torch.tensor([0, 1])
        loss = class_balanced_focal_loss(logits, targets, defect_type)
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        self.assertEqual(logits.grad.shape, logits.shape)

    def test_relation_gate_and_pbd_checkpoint_roundtrip_is_strict(self):
        modules = nn.ModuleDict(
            {
                "relation_pyramid": RelationConditionedDetailPyramid(
                    12, 8, 2, 4
                ),
                "relation_pbd": RelationToPBD(8, 10),
            }
        )
        buffer = io.BytesIO()
        torch.save(modules.state_dict(), buffer)
        buffer.seek(0)
        restored = nn.ModuleDict(
            {
                "relation_pyramid": RelationConditionedDetailPyramid(
                    12, 8, 2, 4
                ),
                "relation_pbd": RelationToPBD(8, 10),
            }
        )
        incompatible = restored.load_state_dict(
            torch.load(buffer, weights_only=True), strict=True
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_gate_threshold_changes_none_bbox_decision(self):
        p_defect = torch.tensor(0.49)
        self.assertFalse(passes_relation_gate(p_defect, 0.5))
        self.assertTrue(passes_relation_gate(p_defect, 0.4))
        self.assertEqual(
            relation_gate_output_override(p_defect, 0.5), "<box>none</box>"
        )
        self.assertIsNone(relation_gate_output_override(p_defect, 0.4))

    def test_relation_switch_changes_bbox_hidden_and_logits(self):
        torch.manual_seed(9)
        pbd = RelationToPBD(8, 10)
        hidden = torch.randn(1, 2, 10)
        relation_summary = torch.randn(1, 8)
        best_relation = torch.randn(1, 8)
        enabled = pbd.enhance_prediction_hidden(
            hidden, relation_summary, best_relation
        )
        delta = enabled - hidden
        self.assertGreater(float(delta.norm()), 0.0)
        head = torch.randn(7, 10)
        disabled_logits = torch.nn.functional.linear(hidden, head)
        enabled_logits = torch.nn.functional.linear(enabled, head)
        self.assertFalse(torch.allclose(disabled_logits, enabled_logits))

    def _run_pbd(self, input_ids, sub_sample_lengths, num_samples=1):
        torch.manual_seed(29)
        pbd = RelationToPBD(8, 10)
        hidden = torch.randn(1, len(input_ids), 10)
        relation_summary = torch.randn(num_samples, 8)
        best_relation = torch.randn(num_samples, 8)
        output = pbd(
            hidden_states=hidden,
            input_ids=torch.tensor([input_ids]),
            sub_sample_lengths=torch.tensor(sub_sample_lengths),
            relation_summary=relation_summary,
            best_relation_token=best_relation,
            box_start_token_id=self.BOX,
            text_mask_token_id=self.MASK,
            block_size=self.BLOCK_SIZE,
        )
        return pbd, hidden, relation_summary, best_relation, output

    def test_ar_pbd_enhances_anchor_only(self):
        _, hidden, _, _, output = self._run_pbd(
            [17, self.BOX, 201, 202, 203, 204, 18], [7]
        )
        delta = (output.hidden_states - hidden).norm(dim=-1).reshape(-1)
        self.assertEqual(output.active_positions.tolist(), [1])
        self.assertGreater(float(delta[1]), 0.0)
        torch.testing.assert_close(
            torch.cat((delta[:1], delta[2:])), torch.zeros(6)
        )

    def test_mtp_pbd_enhances_all_six_positions(self):
        ids = [17, self.BOX, *([self.MASK] * 5), 18]
        _, hidden, _, _, output = self._run_pbd(ids, [8])
        delta = (output.hidden_states - hidden).norm(dim=-1).reshape(-1)
        self.assertEqual(output.active_positions.tolist(), list(range(1, 7)))
        self.assertTrue(bool((delta[1:7] > 0).all()))
        torch.testing.assert_close(delta[[0, 7]], torch.zeros(2))

    def test_mtp_duplicate_history_anchor_is_not_a_seventh_position(self):
        ids = [17, self.BOX, self.BOX, *([self.MASK] * 5)]
        _, hidden, _, _, output = self._run_pbd(ids, [8])
        delta = (output.hidden_states - hidden).norm(dim=-1).reshape(-1)
        self.assertEqual(output.active_positions.tolist(), list(range(2, 8)))
        torch.testing.assert_close(delta[:2], torch.zeros(2))
        self.assertTrue(bool((delta[2:] > 0).all()))

    def test_pbd_does_not_cross_packed_sample_boundary(self):
        ids = [17, self.BOX, *([self.MASK] * 5), 18]
        _, hidden, _, _, output = self._run_pbd(ids, [2, 6], num_samples=2)
        delta = (output.hidden_states - hidden).norm(dim=-1).reshape(-1)
        self.assertEqual(output.active_positions.tolist(), [1])
        self.assertEqual(output.active_samples.tolist(), [0])
        self.assertGreater(float(delta[1]), 0.0)
        torch.testing.assert_close(delta[2:], torch.zeros(6))

    def test_training_inference_pbd_logits_are_numerically_equal(self):
        torch.manual_seed(41)
        training_pbd = RelationToPBD(8, 10)
        inference_pbd = RelationToPBD(8, 10)
        inference_pbd.load_state_dict(training_pbd.state_dict(), strict=True)
        training_ids = torch.tensor([[self.BOX, *([self.MASK] * 5)]])
        training_hidden = torch.randn(1, 6, 10)
        # MTP generation retains the emitted <box> in history and then copies
        # it as the six-position prediction block anchor.
        inference_ids = torch.tensor(
            [[self.BOX, self.BOX, *([self.MASK] * 5)]]
        )
        history_hidden = torch.randn(1, 1, 10)
        inference_hidden = torch.cat((history_hidden, training_hidden), dim=1)
        relation_summary = torch.randn(1, 8)
        best_relation = torch.randn(1, 8)
        lm_head = torch.randn(23, 10)

        shared_kwargs = {
            "relation_summary": relation_summary,
            "best_relation_token": best_relation,
            "box_start_token_id": self.BOX,
            "text_mask_token_id": self.MASK,
            "block_size": self.BLOCK_SIZE,
        }
        training_output = training_pbd(
            hidden_states=training_hidden.clone(),
            input_ids=training_ids,
            sub_sample_lengths=torch.tensor([6]),
            **shared_kwargs,
        )
        inference_output = inference_pbd(
            hidden_states=inference_hidden.clone(),
            input_ids=inference_ids,
            sub_sample_lengths=torch.tensor([7]),
            **shared_kwargs,
        )
        training_logits = torch.nn.functional.linear(
            training_output.hidden_states, lm_head
        )
        inference_logits = torch.nn.functional.linear(
            inference_output.hidden_states[:, 1:, :], lm_head
        )
        torch.testing.assert_close(
            training_output.hidden_states,
            inference_output.hidden_states[:, 1:, :],
        )
        torch.testing.assert_close(training_logits, inference_logits)
        torch.testing.assert_close(
            inference_output.hidden_states[:, :1, :], history_hidden
        )
        baseline_logits = torch.nn.functional.linear(training_hidden, lm_head)
        changed = (training_logits - baseline_logits).abs().sum(dim=-1)
        self.assertTrue(bool((changed > 0).all()))

    def test_pbd_delta_norm_uses_active_positions_only(self):
        ids = [17] * 1000 + [self.BOX, *([self.MASK] * 5)]
        _, hidden, _, _, output = self._run_pbd(ids, [1006])
        measured = pbd_active_delta_norm(
            hidden, output.hidden_states, output.active_positions
        )
        manual = (
            output.hidden_states[:, -6:, :] - hidden[:, -6:, :]
        ).float().norm(dim=-1).mean()
        self.assertEqual(output.active_positions.numel(), 6)
        torch.testing.assert_close(measured, manual)

    def test_position_selector_reports_ar_and_mtp_blocks(self):
        positions, samples = pbd_prediction_positions(
            torch.tensor(
                [[self.BOX, 201, self.BOX, *([self.MASK] * 5)]]
            ),
            torch.tensor([8]),
            self.BOX,
            self.MASK,
            self.BLOCK_SIZE,
        )
        self.assertEqual(positions.tolist(), [0, 2, 3, 4, 5, 6, 7])
        self.assertEqual(samples.tolist(), [0] * 7)

    def test_position_selector_uses_configured_block_size(self):
        positions, _ = pbd_prediction_positions(
            torch.tensor([[self.BOX, self.MASK, self.MASK, self.MASK, 17]]),
            torch.tensor([5]),
            self.BOX,
            self.MASK,
            block_size=4,
        )
        self.assertEqual(positions.tolist(), [0, 1, 2, 3])

    def test_all_detail_levels_and_pbd_receive_nonzero_gradients(self):
        module, features, output = self.make_pyramid()
        pbd = RelationToPBD(8, 10)
        hidden = torch.randn(1, self.BLOCK_SIZE, 10, requires_grad=True)
        pbd_output = pbd(
            hidden_states=hidden,
            input_ids=torch.tensor(
                [[self.BOX, *([self.MASK] * (self.BLOCK_SIZE - 1))]]
            ),
            sub_sample_lengths=torch.tensor([self.BLOCK_SIZE]),
            relation_summary=output.relation_summary,
            best_relation_token=output.best_relation_token,
            box_start_token_id=self.BOX,
            text_mask_token_id=self.MASK,
            block_size=self.BLOCK_SIZE,
        )
        loss = (
            output.gate_loss
            + output.attention_loss
            + pbd_output.hidden_states.square().mean()
            + output.relation_tokens.square().mean()
        )
        loss.backward()
        for index, feature in enumerate(features):
            self.assertIsNotNone(feature.grad, f"detail feature {index}")
            self.assertGreater(float(feature.grad.abs().sum()), 0.0)
            projection_grad = module.level_projections[index][1].weight.grad
            self.assertIsNotNone(projection_grad)
            self.assertGreater(float(projection_grad.abs().sum()), 0.0)
        self.assertGreater(
            float(pbd.semantic_projection[1].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(float(pbd.box_projection[1].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
