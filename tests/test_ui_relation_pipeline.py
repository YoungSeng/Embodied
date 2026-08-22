from __future__ import annotations

import io
import unittest

import torch
from torch import nn

from eaglevl.model.locany.relation_modules import (
    RELATION_FAMILIES,
    DEFAULT_UI_DETAIL_LAYERS,
    UI_RELATION_PROMPT_SPECS,
    RelationConditionedDetailPyramid,
    RelationToPBD,
    match_ui_relation_prompt,
    passes_relation_gate,
    relation_gate_output_override,
)


class UIRelationPipelineTest(unittest.TestCase):
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

    def test_all_detail_levels_and_pbd_receive_nonzero_gradients(self):
        module, features, output = self.make_pyramid()
        pbd = RelationToPBD(8, 10)
        hidden = torch.randn(1, 3, 10, requires_grad=True)
        enhanced = pbd.enhance_prediction_hidden(
            hidden,
            output.relation_summary,
            output.best_relation_token,
        )
        loss = (
            output.gate_loss
            + output.attention_loss
            + enhanced.square().mean()
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
