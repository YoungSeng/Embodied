import unittest

import torch

from eaglevl.model.locany.relation_modules import (
    RelationConditionedDetailPyramid,
    RelationToPBD,
)


class RelationModulesTest(unittest.TestCase):
    def test_relation_pyramid_shapes_losses_and_backward(self):
        torch.manual_seed(3)
        module = RelationConditionedDetailPyramid(
            vision_hidden_size=32,
            detail_hidden_size=16,
            num_slots=2,
            adapter_bottleneck=4,
        )
        # Two 4x4 screenshots, packed exactly like MoonViT.
        pyramid = tuple(torch.randn(32, 32, requires_grad=True) for _ in range(3))
        grid_hws = torch.tensor([[4, 4], [4, 4]], dtype=torch.long)
        families = torch.tensor([0, 3])
        defect_types = torch.tensor([0, 4])
        boxes = torch.zeros(2, 2, 4)
        boxes[0, 0] = torch.tensor([100, 100, 400, 400])
        box_mask = torch.tensor([[True, False], [False, False]])

        output = module(
            pyramid,
            grid_hws,
            families,
            defect_types,
            image_flags=torch.tensor([1, 1]),
            target_boxes=boxes,
            target_box_mask=box_mask,
        )
        self.assertEqual(output.relation_tokens.shape, (2, 2, 16))
        self.assertEqual(output.p_defect.shape, (2,))
        self.assertEqual(output.coarse_boxes.shape, (2, 2, 4))
        self.assertEqual(len(output.query_attention), 2)
        self.assertIsNotNone(output.gate_loss)
        self.assertIsNotNone(output.attention_loss)
        (output.gate_loss + output.attention_loss + output.relation_tokens.mean()).backward()
        self.assertIsNotNone(module.scale_logits.grad)

    def test_pbd_only_changes_box_anchor(self):
        module = RelationToPBD(relation_hidden_size=8, language_hidden_size=12)
        hidden = torch.randn(1, 7, 12)
        input_ids = torch.tensor([[1, 5, 2, 5, 3, 4, 6]])
        enhanced, anchors, samples = module(
            hidden,
            input_ids,
            torch.tensor([7]),
            torch.randn(1, 8),
            torch.randn(1, 8),
            box_start_token_id=5,
        )
        self.assertEqual(anchors.shape, (2, 12))
        self.assertEqual(samples.tolist(), [0, 0])
        unchanged = torch.tensor([0, 2, 4, 5, 6])
        self.assertTrue(torch.equal(enhanced[0, unchanged], hidden[0, unchanged]))

    def test_bfloat_relation_path_contains_nonfinite_values(self):
        module = RelationConditionedDetailPyramid(
            vision_hidden_size=8,
            detail_hidden_size=8,
            num_slots=2,
            adapter_bottleneck=4,
        ).to(dtype=torch.bfloat16)
        with torch.no_grad():
            module.evidence_queries[0, 0, 0] = float("nan")
        pyramid = tuple(torch.randn(4, 8, dtype=torch.bfloat16) for _ in range(3))
        output = module(
            pyramid,
            torch.tensor([[2, 2]]),
            torch.tensor([0]),
            torch.tensor([0]),
            image_flags=torch.tensor([1]),
        )
        self.assertTrue(torch.isfinite(output.relation_tokens).all())
        self.assertTrue(torch.isfinite(output.p_defect).all())

        pbd = RelationToPBD(8, 12).to(dtype=torch.bfloat16)
        hidden = torch.randn(1, 3, 12, dtype=torch.bfloat16)
        enhanced, _, _ = pbd(
            hidden,
            torch.tensor([[1, 5, 2]]),
            torch.tensor([3]),
            torch.full((1, 8), float("nan"), dtype=torch.bfloat16),
            torch.full((1, 8), float("nan"), dtype=torch.bfloat16),
            box_start_token_id=5,
        )
        self.assertTrue(torch.isfinite(enhanced).all())

    def test_default_parameter_budget_is_far_below_five_percent(self):
        pyramid = RelationConditionedDetailPyramid(1152, 256, 8, 64)
        pbd = RelationToPBD(256, 2048)
        added = sum(parameter.numel() for parameter in pyramid.parameters())
        added += sum(parameter.numel() for parameter in pbd.parameters())
        self.assertEqual(added, 2_624_790)
        self.assertLess(added, 0.05 * 3_000_000_000)


if __name__ == "__main__":
    unittest.main()
