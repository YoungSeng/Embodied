import unittest

import torch
from torch import nn
import torch.nn.functional as F

from eaglevl.model.locany.relation_modules import (
    RelationConditionedDetailPyramid,
    RelationToPBD,
    TaskConditionedSetDecoder,
    hungarian_assignment,
    pairwise_generalized_box_iou,
)


class TaskConditionedSetDecoderTest(unittest.TestCase):
    def test_objectness_and_image_gate_do_not_scale_m31_relation_values(self):
        module = RelationConditionedDetailPyramid(
            vision_hidden_size=16,
            detail_hidden_size=16,
            num_slots=8,
            adapter_bottleneck=4,
            task_scale_router=True,
            set_localizer=True,
            task_hard_router=True,
            task_experts=True,
            set_decoder=True,
            set_decoder_layers=1,
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in module.image_gate_heads.parameters())
        )
        features = tuple(torch.randn(16, 16) for _ in range(3))
        arguments = (
            features,
            torch.tensor([[4, 4]]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
        before = module(*arguments, image_flags=torch.tensor([1]))
        with torch.no_grad():
            module.task_set_decoder.objectness_heads[-1].bias.fill_(-30.0)
            module.image_gate_heads[0][-1].bias.fill_(30.0)
        after = module(*arguments, image_flags=torch.tensor([1]))
        torch.testing.assert_close(before.relation_tokens, after.relation_tokens)
        self.assertFalse(torch.equal(before.slot_objectness_logits, after.slot_objectness_logits))

    def test_shapes_boxes_and_finiteness(self):
        decoder = TaskConditionedSetDecoder(16, num_object_queries=8, num_decoder_layers=3)
        output = decoder(torch.randn(3, 20, 16), torch.tensor([0, 2, 4]))
        self.assertEqual(output.global_task_token.shape, (3, 16))
        self.assertEqual(output.slot_tokens.shape, (3, 8, 16))
        self.assertEqual(output.slot_objectness_logits.shape, (3, 8))
        self.assertEqual(output.slot_boxes_norm.shape, (3, 8, 4))
        self.assertTrue(torch.isfinite(output.slot_boxes_norm).all())
        self.assertTrue(((0 <= output.slot_boxes_norm) & (output.slot_boxes_norm <= 1)).all())
        self.assertTrue((output.slot_boxes_norm[..., 0] <= output.slot_boxes_norm[..., 2]).all())
        self.assertTrue((output.slot_boxes_norm[..., 1] <= output.slot_boxes_norm[..., 3]).all())

    def test_direct_refinement_can_move_more_than_old_limit(self):
        decoder = TaskConditionedSetDecoder(16, num_object_queries=8, num_decoder_layers=1)
        with torch.no_grad():
            decoder.shared_box_deltas[0].bias.copy_(torch.tensor([5.0, 5.0, 0.0, 0.0]))
        initial = torch.sigmoid(decoder.reference_box_logits)[0, :2]
        output = decoder(torch.randn(1, 10, 16), torch.tensor([0]))
        center = (output.slot_boxes_norm[0, 0, :2] + output.slot_boxes_norm[0, 0, 2:]) / 2
        self.assertGreater(float(((center - initial).abs().max() * 1000.0).detach()), 100.0)

    def test_hungarian_total_is_permutation_invariant_and_slots_are_unique(self):
        predicted = torch.tensor([[0., 0., 2., 2.], [8., 8., 10., 10.], [4., 4., 6., 6.]])
        target = torch.tensor([[0., 0., 2., 2.], [8., 8., 10., 10.]])

        def total(boxes):
            cost = torch.cdist(predicted, boxes, p=1) + 2 * (1 - pairwise_generalized_box_iou(predicted, boxes))
            slots, targets = hungarian_assignment(cost)
            self.assertEqual(slots.unique().numel(), 2)
            return cost[slots, targets].sum()

        torch.testing.assert_close(total(target), total(target.flip(0)))

    def test_unmatched_slots_receive_no_object_supervision(self):
        decoder = TaskConditionedSetDecoder(16, num_object_queries=8, num_decoder_layers=1)
        output = decoder(torch.randn(1, 10, 16), torch.tensor([0]))
        targets = torch.zeros_like(output.slot_objectness_logits)
        targets[0, 0] = 1.0
        F.binary_cross_entropy_with_logits(output.slot_objectness_logits, targets).backward()
        self.assertIsNotNone(decoder.objectness_heads[0].bias.grad)
        self.assertGreater(float(decoder.objectness_heads[0].bias.grad.abs().sum().detach()), 0.0)

    def test_state_dict_round_trip_preserves_routes_and_boxes(self):
        torch.manual_seed(11)
        decoder = TaskConditionedSetDecoder(16, num_object_queries=8, num_decoder_layers=2)
        memory = torch.randn(2, 12, 16)
        tasks = torch.tensor([1, 4])
        expected = decoder(memory, tasks)
        restored = TaskConditionedSetDecoder(16, num_object_queries=8, num_decoder_layers=2)
        restored.load_state_dict(decoder.state_dict(), strict=True)
        actual = restored(memory, tasks)
        torch.testing.assert_close(actual.slot_tokens, expected.slot_tokens)
        torch.testing.assert_close(actual.slot_boxes_norm, expected.slot_boxes_norm)
        torch.testing.assert_close(actual.slot_objectness_logits, expected.slot_objectness_logits)

    def test_m31_decoder_pbd_checkpoint_round_trip_preserves_logits_and_route(self):
        torch.manual_seed(29)
        bundle = nn.ModuleDict(
            {
                "decoder": TaskConditionedSetDecoder(
                    16, num_object_queries=8, num_decoder_layers=2
                ),
                "pbd": RelationToPBD(
                    16,
                    12,
                    dynamic_slot=True,
                    coordinate_bridge=True,
                    task_experts=True,
                    separate_global_geometry=True,
                ),
            }
        )
        memory = torch.randn(1, 12, 16)
        tasks = torch.tensor([2])
        hidden = torch.randn(1, 12, 12)
        ids = torch.tensor([[5, *([99] * 5), 5, *([99] * 5)]])
        lm_head = torch.randn(31, 12)

        def run(module):
            decoded = module["decoder"](memory, tasks)
            pbd = module["pbd"](
                hidden,
                ids,
                torch.tensor([12]),
                decoded.global_task_token,
                decoded.slot_tokens[:, 0],
                5,
                99,
                6,
                relation_tokens=decoded.slot_tokens,
                slot_objectness_logits=decoded.slot_objectness_logits,
                coarse_boxes=decoded.slot_boxes_norm1000,
                matched_slot_indices=torch.tensor([[2, 1]]),
                defect_type=tasks,
            )
            return decoded, pbd, F.linear(pbd.hidden_states, lm_head)

        expected_decoder, expected_pbd, expected_logits = run(bundle)
        restored = nn.ModuleDict(
            {
                "decoder": TaskConditionedSetDecoder(
                    16, num_object_queries=8, num_decoder_layers=2
                ),
                "pbd": RelationToPBD(
                    16,
                    12,
                    dynamic_slot=True,
                    coordinate_bridge=True,
                    task_experts=True,
                    separate_global_geometry=True,
                ),
            }
        )
        restored.load_state_dict(bundle.state_dict(), strict=True)
        actual_decoder, actual_pbd, actual_logits = run(restored)
        torch.testing.assert_close(
            actual_decoder.slot_boxes_norm, expected_decoder.slot_boxes_norm
        )
        self.assertEqual(
            actual_pbd.selected_slot_indices.tolist(),
            expected_pbd.selected_slot_indices.tolist(),
        )
        torch.testing.assert_close(actual_pbd.hidden_states, expected_pbd.hidden_states)
        torch.testing.assert_close(actual_logits, expected_logits)


if __name__ == "__main__":
    unittest.main()
