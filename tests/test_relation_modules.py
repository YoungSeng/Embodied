import unittest

import torch

from eaglevl.model.locany.relation_modules import (
    FAMILY_SCALE_PRIOR,
    RelationConditionedDetailPyramid,
    RelationToPBD,
    apply_coordinate_logit_prior,
    apply_soft_gate_logit_prior,
    coordinate_bridge_prediction_groups,
    coordinate_gaussian_prior,
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
        output = module(
            hidden,
            input_ids,
            torch.tensor([7]),
            torch.randn(1, 8),
            torch.randn(1, 8),
            box_start_token_id=5,
            text_mask_token_id=99,
            block_size=6,
        )
        self.assertEqual(output.box_anchor_hidden.shape, (2, 12))
        self.assertEqual(output.box_anchor_samples.tolist(), [0, 0])
        self.assertEqual(output.active_positions.tolist(), [1, 3])
        unchanged = torch.tensor([0, 2, 4, 5, 6])
        self.assertTrue(
            torch.equal(output.hidden_states[0, unchanged], hidden[0, unchanged])
        )

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
        self.assertEqual(output.scale_weights.dtype, torch.float32)
        torch.testing.assert_close(
            output.scale_weights.sum(dim=-1),
            torch.ones(output.scale_weights.shape[0]),
            rtol=0.0,
            atol=1.0e-6,
        )

        pbd = RelationToPBD(8, 12).to(dtype=torch.bfloat16)
        hidden = torch.randn(1, 3, 12, dtype=torch.bfloat16)
        pbd_output = pbd(
            hidden,
            torch.tensor([[1, 5, 2]]),
            torch.tensor([3]),
            torch.full((1, 8), float("nan"), dtype=torch.bfloat16),
            torch.full((1, 8), float("nan"), dtype=torch.bfloat16),
            box_start_token_id=5,
            text_mask_token_id=99,
            block_size=6,
        )
        self.assertTrue(torch.isfinite(pbd_output.hidden_states).all())

    def test_default_parameter_budget_is_far_below_five_percent(self):
        pyramid = RelationConditionedDetailPyramid(1152, 256, 8, 64)
        pbd = RelationToPBD(256, 2048)
        added = sum(parameter.numel() for parameter in pyramid.parameters())
        added += sum(parameter.numel() for parameter in pbd.parameters())
        # Includes the five image-level Gate heads introduced when image
        # defectness was separated from slot objectness.
        self.assertEqual(added, 2_709_915)
        self.assertLess(added, 0.05 * 3_000_000_000)

    def test_family_prior_survives_construction_and_task_router_can_separate_tasks(self):
        module = RelationConditionedDetailPyramid(
            8, 8, 2, 4, task_scale_router=True
        )
        torch.testing.assert_close(
            module.scale_logits.float().softmax(dim=-1), FAMILY_SCALE_PRIOR
        )
        features = tuple(torch.randn(8, 8) for _ in range(3))
        with torch.no_grad():
            module.task_scale_embedding.weight.zero_()
            module.task_scale_embedding.weight[0, 0] = 1.0
            module.task_scale_embedding.weight[1, 1] = 1.0
            module.task_scale_projection.weight.zero_()
            module.task_scale_projection.weight[0, 0] = 2.0
            module.task_scale_projection.weight[2, 1] = 2.0
        output = module(
            features,
            torch.tensor([[2, 2], [2, 2]]),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            image_flags=torch.tensor([1, 1]),
        )
        self.assertFalse(torch.allclose(output.scale_weights[0], output.scale_weights[1]))

    def test_family_prior_audit_accepts_only_the_bfloat16_representation(self):
        initialized_before_cast = RelationConditionedDetailPyramid(
            8, 8, 2, 4, task_scale_router=True
        )
        initialized_before_cast.reset_parameters()
        initialized_before_cast.to(dtype=torch.bfloat16)
        initialized_before_cast.assert_family_scale_prior()

        initialized_after_cast = RelationConditionedDetailPyramid(
            8, 8, 2, 4, task_scale_router=True
        ).to(dtype=torch.bfloat16)
        initialized_after_cast.reset_parameters()
        initialized_after_cast.assert_family_scale_prior()
        torch.testing.assert_close(
            initialized_before_cast.scale_logits,
            initialized_after_cast.scale_logits,
        )
        module = initialized_after_cast
        with torch.no_grad():
            module.scale_logits[0, 0].add_(0.25)
        with self.assertRaisesRegex(RuntimeError, "overwritten"):
            module.assert_family_scale_prior()

    def test_hungarian_loss_is_gt_permutation_invariant_and_slots_are_unique(self):
        torch.manual_seed(91)
        module = RelationConditionedDetailPyramid(
            12, 8, 3, 4, task_scale_router=True, set_localizer=True
        )
        features = tuple(torch.randn(16, 12) for _ in range(3))
        boxes = torch.tensor([[[50., 80., 300., 400.], [620., 500., 920., 900.]]])
        mask = torch.tensor([[True, True]])
        first = module(
            features, torch.tensor([[4, 4]]), torch.tensor([0]), torch.tensor([1]),
            image_flags=torch.tensor([1]), target_boxes=boxes, target_box_mask=mask,
        )
        second = module(
            features, torch.tensor([[4, 4]]), torch.tensor([0]), torch.tensor([1]),
            image_flags=torch.tensor([1]), target_boxes=boxes.flip(1), target_box_mask=mask,
        )
        for name in ("box_l1_loss", "box_giou_loss", "attention_kl_loss", "slot_gate_loss"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))
        matched = first.matched_slot_indices[0, :2]
        self.assertEqual(len(set(matched.tolist())), 2)

    def test_dynamic_teacher_forcing_binds_each_box_to_its_matched_slot(self):
        torch.manual_seed(13)
        module = RelationToPBD(8, 10, dynamic_slot=True)
        ids = torch.tensor([[5, *([99] * 5), 7, 5, *([99] * 5)]])
        output = module(
            torch.randn(1, ids.numel(), 10), ids, torch.tensor([ids.numel()]),
            torch.randn(1, 8), torch.randn(1, 8), 5, 99, 6,
            relation_tokens=torch.randn(1, 3, 8),
            slot_gate_logits=torch.zeros(1, 3),
            coarse_boxes=torch.randn(1, 3, 4),
            matched_slot_indices=torch.tensor([[2, 1, -1]]),
            defect_type=torch.tensor([1]),
        )
        first = output.selected_slot_indices[:6]
        second = output.selected_slot_indices[6:]
        self.assertEqual(set(first.tolist()), {2})
        self.assertEqual(set(second.tolist()), {1})
        self.assertEqual(float(output.duplicate_slot_rate), 0.0)
        anchor_weights = output.routing_weights[output.active_offsets == 0]
        torch.testing.assert_close(
            anchor_weights.sum(dim=-1), torch.ones(anchor_weights.shape[0])
        )
        self.assertTrue(bool(((anchor_weights > 0) & (anchor_weights < 1)).any()))
        output.hidden_states.sum().backward()
        for parameter in (
            module.router_query.weight,
            module.router_key.weight,
            module.router_value.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.float().norm()), 0.0)

    def test_dynamic_inference_coverage_routes_consecutive_boxes_to_distinct_slots(self):
        module = RelationToPBD(8, 10, dynamic_slot=True)
        with torch.no_grad():
            module.router_query.weight.zero_()
            module.router_key.weight.zero_()
            module.router_value.weight.copy_(torch.eye(8))
            module.coverage_gamma.fill_(10.0)
        ids = torch.tensor([[5, *([99] * 5), 7, 5, *([99] * 5)]])
        output = module(
            torch.randn(1, ids.numel(), 10), ids, torch.tensor([ids.numel()]),
            torch.randn(1, 8), torch.randn(1, 8), 5, 99, 6,
            relation_tokens=torch.randn(1, 3, 8),
            slot_gate_logits=torch.zeros(1, 3),
            coarse_boxes=torch.randn(1, 3, 4),
            defect_type=torch.tensor([1]),
        )
        anchors = output.selected_slot_indices[output.active_offsets == 0]
        self.assertEqual(anchors.tolist(), [0, 1])
        self.assertEqual(float(output.duplicate_slot_rate), 0.0)

    def test_dynamic_routing_state_is_isolated_between_packed_samples(self):
        module = RelationToPBD(8, 10, dynamic_slot=True)
        ids = torch.tensor([[5, *([99] * 5), 5, *([99] * 5)]])
        output = module(
            torch.randn(1, ids.numel(), 10), ids, torch.tensor([6, 6]),
            torch.randn(2, 8), torch.randn(2, 8), 5, 99, 6,
            relation_tokens=torch.randn(2, 3, 8),
            slot_gate_logits=torch.zeros(2, 3),
            coarse_boxes=torch.randn(2, 3, 4),
            matched_slot_indices=torch.tensor([[2, -1, -1], [1, -1, -1]]),
            defect_type=torch.tensor([1, 4]),
        )
        anchors = output.selected_slot_indices[output.active_offsets == 0]
        self.assertEqual(anchors.tolist(), [2, 1])
        self.assertEqual(output.active_samples[output.active_offsets == 0].tolist(), [0, 1])

    def test_single_slot_dynamic_pbd_degenerates_to_legacy_selected_token(self):
        torch.manual_seed(21)
        legacy = RelationToPBD(8, 10)
        dynamic = RelationToPBD(8, 10, dynamic_slot=True)
        dynamic.semantic_projection.load_state_dict(legacy.semantic_projection.state_dict())
        dynamic.box_projection.load_state_dict(legacy.box_projection.state_dict())
        with torch.no_grad():
            dynamic.semantic_scale.copy_(legacy.semantic_scale)
            dynamic.box_scale.copy_(legacy.box_scale)
        hidden = torch.randn(1, 6, 10)
        relation_summary = torch.randn(1, 8)
        token = torch.randn(1, 8)
        common = dict(
            hidden_states=hidden,
            input_ids=torch.tensor([[5, *([99] * 5)]]),
            sub_sample_lengths=torch.tensor([6]),
            relation_summary=relation_summary,
            best_relation_token=token,
            box_start_token_id=5,
            text_mask_token_id=99,
            block_size=6,
        )
        expected = legacy(**common)
        actual = dynamic(
            **common,
            relation_tokens=token[:, None, :],
            slot_gate_logits=torch.zeros(1, 1),
            coarse_boxes=torch.zeros(1, 1, 4),
            defect_type=torch.tensor([0]),
        )
        torch.testing.assert_close(actual.hidden_states, expected.hidden_states)

    def test_coordinate_prior_touches_only_coordinate_vocabulary_and_zero_is_identity(self):
        logits = torch.randn(1, 2, 32)
        kwargs = dict(
            active_positions=torch.tensor([0]),
            active_offsets=torch.tensor([0]),
            active_samples=torch.tensor([0]),
            selected_coarse_boxes=torch.tensor([[500., 0., 0., 0.]]),
            defect_type=torch.tensor([0]),
            coord_start_token_id=10,
            coord_end_token_id=20,
        )
        identity = apply_coordinate_logit_prior(
            logits, task_lambdas=torch.zeros(5), **kwargs
        )
        torch.testing.assert_close(identity, logits)
        changed = apply_coordinate_logit_prior(
            logits, task_lambdas=torch.ones(5), **kwargs
        )
        torch.testing.assert_close(changed[..., :10], logits[..., :10])
        torch.testing.assert_close(changed[..., 21:], logits[..., 21:])
        self.assertFalse(torch.allclose(changed[..., 10:21], logits[..., 10:21]))

    def test_training_and_inference_coordinate_priors_are_numerically_equal(self):
        logits = torch.randn(1, 6, 32)
        positions = torch.arange(6)
        offsets = torch.arange(6)
        samples = torch.zeros(6, dtype=torch.long)
        boxes = torch.tensor([[100., 200., 700., 800.]]).repeat(6, 1)
        tasks = torch.tensor([2])
        lambdas = torch.tensor([0., 0., 0.7, 0., 0.])
        inference = apply_coordinate_logit_prior(
            logits, positions, offsets, samples, boxes, tasks, lambdas, 10, 20
        )
        training_prior = coordinate_gaussian_prior(
            offsets, samples, boxes, tasks, lambdas, 11
        )
        expected = logits.clone()
        expected.reshape(-1, 32)[positions, 10:21] += training_prior.to(logits.dtype)
        torch.testing.assert_close(inference, expected)

    def test_coordinate_bridge_selects_four_ar_and_mtp_states_without_crossing(self):
        boxes = torch.tensor([[100., 200., 700., 800.]])
        ar = coordinate_bridge_prediction_groups(
            torch.tensor([[5, 10, 11, 12, 13, 6, 5, 10]]),
            torch.tensor([6, 2]),
            torch.tensor([0, 6]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            boxes.repeat(2, 1),
            10,
            20,
        )
        self.assertEqual(ar[0].tolist(), [0, 1, 2, 3, 6, 7])
        self.assertEqual(ar[1].tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(ar[2].tolist(), [0, 1, 2, 3, 0, 1])

        mtp = coordinate_bridge_prediction_groups(
            torch.tensor([[5, 99, 99, 99, 99, 99]]),
            torch.tensor([6]),
            torch.arange(6),
            torch.zeros(6, dtype=torch.long),
            torch.zeros(6, dtype=torch.long),
            torch.arange(6),
            boxes.repeat(6, 1),
            10,
            20,
        )
        self.assertEqual(mtp[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(mtp[2].tolist(), [0, 1, 2, 3])

    def test_soft_gate_beta_zero_is_tokenwise_identity(self):
        logits = torch.randn(2, 3, 20)
        output = apply_soft_gate_logit_prior(
            logits, torch.tensor([0.2, 0.8]), torch.tensor([0, 1]),
            torch.zeros(5), box_token_id=4, none_token_id=7,
        )
        torch.testing.assert_close(output, logits)


if __name__ == "__main__":
    unittest.main()
