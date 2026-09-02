from __future__ import annotations

import unittest

from eaglevl.train.optimizer_utils import (
    optimizer_learning_rates,
    optimizer_parameters,
    two_learning_rate_parameter_groups,
)


class OptimizerParametersTest(unittest.TestCase):
    def setUp(self):
        self.first = object()
        self.second = object()

    def test_reads_standard_optimizer_param_groups(self):
        class Optimizer:
            param_groups = [{"params": [self.first, self.second]}]

        self.assertEqual(
            optimizer_parameters(Optimizer()), [self.first, self.second]
        )

    def test_reads_dummy_optim_grouped_params(self):
        class DummyOptim:
            params = [
                {"params": [self.first], "lr": 1.0e-5},
                {"params": [self.second], "lr": 2.0e-5},
            ]

        self.assertEqual(
            optimizer_parameters(DummyOptim()), [self.first, self.second]
        )

    def test_materializes_dummy_optim_iterator_for_deepspeed(self):
        class DummyOptim:
            def __init__(self, params):
                self.params = iter(params)

        optimizer = DummyOptim([self.first, self.second])
        self.assertEqual(
            optimizer_parameters(optimizer), [self.first, self.second]
        )
        self.assertEqual(list(optimizer.params), [self.first, self.second])

    def test_unsupported_optimizer_has_clear_error(self):
        with self.assertRaisesRegex(TypeError, "param_groups.*params"):
            optimizer_parameters(object())

    def test_two_learning_rate_groups_cover_inherited_and_m32_parameters(self):
        class Parameter:
            requires_grad = True

            def __init__(self, size):
                self.size = size

        inherited = Parameter(1)
        pyramid = Parameter(2)
        pbd = Parameter(3)
        groups = two_learning_rate_parameter_groups(
            [
                ("language_model.weight", inherited),
                ("relation_pyramid.level.weight", pyramid),
                ("module.relation_pbd.box_scale", pbd),
            ],
            decay_parameter_names={
                "language_model.weight",
                "relation_pyramid.level.weight",
            },
            inherited_learning_rate=1.0e-5,
            ui_relation_learning_rate=2.0e-5,
            weight_decay=0.01,
        )
        self.assertEqual(
            {group["ui5_lr_group"] for group in groups},
            {"cpt_inherited", "ui_relation"},
        )
        self.assertEqual(
            {group["lr"] for group in groups if group["ui5_lr_group"] == "ui_relation"},
            {2.0e-5},
        )
        self.assertEqual(
            sum(len(group["params"]) for group in groups), 3
        )

        class Optimizer:
            param_groups = groups

        current = optimizer_learning_rates(Optimizer())
        self.assertEqual(current["cpt_inherited"], 1.0e-5)
        self.assertEqual(current["ui_relation"], 2.0e-5)


if __name__ == "__main__":
    unittest.main()
