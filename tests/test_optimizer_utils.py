from __future__ import annotations

import unittest

from eaglevl.train.optimizer_utils import optimizer_parameters


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


if __name__ == "__main__":
    unittest.main()
