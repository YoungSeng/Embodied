import unittest

import torch

from eaglevl.model.locany.relation_modules import TaskRoutedExpertBank


class TaskRoutedExpertBankTest(unittest.TestCase):
    def _bank(self):
        bank = TaskRoutedExpertBank(8, rank=2, num_defect_types=5)
        with torch.no_grad():
            for index, expert in enumerate(bank.experts):
                expert["down"].weight.fill_(0.1 * (index + 1))
                expert["up"].weight.fill_(0.05 * (index + 1))
        return bank

    def test_mixed_batch_hard_routes_one_expert_per_sample(self):
        bank = self._bank()
        values = torch.randn(5, 3, 8)
        tasks = torch.arange(5)
        expected = torch.cat(
            [bank(values[index:index + 1], tasks[index:index + 1]) for index in range(5)]
        )
        torch.testing.assert_close(bank(values, tasks), expected)

    def test_changing_inactive_expert_does_not_change_output(self):
        bank = self._bank()
        values = torch.randn(2, 8)
        tasks = torch.tensor([1, 1])
        before = bank(values, tasks)
        with torch.no_grad():
            bank.experts[4]["up"].weight.add_(1000.0)
        torch.testing.assert_close(bank(values, tasks), before)

    def test_inactive_experts_receive_no_gradient(self):
        bank = self._bank()
        bank(torch.randn(2, 8), torch.tensor([2, 2])).sum().backward()
        for index, expert in enumerate(bank.experts):
            gradients = [parameter.grad for parameter in expert.parameters()]
            if index == 2:
                self.assertTrue(any(value is not None and value.abs().sum() > 0 for value in gradients))
            else:
                self.assertTrue(all(value is None or torch.count_nonzero(value) == 0 for value in gradients))

    def test_unknown_task_fails_instead_of_falling_back(self):
        bank = self._bank()
        with self.assertRaisesRegex(ValueError, "known UI5 defect_type"):
            bank(torch.randn(1, 8), torch.tensor([-1]))


if __name__ == "__main__":
    unittest.main()
