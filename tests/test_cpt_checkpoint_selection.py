import unittest

from eaglevl.train.cpt_checkpoint_selection import select_checkpoint


class CPTCheckpointSelectionTest(unittest.TestCase):
    def test_incomplete_or_train_pool_current_never_becomes_best(self):
        tasks = {
            task: {"primary_metric": 0.99}
            for task in ("referring_kg", "ui_defect", "vqa")
        }
        for split, complete in (("heldout", False), ("train_pool", True)):
            result = select_checkpoint(
                {
                    "checkpoint": "candidate",
                    "split": split,
                    "complete_ten_task_heldout": complete,
                    "primary_metric": 0.99,
                    "eval_token_ce": 0.1,
                },
                tasks,
                [],
            )
            self.assertFalse(result["is_best_overall"])
            self.assertFalse(result["current_is_complete_heldout"])

    def test_macro_primary_then_ce_tiebreak(self):
        history = [
            {
                "split": "heldout",
                "task": "__task_macro__",
                "checkpoint": "checkpoint-100",
                "primary_metric": 0.8,
                "eval_token_ce": 1.4,
                "complete_ten_task_heldout": True,
            }
        ]
        result = select_checkpoint(
            {
                "split": "heldout",
                "complete_ten_task_heldout": True,
                "primary_metric": 0.8,
                "eval_token_ce": 1.2,
            },
            {},
            history,
        )
        self.assertTrue(result["is_best_overall"])

    def test_critical_task_drop_blocks_best(self):
        history = [
            {
                "split": "heldout",
                "task": "vqa",
                "checkpoint": "checkpoint-100",
                "primary_metric": 0.9,
            },
            {
                "split": "heldout",
                "task": "__task_macro__",
                "checkpoint": "checkpoint-100",
                "primary_metric": 0.7,
                "eval_token_ce": 1.4,
                "complete_ten_task_heldout": True,
            },
        ]
        result = select_checkpoint(
            {
                "split": "heldout",
                "complete_ten_task_heldout": True,
                "primary_metric": 0.8,
                "eval_token_ce": 1.2,
            },
            {"vqa": {"primary_metric": 0.85}},
            history,
        )
        self.assertFalse(result["is_best_overall"])
        self.assertAlmostEqual(result["critical_regressions"]["vqa"], 0.05)

    def test_partial_or_different_manifest_history_cannot_block_best(self):
        history = [
            {
                "split": "heldout",
                "task": "__task_macro__",
                "checkpoint": "partial",
                "manifest_id": "current",
                "primary_metric": 0.99,
                "complete_ten_task_heldout": False,
            },
            {
                "split": "heldout",
                "task": "__task_macro__",
                "checkpoint": "other-split",
                "manifest_id": "other",
                "primary_metric": 0.99,
                "complete_ten_task_heldout": True,
            },
            {
                "split": "heldout",
                "task": "__task_macro__",
                "checkpoint": "other-protocol",
                "manifest_id": "current",
                "evaluation_protocol_id": "other-protocol",
                "primary_metric": 0.99,
                "complete_ten_task_heldout": True,
            },
        ]
        result = select_checkpoint(
            {
                "split": "heldout",
                "complete_ten_task_heldout": True,
                "manifest_id": "current",
                "evaluation_protocol_id": "current-protocol",
                "primary_metric": 0.7,
                "eval_token_ce": 1.2,
            },
            {},
            history,
        )
        self.assertTrue(result["is_best_overall"])


if __name__ == "__main__":
    unittest.main()
