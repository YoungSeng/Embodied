import unittest

from eaglevl.train.cpt_overfitting import analyze_overfitting


class CPTOverfittingTest(unittest.TestCase):
    def test_two_consecutive_val_ce_increases_trigger_risk(self):
        train = [
            {"task": "vqa", "step": 100, "train_main_token_ce": 2.0, "repeat_factor": 1.0},
            {"task": "vqa", "step": 200, "train_main_token_ce": 1.5, "repeat_factor": 1.2},
            {"task": "vqa", "step": 300, "train_main_token_ce": 1.0, "repeat_factor": 1.4},
        ]
        evaluation = [
            {"task": "vqa", "step": 100, "split": "heldout", "eval_token_ce": 2.1, "primary_metric": 0.7},
            {"task": "vqa", "step": 200, "split": "heldout", "eval_token_ce": 2.2, "primary_metric": 0.7},
            {"task": "vqa", "step": 300, "split": "heldout", "eval_token_ce": 2.3, "primary_metric": 0.7},
        ]
        result = analyze_overfitting(train, evaluation)
        self.assertEqual(result["first_overfitting_risk"]["task"], "vqa")
        self.assertEqual(result["first_overfitting_risk"]["step"], 300)

    def test_two_points_are_insufficient_not_proof_of_no_overfit(self):
        result = analyze_overfitting(
            [],
            [
                {"task": "ocr", "step": 100, "split": "heldout", "primary_metric": 0.5},
                {"task": "ocr", "step": 200, "split": "heldout", "primary_metric": 0.6},
            ],
        )
        self.assertEqual(result["per_task"]["ocr"]["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
