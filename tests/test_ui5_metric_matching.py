from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_ui5_train_rollout_worker as train_rollout_worker  # noqa: E402
from ui5_metric_matching import (  # noqa: E402
    maximum_qualified_iou_matches,
    threshold_aware_linear_sum_assignment,
)


def _exact_linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small exhaustive SciPy stand-in for this CPU-only regression test."""

    row_count, column_count = cost.shape
    if row_count <= column_count:
        rows = np.arange(row_count, dtype=np.intp)
        columns = min(
            itertools.permutations(range(column_count), row_count),
            key=lambda chosen: sum(cost[row, column] for row, column in enumerate(chosen)),
        )
        return rows, np.asarray(columns, dtype=np.intp)
    columns = np.arange(column_count, dtype=np.intp)
    rows = min(
        itertools.permutations(range(row_count), column_count),
        key=lambda chosen: sum(cost[row, column] for column, row in enumerate(chosen)),
    )
    return np.asarray(rows, dtype=np.intp), columns


@contextmanager
def _scipy_assignment_stub():
    scipy = types.ModuleType("scipy")
    optimize = types.ModuleType("scipy.optimize")
    optimize.linear_sum_assignment = _exact_linear_sum_assignment
    scipy.optimize = optimize
    with mock.patch.dict(
        sys.modules,
        {"scipy": scipy, "scipy.optimize": optimize},
    ):
        yield


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_inference_module():
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    torch.float16 = object()
    torch.float32 = object()
    torch.Tensor = object
    torch.manual_seed = lambda seed: None
    torch.is_tensor = lambda value: False
    torch.equal = lambda left, right: False
    torch.inference_mode = lambda: (lambda function: function)
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: None,
        device_count=lambda: 0,
        empty_cache=lambda: None,
    )

    transformers = types.ModuleType("transformers")
    for name in ("AutoConfig", "AutoModel", "AutoProcessor", "AutoTokenizer"):
        setattr(transformers, name, type(name, (), {}))

    relation = types.ModuleType("eaglevl.model.locany.relation_modules")
    relation.UI_RELATION_PROMPT_SPECS = tuple(
        SimpleNamespace(task_name=task, prompt_label=task.replace("_", " "))
        for task in (
            "occlusion",
            "cropping",
            "text_overflow",
            "text_ellipsis",
            "content_missing",
        )
    )
    common = types.ModuleType("locany_ui5_common")
    common.aggregate_tiled_gate_diagnostics = lambda *args, **kwargs: {}
    tiling = types.ModuleType("ui5_lossless_tiling")
    tiling.assert_lossless_coverage = lambda *args, **kwargs: None
    tiling.generate_lossless_tiles = lambda *args, **kwargs: []
    tiling.merge_tile_predictions = lambda *args, **kwargs: []

    module_name = "ui5_metric_matching_inference_test"
    with mock.patch.dict(
        sys.modules,
        {
            "torch": torch,
            "transformers": transformers,
            "eaglevl.model.locany.relation_modules": relation,
            "locany_ui5_common": common,
            "ui5_lossless_tiling": tiling,
        },
    ):
        return _load_module(SCRIPTS / "inference_ui_defect_locany.py", module_name)


class _MatrixScorer:
    np = np

    _matrix = np.asarray([[0.90, 0.11], [0.10, 0.0]], dtype=np.float64)

    @classmethod
    def calculate_iou(cls, gt_box, pred_box) -> float:
        return float(cls._matrix[int(gt_box[0]), int(pred_box[0])])


class UI5MetricMatchingTest(unittest.TestCase):
    def test_maximizes_qualified_cardinality_before_raw_iou(self) -> None:
        matrix = np.asarray([[0.90, 0.11], [0.10, 0.0]], dtype=np.float64)
        with _scipy_assignment_stub():
            row_indices, column_indices = threshold_aware_linear_sum_assignment(
                matrix, 0.1
            )
            qualified = maximum_qualified_iou_matches(matrix, 0.1)

        self.assertEqual(
            list(zip(row_indices.tolist(), column_indices.tolist())),
            [(0, 1), (1, 0)],
        )
        self.assertEqual(qualified, [(0, 1), (1, 0)])

    def test_numpy_fallback_matches_exhaustive_rectangular_optimum(self) -> None:
        rng = np.random.default_rng(20260905)
        without_scipy = {"scipy": None, "scipy.optimize": None}
        for row_count, column_count in ((1, 3), (3, 1), (2, 3), (4, 3)):
            for _ in range(10):
                matrix = rng.random((row_count, column_count))
                threshold = 0.4
                qualified = matrix >= threshold
                bonus = float(min(matrix.shape) + 1)
                objective = np.where(qualified, bonus + matrix, 0.0)
                expected_rows, expected_columns = _exact_linear_sum_assignment(
                    -objective
                )
                with mock.patch.dict(sys.modules, without_scipy):
                    rows, columns = threshold_aware_linear_sum_assignment(
                        matrix, threshold
                    )
                self.assertAlmostEqual(
                    float(objective[rows, columns].sum()),
                    float(objective[expected_rows, expected_columns].sum()),
                    places=12,
                )

        counterexample = np.asarray([[0.90, 0.11], [0.10, 0.0]])
        with mock.patch.dict(sys.modules, without_scipy):
            self.assertEqual(
                maximum_qualified_iou_matches(counterexample, 0.1),
                [(0, 1), (1, 0)],
            )

    def test_iou_sum_is_the_secondary_objective(self) -> None:
        matrix = np.asarray([[0.90, 0.80], [0.70, 0.20]], dtype=np.float64)
        with _scipy_assignment_stub():
            matches = maximum_qualified_iou_matches(matrix, 0.1)
        self.assertEqual(matches, [(0, 1), (1, 0)])

    def test_train_and_hard_rollout_scorers_count_two_true_positives(self) -> None:
        gt_boxes = [[0, 0, 1, 1], [1, 0, 2, 1]]
        pred_boxes = [[0, 0, 1, 1], [1, 0, 2, 1]]
        inference = _load_inference_module()

        with _scipy_assignment_stub():
            train_score = train_rollout_worker.score_prediction(
                _MatrixScorer,
                gt_boxes,
                pred_boxes,
                "defect",
                0.1,
                (10, 10),
            )
            hard_score = inference._score_hard_prediction(
                _MatrixScorer,
                gt_boxes,
                pred_boxes,
                "defect",
                0.1,
                (10, 10),
            )

        for score in (train_score, hard_score):
            self.assertEqual(
                (score["TP_box"], score["FP_box"], score["FN_box"]),
                (2, 0, 0),
            )
            self.assertTrue(score["exact_correct"])
            self.assertEqual(
                {(pair["gt_index"], pair["pred_index"]) for pair in score["matched_pairs"]},
                {(0, 1), (1, 0)},
            )

    def test_hard_rollout_pairing_rejects_seed_or_reward_mismatch(self) -> None:
        inference = _load_inference_module()
        seeds = [101, 102, 103, 104]
        row = {
            "record_id": "hard-1",
            "rollouts": {
                "crop": [
                    {
                        "model_id": "crop",
                        "rollout_id": rollout_id,
                        "seed": seed,
                        "exact_correct": False,
                    }
                    for rollout_id, seed in enumerate(seeds)
                ]
            },
        }
        paired = inference._paired_baseline_crop_rollouts(row, seeds)
        self.assertEqual([item["seed"] for item in paired], seeds)

        row["rollouts"]["crop"][2]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "seed is not paired"):
            inference._paired_baseline_crop_rollouts(row, seeds)
        row["rollouts"]["crop"][2]["seed"] = seeds[2]
        row["rollouts"]["crop"][2]["exact_correct"] = True
        with self.assertRaisesRegex(ValueError, "must be false"):
            inference._paired_baseline_crop_rollouts(row, seeds)

    def test_canonical_scorer_counts_two_true_positives(self) -> None:
        scorer = _load_module(
            ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py",
            "ui5_metric_matching_canonical_test",
        )
        gt_boxes = [[0, 0, 1, 1], [1, 0, 2, 1]]
        pred_boxes = [[0, 0, 1, 1], [1, 0, 2, 1]]
        with tempfile.TemporaryDirectory() as temporary:
            merged = Path(temporary) / "merged.jsonl"
            merged.write_text(
                json.dumps({"answer": {}, "pred_ans": {}}) + "\n",
                encoding="utf-8",
            )
            with _scipy_assignment_stub(), mock.patch.object(
                scorer,
                "extract_bboxes_for_issue",
                side_effect=[gt_boxes, pred_boxes],
            ), mock.patch.object(
                scorer,
                "calculate_iou",
                side_effect=_MatrixScorer.calculate_iou,
            ):
                metrics = scorer.evaluate_merged_file(
                    str(merged), "target", 0.1, include_figma=True
                )

        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 0, 0))


if __name__ == "__main__":
    unittest.main()
