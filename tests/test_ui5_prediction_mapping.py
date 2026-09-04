from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scorer = load_module(
    ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py",
    "test_ui5_prediction_mapping_scorer",
)
evaluation = load_module(
    ROOT / "scripts" / "run_ui5_curriculum_evaluation.py",
    "test_ui5_prediction_mapping_evaluation",
)


def load_worker_output_stem_functions() -> dict[str, object]:
    """Load just the worker's dependency-free naming functions from its AST."""

    source_path = ROOT / "scripts" / "inference_ui_defect_locany.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {"legacy_output_stem", "build_output_stems"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    namespace = {
        "Path": Path,
        "Sequence": Sequence,
        "defaultdict": defaultdict,
        "hashlib": hashlib,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class CanonicalPredictionMappingTest(unittest.TestCase):
    def test_collision_hashes_match_worker_and_drive_exact_gt_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_dir = root / "gt"
            pred_dir = root / "pred"
            first_dir = root / "shard-a"
            second_dir = root / "shard-b"
            for directory in (gt_dir, pred_dir, first_dir, second_dir):
                directory.mkdir()
            first = (first_dir / "12.png").resolve()
            second = (second_dir / "12.png").resolve()
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            paths = [str(first), str(second)]

            worker = load_worker_output_stem_functions()
            expected = worker["build_output_stems"](paths)
            self.assertEqual(scorer.build_output_stems(paths), expected)
            self.assertEqual(
                expected,
                {
                    path: "12__"
                    + hashlib.blake2b(path.encode("utf-8"), digest_size=5).hexdigest()
                    for path in paths
                },
            )

            gt_path = gt_dir / "task.jsonl"
            write_jsonl(
                gt_path,
                [
                    {"image": str(first), "answer": {"bboxes": [], "types": []}},
                    {"images": [{"path": str(second)}], "answer": {"bboxes": [], "types": []}},
                ],
            )
            for stem in expected.values():
                (pred_dir / f"{stem}.json").write_text("[]\n", encoding="utf-8")

            merged = root / "merged.jsonl"
            stats = scorer.merge_gt_and_yolo_dir_preds(
                str(gt_path), str(pred_dir), str(merged), "xyxy", "元素重叠"
            )
            self.assertEqual(stats["matched_files"], 2)
            rows = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [row["_ui5_prediction_stem"] for row in rows],
                [expected[str(first)], expected[str(second)]],
            )
            self.assertEqual(
                [Path(row["pred_parse_info"]["prediction_file"]).name for row in rows],
                [f"{expected[str(first)]}.json", f"{expected[str(second)]}.json"],
            )

    def test_prediction_lookup_never_matches_a_longer_numeric_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pred_dir = Path(temporary)
            (pred_dir / "123.json").write_text("[]\n", encoding="utf-8")
            self.assertIsNone(scorer.find_prediction_file(str(pred_dir), "12"))
            (pred_dir / "12_ok.json").write_text("[]\n", encoding="utf-8")
            self.assertEqual(
                Path(scorer.find_prediction_file(str(pred_dir), "12")).name,
                "12_ok.json",
            )
            (pred_dir / "12.json").write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "多个精确状态文件"):
                scorer.find_prediction_file(str(pred_dir), "12")

    def test_prefiltered_sample_is_not_skipped_for_figma_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ui5-no-figma-") as temporary:
            root = Path(temporary)
            merged = root / "merged.jsonl"
            image_path = root / "ordinary.png"
            write_jsonl(
                merged,
                [
                    {
                        "image": str(image_path),
                        "answer": {"bboxes": [], "types": []},
                        "pred_ans": {"has_issue": False, "issues": []},
                        "_ui5_prediction_stem": "ordinary",
                    }
                ],
            )
            sample = json.loads(merged.read_text(encoding="utf-8"))
            self.assertFalse(scorer.is_figma_sample(sample))
            self.assertTrue(scorer.is_figma_sample({"image": "/data/figma:123.png"}))
            metrics = scorer.evaluate_merged_file(
                str(merged), "元素重叠", iou_thresh=0.1, include_figma=False
            )
            self.assertEqual(metrics["total_samples"], 1)
            self.assertEqual(metrics["scored_sample_ids"], ["ordinary"])
            self.assertEqual(metrics["skipped_sample_ids"], [])

    def test_all_task_json_preserves_scored_ids_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "all_tasks_evaluation.txt"
            task_summaries = {}
            for task in scorer.TASK_CONFIG:
                task_summaries[task] = scorer.build_metrics_summary(
                    {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "tn": 0,
                        "img_tp": 0,
                        "img_fp": 0,
                        "img_fn": 0,
                        "img_tn": 1,
                        "count_match": 1,
                        "total_samples": 1,
                        "invalid_pred": 0,
                        "scored_sample_ids": [f"sample-{task}"],
                        "skipped_sample_ids": [],
                    }
                )
            scorer.write_all_tasks_summary(task_summaries, str(report))
            payload = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
            for task in scorer.TASK_CONFIG:
                summary = payload["tasks"][task]
                self.assertEqual(summary["total_samples"], 1)
                self.assertEqual(summary["scored_sample_count"], 1)
                self.assertEqual(summary["scored_sample_ids"], [f"sample-{task}"])
                self.assertEqual(summary["skipped_sample_count"], 0)


class FormalScoredSampleCoverageTest(unittest.TestCase):
    @staticmethod
    def raw_metrics() -> dict:
        metric = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0}
        image = {**metric, "tn": 0, "accuracy": 1.0}
        bbox = {**metric, "count_accuracy": 1.0}
        return {
            "tasks": {
                task: {
                    "image": dict(image),
                    "bbox": dict(bbox),
                    "total_samples": 1,
                    "scored_sample_count": 1,
                    "scored_sample_ids": [f"sample-{task}"],
                    "skipped_sample_count": 0,
                    "skipped_sample_ids": [],
                }
                for task in evaluation.TASKS
            }
        }

    def test_formal_coverage_reconciles_ids_counts_and_duplicates(self) -> None:
        expected_images = {task: 1 for task in evaluation.TASKS}
        expected_stems = {
            task: {f"sample-{task}"} for task in evaluation.TASKS
        }
        raw = self.raw_metrics()
        evaluation.validate_scored_sample_coverage(
            raw, expected_images=expected_images, expected_stems=expected_stems
        )

        raw["tasks"]["occlusion"].update(
            {
                "total_samples": 2,
                "scored_sample_count": 2,
                "scored_sample_ids": ["sample-occlusion", "sample-occlusion"],
            }
        )
        raw["tasks"]["occlusion"]["image"]["tp"] = 2
        with self.assertRaisesRegex(RuntimeError, "duplicates=1"):
            evaluation.validate_scored_sample_coverage(
                raw, expected_images=expected_images, expected_stems=expected_stems
            )

        raw = self.raw_metrics()
        raw["tasks"]["cropping"]["scored_sample_ids"] = ["sample-123"]
        with self.assertRaisesRegex(RuntimeError, "missing=.*sample-cropping"):
            evaluation.validate_scored_sample_coverage(
                raw, expected_images=expected_images, expected_stems=expected_stems
            )


if __name__ == "__main__":
    unittest.main()
