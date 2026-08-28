from __future__ import annotations

import unittest
import json
import sys
import tempfile
from pathlib import Path

from eaglevl.train.cpt_eval_metrics import (
    aggregate_scores,
    canonical_defect_label,
    defect_metrics,
    micro_primary,
    one_to_one_boxes,
    parse_points,
    score_task,
    task_macro_primary,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from recompute_locany_cpt_metrics import recompute


def pair(label: str, box: tuple[int, int, int, int]) -> str:
    return f"<ref>{label}</ref><box><{box[0]}><{box[1]}><{box[2]}><{box[3]}></box>"


class CPTEvalMetricsTest(unittest.TestCase):
    def test_ui_defect_legacy_labels_map_to_fixed_five_classes(self):
        self.assertEqual(canonical_defect_label("文字溢出容器"), "text_overflow")
        self.assertEqual(canonical_defect_label("文字省略异常"), "text_ellipsis")
        self.assertEqual(canonical_defect_label("UI element overlap"), "occlusion")
        self.assertEqual(canonical_defect_label("元素被裁切"), "cropping")
        self.assertEqual(canonical_defect_label("内容未展示"), "content_missing")

    def test_vqa_correct_and_incorrect_are_not_char_f1_accuracy(self):
        score = score_task(
            "vqa",
            "断言结果是否正确: 正确",
            "断言结果是否正确: 错误",
        )
        self.assertEqual(score["parsed_vqa_prediction"], "correct")
        self.assertEqual(score["parsed_vqa_target"], "incorrect")
        self.assertEqual(score["vqa_accuracy"], 0.0)
        self.assertEqual(score["primary_name"], "vqa_accuracy")

    def test_point_parser_supports_qwen_and_agent_action_formats(self):
        self.assertEqual(parse_points("<|point_start|>(123,456)<|point_end|>"), [[123.0, 456.0]])
        self.assertEqual(parse_points("tap(position=(321,654))"), [[321.0, 654.0]])
        hit = score_task("agent_grounding", "tap(position=(500,500))", pair("x", (450, 450, 550, 550)))
        self.assertEqual(hit["point_hit50"], 1.0)
        self.assertEqual(hit["gold_point_count"], 1)
        invalid = score_task(
            "agent_grounding",
            "tap(position=(1200,500))",
            pair("x", (450, 450, 550, 550)),
        )
        self.assertFalse(invalid["point_format_valid"])
        self.assertEqual(invalid["point_hit50"], 0.0)

    def test_one_prediction_cannot_match_two_ground_truth_boxes(self):
        target = "\n".join((pair("a", (0, 0, 100, 100)), pair("b", (0, 0, 100, 100))))
        prediction = pair("x", (0, 0, 100, 100))
        metrics = one_to_one_boxes(prediction, target, label_aware=False)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["recall"], 0.5)

    def test_single_grounding_accepts_unlabeled_box_grammar(self):
        score = score_task(
            "single_grounding",
            "<box><10><20><30><40></box>",
            "<box><10><20><30><40></box>",
        )
        self.assertEqual(score["box_recall"], 1.0)

    def test_ui_defect_requires_matching_class_even_for_identical_box(self):
        metrics = defect_metrics(
            pair("元素重叠", (100, 100, 300, 300)),
            pair("元素被裁切", (100, 100, 300, 300)),
        )
        self.assertEqual(metrics["defect_macro_f1"], 0.0)
        self.assertEqual(metrics["defect_per_class"]["cropping"]["tp"], 0)
        self.assertEqual(metrics["defect_confusion"]["cropping->occlusion"], 1)

    def test_ui_defect_aggregates_five_class_image_and_bbox_granularity(self):
        scores = [
            score_task(
                "ui_defect",
                pair("元素裁切", (100, 100, 300, 300)),
                pair("元素被裁切", (100, 100, 300, 300)),
            ),
            score_task(
                "ui_defect",
                "<box>none</box>",
                pair("文字溢出", (400, 400, 600, 600)),
            ),
        ]
        metrics = aggregate_scores("ui_defect", scores)

        self.assertEqual(
            list(metrics["per_class"])[:5],
            [
                "text_overflow",
                "text_ellipsis",
                "occlusion",
                "cropping",
                "content_missing",
            ],
        )
        self.assertEqual(metrics["per_class"]["cropping"]["image"]["tp"], 1)
        self.assertEqual(metrics["per_class"]["cropping"]["bbox"]["tp"], 1)
        self.assertEqual(metrics["per_class"]["text_overflow"]["image"]["fn"], 1)
        self.assertEqual(metrics["per_class"]["text_overflow"]["bbox"]["fn"], 1)
        self.assertIsNone(metrics["per_class"]["text_ellipsis"]["image"]["f1"])
        self.assertAlmostEqual(metrics["defect_image_macro_f1"], 0.5)
        self.assertAlmostEqual(metrics["defect_bbox_macro_f1"], 0.5)
        self.assertAlmostEqual(metrics["primary_metric"], 0.5)

    def test_all_bbox_tasks_use_the_configured_iou_point_one_threshold(self):
        target = pair("元素裁切", (0, 0, 100, 100))
        prediction = pair("元素裁切", (50, 0, 150, 100))  # IoU = 1/3

        at_point_one = score_task(
            "ui_defect",
            prediction,
            target,
            iou_threshold=0.1,
        )
        at_point_five = score_task(
            "ui_defect",
            prediction,
            target,
            iou_threshold=0.5,
        )

        self.assertEqual(at_point_one["defect_macro_f1"], 1.0)
        self.assertEqual(at_point_five["defect_macro_f1"], 0.0)
        self.assertEqual(at_point_one["iou_threshold"], 0.1)

    def test_ocr_reports_location_and_label_aware_results(self):
        target = pair("设置", (10, 10, 100, 100))
        score = score_task("ocr", pair("设置信", (10, 10, 100, 100)), target)
        self.assertEqual(score["location_metrics"]["f1"], 1.0)
        self.assertEqual(score["ocr_f1"], 0.0)
        self.assertGreater(score["matched_label_char_f1"], 0.0)

    def test_all_elements_reports_type_accuracy_separately(self):
        target = pair("设置 | type=button", (10, 10, 100, 100))
        prediction = pair("设置 | type=text", (10, 10, 100, 100))
        score = score_task("all_ui_elements", prediction, target)
        self.assertEqual(score["box_f1"], 1.0)
        self.assertEqual(score["label_accuracy"], 1.0)
        self.assertEqual(score["type_accuracy"], 0.0)

    def test_task_macro_weights_tasks_not_sample_counts(self):
        vqa = aggregate_scores("vqa", [score_task("vqa", "正确", "正确")] * 10)
        grounding = aggregate_scores(
            "single_grounding",
            [score_task("single_grounding", "invalid", pair("x", (0, 0, 100, 100)))],
        )
        self.assertEqual(task_macro_primary({"vqa": vqa, "single_grounding": grounding}), 0.5)

    def test_micro_uses_box_units_and_inference_errors_are_zero_weighted_scores(self):
        vqa_scores = [score_task("vqa", "正确", "正确")]
        failed = score_task("vqa", "", "正确")
        failed["evaluation_error"] = 1.0
        failed["primary_metric"] = 0.0
        vqa = aggregate_scores("vqa", [*vqa_scores, failed])
        self.assertEqual(vqa["inference_error_count"], 1)
        self.assertEqual(vqa["primary_metric"], 0.5)

        grounding = aggregate_scores(
            "single_grounding",
            [
                score_task(
                    "single_grounding",
                    pair("x", (0, 0, 100, 100)),
                    "\n".join(
                        (pair("a", (0, 0, 100, 100)), pair("b", (200, 200, 300, 300)))
                    ),
                )
            ],
        )
        self.assertEqual(grounding["primary_weight"], 2)
        self.assertAlmostEqual(
            micro_primary({"vqa": vqa, "single_grounding": grounding}),
            (0.5 * 2 + 0.5 * 2) / 4,
        )

    def test_predictions_can_be_rescored_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            rows = [
                {
                    "model": "checkpoint",
                    "split": "heldout",
                    "task": "vqa",
                    "target": "正确",
                    "prediction": "错误",
                    "metrics": {"vqa_accuracy": 1.0},
                    "error": None,
                },
                {
                    "model": "checkpoint",
                    "split": "heldout",
                    "task": "ocr",
                    "target": pair("设置", (10, 10, 100, 100)),
                    "prediction": "",
                    "metrics": {},
                    "error": "RuntimeError: intentional",
                },
            ]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            result = recompute(path)
            self.assertEqual(
                result["models"]["checkpoint"]["per_task"]["vqa"]["vqa_accuracy"],
                0.0,
            )
            self.assertEqual(
                result["models"]["checkpoint"]["per_task"]["ocr"]["primary_metric"],
                0.0,
            )
            self.assertEqual(result["models"]["checkpoint"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
