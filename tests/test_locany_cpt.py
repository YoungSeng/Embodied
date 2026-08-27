from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eaglevl.train.dataset_sampling import resolve_dataset_sampling_weight, resolve_recipe_entry_paths
from eaglevl.train.cpt_sampling import (
    assert_sampling_resume_compatible,
    resolve_cpt_sampling,
)
from eaglevl.train.checkpoint_schedule import PeriodicCheckpointSchedule
from prepare_locany_cpt import extract_image_size, extract_input_size, normalize_box, normalize_record
from simulate_locany_cpt_sampling import simulate


def _record(image: Path, prompt: str, answer: str, *, objects=None, record_id="sample"):
    result = {
        "id": record_id,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "images": [str(image)],
    }
    if objects is not None:
        result["objects"] = objects
    return result


def _write_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


class LocateAnythingCPTTest(unittest.TestCase):
    def test_sampling_simulation_accounts_for_post_skip_exposure(self):
        recipe = {
            "locany_cpt_small": {
                "cpt_task": "small",
                "dataset_rows": 100,
                "mean_total_supervised_tokens": 10,
            },
            "locany_cpt_large": {
                "cpt_task": "large",
                "dataset_rows": 400,
                "mean_total_supervised_tokens": 20,
            },
        }
        stats = {
            "tasks": {
                "small": {"rows": 100, "groups": 100, "oversize_rate": 0.1},
                "large": {"rows": 400, "groups": 400, "oversize_rate": 0.0},
            }
        }
        result = simulate(
            recipe,
            stats,
            exposure=1000,
            min_probability=0.0,
            max_probability=1.0,
        )
        sample_equal = {
            row["name"]: row
            for row in result["modes"]["sample_equal"]["tasks"]
        }
        self.assertEqual(sample_equal["small"]["expected_attempted_samples"], 500)
        self.assertEqual(sample_equal["small"]["expected_trained_samples"], 450)
        self.assertLess(
            sample_equal["small"]["post_skip_sample_share"],
            sample_equal["large"]["post_skip_sample_share"],
        )

    def test_sample_equal_remains_the_default_independent_of_dataset_size(self):
        config = resolve_cpt_sampling(
            [
                {"name": "small", "rows": 5667},
                {"name": "large", "rows": 608896},
            ]
        )
        self.assertEqual(config["mode"], "sample_equal")
        self.assertEqual([task["probability"] for task in config["tasks"]], [0.5, 0.5])

    def test_hybrid_increases_large_coverage_but_accounts_for_tokens(self):
        config = resolve_cpt_sampling(
            [
                {"name": "small", "rows": 100, "mean_total_supervised_tokens": 100},
                {"name": "large", "rows": 10000, "mean_total_supervised_tokens": 400},
            ],
            mode="hybrid",
        )
        probabilities = {task["name"]: task["probability"] for task in config["tasks"]}
        self.assertGreater(probabilities["large"], probabilities["small"])
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_sampling_probability_clamps_and_resume_hash(self):
        config = resolve_cpt_sampling(
            [
                {"name": "small", "rows": 1},
                {"name": "large", "rows": 1000000},
            ],
            mode="sqrt_size",
            min_task_prob=0.1,
            max_task_prob=0.9,
        )
        probabilities = [task["probability"] for task in config["tasks"]]
        self.assertAlmostEqual(probabilities[0], 0.1)
        self.assertAlmostEqual(probabilities[1], 0.9)
        assert_sampling_resume_compatible(config, dict(config))
        changed = dict(config)
        changed["config_hash"] = "different"
        with self.assertRaisesRegex(RuntimeError, "changed across resume"):
            assert_sampling_resume_compatible(config, changed)

    def test_explicit_sampling_weight_decouples_probability_from_size(self):
        self.assertEqual(resolve_dataset_sampling_weight({"sampling_weight": 1.0}, 5667), 1.0)
        self.assertEqual(resolve_dataset_sampling_weight({"sampling_weight": 1.0}, 608896), 1.0)

    def test_legacy_sampling_weight_is_preserved(self):
        self.assertEqual(resolve_dataset_sampling_weight({"repeat_time": 2.0}, 100), 200.0)
        self.assertEqual(resolve_dataset_sampling_weight({"repeat_time": 0.5}, 100), 100.0)

    def test_recipe_relative_paths_are_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            meta_path = Path(temporary) / "bundle" / "recipe" / "train.json"
            resolved = resolve_recipe_entry_paths(
                {
                    "annotation": ["../annotations/task.jsonl"],
                    "root": "..",
                    "paths_relative_to_meta": True,
                },
                meta_path,
            )
            self.assertEqual(
                resolved["annotation"],
                [str((meta_path.parent / "../annotations/task.jsonl").resolve())],
            )
            self.assertEqual(resolved["root"], str((meta_path.parent / "..").resolve()))

    def test_bbox_normalization(self):
        self.assertEqual(normalize_box((0.1, 0.2, 0.3, 0.4), None, None), (100, 200, 300, 400))
        self.assertEqual(
            normalize_box((10, 20, 30, 40), "real", (100, 200)),
            (100, 100, 300, 200),
        )

    def test_png_size_fallback_without_pillow_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "screen.png"
            # A PNG signature and IHDR prefix are enough for the dependency-free
            # size reader; the test intentionally has no image payload.
            image.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR"
                + (400).to_bytes(4, "big")
                + (1200).to_bytes(4, "big")
            )
            self.assertEqual(extract_image_size({}, image), (400.0, 1200.0))

    def test_ocr_pixel_boxes_use_dimensions_read_from_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "screen.png"
            image.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR"
                + (400).to_bytes(4, "big")
                + (1200).to_bytes(4, "big")
            )
            record = _record(
                image,
                "<image>请识别屏幕上的全部文字，以及文字的位置。",
                json.dumps(
                    [{"text": "测试", "bbox_2d": [40, 600, 80, 720]}],
                    ensure_ascii=False,
                ),
            )
            normalized, is_grounding = normalize_record(
                record,
                task="ocr",
                source_file=root / "ocr.jsonl",
                source_root=root,
                check_images=True,
            )
            self.assertTrue(is_grounding)
            self.assertEqual(
                normalized["conversations"][-1]["value"],
                "<ref>测试</ref><box><100><500><200><600></box>",
            )

    def test_ocr_qwen_boxes_use_nested_input_size_not_original_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "screen.jpg"
            image.write_bytes(b"placeholder")
            record = _record(
                image,
                "<image>请识别屏幕上的全部文字，以及文字的位置。",
                "text\t15:00\t<|box_start|>(44,24),(113,48)<|box_end|>\n"
                "text\t弹\t<|box_start|>(31,1073),(69,1109)<|box_end|>",
            )
            record["infos"] = {
                "image_size": [[1080, 2340]],
                "input_size": [[672, 1456]],
            }
            self.assertEqual(extract_image_size(record, image), (1080.0, 2340.0))
            self.assertEqual(extract_input_size(record), (672.0, 1456.0))

            normalized, is_grounding = normalize_record(
                record,
                task="ocr",
                source_file=root / "ocr.jsonl",
                source_root=root,
                check_images=True,
            )
            self.assertTrue(is_grounding)
            self.assertEqual(
                normalized["conversations"][-1]["value"],
                "<ref>15:00</ref><box><65><16><168><33></box>\n"
                "<ref>弹</ref><box><46><737><103><762></box>",
            )

    def test_vqa_is_preserved_and_ui_defect_is_canonical_norm1000(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "screen.jpg"
            image.write_bytes(b"placeholder")
            infos = {
                "image_size": [[828, 1792]],
                "input_size": [[672, 1456]],
            }

            vqa = _record(
                image,
                "<image>根据给定的指令，判断基于图片的断言是否正确：断言：屏幕中存在文字“置顶聊天”。",
                "断言结果是否正确: 正确",
            )
            vqa["infos"] = infos
            normalized_vqa, vqa_is_grounding = normalize_record(
                vqa,
                task="vqa",
                source_file=root / "vqa.jsonl",
                source_root=root,
                check_images=True,
            )
            self.assertFalse(vqa_is_grounding)
            self.assertEqual(
                normalized_vqa["conversations"][-1]["value"],
                "断言结果是否正确: 正确",
            )

            defect = _record(
                image,
                "<image>请发现所有的交互体验问题。",
                "元素被裁切\t<|box_start|>(216,122),(579,154)<|box_end|>\n"
                "\t元素重叠\t<|box_start|>(659,259),(707,275)<|box_end|>",
            )
            defect["infos"] = infos
            normalized_defect, defect_is_grounding = normalize_record(
                defect,
                task="ui_defect",
                source_file=root / "grounding.jsonl",
                source_root=root,
                check_images=True,
            )
            self.assertTrue(defect_is_grounding)
            self.assertEqual(
                normalized_defect["conversations"][-1]["value"],
                "<ref>元素被裁切</ref><box><216><122><579><154></box>\n"
                "<ref>元素重叠</ref><box><659><259><707><275></box>",
            )

    def test_periodic_checkpoint_schedule_is_not_five_minutes(self):
        schedule = PeriodicCheckpointSchedule(interval_hours=12)
        schedule.start(100.0)
        self.assertFalse(schedule.is_due(100.0 + 11.9 * 3600))
        self.assertTrue(schedule.is_due(100.0 + 12.0 * 3600))
        schedule.mark_saved(100.0 + 12.0 * 3600)
        self.assertFalse(schedule.is_due(100.0 + 12.0 * 3600 + 5 * 60))
        self.assertTrue(schedule.is_due(100.0 + 24.0 * 3600))

    def test_ms_swift_messages_and_real_objects_are_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "screen.png"
            image.write_bytes(b"placeholder")
            record = {
                "messages": [
                    {"role": "user", "content": "<image>找出按钮。"},
                    {"role": "assistant", "content": "旧答案"},
                ],
                "images": [{"path": str(image)}],
                "infos": {"image_size": [100, 200]},
                "objects": {
                    "ref": ["按钮"],
                    "bbox": [[10, 20, 30, 40]],
                    "bbox_type": "real",
                },
            }
            normalized, is_grounding = normalize_record(
                record,
                task="single_grounding",
                source_file=root / "data.jsonl",
                source_root=root,
                check_images=True,
            )
            self.assertTrue(is_grounding)
            self.assertEqual(
                normalized["conversations"][-1]["value"],
                "<ref>按钮</ref><box><100><100><300><200></box>",
            )

    def test_end_to_end_prepare_and_validate(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            source = tmp_path / "raw"
            output = tmp_path / "out"
            image = source / "screen.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"test-image-placeholder")
            objects = {
                "ref": [{"label": "确定", "type": "button"}],
                "bbox": [[100, 200, 300, 400]],
                "bbox_type": "norm1000",
            }

            rows = {
                "caption/captions/category_8_dy1_washed.jsonl": _record(
                    image, "<image>请描述页面。", "这是一个设置页面。"
                ),
                "grounding/agent/category_1_dy1_dsample20k_n.jsonl": _record(
                    image, "<image>输入指令：点击确定。", "tap(position=(200,300))"
                ),
                "grounding/agent/category_5_dy1_n.jsonl": _record(
                    image, "<image>确定点击位置。", "旧格式", objects=objects
                ),
                "grounding/multi/category_6_dy1_n.jsonl": _record(
                    image, "<image>请发现所有的交互体验问题。", "元素被裁切", objects=objects
                ),
                "grounding/multi/category_9_mul_train_n.jsonl": _record(
                    image, "<image>标注所有 UI 元素。", "[]", objects=objects
                ),
                "grounding/single/category_9_single_p_train_608k_n.jsonl": _record(
                    image, "<image>找出“确定”的位置。", "{}", objects=objects
                ),
                "ocr/category_7_dy1.jsonl": _record(
                    image, "<image>识别全部文字和位置。", "[]", objects=objects
                ),
                "referring/category_2_dy1_397k_n.jsonl": _record(
                    image,
                    "<image>描述区域功能。<|box_start|>(100,200),(300,400)<|box_end|>",
                    "用于确认当前设置。",
                ),
                "referring/category_3_dy1_297k_n.jsonl": _record(
                    image,
                    "<image>描述区域功能。<|box_start|>(100,200),(300,400)<|box_end|>",
                    "用于确认当前设置。",
                ),
                "vqa/category_4_dy1.jsonl": _record(
                    image,
                    "<image>断言：页面存在确定按钮。",
                    "<answer>断言结果是否正确: 正确</answer>",
                ),
            }
            for relative, row in rows.items():
                _write_jsonl(source / relative, row)

            # These two malformed annotations are explicitly safe data drops:
            # a box separated from its canonical ref/box pair and a zero-height
            # box.  Their high fixture rate must not trip --max-error-rate.
            multi_path = source / "grounding/multi/category_9_mul_train_n.jsonl"
            with multi_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        _record(
                            image,
                            "<image>标注所有 UI 元素。",
                            "换行标签\n<box><52><100><174><200></box>",
                            record_id="known-drop-noncanonical",
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.write(
                    json.dumps(
                        _record(
                            image,
                            "<image>标注所有 UI 元素。",
                            "旧格式",
                            objects={
                                "ref": ["bad box"],
                                "bbox": [[52, 875, 174, 875]],
                                "bbox_type": "norm1000",
                            },
                            record_id="known-drop-degenerate",
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            prepare = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "prepare_locany_cpt.py"),
                    "--source-root",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--recipe-name",
                    "locany_cpt_smoke.json",
                    "--copy-images",
                    "--no-split",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(prepare.returncode, 0, prepare.stdout + prepare.stderr)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_known_dropped"], 2)
            self.assertEqual(manifest["total_rejected"], 0)
            self.assertEqual(
                manifest["tasks"]["all_ui_elements"]["known_dropped_records"], 2
            )
            rejected_rows = [
                json.loads(line)
                for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {row["category"] for row in rejected_rows},
                {"noncanonical_ref_box_pair", "invalid_or_degenerate_bbox"},
            )
            self.assertEqual(
                {row["disposition"] for row in rejected_rows},
                {"known_data_drop"},
            )

            recipe_path = output / "recipe" / "locany_cpt_smoke.json"
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            self.assertEqual(len(recipe), 10)
            self.assertEqual({meta["sampling_weight"] for meta in recipe.values()}, {1.0})

            agent_path = output / "annotations" / "agent_grounding.jsonl"
            agent = json.loads(agent_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(
                agent["conversations"][-1]["value"],
                "<ref>确定 | type=button</ref><box><100><200><300><400></box>",
            )
            self.assertFalse(Path(agent["image"]).is_absolute())

            referring_path = output / "annotations" / "referring.jsonl"
            referring = json.loads(referring_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("<box><100><200><300><400></box>", referring["conversations"][0]["value"])
            self.assertNotIn("<|box_start|>", referring["conversations"][0]["value"])

            validate = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "validate_locany_cpt.py"),
                    "--recipe",
                    str(recipe_path),
                    "--records-per-dataset",
                    "0",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)


if __name__ == "__main__":
    unittest.main()
