from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from split_locany_cpt import split_recipe
from validate_locany_cpt import NormalizeError, validate_split_manifest


def record(record_id: str, image: Path, task: str, answer: str = "ok") -> dict:
    return {
        "id": record_id,
        "cpt_task": task,
        "image": str(image),
        "conversations": [
            {"from": "human", "value": "<image>prompt"},
            {"from": "gpt", "value": answer},
        ],
    }


class CPTSplitTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        annotations = root / "annotations"
        images = root / "images"
        annotations.mkdir(parents=True)
        images.mkdir(parents=True)
        task_rows = {"ui_caption": [], "vqa": []}
        for index in range(40):
            image = images / f"screen-{index}.png"
            image.write_bytes(f"image-{index}".encode())
            task_rows["ui_caption"].append(
                record(f"caption-{index}", image, "ui_caption")
            )
            task_rows["vqa"].append(
                record(
                    f"vqa-{index}",
                    image,
                    "vqa",
                    "断言结果是否正确: 正确" if index % 2 else "断言结果是否正确: 错误",
                )
            )
        # A path alias with identical bytes must still join the original image
        # content group rather than becoming an independent row split.
        alias = images / "screen-alias.png"
        alias.write_bytes((images / "screen-0.png").read_bytes())
        task_rows["ui_caption"].append(record("caption-alias", alias, "ui_caption"))

        for task, rows in task_rows.items():
            with (annotations / f"{task}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        recipe = {
            f"locany_cpt_{task}": {
                "annotation": [f"annotations/{task}.jsonl"],
                "root": "",
                "paths_relative_to_meta": True,
                "sampling_weight": 1.0,
                "cpt_task": task,
            }
            for task in task_rows
        }
        recipe_path = root / "recipe.json"
        recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
        return recipe_path

    def test_same_image_across_tasks_never_crosses_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = self._fixture(root)
            output = root / "split"
            split_recipe(recipe, output, seed=20260826, val_fraction=0.2, val_fast_per_task=3)

            rows = [
                json.loads(line)
                for line in (output / "diagnostics" / "split_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            by_group = {}
            for row in rows:
                self.assertIsInstance(row["record_id_hash"], int)
                self.assertIsInstance(row["group_id_hash"], int)
                by_group.setdefault(row["group_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))

            train = {row["group_id"] for row in rows if row["split"] == "train"}
            val = {row["group_id"] for row in rows if row["split"] == "heldout"}
            self.assertFalse(train & val)
            self.assertTrue(train)
            self.assertTrue(val)

            zero_rows = [
                row for row in rows if row["record_id"] in {"ui_caption:caption-0", "ui_caption:caption-alias"}
            ]
            self.assertEqual(len({row["group_id"] for row in zero_rows}), 1)
            self.assertEqual(len({row["split"] for row in zero_rows}), 1)

    def test_fixed_seed_reproduces_manifest_and_fast_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = self._fixture(root)
            first = root / "first"
            second = root / "second"
            split_recipe(recipe, first, seed=17, val_fraction=0.25, val_fast_per_task=2)
            split_recipe(recipe, second, seed=17, val_fraction=0.25, val_fast_per_task=2)
            self.assertEqual(
                (first / "diagnostics" / "split_manifest.jsonl").read_text(encoding="utf-8"),
                (second / "diagnostics" / "split_manifest.jsonl").read_text(encoding="utf-8"),
            )
            for task in ("ui_caption", "vqa"):
                self.assertEqual(
                    (first / "val_fast" / f"{task}.jsonl").read_text(encoding="utf-8"),
                    (second / "val_fast" / f"{task}.jsonl").read_text(encoding="utf-8"),
                )
            vqa_fast = [
                json.loads(line)
                for line in (first / "val_fast" / "vqa.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            labels = {row["conversations"][-1]["value"] for row in vqa_fast}
            self.assertEqual(
                labels,
                {"断言结果是否正确: 正确", "断言结果是否正确: 错误"},
            )

    def test_multi_image_record_connects_every_shared_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = self._fixture(root)
            images = root / "images"
            bridge = {
                "id": "caption-bridge-0-1",
                "cpt_task": "ui_caption",
                "images": [
                    str(images / "screen-0.png"),
                    str(images / "screen-1.png"),
                ],
                "conversations": [
                    {"from": "human", "value": "<image><image>compare"},
                    {"from": "gpt", "value": "ok"},
                ],
            }
            with (root / "annotations" / "ui_caption.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(bridge, ensure_ascii=False) + "\n")

            output = root / "split"
            split_recipe(
                recipe,
                output,
                seed=20260826,
                val_fraction=0.2,
                val_fast_per_task=3,
            )
            rows = [
                json.loads(line)
                for line in (output / "diagnostics" / "split_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            connected_ids = {
                "ui_caption:caption-0",
                "ui_caption:caption-alias",
                "ui_caption:caption-1",
                "ui_caption:caption-bridge-0-1",
                "vqa:vqa-0",
                "vqa:vqa-1",
            }
            connected = [row for row in rows if row["record_id"] in connected_ids]
            self.assertEqual({row["record_id"] for row in connected}, connected_ids)
            self.assertEqual(len({row["group_id"] for row in connected}), 1)
            self.assertEqual(len({row["split"] for row in connected}), 1)
            validate_split_manifest(output / "diagnostics" / "split_manifest.jsonl")

    def test_validator_rejects_content_and_path_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.jsonl"
            rows = [
                {
                    "split": "train",
                    "group_id": "group-a",
                    "record_id": "record-a",
                    "image": ["/data/same.png"],
                    "image_sha256": ["digest-a"],
                },
                {
                    "split": "heldout",
                    "group_id": "group-b",
                    "record_id": "record-b",
                    "image": ["/data/same.png"],
                    "image_sha256": ["digest-a"],
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(NormalizeError, "leakage"):
                validate_split_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
