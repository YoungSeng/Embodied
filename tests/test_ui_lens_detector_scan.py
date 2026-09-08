from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_ui5_eval_detector_crops as preparer
import run_ui5_parallel_inference as parallel
from locany_ui5_common import TASK_JSONL
from ui5_eval_detector_cache import validate_eval_detector_cache


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ExternalDetectorCacheTest(unittest.TestCase):
    def test_external_count_and_full_dataset_requirement(self) -> None:
        args = argparse.Namespace(cache_scope="external_test", expected_unique_images=None)
        self.assertEqual(
            preparer._resolve_cache_scope(args, {"max_images_per_task": 0}, 17),
            ("external_test", 0, 17),
        )
        with self.assertRaisesRegex(RuntimeError, "contradicts prepared manifest"):
            preparer._resolve_cache_scope(args, {"max_images_per_task": 2}, 17)
        args.expected_unique_images = 18
        with self.assertRaisesRegex(RuntimeError, "count mismatch"):
            preparer._resolve_cache_scope(args, {"max_images_per_task": 0}, 17)

    def test_unequal_tasks_build_validate_and_bind_before_inference(self) -> None:
        # The detector outputs are synthetic; manifest construction, geometry,
        # publication, and the inference launcher's cache validation are real.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, cache = root / "data", root / "cache"
            images = [root / "page1.png", root / "page2.png"]
            for image, color in zip(images, ("white", "gray")):
                Image.new("RGB", (600, 2400), color).save(image)
            counts = {}
            for task, filename in TASK_JSONL.items():
                selected = images[:1] if task == "content_missing" else images
                counts[task] = len(selected)
                # Deliberately omit answers: no GT is needed to create crops.
                write_rows(data / filename, [{"images": [str(p)]} for p in selected])
            args = preparer.parse_args([
                "--stage", "prepare", "--input-dir", str(data),
                "--output-dir", str(cache), "--parser-root", str(root),
                "--cache-scope", "external_test", "--visualization-samples", "2",
            ])
            unique = preparer.prepare_manifest(args)
            self.assertEqual(len(unique), 2)
            detections = [dict(
                row, image=row["image_path"],
                text_detections=[{"bbox": [20, 900, 100, 1020], "score": 0.9}],
                icon_detections=[{"bbox": [500, 910, 580, 1030], "score": 0.8}],
            ) for row in unique]
            write_rows(cache / "detections/merged/detections.jsonl", detections)
            for stage in ("text", "icon"):
                stage_dir = cache / "detections" / stage
                write_json(stage_dir / "stage_summary.json", {
                    "images": 2, "workers": 1, "runtime": {"python": stage},
                })
                write_rows(stage_dir / "shard_00000.jsonl", detections)
                write_json(stage_dir / "shard_00000.done.json", {"stage": stage, "count": 2})
            scans = preparer.build_scan_crops(args)
            self.assertTrue(all(row["gt_used"] is False for row in scans))
            self.assertTrue(any(len(row["tiles"]) > 1 for row in scans))
            scan_root = cache / args.scan_name
            self.assertTrue((scan_root / "gallery/index.html").is_file())
            self.assertTrue(list((scan_root / "preview_crops").glob("*.png")))
            marker = validate_eval_detector_cache(
                cache, scan_name=args.scan_name, input_dir=data,
                expected_unique_images=2, required_cache_scope="external_test",
                require_strict_nonoverlap=True, require_raw_detector_edge_alignment=True,
                require_detector_unique_containment=True,
            )
            self.assertEqual(marker["expected_unique_images"], 2)
            self.assertEqual(
                {row["task"]: row["jsonl_rows"] for row in marker["dataset"]["task_files"]},
                counts,
            )
            summary = json.loads((scan_root / "summary.json").read_text(encoding="utf-8"))
            missing = summary["by_task"]["content_missing"]
            self.assertEqual(missing["effective_mode"], "full_image_global_view")
            self.assertEqual(missing["tile_count_max"], 1)
            manifest = scan_root / "detector_scan_crops.jsonl"
            self.assertIsNone(parallel.validated_full_test_expected_images_per_task(
                manifest, input_dir=data,
            ))
            # The external mode must still reject a different evaluation dataset.
            wrong_data = root / "wrong_data"
            wrong_data.mkdir()
            for filename in TASK_JSONL.values():
                (wrong_data / filename).write_bytes((data / filename).read_bytes())
            with self.assertRaisesRegex(RuntimeError, "dataset path mismatch"):
                parallel.validated_full_test_expected_images_per_task(manifest, input_dir=wrong_data)
            # Editing a task's labels invalidates the bound cache before workers.
            changed = data / TASK_JSONL["content_missing"]
            write_rows(changed, [{"images": [str(p)]} for p in images])
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                parallel.validated_full_test_expected_images_per_task(manifest, input_dir=data)


if __name__ == "__main__":
    unittest.main()
