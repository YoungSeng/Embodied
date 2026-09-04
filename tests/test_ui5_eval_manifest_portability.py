from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from locany_ui5_common import TASK_JSONL  # noqa: E402
from relocate_ui5_eval_detector_manifest import prepare_manifest  # noqa: E402
from ui5_lossless_tiling import generate_detector_scan_plan  # noqa: E402


class DetectorManifestPortabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "h20" / "evaluation"
        self.input_dir.mkdir(parents=True)
        self.images: list[Path] = []
        self.rows: list[dict] = []
        for number, color in enumerate(("red", "blue")):
            image = self.input_dir / f"current-{number}.png"
            Image.new("RGB", (24, 40), color).save(image)
            self.images.append(image)
            digest = hashlib.blake2b(image.read_bytes(), digest_size=20).hexdigest()
            old_path = self.root / "a800" / f"previous-{number}.png"
            self.rows.append(
                {
                    "image_id": f"eval_{digest[:20]}",
                    "content_id": digest,
                    "image_path": str(old_path),
                    "image_paths": [str(old_path)],
                    "width": 24,
                    "height": 40,
                    "tasks": list(TASK_JSONL),
                    "text_detections": [],
                    "icon_detections": [],
                    "geometry_config_digest": "must-remain-identical",
                    "unknown_future_field": {"nested": [1, 2, "preserve"]},
                    **generate_detector_scan_plan(24, 40, []),
                }
            )
        for task in TASK_JSONL:
            self.write_task(task, self.images)
        self.source = self.root / "detector_scan_crops.jsonl"
        self.output = self.root / "detector_scan_crops.h20.jsonl"
        self.write_source()

    def write_source(self) -> None:
        self.source.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8"
        )

    def write_task(self, task: str, paths: list[Path]) -> None:
        (self.input_dir / TASK_JSONL[task]).write_text(
            "".join(json.dumps({"images": [str(path)]}) + "\n" for path in paths),
            encoding="utf-8",
        )

    def prepare(self, **kwargs) -> dict:
        kwargs.setdefault("expected_unique_images", 2)
        return prepare_manifest(self.source, self.input_dir, **kwargs)

    def assert_rejected(self, **kwargs) -> None:
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError, AssertionError)):
            self.prepare(**kwargs)

    def test_cross_mount_matches_content_and_preserves_every_nonpath_field(self) -> None:
        original_bytes = self.source.read_bytes()
        self.prepare(output_manifest=self.output)
        self.assertEqual(self.source.read_bytes(), original_bytes)
        migrated = [json.loads(line) for line in self.output.read_text().splitlines()]
        by_content = {row["content_id"]: row for row in migrated}
        self.assertEqual(len(by_content), 2)
        for original, image in zip(self.rows, self.images):
            row = by_content[original["content_id"]]
            self.assertEqual(Path(row["image_path"]).resolve(), image.resolve())
            aliases = {Path(path).resolve() for path in row["image_paths"]}
            self.assertIn(image.resolve(), aliases)
            self.assertEqual(
                {key: value for key, value in row.items() if key not in {"image_path", "image_paths"}},
                {key: value for key, value in original.items() if key not in {"image_path", "image_paths"}},
            )

    def test_helper_import_never_requires_model_or_gpu_runtime(self) -> None:
        code = """
import builtins, runpy, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'accelerate', 'deepspeed'}:
        raise AssertionError('CPU manifest helper imported model runtime: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
runpy.run_path(sys.argv[2], run_name='manifest_cpu_import_test')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(SCRIPTS), str(SCRIPTS / "relocate_ui5_eval_detector_manifest.py")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_original_a800_aliases_fail_read_only_validation(self) -> None:
        original_bytes = self.source.read_bytes()
        self.assert_rejected()
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertFalse(self.output.exists())

    def test_migrated_manifest_passes_read_only_validation_without_mutation(self) -> None:
        self.prepare(output_manifest=self.output)
        before = self.output.read_bytes()
        prepare_manifest(self.output, self.input_dir, expected_unique_images=2)
        self.assertEqual(self.output.read_bytes(), before)

    def test_same_output_can_be_repeated_but_different_existing_bytes_are_preserved(self) -> None:
        self.prepare(output_manifest=self.output)
        first = self.output.read_bytes()
        self.prepare(output_manifest=self.output)
        self.assertEqual(self.output.read_bytes(), first)
        self.output.write_text("existing-user-file\n", encoding="utf-8")
        self.assert_rejected(output_manifest=self.output)
        self.assertEqual(self.output.read_text(), "existing-user-file\n")

    def test_source_cannot_be_output(self) -> None:
        before = self.source.read_bytes()
        self.assert_rejected(output_manifest=self.source)
        self.assertEqual(self.source.read_bytes(), before)

    def test_symlink_to_source_cannot_be_output(self) -> None:
        alias = self.root / "source-link.jsonl"
        try:
            os.symlink(self.source, alias)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        before = self.source.read_bytes()
        self.assert_rejected(output_manifest=alias)
        self.assertEqual(self.source.read_bytes(), before)

    def test_same_filename_with_changed_bytes_is_not_treated_as_same_image(self) -> None:
        self.rows[0]["image_path"] = str(self.images[0])
        self.rows[0]["image_paths"] = [str(self.images[0])]
        self.write_source()
        Image.new("RGB", (24, 40), "green").save(self.images[0])
        self.assert_rejected(output_manifest=self.output)
        self.assertFalse(self.output.exists())

    def test_same_content_multiple_h20_paths_all_receive_aliases(self) -> None:
        alias = self.input_dir / "another-name.png"
        alias.write_bytes(self.images[0].read_bytes())
        self.write_task("occlusion", [*self.images, alias])
        self.prepare(output_manifest=self.output)
        migrated = [json.loads(line) for line in self.output.read_text().splitlines()]
        row = next(row for row in migrated if row["content_id"] == self.rows[0]["content_id"])
        aliases = {Path(path).resolve() for path in row["image_paths"]}
        self.assertTrue({alias.resolve(), self.images[0].resolve()}.issubset(aliases))

    def test_missing_image_fails_instead_of_silently_shrinking_evaluation(self) -> None:
        self.write_task("occlusion", [*self.images, self.input_dir / "missing.png"])
        self.assert_rejected(output_manifest=self.output)

    def test_figma_colon_names_are_excluded_before_existence_or_content_checks(self) -> None:
        self.write_task("occlusion", [*self.images, self.input_dir / "figma:node.png"])
        self.prepare(output_manifest=self.output)
        migrated = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(len(migrated), 2)
        self.assertFalse(any("figma:node.png" in str(row) for row in migrated))

    def test_missing_task_jsonl_fails(self) -> None:
        (self.input_dir / TASK_JSONL["text_ellipsis"]).unlink()
        self.assert_rejected(output_manifest=self.output)

    def test_malformed_task_json_and_missing_image_field_fail(self) -> None:
        task_path = self.input_dir / TASK_JSONL["occlusion"]
        for invalid in ("{broken JSON\n", '{"messages": []}\n'):
            with self.subTest(invalid=invalid):
                task_path.write_text(invalid, encoding="utf-8")
                self.assert_rejected(output_manifest=self.output)

    def test_source_and_input_must_have_exact_same_content_set(self) -> None:
        self.rows.pop()
        self.write_source()
        self.assert_rejected(output_manifest=self.output)

    def test_unreferenced_source_content_is_rejected(self) -> None:
        for task in TASK_JSONL:
            self.write_task(task, [self.images[0]])
        self.assert_rejected(output_manifest=self.output, expected_unique_images=1)

    def test_expected_content_count_is_enforced(self) -> None:
        self.assert_rejected(output_manifest=self.output, expected_unique_images=3)

    def test_duplicate_source_content_or_image_id_is_ambiguous(self) -> None:
        original = copy.deepcopy(self.rows)
        for field in ("content_id", "image_id"):
            with self.subTest(field=field):
                self.rows = copy.deepcopy(original)
                self.rows[1][field] = self.rows[0][field]
                self.write_source()
                self.assert_rejected(output_manifest=self.output)

    def test_conflicting_source_path_aliases_are_rejected(self) -> None:
        self.rows[1]["image_paths"].append(self.rows[0]["image_path"])
        self.write_source()
        self.assert_rejected(output_manifest=self.output)

    def test_dimensions_must_match_actual_h20_image(self) -> None:
        self.rows[0]["width"] = 25
        self.rows[0].update(generate_detector_scan_plan(25, 40, []))
        self.write_source()
        self.assert_rejected(output_manifest=self.output)

    def test_gt_used_and_incomplete_or_declared_cut_geometry_fail(self) -> None:
        original = copy.deepcopy(self.rows)
        mutations = (
            {"gt_used": True},
            {"tiles": [[0, 0, 24, 39]]},
            {"detector_boundary_cut_count": 1},
        )
        for changes in mutations:
            with self.subTest(changes=changes):
                self.rows = copy.deepcopy(original)
                self.rows[0].update(changes)
                self.write_source()
                self.assert_rejected(output_manifest=self.output)

    def test_cut_detector_box_is_rejected_even_when_declared_counter_is_zero(self) -> None:
        self.rows[0]["tiles"] = [[0, 0, 24, 20], [0, 20, 24, 40]]
        self.rows[0]["text_detections"] = [{"bbox": [2, 15, 22, 25]}]
        self.rows[0]["detector_boundary_cut_count"] = 0
        self.rows[0]["every_seam_is_raw_detector_edge"] = True
        self.write_source()
        self.assert_rejected(output_manifest=self.output)

    def test_content_missing_needs_original_image_but_not_detector_alias(self) -> None:
        for task in TASK_JSONL:
            self.write_task(task, self.images if task == "content_missing" else [self.images[0]])
        self.rows[0]["image_path"] = str(self.images[0])
        self.rows[0]["image_paths"] = [str(self.images[0])]
        self.write_source()
        before = self.source.read_bytes()
        self.prepare()
        self.assertEqual(self.source.read_bytes(), before)
        self.images[1].unlink()
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
