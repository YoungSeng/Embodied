from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import test_ui5_curriculum_recipe as fixtures


recipe = fixtures.curriculum_recipe


class CurriculumCropReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture = fixtures.CurriculumRecipeTests(methodName="runTest")
        self.bundle, self.old_difficulty = self.fixture._fixture(self.root)
        self.source = self.root / "hour009-curriculum"
        self.old = self.build(self.fixture._args(self.bundle, self.old_difficulty, self.source))
        self.target = self.root / "hour018-curriculum"
        self.difficulty = self.root / "hour018" / "complete8.jsonl"
        rows = recipe._read_jsonl(self.old_difficulty)
        # More completed rollout evidence changes both hard membership and count.
        for row in rows:
            if row["sample_id"] in {"hard-pos", "hard-content"}:
                row["crop_correct_count"] = 2
            if row["sample_id"] == "replay-occ-pos":
                row["crop_correct_count"] = 0
        fixtures.write_jsonl(self.difficulty, rows)
        ids = sorted(row["sample_id"] for row in rows if row["crop_correct_count"] == 0)
        (self.difficulty.parent / "summary.json").write_text(json.dumps({
            "unique_complete8_samples": len(rows), "formal_eligible_groups": len(rows),
            "formal_crop_hard_groups": len(ids), "formal_crop_hard_sample_ids": ids,
            "formal_crop_hard_sample_ids_sha256": recipe._json_digest(ids),
        }), encoding="utf-8")

    @staticmethod
    def build(args):
        args.progress_interval_seconds = 1000
        with contextlib.redirect_stderr(io.StringIO()):
            return recipe.build(args)

    def args(self):
        args = self.fixture._args(self.bundle, self.difficulty, self.target)
        args.expected_hard_groups = None
        args.reuse_crops_from = self.source
        return args

    def source_fingerprints(self):
        return {
            path.relative_to(self.source).as_posix(): recipe._sha256_file(path)
            for path in self.source.rglob("*") if path.is_file()
        }

    def test_reuses_every_png_without_cropping_or_encoding_and_rebuilds_pools(self):
        before = self.source_fingerprints()
        with mock.patch.object(recipe.Image.Image, "crop", side_effect=AssertionError("recropping")), \
             mock.patch.object(recipe.Image.Image, "save", side_effect=AssertionError("encoding")):
            new = self.build(self.args())
        self.assertEqual(self.source_fingerprints(), before)
        self.assertNotEqual(new["identity_digest"], self.old["identity_digest"])
        self.assertNotEqual(new["outputs"]["crop_asset_namespace"], self.old["outputs"]["crop_asset_namespace"])
        self.assertEqual(new["hard_groups"], 2)
        self.assertEqual(new["matched_anchor_groups"], 2)
        self.assertEqual(
            {row["sample_id"] for row in recipe._read_jsonl(self.target / "hard_groups.jsonl")},
            {"hard-neg", "replay-occ-pos"},
        )
        self.assertEqual(new["crop_asset_reuse"]["generated_crop_assets"], 0)
        self.assertEqual(new["crop_asset_reuse"]["reused_crop_assets"], len(self.old["crop_assets"]))
        old_assets = {row["crop_id"]: row for row in self.old["crop_assets"]}
        for row in new["crop_assets"]:
            old = old_assets[row["crop_id"]]
            self.assertEqual(row["sha256"], old["sha256"])
            self.assertEqual(row["crop_xyxy"], old["crop_xyxy"])
            self.assertTrue(os.path.samefile(
                self.source / old["relative_path"], self.target / row["relative_path"],
            ))
        events = recipe._read_jsonl(self.target / "progress" / "build_progress.jsonl")
        self.assertIn("reuse_crop_pngs", {row["stage"] for row in events})
        self.assertNotIn("materialize_crop_pngs", {row["stage"] for row in events})
        self.assertTrue((self.target / "_SUCCESS.json").is_file())

    def test_reused_supervision_matches_fresh_build_for_new_snapshot(self):
        reused = self.build(self.args())
        fresh_root = self.root / "fresh-hour018"
        fresh_args = self.fixture._args(self.bundle, self.difficulty, fresh_root)
        fresh_args.expected_hard_groups = None
        fresh = self.build(fresh_args)
        self.assertEqual(reused["crop_assets"], fresh["crop_assets"])
        self.assertEqual(reused["pools"], fresh["pools"])
        for name in ("hard.jsonl", "matched_anchor.jsonl", "global_replay.jsonl"):
            actual = (self.target / name).read_text().replace(str(self.target).replace("\\", "\\\\"), "OUTPUT")
            expected = (fresh_root / name).read_text().replace(str(fresh_root).replace("\\", "\\\\"), "OUTPUT")
            self.assertEqual(actual, expected)
        self.assertEqual(reused["inputs"], fresh["inputs"])

    def test_completed_destination_reuses_without_source_dependency(self):
        new = self.build(self.args())
        args = self.args()
        args.reuse_crops_from = self.root / "not-needed-after-publication"
        with mock.patch.object(recipe, "_materialize_crop_assets", side_effect=AssertionError("materialize")):
            self.assertEqual(self.build(args), new)

    def test_incomplete_source_fails_without_recropping(self):
        (self.source / "_SUCCESS.json").unlink()
        with mock.patch.object(recipe.Image.Image, "crop", side_effect=AssertionError("recropping")):
            with self.assertRaisesRegex(RuntimeError, "source is incomplete"):
                self.build(self.args())
        self.assertFalse((self.target / "_SUCCESS.json").exists())

    def test_corrupt_source_png_fails_without_publication_or_fallback(self):
        image = self.source / self.old["crop_assets"][0]["relative_path"]
        payload = image.read_bytes()
        image.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaisesRegex(RuntimeError, "crop reuse PNG changed"):
            self.build(self.args())
        self.assertFalse((self.target / "_SUCCESS.json").exists())

    def test_bundle_change_rejects_all_asset_reuse(self):
        manifest_path = self.bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["different_bundle_revision"] = True
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "bundle identity differs"):
            self.build(self.args())

    def test_hardlink_failure_never_copies_or_recrops(self):
        before = self.source_fingerprints()
        with mock.patch.object(recipe.os, "link", side_effect=OSError("EXDEV")), \
             mock.patch.object(recipe.shutil, "copyfile", side_effect=AssertionError("copy")), \
             mock.patch.object(recipe.Image.Image, "crop", side_effect=AssertionError("recrop")):
            with self.assertRaisesRegex(RuntimeError, "same filesystem"):
                self.build(self.args())
        self.assertEqual(self.source_fingerprints(), before)
        self.assertFalse((self.target / "_SUCCESS.json").exists())

    def test_exact_crop_ids_and_geometry_are_required(self):
        current_bundle, _ = recipe._verify_rollout_bundle(self.bundle)
        assets = [{
            **row, "source_image": "unused",
            "crop_size": [row["width"], row["height"]],
        } for row in self.old["crop_assets"]]
        with self.assertRaisesRegex(RuntimeError, "ID set differs"):
            recipe._load_crop_reuse_inventory(self.source, self.target, current_bundle, assets[:-1], None)
        assets[0]["crop_xyxy"] = [0, 0, 1, 1]
        with self.assertRaisesRegex(RuntimeError, "geometry/source mismatch"):
            recipe._load_crop_reuse_inventory(self.source, self.target, current_bundle, assets, None)

    def test_tampered_manifest_identity_is_rejected(self):
        path = self.source / "curriculum_manifest.json"
        value = json.loads(path.read_text())
        value["crop_assets"][0]["crop_xyxy"] = [0, 0, 1, 1]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.build(self.args())

    def test_cli_reuse_option_and_shell_forwarding(self):
        parsed = recipe.parse_args([
            "--rollout-difficulty", str(self.difficulty), "--output-dir", str(self.target),
            "--reuse-crops-from", str(self.source),
        ])
        self.assertEqual(parsed.reuse_crops_from, self.source)
        for name in ("run", "preflight"):
            source = (fixtures.PROJECT_ROOT / "shell" / f"{name}_locany_ui5_crop_rollout4_curriculum_h20x2.sh").read_text()
            self.assertIn('--reuse-crops-from "${CURRICULUM_REUSE_CROPS_FROM}"', source)


if __name__ == "__main__":
    unittest.main()
