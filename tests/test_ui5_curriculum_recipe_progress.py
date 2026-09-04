from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests import test_ui5_curriculum_recipe as recipe_tests


class CurriculumRecipeProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture = recipe_tests.CurriculumRecipeTests(methodName="runTest")
        self.bundle, self.difficulty = fixture._fixture(self.root)
        self.output = self.root / "curriculum"
        self.args = fixture._args(self.bundle, self.difficulty, self.output)
        self.args.progress_interval_seconds = 1000.0
        self.stderr = io.StringIO()

    def _build(self) -> dict:
        with contextlib.redirect_stderr(self.stderr):
            return recipe_tests.curriculum_recipe.build(self.args)

    def _events(self) -> list[dict]:
        path = self.output / "progress" / "build_progress.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _latest(self) -> dict:
        return json.loads(
            (self.output / "progress" / "build_progress.json").read_text(encoding="utf-8")
        )

    def _success(self) -> dict:
        return json.loads((self.output / "_SUCCESS.json").read_text(encoding="utf-8"))

    def test_real_asset_counts_and_success_are_reported_without_changing_inventory(self) -> None:
        summary = self._build()
        events = self._events()
        completed = {
            event["stage"]: event
            for event in events
            if event["event"] == "stage_completed"
        }
        source_images = recipe_tests.curriculum_recipe._read_jsonl(
            self.bundle / "manifest" / "unique_images.jsonl"
        )
        expected_counts = {
            "verify_bundle_images": len(source_images),
            "materialize_crop_pngs": len(summary["crop_assets"]),
            "verify_published_crop_pngs": len(summary["crop_assets"]),
            "verify_final_artifacts": len(self._success()["files"]),
        }
        for stage, total in expected_counts.items():
            with self.subTest(stage=stage):
                self.assertGreater(total, 0)
                self.assertEqual(completed[stage]["total"], total)
                self.assertEqual(completed[stage]["completed"], total)
                self.assertEqual(completed[stage]["percent"], 100.0)
                self.assertEqual(completed[stage]["eta_seconds"], 0.0)
                self.assertEqual(completed[stage]["eta_scope"], "current_stage")

        self.assertEqual(events[0]["event"], "build_started")
        self.assertEqual(events[-1]["event"], "build_completed")
        self.assertEqual(events[-1]["status"], "complete")
        self.assertEqual(self._latest(), events[-1])
        self.assertIn("[CURRICULUM PROGRESS]", self.stderr.getvalue())
        self.assertIn("stage_eta=", self.stderr.getvalue())
        for relative in self._success()["files"]:
            self.assertNotIn("progress", Path(relative).parts)
            self.assertNotIn("build_progress", relative)
        self.assertNotIn("build_progress", json.dumps(summary))

    def test_reuse_reports_verification_and_keeps_all_durable_bytes_identical(self) -> None:
        summary = self._build()
        first_run = self._latest()["run_id"]
        durable_paths = set(self._success()["files"]) | {
            "curriculum_manifest.json", "_SUCCESS.json"
        }
        original = {relative: (self.output / relative).read_bytes() for relative in durable_paths}

        reused = self._build()
        latest = self._latest()
        self.assertNotEqual(latest["run_id"], first_run)
        self.assertEqual(reused, summary)
        self.assertEqual(reused["identity_digest"], summary["identity_digest"])
        self.assertEqual(
            {relative: (self.output / relative).read_bytes() for relative in durable_paths},
            original,
        )
        run_events = [event for event in self._events() if event["run_id"] == latest["run_id"]]
        completed = [
            event for event in run_events
            if event["event"] == "stage_completed"
            and event["stage"] == "verify_existing_curriculum"
        ]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["completed"], len(self._success()["files"]))
        self.assertEqual(completed[0]["total"], len(self._success()["files"]))
        self.assertNotIn("materialize_crop_pngs", {event["stage"] for event in run_events})
        self.assertEqual(latest["event"], "build_completed")

    def test_corrupt_asset_leaves_failed_progress_for_that_run(self) -> None:
        summary = self._build()
        first_run = self._latest()["run_id"]
        success_bytes = (self.output / "_SUCCESS.json").read_bytes()
        asset = self.output / summary["crop_assets"][0]["relative_path"]
        original = asset.read_bytes()
        asset.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

        with self.assertRaisesRegex(RuntimeError, "existing curriculum artifact changed"):
            self._build()
        latest = self._latest()
        self.assertNotEqual(latest["run_id"], first_run)
        self.assertEqual(latest["event"], "build_failed")
        self.assertEqual(latest["status"], "failed")
        self.assertIn("existing curriculum artifact changed", latest["error"])
        failing_events = [
            event for event in self._events() if event["run_id"] == latest["run_id"]
        ]
        self.assertNotIn("build_completed", {event["event"] for event in failing_events})
        self.assertEqual((self.output / "_SUCCESS.json").read_bytes(), success_bytes)

    def test_cli_defaults_to_compact_summary_and_full_manifest_is_explicit(self) -> None:
        argv = [
            "--rollout-difficulty", str(self.difficulty),
            "--rollout-bundle-root", str(self.bundle),
            "--output-dir", str(self.output),
            "--progress-interval-seconds", "1000",
            "--seed", "42",
        ]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(self.stderr):
            self.assertEqual(recipe_tests.curriculum_recipe.main(argv), 0)
        compact = json.loads(stdout.getvalue())
        manifest = json.loads(
            (self.output / "curriculum_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIs(compact["complete"], True)
        self.assertIsInstance(compact["crop_assets"], int)
        self.assertEqual(compact["crop_assets"], len(manifest["crop_assets"]))
        self.assertEqual(compact["identity_digest"], manifest["identity_digest"])
        self.assertNotIn("relative_path", stdout.getvalue())
        self.assertNotIn("[CURRICULUM PROGRESS]", stdout.getvalue())
        self.assertIn("[CURRICULUM PROGRESS]", self.stderr.getvalue())

        full_stdout = io.StringIO()
        with contextlib.redirect_stdout(full_stdout), contextlib.redirect_stderr(self.stderr):
            self.assertEqual(
                recipe_tests.curriculum_recipe.main([*argv, "--print-full-summary"]), 0
            )
        self.assertEqual(json.loads(full_stdout.getvalue()), manifest)


if __name__ == "__main__":
    unittest.main()
