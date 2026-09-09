"""CPU regression for stale UI14 environment variables and read-only audit import."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ui14_common import read_json, write_json
from ui14_alignment_common import DATA, NEG_DATA, OLD_MANIFEST, alignment_destination
from ui14_alignment_context import parse_args, run


def state(folder):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ctime_ns)
            for p in Path(folder).rglob("*") if p.is_file()}


class AlignmentPathTests(unittest.TestCase):
    def test_old_generic_environment_is_ignored_in_python_entry(self):
        with mock.patch.dict(os.environ, {"UI14_DATA_ROOT": NEG_DATA,
                "UI14_PARENT_DATA_ROOT": str(Path(OLD_MANIFEST).parent)}, clear=True):
            args = parse_args(["prepare"])
        self.assertEqual(args.data_root, DATA)
        self.assertEqual(args.parent_root, NEG_DATA)

    def test_namespaced_environment_and_explicit_cli_precedence(self):
        with mock.patch.dict(os.environ, {"WORKSPACE": "/workspace",
                "UI14_ALIGNMENT_DATA_ROOT": "/new-alignment",
                "UI14_ALIGNMENT_PARENT_DATA_ROOT": "/frozen-neg11"}, clear=True):
            args = parse_args(["prepare"])
            explicit = parse_args(["prepare", "--data-root", "/explicit", "--parent-root", "/parent"])
        self.assertEqual((args.data_root, args.parent_root), ("/new-alignment", "/frozen-neg11"))
        self.assertEqual((explicit.data_root, explicit.parent_root), ("/explicit", "/parent"))

    def test_old_default_data_roots_are_blocked_even_with_wrong_parent(self):
        for root in (NEG_DATA, str(Path(OLD_MANIFEST).parent)):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "overlaps"):
                alignment_destination(root, Path(root).parent / "another-parent")

    def test_existing_custom_dataset_is_rejected_before_writing_logs(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            old = Path(d) / "old-custom"
            write_json(old / "source_snapshot.json", {"old": True})
            before = state(old)
            args = parse_args(["audit-errors", "--data-root", str(old), "--parent-root", str(Path(d) / "other")])
            with self.assertRaisesRegex(ValueError, "existing UI14 dataset"): run(args)
            self.assertEqual(before, state(old))
            self.assertFalse((old / "progress.json").exists())

    def test_wrong_parent_reports_missing_extension_before_any_output(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            parent, output = Path(d) / "repair", Path(d) / "alignment"
            for name in ("source_snapshot.json", "cpu_check_report.json", "normalization_stats.json",
                         "evaluation_manifest.json"): write_json(parent / name, {})
            before = state(parent)
            args = parse_args(["prepare", "--data-root", str(output), "--parent-root", str(parent)])
            with self.assertRaisesRegex(FileNotFoundError, "negative_extension_manifest.json") as error:
                run(args)
            self.assertIn(str(parent), str(error.exception))
            self.assertIn("UI14_ALIGNMENT_PARENT_DATA_ROOT", str(error.exception))
            self.assertFalse(output.exists())
            self.assertEqual(before, state(parent))

    def test_stale_environment_audit_writes_only_new_experiment(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as log:
            root = Path(d)
            parent = root / "gui_data/ui14_cpt9000_neg11_v1"
            repair = root / "gui_data/ui14_cpt9000_repair_v2"
            write_json(parent / "negative_extension_manifest.json", {"keep": True})
            write_json(repair / "source_snapshot.json", {"keep": True})
            parent_before, repair_before = state(parent), state(repair)
            with mock.patch.dict(os.environ, {"WORKSPACE": str(root), "UI14_DATA_ROOT": str(parent),
                                             "UI14_PARENT_DATA_ROOT": str(repair)}, clear=True):
                args = parse_args(["audit-errors"])
                with mock.patch("ui14_alignment_audit.audit_runs", return_value=[{"status": "complete"}]):
                    run(args)
            self.assertTrue((Path(args.data_root) / "stage_summaries/audit-errors.json").is_file())
            self.assertEqual(parent_before, state(parent))
            self.assertEqual(repair_before, state(repair))
            self.assertIn("ignored inherited UI14_DATA_ROOT=", log.getvalue())
            self.assertIn("ignored inherited UI14_PARENT_DATA_ROOT=", log.getvalue())

    def test_actual_shell_wrapper_ignores_previous_experiment_roots(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else "bash"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            recorder = root / "arguments.py"
            recorder.write_text("import json, os\nprint(json.dumps({k:os.environ.get(k) for k in "
                "['UI14_ALIGNMENT_DATA_ROOT','UI14_ALIGNMENT_PARENT_DATA_ROOT']}))\n", encoding="utf-8")
            # Run the actual wrapper; replace only its final Python program
            # argument with a recorder so no dataset/GPU is required.
            wrapper = (ROOT / "shell/ui14_alignment_context_a800.sh").read_text()
            wrapper = wrapper.replace('-u scripts/ui14_alignment_context.py "$@"', '-u "' + recorder.as_posix() + '" "$@"')
            env = {**os.environ, "UI14_PYTHON": sys.executable, "WORKSPACE": "/workspace",
                   "UI14_DATA_ROOT": "/stale-neg11", "UI14_PARENT_DATA_ROOT": "/stale-repair",
                   "MSYS2_ENV_CONV_EXCL": "*"}
            for name in ("UI14_ALIGNMENT_DATA_ROOT", "UI14_ALIGNMENT_PARENT_DATA_ROOT"): env.pop(name, None)
            result = subprocess.run([str(bash), "-c", "export PATH=/usr/bin:/bin:$PATH\n" + wrapper],
                                    cwd=ROOT, env=env, text=True, capture_output=True, check=True)
            import json
            selected = json.loads(result.stdout)
            self.assertEqual(selected["UI14_ALIGNMENT_DATA_ROOT"], "/workspace/gui_data/ui14_alignment_context_v1")
            self.assertEqual(selected["UI14_ALIGNMENT_PARENT_DATA_ROOT"], "/workspace/gui_data/ui14_cpt9000_neg11_v1")


class AlignmentAuditImportTests(unittest.TestCase):
    def test_completed_533f36b_audit_imports_without_scoring_and_keeps_old_files(self):
        from tests.test_ui14_alignment_audit import evidence_fixture
        from ui14_alignment_audit import audit_runs, audit_step
        from ui14_alignment_audit_reuse import import_completed_audits
        import qwen3vl_merge_and_score_fixed_5tasks as scorer
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            old, manifest, _ = evidence_fixture(root)
            source = root / "old-neg11/historical_audit"
            old_results = audit_runs(old, manifest, ["ui_alignment"], [2000, 4000], source)
            before = state(root / "old-neg11")
            destination = root / "alignment/historical_audit"
            imported = import_completed_audits(source, destination, ["ui_alignment"], [2000, 4000])
            self.assertEqual(imported["imported"], 2)
            with mock.patch.object(scorer, "evaluate_samples", side_effect=AssertionError("completed audit rescored")):
                restored = audit_runs(old, manifest, ["ui_alignment"], [2000, 4000], destination)
            self.assertEqual([r["audit_id"] for r in old_results], [r["audit_id"] for r in restored])
            self.assertTrue(all(str(destination) in r["directory"] for r in restored))
            second = import_completed_audits(source, destination, ["ui_alignment"], [2000, 4000])
            self.assertEqual(second["reused"], 2)
            self.assertEqual(before, state(root / "old-neg11"))
            raw = old / "inference-checkpoint-2000-ui14/ui_alignment/raw/0.json"
            write_json(raw, {"raw_answer": "<box>none<20></box>"})
            changed = audit_step(old, manifest, "ui_alignment", 2000, destination)
            self.assertNotEqual(changed["audit_id"], restored[0]["audit_id"])

    def test_corrupt_or_partial_audit_cannot_publish_completed_import(self):
        from tests.test_ui14_alignment_audit import evidence_fixture
        from ui14_alignment_audit import audit_step
        from ui14_alignment_audit_reuse import import_completed_audits
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            old, manifest, _ = evidence_fixture(root)
            source, destination = root / "old-audit", root / "new-audit"
            report = audit_step(old, manifest, "ui_alignment", 2000, source)
            (Path(report["directory"]) / "samples.html").write_text("partial", encoding="utf-8")
            before = state(source)
            result = import_completed_audits(source, destination, ["ui_alignment"], [2000])
            self.assertEqual(result["imported"], 0)
            self.assertEqual(len(result["skipped"]), 1)
            self.assertFalse(destination.exists())
            self.assertEqual(before, state(source))


if __name__ == "__main__": unittest.main()
