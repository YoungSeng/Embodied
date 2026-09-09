"""Reproduce evaluation-success/startup-failure without GPU or source images."""
import ast
import contextlib
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ui14_common import read_json, write_json, UI_TASKS
from ui14_profile import validate_run_data_binding
from ui14_run_recovery import completed_evaluation, startup_only_changes, PIPELINE_BINDING_BLOCK
from run_ui14_eval import evaluation_identity
from eaglevl.train.ui5_excel_logger import UI5ExcelLogger, build_eval_rows


def export_metadata_from_actual_code(config, checkpoint, base):
    """Execute export -> loading policy -> initializer metadata -> manifest.

    Extract these CPU statements from production sources. Only tensor/model
    loading and parameter statistics are substituted; do not hand-write the
    initialization seed/reason in the successful checkpoint fixtures.
    """
    namespace = {"Any": object, "logger": mock.Mock()}
    setup = ast.parse((ROOT / "eaglevl/model/locany/ui_relation_setup.py").read_text(encoding="utf-8"))
    functions = [n for n in setup.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("ui_relation_loading_state", "initialize_or_validate_ui_relation")]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / "eaglevl/model/locany/ui_relation_setup.py"), "exec"), namespace)
    model_path = ROOT / "eaglevl/model/locany/modeling_locateanything.py"
    model_tree = ast.parse(model_path.read_text(encoding="utf-8"))
    initializer = next(n for n in ast.walk(model_tree) if isinstance(n, ast.FunctionDef)
                       and n.name == "initialize_ui_relation_modules")
    start = next(i for i, n in enumerate(initializer.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == "ui_relation_initialization_seed" for t in n.targets))
    initializer.body = initializer.body[start:]
    initializer.decorator_list = []
    exec(compile(ast.Module(body=[initializer], type_ignores=[]), str(model_path), "exec"), namespace)

    class CPUModel:
        initialize_ui_relation_modules = namespace["initialize_ui_relation_modules"]

        def __init__(self, model_config):
            self.config = model_config

        def named_parameters(self):
            return [("relation_pyramid.fixture", None), ("relation_pbd.fixture", None)]

        def validate_ui_relation_parameters(self):
            return {"parameters": 2, "values": 2, "checksum": .1}

        @classmethod
        def from_pretrained(cls, base_path, *, config, **kwargs):
            model = cls(config)
            return model, {"missing_keys": [name for name, _ in model.named_parameters()]}

    import json
    namespace.update(config=SimpleNamespace(**config), args=SimpleNamespace(seed=42, init_cpt_step=9000),
                     base=Path(base), output=checkpoint,
                     torch=SimpleNamespace(bfloat16="unused CPU fixture"),
                     LocateAnythingForConditionalGeneration=CPUModel, std=.02,
                     json=json, num_new_tokens=0, expanded_tied_aliases=[],
                     complete_marker=checkpoint / "ui5_checkpoint0_manifest.json")
    exporter = ROOT / "scripts/export_ui5_checkpoint0.py"
    main = next(n for n in ast.parse(exporter.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == "ui_relation_initialization_seed" for t in n.targets))
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "init_report" for t in n.targets))
    publish = next(n for n in main.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
                   and n.value.func.value.id == "complete_marker" and n.value.func.attr == "write_text")
    checkpoint.mkdir(parents=True, exist_ok=True)
    exec(compile(ast.Module(body=main.body[start:end + 1] + [publish], type_ignores=[]), str(exporter), "exec"), namespace)
    return vars(namespace["model"].config)


class StartupRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.output = self.root / "data", self.root / "run"
        self.checkpoint = self.output / "checkpoint-0"
        self.runtime = {"OUTPUT_DIR": str(self.output), "UI14_DATA_ROOT": str(self.data),
                        "UI_TASK_REGISTRY": str(self.data / "task_registry.json"),
                        "INIT_CHECKPOINT": str(self.root / "checkpoint-9000")}
        self.snapshot = {"normalization_id": "negatives-version", "repair_run_id": "repair-v2"}
        self.registry = [{**t.to_dict(), **self.snapshot} for t in UI_TASKS]
        write_json(self.data / "source_snapshot.json", self.snapshot)
        write_json(self.runtime["UI_TASK_REGISTRY"], {"tasks": self.registry})
        self.marker = self.output / "ui14_training_data.json"

    def export(self):
        config = export_metadata_from_actual_code(
            {"ui_task_registry": self.registry, "ui_num_tasks": 14},
            self.checkpoint, self.runtime["INIT_CHECKPOINT"])
        write_json(self.checkpoint / "config.json", config)
        # Real validator checks existence/shard completeness, never loads weights.
        (self.checkpoint / "pytorch_model.bin").write_bytes(b"CPU fixture, no model load")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def evaluation(self):
        self.export()
        self.git("init", "-q")
        (self.root / "scripts").mkdir()
        (self.root / "shell").mkdir()
        (self.root / "shell/run_locany_ui5_pipeline.sh").write_text('scripts/run_ui14_eval.py" --output-dir\n')
        (self.root / "scripts/run_ui14_eval.py").write_text("# fixed inference and scoring\n")
        self.git("add", "scripts", "shell")
        self.git("-c", "user.name=CPU Test", "-c", "user.email=cpu@example.invalid", "commit", "-qm", "baseline")
        self.old = self.git("rev-parse", "HEAD")
        (self.root / "scripts/ui14_profile.py").write_text("# startup binding fix\n")
        self.git("add", "scripts")
        self.git("-c", "user.name=CPU Test", "-c", "user.email=cpu@example.invalid", "commit", "-qm", "fix")
        self.new = self.git("rev-parse", "HEAD")
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(os.environ, {"GIT_COMMIT": self.new, "UI5_CONFIG_HASH": "same-runtime"}).start()
        self.manifest = self.data / "evaluation_manifest.json"
        write_json(self.manifest, {"eval_set_id": "unchanged-test-set"})
        identity = {**evaluation_identity(self.manifest, self.checkpoint), "git_commit": self.old}
        metric = dict(precision=.5, recall=.5, f1=.5, tp=1, fp=1, fn=1, tn=1)
        metrics = {"tasks": {t.task_key: {g: dict(metric) for g in ("image", "bbox")} for t in UI_TASKS}}
        now = datetime.now(timezone.utc).isoformat()
        self.state = self.output / "evaluation/ui14-step-0.json"
        write_json(self.state, {"status": "success", "tasks": metrics["tasks"], "identity": identity,
                               "sft_step": 0, "init_checkpoint": self.runtime["INIT_CHECKPOINT"],
                               "started": now, "finished": now, **self.snapshot})
        self.workbook = self.output / "diagnostics/ui5_training_evaluation.xlsx"
        logger = UI5ExcelLogger(self.workbook, [t.task_key for t in UI_TASKS])
        logger.append_eval(0, build_eval_rows(step=0, checkpoint=str(self.checkpoint), metrics=metrics))

    def complete(self):
        return completed_evaluation(self.output, 0, self.manifest, self.checkpoint,
                                    runtime=self.runtime, project=self.root)

    def test_fresh_binding_is_idempotent_and_batch_guard_remains(self):
        validate_run_data_binding(self.runtime, self.snapshot, create=True)
        before = self.marker.stat().st_mtime_ns
        self.export()
        validate_run_data_binding(self.runtime, self.snapshot, create=True)
        self.assertEqual(self.marker.stat().st_mtime_ns, before)
        with self.assertRaisesRegex(RuntimeError, "another repair batch"):
            validate_run_data_binding(self.runtime, {**self.snapshot, "normalization_id": "other"}, create=True)

    def test_legacy_export_readonly_check_then_recovery_never_reads_weight_bytes(self):
        self.export()
        config_bytes = (self.checkpoint / "config.json").read_bytes()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("no weight content reads")):
            validate_run_data_binding(self.runtime, self.snapshot)
            self.assertFalse(self.marker.exists())
            validate_run_data_binding(self.runtime, self.snapshot, create=True)
        self.assertEqual(read_json(self.marker)["normalization_id"], self.snapshot["normalization_id"])
        self.assertEqual((self.checkpoint / "config.json").read_bytes(), config_bytes)

    def test_missing_export_or_wrong_data_cpt_and_training_states_still_rejected(self):
        self.export()
        config_path = self.checkpoint / "config.json"
        original = read_json(config_path)
        for override in ({"init_cpt_step": 3000}, {"init_checkpoint": "/another-cpt"},
                         {"ui_num_tasks": 5}, {"ui_relation_initialization_seed": 43},
                         {"ui_task_registry": [{**r, "normalization_id": "other"} for r in self.registry]}):
            with self.subTest(override=override):
                write_json(config_path, {**original, **override})
                with self.assertRaises(RuntimeError):
                    validate_run_data_binding(self.runtime, self.snapshot, create=True)
                self.assertFalse(self.marker.exists())
        write_json(config_path, original)
        for name in ("trainer_state.json", "optimizer.pt", "global_step0", "rng_state.pth"):
            file = self.checkpoint / name
            file.write_text("state")
            with self.assertRaisesRegex(RuntimeError, "training state"):
                validate_run_data_binding(self.runtime, self.snapshot, create=True)
            file.unlink()
        (self.checkpoint / "ui5_checkpoint0_manifest.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "model-only export"):
            validate_run_data_binding(self.runtime, self.snapshot, create=True)

    def test_unbound_nonzero_checkpoint_even_with_valid_zero_is_rejected(self):
        self.export()
        (self.output / "checkpoint-1000").mkdir()
        with self.assertRaisesRegex(RuntimeError, "no repair data binding"):
            validate_run_data_binding(self.runtime, self.snapshot, create=True)
        self.assertFalse(self.marker.exists())

    def test_actual_initializer_overwrites_reason_and_reports_match(self):
        self.export()
        config = read_json(self.checkpoint / "config.json")
        report = read_json(self.checkpoint / "ui5_checkpoint0_manifest.json")["initialization"]
        self.assertEqual(config["ui_relation_initialization_reason"],
                         "all-ui-relation-keys-missing-checkpoint-0-export")
        self.assertEqual(report, config["ui_relation_initialization_stats"])
        self.assertEqual(report["seed"], 42)
        validate_run_data_binding(self.runtime, self.snapshot)
        self.assertFalse(self.marker.exists())

    def test_report_conflicts_and_training_initialization_reason_are_rejected(self):
        self.export()
        config_path = self.checkpoint / "config.json"
        export_path = self.checkpoint / "ui5_checkpoint0_manifest.json"
        config, exported = read_json(config_path), read_json(export_path)
        for report in (None, {}, "not a report", {**exported["initialization"], "seed": 43},
                       {**exported["initialization"], "reason": "checkpoint-0-export"}):
            with self.subTest(report=report):
                write_json(export_path, {**exported, "initialization": report})
                with self.assertRaisesRegex(RuntimeError, "initialization"):
                    validate_run_data_binding(self.runtime, self.snapshot, create=True)
                self.assertFalse(self.marker.exists())
        write_json(export_path, exported)
        for field, value in (("ui_relation_initialization_reason", "all-ui-relation-keys-missing-from-base-checkpoint"),
                             ("ui_relation_initialization_stats", {**config["ui_relation_initialization_stats"], "seed": 43})):
            write_json(config_path, {**config, field: value})
            with self.assertRaises(RuntimeError):
                validate_run_data_binding(self.runtime, self.snapshot, create=True)
            self.assertFalse(self.marker.exists())

    def test_legacy_short_reason_remains_supported_without_rewriting_checkpoint(self):
        self.export()
        config_path = self.checkpoint / "config.json"
        export_path = self.checkpoint / "ui5_checkpoint0_manifest.json"
        config, exported = read_json(config_path), read_json(export_path)
        config["ui_relation_initialization_reason"] = "checkpoint-0-export"
        config.pop("ui_relation_initialization_stats")
        exported.pop("initialization")
        write_json(config_path, config)
        write_json(export_path, exported)
        before = {p: p.read_bytes() for p in (config_path, export_path)}
        validate_run_data_binding(self.runtime, self.snapshot, create=True)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_completed_fourteen_tasks_survive_startup_fix_without_excel_or_weights_write(self):
        self.evaluation()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in
                  [self.state, self.workbook, *self.checkpoint.iterdir()]}
        validate_run_data_binding(self.runtime, self.snapshot, create=True)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertTrue(self.complete())
            self.assertTrue(self.complete())
        self.assertIn("inference skipped", log.getvalue())
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
        receipt = read_json(self.output / "evaluation/ui14-step-0-startup-reuse.json")
        self.assertEqual(receipt["source_identity"]["git_commit"], self.old)
        self.assertEqual(receipt["current_identity"]["git_commit"], self.new)

    def test_partial_results_changed_runtime_or_test_set_reject_reuse(self):
        self.evaluation()
        state = read_json(self.state)
        for change in ({"status": "failed"}, {"tasks": {t.task_key: {} for t in UI_TASKS[:5]}}):
            write_json(self.state, {**state, **change})
            self.assertFalse(self.complete())
        write_json(self.state, state)
        with mock.patch.dict(os.environ, {"UI5_CONFIG_HASH": "different-generation-config"}):
            self.assertFalse(self.complete())
        write_json(self.manifest, {"eval_set_id": "changed-test-set"})
        self.assertFalse(self.complete())
        write_json(self.manifest, {"eval_set_id": "unchanged-test-set"})
        self.workbook.unlink()
        self.assertFalse(self.complete())

    def test_changed_weights_do_not_reuse_architecture_signature(self):
        self.evaluation()
        weight = self.checkpoint / "pytorch_model.bin"
        future = datetime.now().timestamp() + 10
        os.utime(weight, (future, future))
        self.assertFalse(self.complete())

    def test_inference_changes_dirty_checkout_or_missing_old_commit_reject_reuse(self):
        self.evaluation()
        file = self.root / "scripts/run_ui14_eval.py"
        file.write_text("# changed scorer\n")
        self.assertFalse(self.complete())
        self.git("add", "scripts")
        self.git("-c", "user.name=CPU Test", "-c", "user.email=cpu@example.invalid", "commit", "-qm", "scorer change")
        with mock.patch.dict(os.environ, {"GIT_COMMIT": self.git("rev-parse", "HEAD")}):
            self.assertFalse(self.complete())
        with self.assertRaises((ValueError, subprocess.SubprocessError)):
            startup_only_changes(self.root, "0" * 40, self.new)

    def test_pipeline_allows_only_exact_binding_fix_not_generation_argument_changes(self):
        self.evaluation()
        pipeline = self.root / "shell/run_locany_ui5_pipeline.sh"
        pipeline.write_text(PIPELINE_BINDING_BLOCK + 'scripts/ui14_run_recovery.py" --output-dir\n')
        self.assertTrue(self.complete())
        pipeline.write_text(pipeline.read_text() + "generation_args+=(--temperature 0.9)\n")
        self.assertFalse(self.complete())


if __name__ == "__main__":
    unittest.main()
