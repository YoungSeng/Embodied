"""Run the actual worker startup with an old competing package, without CUDA."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inference_ui_defect_locany.py"


class InferenceImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "old-editable-checkout"
        (self.old / "eaglevl").mkdir(parents=True)
        (self.old / "eaglevl/__init__.py").write_text("# old package without the UI14 registry\n", encoding="utf-8")
        # Stop at the first heavy dependency, after checking actual import
        # resolution. No torch/model/transformers/GPU is loaded by these tests.
        (self.old / "torch.py").write_text(textwrap.dedent('''
            import json, os, sys
            import eaglevl
            import eaglevl.ui_task_registry as registry
            print('CPU_IMPORT_PROBE=' + json.dumps({
                'package': eaglevl.__file__, 'registry': registry.__file__,
                'tasks': [t.task_id for t in registry.UI_TASKS],
                'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'bootstrapped': os.environ.get('_LOCANY_CUDA_BOOTSTRAPPED'),
                'transformers_loaded': 'transformers' in sys.modules,
            }), flush=True)
            raise SystemExit(0)
        '''), encoding="utf-8")
        self.env = {**os.environ, "PYTHONPATH": str(self.old), "PYTHONNOUSERSITE": "1"}
        self.env.pop("_LOCANY_CUDA_BOOTSTRAPPED", None)

    def run_worker(self, script=SCRIPT, *arguments):
        return subprocess.run([sys.executable, str(script), *arguments], cwd=self.root,
                              env=self.env, text=True, capture_output=True, timeout=30)

    def probe(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("CPU_IMPORT_PROBE="))
        record = json.loads(line.split("=", 1)[1])
        self.assertEqual(Path(record["package"]).resolve(), ROOT / "eaglevl/__init__.py")
        self.assertEqual(Path(record["registry"]).resolve(), ROOT / "eaglevl/ui_task_registry.py")
        self.assertEqual(record["tasks"], list(range(14)))
        self.assertFalse(record["transformers_loaded"])
        self.assertIn("[inference imports] checkout=", result.stdout)
        return record

    def test_direct_worker_ignores_old_editable_package_before_torch(self):
        self.probe(self.run_worker())

    def test_cuda_reexec_preserves_checkout_priority(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        if os.name == "nt":
            # The production execvpe path is Linux-only. Verify its exact argv
            # and environment, then launch the replacement worker explicitly
            # on Windows, whose execvpe does not implement POSIX exec.
            tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
            nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                     and any(alias.name in {"argparse", "os", "sys"} for alias in node.names)]
            nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_bootstrap_cuda_visible_devices"]
            namespace = {}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
            with mock.patch.dict(os.environ, self.env, clear=True), \
                    mock.patch.object(sys, "argv", [str(SCRIPT), "--cuda-visible-devices", "3"]), \
                    mock.patch.object(os, "execvpe", side_effect=SystemExit) as reexec:
                with self.assertRaises(SystemExit):
                    namespace["_bootstrap_cuda_visible_devices"]()
            executable, argv, self.env = reexec.call_args.args
            self.assertEqual(executable, sys.executable)
            self.assertEqual(argv, [sys.executable, str(SCRIPT), "--cuda-visible-devices", "3"])
        record = self.probe(self.run_worker(SCRIPT, "--cuda-visible-devices", "3"))
        self.assertEqual(record["cuda"], "3")
        self.assertEqual(record["bootstrapped"], "1")

    def test_preloaded_wrong_package_fails_before_heavy_imports(self):
        (self.old / "sitecustomize.py").write_text("import eaglevl\n", encoding="utf-8")
        result = self.run_worker()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("eaglevl was already imported from", result.stderr)
        self.assertIn("Start a fresh inference process", result.stderr)
        self.assertNotIn("CPU_IMPORT_PROBE=", result.stdout)
        self.assertNotIn('torch.py",', result.stderr)

    def copy_worker(self, *, bootstrap=True, registry=True):
        checkout = self.root / "independent-checkout"
        (checkout / "scripts").mkdir(parents=True)
        (checkout / "eaglevl").mkdir()
        source = SCRIPT.read_text(encoding="utf-8")
        if not bootstrap:
            source = source.replace("\n_bootstrap_project_imports()\n", "\n# bootstrap disabled to reproduce the old startup\n")
        script = checkout / "scripts/inference_ui_defect_locany.py"
        script.write_text(source, encoding="utf-8")
        (checkout / "eaglevl/__init__.py").write_bytes((ROOT / "eaglevl/__init__.py").read_bytes())
        if registry:
            (checkout / "eaglevl/ui_task_registry.py").write_bytes((ROOT / "eaglevl/ui_task_registry.py").read_bytes())
        return script

    def test_without_early_bootstrap_reproduces_reported_missing_registry(self):
        result = self.run_worker(self.copy_worker(bootstrap=False))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No module named 'eaglevl.ui_task_registry'", result.stderr)

    def test_incomplete_checkout_reports_exact_missing_file_before_torch(self):
        result = self.run_worker(self.copy_worker(registry=False))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Incomplete inference checkout: missing", result.stderr)
        self.assertIn("ui_task_registry.py", result.stderr)
        self.assertNotIn("CPU_IMPORT_PROBE=", result.stdout)
        self.assertNotIn('torch.py",', result.stderr)


if __name__ == "__main__":
    unittest.main()
