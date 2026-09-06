"""CPU queue tests; child processes exercise concurrency without loading a model or CUDA."""
import contextlib
from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ui14_common import UI_TASKS, SCAN_NAME, read_json, write_json, write_jsonl
import run_ui5_parallel_inference as parallel
import locany_ui5_common as common


def fixture(root):
    (root / "checkpoint").mkdir()
    specs = []
    for task in UI_TASKS:
        path = root / f"{task.task_key}.jsonl"
        write_jsonl(path, [{"image": f"{task.task_key}.png"}])
        specs.append({**task.to_dict(), "test": str(path), "expected_records": 1,
                      "skip_figma": task.task_id < 5, "cache": str(root / "cache" / task.task_key),
                      "scan_name": SCAN_NAME})
    write_json(root / "evaluation_manifest.json", {"tasks": specs})
    worker = root / "fake_inference.py"
    worker.write_text(textwrap.dedent('''
        import argparse, json, os, time
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument('--output-dir', type=Path)
        parser.add_argument('--summary-path', type=Path)
        parser.add_argument('--tasks', nargs='+')
        args, _ = parser.parse_known_args()
        task = args.tasks[0]
        barrier = args.output_dir / 'barrier'
        barrier.mkdir(exist_ok=True, parents=True)
        started = time.monotonic_ns()
        info = {'task': task, 'pid': os.getpid(), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'started': started}
        arrival = barrier / (task + '.json')
        temporary = barrier / (task + '.tmp')
        temporary.write_text(json.dumps(info), encoding='utf-8')
        os.replace(temporary, arrival)
        # First eight real subprocesses must coexist before any can finish.
        while True:
            arrivals = list(barrier.glob('*.json'))
            if len(arrivals) >= 8:
                try:
                    with (args.output_dir / 'first_wave.json').open('x', encoding='utf-8') as handle:
                        json.dump([json.loads(p.read_text(encoding='utf-8')) for p in arrivals], handle)
                except FileExistsError:
                    pass
                break
            if time.monotonic_ns() - started > 15_000_000_000:
                raise RuntimeError('eight process slots did not become active')
            time.sleep(.01)
        info['finished'] = time.monotonic_ns()
        target = args.output_dir / task
        target.mkdir(parents=True, exist_ok=True)
        (target / 'sample.json').write_text(json.dumps(info), encoding='utf-8')
        args.summary_path.write_text(json.dumps(info), encoding='utf-8')
    '''), encoding="utf-8")
    return ["parallel", "--checkpoint", str(root / "checkpoint"), "--processor-path", str(root / "checkpoint"),
            "--input-dir", str(root), "--output-dir", str(root / "pred"), "--gpu-devices", "0,1,2,3",
            "--workers-per-gpu", "2", "--attn-implementation", "sdpa", "--inference-script", str(worker),
            "--eval-manifest", str(root / "evaluation_manifest.json")]


class InferenceWorkersTests(unittest.TestCase):
    def test_four_gpus_run_eight_processes_and_drain_all_14_tasks_once(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with mock.patch.object(sys, "argv", fixture(root)):
                self.assertEqual(parallel.main(), 0)
            status = read_json(root / "pred/parallel_inference_status.json")
            self.assertTrue(status["success"])
            self.assertEqual(status["workers_per_gpu"], 2)
            self.assertEqual(status["worker_count"], 8)
            self.assertEqual(set(status["tasks"]), {t.task_key for t in UI_TASKS})
            self.assertEqual(len({r["worker_id"] for r in status["tasks"].values()}), 8)
            rows = [read_json(root / "pred" / task.task_key / "sample.json") for task in UI_TASKS]
            self.assertEqual(len({r["pid"] for r in rows}), 14)
            # Use the shared barrier, not comparisons between subprocess clocks.
            first_wave = read_json(root / "pred/first_wave.json")
            self.assertEqual(len(first_wave), 8)
            self.assertEqual(len({r["pid"] for r in first_wave}), 8)
            self.assertEqual(Counter(r["gpu"] for r in first_wave), dict.fromkeys("0123", 2))
            self.assertEqual({(r["physical_gpu"], r["worker_slot"]) for r in status["tasks"].values()},
                             {(gpu, slot) for gpu in "0123" for slot in (0, 1)})
            for task, result in status["tasks"].items():
                self.assertEqual(result["physical_gpu"], read_json(root / "pred" / task / "sample.json")["gpu"])
                self.assertEqual(result["logical_device"], "cuda:0")
            self.assertEqual(len(list((root / "pred/_worker_summaries").glob("*.json"))), 14)
            self.assertEqual(len(list((root / "pred/_worker_logs").glob("*.log"))), 14)

    def test_worker_failure_stops_further_queue_claims_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp)
            argv = fixture(root)
            barrier = threading.Barrier(8, timeout=5)

            def failed_child(*args, **kwargs):
                barrier.wait()
                return subprocess.CompletedProcess(args[0], 17)

            with mock.patch.object(sys, "argv", argv), mock.patch.object(parallel.subprocess, "run", side_effect=failed_child) as child:
                self.assertEqual(parallel.main(), 1)
            status = read_json(root / "pred/parallel_inference_status.json")
            self.assertFalse(status["success"])
            self.assertEqual(child.call_count, 8)
            self.assertEqual(len(status["missing_tasks"]), 6)
            self.assertTrue(all(r["return_code"] == 17 for r in status["tasks"].values()))

    def test_profile_and_yaml_bind_two_inference_workers_only(self):
        from ui14_checks import render_formal_yaml
        from ui14_profile import profile_environment
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            path, runtime_path = render_formal_yaml(Path(tmp))
            runtime = read_json(runtime_path)
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(runtime["EVAL_INFERENCE_WORKERS_PER_GPU"], 2)
            self.assertEqual(str(document["jobRunParams"]["envsList"]["EVAL_INFERENCE_WORKERS_PER_GPU"]), "2")
            self.assertEqual(runtime["GPU_COUNT"], 4)
            self.assertEqual(runtime["PER_DEVICE_TRAIN_BATCH_SIZE"], 1)
            self.assertEqual(runtime["EVAL_DETECTOR_WORKERS_PER_GPU"], 1)
            self.assertEqual(runtime["EVAL_FAIL_POLICY"], "stop")
            env = {**profile_environment(), "GPU_COUNT": "4", "RESOURCE_GROUP": "aiai_locate"}
            with self.assertRaisesRegex(ValueError, "EVAL_INFERENCE_WORKERS_PER_GPU"):
                common.resolve_runtime_config({**env, "EVAL_INFERENCE_WORKERS_PER_GPU": "1"})
        self.assertEqual(common.resolve_runtime_config({"MACHINE_TYPE": "a800", "GPU_COUNT": "4"})["EVAL_INFERENCE_WORKERS_PER_GPU"], 1)
        for value in ("0", "3"):
            with self.assertRaisesRegex(ValueError, "EVAL_INFERENCE_WORKERS_PER_GPU"):
                common.resolve_runtime_config({"EVAL_INFERENCE_WORKERS_PER_GPU": value})


if __name__ == "__main__":
    unittest.main()
