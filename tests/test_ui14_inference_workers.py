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


def fixture(root, workers_per_gpu=2):
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
        # The first wave must coexist before any subprocess can finish.
        while True:
            arrivals = list(barrier.glob('*.json'))
            if len(arrivals) >= FIRST_WAVE_SIZE:
                try:
                    with (args.output_dir / 'first_wave.json').open('x', encoding='utf-8') as handle:
                        json.dump([json.loads(p.read_text(encoding='utf-8')) for p in arrivals], handle)
                except FileExistsError:
                    pass
                break
            if time.monotonic_ns() - started > 15_000_000_000:
                raise RuntimeError('all process slots did not become active')
            time.sleep(.01)
        info['finished'] = time.monotonic_ns()
        target = args.output_dir / task
        target.mkdir(parents=True, exist_ok=True)
        (target / 'sample.json').write_text(json.dumps(info), encoding='utf-8')
        args.summary_path.write_text(json.dumps(info), encoding='utf-8')
    ''').replace('FIRST_WAVE_SIZE', str(4 * workers_per_gpu)), encoding="utf-8")
    return ["parallel", "--checkpoint", str(root / "checkpoint"), "--processor-path", str(root / "checkpoint"),
            "--input-dir", str(root), "--output-dir", str(root / "pred"), "--gpu-devices", "0,1,2,3",
            "--workers-per-gpu", str(workers_per_gpu), "--attn-implementation", "sdpa", "--inference-script", str(worker),
            "--eval-manifest", str(root / "evaluation_manifest.json")]


class InferenceWorkersTests(unittest.TestCase):
    def test_four_gpus_run_eight_processes_and_drain_all_14_tasks_once(self):
        self._exercise_queue(2)

    def test_four_gpus_run_one_process_each_and_drain_all_14_tasks_once(self):
        self._exercise_queue(1)

    def _exercise_queue(self, workers_per_gpu):
        slots = 4 * workers_per_gpu
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with mock.patch.object(sys, "argv", fixture(root, workers_per_gpu)):
                self.assertEqual(parallel.main(), 0)
            status = read_json(root / "pred/parallel_inference_status.json")
            self.assertTrue(status["success"])
            self.assertEqual(status["workers_per_gpu"], workers_per_gpu)
            self.assertEqual(status["worker_count"], slots)
            self.assertEqual(set(status["tasks"]), {t.task_key for t in UI_TASKS})
            self.assertEqual(len({r["worker_id"] for r in status["tasks"].values()}), slots)
            rows = [read_json(root / "pred" / task.task_key / "sample.json") for task in UI_TASKS]
            self.assertEqual(len({r["pid"] for r in rows}), 14)
            # Use the shared barrier, not comparisons between subprocess clocks.
            first_wave = read_json(root / "pred/first_wave.json")
            self.assertEqual(len(first_wave), slots)
            self.assertEqual(len({r["pid"] for r in first_wave}), slots)
            self.assertEqual(Counter(r["gpu"] for r in first_wave), dict.fromkeys("0123", workers_per_gpu))
            self.assertEqual({(r["physical_gpu"], r["worker_slot"]) for r in status["tasks"].values()},
                             {(gpu, slot) for gpu in "0123" for slot in range(workers_per_gpu)})
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

    def test_loneword_exclusivity_keeps_other_gpus_parallel_then_restores_both_slots(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            argv = fixture(root) + ["--exclusive-gpu-tasks", "synth_loneword"]
            barrier = threading.Barrier(7, timeout=10)
            refilled = threading.Event()
            guard = threading.Lock()
            active = {gpu: set() for gpu in "0123"}
            arrivals = []
            first_wave = []
            exclusive_gpu = None
            refilled_tasks = []

            def child(command, **kwargs):
                nonlocal exclusive_gpu
                task = command[command.index("--tasks") + 1]
                gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
                with guard:
                    active[gpu].add(task)
                    arrivals.append((task, gpu))
                    wave_one = len(arrivals) <= 7
                    if task == "synth_loneword":
                        exclusive_gpu = gpu
                    if "synth_loneword" in active[gpu]:
                        self.assertEqual(active[gpu], {"synth_loneword"})
                    self.assertLessEqual(len(active[gpu]), 2)
                    if len(arrivals) == 7:
                        first_wave.extend(arrivals)
                    if not wave_one and gpu == exclusive_gpu:
                        refilled_tasks.append(task)
                        if len(active[gpu]) == 2:
                            refilled.set()
                if wave_one:
                    barrier.wait()
                if task != "synth_loneword":
                    # Keep all other cards busy while the exclusive task exits.
                    # Both slots of its card must then accept ordinary tasks.
                    self.assertTrue(refilled.wait(10), "exclusive GPU failed to resume two slots")
                path = root / "pred" / task / "sample.json"
                write_json(path, {"task": task, "gpu": gpu})
                with guard:
                    active[gpu].remove(task)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(sys, "argv", argv), mock.patch.object(parallel.subprocess, "run", side_effect=child) as run:
                self.assertEqual(parallel.main(), 0)
            status = read_json(root / "pred/parallel_inference_status.json")
            self.assertEqual(run.call_count, 14)
            self.assertTrue(status["success"])
            self.assertEqual(status["exclusive_gpu_tasks"], ["synth_loneword"])
            self.assertEqual(Counter(gpu for _, gpu in first_wave), {gpu: 1 if gpu == exclusive_gpu else 2 for gpu in "0123"})
            self.assertGreaterEqual(len(refilled_tasks), 2)
            for task, result in status["tasks"].items():
                self.assertEqual(result["exclusive_gpu"], task == "synth_loneword")
                self.assertEqual(result["gpu_slots_reserved"], 2 if task == "synth_loneword" else 1)

    def test_failed_exclusive_task_wakes_blocked_sibling_without_claiming_more(self):
        queue = parallel.GpuTaskQueue(["synth_loneword", "cropping"], ["synth_loneword"])
        self.assertEqual(queue.claim("0", 0), "synth_loneword")
        entered = threading.Event()
        result = []

        def sibling():
            entered.set()
            result.append(queue.claim("0", 1))

        thread = threading.Thread(target=sibling, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(2))
        queue.finish("0", 0, failed=True)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])
        self.assertEqual(queue.pending, ["cropping"])

    def test_launch_error_releases_exclusive_gpu_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp)
            argv = fixture(root) + ["--exclusive-gpu-tasks", "synth_loneword"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(parallel.subprocess, "run", side_effect=OSError("fixture launch failure")):
                self.assertEqual(parallel.main(), 1)
            status = read_json(root / "pred/parallel_inference_status.json")
            self.assertFalse(status["success"])
            self.assertEqual(status["tasks"]["synth_loneword"]["return_code"], 91)
            self.assertIn("fixture launch failure", status["tasks"]["synth_loneword"]["error"])

    def test_profile_and_yaml_bind_two_inference_workers(self):
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
