from __future__ import annotations

import threading
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CPTSegmentedPipelineTest(unittest.TestCase):
    def test_shared_priority_scheduler_retries_on_same_gpu(self) -> None:
        import sys

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from run_ui5_parallel_inference import run_priority_gpu_tasks

        lock = threading.Lock()
        attempts: dict[str, list[tuple[str, int]]] = {}

        def runner(task: str, gpu: str, attempt: int):
            with lock:
                attempts.setdefault(task, []).append((gpu, attempt))
            return {
                "return_code": 7 if task == "ui_defect" and attempt == 1 else 0,
                "task": task,
                "physical_gpu": gpu,
            }

        tasks = ["ui_defect", "ocr", "vqa", "referring"]
        result = run_priority_gpu_tasks(
            tasks=tasks,
            gpu_devices=["0", "1"],
            estimates={task: 1 for task in tasks},
            runner=runner,
            retries=1,
            continue_on_failure=True,
        )
        self.assertEqual(set(result), set(tasks))
        self.assertTrue(all(row["return_code"] == 0 for row in result.values()))
        first_gpu, _ = attempts["ui_defect"][0]
        retry_gpu, retry_number = attempts["ui_defect"][1]
        self.assertEqual(first_gpu, retry_gpu)
        self.assertEqual(retry_number, 2)

    def test_formal_yaml_selects_one_job_segmented_h20x2_profile(self) -> None:
        yaml = (
            REPO_ROOT / "locany_cpt_v4_h20x2_formal_segmented_eval_merlin.yaml"
        ).read_text(encoding="utf-8")
        for value in (
            'GPU_COUNT: "2"',
            'EVAL_GPU_DEVICES: "0,1"',
            'MAX_STEPS: "20000"',
            'EVAL_INTERVAL_STEPS: "1000"',
            'EVAL_SAMPLES_PER_TASK: "200"',
            'CPT_SEGMENTED_PIPELINE: "1"',
            'CPT_INTEGRATED_EVAL: "0"',
            'CPT_EXTERNAL_UI5_EVAL: "1"',
            'GRADIENT_ACCUMULATION_STEPS: "4"',
            'MAX_SEQ_LENGTH: "7268"',
            'ENABLE_UI_RELATION: "False"',
            'EVAL_FAIL_POLICY: "warn"',
        ):
            self.assertIn(value, yaml)
        self.assertIn("gpu: 2", yaml)
        self.assertNotIn("run_locany_cpt_eval_merlin.sh", yaml)

    def test_pipeline_uses_natural_targets_then_validates_before_eval(self) -> None:
        pipeline = (
            REPO_ROOT / "shell/run_locany_cpt_segmented_pipeline.sh"
        ).read_text(encoding="utf-8")
        trainer = (
            REPO_ROOT / "eaglevl/train/locany_finetune_magi_stream.py"
        ).read_text(encoding="utf-8")
        self.assertIn('export MAX_STEPS="${target}"', pipeline)
        self.assertIn("TORCHRUN_ALL_RANKS_EXITED", pipeline)
        self.assertLess(
            pipeline.index('checkpoint_status "${target}"'),
            pipeline.index('run_eval "${CURRENT_STEP}"'),
        )
        self.assertIn("unset LOCANY_STOP_AFTER_STEP LOCANY_STOP_AFTER_PERIODIC_SAVE", pipeline)
        self.assertIn("LOCANY_LR_SCHEDULER_TOTAL_STEPS", pipeline)
        self.assertIn("LOCANY_PIPELINE_FINAL_STEP", trainer)
        self.assertIn("def create_scheduler", trainer)

    def test_heldout_workers_are_fragmented_and_base_cache_is_strict_after_step0(self) -> None:
        parallel = (
            REPO_ROOT / "scripts/run_locany_cpt_parallel_eval.py"
        ).read_text(encoding="utf-8")
        evaluator = (
            REPO_ROOT / "scripts/eval_locany_cpt_learning.py"
        ).read_text(encoding="utf-8")
        self.assertIn("run_priority_gpu_tasks", parallel)
        self.assertIn('command.append("--skip-base-if-cached")', parallel)
        self.assertIn('"--output-fragment"', parallel)
        self.assertIn('"--gpu-device"', parallel)
        self.assertIn("complete_ten_task_heldout", parallel)
        self.assertIn("--skip-base-if-cached", evaluator)
        self.assertIn("--output-fragment", evaluator)


if __name__ == "__main__":
    unittest.main()
