from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class LocateAnythingCPTMerlinTest(unittest.TestCase):
    PROFILES = {
        "locany_cpt_v4_a100x4_smoke_merlin.yaml": (
            "shell/run_locany_cpt_merlin.sh a100 smoke",
            "gpuv: A800_SXM_40GB",
            "clusterId: 24",
            "gpu: 4",
            'CUDA_DEVICES: "0,1,2,3"',
        ),
        "locany_cpt_v4_a100x4_formal_merlin.yaml": (
            "shell/run_locany_cpt_merlin.sh a100 formal",
            "gpuv: A800_SXM_40GB",
            "clusterId: 24",
            "gpu: 4",
            'CUDA_DEVICES: "0,1,2,3"',
        ),
        "locany_cpt_v4_h20x4_formal_merlin.yaml": (
            "shell/run_locany_cpt_merlin.sh h20 formal",
            "gpuv: NVIDIA_H20",
            "clusterId: 20",
            "gpu: 4",
            'CUDA_DEVICES: "0,1,2,3"',
        ),
        "locany_cpt_v4_h20x2_formal_merlin.yaml": (
            "shell/run_locany_cpt_merlin.sh h20 formal",
            "gpuv: NVIDIA_H20",
            "clusterId: 20",
            "gpu: 2",
            'CUDA_DEVICES: "0,1"',
        ),
        "locany_cpt_v4_h20x2_smoke_merlin.yaml": (
            "shell/run_locany_cpt_merlin.sh h20 smoke",
            "gpuv: NVIDIA_H20",
            "clusterId: 20",
            "gpu: 2",
            'CUDA_DEVICES: "0,1"',
        ),
    }

    def test_merlin_profiles_have_expected_resources_and_commands(self):
        for filename, expected in self.PROFILES.items():
            with self.subTest(filename=filename):
                text = (REPO_ROOT / filename).read_text(encoding="utf-8")
                for value in expected:
                    self.assertIn(value, text)
                self.assertIn("Embodied-CPT", text)
                self.assertIn('INSTALL_SYSTEM_RUNTIME_DEPS: "1"', text)

    def test_yaml_pins_smoke_and_formal_training_parameters(self):
        smoke = (REPO_ROOT / "locany_cpt_v4_a100x4_smoke_merlin.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn('MAX_STEPS: "20"', smoke)
        self.assertIn('CPT_METRICS_INTERVAL: "5"', smoke)
        self.assertIn('CPT_TABLE_INTERVAL: "20"', smoke)
        self.assertIn('CPT_SMOKE_RESUME_STEP: "10"', smoke)
        self.assertIn('SAVE_STEPS: "20"', smoke)
        self.assertIn('MAX_NUM_TOKENS: "12800"', smoke)

        a100 = (REPO_ROOT / "locany_cpt_v4_a100x4_formal_merlin.yaml").read_text(
            encoding="utf-8"
        )
        h20 = (REPO_ROOT / "locany_cpt_v4_h20x4_formal_merlin.yaml").read_text(
            encoding="utf-8"
        )
        for text in (a100, h20):
            self.assertIn('GRADIENT_ACCUMULATION_STEPS: "2"', text)
            self.assertIn('MAX_STEPS: "20000"', text)
            self.assertIn('LEARNING_RATE: "5e-6"', text)
            self.assertIn('SAVE_EVERY_N_HOURS: "12"', text)
            self.assertIn('SAVE_STEPS: "1000000000"', text)
        self.assertIn('MAX_NUM_TOKENS: "12800"', a100)
        self.assertIn('ATTN_IMPLEMENTATION: "sdpa"', a100)
        self.assertIn('MAX_SEQ_LENGTH: "7268"', h20)
        self.assertIn('MAX_NUM_TOKENS_PER_SAMPLE: "7268"', h20)
        self.assertIn('MAX_NUM_TOKENS: "7268"', h20)
        self.assertIn('PACKING_BUFFER_SIZE: "16"', h20)
        self.assertIn('ATTN_IMPLEMENTATION: "sdpa"', h20)

        h20x2 = (REPO_ROOT / "locany_cpt_v4_h20x2_formal_merlin.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("caption: 'LocateAnything UI CPT v2 Formal - H20x2 SDPA'", h20x2)
        self.assertIn("name: 'locany-cpt-v4-v2-h20x2-formal'", h20x2)
        self.assertIn("cpu: 40", h20x2)
        self.assertIn("memory: 460800", h20x2)
        self.assertIn('GPU_COUNT: "2"', h20x2)
        self.assertIn('GRADIENT_ACCUMULATION_STEPS: "4"', h20x2)
        self.assertIn('ATTN_IMPLEMENTATION: "sdpa"', h20x2)
        self.assertIn('MAX_SEQ_LENGTH: "7268"', h20x2)
        self.assertIn('MAX_NUM_TOKENS_PER_SAMPLE: "7268"', h20x2)
        self.assertIn('MAX_NUM_TOKENS: "7268"', h20x2)
        self.assertIn('PACKING_BUFFER_SIZE: "16"', h20x2)

        h20x2_smoke = (
            REPO_ROOT / "locany_cpt_v4_h20x2_smoke_merlin.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn('CPT_METRICS_INTERVAL: "5"', h20x2_smoke)
        self.assertIn('CPT_TABLE_INTERVAL: "20"', h20x2_smoke)
        self.assertIn('CPT_SMOKE_RESUME_STEP: "10"', h20x2_smoke)
        self.assertIn('RUN_NAME: "locany-3b-ui-cpt-v4-v2-h20x2-smoke"', h20x2_smoke)
        for text in (h20x2, h20x2_smoke):
            self.assertIn("locany_cpt_v4_split_v2", text)
            self.assertNotIn('ATTN_IMPLEMENTATION: "magi"', text)
            self.assertNotIn('MAX_NUM_TOKENS: "25600"', text)

    def test_formal_defaults_keep_four_card_rank_batch_and_twelve_hour_saves(self):
        launcher = (REPO_ROOT / "shell" / "run_locany_cpt.sh").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_GRADIENT_ACCUMULATION_STEPS=4', launcher)
        self.assertIn('DEFAULT_GRADIENT_ACCUMULATION_STEPS=2', launcher)
        self.assertIn(
            'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_GRADIENT_ACCUMULATION_STEPS}}"',
            launcher,
        )
        self.assertIn('SAVE_EVERY_N_HOURS="${SAVE_EVERY_N_HOURS:-12}"', launcher)
        self.assertIn('MAX_STEPS="${MAX_STEPS:-20000}"', launcher)
        self.assertIn('GPU_COUNT="${GPU_COUNT:-4}"', launcher)
        self.assertIn('export GPUS="${GPU_COUNT}" GPU_COUNT', launcher)

        merlin = (REPO_ROOT / "shell" / "run_locany_cpt_merlin.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('expected_gpu_count = int(os.environ.get("GPU_COUNT", "4"))', merlin)
        self.assertIn('x${GPU_COUNT}-formal', merlin)
        self.assertIn('run_training_phase "pre-resume-${SMOKE_RESUME_STEP}"', merlin)
        self.assertIn('export LOCANY_SEGMENT_MODE=1', merlin)
        self.assertIn(
            "SMOKE_PRE_RESUME=SKIPPED_EXISTING_RESUMABLE_CHECKPOINT", merlin
        )
        self.assertIn("scripts/locany_ui5_checkpoint.py", merlin)

    def test_disabled_ui_relation_skips_ui5_only_trainer_audits(self):
        launcher = (REPO_ROOT / "shell" / "run_locany_cpt.sh").read_text(
            encoding="utf-8"
        )
        trainer = (
            REPO_ROOT / "eaglevl" / "train" / "locany_finetune_magi_stream.py"
        ).read_text(encoding="utf-8")

        self.assertIn("export ENABLE_UI_RELATION=False", launcher)
        self.assertIn(
            'getattr(self.model, "enable_ui_relation", False)', trainer
        )
        self.assertIn(
            "if not self._ui5_enabled:\n            return optimizer", trainer
        )
        self.assertIn('if self._ui5_enabled and "grad_norm" in logs:', trainer)
        self.assertIn("and step % self._cpt_metrics_interval == 0", trainer)
        self.assertIn("self._write_cpt_metrics(step, logs)", trainer)
        self.assertIn("if self._ui5_enabled and step in {1, 20, 100}:", trainer)
        self.assertIn("@record\ndef main():", trainer)
        self.assertIn("publish_cpt_checkpoint_completion", trainer)
        self.assertIn("reconcile_cpt_checkpoint_completion", trainer)
        self.assertIn("dist.destroy_process_group()", trainer)
        self.assertIn("CPT resume dataloader state is missing", trainer)
        self.assertIn("'version': 6", trainer)
        self.assertIn("truncation=not self.cpt_enabled", trainer)
        self.assertIn('"cpt_eval_queue.jsonl"', trainer)

    def test_v2_data_and_eval_jobs_are_explicit(self):
        prepare = (REPO_ROOT / "shell" / "prepare_locany_cpt_v2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("locany_cpt_v4_split_v2_smoke", prepare)
        self.assertIn("locany_cpt_val_fast.json", prepare)
        self.assertIn("--minimum-records-per-dataset", prepare)
        self.assertIn("--allow-manifest-subset", prepare)

        eval_yaml = (REPO_ROOT / "locany_cpt_v4_h20x1_eval_merlin.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("gpu: 1", eval_yaml)
        self.assertIn("NVIDIA_H20", eval_yaml)
        self.assertIn("shell/run_locany_cpt_eval_merlin.sh h20", eval_yaml)
        self.assertIn('EVAL_ATTN_IMPLEMENTATION: "sdpa"', eval_yaml)
        self.assertIn(
            'EVAL_VISION_ATTN_IMPLEMENTATION: "flash_attention_2"', eval_yaml
        )
        self.assertIn('EVAL_SAMPLES_PER_TASK: "10"', eval_yaml)

        smoke_eval_yaml = (
            REPO_ROOT / "locany_cpt_v4_h20x1_smoke_eval_merlin.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn('RUN_NAME: "locany-3b-ui-cpt-v4-v2-h20x2-smoke"', smoke_eval_yaml)
        self.assertIn("locany_cpt_v4_split_v2_smoke", smoke_eval_yaml)
        self.assertIn('EVAL_MAX_PENDING: "2"', smoke_eval_yaml)
        self.assertIn(
            'EVAL_VISION_ATTN_IMPLEMENTATION: "flash_attention_2"',
            smoke_eval_yaml,
        )

        eval_launcher = (
            REPO_ROOT / "shell" / "run_locany_cpt_eval_merlin.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("scripts/run_locany_cpt_eval_queue.py", eval_launcher)
        self.assertIn("--require-zero-inference-errors", eval_launcher)
        self.assertIn("shell/ensure_locany_cpt_runtime.sh", eval_launcher)
        self.assertIn("EVAL_VISION_ATTN_IMPLEMENTATION:-flash_attention_2", eval_launcher)
        self.assertIn("expandable_segments:True", eval_launcher)

        queue_runner = (
            REPO_ROOT / "scripts" / "run_locany_cpt_eval_queue.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--fail-fast-inference-errors"', queue_runner)

    def test_cpt_runtime_preflight_installs_libgl_before_worker_imports(self):
        helper = (
            REPO_ROOT / "shell" / "ensure_locany_cpt_runtime.sh"
        ).read_text(encoding="utf-8")
        launcher = (REPO_ROOT / "shell" / "run_locany_cpt.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/preflight_locany_runtime.py", helper)
        self.assertIn("preflight_code != 42", helper)
        self.assertIn("apt-get install -y", helper)
        self.assertIn("libgl1 libglib2.0-0", helper)
        self.assertIn("sudo -n", helper)
        self.assertIn("shell/ensure_locany_cpt_runtime.sh", launcher)


if __name__ == "__main__":
    unittest.main()
