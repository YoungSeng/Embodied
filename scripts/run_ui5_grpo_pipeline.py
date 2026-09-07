#!/usr/bin/env python3
"""Release training GPUs -> five formal hybrid workers -> metrics -> full resume."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from eaglevl.train.ui5_grpo_core import GRPOConfig, digest, file_sha
from eaglevl.train.ui5_grpo_checkpoint import recover, validate_checkpoint
from eaglevl.train.ui5_grpo_runtime import verify_execution_code
from scripts.build_ui5_grpo_mixed_manifest import read_json, write_json
from scripts import run_ui5_curriculum_evaluation as evaluation
from scripts.ui5_grpo_artifacts import register_evaluation, refresh_workbook


def verify_evaluation_inputs(run):
    source = run["evaluation_inputs"]
    if file_sha(source["path"]) != source["sha256"]:
        raise ValueError("fixed formal evaluation input inventory changed")
    for path, expected in read_json(source["path"]).items():
        if file_sha(path) != expected:
            raise ValueError(f"formal evaluation data/cache differs from step 0: {path}")


def evaluate(run, step, candidate):
    verify_evaluation_inputs(run)
    output = Path(run["output_dir"]) / "evaluation" / f"step-{step:06d}"
    e = run["evaluation"]
    paths = {task: evaluation._input_image_paths(Path(e["input_dir"]) / name, 0)
             for task, name in evaluation.TASK_GT_FILE.items()}
    counts = {task: len(values) for task, values in paths.items()}
    stems = {task: evaluation._expected_output_stems(values) for task, values in paths.items()}
    identity = dict(run_identity=run["identity"], step=step, candidate=str(candidate),
                    candidate_weights={p.name: file_sha(p) for p in sorted(Path(candidate).glob("*.safetensors"))},
                    checkpoint_manifest=file_sha(Path(candidate) / "grpo_complete.json") if step else run["reference"]["identity"],
                    config=e, inputs={name: file_sha(Path(e["input_dir"]) / name) for name in evaluation.TASK_GT_FILE.values()},
                    detector_manifest=file_sha(e["detector_crop_manifest"]),
                    scorer=file_sha(ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py"),
                    parser=file_sha(ROOT / "scripts/inference_ui_defect_locany.py"),
                    matcher=file_sha(ROOT / "scripts/ui5_metric_matching.py"))
    status_path = output / "evaluation_status.json"
    if status_path.is_file():
        status = read_json(status_path)
        if status.get("success") and status["identity"] == digest(identity):
            metrics = read_json(output / "ui5_metrics.json")
            if file_sha(output / "ui5_metrics.json") != status["metrics_sha256"]:
                raise ValueError("durable UI5 metrics changed")
            return register_evaluation(run, step, candidate, metrics, status["evaluation_seconds"])
    output.mkdir(parents=True, exist_ok=True)
    evaluation.clean_owned_outputs(output)
    identity_path = output / "evaluation_manifest.json"
    write_json(identity_path, identity)
    args = SimpleNamespace(**dict(e, output_dir=output, checkpoint=Path(candidate),
            processor_path=Path(run["processor_path"]), input_dir=Path(e["input_dir"]),
            python=run["python"], worker_script=ROOT / "scripts/inference_ui_defect_locany.py",
            scorer_script=ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py", project_root=ROOT,
            evaluation_purpose="grpo_full", anchor_groups_jsonl=None,
            rollout_bundle_root=Path(run["bundle"]), max_images_per_task=0,
            score_run_name="ui5_score", overwrite=True))
    args.detector_crop_manifest = Path(e["detector_crop_manifest"])
    empty = {task: 0 for task in evaluation.TASKS}
    specs = evaluation.build_worker_specs(args, ("0", "1"), empty, Path(run["mixed_dir"]) / "groups.jsonl", identity_path)
    started = time.monotonic()
    workers = evaluation.launch_workers(specs, project_root=ROOT, heartbeat_seconds=30)
    evaluation.validate_worker_results(specs, workers, expected_images=counts, expected_stems=stems,
        expected_hard=empty, expected_hard_ids={task: set() for task in evaluation.TASKS},
        rollout_seeds=evaluation.FORMAL_ROLLOUT_SEEDS, identity_digest=file_sha(identity_path),
        fake_worker=False, require_mining=False)
    _, metrics, command = evaluation.run_scorer(args, expected_images=counts, expected_stems=stems)
    verify_evaluation_inputs(run)
    seconds = time.monotonic() - started
    write_json(status_path, dict(success=True, status="completed", step=step, identity=digest(identity),
                                metrics_sha256=file_sha(output / "ui5_metrics.json"),
                                evaluation_seconds=seconds, worker_results=workers, scorer_command=command))
    return register_evaluation(run, step, candidate, metrics, seconds)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    args = parser.parse_args()
    run = read_json(args.run_config)
    payload = dict(run)
    if payload.pop("identity") != digest(payload):
        raise ValueError("formal run config was modified")
    config = GRPOConfig(**run["grpo"]).validate()
    if os.environ.get("GRPO_CONFIG_JSON") and json.loads(os.environ["GRPO_CONFIG_JSON"]) != run["grpo"]:
        raise ValueError("rendered YAML GRPO settings differ from immutable run.json")
    sha = verify_execution_code(run)
    if os.environ.get("CODE_REVISION") and os.environ["CODE_REVISION"] != sha:
        raise ValueError("formal checkout SHA differs from the submitted YAML")
    output = Path(run["output_dir"])
    with (output / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = recover(output / "resume", run["identity"])
        # Remove only uncommitted future telemetry after a crash. Training and
        # sampling restart exactly at the durable optimizer checkpoint.
        for directory, pattern, number in ((output / "metrics", "step*.json", lambda p: int(p.stem[4:])),
                (output / "diagnostics", "sampling_step*.json", lambda p: int(p.stem[13:])),
                (output / "diagnostics", "train_ar_step*.json", lambda p: int(p.stem[13:]))):
            for path in directory.glob(pattern):
                if number(path) > current:
                    path.unlink()
        state = read_json(output / "checkpoints.json") if (output / "checkpoints.json").exists() else {"evaluations": []}
        recorded = {row["step"] for row in state["evaluations"]}
        if recorded and max(recorded) > current:
            raise ValueError("formal evaluation advanced beyond durable resume")
        # Revalidate the durable step-0 identity/metrics on resume; a matching
        # result is registered idempotently without launching any GPU worker.
        evaluate(run, 0, Path(run["initial_model"]))
        if current and current % config.eval_interval == 0:
            evaluate(run, current, output / "resume/latest")
        while current < config.total_steps:
            target = min(config.total_steps, (current // config.eval_interval + 1) * config.eval_interval)
            print(f"[GRPO TRAIN] resume_step={current} until={target}", flush=True)
            command = [run["python"], "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       "--nproc_per_node=2", "--module", "eaglevl.train.ui5_grpo_trainer",
                       "--run-config", str(args.run_config.resolve()), "--until-step", str(target)]
            child_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", TOKENIZERS_PARALLELISM="false",
                             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            subprocess.run(command, cwd=ROOT, env=child_env, check=True)
            # subprocess completion is the GPU release barrier for both ranks.
            marker = validate_checkpoint(output / "resume/latest", run["identity"])
            if marker["step"] != target:
                raise ValueError("training segment did not finish its intended optimizer step")
            evaluate(run, target, output / "resume/latest")
            current = target
        refresh_workbook(run)
        write_json(output / "completed.json", dict(step=current, run_identity=run["identity"],
                                                  workbook=str(output / "diagnostics/ui5_grpo_training_evaluation.xlsx")))


if __name__ == "__main__":
    main()
