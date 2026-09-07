#!/usr/bin/env python3
"""Prepare all internal artifacts, render the inherited H20x2 YAML, submit once.

Defaults are the user's real submission/checkpoint/rollout paths. All source
data stays on the internal mount. --resume reuses the same run identity and
full rolling state after the previous platform job has stopped.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from eaglevl.train.ui5_grpo_core import GRPOConfig, digest, file_sha
from eaglevl.train.ui5_grpo_runtime import bind_execution_code
from scripts.build_ui5_grpo_mixed_manifest import build, read_json, write_json
from scripts.ui5_grpo_bootstrap import PREVIOUS, recorded_source

WORKSPACE = Path("/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace")
ORIGINAL = WORKSPACE / "gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000"
ROLLOUT = WORKSPACE / "gui_rollouts/ui5-train-rollout8-h20x2-v6-20260904"
RUN_NAME = "ui5-crop-grpo-mixed-v1-h20x2-20260907"
CHECKOUT = WORKSPACE / "code/Embodied-ui5-crop-grpo-mixed-v1"
BASELINE = "b29590f88c4b4a102c742f8410e1c6751b0859d2"
SUBMISSION_CLUSTERS = {
    "default": None,  # Preserve the actual bound v3 resource group and queue.
    "ies_aiai_experience": dict(
        group_id=1602,
        queue_name="compute-329-hl-cloudnative-ai-ies.aiai.experience-guarantee",
    ),
}


def inherited_source(previous):
    import yaml
    state_path = recorded_source(previous)
    state = read_json(state_path)
    job_path = Path(state["job_yaml"]).resolve(strict=True)
    if job_path.parent != state_path.parent:
        raise ValueError("inherited YAML is outside the recorded submission")
    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    env = dict(job["jobRunParams"]["envsList"])
    if any(env.get(key) != value for key, value in state["runtime"].items()):
        raise ValueError("inherited YAML/runtime differs from the recorded job")
    return state_path, state, job, env


def pin_evaluation_inputs(config, path):
    from scripts.run_ui5_curriculum_evaluation import TASK_GT_FILE, _input_image_paths
    paths = {Path(config["detector_crop_manifest"]).resolve(strict=True)}
    for name in TASK_GT_FILE.values():
        source = Path(config["input_dir"]) / name
        paths.add(source.resolve(strict=True))
        paths.update(Path(value).resolve(strict=True) for value in _input_image_paths(source, 0))
    inventory = {str(item): file_sha(item) for item in sorted(paths)}
    write_json(path, inventory)
    return dict(path=str(path), sha256=file_sha(path))


def evaluation_config(env):
    def value(name, default, convert=str):
        return convert(env.get(name, default))
    return dict(input_dir=env["EVAL_INPUT_DIR"], detector_crop_manifest=env["EVAL_DETECTOR_MANIFEST"],
                dtype="bf16", attn_implementation="sdpa", vision_attn_implementation="flash_attention_2",
                generation_mode="hybrid", decoder_policy="boundary_v3", inference_crop_mode="detector_scan",
                max_new_tokens=value("EVAL_MAX_NEW_TOKENS", 4096, int),
                n_future_tokens=value("EVAL_N_FUTURE_TOKENS", 6, int),
                temperature=value("EVAL_TEMPERATURE", 0.7, float), top_p=value("EVAL_TOP_P", 0.9, float),
                top_k=value("EVAL_TOP_K", 0, int), repetition_penalty=value("EVAL_REPETITION_PENALTY", 1.1, float),
                greedy=False, relation_gate_mode=value("RELATION_GATE_MODE", "observe"), relation_gate_threshold=None,
                seed=value("SEED", 42, int), tile_max_count=value("EVAL_TILE_MAX_COUNT", 10, int),
                tile_target_long_side=value("EVAL_TILE_TARGET_LONG_SIDE", 1600, int),
                tile_overlap_ratio=value("EVAL_TILE_OVERLAP_RATIO", 0.1, float),
                tile_nms_iou=value("EVAL_TILE_NMS_IOU", 0.5, float),
                evaluator_iou_threshold=value("EVAL_IOU_THRESHOLD", 0.1, float))


def submission_target(job):
    arnold = job["jobDefVersion"]["resource"]["arnoldConfig"]
    return dict(profile=job["jobRunParams"]["envsList"]["UI5_SUBMISSION_CLUSTER"],
                cluster_id=arnold["clusterId"], group_ids=list(arnold["groupIds"]),
                queue_name=arnold["roles"][0].get("queueName", ""))


def render_job(old_job, old_env, run, run_config, *, execution_sha=None, cluster="default"):
    import yaml
    if cluster not in SUBMISSION_CLUSTERS:
        raise ValueError(f"unknown H20 submission cluster: {cluster}")
    job = yaml.safe_load((ROOT / "jobs/ui5_crop_grpo_mixed_v1_h20x2.yaml").read_text(encoding="utf-8"))
    for key in ("resource", "imageMeta", "volumes"):
        job["jobDefVersion"][key] = copy.deepcopy(old_job["jobDefVersion"][key])
    job["namespace"] = old_job["namespace"]
    arnold = job["jobDefVersion"]["resource"]["arnoldConfig"]
    roles = arnold["roles"]
    if len(roles) != 1 or (roles[0]["num"], roles[0]["gpu"], roles[0]["gpuv"]) != (1, 2, "NVIDIA_H20"):
        raise ValueError("recorded resource is not H20x2")
    override = SUBMISSION_CLUSTERS[cluster]
    if override is not None:
        arnold["groupIds"] = [override["group_id"]]
        roles[0]["queueName"] = override["queue_name"]
    job["caption"] = "UI5 Crop Mixed GRPO H20x2 " + run["run_name"][-8:]
    job["jobDefVersion"]["name"] = run["run_name"]
    job["jobDefVersion"]["gitRepo"] = dict(mnt=run["project_root"])
    env = {str(k): str(v) for k, v in old_env.items()}
    for key in ("RESUME_FROM_CHECKPOINT", "LOCANY_STOP_AFTER_STEP", "CURRICULUM_START_STEP", "CURRICULUM_MODE",
                "HARD_RATIOS", "ANCHOR_RATIOS", "GLOBAL_REPLAY_RATIOS", "LLM_LRS", "UI5_CURRICULUM_PROFILE"):
        env.pop(key, None)
    env.update(PROJECT_ROOT=run["project_root"], CODE_REVISION=execution_sha or run["code_sha"],
               UI5_SUBMISSION_CLUSTER=cluster,
               RUN_NAME=run["run_name"], OUTPUT_DIR=run["output_dir"], GRPO_RUN_CONFIG=str(run_config),
               PYTHON_BIN=run["python"], MODEL_PATH=run["initial_model"],
               GRPO_CONFIG_JSON=json.dumps(run["grpo"], sort_keys=True),
               GRADIENT_ACCUMULATION_STEPS="2", TOTAL_STEPS="1200", WARMUP_STEPS="40",
               MAX_SEQ_LENGTH="7268", MAX_NUM_TOKENS_PER_SAMPLE="7268", MAX_NUM_TOKENS="7268",
               ATTN_IMPLEMENTATION="sdpa", UI5_GPU0_WORKERS="2", UI5_GPU1_WORKERS="3",
               UI5_GRPO_PROFILE="mixed_v1", EVAL_INTERVAL_STEPS="200")
    job["jobRunParams"]["envsList"] = env
    job["jobRunParams"]["entrypointFullScript"] = (
        'set -Eeuo pipefail\ncd "${PROJECT_ROOT}"\n'
        'test "$(git rev-parse HEAD)" = "${CODE_REVISION}"\n'
        'exec bash shell/run_ui5_crop_grpo_mixed_h20x2.sh\n')
    return job


def prepare(args):
    import yaml
    from scripts.ui5_curriculum_v3 import model_view
    from scripts.ui5_curriculum_text_revision import read_publication
    from scripts.patch_locany_checkpoint import patch_checkpoint
    code_update = getattr(args, "resume_code_update", False)
    if code_update:
        args.resume = True
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,70}", args.run_name):
        raise ValueError("RUN_NAME must be a unique safe lowercase name")
    if ROOT.resolve() != CHECKOUT.resolve():
        raise ValueError(f"formal preparation requires the independent checkout {CHECKOUT}")
    source_path, source, old_job, env = inherited_source(args.previous_submission_dir)
    for key in ("ENV_DIR", "PROCESSOR_PATH", "ROLLOUT_BUNDLE_ROOT", "FROZEN_SELECTION", "CURRICULUM_DATA_DIR", "EVAL_INPUT_DIR"):
        if not env.get(key):
            raise ValueError(f"actual inherited runtime lacks {key}; no guessed input")
    env.setdefault("PYTHON_BIN", str(Path(env["ENV_DIR"]) / "bin/python"))
    if "EVAL_DETECTOR_MANIFEST" not in env:
        env["EVAL_DETECTOR_MANIFEST"] = str(Path(env["EVAL_DETECTOR_CACHE"]) /
            env.get("EVAL_SCAN_NAME", "horizontal_scan_v5_raw_detector_edge_aligned") / "detector_scan_crops.jsonl")
    # Source-bound paths/conda/processor remain local to the internal host.
    if Path(sys.executable).resolve() != Path(env["PYTHON_BIN"]).resolve():
        os.execv(env["PYTHON_BIN"], [env["PYTHON_BIN"], str(Path(__file__).resolve()), *sys.argv[1:]])
    if Path(env["PROJECT_ROOT"]).resolve() == ROOT.resolve():
        raise ValueError("GRPO must not train in the prior v3 checkout")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, sha], cwd=ROOT, check=True)
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("formal checkout has uncommitted source modifications")
    config = GRPOConfig().validate()
    output = WORKSPACE / "gui_models/Embodied-ui5-crop-grpo-mixed" / args.run_name
    submission = WORKSPACE / "gui_logs/ui5_grpo" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    submission.mkdir(parents=True, exist_ok=True)
    run_config = output / "run.json"
    if run_config.exists():
        run = read_json(run_config)
        if run["source_submission"] != str(source_path):
            raise ValueError("existing RUN_NAME belongs to another source identity")
        # Match the pipeline's process lock before publishing a runtime update.
        import fcntl
        with (output / ".pipeline.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            bind_execution_code(run, allow_update=code_update)
        if args.submit and not args.resume and (submission / "submission-attempt.started").exists():
            raise ValueError("submission already attempted; inspect its receipt, or --resume after the prior job stops")
    else:
        if args.resume:
            raise ValueError("cannot resume a run that has not been prepared")
        curriculum_dir = Path(env["CURRICULUM_DATA_DIR"])
        read_publication(curriculum_dir)
        mixed = build(snapshot=Path(source["snapshot"]), frozen_selection=Path(env["FROZEN_SELECTION"]),
                      rollout_root=ROLLOUT, bundle=Path(env["ROLLOUT_BUNDLE_ROOT"]), curriculum_dir=curriculum_dir,
                      eval_input=Path(env["EVAL_INPUT_DIR"]), output=output / "mixed", config=config)
        initial = output / "initial_model"
        if not initial.exists():
            model_view(ORIGINAL, initial)
        patch_checkpoint(base_model=Path(env["PROCESSOR_PATH"]), checkpoint=initial,
                         project_root=ROOT, force=True, validate_relation_weights=True)
        reference_files = {p.name: file_sha(p) for p in sorted(initial.iterdir())
                           if p.is_file() and (p.suffix == ".safetensors" or p.name == "config.json"
                                              or p.name.startswith("pytorch_model") and p.suffix == ".bin")}
        scales = {"llm": 1.0, "mlp": 1.0}
        if env.get("LR_SCALE"):
            scales.update({k.strip(): float(v) for k, v in (pair.split(":") for pair in env["LR_SCALE"].split(","))})
        if any(value <= 0 for value in scales.values()):
            raise ValueError("inherited LR multipliers must be positive")
        scales = {key: value / scales["llm"] for key, value in scales.items()}
        run = dict(schema_version=1, run_name=args.run_name, code_sha=sha, baseline_sha=BASELINE,
                   project_root=str(ROOT), output_dir=str(output), python=env["PYTHON_BIN"],
                   processor_path=env["PROCESSOR_PATH"], processor_in_token_limit=int(env.get("PROCESSOR_IN_TOKEN_LIMIT", 25600)),
                   initial_model=str(initial), original_checkpoint=str(ORIGINAL),
                   reference=dict(path=str(initial), files=reference_files, identity=digest(reference_files)),
                   mixed_dir=str(output / "mixed"), mixed_identity=mixed["identity"],
                   bundle=env["ROLLOUT_BUNDLE_ROOT"], source_submission=str(source_path),
                   inherited_yaml_sha256=file_sha(source["job_yaml"]),
                   inherited_runtime=env, grpo=config.to_dict(), evaluation=evaluation_config(env),
                   optimizer=dict(lr_scale=scales, weight_decay=float(env.get("WEIGHT_DECAY", 0.01)),
                                  betas=[float(env.get("ADAM_BETA1", 0.9)), float(env.get("ADAM_BETA2", 0.999))],
                                  eps=float(env.get("ADAM_EPSILON", 1e-8)),
                                  replay_negative_to_positive_ratio=float(env.get("UI_NEGATIVE_TO_POSITIVE_RATIO", 2.0))))
        if (run["evaluation"]["tile_nms_iou"], run["evaluation"]["evaluator_iou_threshold"]) != (0.5, 0.1):
            raise ValueError("actual inherited evaluator differs from required NMS=0.5/IoU=0.1")
        run["evaluation_inputs"] = pin_evaluation_inputs(run["evaluation"], output / "evaluation_inputs.json")
        run["identity"] = digest(run)
        write_json(run_config, run)
    destination = submission
    # Never overwrite a platform receipt when merely displaying prepared paths.
    if not args.resume and (submission / "submission-attempt.started").exists():
        print(json.dumps(read_json(output / "delivery_paths.json"), indent=2), flush=True)
        return submission / "formal.yaml"
    if args.resume:
        from datetime import datetime, timezone
        destination = submission / ("resume-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        destination.mkdir(exist_ok=False)
    job = render_job(old_job, env, run, run_config, execution_sha=sha,
                     cluster=getattr(args, "cluster", "default"))
    target = submission_target(job)
    job_path = destination / "formal.yaml"
    job_path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    state_path = destination / "submission.json"
    state = dict(status="prepared", runtime=job["jobRunParams"]["envsList"], job_yaml=str(job_path),
                 run_identity=run["identity"], run_config=str(run_config), resume=args.resume,
                 execution_code_sha=sha, run_code_sha=run["code_sha"], submission_target=target)
    write_json(state_path, state)
    write_json(output / "delivery_paths.json", dict(code_sha=sha, yaml=str(job_path), run_config=str(run_config),
               submission_target=target,
               mixed_manifest=str(output / "mixed/manifest.json"),
               excel=str(output / "diagnostics/ui5_grpo_training_evaluation.xlsx"),
               latest=str(output / "resume/latest"),
               best_image=str(output / "checkpoints/best-image"), best_bbox=str(output / "checkpoints/best-bbox"),
               best_joint=str(output / "checkpoints/best-joint")))
    print(json.dumps(read_json(output / "delivery_paths.json"), indent=2), flush=True)
    if args.submit:
        from scripts.prepare_ui5_curriculum_snapshot import submit_job
        mlx = shutil.which(args.mlx_bin)
        if not mlx:
            raise FileNotFoundError("mlx is unavailable in the inherited internal submission environment")
        submit_job(mlx, job_path, state_path, state)
    return job_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-submission-dir", type=Path, default=PREVIOUS)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--cluster", choices=tuple(SUBMISSION_CLUSTERS), default="default",
                        help="H20 submission target; default inherits v3, ies_aiai_experience selects group 1602 and its HL queue")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true", help="previous platform job must be stopped; preserve the existing RUN_NAME")
    parser.add_argument("--resume-code-update", action="store_true",
                        help="implies --resume; audit a runtime-only code repair while preserving run.json and completed step 0")
    parser.add_argument("--mlx-bin", default="mlx")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
