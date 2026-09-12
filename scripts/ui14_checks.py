"""CPU-only production checkpoint/path and rendered YAML checks."""
from pathlib import Path
from ui14_common import read_json, write_json, file_digest, INIT_CHECKPOINT, CLUSTER_PROJECT


def validate_initial_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    config = read_json(checkpoint / "config.json")
    shards = set()
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (checkpoint / index).is_file():
            shards.update(read_json(checkpoint / index)["weight_map"].values())
    if not shards:
        shards = {p.name for pattern in ("model*.safetensors", "pytorch_model*.bin") for p in checkpoint.glob(pattern)}
    if not shards: raise ValueError("CPT checkpoint has no model weights")
    for name in shards:
        path = (checkpoint / name).resolve(strict=True)
        if checkpoint not in path.parents or path.stat().st_size <= 0: raise ValueError(f"Unreadable CPT weight shard: {path}")
        with path.open("rb") as handle: handle.read(16)
    return {"readable_weight_shards": len(shards), "config_model_type": config.get("model_type"),
            "optimizer_source": "new SFT optimizer/scheduler; CPT optimizer is never resumed",
            "sft_start_step": 0, "gpu_loaded": False}


def validate_formal_yaml(rendered, runtime, *, config_path=None):
    import yaml
    from locany_ui5_common import machine_resource_config
    selected = machine_resource_config("a800", resource_group=runtime["RESOURCE_GROUP"], config_path=config_path)
    parsed = yaml.safe_load(rendered)
    if not isinstance(parsed, dict): raise ValueError("Formal YAML did not parse")
    required = {"GPU_COUNT": 4, "GRADIENT_ACCUMULATION_STEPS": 2, "MAX_STEPS": 16000,
                "INIT_CPT_STEP": 9000, "MAX_SEQ_LENGTH": 7268, "MAX_NUM_TOKENS_PER_SAMPLE": 7268,
                "MAX_NUM_TOKENS": 12800, "RESOURCE_GROUP_ID": selected["group_id"], "SEED": 42,
                "EVAL_FAIL_POLICY": "stop", "EVAL_INTERVAL_STEPS": 1000, "SAVE_STEPS": 4000,
                "EVAL_INFERENCE_WORKERS_PER_GPU": 1, "EVAL_AT_START": 1}
    for key, expected in required.items():
        if str(runtime[key]) != str(expected): raise ValueError(f"Formal runtime drift: {key}")
    resource = parsed["jobDefVersion"]["resource"]["arnoldConfig"]
    roles = resource["roles"]
    if (resource["clusterId"] != selected["cluster_id"] or resource["groupIds"] != [selected["group_id"]]
        or len(roles) != 1 or roles[0]["num"] != 1
        or roles[0]["gpu"] != 4 or roles[0]["gpuv"] != "A800_SXM_40GB"
        or roles[0].get("queueName", "") != selected.get("queue_name", "")):
        raise ValueError("Rendered YAML resource does not describe one four-card A800 worker")
    envs = parsed["jobRunParams"]["envsList"]
    for key in ("INIT_CHECKPOINT", "UI14_DATA_ROOT", "UI_TASK_REGISTRY", "UI_EVAL_MANIFEST",
                "META_PATH", "EVAL_AT_START", "EVAL_FAIL_POLICY", "EVAL_INFERENCE_WORKERS_PER_GPU", "OUTPUT_DIR", "RESOURCE_GROUP"):
        if str(envs[key]) != str(runtime[key]): raise ValueError(f"Rendered YAML environment drift: {key}")
    if not runtime["BASE_MODEL"] == runtime["MODEL_PATH"] == runtime["INIT_CHECKPOINT"] == INIT_CHECKPOINT:
        raise ValueError("CPT initialization path drift")
    from ui14_neg11_data import NEG_PROJECT
    expected_project = NEG_PROJECT if runtime.get("UI_TRAIN_PROFILE") == "m32-cpt9000-ui14-neg11-v1" else CLUSTER_PROJECT
    from ui14_alignment_common import PROFILE, PROJECT
    if runtime.get("UI_TRAIN_PROFILE") == PROFILE:
        expected_project = PROJECT
        if runtime.get("UI_EVAL_ANSWER_GRAMMAR") != "ui14_answer_v1": raise ValueError("Alignment answer grammar drift")
    from ui14_alignment_crops_common import PROFILE as CROPS_PROFILE, PROJECT as CROPS_PROJECT, OUTPUT as CROPS_OUTPUT
    if runtime.get("UI_TRAIN_PROFILE") == CROPS_PROFILE:
        expected_project = CROPS_PROJECT
        if runtime.get("UI_EVAL_ANSWER_GRAMMAR") != "legacy": raise ValueError("Production answer grammar drift")
        if runtime["OUTPUT_DIR"] != CROPS_OUTPUT: raise ValueError("Independent alignment-crops run output drift")
        from locany_ui5_common import UI14_EXCLUSIVE_GPU_TASKS
        if not set(UI14_EXCLUSIVE_GPU_TASKS).issubset(runtime.get("EVAL_EXCLUSIVE_GPU_TASKS", "").split()):
            raise ValueError("Alignment-crops run must reserve a physical GPU for known OOM tasks")
    if runtime["PROJECT_ROOT"] != expected_project: raise ValueError("Formal project path drift")


def render_formal_yaml(data_root, resource_group="aiai_locate", profile="m32-cpt9000-ui14-v1"):
    from submit_locany_ui5 import parse_args, render_job
    args = parse_args(["--profile", profile, "--machine", "a800",
        "--resource-group", resource_group, "--gpus", "4", "--ui14-data-root", str(data_root), "--render-only"])
    rendered, runtime = render_job(args)
    validate_formal_yaml(rendered, runtime, config_path=args.config)
    path, runtime_path = Path(data_root) / "formal_job.yaml", Path(data_root) / "formal_runtime.json"
    path.write_text(rendered, encoding="utf-8")
    write_json(runtime_path, runtime)
    return path, runtime_path


def write_submission_artifacts(output_yaml, rendered, runtime, *, render_only):
    """Keep the finalized data evidence immutable when selecting a resource group."""
    import hashlib
    import os
    output_yaml = Path(output_yaml).resolve()
    root = Path(runtime["UI14_DATA_ROOT"]).resolve()
    report_path = root / "cpu_check_report.json"
    report = read_json(report_path) if report_path.is_file() else {}
    payload = rendered.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    runtime_path = output_yaml.with_suffix(".runtime.json")
    binding_path = output_yaml.with_suffix(".binding.json")
    protected = {(root / name).resolve(): expected for name, expected in report.get("artifact_digests", {}).items()}
    if (output_yaml in protected and protected[output_yaml] != digest
            or runtime_path in protected or binding_path in protected):
        raise ValueError("Submission would overwrite CPU-checked artifacts; use UI14_DATA_ROOT/submissions/ for --output-yaml")
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    if not output_yaml.is_file() or output_yaml.read_bytes() != payload:
        temporary = output_yaml.with_name(output_yaml.name + f".tmp.{os.getpid()}")
        temporary.write_bytes(payload)
        os.replace(temporary, output_yaml)
    write_json(runtime_path, runtime)
    write_json(binding_path, {
        "render_only": bool(render_only), "yaml": str(output_yaml), "yaml_sha256": digest,
        "runtime": str(runtime_path), "runtime_sha256": file_digest(runtime_path),
        "cpu_check_report": str(report_path), "cpu_check_report_sha256": file_digest(report_path) if report_path.is_file() else None,
        "normalization_id": report.get("normalization_id"), "repair_run_id": report.get("repair_run_id"),
        "resource_group": runtime["RESOURCE_GROUP"], "resource_group_id": runtime["RESOURCE_GROUP_ID"],
        "resource_queue_name": runtime["RESOURCE_QUEUE_NAME"],
    })
