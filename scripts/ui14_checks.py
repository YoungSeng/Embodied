"""CPU-only production checkpoint/path and rendered YAML checks."""
from pathlib import Path
from ui14_common import read_json, write_json, INIT_CHECKPOINT, CLUSTER_PROJECT


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


def render_formal_yaml(data_root):
    import yaml
    from submit_locany_ui5 import parse_args, render_job
    args = parse_args(["--profile", "m32-cpt9000-ui14-v1", "--machine", "a800",
        "--resource-group", "aiai_locate", "--gpus", "4", "--ui14-data-root", str(data_root), "--render-only"])
    rendered, runtime = render_job(args)
    parsed = yaml.safe_load(rendered)
    if not isinstance(parsed, dict): raise ValueError("Formal YAML did not parse")
    required = {"GPU_COUNT": 4, "GRADIENT_ACCUMULATION_STEPS": 2, "MAX_STEPS": 16000,
                "INIT_CPT_STEP": 9000, "MAX_SEQ_LENGTH": 7268, "MAX_NUM_TOKENS_PER_SAMPLE": 7268,
                "MAX_NUM_TOKENS": 12800, "RESOURCE_GROUP_ID": 2146, "SEED": 42,
                "EVAL_FAIL_POLICY": "stop", "EVAL_INTERVAL_STEPS": 1000, "SAVE_STEPS": 4000}
    for key, expected in required.items():
        if str(runtime[key]) != str(expected): raise ValueError(f"Formal runtime drift: {key}")
    resource = parsed["jobDefVersion"]["resource"]["arnoldConfig"]
    roles = resource["roles"]
    if (resource["groupIds"] != [2146] or len(roles) != 1 or roles[0]["num"] != 1
        or roles[0]["gpu"] != 4 or roles[0]["gpuv"] != "A800_SXM_40GB"
        or roles[0]["queueName"] != "compute-3302-yg-cloudnative-ai-aiai.locate-guarantee"):
        raise ValueError("Rendered YAML resource does not describe one four-card A800 worker")
    envs = parsed["jobRunParams"]["envsList"]
    for key in ("INIT_CHECKPOINT", "UI14_DATA_ROOT", "UI_TASK_REGISTRY", "UI_EVAL_MANIFEST",
                "META_PATH", "EVAL_FAIL_POLICY", "OUTPUT_DIR"):
        if str(envs[key]) != str(runtime[key]): raise ValueError(f"Rendered YAML environment drift: {key}")
    if not runtime["BASE_MODEL"] == runtime["MODEL_PATH"] == runtime["INIT_CHECKPOINT"] == INIT_CHECKPOINT:
        raise ValueError("CPT initialization path drift")
    if runtime["PROJECT_ROOT"] != CLUSTER_PROJECT: raise ValueError("Formal project path drift")
    for token in ("compute-3302-yg-cloudnative-ai-aiai.locate-guarantee", "2146"):
        if token not in rendered: raise ValueError(f"Missing resource field in YAML: {token}")
    path, runtime_path = Path(data_root) / "formal_job.yaml", Path(data_root) / "formal_runtime.json"
    path.write_text(rendered, encoding="utf-8")
    write_json(runtime_path, runtime)
    return path, runtime_path
