#!/usr/bin/env python3
"""Shared configuration helpers for the LocateAnything UI5 v4 pipeline."""

from __future__ import annotations

import json
import os
import posixpath
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "locany_ui5_machines.json"
TASKS = (
    "occlusion",
    "cropping",
    "text_overflow",
    "text_ellipsis",
    "content_missing",
)
TASK_JSONL = {
    "occlusion": "test_ui_occlusion_wcnt_no_figma.jsonl",
    "cropping": "test_ui_cropping_wcnt_no_figma.jsonl",
    "text_overflow": "test_ui_text_overflow_wcnt_no_figma.jsonl",
    "text_ellipsis": "test_ui_text_ellipsis_wcnt_no_figma.jsonl",
    "content_missing": "test_ui_content_missing_wcnt_no_figma.jsonl",
}
TASK_ISSUE_NAMES = {
    "occlusion": "元素重叠",
    "cropping": "元素被裁切",
    "text_overflow": "文字溢出容器",
    "text_ellipsis": "文字省略异常",
    "content_missing": "内容未展示",
}


def parse_bool(value: Any, *, name: str = "value") -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def parse_gpu_devices(value: str, expected_count: int | None = None) -> list[str]:
    devices = [part.strip() for part in str(value).split(",") if part.strip()]
    if not devices:
        raise ValueError("GPU device list is empty")
    if len(set(devices)) != len(devices):
        raise ValueError(f"GPU device list contains duplicates: {value!r}")
    if expected_count is not None and len(devices) != expected_count:
        raise ValueError(
            f"GPU_COUNT={expected_count}, but CUDA_DEVICES resolves to {len(devices)} devices: {value}"
        )
    return devices


def load_machine_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported machine config schema: {config.get('schema_version')!r}")
    return config


def _env_value(env: Mapping[str, str], name: str, default: Any) -> Any:
    value = env.get(name)
    return default if value is None or value == "" else value


def join_runtime_path(root: str, *parts: str) -> str:
    """Join cluster POSIX paths correctly even when rendering on Windows."""

    if str(root).startswith("/"):
        return posixpath.join(str(root), *(str(part) for part in parts))
    return str(Path(root).joinpath(*parts))


def resolve_runtime_config(
    env: Mapping[str, str] | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Resolve final runtime values with environment variables taking precedence."""

    env = os.environ if env is None else env
    raw = load_machine_config(config_path)
    shared = raw["shared"]

    machine_type = str(_env_value(env, "MACHINE_TYPE", "a800")).lower()
    if machine_type not in raw["machines"]:
        raise ValueError(
            f"Unsupported MACHINE_TYPE={machine_type!r}; choose one of {sorted(raw['machines'])}"
        )
    machine = raw["machines"][machine_type]

    gpu_count = int(_env_value(env, "GPU_COUNT", 8 if machine_type == "a800" else 4))
    gpu_defaults = machine["gpu_defaults"].get(str(gpu_count))
    if gpu_defaults is None:
        raise ValueError(
            f"Unsupported GPU_COUNT={gpu_count} for {machine_type}; "
            f"choose one of {sorted(machine['gpu_defaults'])}"
        )

    cuda_devices = str(
        _env_value(env, "CUDA_DEVICES", ",".join(str(index) for index in range(gpu_count)))
    )
    train_devices = parse_gpu_devices(cuda_devices, gpu_count)

    eval_gpu_default = ",".join(train_devices[: min(4, gpu_count)])
    eval_gpu_devices = str(_env_value(env, "EVAL_GPU_DEVICES", eval_gpu_default))
    evaluation_devices = parse_gpu_devices(eval_gpu_devices)
    if not set(evaluation_devices).issubset(set(train_devices)):
        raise ValueError(
            f"EVAL_GPU_DEVICES={eval_gpu_devices} must be a subset of "
            f"CUDA_DEVICES={cuda_devices}"
        )

    workspace = str(machine["workspace"])
    project_root_default = join_runtime_path(workspace, shared["project_relative_path"])
    project_root = str(_env_value(env, "PROJECT_ROOT", project_root_default))
    data_version = str(_env_value(env, "DATA_VERSION", "v3"))
    version = str(_env_value(env, "VERSION", "v4"))

    format_values = {"workspace": workspace, "project_root": project_root}
    output_base = str(
        _env_value(
            env,
            "OUTPUT_BASE",
            machine["output_base_template"].format(**format_values),
        )
    )
    run_name = str(
        _env_value(
            env,
            "RUN_NAME",
            f"locany-3b-ui5-{machine_type}-full-{version}-en",
        )
    )
    output_dir = str(
        _env_value(env, "OUTPUT_DIR", join_runtime_path(output_base, run_name))
    )

    training_data_rel = shared["training_data_relative_path"].format(
        data_version=data_version
    )
    training_data_dir = str(
        _env_value(env, "TRAINING_DATA_DIR", join_runtime_path(project_root, training_data_rel))
    )
    training_data_source_rel = shared["training_data_source_relative_path"].format(
        data_version=data_version
    )
    training_data_source_dir = str(
        _env_value(
            env,
            "TRAINING_DATA_SOURCE_DIR",
            join_runtime_path(workspace, training_data_source_rel),
        )
    )
    meta_path = str(
        _env_value(
            env,
            "META_PATH",
            join_runtime_path(training_data_dir, "recipe", "ui_defect_5class_train.json"),
        )
    )
    eval_input_dir = str(
        _env_value(
            env,
            "EVAL_INPUT_DIR",
            join_runtime_path(workspace, shared["eval_data_relative_path"]),
        )
    )

    max_steps = int(_env_value(env, "MAX_STEPS", 16000))
    save_steps = int(_env_value(env, "SAVE_STEPS", 4000))
    eval_interval = int(_env_value(env, "EVAL_INTERVAL_STEPS", 1000))
    if min(max_steps, save_steps, eval_interval) <= 0:
        raise ValueError("MAX_STEPS, SAVE_STEPS, and EVAL_INTERVAL_STEPS must be positive")

    enable_eval = parse_bool(_env_value(env, "ENABLE_EVAL", "1"), name="ENABLE_EVAL")
    eval_at_start = parse_bool(
        _env_value(env, "EVAL_AT_START", "1"), name="EVAL_AT_START"
    )
    eval_fail_policy = str(_env_value(env, "EVAL_FAIL_POLICY", "stop")).lower()
    if eval_fail_policy not in {"stop", "warn"}:
        raise ValueError("EVAL_FAIL_POLICY must be 'stop' or 'warn'")

    base_model = str(
        _env_value(
            env,
            "BASE_MODEL",
            join_runtime_path(workspace, shared["base_model_relative_path"]),
        )
    )
    scorer_root = str(
        _env_value(
            env,
            "SCORER_ROOT",
            join_runtime_path(workspace, shared["scorer_relative_path"]),
        )
    )

    resolved: dict[str, Any] = {
        "MACHINE_TYPE": machine_type,
        "GPU_COUNT": gpu_count,
        "CUDA_DEVICES": cuda_devices,
        "EVAL_GPU_DEVICES": eval_gpu_devices,
        "WORKSPACE": workspace,
        "PROJECT_ROOT": project_root,
        "ENV_DIR": str(
            _env_value(
                env,
                "ENV_DIR",
                join_runtime_path(workspace, shared["conda_env_relative_path"]),
            )
        ),
        "HF_HOME": str(
            _env_value(
                env,
                "HF_HOME",
                machine["hf_home_template"].format(**format_values),
            )
        ),
        "BASE_MODEL": base_model,
        "MODEL_PATH": str(_env_value(env, "MODEL_PATH", base_model)),
        "SCORER_ROOT": scorer_root,
        "DATA_VERSION": data_version,
        "VERSION": version,
        "TRAINING_DATA_DIR": training_data_dir,
        "TRAINING_DATA_SOURCE_DIR": training_data_source_dir,
        "META_PATH": meta_path,
        "EVAL_INPUT_DIR": eval_input_dir,
        "OUTPUT_BASE": output_base,
        "RUN_NAME": run_name,
        "OUTPUT_DIR": output_dir,
        "ATTN_IMPLEMENTATION": str(
            _env_value(env, "ATTN_IMPLEMENTATION", machine["attention_implementation"])
        ),
        "MAX_SEQ_LENGTH": int(
            _env_value(env, "MAX_SEQ_LENGTH", machine["max_seq_length"])
        ),
        "MAX_NUM_TOKENS_PER_SAMPLE": int(
            _env_value(
                env,
                "MAX_NUM_TOKENS_PER_SAMPLE",
                machine["max_num_tokens_per_sample"],
            )
        ),
        "MAX_NUM_TOKENS": int(
            _env_value(env, "MAX_NUM_TOKENS", gpu_defaults["max_num_tokens"])
        ),
        "MAX_NUM_TOKENS_SCOPE": "per_rank_packed_batch",
        "MAX_STEPS": max_steps,
        "WARMUP_STEPS": int(_env_value(env, "WARMUP_STEPS", 500)),
        "LEARNING_RATE": str(_env_value(env, "LEARNING_RATE", "2e-5")),
        "SAVE_STEPS": save_steps,
        "ENABLE_EVAL": int(enable_eval),
        "EVAL_AT_START": int(eval_at_start),
        "EVAL_INTERVAL_STEPS": eval_interval,
        "EVAL_FAIL_POLICY": eval_fail_policy,
        "EVAL_MAX_IMAGES_PER_TASK": int(
            _env_value(env, "EVAL_MAX_IMAGES_PER_TASK", 0)
        ),
        "PIPELINE_MODE": str(_env_value(env, "PIPELINE_MODE", "train")).lower(),
    }
    if resolved["PIPELINE_MODE"] not in {"train", "eval"}:
        raise ValueError("PIPELINE_MODE must be 'train' or 'eval'")
    if resolved["MAX_NUM_TOKENS"] < resolved["MAX_NUM_TOKENS_PER_SAMPLE"]:
        raise ValueError(
            "MAX_NUM_TOKENS cannot be smaller than MAX_NUM_TOKENS_PER_SAMPLE; "
            "otherwise some accepted samples can never fit in a packed batch"
        )
    return resolved


def machine_resource_config(
    machine_type: str,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    raw = load_machine_config(config_path)
    machine_type = machine_type.lower()
    if machine_type not in raw["machines"]:
        raise ValueError(f"Unknown machine type: {machine_type}")
    machine = raw["machines"][machine_type]
    workspace = machine["workspace"]
    project_root = join_runtime_path(workspace, raw["shared"]["project_relative_path"])
    return {
        "workspace": workspace,
        "project_root": project_root,
        "env_dir": join_runtime_path(
            workspace, raw["shared"]["conda_env_relative_path"]
        ),
        "image_url": raw["shared"]["image_url"],
        "image_vid": raw["shared"]["image_vid"],
        **machine["merlin"],
    }
