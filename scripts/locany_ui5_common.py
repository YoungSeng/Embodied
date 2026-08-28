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
            f"locany-3b-ui5-{machine_type}x{gpu_count}-full-{version}-en",
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
    use_detection_crops = parse_bool(
        _env_value(env, "UI5_USE_DETECTION_CROPS", "0"),
        name="UI5_USE_DETECTION_CROPS",
    )
    crop_audit_dir = str(_env_value(env, "UI5_CROP_AUDIT_DIR", ""))
    crop_train_mode = str(
        _env_value(
            env,
            "UI5_CROP_TRAIN_MODE",
            "full_plus_crop" if use_detection_crops else "full_only",
        )
    )
    if crop_train_mode not in {"full_only", "full_plus_crop"}:
        raise ValueError("UI5_CROP_TRAIN_MODE must be full_only or full_plus_crop")
    if use_detection_crops and crop_train_mode != "full_plus_crop":
        raise ValueError(
            "UI5_USE_DETECTION_CROPS=1 requires UI5_CROP_TRAIN_MODE=full_plus_crop"
        )
    if use_detection_crops and not crop_audit_dir:
        raise ValueError("UI5_USE_DETECTION_CROPS=1 requires UI5_CROP_AUDIT_DIR")
    crop_meta_path = str(_env_value(env, "UI5_CROP_META_PATH", ""))
    if crop_audit_dir:
        if not crop_meta_path:
            crop_meta_path = join_runtime_path(
                crop_audit_dir,
                "training_recipes",
                f"ui_defect_5class_train_{crop_train_mode}.json",
            )
        meta_path = crop_meta_path
    elif crop_meta_path:
        raise ValueError("UI5_CROP_META_PATH requires UI5_CROP_AUDIT_DIR")
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
    eval_inference_crop_mode = str(
        _env_value(env, "EVAL_INFERENCE_CROP_MODE", "detector_scan")
    ).lower()
    if eval_inference_crop_mode not in {"full_image", "lossless_tiling", "detector_scan"}:
        raise ValueError(
            "EVAL_INFERENCE_CROP_MODE must be full_image, lossless_tiling, or detector_scan"
        )

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
        "RESOURCE_GROUP": str(_env_value(env, "RESOURCE_GROUP", "default")),
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
        "UI5_USE_DETECTION_CROPS": int(use_detection_crops),
        "UI5_CROP_AUDIT_DIR": crop_audit_dir,
        "UI5_CROP_TRAIN_MODE": crop_train_mode,
        "UI5_CROP_META_PATH": crop_meta_path,
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
        "GRADIENT_ACCUMULATION_STEPS": int(
            _env_value(
                env,
                "GRADIENT_ACCUMULATION_STEPS",
                2 if gpu_count == 4 else 1,
            )
        ),
        "RELATION_GATE_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_GATE_LOSS_WEIGHT", 1.0)
        ),
        "RELATION_SLOT_GATE_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_SLOT_GATE_LOSS_WEIGHT", 0.1)
        ),
        "RELATION_ATTENTION_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_ATTENTION_LOSS_WEIGHT", 0.1)
        ),
        "RELATION_GATE_THRESHOLD": float(
            _env_value(env, "RELATION_GATE_THRESHOLD", 0.5)
        ),
        "RELATION_GATE_MODE": str(
            _env_value(env, "RELATION_GATE_MODE", "observe")
        ).lower(),
        "RELATION_FOCAL_BETA": float(
            _env_value(env, "RELATION_FOCAL_BETA", 0.999)
        ),
        "RELATION_FOCAL_GAMMA": float(
            _env_value(env, "RELATION_FOCAL_GAMMA", 2.0)
        ),
        "RELATION_NUM_SLOTS": int(
            _env_value(env, "RELATION_NUM_SLOTS", 8)
        ),
        "SAVE_STEPS": save_steps,
        "ENABLE_EVAL": int(enable_eval),
        "EVAL_AT_START": int(eval_at_start),
        "EVAL_INTERVAL_STEPS": eval_interval,
        "EVAL_FAIL_POLICY": eval_fail_policy,
        "EVAL_INFERENCE_CROP_MODE": eval_inference_crop_mode,
        "EVAL_PARSER_ROOT": str(
            _env_value(
                env,
                "EVAL_PARSER_ROOT",
                join_runtime_path(str(Path(project_root).parent), "ui-region-parser")
                if not project_root.startswith("/")
                else posixpath.join(posixpath.dirname(project_root), "ui-region-parser"),
            )
        ),
        "EVAL_DETECTOR_CACHE": str(
            _env_value(
                env,
                "EVAL_DETECTOR_CACHE",
                join_runtime_path(output_dir, "evaluation", "detector_scan_cache"),
            )
        ),
        "EVAL_DETECTOR_CACHE_MODE": str(
            _env_value(env, "EVAL_DETECTOR_CACHE_MODE", "readonly")
        ).lower(),
        "EVAL_SCAN_NAME": str(
            _env_value(env, "EVAL_SCAN_NAME", "horizontal_scan_v3_no_overlap")
        ),
        "EVAL_REQUIRE_CACHE_SCOPE": str(
            _env_value(env, "EVAL_REQUIRE_CACHE_SCOPE", "full_test")
        ).lower(),
        "EVAL_REQUIRE_STRICT_NONOVERLAP": int(
            parse_bool(
                _env_value(env, "EVAL_REQUIRE_STRICT_NONOVERLAP", "1"),
                name="EVAL_REQUIRE_STRICT_NONOVERLAP",
            )
        ),
        "EVAL_EXPECTED_UNIQUE_IMAGES": int(
            _env_value(env, "EVAL_EXPECTED_UNIQUE_IMAGES", 17281)
        ),
        "EVAL_TEXT_PYTHON": str(_env_value(env, "EVAL_TEXT_PYTHON", "")),
        "EVAL_ICON_PYTHON": str(_env_value(env, "EVAL_ICON_PYTHON", "")),
        "EVAL_TEXT_MODEL_DIR": str(_env_value(env, "EVAL_TEXT_MODEL_DIR", "")),
        "EVAL_ICON_MODEL": str(
            _env_value(
                env,
                "EVAL_ICON_MODEL",
                join_runtime_path(
                    str(_env_value(env, "EVAL_PARSER_ROOT", posixpath.join(posixpath.dirname(project_root), "ui-region-parser"))),
                    "weights",
                    "icon_detect_v3",
                    "model.pt",
                ),
            )
        ),
        "EVAL_DETECTOR_WORKERS_PER_GPU": int(
            _env_value(env, "EVAL_DETECTOR_WORKERS_PER_GPU", 1)
        ),
        "EVAL_TILE_MAX_COUNT": int(_env_value(env, "EVAL_TILE_MAX_COUNT", 10)),
        "EVAL_TILE_TARGET_LONG_SIDE": int(
            _env_value(env, "EVAL_TILE_TARGET_LONG_SIDE", 1600)
        ),
        "EVAL_TILE_OVERLAP_RATIO": float(
            _env_value(env, "EVAL_TILE_OVERLAP_RATIO", 0.10)
        ),
        "EVAL_TILE_NMS_IOU": float(_env_value(env, "EVAL_TILE_NMS_IOU", 0.50)),
        "EVAL_SCAN_TARGET_HEIGHT": int(
            _env_value(env, "EVAL_SCAN_TARGET_HEIGHT", 960)
        ),
        "EVAL_SCAN_VERTICAL_LINK_RATIO": float(
            _env_value(env, "EVAL_SCAN_VERTICAL_LINK_RATIO", 0.025)
        ),
        "EVAL_SCAN_CONTEXT_RATIO": float(
            _env_value(env, "EVAL_SCAN_CONTEXT_RATIO", 0.20)
        ),
        "EVAL_SCAN_MIN_CONTEXT_IMAGE_RATIO": float(
            _env_value(env, "EVAL_SCAN_MIN_CONTEXT_IMAGE_RATIO", 0.015)
        ),
        "EVAL_SCAN_DENSE_BAND_RATIO": float(
            _env_value(env, "EVAL_SCAN_DENSE_BAND_RATIO", 0.80)
        ),
        "EVAL_SCAN_VISUALIZATION_SAMPLES": int(
            _env_value(env, "EVAL_SCAN_VISUALIZATION_SAMPLES", 20)
        ),
        "INSTALL_SYSTEM_RUNTIME_DEPS": int(
            parse_bool(
                _env_value(env, "INSTALL_SYSTEM_RUNTIME_DEPS", "0"),
                name="INSTALL_SYSTEM_RUNTIME_DEPS",
            )
        ),
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
    if resolved["GRADIENT_ACCUMULATION_STEPS"] < 1:
        raise ValueError("GRADIENT_ACCUMULATION_STEPS must be positive")
    if not 0.0 <= resolved["RELATION_GATE_THRESHOLD"] <= 1.0:
        raise ValueError("RELATION_GATE_THRESHOLD must be in [0, 1]")
    if resolved["RELATION_GATE_MODE"] not in {"observe", "hard"}:
        raise ValueError("RELATION_GATE_MODE must be observe or hard")
    if not 1 <= resolved["EVAL_TILE_MAX_COUNT"] <= 10:
        raise ValueError("EVAL_TILE_MAX_COUNT must be in [1, 10]")
    if resolved["EVAL_TILE_TARGET_LONG_SIDE"] <= 0:
        raise ValueError("EVAL_TILE_TARGET_LONG_SIDE must be positive")
    if not 0 < resolved["EVAL_TILE_OVERLAP_RATIO"] < 1:
        raise ValueError("EVAL_TILE_OVERLAP_RATIO must be in (0, 1)")
    if not 0 <= resolved["EVAL_TILE_NMS_IOU"] <= 1:
        raise ValueError("EVAL_TILE_NMS_IOU must be in [0, 1]")
    if resolved["EVAL_DETECTOR_WORKERS_PER_GPU"] not in {1, 2}:
        raise ValueError("EVAL_DETECTOR_WORKERS_PER_GPU must be 1 or 2")
    if resolved["EVAL_DETECTOR_CACHE_MODE"] not in {"build", "readonly"}:
        raise ValueError("EVAL_DETECTOR_CACHE_MODE must be build or readonly")
    if resolved["EVAL_REQUIRE_CACHE_SCOPE"] not in {"preview", "full_test"}:
        raise ValueError("EVAL_REQUIRE_CACHE_SCOPE must be preview or full_test")
    if resolved["EVAL_EXPECTED_UNIQUE_IMAGES"] < 0:
        raise ValueError("EVAL_EXPECTED_UNIQUE_IMAGES cannot be negative")
    if resolved["EVAL_SCAN_TARGET_HEIGHT"] <= 0:
        raise ValueError("EVAL_SCAN_TARGET_HEIGHT must be positive")
    for name in (
        "EVAL_SCAN_VERTICAL_LINK_RATIO",
        "EVAL_SCAN_CONTEXT_RATIO",
        "EVAL_SCAN_MIN_CONTEXT_IMAGE_RATIO",
    ):
        if resolved[name] < 0:
            raise ValueError(f"{name} cannot be negative")
    if not 0 < resolved["EVAL_SCAN_DENSE_BAND_RATIO"] <= 1:
        raise ValueError("EVAL_SCAN_DENSE_BAND_RATIO must be in (0, 1]")
    if not 0.0 <= resolved["RELATION_FOCAL_BETA"] < 1.0:
        raise ValueError("RELATION_FOCAL_BETA must be in [0, 1)")
    if resolved["RELATION_FOCAL_GAMMA"] < 0.0:
        raise ValueError("RELATION_FOCAL_GAMMA cannot be negative")
    if resolved["RELATION_NUM_SLOTS"] < 1:
        raise ValueError("RELATION_NUM_SLOTS must be positive")
    return resolved


GPU_PARITY_ALLOWED_DIFFERENCES = frozenset(
    {
        "GPU_COUNT",
        "CUDA_DEVICES",
        "MAX_NUM_TOKENS",
        "GRADIENT_ACCUMULATION_STEPS",
        "RUN_NAME",
        "OUTPUT_DIR",
        "EVAL_DETECTOR_CACHE",
    }
)


def assert_gpu_mode_consistency(
    four_gpu: Mapping[str, Any],
    eight_gpu: Mapping[str, Any],
) -> None:
    """Fail if a 4/8-GPU pair differs in any training/eval setting not allowed."""

    accumulation_pair = (
        int(four_gpu.get("GRADIENT_ACCUMULATION_STEPS", -1)),
        int(eight_gpu.get("GRADIENT_ACCUMULATION_STEPS", -1)),
    )
    if accumulation_pair != (2, 1):
        raise ValueError(
            "4/8-GPU gradient accumulation must preserve the original 2/1 "
            f"schedule, got {accumulation_pair[0]}/{accumulation_pair[1]}"
        )

    keys = set(four_gpu) | set(eight_gpu)
    unexpected = {
        key: (four_gpu.get(key), eight_gpu.get(key))
        for key in sorted(keys - GPU_PARITY_ALLOWED_DIFFERENCES)
        if four_gpu.get(key) != eight_gpu.get(key)
    }
    if unexpected:
        details = ", ".join(
            f"{key}={values[0]!r}/{values[1]!r}"
            for key, values in unexpected.items()
        )
        raise ValueError(
            "4/8-GPU pipeline parity violation; only GPU count, visible devices, "
            "MAX_NUM_TOKENS, gradient accumulation 2/1 and output naming may "
            f"differ: {details}"
        )


def machine_resource_config(
    machine_type: str,
    *,
    resource_group: str = "default",
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    raw = load_machine_config(config_path)
    machine_type = machine_type.lower()
    if machine_type not in raw["machines"]:
        raise ValueError(f"Unknown machine type: {machine_type}")
    machine = raw["machines"][machine_type]
    resource_groups = machine.get("resource_groups", {"default": {}})
    if resource_group not in resource_groups:
        raise ValueError(
            f"Unknown resource group {resource_group!r} for {machine_type}; "
            f"choose one of {sorted(resource_groups)}"
        )
    resource_override = resource_groups[resource_group]
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
        **resource_override,
        "resource_group": resource_group,
    }
