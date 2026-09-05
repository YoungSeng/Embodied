#!/usr/bin/env python3
"""Shared configuration helpers for the LocateAnything UI5 v4 pipeline."""

from __future__ import annotations

import json
import math
import os
import posixpath
import re
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

# The five production evaluation JSONL files share one pool of 1,555
# content-unique images.  Keep this separate from the 17,281-image training
# pool used by crop_audit_v4_gt_repair.
DEFAULT_UI5_FULL_TEST_UNIQUE_IMAGES = 1555


def _finite_gate_probability(value: Any) -> float | None:
    """Return a finite Gate probability without accepting booleans as numbers."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def aggregate_tiled_gate_diagnostics(
    tile_gates: list[dict[str, Any]],
    *,
    crop_mode: str,
) -> dict[str, Any]:
    """Aggregate per-tile image-Gate diagnostics into one source-image record.

    Detector-scan inference evaluates several non-overlapping views for one
    source image.  The source image is positive when any view is positive, so
    its image-level Gate score is the maximum finite per-view ``p_defect``.
    Keeping the original tile diagnostics makes the aggregation auditable and
    lets historical runs created before the top-level score was added recover
    without repeating model inference.
    """

    scores = [
        score
        for row in tile_gates
        if (score := _finite_gate_probability(row.get("p_defect"))) is not None
    ]
    available_count = sum(bool(row.get("available")) for row in tile_gates)
    return {
        "available": available_count > 0,
        "available_tile_count": available_count,
        "p_defect": max(scores) if scores else None,
        "p_defect_tile_count": len(scores),
        "p_defect_aggregation": "max_tile",
        "would_pass": any(bool(row.get("would_pass")) for row in tile_gates),
        "gate_filtered": bool(tile_gates)
        and all(bool(row.get("gate_filtered")) for row in tile_gates),
        "mode": crop_mode,
        "tile_count": len(tile_gates),
        "tile_union_full_image": True,
        "gt_repair_used": False,
        "tile_gates": tile_gates,
    }


def image_gate_probability(record: Mapping[str, Any]) -> tuple[float | None, str]:
    """Read an image Gate score, recovering legacy tiled sidecars when needed."""

    top_level = _finite_gate_probability(record.get("p_defect"))
    if top_level is not None:
        return top_level, "top_level"
    tile_gates = record.get("tile_gates")
    if isinstance(tile_gates, list):
        scores = [
            score
            for row in tile_gates
            if isinstance(row, dict)
            and (score := _finite_gate_probability(row.get("p_defect"))) is not None
        ]
        if scores:
            return max(scores), "legacy_tile_gates_max"
    return None, "missing"


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
    if env.get("UI_TRAIN_PROFILE") == "m32-cpt9000-ui14-v1":
        from ui14_profile import profile_environment
        env = {**profile_environment(project_root=env.get("PROJECT_ROOT"), data_root=env.get("UI14_DATA_ROOT")), **env}
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
            "crop_only" if use_detection_crops else "full_only",
        )
    )
    if crop_train_mode not in {"full_only", "crop_only"}:
        raise ValueError(
            "UI5_CROP_TRAIN_MODE must be full_only or crop_only"
        )
    if use_detection_crops and crop_train_mode != "crop_only":
        raise ValueError("UI5_USE_DETECTION_CROPS=1 requires crop_only")
    ui_sampling_mode = str(
        _env_value(
            env,
            "UI5_UI_SAMPLING_MODE",
            (
                "task_source_balanced_rotating"
                if crop_train_mode == "crop_only"
                else "fixed_ratio"
            ),
        )
    )
    if ui_sampling_mode not in {
        "fixed_ratio",
        "task_source_balanced_rotating",
    }:
        raise ValueError(
            "UI5_UI_SAMPLING_MODE must be fixed_ratio or "
            "task_source_balanced_rotating"
        )
    if crop_train_mode == "crop_only" and ui_sampling_mode != "task_source_balanced_rotating":
        raise ValueError(
            "crop_only requires UI5_UI_SAMPLING_MODE=task_source_balanced_rotating"
        )
    ui_negative_to_positive_ratio = float(
        _env_value(env, "UI_NEGATIVE_TO_POSITIVE_RATIO", 2.0)
    )
    if ui_negative_to_positive_ratio <= 0:
        raise ValueError("UI_NEGATIVE_TO_POSITIVE_RATIO must be positive")
    if crop_train_mode == "crop_only" and not math.isclose(
        ui_negative_to_positive_ratio, 2.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("crop_only requires UI_NEGATIVE_TO_POSITIVE_RATIO=2.0")
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
    eval_data_split_hint = str(
        _env_value(env, "EVAL_DATA_SPLIT", "test")
    ).lower()
    eval_input_default = join_runtime_path(
        workspace, shared["eval_data_relative_path"]
    )
    eval_input_dir = str(
        _env_value(
            env,
            "EVAL_INPUT_DIR",
            eval_input_default,
        )
    )

    max_steps = int(_env_value(env, "MAX_STEPS", 16000))
    save_steps = int(_env_value(env, "SAVE_STEPS", 4000))
    eval_interval = int(_env_value(env, "EVAL_INTERVAL_STEPS", 1000))
    if min(max_steps, save_steps, eval_interval) <= 0:
        raise ValueError("MAX_STEPS, SAVE_STEPS, and EVAL_INTERVAL_STEPS must be positive")

    enable_eval = parse_bool(_env_value(env, "ENABLE_EVAL", "1"), name="ENABLE_EVAL")
    eval_at_start = parse_bool(
        _env_value(env, "EVAL_AT_START", "0"), name="EVAL_AT_START"
    )
    eval_fail_policy = str(_env_value(env, "EVAL_FAIL_POLICY", "stop")).lower()
    if eval_fail_policy not in {"stop", "warn"}:
        raise ValueError("EVAL_FAIL_POLICY must be 'stop' or 'warn'")
    eval_validation_early_stop = parse_bool(
        _env_value(env, "EVAL_VALIDATION_EARLY_STOP", "0"),
        name="EVAL_VALIDATION_EARLY_STOP",
    )
    if eval_validation_early_stop:
        raise ValueError("validation early stop is disabled for formal UI5 training")
    eval_data_split = eval_data_split_hint
    if eval_data_split != "test":
        raise ValueError("This crop-only formal pipeline requires EVAL_DATA_SPLIT=test")
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
    model_path = str(_env_value(env, "MODEL_PATH", base_model))
    init_checkpoint = str(_env_value(env, "INIT_CHECKPOINT", base_model))
    init_step_match = re.search(r"(?:^|/)checkpoint-(\d+)/?$", init_checkpoint)
    init_cpt_step = int(
        _env_value(
            env,
            "INIT_CPT_STEP",
            init_step_match.group(1) if init_step_match is not None else 0,
        )
    )
    learning_rate = str(_env_value(env, "LEARNING_RATE", "2e-5"))
    ui_relation_learning_rate = str(
        _env_value(env, "UI_RELATION_LEARNING_RATE", learning_rate)
    )
    scorer_root = str(
        _env_value(
            env,
            "SCORER_ROOT",
            join_runtime_path(workspace, shared["scorer_relative_path"]),
        )
    )

    tc_msed_stage = str(_env_value(env, "TC_MSED_STAGE", "v4")).lower()
    if tc_msed_stage not in {"v4", "m1", "m2", "m3", "m4", "m5", "m31", "m32"}:
        raise ValueError("TC_MSED_STAGE must be v4/m1/m2/m3/m4/m5/m31/m32")
    tc_enabled = tc_msed_stage != "v4"
    set_enabled = tc_msed_stage in {"m2", "m3", "m4", "m5", "m31", "m32"}
    dynamic_enabled = tc_msed_stage in {"m3", "m4", "m5", "m31", "m32"}
    m31_family_enabled = tc_msed_stage in {"m31", "m32"}
    resolved: dict[str, Any] = {
        "MACHINE_TYPE": machine_type,
        "RESOURCE_GROUP": str(_env_value(env, "RESOURCE_GROUP", "default")),
        "GPU_COUNT": gpu_count,
        "CUDA_DEVICES": cuda_devices,
        "EVAL_GPU_DEVICES": eval_gpu_devices,
        "EVAL_ENABLE_PBD": int(_env_value(env, "EVAL_ENABLE_PBD", 1)),
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
        "MODEL_PATH": model_path,
        "INIT_CHECKPOINT": init_checkpoint,
        "INIT_CPT_STEP": init_cpt_step,
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
        "UI5_UI_SAMPLING_MODE": ui_sampling_mode,
        "UI_NEGATIVE_TO_POSITIVE_RATIO": ui_negative_to_positive_ratio,
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
        "SEED": int(_env_value(env, "SEED", 42)),
        "WARMUP_STEPS": int(_env_value(env, "WARMUP_STEPS", 500)),
        "LEARNING_RATE": learning_rate,
        "UI_RELATION_LEARNING_RATE": ui_relation_learning_rate,
        "WEIGHT_DECAY": float(_env_value(env, "WEIGHT_DECAY", 0.01)),
        "MAX_GRAD_NORM": float(_env_value(env, "MAX_GRAD_NORM", 1.0)),
        "LR_SCHEDULER_TYPE": str(
            _env_value(env, "LR_SCHEDULER_TYPE", "cosine")
        ).lower(),
        "BF16": int(parse_bool(_env_value(env, "BF16", "1"), name="BF16")),
        "PER_DEVICE_TRAIN_BATCH_SIZE": int(
            _env_value(env, "PER_DEVICE_TRAIN_BATCH_SIZE", 1)
        ),
        "DEEPSPEED_CONFIG": str(
            _env_value(
                env,
                "DEEPSPEED_CONFIG",
                "deepspeed_configs/zero_stage2_two_lr_config.json",
            )
        ),
        "GRADIENT_ACCUMULATION_STEPS": int(
            _env_value(
                env,
                "GRADIENT_ACCUMULATION_STEPS",
                2 if gpu_count == 4 else 1,
            )
        ),
        "RELATION_GATE_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_GATE_LOSS_WEIGHT", 0.0 if m31_family_enabled else (0.2 if tc_enabled else 1.0))
        ),
        "RELATION_SLOT_GATE_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_SLOT_GATE_LOSS_WEIGHT", 0.5 if tc_enabled else 0.1)
        ),
        "RELATION_SLOT_OBJECTNESS_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_SLOT_OBJECTNESS_LOSS_WEIGHT", 0.5 if m31_family_enabled else (0.5 if tc_enabled else 0.1))
        ),
        "RELATION_ATTENTION_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_ATTENTION_LOSS_WEIGHT", 0.2 if m31_family_enabled else (1.0 if tc_enabled else 0.1))
        ),
        "RELATION_GATE_THRESHOLD": float(
            _env_value(env, "RELATION_GATE_THRESHOLD", 0.5)
        ),
        "RELATION_GATE_MODE": str(
            _env_value(env, "RELATION_GATE_MODE", "soft" if tc_msed_stage in {"m4", "m5"} else "observe")
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
        "TC_MSED_STAGE": tc_msed_stage,
        "RELATION_BOX_L1_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_BOX_L1_LOSS_WEIGHT", 1.0 if m31_family_enabled else (5.0 if set_enabled else 0.0))
        ),
        "RELATION_BOX_GIOU_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_BOX_GIOU_LOSS_WEIGHT", 1.0 if m31_family_enabled else (2.0 if set_enabled else 0.0))
        ),
        "RELATION_COVERAGE_LOSS_WEIGHT": float(
            _env_value(env, "RELATION_COVERAGE_LOSS_WEIGHT", 0.05 if m31_family_enabled else (0.1 if dynamic_enabled else 0.0))
        ),
        "RELATION_TASK_HARD_ROUTER": int(
            _env_value(env, "RELATION_TASK_HARD_ROUTER", 1 if m31_family_enabled else 0)
        ),
        "RELATION_TASK_EXPERT_RANK": int(
            _env_value(env, "RELATION_TASK_EXPERT_RANK", 8)
        ),
        "RELATION_SET_DECODER_LAYERS": int(
            _env_value(env, "RELATION_SET_DECODER_LAYERS", 3)
        ),
        "RELATION_COORD_PRIOR_SIGMA": float(
            _env_value(env, "RELATION_COORD_PRIOR_SIGMA", 0.05)
        ),
        "RELATION_AUX_BUDGET_RATIO": float(
            _env_value(env, "RELATION_AUX_BUDGET_RATIO", 1.0)
        ),
        "SAVE_STEPS": save_steps,
        "ENABLE_EVAL": int(enable_eval),
        "EVAL_AT_START": int(eval_at_start),
        "EVAL_INTERVAL_STEPS": eval_interval,
        "EVAL_FAIL_POLICY": eval_fail_policy,
        "EVAL_VALIDATION_EARLY_STOP": int(eval_validation_early_stop),
        "EVAL_DATA_SPLIT": eval_data_split,
        "EVAL_FROZEN_GATE_THRESHOLDS": str(
            _env_value(env, "EVAL_FROZEN_GATE_THRESHOLDS", "")
        ),
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
                join_runtime_path(
                    project_root,
                    "work_dirs",
                    "ui5_eval_detector_cache_horizontal_v5",
                ),
            )
        ),
        "EVAL_DETECTOR_CACHE_MODE": str(
            _env_value(env, "EVAL_DETECTOR_CACHE_MODE", "readonly")
        ).lower(),
        "EVAL_SCAN_NAME": str(
            _env_value(
                env,
                "EVAL_SCAN_NAME",
                "horizontal_scan_v5_raw_detector_edge_aligned",
            )
        ),
        "EVAL_REQUIRE_CACHE_SCOPE": str(
            _env_value(
                env,
                "EVAL_REQUIRE_CACHE_SCOPE",
                "full_test",
            )
        ).lower(),
        "EVAL_REQUIRE_STRICT_NONOVERLAP": int(
            parse_bool(
                _env_value(env, "EVAL_REQUIRE_STRICT_NONOVERLAP", "1"),
                name="EVAL_REQUIRE_STRICT_NONOVERLAP",
            )
        ),
        "EVAL_REQUIRE_RAW_DETECTOR_EDGE_ALIGNMENT": int(
            parse_bool(
                _env_value(env, "EVAL_REQUIRE_RAW_DETECTOR_EDGE_ALIGNMENT", "1"),
                name="EVAL_REQUIRE_RAW_DETECTOR_EDGE_ALIGNMENT",
            )
        ),
        "EVAL_REQUIRE_DETECTOR_UNIQUE_CONTAINMENT": int(
            parse_bool(
                _env_value(
                    env, "EVAL_REQUIRE_DETECTOR_UNIQUE_CONTAINMENT", "1"
                ),
                name="EVAL_REQUIRE_DETECTOR_UNIQUE_CONTAINMENT",
            )
        ),
        "EVAL_EXPECTED_UNIQUE_IMAGES": int(
            _env_value(
                env,
                "EVAL_EXPECTED_UNIQUE_IMAGES",
                DEFAULT_UI5_FULL_TEST_UNIQUE_IMAGES,
            )
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
        "EVAL_SCAN_TARGET_GUARD_RATIO": float(
            _env_value(env, "EVAL_SCAN_TARGET_GUARD_RATIO", 0.0)
        ),
        "EVAL_SCAN_TARGET_GUARD_MIN_PIXELS": int(
            _env_value(env, "EVAL_SCAN_TARGET_GUARD_MIN_PIXELS", 0)
        ),
        "EVAL_SCAN_TARGET_GUARD_MAX_PIXELS": int(
            _env_value(env, "EVAL_SCAN_TARGET_GUARD_MAX_PIXELS", 0)
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
    for name in ("UI_TRAIN_PROFILE", "UI14_DATA_ROOT", "UI_TASK_REGISTRY", "UI_EVAL_MANIFEST", "UI14_CHECK_REPORT", "UI_NUM_TASKS", "LOGGING_STEPS", "SAMPLE_LOG_INTERVAL", "LOCANY_CPT_MODE"):
        if name in env: resolved[name] = env[name]
    if resolved["PIPELINE_MODE"] not in {"train", "eval"}:
        raise ValueError("PIPELINE_MODE must be 'train' or 'eval'")
    if resolved["MAX_NUM_TOKENS"] < resolved["MAX_NUM_TOKENS_PER_SAMPLE"]:
        raise ValueError(
            "MAX_NUM_TOKENS cannot be smaller than MAX_NUM_TOKENS_PER_SAMPLE; "
            "otherwise some accepted samples can never fit in a packed batch"
        )
    if resolved["GRADIENT_ACCUMULATION_STEPS"] < 1:
        raise ValueError("GRADIENT_ACCUMULATION_STEPS must be positive")
    if resolved["SEED"] != 42:
        raise ValueError("UI5 M32 initialization requires SEED=42")
    if float(resolved["LEARNING_RATE"]) <= 0.0:
        raise ValueError("LEARNING_RATE must be positive")
    if float(resolved["UI_RELATION_LEARNING_RATE"]) <= 0.0:
        raise ValueError("UI_RELATION_LEARNING_RATE must be positive")
    if resolved["WEIGHT_DECAY"] < 0.0:
        raise ValueError("WEIGHT_DECAY cannot be negative")
    if resolved["MAX_GRAD_NORM"] <= 0.0:
        raise ValueError("MAX_GRAD_NORM must be positive")
    if resolved["LR_SCHEDULER_TYPE"] != "cosine":
        raise ValueError("UI5 formal training requires LR_SCHEDULER_TYPE=cosine")
    if resolved["BF16"] != 1:
        raise ValueError("UI5 formal training requires BF16=True")
    if resolved["PER_DEVICE_TRAIN_BATCH_SIZE"] != 1:
        raise ValueError("UI5 formal training requires PER_DEVICE_TRAIN_BATCH_SIZE=1")
    if resolved["EVAL_ENABLE_PBD"] not in {0, 1}:
        raise ValueError("EVAL_ENABLE_PBD must be 0 or 1")
    if not 0.0 <= resolved["RELATION_GATE_THRESHOLD"] <= 1.0:
        raise ValueError("RELATION_GATE_THRESHOLD must be in [0, 1]")
    if resolved["RELATION_GATE_MODE"] not in {"observe", "hard", "soft"}:
        raise ValueError("RELATION_GATE_MODE must be observe, hard, or soft")

    if crop_train_mode == "crop_only" and not parse_bool(
        _env_value(env, "UI5_GPU_PARITY_PROBE", "0"),
        name="UI5_GPU_PARITY_PROBE",
    ):
        formal_exact = {
            "MACHINE_TYPE": "a800",
            "GPU_COUNT": 4,
            "MAX_SEQ_LENGTH": 7268,
            "MAX_NUM_TOKENS_PER_SAMPLE": 7268,
            "MAX_NUM_TOKENS": 12800,
            "MAX_NUM_TOKENS_SCOPE": "per_rank_packed_batch",
            "MAX_STEPS": 16000,
            "SEED": 42,
            "WARMUP_STEPS": 500,
            "GRADIENT_ACCUMULATION_STEPS": 2,
            "PER_DEVICE_TRAIN_BATCH_SIZE": 1,
            "BF16": 1,
            "SAVE_STEPS": 4000,
            "ENABLE_EVAL": 1,
            "EVAL_AT_START": 0,
            "EVAL_INTERVAL_STEPS": 1000,
            "EVAL_FAIL_POLICY": "warn",
            "EVAL_VALIDATION_EARLY_STOP": 0,
            "EVAL_DATA_SPLIT": "test",
            "EVAL_INFERENCE_CROP_MODE": "detector_scan",
            "EVAL_DETECTOR_CACHE_MODE": "readonly",
            "EVAL_SCAN_NAME": "horizontal_scan_v5_raw_detector_edge_aligned",
            "EVAL_REQUIRE_CACHE_SCOPE": "full_test",
            "EVAL_EXPECTED_UNIQUE_IMAGES": 1555,
            "EVAL_REQUIRE_STRICT_NONOVERLAP": 1,
            "EVAL_REQUIRE_RAW_DETECTOR_EDGE_ALIGNMENT": 1,
            "EVAL_REQUIRE_DETECTOR_UNIQUE_CONTAINMENT": 1,
            "EVAL_MAX_IMAGES_PER_TASK": 0,
            "TC_MSED_STAGE": "m32",
            "RELATION_GATE_MODE": "observe",
            "EVAL_ENABLE_PBD": 1,
            "INIT_CPT_STEP": 3000,
            "UI5_USE_DETECTION_CROPS": 1,
            "UI5_CROP_TRAIN_MODE": "crop_only",
            "UI5_UI_SAMPLING_MODE": "task_source_balanced_rotating",
            "INSTALL_SYSTEM_RUNTIME_DEPS": 1,
        }
        if env.get("UI_TRAIN_PROFILE") == "m32-cpt9000-ui14-v1":
            formal_exact.update(INIT_CPT_STEP=9000, EVAL_FAIL_POLICY="stop", RESOURCE_GROUP="aiai_locate", ATTN_IMPLEMENTATION="sdpa", UI_NUM_TASKS="14", LOCANY_CPT_MODE="0")
            from ui14_common import INIT_CHECKPOINT as UI14_INIT_CHECKPOINT
            formal_exact.update(BASE_MODEL=UI14_INIT_CHECKPOINT, MODEL_PATH=UI14_INIT_CHECKPOINT, INIT_CHECKPOINT=UI14_INIT_CHECKPOINT)
        drift = {
            name: {"expected": expected, "actual": resolved.get(name)}
            for name, expected in formal_exact.items()
            if resolved.get(name) != expected
        }
        formal_floats = {
            "LEARNING_RATE": 1e-5,
            "UI_RELATION_LEARNING_RATE": 2e-5,
            "WEIGHT_DECAY": 0.01,
            "MAX_GRAD_NORM": 1.0,
            "UI_NEGATIVE_TO_POSITIVE_RATIO": 2.0,
        }
        drift.update(
            {
                name: {"expected": expected, "actual": resolved.get(name)}
                for name, expected in formal_floats.items()
                if not math.isclose(
                    float(resolved.get(name)),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            }
        )
        if not str(resolved["DEEPSPEED_CONFIG"]).endswith(
            "deepspeed_configs/zero_stage2_two_lr_config.json"
        ):
            drift["DEEPSPEED_CONFIG"] = {
                "expected": "deepspeed_configs/zero_stage2_two_lr_config.json",
                "actual": resolved["DEEPSPEED_CONFIG"],
            }
        if not (
            resolved["BASE_MODEL"]
            == resolved["MODEL_PATH"]
            == resolved["INIT_CHECKPOINT"]
        ):
            drift["CPT_INITIALIZATION_CHAIN"] = {
                "expected": "BASE_MODEL=MODEL_PATH=INIT_CHECKPOINT",
                "actual": [
                    resolved["BASE_MODEL"],
                    resolved["MODEL_PATH"],
                    resolved["INIT_CHECKPOINT"],
                ],
            }
        if drift:
            raise ValueError(
                f"{env.get('UI_TRAIN_PROFILE', 'crop-only M32+CPT-3000')} formal profile drift: "
                + json.dumps(drift, ensure_ascii=False, sort_keys=True)
            )
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
    if resolved["EVAL_DETECTOR_CACHE_MODE"] != "readonly":
        raise ValueError("This formal pipeline requires EVAL_DETECTOR_CACHE_MODE=readonly")
    if resolved["EVAL_REQUIRE_CACHE_SCOPE"] != "full_test":
        raise ValueError(
            "This formal pipeline requires EVAL_REQUIRE_CACHE_SCOPE=full_test"
        )
    if resolved["EVAL_EXPECTED_UNIQUE_IMAGES"] < 0:
        raise ValueError("EVAL_EXPECTED_UNIQUE_IMAGES cannot be negative")
    if resolved["EVAL_SCAN_TARGET_HEIGHT"] <= 0:
        raise ValueError("EVAL_SCAN_TARGET_HEIGHT must be positive")
    if resolved["EVAL_INFERENCE_CROP_MODE"] == "detector_scan" and (
        resolved["EVAL_SCAN_TARGET_GUARD_RATIO"],
        resolved["EVAL_SCAN_TARGET_GUARD_MIN_PIXELS"],
        resolved["EVAL_SCAN_TARGET_GUARD_MAX_PIXELS"],
    ) != (0.0, 0, 0):
        raise ValueError("raw detector-edge evaluation requires target guard 0/0/0")
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
    if m31_family_enabled:
        required_m31 = {
            "RELATION_TASK_HARD_ROUTER": 1,
            "RELATION_TASK_EXPERT_RANK": 8,
            "RELATION_SET_DECODER_LAYERS": 3,
            "RELATION_NUM_SLOTS": 8,
            "RELATION_GATE_MODE": "observe",
            "RELATION_GATE_LOSS_WEIGHT": 0.0,
        }
        drift = {
            key: (resolved[key], expected)
            for key, expected in required_m31.items()
            if resolved[key] != expected
        }
        if drift:
            details = ", ".join(
                f"{key}={actual!r} (required {expected!r})"
                for key, (actual, expected) in sorted(drift.items())
            )
            raise ValueError(
                f"{tc_msed_stage} has a fixed P0/P1 architecture, task-expert "
                "configuration and Gate policy; "
                f"refusing configuration drift: {details}"
            )
    if not 0.0 <= resolved["RELATION_AUX_BUDGET_RATIO"] <= 1.0:
        raise ValueError("RELATION_AUX_BUDGET_RATIO must be in [0, 1]")
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
