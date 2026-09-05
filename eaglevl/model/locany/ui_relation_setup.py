"""Shared UI5 model configuration and checkpoint-loading policy.

Training and checkpoint-0 export must pass through this module so their MTP,
attention, token IDs, Relation/Gate/PBD structure, and initialization decision
cannot drift independently.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributed as dist


def audit_ui5_detail_scale_weights(
    weights: torch.Tensor,
    *,
    global_step: int,
    load_state: str,
    resuming_from_checkpoint: bool,
) -> dict[str, Any]:
    """Check the simplex without mistaking a new run for a newly created model.

    ``load_state`` comes from this process's loading report, not a checkpoint's
    persisted initialization metadata. Learned weights are never reset here.
    A checkpoint restore takes precedence even when its trainer step is zero.
    """
    values = weights.detach().float()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
        raise RuntimeError(f"Detail Pyramid scale weights must have nonempty [N, 3] shape: {list(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Detail Pyramid scale weights are non-finite")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise RuntimeError("Detail Pyramid scale weights must lie in [0, 1]")
    weight_sums = values.sum(dim=-1)
    if not bool(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1.0e-5, rtol=0.0)):
        raise RuntimeError(f"Detail Pyramid scale weights do not sum to one: {weight_sums.cpu().tolist()}")
    expect_thirds = (
        load_state in {"all_missing", "new_model"}
        and int(global_step) == 0
        and not resuming_from_checkpoint
    )
    if expect_thirds and not bool(torch.allclose(
        values, torch.full_like(values, 1.0 / 3.0), atol=1.0e-4, rtol=0.0,
    )):
        raise RuntimeError(
            "Freshly initialized Detail Pyramid scale weights are not thirds: "
            f"{values.cpu().tolist()}"
        )
    return {
        "global_step": int(global_step),
        "ui_relation_load_state": str(load_state),
        "resuming_from_checkpoint": bool(resuming_from_checkpoint),
        "initial_thirds_required": expect_thirds,
        "scale_weights_valid": True,
        "scale_weight_max_deviation_from_thirds": float((values - 1.0 / 3.0).abs().max().item()),
    }


def ui_relation_collective_device(
    parameter_device: torch.device | str | None = None,
) -> torch.device:
    """Choose a tensor device supported by the active process-group backend.

    UI relation modules are audited before Trainer/DeepSpeed moves the model to
    CUDA.  Consequently, ``next(model.parameters()).device`` is still CPU at
    that point.  NCCL collectives cannot consume CPU tensors, so they must use
    the CUDA device selected for the current torchrun rank.  CPU-capable
    backends keep the audit tensor on CPU.
    """

    fallback = torch.device(parameter_device or "cpu")
    if not dist.is_available() or not dist.is_initialized():
        return fallback
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "NCCL UI relation consistency audit requires an available CUDA device"
            )
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def configure_ui5_model_config(
    config,
    *,
    attn_implementation: str,
    image_token_index: int,
    block_size: int,
    causal_attn: bool,
    text_mask_token_id: int,
    null_token_id: int,
    box_start_token_id: int,
    box_end_token_id: int,
    coord_start_token_id: int,
    coord_end_token_id: int,
    ref_start_token_id: int,
    ref_end_token_id: int,
    none_token_id: int,
    enable_ui_relation: bool,
    relation_detail_hidden_size: int,
    relation_num_slots: int,
    relation_adapter_bottleneck: int,
    relation_detail_layers: Sequence[int] | None,
    relation_gate_loss_weight: float,
    relation_slot_gate_loss_weight: float,
    relation_attention_loss_weight: float,
    relation_gate_threshold: float,
    relation_focal_beta: float,
    relation_focal_gamma: float,
):
    """Apply the one authoritative UI5 configuration to a model config."""

    config._attn_implementation = str(attn_implementation)
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = str(attn_implementation)
    config.text_config._attn_implementation_autoset = False
    config.vision_config._attn_implementation = "flash_attention_2"
    config.vision_config._attn_implementation_autoset = False

    config.image_token_index = int(image_token_index)
    config.text_config.block_size = int(block_size)
    config.text_config.causal_attn = bool(causal_attn)
    config.text_config.text_mask_token_id = int(text_mask_token_id)
    config.text_config.null_token_id = int(null_token_id)
    config.box_start_token_id = int(box_start_token_id)
    config.box_end_token_id = int(box_end_token_id)
    config.coord_start_token_id = int(coord_start_token_id)
    config.coord_end_token_id = int(coord_end_token_id)
    config.ref_start_token_id = int(ref_start_token_id)
    config.ref_end_token_id = int(ref_end_token_id)
    config.none_token_id = int(none_token_id)

    config.enable_ui_relation = bool(enable_ui_relation)
    config.relation_detail_hidden_size = int(relation_detail_hidden_size)
    config.relation_num_slots = int(relation_num_slots)
    config.relation_adapter_bottleneck = int(relation_adapter_bottleneck)
    if relation_detail_layers is not None:
        layers = [int(value) for value in relation_detail_layers]
        if len(layers) != 3:
            raise ValueError("UI5 relation_detail_layers must contain exactly 3 indices")
        config.relation_detail_layers = layers
    config.relation_gate_loss_weight = float(relation_gate_loss_weight)
    config.relation_slot_gate_loss_weight = float(relation_slot_gate_loss_weight)
    config.relation_attention_loss_weight = float(relation_attention_loss_weight)
    config.relation_gate_threshold = float(relation_gate_threshold)
    config.relation_gate_mode = "observe"
    if not hasattr(config, "relation_gate_thresholds"):
        config.relation_gate_thresholds = {}
    config.relation_focal_beta = float(relation_focal_beta)
    config.relation_focal_gamma = float(relation_focal_gamma)
    return config


def ui_relation_loading_state(model, loading_info: dict[str, Any]) -> dict[str, Any]:
    """Classify Relation/Gate/PBD loading as all-missing, complete, or partial."""

    expected = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("relation_pyramid.") or name.startswith("relation_pbd.")
    }
    reported_missing = set(loading_info.get("missing_keys", ()))
    missing = {
        expected_name
        for expected_name in expected
        if any(
            key == expected_name or key.endswith(expected_name)
            for key in reported_missing
        )
    }
    unexpected = sorted(
        key
        for key in loading_info.get("unexpected_keys", ())
        if "relation_pyramid" in key or "relation_pbd" in key
    )
    if unexpected:
        state = "unexpected"
    elif not expected:
        state = "disabled"
    elif missing == expected:
        state = "all_missing"
    elif missing:
        state = "partial"
    else:
        state = "complete"
    return {
        "state": state,
        "expected": expected,
        "missing": missing,
        "unexpected": unexpected,
    }


def initialize_or_validate_ui_relation(
    model,
    loading_info: dict[str, Any],
    *,
    seed: int,
    all_missing_reason: str,
) -> dict[str, Any]:
    """Enforce the single all/none checkpoint policy and return its report."""

    status = ui_relation_loading_state(model, loading_info)
    if status["state"] == "unexpected":
        raise RuntimeError(
            "Unexpected UI Relation/Gate/PBD checkpoint keys: "
            f"{status['unexpected']}"
        )
    if status["state"] == "partial":
        raise RuntimeError(
            "Partial UI Relation/Gate/PBD checkpoint is forbidden; "
            f"missing={sorted(status['missing'])}"
        )
    if status["state"] == "all_missing":
        initialization = model.initialize_ui_relation_modules(seed, all_missing_reason)
    elif status["state"] == "complete":
        initialization = model.validate_ui_relation_parameters()
    else:
        initialization = {"parameters": 0, "values": 0, "checksum": 0.0}
    return {
        "state": status["state"],
        "expected_key_count": len(status["expected"]),
        "missing_key_count": len(status["missing"]),
        "initialization": initialization,
    }
