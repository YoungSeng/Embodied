# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from contextlib import nullcontext
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from .modeling_qwen2 import Qwen2ForCausalLM
# from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import torch.utils.checkpoint as cp
from ..moon_vit.modeling_vit import MoonVitPretrainedModel
from peft import LoraConfig, get_peft_model
from transformers.generation import GenerationMixin
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from .configuration_locateanything import LocateAnythingConfig
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from eaglevl.sp_utils import  (get_pg_manager, ring_split_for_sequence_parallel)
from eaglevl.train.liger_loss_weight_ops import LigerFusedLinearCrossEntropyLoss
from .relation_modules import (
    RelationConditionedDetailPyramid,
    RelationPyramidOutput,
    RelationToPBD,
    apply_coordinate_logit_prior,
    apply_soft_gate_logit_prior,
    coordinate_bridge_prediction_groups,
    coordinate_gaussian_prior,
    pbd_active_delta_norm,
)
from .ui_relation_setup import ui_relation_collective_device


logger = logging.get_logger(__name__)


# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L241C1-L280C1
LOCATEANYTHING_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LocateAnythingConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare LocateAnything Model outputting raw hidden-states without any specific head on top.",
    LOCATEANYTHING_START_DOCSTRING,
)
class LocateAnythingPreTrainedModel(PreTrainedModel):
    config_class = LocateAnythingConfig
    base_model_prefix = "model"
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True
    
    def _init_weights(self, module):
        std = getattr(self.config, 'initializer_range', None) or self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

IGNORE_INDEX = -100


@dataclass
class UIDefectModelOutput(CausalLMOutputWithPast):
    """LocateAnything output with the stable UI-relation interface."""

    relation_tokens: Optional[torch.Tensor] = None
    relation_family: Optional[torch.Tensor] = None
    p_defect: Optional[torch.Tensor] = None
    coarse_boxes: Optional[torch.Tensor] = None
    query_attention: Optional[Tuple[torch.Tensor, ...]] = None
    box_anchor_hidden: Optional[torch.Tensor] = None
    box_anchor_samples: Optional[torch.Tensor] = None
    coordinate_logits: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    gate_loss: Optional[torch.Tensor] = None
    image_gate_loss: Optional[torch.Tensor] = None
    slot_gate_loss: Optional[torch.Tensor] = None
    slot_objectness_loss: Optional[torch.Tensor] = None
    attention_loss: Optional[torch.Tensor] = None
    per_task_gate_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_lm_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_image_gate_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_slot_gate_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_slot_objectness_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_attention_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_box_l1_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_box_giou_loss: Optional[Dict[int, torch.Tensor]] = None
    gate_targets: Optional[torch.Tensor] = None
    image_gate_targets: Optional[torch.Tensor] = None
    slot_gate_logits: Optional[torch.Tensor] = None
    slot_objectness_logits: Optional[torch.Tensor] = None
    image_gate_logits: Optional[torch.Tensor] = None
    detail_layer_weights: Optional[torch.Tensor] = None
    detail_feature_norm: Optional[torch.Tensor] = None
    detail_feature_abs_max: Optional[torch.Tensor] = None
    detail_saturation_fraction: Optional[torch.Tensor] = None
    detail_norm_ratio: Optional[torch.Tensor] = None
    detail_fused_norm: Optional[torch.Tensor] = None
    relation_context_norm: Optional[torch.Tensor] = None
    relation_gate_prob_mean: Optional[torch.Tensor] = None
    pbd_delta_norm: Optional[torch.Tensor] = None
    pbd_active_positions: Optional[torch.Tensor] = None
    box_anchor_count: Optional[torch.Tensor] = None
    pbd_to_hidden_ratio: Optional[torch.Tensor] = None
    loss_lm_contribution: Optional[torch.Tensor] = None
    loss_image_gate_contribution: Optional[torch.Tensor] = None
    loss_slot_gate_contribution: Optional[torch.Tensor] = None
    loss_attention_contribution: Optional[torch.Tensor] = None
    loss_box_l1_contribution: Optional[torch.Tensor] = None
    loss_box_giou_contribution: Optional[torch.Tensor] = None
    loss_coverage_contribution: Optional[torch.Tensor] = None
    loss_coordinate_bridge_contribution: Optional[torch.Tensor] = None
    loss_reconstructed: Optional[torch.Tensor] = None
    loss_reconstruction_error: Optional[torch.Tensor] = None
    attention_active: Optional[torch.Tensor] = None
    attention_kl_loss: Optional[torch.Tensor] = None
    attention_ce_diagnostic: Optional[torch.Tensor] = None
    box_l1_loss: Optional[torch.Tensor] = None
    box_giou_loss: Optional[torch.Tensor] = None
    coverage_loss: Optional[torch.Tensor] = None
    coordinate_bridge_loss: Optional[torch.Tensor] = None
    matched_slot_indices: Optional[torch.Tensor] = None
    scale_entropy: Optional[torch.Tensor] = None
    scale_batch_std: Optional[torch.Tensor] = None
    coarse_iou_mean: Optional[torch.Tensor] = None
    coarse_recall_03: Optional[torch.Tensor] = None
    coarse_recall_05: Optional[torch.Tensor] = None
    matched_slots: Optional[torch.Tensor] = None
    unmatched_slots: Optional[torch.Tensor] = None
    slot_usage_entropy: Optional[torch.Tensor] = None
    unique_slot_count: Optional[torch.Tensor] = None
    duplicate_slot_rate: Optional[torch.Tensor] = None
    predicted_center_diversity: Optional[torch.Tensor] = None
    attention_diversity: Optional[torch.Tensor] = None
    route_top1_match_accuracy: Optional[torch.Tensor] = None
    pre_mask_route_top1_match_accuracy: Optional[torch.Tensor] = None
    slot_usage_histogram: Optional[torch.Tensor] = None
    relation_aux_budget_scale: Optional[torch.Tensor] = None
    relation_aux_raw_contribution: Optional[torch.Tensor] = None
    relation_aux_scaled_contribution: Optional[torch.Tensor] = None
    coord_prior_lambdas: Optional[torch.Tensor] = None
    soft_gate_betas: Optional[torch.Tensor] = None
    cpt_token_losses: Optional[torch.Tensor] = None
    global_visual_cache: Optional[Any] = None


class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, GenerationMixin):
    config_class = LocateAnythingConfig
    def __init__(self, config: LocateAnythingConfig, vision_model=None, language_model=None):
        super().__init__(config)

        self.template = config.template
        self.mlp_checkpoint = config.mlp_checkpoint

        logger.info(f'mlp_checkpoint: {self.mlp_checkpoint}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'moonvit':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = MoonVitPretrainedModel(config.vision_config)
            else:
                raise ValueError(f'Unsupported vision model type: {config.vision_config.model_type}. Only moonvit is supported.')

        text_attn_impl = (
            getattr(config, '_attn_implementation', None)
            or getattr(config.text_config, '_attn_implementation', None)
            or 'magi'
        )
        config.text_config._attn_implementation = text_attn_impl

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.text_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.text_config)
            else:
                raise ValueError(f'Unsupported language model architecture: {config.text_config.architectures[0]}. Only Qwen2ForCausalLM and Qwen3ForCausalLM are supported.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        # MLP for moonvit (without pixel_shuffle_back, direct mapping)
        self.mlp1 = nn.Sequential(
                nn.LayerNorm(vit_hidden_size*4),
                nn.Linear(vit_hidden_size*4, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        self.image_token_index = config.image_token_index
        self.neftune_alpha = None

        self.enable_ui_relation = bool(getattr(config, "enable_ui_relation", True))
        if self.enable_ui_relation:
            detail_hidden_size = int(getattr(config, "relation_detail_hidden_size", 256))
            self.relation_pyramid = RelationConditionedDetailPyramid(
                num_defect_types=int(getattr(config, "ui_num_tasks", 5)),
                vision_hidden_size=vit_hidden_size,
                detail_hidden_size=detail_hidden_size,
                num_slots=int(getattr(config, "relation_num_slots", 8)),
                adapter_bottleneck=int(getattr(config, "relation_adapter_bottleneck", 64)),
                focal_gamma=float(getattr(config, "relation_focal_gamma", 2.0)),
                focal_beta=float(getattr(config, "relation_focal_beta", 0.999)),
                task_scale_router=bool(getattr(config, "relation_task_scale_router", False)),
                set_localizer=bool(getattr(config, "relation_set_localizer", False)),
                soft_gate=bool(getattr(config, "relation_soft_gate", False)),
                task_hard_router=bool(getattr(config, "relation_task_hard_router", False)),
                task_experts=bool(getattr(config, "relation_task_experts", False)),
                task_expert_rank=int(getattr(config, "relation_task_expert_rank", 8)),
                set_decoder=bool(getattr(config, "relation_set_decoder", False)),
                set_decoder_layers=int(getattr(config, "relation_set_decoder_layers", 3)),
                set_decoder_deep_supervision=bool(
                    getattr(config, "relation_set_decoder_deep_supervision", False)
                ),
                reference_position_encoding=bool(
                    getattr(config, "relation_reference_position_encoding", False)
                ),
                per_level_scale_router=bool(
                    getattr(config, "relation_per_level_scale_router", False)
                ),
            )
            self.relation_pbd = RelationToPBD(
                detail_hidden_size,
                llm_hidden_size,
                num_defect_types=int(getattr(config, "ui_num_tasks", 5)),
                dynamic_slot=bool(getattr(config, "relation_dynamic_slot_pbd", False)),
                overlap_adapter=bool(getattr(config, "relation_overlap_adapter", False)),
                coordinate_bridge=bool(getattr(config, "relation_coordinate_bridge", False)),
                task_experts=bool(getattr(config, "relation_task_experts", False)),
                task_expert_rank=int(getattr(config, "relation_task_expert_rank", 8)),
                separate_global_geometry=bool(getattr(config, "relation_set_decoder", False)),
                straight_through_hard_routing=bool(
                    getattr(config, "relation_straight_through_slot_router", False)
                ),
            )
            if str(getattr(config, "tc_msed_stage", "v4")).lower() == "m32":
                self.register_buffer(
                    "relation_aux_lm_ema", torch.zeros((), dtype=torch.float32)
                )
                self.register_buffer(
                    "relation_aux_loss_ema", torch.zeros((), dtype=torch.float32)
                )
                self.register_buffer(
                    "relation_aux_ema_initialized", torch.zeros((), dtype=torch.bool)
                )

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        self.use_llm_lora = config.use_llm_lora 
        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        # Set _no_split_modules dynamically based on the actual LLM architecture
        arch = config.text_config.architectures[0] if hasattr(config.text_config, 'architectures') and config.text_config.architectures else 'Qwen2ForCausalLM'
        if 'Qwen3' in arch:
            self._no_split_modules = ["Qwen3DecoderLayer"]
        else:
            self._no_split_modules = ["Qwen2DecoderLayer"]

    @torch.no_grad()
    def initialize_ui_relation_modules(self, seed: int, reason: str) -> dict:
        """Deterministically initialize only checkpoint-optional UI modules."""

        if not self.enable_ui_relation:
            return {"seed": int(seed), "reason": str(reason), "parameters": 0}
        std = float(
            getattr(self.config, "initializer_range", None)
            or self.config.text_config.initializer_range
        )
        relation_parameters = [
            *self.relation_pyramid.parameters(),
            *self.relation_pbd.parameters(),
        ]
        gather_context = nullcontext()
        zero_partitioned = any(
            hasattr(parameter, "ds_id") for parameter in relation_parameters
        )
        if zero_partitioned:
            import deepspeed

            gather_context = deepspeed.zero.GatheredParameters(
                relation_parameters, modifier_rank=0
            )
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with gather_context:
            should_initialize = not zero_partitioned or not dist.is_initialized() or dist.get_rank() == 0
            if should_initialize:
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(int(seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed))
                    for root in (self.relation_pyramid, self.relation_pbd):
                        for module in root.modules():
                            if isinstance(module, nn.LayerNorm):
                                nn.init.ones_(module.weight)
                                if module.bias is not None:
                                    nn.init.zeros_(module.bias)
                            elif isinstance(module, nn.Linear):
                                nn.init.normal_(module.weight, mean=0.0, std=std)
                                if module.bias is not None:
                                    nn.init.zeros_(module.bias)
                            elif isinstance(module, nn.Embedding):
                                nn.init.normal_(module.weight, mean=0.0, std=std)
                    nn.init.normal_(self.relation_pyramid.evidence_queries, mean=0.0, std=0.02)
                    nn.init.normal_(self.relation_pyramid.context_queries, mean=0.0, std=0.02)
                    self.relation_pyramid.scale_logits.copy_(
                        self.relation_pyramid.expected_family_scale_logits()
                    )
                    if self.relation_pyramid.task_scale_router:
                        nn.init.zeros_(self.relation_pyramid.task_scale_projection.weight)
                        nn.init.zeros_(self.relation_pyramid.image_scale_projection.weight)
                        if self.relation_pyramid.per_level_scale_router:
                            nn.init.zeros_(
                                self.relation_pyramid.per_level_scale_scorer[-1].weight
                            )
                    if self.relation_pyramid.set_localizer:
                        nn.init.zeros_(self.relation_pyramid.coarse_box_head[-1].weight)
                        nn.init.zeros_(self.relation_pyramid.coarse_box_head[-1].bias)
                    if self.relation_pyramid.soft_gate:
                        nn.init.zeros_(self.relation_pyramid.soft_gate_beta)
                    if self.relation_pyramid.set_decoder:
                        self.relation_pyramid.task_set_decoder.reset_parameters()
                        self.relation_pyramid.relation_semantic_experts.reset_parameters()
                    for adapter in self.relation_pyramid.family_adapters:
                        adapter.scale.fill_(0.1)
                    for head in (*self.relation_pyramid.gate_heads, *self.relation_pyramid.image_gate_heads):
                        nn.init.constant_(head[-1].bias, -2.0)
                    self.relation_pbd.semantic_scale.fill_(0.01)
                    self.relation_pbd.box_scale.fill_(0.01)
                    if self.relation_pbd.dynamic_slot:
                        self.relation_pbd.coverage_gamma.fill_(1.0)
                    if self.relation_pbd.overlap_adapter:
                        nn.init.zeros_(self.relation_pbd.overlap_adapter_up.weight)
                    if self.relation_pbd.coordinate_bridge:
                        nn.init.zeros_(self.relation_pbd.coord_prior_lambda)
                    if self.relation_pbd.task_experts:
                        self.relation_pbd.semantic_task_experts.reset_parameters()
                        self.relation_pbd.geometry_task_experts.reset_parameters()
                    self.relation_pyramid.assert_family_scale_prior()

        self.config.ui_relation_initialization_seed = int(seed)
        self.config.ui_relation_initialization_reason = str(reason)
        report = self.validate_ui_relation_parameters()
        report.update({"seed": int(seed), "reason": str(reason), "initializer_range": std})
        self.config.ui_relation_initialization_stats = dict(report)
        logger.warning("Initialized UI Relation/Gate/PBD modules: %s", report)
        return report

    @torch.no_grad()
    def validate_ui_relation_parameters(self) -> dict:
        """Validate and summarize UI parameters; never mutate learned weights."""

        if not self.enable_ui_relation:
            return {"parameters": 0, "values": 0, "checksum": 0.0}
        names = []
        values = 0
        checksum = 0.0
        square_checksum = 0.0
        for prefix, module in (("relation_pyramid", self.relation_pyramid), ("relation_pbd", self.relation_pbd)):
            for name, parameter in module.named_parameters():
                local_parameter = getattr(parameter, "ds_tensor", parameter)
                if not bool(torch.isfinite(local_parameter).all()):
                    names.append(f"{prefix}.{name}")
                tensor = local_parameter.detach().double()
                values += tensor.numel()
                checksum += float(tensor.sum().item())
                square_checksum += float(tensor.square().sum().item())
        nonfinite_count = len(names)
        zero_partitioned = any(
            hasattr(parameter, "ds_id")
            for module in (self.relation_pyramid, self.relation_pbd)
            for parameter in module.parameters()
        )
        if zero_partitioned and dist.is_available() and dist.is_initialized():
            gathered_names = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_names, names)
            names = sorted(
                {
                    name
                    for rank_names in gathered_names
                    for name in (rank_names or [])
                }
            )
            device = ui_relation_collective_device(next(self.parameters()).device)
            totals = torch.tensor(
                [float(values), checksum, square_checksum, float(nonfinite_count)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            values = int(totals[0].item())
            checksum = float(totals[1].item())
            square_checksum = float(totals[2].item())
            nonfinite_count = int(totals[3].item())
        if nonfinite_count:
            raise FloatingPointError(f"Non-finite UI relation parameters: {names}")
        return {
            "parameters": sum(1 for _ in self.relation_pyramid.parameters())
            + sum(1 for _ in self.relation_pbd.parameters()),
            "values": values,
            "checksum": checksum,
            "square_checksum": square_checksum,
        }

    @torch.no_grad()
    def assert_ui_relation_rank_consistency(self, atol: float = 1.0e-8) -> dict:
        report = self.validate_ui_relation_parameters()
        if not dist.is_available() or not dist.is_initialized():
            report["world_size"] = 1
            return report
        collective_device = ui_relation_collective_device(
            next(self.parameters()).device
        )
        local = torch.tensor(
            [report["checksum"], report["square_checksum"]],
            dtype=torch.float64,
            device=collective_device,
        )
        gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        stacked = torch.stack(gathered)
        max_diff = float((stacked - stacked[0]).abs().max().item())
        if max_diff > float(atol):
            raise RuntimeError(
                f"UI relation initialization differs across ranks: max_diff={max_diff}, checksums={stacked.cpu().tolist()}"
            )
        report.update(
            {
                "world_size": dist.get_world_size(),
                "rank_max_diff": max_diff,
                "collective_backend": str(dist.get_backend()),
                "collective_device": str(collective_device),
            }
        )
        return report

        
    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj',
                            'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
        self.use_llm_lora = True

    def get_sub_sample_lengths(self, input_ids):
        # for compatibility with packing
        sub_sample_lengths = [torch.tensor([each.shape[0]], device=input_ids.device, dtype=torch.int32) for each in input_ids]
        return sub_sample_lengths
    
    def forward(
            self,
            pixel_values: List[torch.FloatTensor],
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_grid_hws: Optional[torch.Tensor] = None,
            image_flags: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            loss_weight: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            sub_sample_lengths: Optional[List[torch.Tensor]] = None,
            relation_family: Optional[torch.LongTensor] = None,
            defect_type: Optional[torch.LongTensor] = None,
            target_boxes: Optional[torch.FloatTensor] = None,
            target_box_mask: Optional[torch.BoolTensor] = None,
            return_ui_defect_outputs: bool = False,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, UIDefectModelOutput]:
        RING_ZIGZAG = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if sub_sample_lengths is None:
            sub_sample_lengths = self.get_sub_sample_lengths(input_ids)

        model_input_ids = input_ids
        input_embeds = self.language_model.get_input_embeddings()(model_input_ids)

        has_images = image_flags is not None and image_flags.sum() > 0
        
        relation_output = None
        global_visual_cache = None
        if self.enable_ui_relation and relation_family is not None:
            if defect_type is None:
                raise ValueError("defect_type is required when relation_family is provided")
            vit_embeds, relation_output, global_visual_cache = self.extract_ui_features(
                pixel_values=pixel_values,
                image_grid_hws=image_grid_hws,
                relation_family=relation_family,
                defect_type=defect_type,
                image_flags=image_flags,
                target_boxes=target_boxes,
                target_box_mask=target_box_mask,
                return_global_visual_cache=return_ui_defect_outputs,
            )
        else:
            vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
            
        B, N, C = input_embeds.shape
        # LoRA's input-gradient hook can make embedding outputs leaf tensors.
        # Clone before indexed writes so the same path works with and without LoRA.
        input_embeds = input_embeds.reshape(B * N, C).clone()

        if has_images:
            filtered_vit_embeds = []
            idx = 0
            for flag in image_flags:
                flag_val = flag.item()
                if flag_val != 0:
                    filtered_vit_embeds.extend(vit_embeds[idx:idx + flag_val])
                    idx += flag_val
                else:
                    idx += 1

            vit_embeds = filtered_vit_embeds
            vit_embeds = torch.cat(vit_embeds, dim=0)

            vit_embeds = self.mlp1(vit_embeds)
            flat_input_ids = model_input_ids.reshape(B * N)
            selected = (flat_input_ids == self.image_token_index)
            n_token = int(selected.sum().item())
            n_embed = vit_embeds.shape[0]
            if n_embed == n_token:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
                ignore_flag = False
            else:
                print(f'warning: image token/feature mismatch, input_embeds[selected].shape={input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_assign = min(n_token, n_embed)
                if n_assign > 0:
                    selected_indices = selected.nonzero(as_tuple=False).squeeze(1)[:n_assign]
                    input_embeds[selected_indices] = input_embeds[selected_indices] * 0.0 + vit_embeds[:n_assign]
                ignore_flag = True
        else:
            ignore_flag = False
            vit_embeds = torch.cat(vit_embeds, dim=0)     
            vit_embeds = self.mlp1(vit_embeds)
            input_embeds[0] = vit_embeds.sum()
        input_embeds = input_embeds.reshape(B, N, C)


        if self.use_llm_lora:
            language_model_forward = self.language_model.model.model.forward
        else:
            language_model_forward = self.language_model.model.forward
        
        ssl = None
        ssl_tensor = None
        if sub_sample_lengths is not None:
            ssl = sub_sample_lengths[0] if isinstance(sub_sample_lengths, list) else sub_sample_lengths
            total_packed_len = int(ssl.sum().item()) if isinstance(ssl, torch.Tensor) else int(sum(ssl))
            seq_len = int(model_input_ids.shape[-1])
            if total_packed_len != seq_len:
                raise ValueError(
                    f"Packed sequence length mismatch: seq_len={seq_len}, "
                    f"sum(sub_sample_lengths)={total_packed_len}, "
                    f"sub_sample_lengths={ssl.tolist() if isinstance(ssl, torch.Tensor) else ssl}"
                )
            if len(ssl) > 1:  # Multiple samples packed together
                ssl_tensor = ssl

        outputs = language_model_forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            sub_sample_lengths=ssl_tensor,  # Pass sub_sample_lengths for stream packing
            )  
        
        # not every token needs to be computed by lm_head, we only compute the tokens that have valid labels
        decoder_hidden_states = outputs.last_hidden_state
        hidden_states = decoder_hidden_states
        lm_head_weight = self.language_model.lm_head.weight
        box_anchor_hidden = None
        box_anchor_samples = None
        coordinate_logits = None
        pbd_delta_norm = None
        pbd_active_positions = None
        box_anchor_count = None
        pbd_to_hidden_ratio = None
        coverage_loss = None
        coordinate_bridge_loss = None
        soft_gate_bridge_loss = None
        pbd_output = None

        if relation_output is not None:
            if isinstance(ssl, torch.Tensor):
                ssl_for_fusion = ssl.to(
                    device=hidden_states.device, dtype=torch.long
                )
            elif ssl is not None:
                ssl_for_fusion = torch.as_tensor(
                    ssl, device=hidden_states.device, dtype=torch.long
                )
            else:
                # The stream trainer always supplies packed lengths, while
                # unit/single-sample callers may not.  Preserve the identical
                # PBD path by treating each batch row as one complete sample.
                ssl_for_fusion = torch.full(
                    (model_input_ids.shape[0],),
                    model_input_ids.shape[1],
                    device=hidden_states.device,
                    dtype=torch.long,
                )
            text_config = self.config.text_config
            pbd_output = self.relation_pbd(
                hidden_states=hidden_states,
                input_ids=model_input_ids,
                sub_sample_lengths=ssl_for_fusion,
                relation_summary=relation_output.relation_summary,
                best_relation_token=relation_output.best_relation_token,
                box_start_token_id=int(self.config.box_start_token_id),
                text_mask_token_id=int(text_config.text_mask_token_id),
                block_size=int(text_config.block_size),
                relation_tokens=relation_output.relation_tokens,
                slot_gate_logits=(
                    None if self.relation_pyramid.set_decoder else relation_output.slot_gate_logits
                ),
                slot_objectness_logits=(
                    relation_output.slot_objectness_logits
                    if self.relation_pyramid.set_decoder
                    else None
                ),
                coarse_boxes=relation_output.coarse_boxes,
                matched_slot_indices=relation_output.matched_slot_indices,
                defect_type=defect_type,
            )
            hidden_states = pbd_output.hidden_states
            box_anchor_hidden = pbd_output.box_anchor_hidden
            box_anchor_samples = pbd_output.box_anchor_samples
            pbd_delta_norm = pbd_active_delta_norm(
                decoder_hidden_states.detach(),
                hidden_states.detach(),
                pbd_output.active_positions,
            )
            if pbd_output.active_positions.numel() > 0:
                active_hidden_norm = decoder_hidden_states.reshape(
                    -1, decoder_hidden_states.shape[-1]
                )[pbd_output.active_positions].detach().float().norm(dim=-1).mean()
                pbd_to_hidden_ratio = pbd_delta_norm / active_hidden_norm.clamp_min(1.0e-12)
            pbd_active_positions = torch.tensor(
                pbd_output.active_positions.numel(),
                device=hidden_states.device,
                dtype=torch.long,
            )
            box_anchor_count = torch.tensor(
                int((pbd_output.active_offsets == 0).sum().item())
                if pbd_output.active_offsets is not None
                else 0,
                device=hidden_states.device,
                dtype=torch.long,
            )
            coverage_loss = pbd_output.coverage_loss
            coord_start = int(self.config.coord_start_token_id)
            coord_end = int(self.config.coord_end_token_id) + 1
            if (
                self.relation_pbd.coordinate_bridge
                and pbd_output.active_positions.numel() > 0
            ):
                (
                    coord_bridge_positions,
                    coord_bridge_samples,
                    coord_bridge_offsets,
                    coord_bridge_boxes,
                ) = coordinate_bridge_prediction_groups(
                    model_input_ids,
                    ssl_for_fusion,
                    pbd_output.active_positions,
                    pbd_output.active_samples,
                    pbd_output.active_anchor_ordinals,
                    pbd_output.active_offsets,
                    pbd_output.selected_coarse_boxes,
                    coord_start,
                    coord_end - 1,
                )
                active_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])[
                    coord_bridge_positions
                ]
                base_coordinate_logits = F.linear(
                    active_hidden, lm_head_weight[coord_start:coord_end]
                )
                coordinate_prior = coordinate_gaussian_prior(
                    coord_bridge_offsets,
                    coord_bridge_samples,
                    coord_bridge_boxes,
                    defect_type,
                    self.relation_pbd.coord_prior_lambda,
                    coord_end - coord_start,
                    sigma=float(getattr(self.config, "relation_coord_prior_sigma", 0.05)),
                ).to(dtype=base_coordinate_logits.dtype)
                coordinate_logits = base_coordinate_logits + coordinate_prior
                if labels is not None:
                    flat_labels = labels.reshape(-1)
                    next_positions = coord_bridge_positions + 1
                    valid_coordinate = (
                        (coord_bridge_offsets < 4)
                        & (next_positions < flat_labels.numel())
                    )
                    safe_next = next_positions.clamp_max(max(flat_labels.numel() - 1, 0))
                    coordinate_targets = flat_labels[safe_next] - coord_start
                    valid_coordinate = valid_coordinate & (
                        (coordinate_targets >= 0)
                        & (coordinate_targets < coord_end - coord_start)
                    )
                    if bool(valid_coordinate.any()):
                        # Replace exactly the affected full-vocabulary LM CE
                        # terms.  A CE over only the coordinate sub-vocabulary
                        # has a different normalizer and would over-weight the
                        # bridge relative to the packed-token LM mean.
                        full_targets = flat_labels[safe_next][valid_coordinate].long()
                        base_full_logits = F.linear(
                            active_hidden[valid_coordinate], lm_head_weight
                        ).float()
                        enhanced_full_logits = base_full_logits.clone()
                        enhanced_full_logits[:, coord_start:coord_end] += (
                            coordinate_prior[valid_coordinate].float()
                        )
                        supervised_tokens = labels[..., 1:].ne(IGNORE_INDEX).sum().clamp_min(1)
                        base_ce = F.cross_entropy(
                            base_full_logits,
                            full_targets,
                            reduction="sum",
                        )
                        enhanced_ce = F.cross_entropy(
                            enhanced_full_logits,
                            full_targets,
                            reduction="sum",
                        )
                        coordinate_bridge_loss = (
                            enhanced_ce - base_ce
                        ) / supervised_tokens.float()
            else:
                coordinate_logits = F.linear(
                    box_anchor_hidden,
                    lm_head_weight[coord_start:coord_end],
                )
        
        hidden_dim = hidden_states.shape[-1]

        loss = None
        lm_loss = None
        image_gate_contribution = None
        slot_gate_contribution = None
        attention_contribution = None
        box_l1_contribution = None
        box_giou_contribution = None
        coverage_contribution = None
        coordinate_bridge_contribution = None
        soft_gate_contribution = None
        loss_reconstructed = None
        loss_reconstruction_error = None
        relation_aux_budget_scale = None
        relation_aux_raw_contribution = None
        relation_aux_scaled_contribution = None
        cpt_token_losses = None
        per_task_lm_loss = None
        logits = None
        if labels is not None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_hidden_states = shift_hidden_states.view(-1, hidden_dim)
            shift_labels = shift_labels.view(-1)
            valid_shift_labels = int(shift_labels.ne(IGNORE_INDEX).sum().item())
            if valid_shift_labels == 0:
                raise ValueError(
                    f"No valid shifted labels in packed batch: labels_shape={tuple(labels.shape)}, "
                    f"input_ids_shape={tuple(model_input_ids.shape)}, "
                    f"sub_sample_lengths={ssl.tolist() if 'ssl' in locals() and isinstance(ssl, torch.Tensor) else ssl if 'ssl' in locals() else None}"
                )

            # Process loss_weight: shift it like labels and flatten
            shift_loss_weight = None
            if loss_weight is not None:
                shift_loss_weight = loss_weight[..., 1:].contiguous()
                shift_loss_weight = shift_loss_weight.view(-1)

            liger_loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='mean')
            collect_token_losses = bool(
                getattr(self, "_cpt_observability_enabled", False)
            ) or str(getattr(self.config, "tc_msed_stage", "v4")).lower() in {"m31", "m32"}
            if collect_token_losses:
                lm_loss, cpt_token_losses = liger_loss_fn.forward_with_token_losses(
                    lm_head_weight, shift_hidden_states, shift_labels
                )
            else:
                lm_loss = liger_loss_fn(lm_head_weight, shift_hidden_states, shift_labels)
            if (
                str(getattr(self.config, "tc_msed_stage", "v4")).lower() in {"m31", "m32"}
                and cpt_token_losses is not None
            ):
                sample_ids = torch.repeat_interleave(
                    torch.arange(
                        ssl_for_fusion.numel(),
                        device=ssl_for_fusion.device,
                        dtype=torch.long,
                    ),
                    ssl_for_fusion.reshape(-1).long(),
                )
                if sample_ids.numel() != model_input_ids.numel():
                    raise RuntimeError(
                        "packed sample lengths do not cover the model input: "
                        f"length_sum={sample_ids.numel()}, "
                        f"input_tokens={model_input_ids.numel()}"
                    )
                shifted_sample_ids = sample_ids.reshape_as(model_input_ids)[
                    ..., 1:
                ].reshape(-1)
                if shifted_sample_ids.numel() != shift_labels.numel():
                    raise RuntimeError(
                        "packed sample/token mapping disagrees with shifted LM labels: "
                        f"sample_ids={shifted_sample_ids.numel()}, "
                        f"labels={shift_labels.numel()}"
                    )
                valid_lm = shift_labels.ne(IGNORE_INDEX)
                per_task_lm_loss = {}
                for task_id in range(self.relation_pyramid.num_defect_types):
                    selected = valid_lm & (
                        defect_type[shifted_sample_ids].long() == task_id
                    )
                    if bool(selected.any()):
                        per_task_lm_loss[task_id] = cpt_token_losses[
                            selected
                        ].float().mean()
            loss = lm_loss
            m32_stage = (
                str(getattr(self.config, "tc_msed_stage", "v4")).lower() == "m32"
            )
            if relation_output is not None:
                if relation_output.image_gate_loss is not None:
                    image_gate_contribution = (
                        float(self.config.relation_gate_loss_weight)
                        * relation_output.image_gate_loss
                    )
                    loss = loss + image_gate_contribution
                if relation_output.slot_gate_loss is not None:
                    slot_gate_contribution = (
                        float(getattr(
                            self.config,
                            "relation_slot_objectness_loss_weight",
                            getattr(self.config, "relation_slot_gate_loss_weight", 0.1),
                        ))
                        * relation_output.slot_gate_loss
                    )
                    if not m32_stage:
                        loss = loss + slot_gate_contribution
                if relation_output.attention_loss is not None:
                    attention_contribution = (
                        float(self.config.relation_attention_loss_weight)
                        * relation_output.attention_loss
                    )
                    if not m32_stage:
                        loss = loss + attention_contribution
                if relation_output.box_l1_loss is not None:
                    box_l1_contribution = (
                        float(getattr(self.config, "relation_box_l1_loss_weight", 0.0))
                        * relation_output.box_l1_loss
                    )
                    if not m32_stage:
                        loss = loss + box_l1_contribution
                if relation_output.box_giou_loss is not None:
                    box_giou_contribution = (
                        float(getattr(self.config, "relation_box_giou_loss_weight", 0.0))
                        * relation_output.box_giou_loss
                    )
                    if not m32_stage:
                        loss = loss + box_giou_contribution
                if coverage_loss is not None:
                    coverage_contribution = (
                        float(getattr(self.config, "relation_coverage_loss_weight", 0.0))
                        * coverage_loss
                    )
                    if not m32_stage:
                        loss = loss + coverage_contribution
                if m32_stage:
                    zero_aux = lm_loss.new_zeros(())
                    raw_parts = (
                        slot_gate_contribution,
                        attention_contribution,
                        box_l1_contribution,
                        box_giou_contribution,
                        coverage_contribution,
                    )
                    relation_aux_raw_contribution = sum(
                        (part if part is not None else zero_aux)
                        for part in raw_parts
                    )
                    with torch.no_grad():
                        current_lm = lm_loss.detach().float().abs()
                        current_aux = relation_aux_raw_contribution.detach().float().abs()
                        if not bool(self.relation_aux_ema_initialized.item()):
                            self.relation_aux_lm_ema.copy_(current_lm)
                            self.relation_aux_loss_ema.copy_(current_aux)
                            self.relation_aux_ema_initialized.fill_(True)
                        else:
                            decay = 0.95
                            self.relation_aux_lm_ema.mul_(decay).add_(
                                current_lm, alpha=1.0 - decay
                            )
                            self.relation_aux_loss_ema.mul_(decay).add_(
                                current_aux, alpha=1.0 - decay
                            )
                        budget = float(
                            getattr(self.config, "relation_aux_budget_ratio", 1.0)
                        ) * self.relation_aux_lm_ema
                        current_budget = float(
                            getattr(self.config, "relation_aux_budget_ratio", 1.0)
                        ) * current_lm
                        relation_aux_budget_scale = torch.minimum(
                            torch.minimum(
                                torch.ones_like(budget),
                                budget
                                / self.relation_aux_loss_ema.clamp_min(1.0e-12),
                            ),
                            current_budget / current_aux.clamp_min(1.0e-12),
                        ).detach()
                    slot_gate_contribution = (
                        zero_aux if slot_gate_contribution is None
                        else relation_aux_budget_scale * slot_gate_contribution
                    )
                    attention_contribution = (
                        zero_aux if attention_contribution is None
                        else relation_aux_budget_scale * attention_contribution
                    )
                    box_l1_contribution = (
                        zero_aux if box_l1_contribution is None
                        else relation_aux_budget_scale * box_l1_contribution
                    )
                    box_giou_contribution = (
                        zero_aux if box_giou_contribution is None
                        else relation_aux_budget_scale * box_giou_contribution
                    )
                    coverage_contribution = (
                        zero_aux if coverage_contribution is None
                        else relation_aux_budget_scale * coverage_contribution
                    )
                    relation_aux_scaled_contribution = (
                        slot_gate_contribution
                        + attention_contribution
                        + box_l1_contribution
                        + box_giou_contribution
                        + coverage_contribution
                    )
                    loss = loss + relation_aux_scaled_contribution
                if coordinate_bridge_loss is not None:
                    coordinate_bridge_contribution = coordinate_bridge_loss
                    loss = loss + coordinate_bridge_contribution
                if (
                    self.relation_pyramid.soft_gate
                    and relation_output.p_defect is not None
                ):
                    flat_hidden_for_gate = hidden_states[..., :-1, :].contiguous().view(
                        -1, hidden_dim
                    )
                    flat_gate_labels = labels[..., 1:].contiguous().view(-1)
                    box_id = int(self.config.box_start_token_id)
                    none_id = int(self.config.none_token_id)
                    candidate_gate_positions = (
                        (flat_gate_labels == box_id) | (flat_gate_labels == none_id)
                    )
                    lengths = ssl_for_fusion.reshape(-1).long()
                    sample_ids = torch.repeat_interleave(
                        torch.arange(lengths.numel(), device=lengths.device), lengths
                    )
                    shifted_sample_ids = sample_ids[1:]
                    if shifted_sample_ids.numel() > flat_gate_labels.numel():
                        shifted_sample_ids = shifted_sample_ids[: flat_gate_labels.numel()]
                    elif shifted_sample_ids.numel() < flat_gate_labels.numel():
                        shifted_sample_ids = F.pad(
                            shifted_sample_ids,
                            (0, flat_gate_labels.numel() - shifted_sample_ids.numel()),
                            value=int(lengths.numel() - 1),
                        )
                    # Image defectness is one decision per packed sample.  Bias
                    # only its first <box>/none target; applying it to every
                    # later box anchor would train a box-count prior while
                    # generation uses an image-level prior.
                    gate_positions = torch.zeros_like(candidate_gate_positions)
                    for sample_index in range(lengths.numel()):
                        candidates = (
                            candidate_gate_positions
                            & (shifted_sample_ids == sample_index)
                        ).nonzero(as_tuple=False).flatten()
                        if candidates.numel() > 0:
                            gate_positions[candidates[0]] = True
                    if bool(gate_positions.any()):
                        base_gate_logits = F.linear(
                            flat_hidden_for_gate[gate_positions], lm_head_weight
                        ).float()
                        selected_samples = shifted_sample_ids[gate_positions]
                        selected_tasks = defect_type[selected_samples].long().clamp(
                            0, self.relation_pyramid.soft_gate_beta.numel() - 1
                        )
                        betas = self.relation_pyramid.soft_gate_beta[selected_tasks].float()
                        probabilities = relation_output.p_defect[selected_samples].float()
                        enhanced_gate_logits = base_gate_logits.clone()
                        enhanced_gate_logits[:, box_id] += betas * probabilities
                        enhanced_gate_logits[:, none_id] += betas * (1.0 - probabilities)
                        gate_targets = flat_gate_labels[gate_positions].long()
                        base_gate_ce = F.cross_entropy(
                            base_gate_logits, gate_targets, reduction="sum"
                        )
                        enhanced_gate_ce = F.cross_entropy(
                            enhanced_gate_logits, gate_targets, reduction="sum"
                        )
                        soft_gate_bridge_loss = (
                            enhanced_gate_ce - base_gate_ce
                        ) / float(valid_shift_labels)
                        soft_gate_contribution = soft_gate_bridge_loss
                        loss = loss + soft_gate_contribution
            zero = lm_loss.new_zeros(())
            image_gate_contribution = image_gate_contribution if image_gate_contribution is not None else zero
            slot_gate_contribution = slot_gate_contribution if slot_gate_contribution is not None else zero
            attention_contribution = attention_contribution if attention_contribution is not None else zero
            box_l1_contribution = box_l1_contribution if box_l1_contribution is not None else zero
            box_giou_contribution = box_giou_contribution if box_giou_contribution is not None else zero
            coverage_contribution = coverage_contribution if coverage_contribution is not None else zero
            coordinate_bridge_contribution = coordinate_bridge_contribution if coordinate_bridge_contribution is not None else zero
            soft_gate_contribution = soft_gate_contribution if soft_gate_contribution is not None else zero
            loss_reconstructed = (
                lm_loss
                + image_gate_contribution
                + slot_gate_contribution
                + attention_contribution
                + box_l1_contribution
                + box_giou_contribution
                + coverage_contribution
                + coordinate_bridge_contribution
                + soft_gate_contribution
            )
            loss_reconstruction_error = (loss - loss_reconstructed).abs()
            if float(loss_reconstruction_error.detach().float().item()) >= 1.0e-4:
                raise FloatingPointError(
                    "UI5 loss decomposition mismatch: "
                    f"loss={loss.detach().float().item()}, "
                    f"reconstructed={loss_reconstructed.detach().float().item()}"
                )
            if not torch.isfinite(loss):
                gate_loss_value = (
                    relation_output.gate_loss.detach().float().item()
                    if relation_output is not None
                    and relation_output.gate_loss is not None
                    else None
                )
                attention_loss_value = (
                    relation_output.attention_loss.detach().float().item()
                    if relation_output is not None
                    and relation_output.attention_loss is not None
                    else None
                )
                raise FloatingPointError(
                    f"Non-finite loss detected before backward: loss={loss.detach().float().item()}, "
                    f"lm_loss={lm_loss.detach().float().item()}, "
                    f"gate_loss={gate_loss_value}, attention_loss={attention_loss_value}, "
                    f"valid_shift_labels={valid_shift_labels}, "
                    f"decoder_hidden_has_nan={bool(torch.isnan(decoder_hidden_states).any().item())}, "
                    f"hidden_states_has_nan={bool(torch.isnan(shift_hidden_states).any().item())}, "
                    f"hidden_states_has_inf={bool(torch.isinf(shift_hidden_states).any().item())}, "
                    f"relation_tokens_has_nan={bool(torch.isnan(relation_output.relation_tokens).any().item()) if relation_output is not None else None}, "
                    f"p_defect={relation_output.p_defect.detach().float().tolist() if relation_output is not None else None}"
                )
        else:
            logits = F.linear(hidden_states, lm_head_weight)

        if ignore_flag and loss is not None:
            loss = loss * 0.0
            lm_loss = lm_loss * 0.0
            image_gate_contribution = image_gate_contribution * 0.0
            slot_gate_contribution = slot_gate_contribution * 0.0
            attention_contribution = attention_contribution * 0.0
            loss_reconstructed = loss_reconstructed * 0.0
            loss_reconstruction_error = loss_reconstruction_error * 0.0
            if cpt_token_losses is not None:
                cpt_token_losses = cpt_token_losses * 0.0
            if per_task_lm_loss is not None:
                per_task_lm_loss = {
                    task_id: value * 0.0
                    for task_id, value in per_task_lm_loss.items()
                }
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return UIDefectModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            relation_tokens=relation_output.relation_tokens if relation_output is not None else None,
            relation_family=relation_output.relation_family if relation_output is not None else relation_family,
            p_defect=relation_output.p_defect if relation_output is not None else None,
            coarse_boxes=relation_output.coarse_boxes if relation_output is not None else None,
            query_attention=relation_output.query_attention if relation_output is not None and return_ui_defect_outputs else None,
            box_anchor_hidden=box_anchor_hidden,
            box_anchor_samples=box_anchor_samples,
            coordinate_logits=coordinate_logits,
            lm_loss=lm_loss,
            gate_loss=relation_output.gate_loss if relation_output is not None else None,
            image_gate_loss=relation_output.image_gate_loss if relation_output is not None else None,
            slot_gate_loss=relation_output.slot_gate_loss if relation_output is not None else None,
            slot_objectness_loss=(
                relation_output.slot_gate_loss if relation_output is not None else None
            ),
            attention_loss=relation_output.attention_loss if relation_output is not None else None,
            attention_kl_loss=relation_output.attention_kl_loss if relation_output is not None else None,
            attention_ce_diagnostic=relation_output.attention_ce_diagnostic if relation_output is not None else None,
            box_l1_loss=relation_output.box_l1_loss if relation_output is not None else None,
            box_giou_loss=relation_output.box_giou_loss if relation_output is not None else None,
            coverage_loss=coverage_loss,
            coordinate_bridge_loss=coordinate_bridge_loss,
            matched_slot_indices=relation_output.matched_slot_indices if relation_output is not None else None,
            scale_entropy=relation_output.scale_entropy if relation_output is not None else None,
            scale_batch_std=relation_output.scale_batch_std if relation_output is not None else None,
            coarse_iou_mean=relation_output.coarse_iou_mean if relation_output is not None else None,
            coarse_recall_03=relation_output.coarse_recall_03 if relation_output is not None else None,
            coarse_recall_05=relation_output.coarse_recall_05 if relation_output is not None else None,
            matched_slots=relation_output.matched_slots if relation_output is not None else None,
            unmatched_slots=relation_output.unmatched_slots if relation_output is not None else None,
            slot_usage_entropy=relation_output.slot_usage_entropy if relation_output is not None else None,
            unique_slot_count=pbd_output.unique_slot_count if pbd_output is not None else None,
            duplicate_slot_rate=pbd_output.duplicate_slot_rate if pbd_output is not None else None,
            predicted_center_diversity=(
                relation_output.predicted_center_diversity
                if relation_output is not None else None
            ),
            attention_diversity=(
                relation_output.attention_diversity
                if relation_output is not None else None
            ),
            route_top1_match_accuracy=(
                pbd_output.route_top1_match_accuracy if pbd_output is not None else None
            ),
            pre_mask_route_top1_match_accuracy=(
                pbd_output.pre_mask_route_top1_match_accuracy
                if pbd_output is not None else None
            ),
            slot_usage_histogram=(
                pbd_output.slot_usage_histogram if pbd_output is not None else None
            ),
            relation_aux_budget_scale=relation_aux_budget_scale,
            relation_aux_raw_contribution=relation_aux_raw_contribution,
            relation_aux_scaled_contribution=relation_aux_scaled_contribution,
            coord_prior_lambdas=(self.relation_pbd.coord_prior_lambda if self.enable_ui_relation and self.relation_pbd.coordinate_bridge else None),
            soft_gate_betas=(self.relation_pyramid.soft_gate_beta if self.enable_ui_relation and self.relation_pyramid.soft_gate else None),
            per_task_gate_loss=(
                relation_output.per_task_gate_loss if relation_output is not None else None
            ),
            per_task_lm_loss=per_task_lm_loss,
            per_task_image_gate_loss=(
                relation_output.per_task_image_gate_loss if relation_output is not None else None
            ),
            per_task_slot_gate_loss=(
                relation_output.per_task_slot_gate_loss if relation_output is not None else None
            ),
            per_task_slot_objectness_loss=(
                relation_output.per_task_slot_objectness_loss
                if relation_output is not None
                else None
            ),
            per_task_attention_loss=(
                relation_output.per_task_attention_loss if relation_output is not None else None
            ),
            per_task_box_l1_loss=(
                relation_output.per_task_box_l1_loss if relation_output is not None else None
            ),
            per_task_box_giou_loss=(
                relation_output.per_task_box_giou_loss if relation_output is not None else None
            ),
            gate_targets=(
                relation_output.gate_targets if relation_output is not None else None
            ),
            image_gate_targets=(
                relation_output.image_gate_targets if relation_output is not None else None
            ),
            slot_gate_logits=(
                relation_output.slot_gate_logits if relation_output is not None else None
            ),
            slot_objectness_logits=(
                relation_output.slot_objectness_logits if relation_output is not None else None
            ),
            image_gate_logits=(
                relation_output.image_gate_logits if relation_output is not None else None
            ),
            detail_layer_weights=(
                relation_output.scale_weights if relation_output is not None else None
            ),
            detail_feature_norm=(
                relation_output.projected_level_norms if relation_output is not None else None
            ),
            detail_feature_abs_max=(
                relation_output.projected_level_abs_max if relation_output is not None else None
            ),
            detail_saturation_fraction=(
                relation_output.projected_level_saturation_fraction if relation_output is not None else None
            ),
            detail_norm_ratio=(
                relation_output.projected_level_norm_ratio if relation_output is not None else None
            ),
            detail_fused_norm=(
                relation_output.fused_feature_norm if relation_output is not None else None
            ),
            relation_context_norm=(
                relation_output.relation_context_norm if relation_output is not None else None
            ),
            relation_gate_prob_mean=(
                torch.sigmoid(relation_output.image_gate_logits.detach()).float().mean()
                if relation_output is not None
                else None
            ),
            pbd_delta_norm=pbd_delta_norm,
            pbd_active_positions=pbd_active_positions,
            box_anchor_count=box_anchor_count,
            pbd_to_hidden_ratio=pbd_to_hidden_ratio,
            loss_lm_contribution=lm_loss,
            loss_image_gate_contribution=image_gate_contribution,
            loss_slot_gate_contribution=slot_gate_contribution,
            loss_attention_contribution=attention_contribution,
            loss_box_l1_contribution=box_l1_contribution,
            loss_box_giou_contribution=box_giou_contribution,
            loss_coverage_contribution=coverage_contribution,
            loss_coordinate_bridge_contribution=coordinate_bridge_contribution,
            loss_reconstructed=loss_reconstructed,
            loss_reconstruction_error=loss_reconstruction_error,
            attention_active=(
                torch.tensor(
                    float(relation_output is not None and relation_output.attention_loss is not None),
                    device=hidden_states.device,
                )
                if labels is not None
                else None
            ),
            cpt_token_losses=cpt_token_losses,
            global_visual_cache=global_visual_cache,
        )

    
    def extract_feature(self, pixel_values, image_grid_hws):
        vit_embeds = self.vision_model(pixel_values=pixel_values, grid_hws=image_grid_hws)

        return vit_embeds

    def extract_ui_features(
        self,
        pixel_values,
        image_grid_hws,
        relation_family,
        defect_type,
        image_flags=None,
        target_boxes=None,
        target_box_mask=None,
        return_global_visual_cache=False,
    ):
        vit_embeds, detail_features = self.vision_model(
            pixel_values=pixel_values,
            grid_hws=image_grid_hws,
            output_detail_features=True,
            detail_layer_indices=self.config.relation_detail_layers,
        )
        relation_output = self.relation_pyramid(
            pyramid_features=detail_features,
            grid_hws=image_grid_hws,
            relation_family=relation_family,
            defect_type=defect_type,
            image_flags=image_flags,
            target_boxes=target_boxes,
            target_box_mask=target_box_mask,
        )
        global_visual_cache = None
        if return_global_visual_cache:
            global_visual_cache = {
                "merged_visual_features": vit_embeds,
                "detail_features": detail_features,
                "image_grid_hws": image_grid_hws,
            }
        return vit_embeds, relation_output, global_visual_cache

    def ui_relation_parameter_report(self) -> dict:
        relation_parameters = 0
        if self.enable_ui_relation:
            relation_parameters = sum(
                parameter.numel()
                for module in (self.relation_pyramid, self.relation_pbd)
                for parameter in module.parameters()
            )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        return {
            "relation_parameters": relation_parameters,
            "total_parameters": total_parameters,
            "relation_percent": 100.0 * relation_parameters / max(total_parameters, 1),
            "within_five_percent": relation_parameters <= 0.05 * total_parameters,
        }

    @torch.no_grad()
    def repair_nonfinite_ui_relation_parameters(self, absolute_limit: float = 1.0e4) -> dict:
        """Backward-compatible validator; invalid tensors are never repaired silently."""

        warnings.warn(
            "repair_nonfinite_ui_relation_parameters is now validation-only; "
            "use initialize_ui_relation_modules for an all-missing base checkpoint",
            DeprecationWarning,
            stacklevel=2,
        )
        report = self.validate_ui_relation_parameters()
        report.update({"parameters_repaired": [], "values_repaired": 0})
        return report

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            image_grid_hws: Optional[torch.Tensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Convert numpy array to tensor if needed
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)
                    
        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
        
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        if image_grid_hws is not None:
            vit_embeds = torch.cat(vit_embeds, dim=0)
            vit_embeds = self.mlp1(vit_embeds)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.image_token_index)
            input_embeds[selected] = vit_embeds
            
        input_embeds = input_embeds.reshape(B, N, C)
        
        if 'use_cache' not in generate_kwargs:
            generate_kwargs['use_cache'] = True
            
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )

        return outputs

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.language_model.get_decoder()
