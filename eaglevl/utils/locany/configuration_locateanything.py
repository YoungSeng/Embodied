# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from .relation_modules import DEFAULT_UI_DETAIL_LAYERS
logger = logging.get_logger(__name__)

class MoonViTConfig(PretrainedConfig):
    model_type = "moonvit"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # Positional embedding config
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        # Transformer config
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Patch merger config
        self.merge_kernel_size = merge_kernel_size


class LocateAnythingConfig(PretrainedConfig):
    model_type = 'locateanything'
    is_composition = True
    sub_configs = {"vision_config": MoonViTConfig, "text_config": Qwen2Config}
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            downsample_ratio=0.5,
            template=None,
            loss_version='v1',
            mlp_checkpoint=False,
            image_token_index=151667,
            box_start_token_id=151668,
            box_end_token_id=151669,
            coord_start_token_id=151677,
            coord_end_token_id=152677,
            ref_start_token_id=151672,
            ref_end_token_id=151673,
            none_token_id=4064,
            enable_ui_relation=True,
            relation_detail_hidden_size=256,
            relation_num_slots=8,
            relation_adapter_bottleneck=64,
            relation_detail_layers=None,
            relation_gate_loss_weight=1.0,
            relation_slot_gate_loss_weight=0.1,
            relation_slot_objectness_loss_weight=None,
            relation_attention_loss_weight=0.1,
            relation_focal_gamma=2.0,
            relation_focal_beta=0.999,
            relation_gate_threshold=0.5,
            relation_gate_mode="observe",
            relation_gate_thresholds=None,
            tc_msed_stage="v4",
            relation_task_scale_router=False,
            relation_set_localizer=False,
            relation_dynamic_slot_pbd=False,
            relation_coordinate_bridge=False,
            relation_soft_gate=False,
            relation_overlap_adapter=False,
            relation_task_hard_router=False,
            relation_task_experts=False,
            relation_task_expert_rank=8,
            relation_set_decoder=False,
            relation_set_decoder_layers=3,
            relation_box_l1_loss_weight=0.0,
            relation_box_giou_loss_weight=0.0,
            relation_coverage_loss_weight=0.0,
            relation_coord_prior_sigma=0.05,
            ui_relation_initialization_seed=20260823,
            ui_relation_initialization_reason=None,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'model_type': 'moonvit'}
            logger.info('vision_config is None. Initializing the MoonViTConfig with default values.')

        if text_config is None:
            text_config = {'architectures': ['Qwen2ForCausalLM']}
            logger.info('text_config is None. Initializing the Qwen2Config config with default values.')

        if vision_config['model_type'] == 'moonvit':
            self.vision_config = MoonViTConfig(**vision_config)
        else:
            raise ValueError('Unsupported model_type: {}. Only moonvit is supported.'.format(vision_config['model_type']))


        if text_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.text_config = Qwen2Config(**text_config)
        elif text_config['architectures'][0] == 'Qwen3ForCausalLM':
            self.text_config = Qwen3Config(**text_config)
        else:
            raise ValueError('Unsupported architecture: {}. Only Qwen2ForCausalLM and Qwen3ForCausalLM are supported.'.format(text_config['architectures'][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.mlp_checkpoint = mlp_checkpoint
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.loss_version = loss_version
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.image_token_index = image_token_index
        self.box_start_token_id = box_start_token_id
        self.box_end_token_id = box_end_token_id
        self.coord_start_token_id = coord_start_token_id
        self.coord_end_token_id = coord_end_token_id
        self.ref_start_token_id = ref_start_token_id
        self.ref_end_token_id = ref_end_token_id
        self.none_token_id = none_token_id
        self.enable_ui_relation = enable_ui_relation
        self.relation_detail_hidden_size = relation_detail_hidden_size
        self.relation_num_slots = relation_num_slots
        self.relation_adapter_bottleneck = relation_adapter_bottleneck
        if relation_detail_layers is None:
            relation_detail_layers = DEFAULT_UI_DETAIL_LAYERS
        if len(relation_detail_layers) != 3:
            raise ValueError("relation_detail_layers must contain early, middle, and final layer indices")
        self.relation_detail_layers = [int(index) for index in relation_detail_layers]
        if min(self.relation_detail_layers) < 0 or max(self.relation_detail_layers) >= self.vision_config.num_hidden_layers:
            raise ValueError(
                "relation_detail_layers must be valid MoonViT block indices; "
                f"layers={self.relation_detail_layers}, num_layers={self.vision_config.num_hidden_layers}"
            )
        self.relation_gate_loss_weight = relation_gate_loss_weight
        self.relation_slot_gate_loss_weight = relation_slot_gate_loss_weight
        self.relation_slot_objectness_loss_weight = float(
            relation_slot_gate_loss_weight
            if relation_slot_objectness_loss_weight is None
            else relation_slot_objectness_loss_weight
        )
        self.relation_attention_loss_weight = relation_attention_loss_weight
        self.relation_focal_gamma = relation_focal_gamma
        self.relation_focal_beta = relation_focal_beta
        self.relation_gate_threshold = relation_gate_threshold
        self.relation_gate_mode = str(relation_gate_mode).lower()
        if self.relation_gate_mode not in {"observe", "hard", "soft"}:
            raise ValueError("relation_gate_mode must be 'observe', 'hard', or 'soft'")
        self.relation_gate_thresholds = dict(relation_gate_thresholds or {})
        self.tc_msed_stage = str(tc_msed_stage).lower()
        if self.tc_msed_stage not in {"v4", "m1", "m2", "m3", "m4", "m5", "m31"}:
            raise ValueError("tc_msed_stage must be one of v4/m1/m2/m3/m4/m5/m31")
        self.relation_task_scale_router = bool(relation_task_scale_router)
        self.relation_set_localizer = bool(relation_set_localizer)
        self.relation_dynamic_slot_pbd = bool(relation_dynamic_slot_pbd)
        self.relation_coordinate_bridge = bool(relation_coordinate_bridge)
        self.relation_soft_gate = bool(relation_soft_gate)
        self.relation_overlap_adapter = bool(relation_overlap_adapter)
        self.relation_task_hard_router = bool(relation_task_hard_router)
        self.relation_task_experts = bool(relation_task_experts)
        self.relation_task_expert_rank = int(relation_task_expert_rank)
        self.relation_set_decoder = bool(relation_set_decoder)
        self.relation_set_decoder_layers = int(relation_set_decoder_layers)
        self.relation_box_l1_loss_weight = float(relation_box_l1_loss_weight)
        self.relation_box_giou_loss_weight = float(relation_box_giou_loss_weight)
        self.relation_coverage_loss_weight = float(relation_coverage_loss_weight)
        self.relation_coord_prior_sigma = float(relation_coord_prior_sigma)
        if self.relation_coord_prior_sigma <= 0.0:
            raise ValueError("relation_coord_prior_sigma must be positive")
        self.ui_relation_initialization_seed = int(ui_relation_initialization_seed)
        self.ui_relation_initialization_reason = ui_relation_initialization_reason
        if not 0.0 <= float(self.relation_gate_threshold) <= 1.0:
            raise ValueError("relation_gate_threshold must be in [0, 1]")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['text_config'] = self.text_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['image_token_index'] = self.image_token_index
        output['box_start_token_id'] = self.box_start_token_id
        output['box_end_token_id'] = self.box_end_token_id
        output['coord_start_token_id'] = self.coord_start_token_id
        output['coord_end_token_id'] = self.coord_end_token_id
        output['ref_start_token_id'] = self.ref_start_token_id
        output['ref_end_token_id'] = self.ref_end_token_id
        output['none_token_id'] = self.none_token_id
        output['enable_ui_relation'] = self.enable_ui_relation
        output['relation_detail_hidden_size'] = self.relation_detail_hidden_size
        output['relation_num_slots'] = self.relation_num_slots
        output['relation_adapter_bottleneck'] = self.relation_adapter_bottleneck
        output['relation_detail_layers'] = self.relation_detail_layers
        output['relation_gate_loss_weight'] = self.relation_gate_loss_weight
        output['relation_slot_gate_loss_weight'] = self.relation_slot_gate_loss_weight
        output['relation_slot_objectness_loss_weight'] = self.relation_slot_objectness_loss_weight
        output['relation_attention_loss_weight'] = self.relation_attention_loss_weight
        output['relation_focal_gamma'] = self.relation_focal_gamma
        output['relation_focal_beta'] = self.relation_focal_beta
        output['relation_gate_threshold'] = self.relation_gate_threshold
        output['relation_gate_mode'] = self.relation_gate_mode
        output['relation_gate_thresholds'] = self.relation_gate_thresholds
        output['tc_msed_stage'] = self.tc_msed_stage
        output['relation_task_scale_router'] = self.relation_task_scale_router
        output['relation_set_localizer'] = self.relation_set_localizer
        output['relation_dynamic_slot_pbd'] = self.relation_dynamic_slot_pbd
        output['relation_coordinate_bridge'] = self.relation_coordinate_bridge
        output['relation_soft_gate'] = self.relation_soft_gate
        output['relation_overlap_adapter'] = self.relation_overlap_adapter
        output['relation_task_hard_router'] = self.relation_task_hard_router
        output['relation_task_experts'] = self.relation_task_experts
        output['relation_task_expert_rank'] = self.relation_task_expert_rank
        output['relation_set_decoder'] = self.relation_set_decoder
        output['relation_set_decoder_layers'] = self.relation_set_decoder_layers
        output['relation_box_l1_loss_weight'] = self.relation_box_l1_loss_weight
        output['relation_box_giou_loss_weight'] = self.relation_box_giou_loss_weight
        output['relation_coverage_loss_weight'] = self.relation_coverage_loss_weight
        output['relation_coord_prior_sigma'] = self.relation_coord_prior_sigma
        output['ui_relation_initialization_seed'] = self.ui_relation_initialization_seed
        output['ui_relation_initialization_reason'] = self.ui_relation_initialization_reason
        output['_attn_implementation'] = self._attn_implementation
        if hasattr(self, '_attn_implementation_autoset'):
            output['_attn_implementation_autoset'] = self._attn_implementation_autoset
        return output
