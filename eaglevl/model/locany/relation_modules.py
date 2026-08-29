"""Lightweight relation reasoning for unified UI-defect localization.

The module intentionally stays on the vision side.  It consumes three
same-resolution MoonViT feature maps and returns a fixed number of relation
tokens; it never appends patch or relation tokens to the language sequence.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


RELATION_FAMILIES = ("boundary", "pairwise", "text", "presence")
DEFECT_TYPES = (
    "text_overflow",
    "cropping",
    "overlap",
    "ellipsis",
    "missing",
)
DEFAULT_UI_DETAIL_LAYERS = (5, 15, 26)
FAMILY_SCALE_PRIOR = torch.tensor(
    [
        [0.50, 0.35, 0.15],  # boundary
        [0.15, 0.40, 0.45],  # pairwise
        [0.40, 0.25, 0.35],  # text
        [0.10, 0.20, 0.70],  # presence
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class UIRelationPromptSpec:
    """One of the five fixed UI5 prompts and its relation routing target."""

    task_name: str
    diagnostic_name: str
    relation_family: int
    defect_type: int
    prompt_label: str
    aliases: Tuple[str, ...]

    @property
    def prompt(self) -> str:
        return (
            "Locate all the instances that match the following description: "
            f"{self.prompt_label}."
        )


UI_RELATION_PROMPT_SPECS = (
    UIRelationPromptSpec(
        "text_overflow", "text_overflow", 0, 0, "text overflow",
        ("text overflow", "文字溢出"),
    ),
    UIRelationPromptSpec(
        "cropping", "element_cropping", 0, 1, "cropped element",
        ("cropped element", "element cropping", "元素裁切"),
    ),
    UIRelationPromptSpec(
        "occlusion", "element_overlap", 1, 2, "overlapping elements",
        ("overlapping elements", "element overlap", "元素重叠"),
    ),
    UIRelationPromptSpec(
        "text_ellipsis", "text_ellipsis", 2, 3, "abnormal text ellipsis",
        ("abnormal text ellipsis", "ellipsis anomaly", "省略异常"),
    ),
    UIRelationPromptSpec(
        "content_missing", "content_missing", 3, 4, "missing content",
        ("missing content", "content missing", "内容缺失"),
    ),
)


def match_ui_relation_prompt(text: str) -> Optional[UIRelationPromptSpec]:
    """Route the fixed UI5 prompts through one shared training/inference table."""

    normalized = str(text).lower()
    for spec in UI_RELATION_PROMPT_SPECS:
        if any(alias.lower() in normalized for alias in spec.aliases):
            return spec
    return None


def passes_relation_gate(p_defect: torch.Tensor | float, threshold: float) -> bool:
    """Single inference decision used before the bbox generation block."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"relation gate threshold must be in [0, 1], got {threshold}")
    if torch.is_tensor(p_defect):
        if p_defect.numel() != 1:
            raise ValueError("passes_relation_gate expects one sample")
        probability = float(p_defect.detach().float().item())
    else:
        probability = float(p_defect)
    return probability >= float(threshold)


def relation_gate_output_override(
    p_defect: torch.Tensor | float, threshold: float
) -> Optional[str]:
    """Return the forced negative answer, or None when bbox generation may run."""

    return None if passes_relation_gate(p_defect, threshold) else "<box>none</box>"


@dataclass
class RelationPyramidOutput:
    relation_tokens: torch.Tensor
    relation_family: torch.Tensor
    p_defect: torch.Tensor
    image_gate_logits: torch.Tensor
    slot_gate_logits: torch.Tensor
    gate_logits: torch.Tensor
    coarse_boxes: torch.Tensor
    query_attention: Tuple[torch.Tensor, ...]
    relation_summary: torch.Tensor
    best_relation_token: torch.Tensor
    scale_weights: torch.Tensor
    global_task_token: Optional[torch.Tensor] = None
    slot_objectness_logits: Optional[torch.Tensor] = None
    gate_targets: Optional[torch.Tensor] = None
    image_gate_targets: Optional[torch.Tensor] = None
    projected_level_norms: Optional[torch.Tensor] = None
    projected_level_abs_max: Optional[torch.Tensor] = None
    projected_level_saturation_fraction: Optional[torch.Tensor] = None
    projected_level_norm_ratio: Optional[torch.Tensor] = None
    fused_feature_norm: Optional[torch.Tensor] = None
    relation_context_norm: Optional[torch.Tensor] = None
    per_task_gate_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_attention_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_box_l1_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_box_giou_loss: Optional[Dict[int, torch.Tensor]] = None
    gate_loss: Optional[torch.Tensor] = None
    image_gate_loss: Optional[torch.Tensor] = None
    slot_gate_loss: Optional[torch.Tensor] = None
    per_task_image_gate_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_slot_gate_loss: Optional[Dict[int, torch.Tensor]] = None
    attention_loss: Optional[torch.Tensor] = None
    attention_kl_loss: Optional[torch.Tensor] = None
    attention_ce_diagnostic: Optional[torch.Tensor] = None
    box_l1_loss: Optional[torch.Tensor] = None
    box_giou_loss: Optional[torch.Tensor] = None
    matched_slot_indices: Optional[torch.Tensor] = None
    scale_entropy: Optional[torch.Tensor] = None
    scale_batch_std: Optional[torch.Tensor] = None
    coarse_iou_mean: Optional[torch.Tensor] = None
    coarse_recall_03: Optional[torch.Tensor] = None
    coarse_recall_05: Optional[torch.Tensor] = None
    matched_slots: Optional[torch.Tensor] = None
    unmatched_slots: Optional[torch.Tensor] = None
    slot_usage_entropy: Optional[torch.Tensor] = None


def class_balanced_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    defect_type: torch.Tensor,
    gamma: float = 2.0,
    beta: float = 0.999,
    positive_counts: Sequence[int] = (742, 4480, 3068, 3267, 2125),
    total_counts: Sequence[int] = (17604, 17604, 17604, 17604, 17604),
) -> torch.Tensor:
    """Effective-number class-balanced focal loss.

    Positive and negative effective-number weights are normalized within each
    task.  This keeps text-overflow from being ignored without applying its raw
    22.7x inverse-frequency multiplier, which would be harmful to precision.
    """
    if logits.numel() == 0:
        return logits.sum()

    device = logits.device
    dtype = logits.dtype
    pos = torch.as_tensor(positive_counts, device=device, dtype=torch.float32)
    total = torch.as_tensor(total_counts, device=device, dtype=torch.float32)
    neg = total - pos
    beta_tensor = torch.tensor(beta, device=device, dtype=torch.float32)
    pos_weight = (1.0 - beta_tensor) / (1.0 - beta_tensor.pow(pos))
    neg_weight = (1.0 - beta_tensor) / (1.0 - beta_tensor.pow(neg))
    normalizer = 2.0 / (pos_weight + neg_weight)
    pos_weight = pos_weight * normalizer
    neg_weight = neg_weight * normalizer

    class_ids = defect_type.clamp(min=0, max=len(positive_counts) - 1).long()
    trailing_dims = (1,) * max(0, logits.ndim - 1)
    sample_pos_weight = pos_weight[class_ids].to(dtype=dtype).view(-1, *trailing_dims)
    sample_neg_weight = neg_weight[class_ids].to(dtype=dtype).view(-1, *trailing_dims)
    balance_weight = torch.where(targets > 0.5, sample_pos_weight, sample_neg_weight)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    return (balance_weight * (1.0 - p_t).pow(gamma) * bce).mean()


def canonicalize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Clamp boxes to the UI coordinate range and enforce x1<=x2/y1<=y2."""

    first = torch.minimum(boxes[..., :2], boxes[..., 2:])
    second = torch.maximum(boxes[..., :2], boxes[..., 2:])
    return torch.cat((first, second), dim=-1).clamp(0.0, 1000.0)


def aligned_generalized_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Aligned GIoU for equally-shaped ``[..., 4]`` xyxy tensors."""

    boxes1 = canonicalize_xyxy(boxes1.float())
    boxes2 = canonicalize_xyxy(boxes2.float())
    intersection_xy1 = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    intersection_xy2 = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (intersection_xy2 - intersection_xy1).clamp_min(0.0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0.0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0.0).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp_min(1.0e-7)
    enclosing_xy1 = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enclosing_xy2 = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing = (enclosing_xy2 - enclosing_xy1).clamp_min(0.0).prod(dim=-1)
    return iou - (enclosing - union) / enclosing.clamp_min(1.0e-7)


def pairwise_generalized_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    return aligned_generalized_box_iou(boxes1[:, None, :], boxes2[None, :, :])


def hungarian_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact rectangular assignment for small UI slot sets without SciPy.

    UI5 uses at most eight slots/boxes, so dynamic programming over slot bitmasks
    is deterministic and cheaper than adding a runtime SciPy dependency.
    """

    if cost.ndim != 2:
        raise ValueError(f"Hungarian cost must be rank-2, got {tuple(cost.shape)}")
    slots, targets = cost.shape
    if targets == 0 or slots == 0:
        empty = torch.zeros(0, device=cost.device, dtype=torch.long)
        return empty, empty
    if targets > slots:
        raise ValueError(f"targets ({targets}) cannot exceed slots ({slots})")
    detached = cost.detach().float().cpu()
    states: Dict[int, Tuple[float, Tuple[int, ...]]] = {0: (0.0, tuple())}
    for target in range(targets):
        next_states: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
        for used_mask, (value, assignment) in states.items():
            for slot in range(slots):
                bit = 1 << slot
                if used_mask & bit:
                    continue
                candidate = value + float(detached[slot, target].item())
                new_mask = used_mask | bit
                previous = next_states.get(new_mask)
                if previous is None or candidate < previous[0]:
                    next_states[new_mask] = (candidate, assignment + (slot,))
        states = next_states
    _, best_assignment = min(states.values(), key=lambda item: item[0])
    target_indices = torch.arange(targets, device=cost.device, dtype=torch.long)
    slot_indices = torch.tensor(best_assignment, device=cost.device, dtype=torch.long)
    return slot_indices, target_indices


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(hidden_states))))
        return hidden_states + self.scale.tanh() * residual


class TaskRoutedExpertBank(nn.Module):
    """Exactly one residual expert is activated by each known defect type.

    This is deterministic sparse adapter-MoE routing: no learned router, soft
    mixture, top-k selection, or load-balancing loss is involved.  Grouped
    execution is important here; evaluating all experts and masking afterwards
    would still build autograd graphs (and gradients) for inactive experts.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 8,
        num_defect_types: int = 5,
        initial_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0 or num_defect_types <= 0:
            raise ValueError("hidden_size, rank and num_defect_types must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.num_defect_types = int(num_defect_types)
        self.initial_alpha = float(initial_alpha)
        self.experts = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_size),
                        "down": nn.Linear(hidden_size, rank, bias=False),
                        "up": nn.Linear(rank, hidden_size, bias=False),
                    }
                )
                for _ in range(num_defect_types)
            ]
        )
        self.alpha = nn.Parameter(
            torch.full((num_defect_types,), self.initial_alpha)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # ``from_pretrained(..., low_cpu_mem_usage=True)`` may materialize
        # checkpoint-missing direct Parameters with uninitialized storage.
        # Resetting only child Linear modules is therefore insufficient.
        with torch.no_grad():
            self.alpha.fill_(self.initial_alpha)
        for expert in self.experts:
            nn.init.zeros_(expert["up"].weight)

    def _validate_tasks(self, defect_type: torch.Tensor, batch: int) -> torch.Tensor:
        tasks = defect_type.reshape(-1).long()
        if tasks.numel() != batch:
            raise ValueError(
                f"defect_type has {tasks.numel()} values for expert batch {batch}"
            )
        invalid = (tasks < 0) | (tasks >= self.num_defect_types)
        if bool(invalid.any()):
            raise ValueError(
                "TaskRoutedExpertBank requires a known UI5 defect_type; "
                f"got {tasks[invalid].detach().cpu().tolist()}"
            )
        return tasks

    def forward(
        self, hidden_states: torch.Tensor, defect_type: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "expert input must be [B,...,H] with "
                f"H={self.hidden_size}, got {tuple(hidden_states.shape)}"
            )
        tasks = self._validate_tasks(defect_type, hidden_states.shape[0])
        output = hidden_states
        # Sorting the tiny set makes route execution/checkpoint tests stable.
        for task_tensor in torch.unique(tasks, sorted=True):
            task = int(task_tensor.item())
            indices = (tasks == task).nonzero(as_tuple=False).flatten()
            selected = hidden_states.index_select(0, indices)
            expert = self.experts[task]
            residual = expert["up"](
                F.gelu(expert["down"](expert["norm"](selected)))
            )
            selected_output = selected + self.alpha[task].to(
                dtype=selected.dtype
            ) * residual
            output = output.index_copy(0, indices, selected_output)
        return output


class _TaskRoutedLinearBank(nn.Module):
    """Task-private delta heads with the same strict grouped routing policy."""

    def __init__(
        self, hidden_size: int, output_size: int, num_defect_types: int = 5
    ) -> None:
        super().__init__()
        self.num_defect_types = int(num_defect_types)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, output_size) for _ in range(num_defect_types)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hidden_states: torch.Tensor, defect_type: torch.Tensor) -> torch.Tensor:
        tasks = defect_type.reshape(-1).long()
        if tasks.numel() != hidden_states.shape[0]:
            raise ValueError("task delta head batch mismatch")
        if bool(((tasks < 0) | (tasks >= self.num_defect_types)).any()):
            raise ValueError("task delta head received unknown defect_type")
        output = hidden_states.new_zeros((*hidden_states.shape[:-1], self.heads[0].out_features))
        for task_tensor in torch.unique(tasks, sorted=True):
            task = int(task_tensor.item())
            indices = (tasks == task).nonzero(as_tuple=False).flatten()
            selected = self.heads[task](hidden_states.index_select(0, indices))
            output = output.index_copy(0, indices, selected)
        return output


def inverse_sigmoid(values: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    values = values.float().clamp(min=eps, max=1.0 - eps)
    return torch.log(values) - torch.log1p(-values)


def cxcywh_to_xyxy_unit(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half_size = boxes[..., 2:].clamp_min(0.0) * 0.5
    first = (center - half_size).clamp(0.0, 1.0)
    second = (center + half_size).clamp(0.0, 1.0)
    return torch.cat((torch.minimum(first, second), torch.maximum(first, second)), dim=-1)


@dataclass
class TaskConditionedSetDecoderOutput:
    global_task_token: torch.Tensor
    slot_tokens: torch.Tensor
    slot_objectness_logits: torch.Tensor
    slot_boxes_norm: torch.Tensor
    slot_boxes_norm1000: torch.Tensor
    slot_attention: torch.Tensor
    auxiliary_boxes_norm: Tuple[torch.Tensor, ...]
    auxiliary_objectness_logits: Tuple[torch.Tensor, ...]


class _TaskConditionedSetDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expert_rank: int,
        num_defect_types: int,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.task_expert = TaskRoutedExpertBank(
            hidden_size,
            rank=expert_rank,
            num_defect_types=num_defect_types,
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        defect_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = self.self_norm(queries)
        update, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + update
        normalized = self.cross_norm(queries)
        update, attention = self.cross_attention(
            normalized,
            memory,
            memory,
            need_weights=True,
            average_attn_weights=True,
        )
        queries = queries + update
        queries = queries + self.ffn(self.ffn_norm(queries))
        queries = self.task_expert(queries, defect_type)
        return queries, attention


class TaskConditionedSetDecoder(nn.Module):
    """One global query plus K task-conditioned direct object queries."""

    def __init__(
        self,
        hidden_size: int,
        num_object_queries: int = 8,
        num_decoder_layers: int = 3,
        num_attention_heads: Optional[int] = None,
        num_defect_types: int = 5,
        expert_rank: int = 8,
    ) -> None:
        super().__init__()
        if num_object_queries <= 0 or num_decoder_layers <= 0:
            raise ValueError("set decoder queries/layers must be positive")
        if num_attention_heads is None:
            num_attention_heads = next(
                heads for heads in (8, 4, 2, 1) if hidden_size % heads == 0
            )
        if hidden_size % int(num_attention_heads) != 0:
            raise ValueError("set decoder hidden size must divide attention heads")
        self.hidden_size = int(hidden_size)
        self.num_object_queries = int(num_object_queries)
        self.num_defect_types = int(num_defect_types)
        self.global_query = nn.Parameter(torch.empty(1, hidden_size))
        self.object_queries = nn.Parameter(torch.empty(num_object_queries, hidden_size))
        self.task_embedding = nn.Embedding(num_defect_types, hidden_size)
        self.reference_box_logits = nn.Parameter(
            torch.empty(num_object_queries, 4)
        )
        self.layers = nn.ModuleList(
            [
                _TaskConditionedSetDecoderLayer(
                    hidden_size,
                    int(num_attention_heads),
                    expert_rank,
                    num_defect_types,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.shared_box_deltas = nn.ModuleList(
            [nn.Linear(hidden_size, 4) for _ in range(num_decoder_layers)]
        )
        self.task_box_deltas = nn.ModuleList(
            [
                _TaskRoutedLinearBank(hidden_size, 4, num_defect_types)
                for _ in range(num_decoder_layers)
            ]
        )
        self.objectness_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_decoder_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.global_query, std=0.02)
        nn.init.normal_(self.object_queries, std=0.02)
        # Deterministic spread across the image; refinement is unconstrained in
        # inverse-sigmoid space and can move arbitrarily far from these points.
        side = int(math.ceil(math.sqrt(self.num_object_queries)))
        references = []
        for index in range(self.num_object_queries):
            row, column = divmod(index, side)
            references.append(
                ((column + 0.5) / side, (row + 0.5) / side, 0.25, 0.25)
            )
        with torch.no_grad():
            self.reference_box_logits.copy_(
                inverse_sigmoid(torch.tensor(references, dtype=torch.float32))
            )
        for shared in self.shared_box_deltas:
            nn.init.zeros_(shared.weight)
            nn.init.zeros_(shared.bias)
        for objectness in self.objectness_heads:
            nn.init.zeros_(objectness.weight)
            nn.init.constant_(objectness.bias, -2.0)
        for layer in self.layers:
            layer.self_attention._reset_parameters()
            layer.cross_attention._reset_parameters()
            layer.task_expert.reset_parameters()
        for task_delta in self.task_box_deltas:
            task_delta.reset_parameters()

    def forward(
        self, memory: torch.Tensor, defect_type: torch.Tensor
    ) -> TaskConditionedSetDecoderOutput:
        if memory.ndim != 3 or memory.shape[-1] != self.hidden_size:
            raise ValueError(
                f"set decoder memory must be [B,P,{self.hidden_size}]"
            )
        tasks = defect_type.reshape(-1).long()
        if tasks.numel() != memory.shape[0]:
            raise ValueError("set decoder defect_type batch mismatch")
        if bool(((tasks < 0) | (tasks >= self.num_defect_types)).any()):
            raise ValueError("set decoder requires one of the five known defect types")
        batch = memory.shape[0]
        task_condition = self.task_embedding(tasks).unsqueeze(1)
        base_queries = torch.cat((self.global_query, self.object_queries), dim=0)
        queries = base_queries.unsqueeze(0).expand(batch, -1, -1) + task_condition
        reference = torch.sigmoid(self.reference_box_logits.float()).unsqueeze(0).expand(
            batch, -1, -1
        )
        auxiliary_boxes: List[torch.Tensor] = []
        auxiliary_objectness: List[torch.Tensor] = []
        final_attention = memory.new_zeros(
            (batch, self.num_object_queries, memory.shape[1]), dtype=torch.float32
        )
        for index, layer in enumerate(self.layers):
            queries, attention = layer(queries, memory, tasks)
            slots = queries[:, 1:]
            delta = self.shared_box_deltas[index](slots).float()
            delta = delta + self.task_box_deltas[index](slots, tasks).float()
            reference = torch.sigmoid(inverse_sigmoid(reference) + delta)
            objectness = self.objectness_heads[index](slots).squeeze(-1)
            auxiliary_boxes.append(cxcywh_to_xyxy_unit(reference))
            auxiliary_objectness.append(objectness)
            final_attention = attention[:, 1:].float()
        queries = self.output_norm(queries)
        boxes_norm = auxiliary_boxes[-1]
        return TaskConditionedSetDecoderOutput(
            global_task_token=queries[:, 0],
            slot_tokens=queries[:, 1:],
            slot_objectness_logits=auxiliary_objectness[-1],
            slot_boxes_norm=boxes_norm,
            slot_boxes_norm1000=boxes_norm * 1000.0,
            slot_attention=final_attention,
            auxiliary_boxes_norm=tuple(auxiliary_boxes),
            auxiliary_objectness_logits=tuple(auxiliary_objectness),
        )


class RelationConditionedDetailPyramid(nn.Module):
    """Relation-specific scale selection and implicit UI relation queries."""

    def __init__(
        self,
        vision_hidden_size: int,
        detail_hidden_size: int = 256,
        num_slots: int = 8,
        adapter_bottleneck: int = 64,
        num_families: int = 4,
        num_defect_types: int = 5,
        focal_gamma: float = 2.0,
        focal_beta: float = 0.999,
        task_scale_router: bool = False,
        set_localizer: bool = False,
        soft_gate: bool = False,
        task_hard_router: bool = False,
        task_experts: bool = False,
        task_expert_rank: int = 8,
        set_decoder: bool = False,
        set_decoder_layers: int = 3,
    ) -> None:
        super().__init__()
        self.detail_hidden_size = detail_hidden_size
        self.num_slots = num_slots
        self.num_families = num_families
        self.num_defect_types = num_defect_types
        self.focal_gamma = focal_gamma
        self.focal_beta = focal_beta
        self.task_scale_router = bool(task_scale_router)
        self.set_localizer = bool(set_localizer)
        self.soft_gate = bool(soft_gate)
        self.task_hard_router = bool(task_hard_router)
        self.task_experts = bool(task_experts)
        self.task_expert_rank = int(task_expert_rank)
        self.set_decoder = bool(set_decoder)
        self.set_decoder_layers = int(set_decoder_layers)
        if self.set_decoder and not (self.task_hard_router and self.task_experts):
            raise ValueError(
                "m31 set decoder requires deterministic task routing and experts"
            )

        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(vision_hidden_size),
                    nn.Linear(vision_hidden_size, detail_hidden_size, bias=False),
                )
                for _ in range(3)
            ]
        )

        # early / middle / final initialization.  These remain fully learnable.
        initial_scale_weights = FAMILY_SCALE_PRIOR.clone()
        self.register_buffer(
            "family_scale_prior", initial_scale_weights.clone(), persistent=False
        )
        self.scale_logits = nn.Parameter(initial_scale_weights.log())
        if self.task_scale_router:
            self.task_scale_embedding = nn.Embedding(num_defect_types, detail_hidden_size)
            self.task_scale_projection = nn.Linear(detail_hidden_size, 3, bias=False)
            self.image_scale_projection = nn.Linear(detail_hidden_size, 3, bias=False)

        self.evidence_queries = nn.Parameter(
            torch.empty(num_families, num_slots, detail_hidden_size)
        )
        self.context_queries = nn.Parameter(
            torch.empty(num_families, num_slots, detail_hidden_size)
        )
        self.family_embedding = nn.Embedding(num_families, detail_hidden_size)
        self.defect_embedding = nn.Embedding(num_defect_types, detail_hidden_size)
        self.key_projection = nn.Linear(detail_hidden_size, detail_hidden_size, bias=False)
        self.value_projection = nn.Linear(detail_hidden_size, detail_hidden_size, bias=False)
        self.query_norm = nn.LayerNorm(detail_hidden_size)
        self.token_norm = nn.LayerNorm(detail_hidden_size)

        self.relation_mlp = nn.Sequential(
            nn.LayerNorm(4 * detail_hidden_size),
            nn.Linear(4 * detail_hidden_size, detail_hidden_size),
            nn.GELU(),
            nn.Linear(detail_hidden_size, detail_hidden_size),
        )
        self.family_adapters = nn.ModuleList(
            [ResidualAdapter(detail_hidden_size, adapter_bottleneck) for _ in range(num_families)]
        )
        self.gate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(detail_hidden_size),
                    nn.Linear(detail_hidden_size, adapter_bottleneck),
                    nn.GELU(),
                    nn.Linear(adapter_bottleneck, 1),
                )
                for _ in range(num_families)
            ]
        )
        # Slot objectness and image defectness are deliberately separate.
        # Slot heads select relation evidence for PBD; these five fixed-task
        # heads answer whether the screenshot contains that defect at all.
        self.image_gate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(detail_hidden_size),
                    nn.Linear(detail_hidden_size, adapter_bottleneck),
                    nn.GELU(),
                    nn.Linear(adapter_bottleneck, 1),
                )
                for _ in range(num_defect_types)
            ]
        )
        if self.set_localizer:
            self.coarse_box_head = nn.Sequential(
                nn.LayerNorm(detail_hidden_size),
                nn.Linear(detail_hidden_size, adapter_bottleneck),
                nn.GELU(),
                nn.Linear(adapter_bottleneck, 4),
            )
        if self.set_decoder:
            self.task_set_decoder = TaskConditionedSetDecoder(
                detail_hidden_size,
                num_object_queries=num_slots,
                num_decoder_layers=set_decoder_layers,
                num_defect_types=num_defect_types,
                expert_rank=task_expert_rank,
            )
            self.relation_semantic_experts = TaskRoutedExpertBank(
                detail_hidden_size,
                rank=task_expert_rank,
                num_defect_types=num_defect_types,
            )
        if self.soft_gate:
            self.soft_gate_beta = nn.Parameter(torch.zeros(num_defect_types))
        self.reset_parameters()
        if self.set_decoder:
            # M3.1 retains these weights solely for checkpoint compatibility
            # and detached diagnostics; they must not enter optimization.
            self.image_gate_heads.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.evidence_queries, std=0.02)
        nn.init.normal_(self.context_queries, std=0.02)
        with torch.no_grad():
            self.scale_logits.copy_(self.expected_family_scale_logits())
        if self.task_scale_router:
            nn.init.zeros_(self.task_scale_projection.weight)
            nn.init.zeros_(self.image_scale_projection.weight)
        if self.set_localizer:
            nn.init.zeros_(self.coarse_box_head[-1].weight)
            nn.init.zeros_(self.coarse_box_head[-1].bias)
        if self.soft_gate:
            nn.init.zeros_(self.soft_gate_beta)
        # Start conservatively: an unseen slot should prefer non-defect.
        for head in self.gate_heads:
            nn.init.constant_(head[-1].bias, -2.0)
        for head in self.image_gate_heads:
            nn.init.constant_(head[-1].bias, -2.0)

    def expected_family_scale_logits(self) -> torch.Tensor:
        """Return canonical FP32 log-priors cast to the parameter contract.

        DeepSpeed may cast the module to BF16 *after* the UI branch is
        initialized.  A tensor buffer follows that cast, so recomputing
        ``log(buffer)`` later is not equivalent to casting the original FP32
        log-prior.  Keep the source of truth in the module-level FP32 constant
        so initialization-before-cast and initialization-after-cast agree.
        """

        return (
            FAMILY_SCALE_PRIOR.to(
                device=self.scale_logits.device, dtype=torch.float32
            )
            .log()
            .to(dtype=self.scale_logits.dtype)
        )

    def expected_family_scale_weights(self) -> torch.Tensor:
        """Return the configured prior through the module's real dtype path.

        ``family_scale_prior`` and ``scale_logits`` are cast with the model.
        Reconstructing the expectation this way mirrors ``reset_parameters``
        followed by the FP32 softmax used in ``forward``.  In particular, it
        must not compare that result with probabilities rounded directly to
        BF16, because those are two different numerical operations.
        """

        return self.expected_family_scale_logits().float().softmax(dim=-1)

    def assert_family_scale_prior(self, atol: float = 1.0e-6) -> None:
        actual = self.scale_logits.detach().float().softmax(dim=-1)
        # ``from_pretrained(..., torch_dtype=bfloat16)`` casts the learnable
        # logits before the optional UI branch is initialized.  Copying the
        # FP32 log-prior into that parameter therefore produces the *BF16
        # representation* of the prior, not bitwise FP32 probabilities.  Build
        # the expected value through the same dtype round-trip so this remains
        # a real overwrite check instead of rejecting normal quantization.
        expected = self.expected_family_scale_weights()
        if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
            raise RuntimeError(
                "TC-MSED family scale prior was overwritten during initialization: "
                f"actual={actual.tolist()}, expected={expected.tolist()}"
            )

    @staticmethod
    def _sanitize(hidden_states: torch.Tensor, limit: float = 128.0) -> torch.Tensor:
        """Keep the new BF16 branch finite without changing ordinary values."""
        return torch.nan_to_num(
            hidden_states,
            nan=0.0,
            posinf=limit,
            neginf=-limit,
        ).clamp(min=-limit, max=limit)

    @staticmethod
    def _split_features(
        features: Sequence[torch.Tensor], grid_hws: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        lengths = (grid_hws[:, 0] * grid_hws[:, 1]).tolist()
        output: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        start = 0
        for length in lengths:
            length = int(length)
            output.append(tuple(level[start : start + length] for level in features))
            start += length
        return output

    @staticmethod
    def _patch_coordinates(height: int, width: int, device: torch.device) -> torch.Tensor:
        ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
        xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)

    @staticmethod
    def _mask_distribution(mask: torch.Tensor, fallback_distance: torch.Tensor) -> torch.Tensor:
        if bool(mask.any()):
            distribution = mask.to(dtype=torch.float32)
        else:
            distribution = torch.zeros_like(fallback_distance, dtype=torch.float32)
            distribution[fallback_distance.argmin()] = 1.0
        return distribution / distribution.sum().clamp_min(1.0)

    def _attention_targets(
        self,
        boxes: torch.Tensor,
        box_mask: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = self._patch_coordinates(height, width, device)
        evidence = torch.zeros(self.num_slots, height * width, device=device)
        context = torch.zeros_like(evidence)
        active_slots = torch.zeros(self.num_slots, device=device)

        # This helper is the legacy ordered-slot target builder used by v4/M1.
        # It cannot represent more targets than slots, so cap explicitly here;
        # the set-localizer path below keeps the complete GT set and applies its
        # own order-invariant capacity rule before Hungarian matching.
        valid_indices = box_mask.nonzero(as_tuple=False).flatten()[: self.num_slots]
        for slot, box_index in enumerate(valid_indices.tolist()):
            box = boxes[box_index].float().to(device=device) / 1000.0
            x1, y1, x2, y2 = box.unbind()
            inside = (
                (coords[:, 0] >= x1)
                & (coords[:, 0] <= x2)
                & (coords[:, 1] >= y1)
                & (coords[:, 1] <= y2)
            )
            center = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
            center_distance = (coords - center).square().sum(dim=-1)
            evidence[slot] = self._mask_distribution(inside, center_distance)

            half_width = (x2 - x1) * 1.5
            half_height = (y2 - y1) * 1.5
            expanded = (
                (coords[:, 0] >= center[0] - half_width)
                & (coords[:, 0] <= center[0] + half_width)
                & (coords[:, 1] >= center[1] - half_height)
                & (coords[:, 1] <= center[1] + half_height)
            )
            ring = expanded & ~inside
            outside_distance = center_distance.masked_fill(inside, float("inf"))
            context[slot] = self._mask_distribution(ring, outside_distance)
            active_slots[slot] = 1.0
        return evidence, context, active_slots

    def _box_attention_targets(
        self,
        boxes: torch.Tensor,
        box_mask: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one spatial target per GT box, independent of slot order."""

        # Preserve the complete unordered target set here.  The caller owns the
        # deterministic K-slot capacity policy and must retain the selected GT
        # ordinals so teacher-forced box spans stay aligned with matched slots.
        valid_indices = box_mask.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            empty = torch.zeros(0, height * width, device=device)
            return empty, empty, boxes.new_zeros((0, 4), device=device).float()
        coords = self._patch_coordinates(height, width, device)
        evidence: List[torch.Tensor] = []
        context: List[torch.Tensor] = []
        valid_boxes: List[torch.Tensor] = []
        for box_index in valid_indices.tolist():
            raw_box = canonicalize_xyxy(boxes[box_index].float().to(device=device))
            box = raw_box / 1000.0
            x1, y1, x2, y2 = box.unbind()
            inside = (
                (coords[:, 0] >= x1)
                & (coords[:, 0] <= x2)
                & (coords[:, 1] >= y1)
                & (coords[:, 1] <= y2)
            )
            center = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
            center_distance = (coords - center).square().sum(dim=-1)
            evidence.append(self._mask_distribution(inside, center_distance))
            half_width = (x2 - x1) * 1.5
            half_height = (y2 - y1) * 1.5
            expanded = (
                (coords[:, 0] >= center[0] - half_width)
                & (coords[:, 0] <= center[0] + half_width)
                & (coords[:, 1] >= center[1] - half_height)
                & (coords[:, 1] <= center[1] + half_height)
            )
            ring = expanded & ~inside
            outside_distance = center_distance.masked_fill(inside, float("inf"))
            context.append(self._mask_distribution(ring, outside_distance))
            valid_boxes.append(raw_box)
        return torch.stack(evidence), torch.stack(context), torch.stack(valid_boxes)

    @staticmethod
    def _coarse_boxes(attention: torch.Tensor, height: int, width: int) -> torch.Tensor:
        coords = RelationConditionedDetailPyramid._patch_coordinates(
            height, width, attention.device
        ).to(dtype=attention.dtype)
        center = attention @ coords
        delta = coords.unsqueeze(0) - center.unsqueeze(1)
        variance = (attention.unsqueeze(-1) * delta.square()).sum(dim=1)
        radius = 2.0 * variance.clamp_min(1e-6).sqrt()
        xy1 = (center - radius).clamp(0.0, 1.0)
        xy2 = (center + radius).clamp(0.0, 1.0)
        return torch.cat((xy1, xy2), dim=-1) * 1000.0

    def forward(
        self,
        pyramid_features: Sequence[torch.Tensor],
        grid_hws: torch.Tensor,
        relation_family: torch.Tensor,
        defect_type: torch.Tensor,
        image_flags: Optional[torch.Tensor] = None,
        target_boxes: Optional[torch.Tensor] = None,
        target_box_mask: Optional[torch.Tensor] = None,
    ) -> RelationPyramidOutput:
        if len(pyramid_features) != 3:
            raise ValueError(f"Expected three pyramid levels, got {len(pyramid_features)}")

        relation_family = relation_family.reshape(-1).long()
        defect_type = defect_type.reshape(-1).long()
        num_samples = relation_family.numel()
        if self.set_decoder:
            invalid_tasks = (defect_type < 0) | (
                defect_type >= self.num_defect_types
            )
            if bool(invalid_tasks.any()):
                raise ValueError(
                    "m31 requires a known defect_type for every packed sample; "
                    f"got {defect_type[invalid_tasks].detach().cpu().tolist()}"
                )
        if image_flags is None:
            image_flags = torch.ones(num_samples, device=relation_family.device, dtype=torch.long)
        else:
            image_flags = image_flags.reshape(-1).long()
        if image_flags.numel() != num_samples:
            raise ValueError(
                f"image_flags ({image_flags.numel()}) and relation_family ({num_samples}) disagree"
            )

        projected_raw = [
            projection(self._sanitize(level))
            for projection, level in zip(self.level_projections, pyramid_features)
        ]
        nonfinite_levels = [
            index
            for index, level in enumerate(projected_raw)
            if not bool(torch.isfinite(level).all())
        ]
        if nonfinite_levels:
            raise FloatingPointError(
                "Detail Pyramid projection produced NaN/Inf at level indices "
                f"{nonfinite_levels}"
            )
        projected_levels = [self._sanitize(level) for level in projected_raw]
        image_features = self._split_features(projected_levels, grid_hws)
        # Keep the simplex in FP32 even when the module is cast to BF16.  Three
        # BF16 representations of 1/3 sum to 1.001953125, which is expected
        # quantization rather than a broken scale gate.  Returning FP32 weights
        # also makes the training audit and Excel diagnostics reflect the
        # mathematical softmax instead of that storage artifact.
        scale_weights_all = self.scale_logits.float().softmax(dim=-1)

        relation_tokens_list: List[torch.Tensor] = []
        global_task_tokens_list: List[torch.Tensor] = []
        gate_logits_list: List[torch.Tensor] = []
        image_gate_logits_list: List[torch.Tensor] = []
        coarse_boxes_list: List[torch.Tensor] = []
        attention_list: List[torch.Tensor] = []
        scale_weights_list: List[torch.Tensor] = []
        gate_targets_list: List[torch.Tensor] = []
        evidence_losses: List[torch.Tensor] = []
        attention_losses_by_task: Dict[int, List[torch.Tensor]] = {
            task: [] for task in range(self.num_defect_types)
        }
        box_l1_losses_by_task: Dict[int, List[torch.Tensor]] = {
            task: [] for task in range(self.num_defect_types)
        }
        box_giou_losses_by_task: Dict[int, List[torch.Tensor]] = {
            task: [] for task in range(self.num_defect_types)
        }
        fused_norms: List[torch.Tensor] = []
        relation_context_norms: List[torch.Tensor] = []
        box_l1_losses: List[torch.Tensor] = []
        box_giou_losses: List[torch.Tensor] = []
        attention_kl_losses: List[torch.Tensor] = []
        attention_ce_values: List[torch.Tensor] = []
        matched_indices_list: List[torch.Tensor] = []
        coarse_ious: List[torch.Tensor] = []
        matched_counts: List[torch.Tensor] = []
        unmatched_counts: List[torch.Tensor] = []

        image_index = 0
        parameter_zero = self.evidence_queries.sum() * 0.0
        target_capacity = (
            int(target_boxes.shape[1])
            if target_boxes is not None and target_boxes.ndim >= 3
            else self.num_slots
        )
        for sample_index in range(num_samples):
            raw_family = int(relation_family[sample_index].item())
            family = int(relation_family[sample_index].clamp(0, self.num_families - 1).item())
            task = int(defect_type[sample_index].clamp(0, self.num_defect_types - 1).item())
            num_images = int(image_flags[sample_index].item())
            if raw_family < 0 or num_images <= 0 or image_index >= len(image_features):
                empty_tokens = torch.zeros(
                    self.num_slots,
                    self.detail_hidden_size,
                    device=pyramid_features[0].device,
                    dtype=pyramid_features[0].dtype,
                ) + parameter_zero
                relation_tokens_list.append(empty_tokens)
                global_task_tokens_list.append(empty_tokens.mean(dim=0))
                gate_logits_list.append(torch.full((self.num_slots,), -20.0, device=empty_tokens.device, dtype=empty_tokens.dtype))
                image_gate_logits_list.append(
                    torch.full((), -20.0, device=empty_tokens.device, dtype=empty_tokens.dtype)
                    + parameter_zero
                )
                coarse_boxes_list.append(torch.zeros(self.num_slots, 4, device=empty_tokens.device, dtype=empty_tokens.dtype))
                attention_list.append(torch.zeros(2, self.num_slots, 1, device=empty_tokens.device, dtype=empty_tokens.dtype))
                scale_weights_list.append(scale_weights_all[family])
                gate_targets_list.append(torch.zeros(self.num_slots, device=empty_tokens.device, dtype=empty_tokens.dtype))
                matched_indices_list.append(
                    torch.full(
                        (target_capacity,),
                        -1,
                        device=empty_tokens.device,
                        dtype=torch.long,
                    )
                )
                matched_counts.append(parameter_zero.detach().float())
                unmatched_counts.append(parameter_zero.detach().float() + self.num_slots)
                fused_norms.append(parameter_zero.detach().float())
                relation_context_norms.append(parameter_zero.detach().float())
                # Even text-only samples carry one dummy 2x2 grid in the base
                # pipeline, so keep the image cursor aligned with grid_hws.
                image_index += max(num_images, 1)
                continue

            # UI training uses one screenshot.  If a generic sample has visual
            # prompt images, relation supervision intentionally stays on image-1.
            levels = image_features[image_index]
            height, width = [int(v) for v in grid_hws[image_index].tolist()]
            image_index += num_images
            if self.task_scale_router:
                image_descriptor = torch.stack(
                    [level.float().mean(dim=0) for level in levels], dim=0
                ).mean(dim=0)
                task_residual = self.task_scale_projection(
                    self.task_scale_embedding.weight[task]
                ).float()
                image_residual = self.image_scale_projection(
                    image_descriptor.to(dtype=self.image_scale_projection.weight.dtype)
                ).float()
                weights = (
                    self.scale_logits[family].float()
                    + task_residual
                    + image_residual
                ).softmax(dim=-1)
            else:
                weights = scale_weights_all[family]
            fused = sum(
                weights[level].to(device=levels[level].device, dtype=levels[level].dtype)
                * levels[level]
                for level in range(3)
            )
            fused = self._sanitize(self.token_norm(self._sanitize(fused)))
            fused_norms.append(fused.detach().float().norm(dim=-1).mean())
            keys = self._sanitize(self.key_projection(fused))
            values = self._sanitize(self.value_projection(fused))

            if self.set_decoder:
                task_tensor = defect_type[sample_index : sample_index + 1]
                decoded = self.task_set_decoder(fused.unsqueeze(0), task_tensor)
                adapted_relation = self.relation_semantic_experts(
                    decoded.slot_tokens, task_tensor
                )[0]
                global_task_token = self.relation_semantic_experts(
                    decoded.global_task_token.unsqueeze(1), task_tensor
                )[0, 0]
                gate_logits = self._sanitize(
                    decoded.slot_objectness_logits[0], limit=30.0
                )
                sample_coarse_boxes = canonicalize_xyxy(
                    decoded.slot_boxes_norm1000[0].float()
                )
                evidence_attention = decoded.slot_attention[0].to(dtype=fused.dtype)
                context_attention = (1.0 - evidence_attention.float()).clamp_min(0.0)
                context_attention = (
                    context_attention
                    / context_attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-7)
                ).to(dtype=fused.dtype)
                # Image Gate is diagnostics-only in m31.  Detaching both input
                # and output prevents it from affecting hidden states/loss.
                image_gate_logit = self.image_gate_heads[task](
                    global_task_token.detach()
                ).squeeze(-1).detach()
                relation_tokens_for_sample = adapted_relation
            else:
                conditioning = self._sanitize(
                    self.family_embedding.weight[family] + self.defect_embedding.weight[task],
                    limit=32.0,
                )
                evidence_query = self._sanitize(
                    self.query_norm(self._sanitize(self.evidence_queries[family] + conditioning)),
                    limit=32.0,
                )
                context_query = self._sanitize(
                    self.query_norm(self._sanitize(self.context_queries[family] + conditioning)),
                    limit=32.0,
                )
                scale = math.sqrt(self.detail_hidden_size)
                evidence_logits = self._sanitize(
                    evidence_query @ keys.transpose(0, 1) / scale, limit=30.0
                )
                context_logits = self._sanitize(
                    context_query @ keys.transpose(0, 1) / scale, limit=30.0
                )
                evidence_attention = evidence_logits.float().softmax(dim=-1).to(dtype=fused.dtype)
                context_attention = context_logits.float().softmax(dim=-1).to(dtype=fused.dtype)
                evidence_state = evidence_attention @ values
                context_state = context_attention @ values
                relation_input = self._sanitize(torch.cat(
                    (
                        evidence_state,
                        context_state,
                        evidence_state - context_state,
                        evidence_state * context_state,
                    ),
                    dim=-1,
                ))
                base_relation = self._sanitize(self.relation_mlp(relation_input), limit=32.0)
                adapted_all = torch.stack(
                    [self._sanitize(adapter(base_relation), limit=32.0) for adapter in self.family_adapters], dim=0
                )
                adapted_relation = adapted_all[family]
                logits_all = torch.stack(
                    [head(adapted_relation).squeeze(-1) for head in self.gate_heads], dim=0
                )
                gate_logits = self._sanitize(logits_all[family], limit=30.0)
                global_task_token = adapted_relation.mean(dim=0)
                image_gate_logit = self._sanitize(
                    self.image_gate_heads[task](global_task_token).squeeze(-1),
                    limit=30.0,
                )
                relation_tokens_for_sample = self._sanitize(
                    torch.sigmoid(gate_logits).unsqueeze(-1) * adapted_relation,
                    limit=32.0,
                )
                attention_boxes = self._coarse_boxes(evidence_attention, height, width)
                if self.set_localizer:
                    residual = 100.0 * torch.tanh(self.coarse_box_head(adapted_relation).float())
                    sample_coarse_boxes = canonicalize_xyxy(attention_boxes.float() + residual)
                else:
                    sample_coarse_boxes = attention_boxes

            relation_context_norms.append(
                adapted_relation.detach().float().norm(dim=-1).mean()
            )
            relation_tokens_list.append(relation_tokens_for_sample)
            global_task_tokens_list.append(global_task_token)
            gate_logits_list.append(gate_logits)
            image_gate_logits_list.append(image_gate_logit)
            coarse_boxes_list.append(sample_coarse_boxes.to(dtype=relation_tokens_for_sample.dtype))
            attention_list.append(torch.stack((evidence_attention, context_attention), dim=0))
            scale_weights_list.append(weights)

            gate_targets = torch.zeros(self.num_slots, device=fused.device, dtype=fused.dtype)
            matched_for_sample = torch.full(
                (target_capacity,), -1, device=fused.device, dtype=torch.long
            )
            if target_boxes is not None and target_box_mask is not None and self.set_localizer:
                box_evidence, box_context, valid_boxes = self._box_attention_targets(
                    target_boxes[sample_index],
                    target_box_mask[sample_index].bool(),
                    height,
                    width,
                    fused.device,
                )
                # K is intentionally fixed at eight for the first ablation.
                # A pathological screenshot with more GT boxes must not crash
                # a long run; keep the K largest boxes using an order-invariant
                # geometric criterion until a larger-K experiment is explicit.
                valid_target_ordinals = torch.arange(
                    valid_boxes.shape[0], device=fused.device, dtype=torch.long
                )
                if valid_boxes.shape[0] > self.num_slots:
                    areas = (
                        (valid_boxes[:, 2] - valid_boxes[:, 0]).clamp_min(0.0)
                        * (valid_boxes[:, 3] - valid_boxes[:, 1]).clamp_min(0.0)
                    )
                    keep = torch.argsort(areas, descending=True)[: self.num_slots]
                    valid_boxes = valid_boxes[keep]
                    box_evidence = box_evidence[keep]
                    box_context = box_context[keep]
                    valid_target_ordinals = valid_target_ordinals[keep]
                if valid_boxes.shape[0] > 0:
                    positive_focal_cost = (
                        (1.0 - torch.sigmoid(gate_logits.float())).pow(self.focal_gamma)
                        * F.softplus(-gate_logits.float())
                    )[:, None]
                    l1_cost = torch.cdist(
                        sample_coarse_boxes.float() / 1000.0,
                        valid_boxes.float() / 1000.0,
                        p=1,
                    )
                    giou_cost = 1.0 - pairwise_generalized_box_iou(
                        sample_coarse_boxes.float(), valid_boxes.float()
                    )
                    evidence_log = evidence_attention.float().clamp_min(1.0e-7).log()
                    target_log = box_evidence.float().clamp_min(1.0e-7).log()
                    attention_kl_cost = (
                        box_evidence[None].float()
                        * (target_log[None] - evidence_log[:, None])
                    ).sum(dim=-1)
                    cost = (
                        (2.0 if self.set_decoder else 1.0) * positive_focal_cost
                        + 5.0 * l1_cost
                        + 2.0 * giou_cost
                        + (0.0 if self.set_decoder else 1.0) * attention_kl_cost
                    )
                    matched_slots, matched_targets = hungarian_assignment(cost)
                    matched_for_sample[
                        valid_target_ordinals[matched_targets]
                    ] = matched_slots
                    gate_targets[matched_slots] = 1.0
                    predicted = sample_coarse_boxes[matched_slots].float()
                    expected = valid_boxes[matched_targets].float()
                    sample_l1 = F.l1_loss(
                        predicted / 1000.0, expected / 1000.0, reduction="mean"
                    )
                    sample_giou = (
                        1.0 - aligned_generalized_box_iou(predicted, expected)
                    ).mean()
                    box_l1_losses.append(sample_l1)
                    box_giou_losses.append(sample_giou)
                    box_l1_losses_by_task[task].append(sample_l1)
                    box_giou_losses_by_task[task].append(sample_giou)
                    matched_evidence = evidence_attention[matched_slots].float().clamp_min(1.0e-7)
                    matched_context = context_attention[matched_slots].float().clamp_min(1.0e-7)
                    target_evidence = box_evidence[matched_targets].float()
                    target_context = box_context[matched_targets].float()
                    evidence_ce = -(target_evidence * matched_evidence.log()).sum(dim=-1)
                    context_ce = -(target_context * matched_context.log()).sum(dim=-1)
                    evidence_entropy = -(
                        target_evidence * target_evidence.clamp_min(1.0e-7).log()
                    ).sum(dim=-1)
                    context_entropy = -(
                        target_context * target_context.clamp_min(1.0e-7).log()
                    ).sum(dim=-1)
                    sample_attention_kl = (
                        evidence_ce - evidence_entropy + context_ce - context_entropy
                    ).mean()
                    sample_attention_ce = (evidence_ce + context_ce).mean()
                    attention_kl_losses.append(sample_attention_kl)
                    attention_ce_values.append(sample_attention_ce)
                    evidence_losses.append(sample_attention_kl)
                    attention_losses_by_task[task].append(sample_attention_kl)
                    coarse_ious.append(
                        aligned_generalized_box_iou(predicted, expected).clamp(0.0, 1.0)
                    )
                    matched_counts.append(predicted.new_tensor(float(predicted.shape[0])))
                    unmatched_counts.append(
                        predicted.new_tensor(float(self.num_slots - predicted.shape[0]))
                    )
                else:
                    matched_counts.append(parameter_zero.detach().float())
                    unmatched_counts.append(parameter_zero.detach().float() + self.num_slots)
            elif target_boxes is not None and target_box_mask is not None:
                evidence_target, context_target, gate_targets = self._attention_targets(
                    target_boxes[sample_index],
                    target_box_mask[sample_index].bool(),
                    height,
                    width,
                    fused.device,
                )
                active = gate_targets.bool()
                if bool(active.any()):
                    evidence_ce = -(
                        evidence_target[active] * evidence_attention[active].float().clamp_min(1e-7).log()
                    ).sum(dim=-1)
                    context_ce = -(
                        context_target[active] * context_attention[active].float().clamp_min(1e-7).log()
                    ).sum(dim=-1)
                    sample_attention_loss = (evidence_ce + context_ce).mean()
                    evidence_losses.append(sample_attention_loss)
                    attention_losses_by_task[task].append(sample_attention_loss)
                valid_target_ordinals = target_box_mask[sample_index].bool().nonzero(
                    as_tuple=False
                ).flatten()[: self.num_slots]
                matched_count = int(valid_target_ordinals.numel())
                matched_for_sample[:matched_count] = torch.arange(
                    matched_count, device=fused.device, dtype=torch.long
                )
                matched_counts.append(parameter_zero.detach().float() + matched_count)
                unmatched_counts.append(
                    parameter_zero.detach().float() + self.num_slots - matched_count
                )
            else:
                matched_counts.append(parameter_zero.detach().float())
                unmatched_counts.append(parameter_zero.detach().float() + self.num_slots)
            gate_targets_list.append(gate_targets)
            matched_indices_list.append(matched_for_sample)

        relation_tokens = torch.stack(relation_tokens_list, dim=0)
        global_task_tokens = torch.stack(global_task_tokens_list, dim=0)
        slot_gate_logits = torch.stack(gate_logits_list, dim=0)
        image_gate_logits = torch.stack(image_gate_logits_list, dim=0)
        coarse_boxes = torch.stack(coarse_boxes_list, dim=0)
        scale_weights = torch.stack(scale_weights_list, dim=0)
        gate_probability = torch.sigmoid(slot_gate_logits)
        p_defect = torch.sigmoid(image_gate_logits)
        # Do not divide by the summed gate probability: that would cancel the
        # gate on negative samples.  A mean preserves defectness attenuation.
        relation_summary = (
            global_task_tokens if self.set_decoder else relation_tokens.mean(dim=1)
        )
        best_indices = gate_probability.argmax(dim=-1)
        best_relation = relation_tokens[
            torch.arange(num_samples, device=relation_tokens.device), best_indices
        ]

        gate_targets = None
        image_gate_targets = None
        image_gate_loss = None
        slot_gate_loss = None
        per_task_image_gate_loss: Dict[int, torch.Tensor] = {}
        per_task_slot_gate_loss: Dict[int, torch.Tensor] = {}
        if target_boxes is not None and target_box_mask is not None:
            gate_targets = torch.stack(gate_targets_list, dim=0).to(dtype=slot_gate_logits.dtype)
            image_gate_targets = target_box_mask.bool().any(dim=-1).to(
                dtype=image_gate_logits.dtype
            )
            valid_samples = relation_family >= 0
            if bool(valid_samples.any()):
                image_gate_loss = class_balanced_focal_loss(
                    image_gate_logits[valid_samples],
                    image_gate_targets[valid_samples],
                    defect_type[valid_samples],
                    gamma=self.focal_gamma,
                    beta=self.focal_beta,
                )
                slot_gate_loss = class_balanced_focal_loss(
                    slot_gate_logits[valid_samples],
                    gate_targets[valid_samples],
                    defect_type[valid_samples],
                    gamma=self.focal_gamma,
                    beta=self.focal_beta,
                )
                for task in range(self.num_defect_types):
                    task_mask = valid_samples & (defect_type == task)
                    if bool(task_mask.any()):
                        per_task_image_gate_loss[task] = class_balanced_focal_loss(
                            image_gate_logits[task_mask],
                            image_gate_targets[task_mask],
                            defect_type[task_mask],
                            gamma=self.focal_gamma,
                            beta=self.focal_beta,
                        )
                        per_task_slot_gate_loss[task] = class_balanced_focal_loss(
                            slot_gate_logits[task_mask],
                            gate_targets[task_mask],
                            defect_type[task_mask],
                            gamma=self.focal_gamma,
                            beta=self.focal_beta,
                        )
        attention_loss = torch.stack(evidence_losses).mean() if evidence_losses else None
        attention_kl_loss = (
            torch.stack(attention_kl_losses).mean() if attention_kl_losses else None
        )
        attention_ce_diagnostic = (
            torch.stack(attention_ce_values).mean() if attention_ce_values else None
        )
        box_l1_loss = torch.stack(box_l1_losses).mean() if box_l1_losses else None
        box_giou_loss = torch.stack(box_giou_losses).mean() if box_giou_losses else None
        per_task_attention_loss = {
            task: torch.stack(values).mean()
            for task, values in attention_losses_by_task.items()
            if values
        }
        per_task_box_l1_loss = {
            task: torch.stack(values).mean()
            for task, values in box_l1_losses_by_task.items() if values
        }
        per_task_box_giou_loss = {
            task: torch.stack(values).mean()
            for task, values in box_giou_losses_by_task.items() if values
        }
        projected_level_norms = torch.stack(
            [level.detach().float().norm(dim=-1).mean() for level in projected_levels]
        )
        projected_level_abs_max = torch.stack(
            [level.detach().float().abs().max() for level in projected_raw]
        )
        projected_level_saturation_fraction = torch.stack(
            [
                ((~torch.isfinite(level)) | (level.detach().float().abs() >= 128.0))
                .float()
                .mean()
                for level in projected_raw
            ]
        )
        projected_level_norm_ratio = (
            projected_level_norms.max()
            / projected_level_norms.min().clamp_min(1.0e-12)
        )
        fused_feature_norm = torch.stack(fused_norms).mean()
        relation_context_norm = torch.stack(relation_context_norms).mean()
        scale_entropy = -(
            scale_weights.float() * scale_weights.float().clamp_min(1.0e-7).log()
        ).sum(dim=-1)
        scale_batch_std = scale_weights.float().std(dim=0, unbiased=False)
        matched_slot_indices = torch.stack(matched_indices_list, dim=0)
        if coarse_ious:
            flat_coarse_iou = torch.cat([value.reshape(-1) for value in coarse_ious])
            coarse_iou_mean = flat_coarse_iou.mean()
            coarse_recall_03 = (flat_coarse_iou >= 0.3).float().mean()
            coarse_recall_05 = (flat_coarse_iou >= 0.5).float().mean()
        else:
            coarse_iou_mean = parameter_zero.detach().float()
            coarse_recall_03 = parameter_zero.detach().float()
            coarse_recall_05 = parameter_zero.detach().float()
        usage = gate_targets.float().sum(dim=0) if gate_targets is not None else None
        if usage is not None and float(usage.sum().item()) > 0:
            usage_probability = usage / usage.sum()
            slot_usage_entropy = -(
                usage_probability * usage_probability.clamp_min(1.0e-7).log()
            ).sum()
        else:
            slot_usage_entropy = parameter_zero.detach().float()

        return RelationPyramidOutput(
            relation_tokens=relation_tokens,
            relation_family=relation_family,
            p_defect=p_defect,
            image_gate_logits=image_gate_logits,
            slot_gate_logits=slot_gate_logits,
            gate_logits=slot_gate_logits,
            coarse_boxes=coarse_boxes,
            query_attention=tuple(attention_list),
            relation_summary=relation_summary,
            best_relation_token=best_relation,
            scale_weights=scale_weights,
            global_task_token=global_task_tokens,
            slot_objectness_logits=slot_gate_logits,
            gate_targets=gate_targets,
            image_gate_targets=image_gate_targets,
            projected_level_norms=projected_level_norms,
            projected_level_abs_max=projected_level_abs_max,
            projected_level_saturation_fraction=projected_level_saturation_fraction,
            projected_level_norm_ratio=projected_level_norm_ratio,
            fused_feature_norm=fused_feature_norm,
            relation_context_norm=relation_context_norm,
            per_task_gate_loss=per_task_image_gate_loss,
            per_task_image_gate_loss=per_task_image_gate_loss,
            per_task_slot_gate_loss=per_task_slot_gate_loss,
            per_task_attention_loss=per_task_attention_loss,
            per_task_box_l1_loss=per_task_box_l1_loss,
            per_task_box_giou_loss=per_task_box_giou_loss,
            gate_loss=image_gate_loss,
            image_gate_loss=image_gate_loss,
            slot_gate_loss=slot_gate_loss,
            attention_loss=attention_loss,
            attention_kl_loss=attention_kl_loss,
            attention_ce_diagnostic=attention_ce_diagnostic,
            box_l1_loss=box_l1_loss,
            box_giou_loss=box_giou_loss,
            matched_slot_indices=matched_slot_indices,
            scale_entropy=scale_entropy,
            scale_batch_std=scale_batch_std,
            coarse_iou_mean=coarse_iou_mean,
            coarse_recall_03=coarse_recall_03,
            coarse_recall_05=coarse_recall_05,
            matched_slots=torch.stack(matched_counts).mean(),
            unmatched_slots=torch.stack(unmatched_counts).mean(),
            slot_usage_entropy=slot_usage_entropy,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def pbd_prediction_positions(
    input_ids: torch.Tensor,
    sub_sample_lengths: torch.Tensor,
    box_start_token_id: int,
    text_mask_token_id: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select PBD prediction positions without crossing packed samples.

    A normal autoregressive ``<box>`` contributes its anchor only.  A complete
    MTP block ``[<box>, <text_mask> * (block_size - 1)]`` contributes every
    position in that block.  Returned indices address the flattened input and
    are paired with their packed-sample indices.
    """

    if int(block_size) <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    flat_ids = input_ids.reshape(-1)
    lengths = torch.as_tensor(
        sub_sample_lengths,
        device=flat_ids.device,
        dtype=torch.long,
    ).reshape(-1)
    if bool((lengths < 0).any()):
        raise ValueError("sub_sample_lengths cannot contain negative values")
    if int(lengths.sum().item()) != flat_ids.numel():
        raise ValueError(
            "PBD packed length mismatch: "
            f"input_tokens={flat_ids.numel()}, "
            f"sub_sample_lengths={lengths.tolist()}"
        )

    positions: List[torch.Tensor] = []
    sample_indices: List[torch.Tensor] = []
    sample_start = 0
    for sample_index, raw_length in enumerate(lengths.tolist()):
        sample_end = sample_start + int(raw_length)
        local_anchors = (
            flat_ids[sample_start:sample_end] == int(box_start_token_id)
        ).nonzero(as_tuple=False).flatten()
        mtp_anchors = set()
        for raw_local_anchor in local_anchors.tolist():
            candidate = sample_start + int(raw_local_anchor)
            candidate_end = candidate + int(block_size)
            if candidate_end > sample_end:
                continue
            following = flat_ids[candidate + 1 : candidate_end]
            if following.numel() == int(block_size) - 1 and bool(
                (following == int(text_mask_token_id)).all()
            ):
                mtp_anchors.add(candidate)
        for local_anchor in local_anchors.tolist():
            anchor = sample_start + int(local_anchor)
            # Generation keeps the newly emitted token in history and copies
            # it once as the MTP anchor.  For ``[..., <box>, <box>, mask...]``
            # the first box is history, not a seventh prediction position.
            if anchor + 1 in mtp_anchors:
                continue
            width = int(block_size) if anchor in mtp_anchors else 1
            selected = torch.arange(
                anchor,
                anchor + width,
                device=flat_ids.device,
                dtype=torch.long,
            )
            positions.append(selected)
            sample_indices.append(
                torch.full_like(selected, sample_index, dtype=torch.long)
            )
        sample_start = sample_end

    if not positions:
        empty = torch.zeros(0, device=flat_ids.device, dtype=torch.long)
        return empty, empty
    return torch.cat(positions), torch.cat(sample_indices)


def pbd_active_delta_norm(
    hidden_before: torch.Tensor,
    hidden_after: torch.Tensor,
    active_positions: torch.Tensor,
) -> torch.Tensor:
    """Mean PBD delta norm over modified positions only."""

    if hidden_before.shape != hidden_after.shape:
        raise ValueError(
            "PBD hidden shapes disagree: "
            f"before={tuple(hidden_before.shape)}, after={tuple(hidden_after.shape)}"
        )
    flat_before = hidden_before.reshape(-1, hidden_before.shape[-1])
    flat_after = hidden_after.reshape(-1, hidden_after.shape[-1])
    active_positions = active_positions.reshape(-1).long()
    if active_positions.numel() == 0:
        return flat_before.new_zeros((), dtype=torch.float32)
    delta = flat_after[active_positions] - flat_before[active_positions]
    return delta.float().norm(dim=-1).mean()


def pbd_prediction_groups(
    input_ids: torch.Tensor,
    sub_sample_lengths: torch.Tensor,
    box_start_token_id: int,
    text_mask_token_id: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return active positions plus sample/anchor/offset routing identities.

    This is the single grouping rule used by dynamic PBD in training and
    generation.  Anchor ordinals restart at every packed-sample boundary.
    """

    positions, samples = pbd_prediction_positions(
        input_ids,
        sub_sample_lengths,
        box_start_token_id,
        text_mask_token_id,
        block_size,
    )
    if positions.numel() == 0:
        return positions, samples, positions.clone(), positions.clone()
    flat_ids = input_ids.reshape(-1)
    anchor_ordinals = torch.zeros_like(positions)
    offsets = torch.zeros_like(positions)
    per_sample_anchor = [0 for _ in range(int(samples.max().item()) + 1)]
    cursor = 0
    while cursor < positions.numel():
        sample = int(samples[cursor].item())
        anchor = int(positions[cursor].item())
        width = 1
        while (
            cursor + width < positions.numel()
            and int(samples[cursor + width].item()) == sample
            and int(positions[cursor + width].item()) == anchor + width
            and int(flat_ids[anchor].item()) == int(box_start_token_id)
            and int(flat_ids[anchor + width].item()) == int(text_mask_token_id)
        ):
            width += 1
        anchor_ordinals[cursor : cursor + width] = per_sample_anchor[sample]
        offsets[cursor : cursor + width] = torch.arange(
            width, device=positions.device, dtype=torch.long
        )
        per_sample_anchor[sample] += 1
        cursor += width
    return positions, samples, anchor_ordinals, offsets


def coordinate_bridge_prediction_groups(
    input_ids: torch.Tensor,
    sub_sample_lengths: torch.Tensor,
    active_positions: torch.Tensor,
    active_samples: torch.Tensor,
    active_anchor_ordinals: torch.Tensor,
    active_offsets: torch.Tensor,
    selected_coarse_boxes: torch.Tensor,
    coord_start_token_id: int,
    coord_end_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select all four coordinate-prediction states for AR and MTP.

    PBD itself deliberately remains anchor-only in ordinary AR.  The geometry
    bridge has a different contract: the hidden state at ``<box>`` predicts x1
    and the next three coordinate-token states predict y1/x2/y2.  MTP already
    exposes those states as offsets 0..3, while a teacher-forced AR sequence
    needs them expanded from the anchor.  Expansion is bounded by the current
    packed sub-sample and only follows observed coordinate tokens.
    """

    if not (
        active_positions.numel()
        == active_samples.numel()
        == active_anchor_ordinals.numel()
        == active_offsets.numel()
        == selected_coarse_boxes.shape[0]
    ):
        raise ValueError("coordinate bridge PBD metadata lengths disagree")
    if active_positions.numel() == 0:
        empty = active_positions.reshape(-1).long()
        return empty, empty.clone(), empty.clone(), selected_coarse_boxes[:0]

    flat_ids = input_ids.reshape(-1)
    lengths = sub_sample_lengths.reshape(-1).long()
    if int(lengths.sum().item()) != flat_ids.numel():
        raise ValueError(
            "coordinate bridge packed lengths disagree with input_ids: "
            f"sum={int(lengths.sum().item())}, tokens={flat_ids.numel()}"
        )
    starts = torch.cat(
        (
            lengths.new_zeros(1),
            lengths.cumsum(dim=0)[:-1],
        )
    )
    ends = starts + lengths

    output_positions: List[int] = []
    output_samples: List[int] = []
    output_offsets: List[int] = []
    output_boxes: List[torch.Tensor] = []
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index in range(active_positions.numel()):
        group = (
            int(active_samples[index].item()),
            int(active_anchor_ordinals[index].item()),
        )
        groups.setdefault(group, []).append(index)

    for (sample, _), indices in groups.items():
        ordered = sorted(indices, key=lambda index: int(active_offsets[index].item()))
        by_offset = {int(active_offsets[index].item()): index for index in ordered}
        anchor_index = by_offset.get(0)
        if anchor_index is None:
            continue
        anchor_position = int(active_positions[anchor_index].item())
        anchor_box = selected_coarse_boxes[anchor_index]

        # MTP exposes the entire block through the shared PBD selector.
        if any(offset > 0 for offset in by_offset):
            for offset in range(4):
                index = by_offset.get(offset)
                if index is None:
                    break
                output_positions.append(int(active_positions[index].item()))
                output_samples.append(sample)
                output_offsets.append(offset)
                output_boxes.append(selected_coarse_boxes[index])
            continue

        # AR: <box> hidden predicts x1; hidden states on coord1/2/3 predict the
        # remaining three coordinates.  Never inspect or cross another packed
        # sample, and stop as soon as the teacher-forced coordinate chain ends.
        output_positions.append(anchor_position)
        output_samples.append(sample)
        output_offsets.append(0)
        output_boxes.append(anchor_box)
        sample_end = int(ends[sample].item())
        for offset in range(1, 4):
            position = anchor_position + offset
            if position >= sample_end:
                break
            token = int(flat_ids[position].item())
            if token < int(coord_start_token_id) or token > int(coord_end_token_id):
                break
            output_positions.append(position)
            output_samples.append(sample)
            output_offsets.append(offset)
            output_boxes.append(anchor_box)

    device = active_positions.device
    if not output_positions:
        empty = active_positions[:0]
        return empty, empty.clone(), empty.clone(), selected_coarse_boxes[:0]
    return (
        torch.tensor(output_positions, device=device, dtype=torch.long),
        torch.tensor(output_samples, device=device, dtype=torch.long),
        torch.tensor(output_offsets, device=device, dtype=torch.long),
        torch.stack(output_boxes, dim=0),
    )


def apply_coordinate_logit_prior(
    logits: torch.Tensor,
    active_positions: torch.Tensor,
    active_offsets: torch.Tensor,
    active_samples: torch.Tensor,
    selected_coarse_boxes: torch.Tensor,
    defect_type: torch.Tensor,
    task_lambdas: torch.Tensor,
    coord_start_token_id: int,
    coord_end_token_id: int,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Add a Gaussian coarse-geometry prior only to coordinate-token logits."""

    if float(sigma) <= 0.0:
        raise ValueError("coordinate prior sigma must be positive")
    output = logits.clone()
    flat = output.reshape(-1, output.shape[-1])
    coord_count = int(coord_end_token_id) - int(coord_start_token_id) + 1
    if coord_count <= 0:
        raise ValueError("invalid coordinate token range")
    coordinate_values = torch.linspace(
        0.0, 1000.0, coord_count, device=flat.device, dtype=torch.float32
    )
    for index in range(active_positions.numel()):
        dimension = int(active_offsets[index].item())
        if dimension < 0 or dimension >= 4:
            continue
        position = int(active_positions[index].item())
        sample = int(active_samples[index].item())
        task = int(defect_type[sample].clamp(0, task_lambdas.numel() - 1).item())
        strength = task_lambdas[task].float()
        if not bool((strength.detach().abs() > 0).item()):
            continue
        center = selected_coarse_boxes[index, dimension].float()
        width = 1000.0 * float(sigma)
        prior = -strength * (coordinate_values - center).square() / (2.0 * width * width)
        flat[position, int(coord_start_token_id) : int(coord_end_token_id) + 1] += prior.to(
            dtype=flat.dtype
        )
    return output


def coordinate_gaussian_prior(
    active_offsets: torch.Tensor,
    active_samples: torch.Tensor,
    selected_coarse_boxes: torch.Tensor,
    defect_type: torch.Tensor,
    task_lambdas: torch.Tensor,
    coordinate_count: int,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Vectorized coordinate-vocabulary prior shared by train and inference."""

    count = int(coordinate_count)
    if count <= 0 or float(sigma) <= 0.0:
        raise ValueError("coordinate_count and sigma must be positive")
    values = torch.linspace(
        0.0,
        1000.0,
        count,
        device=selected_coarse_boxes.device,
        dtype=torch.float32,
    )
    result = selected_coarse_boxes.new_zeros(
        (active_offsets.numel(), count), dtype=torch.float32
    )
    valid = (active_offsets >= 0) & (active_offsets < 4)
    if not bool(valid.any()):
        return result
    indices = valid.nonzero(as_tuple=False).flatten()
    dimensions = active_offsets[indices].long()
    samples = active_samples[indices].long()
    tasks = defect_type[samples].long().clamp(0, task_lambdas.numel() - 1)
    strengths = task_lambdas[tasks].float()
    centers = selected_coarse_boxes[indices, dimensions].float()
    width = 1000.0 * float(sigma)
    result[indices] = (
        -strengths[:, None]
        * (values[None, :] - centers[:, None]).square()
        / (2.0 * width * width)
    )
    return result


def apply_soft_gate_logit_prior(
    logits: torch.Tensor,
    p_defect: torch.Tensor,
    defect_type: torch.Tensor,
    task_betas: torch.Tensor,
    box_token_id: int,
    none_token_id: int,
) -> torch.Tensor:
    """Apply task-specific soft none/box evidence without hard early exit."""

    output = logits.clone()
    if logits.ndim < 2:
        raise ValueError("soft gate expects logits with a vocabulary dimension")
    batch = logits.shape[0]
    for sample in range(batch):
        task = int(defect_type[sample].clamp(0, task_betas.numel() - 1).item())
        beta = task_betas[task].to(dtype=output.dtype, device=output.device)
        probability = p_defect[sample].to(dtype=output.dtype, device=output.device)
        output[sample, ..., int(box_token_id)] += beta * probability
        output[sample, ..., int(none_token_id)] += beta * (1.0 - probability)
    return output


def replace_pbd_active_logits(
    destination_logits: torch.Tensor,
    replacement_logits: torch.Tensor,
    active_positions: torch.Tensor,
) -> torch.Tensor:
    """Replace PBD positions while preserving the decoder logits dtype/device.

    Some attention implementations expose decoder logits as float32 even when
    the LM head/PBD hidden path is bfloat16. PyTorch indexed assignment does
    not perform this conversion implicitly, so the source is cast explicitly
    to the destination contract before the write.
    """

    if destination_logits.shape != replacement_logits.shape:
        raise ValueError(
            "PBD replacement logits shape mismatch: "
            f"destination={tuple(destination_logits.shape)}, "
            f"replacement={tuple(replacement_logits.shape)}"
        )
    output = destination_logits.clone()
    flat_output = output.reshape(-1, output.shape[-1])
    flat_replacement = replacement_logits.reshape(
        -1, replacement_logits.shape[-1]
    )
    positions = active_positions.reshape(-1).to(
        device=flat_output.device, dtype=torch.long
    )
    if positions.numel() == 0:
        return output
    if int(positions.min().item()) < 0 or int(positions.max().item()) >= flat_output.shape[0]:
        raise IndexError(
            "PBD active position is outside logits: "
            f"positions={positions.tolist()}, tokens={flat_output.shape[0]}"
        )
    selected = flat_replacement[positions.to(flat_replacement.device)].to(
        device=flat_output.device,
        dtype=flat_output.dtype,
    )
    flat_output[positions] = selected
    return output


@dataclass
class PBDForwardOutput:
    hidden_states: torch.Tensor
    box_anchor_hidden: torch.Tensor
    box_anchor_samples: torch.Tensor
    active_positions: torch.Tensor
    active_samples: torch.Tensor
    active_anchor_ordinals: Optional[torch.Tensor] = None
    active_offsets: Optional[torch.Tensor] = None
    selected_slot_indices: Optional[torch.Tensor] = None
    routing_weights: Optional[torch.Tensor] = None
    selected_coarse_boxes: Optional[torch.Tensor] = None
    final_slot_usage: Optional[torch.Tensor] = None
    coverage_loss: Optional[torch.Tensor] = None
    unique_slot_count: Optional[torch.Tensor] = None
    duplicate_slot_rate: Optional[torch.Tensor] = None


class RelationToPBD(nn.Module):
    """Inject relation evidence into semantic/negative and box anchor states."""

    def __init__(
        self,
        relation_hidden_size: int,
        language_hidden_size: int,
        dynamic_slot: bool = False,
        overlap_adapter: bool = False,
        coordinate_bridge: bool = False,
        num_defect_types: int = 5,
        adapter_rank: int = 8,
        task_experts: bool = False,
        task_expert_rank: int = 8,
        separate_global_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(relation_hidden_size),
            nn.Linear(relation_hidden_size, language_hidden_size, bias=False),
        )
        self.box_projection = nn.Sequential(
            nn.LayerNorm(relation_hidden_size),
            nn.Linear(relation_hidden_size, language_hidden_size, bias=False),
        )
        # Small non-zero initialization lets the new path learn while preserving
        # the pretrained decoder distribution at step zero.
        self.semantic_scale = nn.Parameter(torch.tensor(0.01))
        self.box_scale = nn.Parameter(torch.tensor(0.01))
        self.dynamic_slot = bool(dynamic_slot)
        self.overlap_adapter = bool(overlap_adapter)
        self.coordinate_bridge = bool(coordinate_bridge)
        self.task_experts = bool(task_experts)
        self.separate_global_geometry = bool(separate_global_geometry)
        if self.task_experts:
            self.semantic_task_experts = TaskRoutedExpertBank(
                relation_hidden_size,
                rank=task_expert_rank,
                num_defect_types=num_defect_types,
            )
            self.geometry_task_experts = TaskRoutedExpertBank(
                relation_hidden_size,
                rank=task_expert_rank,
                num_defect_types=num_defect_types,
            )
        if self.dynamic_slot:
            self.router_query = nn.Linear(language_hidden_size, relation_hidden_size, bias=False)
            self.router_key = nn.Linear(relation_hidden_size, relation_hidden_size, bias=False)
            self.router_value = nn.Linear(relation_hidden_size, relation_hidden_size, bias=False)
            self.coverage_gamma = nn.Parameter(torch.tensor(1.0))
        if self.overlap_adapter:
            self.overlap_adapter_down = nn.Linear(relation_hidden_size, adapter_rank, bias=False)
            self.overlap_adapter_up = nn.Linear(adapter_rank, relation_hidden_size, bias=False)
            nn.init.zeros_(self.overlap_adapter_up.weight)
        if self.coordinate_bridge:
            self.coord_prior_lambda = nn.Parameter(torch.zeros(num_defect_types))

    def enhance_prediction_hidden(
        self,
        hidden_states: torch.Tensor,
        relation_summary: torch.Tensor,
        best_relation_token: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the same semantic/box PBD delta used by training and generation."""

        semantic_delta = self.semantic_projection(
            torch.nan_to_num(
                relation_summary, nan=0.0, posinf=32.0, neginf=-32.0
            ).clamp(-32.0, 32.0)
        )
        box_delta = self.box_projection(
            torch.nan_to_num(
                best_relation_token, nan=0.0, posinf=32.0, neginf=-32.0
            ).clamp(-32.0, 32.0)
        )
        while semantic_delta.ndim < hidden_states.ndim:
            semantic_delta = semantic_delta.unsqueeze(1)
            box_delta = box_delta.unsqueeze(1)
        return (
            hidden_states
            + torch.nan_to_num(self.semantic_scale.tanh(), nan=0.0) * semantic_delta
            + torch.nan_to_num(self.box_scale.tanh(), nan=0.0) * box_delta
        )

    def enhance_routed_hidden(
        self,
        hidden_states: torch.Tensor,
        relation_summary: torch.Tensor,
        selected_relation_token: torch.Tensor,
        active_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """M3.1 global semantics at anchors, slot geometry across the box span."""

        if not self.separate_global_geometry:
            return self.enhance_prediction_hidden(
                hidden_states, relation_summary, selected_relation_token
            )
        semantic_delta = self.semantic_projection(relation_summary)
        box_delta = self.box_projection(selected_relation_token)
        anchor = (active_offsets == 0).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ).unsqueeze(-1)
        return (
            hidden_states
            + anchor
            * torch.nan_to_num(self.semantic_scale.tanh(), nan=0.0)
            * semantic_delta
            + torch.nan_to_num(self.box_scale.tanh(), nan=0.0) * box_delta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        sub_sample_lengths: torch.Tensor,
        relation_summary: torch.Tensor,
        best_relation_token: torch.Tensor,
        box_start_token_id: int,
        text_mask_token_id: int,
        block_size: int,
        relation_tokens: Optional[torch.Tensor] = None,
        slot_gate_logits: Optional[torch.Tensor] = None,
        slot_objectness_logits: Optional[torch.Tensor] = None,
        coarse_boxes: Optional[torch.Tensor] = None,
        matched_slot_indices: Optional[torch.Tensor] = None,
        defect_type: Optional[torch.Tensor] = None,
        initial_slot_usage: Optional[torch.Tensor] = None,
    ) -> PBDForwardOutput:
        enhanced = hidden_states.clone()
        flat_hidden = enhanced.reshape(-1, enhanced.shape[-1])
        flat_ids = input_ids.reshape(-1)
        if flat_hidden.shape[0] != flat_ids.numel():
            raise ValueError(
                "PBD hidden/input token counts disagree: "
                f"hidden_tokens={flat_hidden.shape[0]}, input_tokens={flat_ids.numel()}"
            )
        active_positions, active_samples, active_anchor_ordinals, active_offsets = pbd_prediction_groups(
            input_ids=input_ids,
            sub_sample_lengths=sub_sample_lengths,
            box_start_token_id=box_start_token_id,
            text_mask_token_id=text_mask_token_id,
            block_size=block_size,
        )
        selected_slot_indices = torch.full_like(active_positions, -1)
        routing_weights = None
        selected_coarse_boxes = hidden_states.new_zeros((active_positions.numel(), 4))
        final_slot_usage = None
        coverage_loss = hidden_states.new_zeros((), dtype=torch.float32)
        unique_slot_count = hidden_states.new_zeros((), dtype=torch.float32)
        duplicate_slot_rate = hidden_states.new_zeros((), dtype=torch.float32)
        if active_positions.numel() > 0:
            required_samples = int(active_samples.max().item()) + 1
            if relation_summary.shape[0] < required_samples:
                raise ValueError(
                    "relation_summary has fewer samples than the packed PBD selection"
                )
            if best_relation_token.shape[0] < required_samples:
                raise ValueError(
                    "best_relation_token has fewer samples than the packed PBD selection"
                )
            safe_summaries = torch.nan_to_num(
                relation_summary[active_samples], nan=0.0, posinf=32.0, neginf=-32.0
            ).clamp(-32.0, 32.0)
            safe_best_tokens = torch.nan_to_num(
                best_relation_token[active_samples], nan=0.0, posinf=32.0, neginf=-32.0
            ).clamp(-32.0, 32.0)
            if slot_objectness_logits is not None:
                if slot_gate_logits is not None and slot_gate_logits is not slot_objectness_logits:
                    raise ValueError("provide slot objectness through only one argument")
                slot_gate_logits = slot_objectness_logits
            if self.task_experts:
                if defect_type is None:
                    raise ValueError("m31 PBD task experts require defect_type")
                active_tasks = defect_type.reshape(-1)[active_samples].long()
                safe_summaries = self.semantic_task_experts(
                    safe_summaries, active_tasks
                )
            if self.dynamic_slot:
                if relation_tokens is None or slot_gate_logits is None:
                    raise ValueError("dynamic slot PBD requires relation_tokens and slot_gate_logits")
                num_samples, num_slots, relation_dim = relation_tokens.shape
                if initial_slot_usage is None:
                    final_slot_usage = relation_tokens.new_zeros((num_samples, num_slots)).float()
                else:
                    final_slot_usage = initial_slot_usage.to(
                        device=relation_tokens.device, dtype=torch.float32
                    ).clone()
                routed_tokens = safe_best_tokens.clone()
                all_weights = relation_tokens.new_zeros(
                    (active_positions.numel(), num_slots), dtype=torch.float32
                )
                soft_anchor_weights: List[torch.Tensor] = []
                seen_route_weights: Dict[Tuple[int, int], torch.Tensor] = {}
                seen_selected_slots: Dict[Tuple[int, int], int] = {}
                for active_index in range(active_positions.numel()):
                    sample = int(active_samples[active_index].item())
                    ordinal = int(active_anchor_ordinals[active_index].item())
                    group = (sample, ordinal)
                    if group not in seen_route_weights:
                        query = self.router_query(flat_hidden[active_positions[active_index]]).float()
                        keys = self.router_key(relation_tokens[sample]).float()
                        route_logits = (
                            query @ keys.transpose(0, 1) / math.sqrt(float(relation_dim))
                            + F.logsigmoid(slot_gate_logits[sample].float())
                            - self.coverage_gamma.float().abs() * final_slot_usage[sample]
                        )
                        if self.task_experts:
                            available = final_slot_usage[sample] <= 0
                            if bool(available.any()):
                                route_logits = route_logits.masked_fill(~available, -1.0e4)
                        soft_weights = route_logits.softmax(dim=-1)
                        chosen = int(soft_weights.argmax().item())
                        if matched_slot_indices is not None and ordinal < matched_slot_indices.shape[1]:
                            teacher_slot = int(matched_slot_indices[sample, ordinal].item())
                            if teacher_slot >= 0:
                                coverage_loss = coverage_loss + F.cross_entropy(
                                    route_logits.unsqueeze(0),
                                    torch.tensor([teacher_slot], device=route_logits.device),
                                )
                                chosen = teacher_slot
                        selected = F.one_hot(
                            torch.tensor(chosen, device=route_logits.device), num_slots
                        ).float()
                        # The first TC-MSED experiment keeps the differentiable
                        # soft mixture.  The discrete (teacher/argmax) slot is
                        # used only for coverage state, coarse geometry and
                        # diagnostics.  This lets LM loss train q/k/v routing;
                        # straight-through top-1 remains a separate ablation.
                        seen_route_weights[group] = (
                            selected if self.task_experts else soft_weights
                        )
                        seen_selected_slots[group] = chosen
                        soft_anchor_weights.append(soft_weights)
                        # ``final_slot_usage[sample]`` participates in the
                        # route_logits graph for this and previous anchors.
                        # Mutating that storage in place invalidates autograd's
                        # saved version counter.  Carry coverage state forward
                        # functionally instead.
                        updated_usage = final_slot_usage.clone()
                        updated_usage[sample] = final_slot_usage[sample] + selected
                        final_slot_usage = updated_usage
                    route_weights = seen_route_weights[group]
                    all_weights[active_index] = route_weights
                    selected_index = seen_selected_slots[group]
                    selected_slot_indices[active_index] = selected_index
                    # K=1 is a strict degeneration to the legacy selected token;
                    # do not add an otherwise unidentifiable value projection.
                    if self.task_experts:
                        token = self.router_value(
                            relation_tokens[sample, selected_index]
                        )
                    elif num_slots == 1:
                        token = relation_tokens[sample, 0]
                    else:
                        projected_tokens = self.router_value(relation_tokens[sample])
                        token = route_weights.to(dtype=projected_tokens.dtype) @ projected_tokens
                    if self.overlap_adapter and defect_type is not None and int(defect_type[sample].item()) == 2:
                        token = token + self.overlap_adapter_up(
                            self.overlap_adapter_down(token)
                        )
                    routed_tokens[active_index] = token
                    if coarse_boxes is not None:
                        selected_coarse_boxes[active_index] = coarse_boxes[sample, selected_index]
                if len(soft_anchor_weights) > 1:
                    by_sample: Dict[int, List[torch.Tensor]] = {}
                    for (sample, _), weights in zip(
                        seen_route_weights.keys(), soft_anchor_weights
                    ):
                        by_sample.setdefault(sample, []).append(weights)
                    overlap_terms = []
                    for values in by_sample.values():
                        for left in range(len(values)):
                            for right in range(left + 1, len(values)):
                                overlap_terms.append((values[left] * values[right]).sum())
                    if overlap_terms:
                        coverage_loss = coverage_loss + torch.stack(overlap_terms).mean()
                routing_weights = all_weights
                safe_best_tokens = routed_tokens
                if self.task_experts:
                    active_tasks = defect_type.reshape(-1)[active_samples].long()
                    safe_best_tokens = self.geometry_task_experts(
                        safe_best_tokens, active_tasks
                    )
                anchor_selected = []
                for group, selected_index in seen_selected_slots.items():
                    anchor_selected.append((group[0], int(selected_index)))
                if anchor_selected:
                    total = len(anchor_selected)
                    unique_per_sample = []
                    duplicates = 0
                    for sample in sorted({item[0] for item in anchor_selected}):
                        values = [slot for owner, slot in anchor_selected if owner == sample]
                        unique_per_sample.append(float(len(set(values))))
                        duplicates += len(values) - len(set(values))
                    unique_slot_count = hidden_states.new_tensor(unique_per_sample).mean()
                    duplicate_slot_rate = hidden_states.new_tensor(float(duplicates / max(total, 1)))
                    coverage_loss = coverage_loss / max(total, 1)
            active_hidden = flat_hidden.index_select(0, active_positions)
            enhanced_active_hidden = self.enhance_routed_hidden(
                active_hidden,
                safe_summaries,
                safe_best_tokens,
                active_offsets,
            )
            # The replacement is computed from ``flat_hidden`` itself.  An
            # indexed in-place write into that storage invalidates the views
            # saved by autograd (and fails during dynamic-slot backward).
            # ``index_copy`` returns fresh storage and keeps both AR and MTP
            # routing fully differentiable.
            flat_hidden = flat_hidden.index_copy(
                0, active_positions, enhanced_active_hidden
            )
            enhanced = flat_hidden.reshape_as(hidden_states)

        anchor_mask = flat_ids[active_positions] == int(box_start_token_id)
        anchor_positions = active_positions[anchor_mask]
        if anchor_positions.numel() > 0:
            box_anchor_hidden = flat_hidden[anchor_positions]
            box_anchor_samples = active_samples[anchor_mask]
        else:
            box_anchor_hidden = hidden_states.new_zeros((0, hidden_states.shape[-1]))
            box_anchor_samples = torch.zeros(0, device=hidden_states.device, dtype=torch.long)
        return PBDForwardOutput(
            hidden_states=enhanced,
            box_anchor_hidden=box_anchor_hidden,
            box_anchor_samples=box_anchor_samples,
            active_positions=active_positions,
            active_samples=active_samples,
            active_anchor_ordinals=active_anchor_ordinals,
            active_offsets=active_offsets,
            selected_slot_indices=selected_slot_indices,
            routing_weights=routing_weights,
            selected_coarse_boxes=selected_coarse_boxes,
            final_slot_usage=final_slot_usage,
            coverage_loss=coverage_loss,
            unique_slot_count=unique_slot_count,
            duplicate_slot_rate=duplicate_slot_rate,
        )
