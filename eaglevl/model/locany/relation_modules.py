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
    gate_logits: torch.Tensor
    coarse_boxes: torch.Tensor
    query_attention: Tuple[torch.Tensor, ...]
    relation_summary: torch.Tensor
    best_relation_token: torch.Tensor
    scale_weights: torch.Tensor
    gate_targets: Optional[torch.Tensor] = None
    projected_level_norms: Optional[torch.Tensor] = None
    fused_feature_norm: Optional[torch.Tensor] = None
    relation_context_norm: Optional[torch.Tensor] = None
    per_task_gate_loss: Optional[Dict[int, torch.Tensor]] = None
    per_task_attention_loss: Optional[Dict[int, torch.Tensor]] = None
    gate_loss: Optional[torch.Tensor] = None
    attention_loss: Optional[torch.Tensor] = None


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
    sample_pos_weight = pos_weight[class_ids].to(dtype=dtype).unsqueeze(-1)
    sample_neg_weight = neg_weight[class_ids].to(dtype=dtype).unsqueeze(-1)
    balance_weight = torch.where(targets > 0.5, sample_pos_weight, sample_neg_weight)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    return (balance_weight * (1.0 - p_t).pow(gamma) * bce).mean()


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
    ) -> None:
        super().__init__()
        self.detail_hidden_size = detail_hidden_size
        self.num_slots = num_slots
        self.num_families = num_families
        self.num_defect_types = num_defect_types
        self.focal_gamma = focal_gamma
        self.focal_beta = focal_beta

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
        initial_scale_weights = torch.tensor(
            [
                [0.50, 0.35, 0.15],  # boundary
                [0.15, 0.40, 0.45],  # pairwise
                [0.40, 0.25, 0.35],  # text
                [0.10, 0.20, 0.70],  # presence
            ],
            dtype=torch.float32,
        )
        self.scale_logits = nn.Parameter(initial_scale_weights.log())

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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.evidence_queries, std=0.02)
        nn.init.normal_(self.context_queries, std=0.02)
        # Start conservatively: an unseen slot should prefer non-defect.
        for head in self.gate_heads:
            nn.init.constant_(head[-1].bias, -2.0)

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
        if image_flags is None:
            image_flags = torch.ones(num_samples, device=relation_family.device, dtype=torch.long)
        else:
            image_flags = image_flags.reshape(-1).long()
        if image_flags.numel() != num_samples:
            raise ValueError(
                f"image_flags ({image_flags.numel()}) and relation_family ({num_samples}) disagree"
            )

        projected_levels = [
            self._sanitize(projection(self._sanitize(level)))
            for projection, level in zip(self.level_projections, pyramid_features)
        ]
        image_features = self._split_features(projected_levels, grid_hws)
        scale_weights_all = self.scale_logits.softmax(dim=-1)

        relation_tokens_list: List[torch.Tensor] = []
        gate_logits_list: List[torch.Tensor] = []
        coarse_boxes_list: List[torch.Tensor] = []
        attention_list: List[torch.Tensor] = []
        scale_weights_list: List[torch.Tensor] = []
        gate_targets_list: List[torch.Tensor] = []
        evidence_losses: List[torch.Tensor] = []
        attention_losses_by_task: Dict[int, List[torch.Tensor]] = {
            task: [] for task in range(self.num_defect_types)
        }
        fused_norms: List[torch.Tensor] = []
        relation_context_norms: List[torch.Tensor] = []

        image_index = 0
        parameter_zero = self.evidence_queries.sum() * 0.0
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
                gate_logits_list.append(torch.full((self.num_slots,), -20.0, device=empty_tokens.device, dtype=empty_tokens.dtype))
                coarse_boxes_list.append(torch.zeros(self.num_slots, 4, device=empty_tokens.device, dtype=empty_tokens.dtype))
                attention_list.append(torch.zeros(2, self.num_slots, 1, device=empty_tokens.device, dtype=empty_tokens.dtype))
                scale_weights_list.append(scale_weights_all[family])
                gate_targets_list.append(torch.zeros(self.num_slots, device=empty_tokens.device, dtype=empty_tokens.dtype))
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
            weights = scale_weights_all[family]
            fused = sum(weights[level] * levels[level] for level in range(3))
            fused = self._sanitize(self.token_norm(self._sanitize(fused)))
            fused_norms.append(fused.detach().float().norm(dim=-1).mean())
            keys = self._sanitize(self.key_projection(fused))
            values = self._sanitize(self.value_projection(fused))

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
            relation_context_norms.append(
                adapted_relation.detach().float().norm(dim=-1).mean()
            )
            logits_all = torch.stack(
                [head(adapted_relation).squeeze(-1) for head in self.gate_heads], dim=0
            )
            gate_logits = self._sanitize(logits_all[family], limit=30.0)
            gated_relation = self._sanitize(
                torch.sigmoid(gate_logits).unsqueeze(-1) * adapted_relation,
                limit=32.0,
            )

            relation_tokens_list.append(gated_relation)
            gate_logits_list.append(gate_logits)
            coarse_boxes_list.append(self._coarse_boxes(evidence_attention, height, width))
            attention_list.append(torch.stack((evidence_attention, context_attention), dim=0))
            scale_weights_list.append(weights)

            gate_targets = torch.zeros(self.num_slots, device=fused.device, dtype=fused.dtype)
            if target_boxes is not None and target_box_mask is not None:
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
            gate_targets_list.append(gate_targets)

        relation_tokens = torch.stack(relation_tokens_list, dim=0)
        gate_logits = torch.stack(gate_logits_list, dim=0)
        coarse_boxes = torch.stack(coarse_boxes_list, dim=0)
        scale_weights = torch.stack(scale_weights_list, dim=0)
        gate_probability = torch.sigmoid(gate_logits)
        p_defect = gate_probability.max(dim=-1).values
        # Do not divide by the summed gate probability: that would cancel the
        # gate on negative samples.  A mean preserves defectness attenuation.
        relation_summary = relation_tokens.mean(dim=1)
        best_indices = gate_probability.argmax(dim=-1)
        best_relation = relation_tokens[
            torch.arange(num_samples, device=relation_tokens.device), best_indices
        ]

        gate_targets = None
        gate_loss = None
        per_task_gate_loss: Dict[int, torch.Tensor] = {}
        if target_boxes is not None and target_box_mask is not None:
            gate_targets = torch.stack(gate_targets_list, dim=0).to(dtype=gate_logits.dtype)
            valid_samples = relation_family >= 0
            if bool(valid_samples.any()):
                gate_loss = class_balanced_focal_loss(
                    gate_logits[valid_samples],
                    gate_targets[valid_samples],
                    defect_type[valid_samples],
                    gamma=self.focal_gamma,
                    beta=self.focal_beta,
                )
                for task in range(self.num_defect_types):
                    task_mask = valid_samples & (defect_type == task)
                    if bool(task_mask.any()):
                        per_task_gate_loss[task] = class_balanced_focal_loss(
                            gate_logits[task_mask],
                            gate_targets[task_mask],
                            defect_type[task_mask],
                            gamma=self.focal_gamma,
                            beta=self.focal_beta,
                        )
        attention_loss = torch.stack(evidence_losses).mean() if evidence_losses else None
        per_task_attention_loss = {
            task: torch.stack(values).mean()
            for task, values in attention_losses_by_task.items()
            if values
        }
        projected_level_norms = torch.stack(
            [level.detach().float().norm(dim=-1).mean() for level in projected_levels]
        )
        fused_feature_norm = torch.stack(fused_norms).mean()
        relation_context_norm = torch.stack(relation_context_norms).mean()

        return RelationPyramidOutput(
            relation_tokens=relation_tokens,
            relation_family=relation_family,
            p_defect=p_defect,
            gate_logits=gate_logits,
            coarse_boxes=coarse_boxes,
            query_attention=tuple(attention_list),
            relation_summary=relation_summary,
            best_relation_token=best_relation,
            scale_weights=scale_weights,
            gate_targets=gate_targets,
            projected_level_norms=projected_level_norms,
            fused_feature_norm=fused_feature_norm,
            relation_context_norm=relation_context_norm,
            per_task_gate_loss=per_task_gate_loss,
            per_task_attention_loss=per_task_attention_loss,
            gate_loss=gate_loss,
            attention_loss=attention_loss,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class RelationToPBD(nn.Module):
    """Inject relation evidence into semantic/negative and box anchor states."""

    def __init__(self, relation_hidden_size: int, language_hidden_size: int) -> None:
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        sub_sample_lengths: torch.Tensor,
        relation_summary: torch.Tensor,
        best_relation_token: torch.Tensor,
        box_start_token_id: int,
        slot_relation_tokens: Optional[torch.Tensor] = None,
        slot_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enhanced = hidden_states.clone()
        flat_hidden = enhanced.reshape(-1, enhanced.shape[-1])
        flat_ids = input_ids.reshape(-1)
        anchor_states: List[torch.Tensor] = []
        anchor_samples: List[int] = []
        start = 0
        for sample_index, raw_length in enumerate(sub_sample_lengths.tolist()):
            end = start + int(raw_length)
            local_anchor = (flat_ids[start:end] == box_start_token_id).nonzero(as_tuple=False).flatten()
            if local_anchor.numel() > 0:
                indices = local_anchor + start
                safe_summary = torch.nan_to_num(
                    relation_summary[sample_index],
                    nan=0.0,
                    posinf=32.0,
                    neginf=-32.0,
                ).clamp(-32.0, 32.0)
                if slot_relation_tokens is not None:
                    requested_slots = (
                        int(slot_counts[sample_index].item())
                        if slot_counts is not None
                        else slot_relation_tokens.shape[1]
                    )
                    num_specific = min(
                        indices.numel(), slot_relation_tokens.shape[1], requested_slots
                    )
                    selected_tokens = best_relation_token[sample_index].unsqueeze(0).expand(
                        indices.numel(), -1
                    ).clone()
                    selected_tokens[:num_specific] = slot_relation_tokens[
                        sample_index, :num_specific
                    ]
                    selected_tokens = torch.nan_to_num(
                        selected_tokens,
                        nan=0.0,
                        posinf=32.0,
                        neginf=-32.0,
                    ).clamp(-32.0, 32.0)
                else:
                    safe_best = torch.nan_to_num(
                        best_relation_token[sample_index],
                        nan=0.0,
                        posinf=32.0,
                        neginf=-32.0,
                    ).clamp(-32.0, 32.0)
                    selected_tokens = safe_best.unsqueeze(0).expand(
                        indices.numel(), -1
                    )
                sample_summaries = safe_summary.unsqueeze(0).expand(
                    indices.numel(), -1
                )
                flat_hidden[indices] = self.enhance_prediction_hidden(
                    flat_hidden[indices], sample_summaries, selected_tokens
                )
                anchor_states.extend(flat_hidden[index] for index in indices.tolist())
                anchor_samples.extend([sample_index] * indices.numel())
            start = end

        if anchor_states:
            box_anchor_hidden = torch.stack(anchor_states, dim=0)
            box_anchor_samples = torch.tensor(
                anchor_samples, device=hidden_states.device, dtype=torch.long
            )
        else:
            box_anchor_hidden = hidden_states.new_zeros((0, hidden_states.shape[-1]))
            box_anchor_samples = torch.zeros(0, device=hidden_states.device, dtype=torch.long)
        return enhanced, box_anchor_hidden, box_anchor_samples
