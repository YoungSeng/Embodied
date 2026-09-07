"""Original-image GRPO mathematics and deterministic, task-first sampling."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random


TASKS = ("occlusion", "cropping", "text_overflow", "text_ellipsis", "content_missing")


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 4
    groups_per_rank: int = 2
    world_size: int = 2
    total_steps: int = 1200
    warmup_steps: int = 40
    learning_rate: float = 1e-6
    min_learning_rate: float = 5e-7
    clip: float = 0.2
    kl_beta: float = 0.02
    replay_weight: float = 0.1
    num_iterations: int = 1
    max_grad_norm: float = 1.0
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_seq_length: int = 7268
    max_num_tokens_per_sample: int = 7268
    max_num_tokens: int = 7268
    max_new_tokens: int = 512
    reward_box: float = 0.70
    reward_image: float = 0.20
    reward_iou: float = 0.10
    iou_threshold: float = 0.1
    nms_iou: float = 0.5
    eval_interval: int = 200
    log_interval: int = 10
    excel_interval: int = 100
    save_interval: int = 100
    diagnostic_per_task: int = 12
    seed: int = 42

    def validate(self):
        if self.group_size != 4 or self.num_iterations != 1:
            raise ValueError("v1 requires G=4 and num_iterations=1")
        if (self.world_size, self.groups_per_rank) != (2, 2):
            raise ValueError("formal batch: two ranks, two original-image groups/rank/step")
        if (self.top_p, self.top_k, self.repetition_penalty) != (1.0, 0, 1.0):
            raise ValueError("AR log probabilities require an unfiltered temperature distribution")
        if self.temperature <= 0 or not 0 < self.clip < 1:
            raise ValueError("invalid temperature/clip")
        if len({self.max_seq_length, self.max_num_tokens_per_sample, self.max_num_tokens}) != 1:
            raise ValueError("all three sequence budgets must agree")
        if self.max_seq_length != 7268 or self.max_new_tokens != 512:
            raise ValueError("formal sequence/completion caps must be 7268/512")
        if not 0 < self.warmup_steps < self.total_steps or not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("invalid optimizer learning-rate horizon")
        if min(self.reward_box, self.reward_image, self.reward_iou, self.kl_beta, self.replay_weight) < 0:
            raise ValueError("loss and reward coefficients cannot be negative")
        if (self.iou_threshold, self.nms_iou) != (0.1, 0.5):
            raise ValueError("formal UI5 matching/NMS thresholds must be 0.1/0.5")
        if not math.isclose(self.reward_box + self.reward_image + self.reward_iou, 1.0):
            raise ValueError("reward component weights must sum to one")
        return self

    def to_dict(self):
        return asdict(self)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_seed(*parts):
    return int(digest(parts)[:15], 16)


def lr_multiplier(completed_steps, config):
    """Scheduler position is completed optimizer updates, never microbatch slots."""
    if completed_steps < config.warmup_steps:
        return completed_steps / config.warmup_steps
    progress = min(1.0, (completed_steps - config.warmup_steps) /
                   (config.total_steps - config.warmup_steps))
    floor = config.min_learning_rate / config.learning_rate
    return floor + (1.0 - floor) * (1.0 + math.cos(math.pi * progress)) / 2.0


def remaining_budget(input_tokens, config):
    remaining = min(config.max_seq_length, config.max_num_tokens_per_sample,
                    config.max_num_tokens) - input_tokens
    if remaining <= 0:
        raise ValueError(f"prompt consumes the entire token budget: {input_tokens}")
    return min(config.max_new_tokens, remaining)


def centered_advantages(rewards):
    if len(rewards) != 4 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("one original-image group must have four finite rewards")
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def completion_loss_sums(current, old, reference, advantage, clip=0.2):
    """Unnormalised completion-token sums; the caller owns group normalisation."""
    import torch
    if current.ndim != 1 or current.shape != old.shape or current.shape != reference.shape:
        raise ValueError("current/old/reference must align with actual completion tokens")
    if current.numel() == 0 or old.requires_grad or reference.requires_grad:
        raise ValueError("nonempty completions and detached old/reference probabilities required")
    ratio = (current - old).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1.0 - clip, 1.0 + clip) * advantage)
    # Non-negative sampled reverse-KL estimator, all distributions use T=0.7.
    delta = reference - current
    kl = delta.exp() - delta - 1.0
    if not torch.isfinite(policy).all() or not torch.isfinite(kl).all():
        raise FloatingPointError("non-finite GRPO objective")
    return policy.sum(), kl.sum()


class MixedSampler:
    """Stateless draws: task -> available polarity -> available k/4 -> group.

    Every complete task cycle is exactly balanced. Within each nested bucket,
    a shuffled cycle visits every member before repeating; no empty stratum is
    fabricated. A rank cursor plus the manifest identity completely restores it.
    """
    def __init__(self, groups, seed=42):
        self.groups = groups
        self.seed = seed
        self.buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, row in enumerate(groups):
            self.buckets[row["task"]][row["polarity"]][row["crop_correct_count"]].append(index)
        if not self.buckets or set(self.buckets) - set(TASKS):
            raise ValueError("mixed pool must contain eligible UI5 train groups")
        self.tasks = tuple(task for task in TASKS if task in self.buckets)
        self.identity = digest([(r["group_id"], r["crop_correct_count"]) for r in groups])

    def _pick(self, values, ordinal, namespace):
        values = sorted(values)
        cycle, position = divmod(ordinal, len(values))
        random.Random(stable_seed(self.seed, namespace, cycle)).shuffle(values)
        return values[position], cycle

    def index_at(self, ordinal):
        task, occurrence = self._pick(self.tasks, ordinal, "task")
        polarity, occurrence = self._pick(self.buckets[task], occurrence, (task, "polarity"))
        count, occurrence = self._pick(self.buckets[task][polarity], occurrence, (task, polarity, "k"))
        index, _ = self._pick(self.buckets[task][polarity][count], occurrence, (task, polarity, count))
        return index

    def report(self, draws):
        counts = Counter()
        for ordinal in range(draws):
            row = self.groups[self.index_at(ordinal)]
            counts[(row["task"], row["polarity"], row["crop_correct_count"])] += 1
        rows = []
        for task in TASKS:
            for polarity in ("positive", "negative"):
                for count in (1, 2, 3):
                    available = len(self.buckets[task].get(polarity, {}).get(count, []))
                    sampled = counts[(task, polarity, count)]
                    rows.append(dict(task=task, polarity=polarity, crop_correct_count=count,
                                     available_groups=available, empty=available == 0,
                                     planned_draws=sampled, planned_fraction=sampled / draws))
        return rows


def aligned_slot_count(local_slots, device):
    import torch
    import torch.distributed as dist
    count = torch.tensor(local_slots, device=device, dtype=torch.long)
    dist.all_reduce(count, op=dist.ReduceOp.MAX)
    return int(count.item())


def zero_parameter_touch(model):
    """Include every trainable parameter on every rank, even in padding slots.

    Gate heads can be unused by the policy but used by replay. This zero term
    gives ZeRO-2 the same parameter/collective participation in both cases.
    """
    return sum(parameter.reshape(-1)[0] * 0.0 for parameter in model.parameters()
               if parameter.requires_grad)


def deepspeed_slot_backward(engine, loss, *, final_slot):
    """One explicit boundary per update; GAS=2 denotes group rounds, not crops.

    The caller has already divided by real group/token counts. Disabling the
    engine's additional GAS scaling avoids an accidental factor of two. These
    public APIs also work in v3's older managed-accumulation DeepSpeed runtime.
    """
    engine.set_gradient_accumulation_boundary(final_slot)
    engine.backward(loss, scale_wrt_gas=False)
    engine.step()
