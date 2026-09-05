"""UI-defect task parsing and deterministic class/label balancing."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from eaglevl.ui_task_registry import UI_TASKS, get_task


BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", re.IGNORECASE)

# defect_type, relation_family. Boundary is shared by overflow and cropping.
# The routing table itself is shared with trust_remote_code inference.
_TRAIN_TASK_NAMES = {
    "occlusion": "overlap",
    "text_ellipsis": "ellipsis",
    "content_missing": "missing",
}
UI_TASK_SPECS = tuple(
    (_TRAIN_TASK_NAMES.get(t.task_key, t.task_key), t.task_id, t.family_id, t.aliases)
    for t in UI_TASKS
)
TASK_SPECS = UI_TASK_SPECS[:5]  # Legacy public prompt table.



def _conversation_text(record: dict, role: str) -> str:
    values = [
        str(turn.get("value", ""))
        for turn in record.get("conversations", [])
        if turn.get("from") == role
    ]
    return "\n".join(values)


def identify_ui_defect_task(record: dict) -> Optional[Tuple[str, int, int]]:
    """Return (task_name, defect_type, relation_family), if this is a UI task."""
    explicit = record.get("task_id", record.get("model_task_id", record.get("task_key")))
    if explicit is not None:
        task = get_task(explicit)
        if record.get("task_key") is not None and get_task(record["task_key"]).task_id != task.task_id:
            raise ValueError("Conflicting task_id/task_key")
        if isinstance(record.get("defect_type"), int) and record["defect_type"] != task.task_id:
            raise ValueError("Conflicting task_id/defect_type")
        family = record.get("relation_family", task.family_id)
        if family not in (task.family_id, task.relation_family):
            raise ValueError("Conflicting task ID/relation family")
        return UI_TASK_SPECS[task.task_id][:3]
    explicit_type = record.get("defect_type")
    explicit_family = record.get("relation_family")
    if (explicit_type is None) != (explicit_family is None):
        raise ValueError("Explicit UI routing metadata must provide both defect_type and relation_family")
    if explicit_type is not None:
        try:
            task = get_task(explicit_type)
        except ValueError:
            task = next((t for t in UI_TASKS if t.diagnostic_name == explicit_type), None)
            if task is None:
                raise ValueError(f"Unknown explicit UI defect type: {explicit_type}")
        if explicit_family not in (task.family_id, task.relation_family):
            raise ValueError("Explicit UI relation_family disagrees with the fixed task table")
        return UI_TASK_SPECS[task.task_id][:3]

    prompt = _conversation_text(record, "human").lower()
    for task_name, defect_type, relation_family, aliases in TASK_SPECS[:5]:
        if any(alias.lower() in prompt for alias in aliases):
            return task_name, defect_type, relation_family
    return None


def is_positive_ui_defect(record: dict) -> bool:
    return bool(BOX_PATTERN.search(_conversation_text(record, "gpt")))


def extract_ui_defect_targets(record: dict, max_boxes: int = 8) -> Dict[str, torch.Tensor]:
    import torch
    task = identify_ui_defect_task(record)
    if task is None:
        defect_type = -1
        relation_family = -1
    else:
        _, defect_type, relation_family = task

    boxes = torch.zeros(max_boxes, 4, dtype=torch.float32)
    box_mask = torch.zeros(max_boxes, dtype=torch.bool)
    answer = _conversation_text(record, "gpt")
    parsed = []
    for match in BOX_PATTERN.finditer(answer):
        box = tuple(float(value) for value in match.groups())
        x1, y1, x2, y2 = box
        if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
            parsed.append(box)
    for index, box in enumerate(parsed[:max_boxes]):
        boxes[index] = torch.tensor(box, dtype=torch.float32)
        box_mask[index] = True

    return {
        "relation_family": torch.tensor([relation_family], dtype=torch.long),
        "defect_type": torch.tensor([defect_type], dtype=torch.long),
        "target_boxes": boxes.unsqueeze(0),
        "target_box_mask": box_mask.unsqueeze(0),
    }


def build_balanced_ui_indices(
    records: Sequence[dict],
    records_per_class: int = 17604,
    negative_to_positive_ratio: float = 2.0,
    seed: int = 202603,
) -> List[int]:
    """Build exactly ``records_per_class`` indices per UI task.

    Sampling is with replacement for the minority side.  Every
    ``manual_gt_repair`` record is pinned once inside its task/polarity quota;
    ordinary records fill the remaining quota.  With the default 1:2 effective
    positive/negative ratio this strongly oversamples text overflow while
    preserving the requested 88,020-record epoch size.
    """
    if records_per_class <= 0:
        raise ValueError("records_per_class must be positive")
    if negative_to_positive_ratio <= 0:
        raise ValueError("negative_to_positive_ratio must be positive")

    buckets: Dict[int, Dict[str, List[int]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    passthrough = []
    for index, record in enumerate(records):
        task = identify_ui_defect_task(record)
        if task is None:
            passthrough.append(index)
            continue
        defect_type = task[1]
        is_positive = is_positive_ui_defect(record)
        buckets[defect_type]["positive" if is_positive else "negative"].append(index)

    if not buckets:
        return list(range(len(records)))

    if passthrough:
        raise ValueError(
            "UI balancing was enabled for a mixed dataset; move non-UI records to a separate recipe entry"
        )

    rng = random.Random(seed)
    positive_count = max(1, round(records_per_class / (1.0 + negative_to_positive_ratio)))
    negative_count = records_per_class - positive_count

    def sample_bucket(values: List[int], count: int) -> List[int]:
        if not values:
            raise ValueError("Cannot balance a task with an empty positive or negative bucket")
        required = [
            index
            for index in values
            if records[index].get("_ui5_crop_source") == "manual_gt_repair"
        ]
        if len(required) > count:
            raise ValueError(
                "UI balancing quota is smaller than the number of required "
                f"manual_gt_repair records: required={len(required)}, quota={count}"
            )
        required_set = set(required)
        ordinary = [index for index in values if index not in required_set]
        remaining = count - len(required)
        if remaining == 0:
            return required
        if not ordinary:
            raise ValueError(
                "Cannot fill the remaining UI balancing quota without resampling "
                "manual_gt_repair records"
            )
        if remaining <= len(ordinary):
            sampled = rng.sample(ordinary, remaining)
        else:
            sampled = [rng.choice(ordinary) for _ in range(remaining)]
        return [*required, *sampled]

    result: List[int] = []
    for defect_type in sorted(buckets):
        positive = buckets[defect_type]["positive"]
        negative = buckets[defect_type]["negative"]
        result.extend(sample_bucket(positive, positive_count if negative else records_per_class) if positive else [])
        result.extend(sample_bucket(negative, negative_count if positive else records_per_class) if negative else [])
    rng.shuffle(result)
    required_indices = {
        index
        for index, record in enumerate(records)
        if record.get("_ui5_crop_source") == "manual_gt_repair"
    }
    missing_required = required_indices - set(result)
    if missing_required:
        raise RuntimeError(
            "UI balancing dropped required manual_gt_repair records: "
            f"{sorted(missing_required)[:20]}"
        )
    return result


def build_task_balanced_all_records_indices(
    records: Sequence[dict],
    seed: int = 202603,
) -> List[int]:
    """Return one deterministic macro-balanced epoch without dropping records.

    Every task is shuffled independently.  The five task streams are then
    round-robin interleaved.  A shorter stream is repeated only after every
    unique record in that stream has appeared once; the longest stream appears
    exactly once.  Positive/negative labels are deliberately not rebalanced so
    the recipe's natural crop distribution is preserved.
    """
    buckets: Dict[int, List[int]] = defaultdict(list)
    passthrough: List[int] = []
    for index, record in enumerate(records):
        task = identify_ui_defect_task(record)
        if task is None:
            passthrough.append(index)
        else:
            buckets[task[1]].append(index)

    if not buckets:
        return list(range(len(records)))
    if passthrough:
        raise ValueError(
            "task_balanced_all_records was enabled for a mixed dataset; "
            "move non-UI records to a separate recipe entry"
        )
    if any(not values for values in buckets.values()):
        raise ValueError("task_balanced_all_records cannot use an empty task stream")

    rng = random.Random(seed)
    streams: Dict[int, List[int]] = {}
    for defect_type, values in sorted(buckets.items()):
        stream = list(values)
        rng.shuffle(stream)
        streams[defect_type] = stream

    task_order = sorted(streams)
    rng.shuffle(task_order)
    longest = max(len(stream) for stream in streams.values())
    result: List[int] = []
    for position in range(longest):
        rotated = task_order[position % len(task_order):] + task_order[:position % len(task_order)]
        for defect_type in rotated:
            stream = streams[defect_type]
            result.append(stream[position % len(stream)])

    if set(result) != set(range(len(records))):
        missing = sorted(set(range(len(records))) - set(result))
        raise RuntimeError(
            "task_balanced_all_records dropped legal records: "
            f"count={len(missing)}, first={missing[:20]}"
        )
    required = {
        index
        for index, record in enumerate(records)
        if record.get("_ui5_crop_source") == "manual_gt_repair"
    }
    if not required.issubset(result):
        raise RuntimeError(
            "task_balanced_all_records dropped manual_gt_repair records: "
            f"{sorted(required - set(result))[:20]}"
        )
    return result


def _ui_source_group_id(record: Mapping[str, Any]) -> str:
    value = (
        record.get("source_image_id")
        or record.get("_ui5_image_id")
        or record.get("_ui5_source_image")
        or record.get("image")
    )
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    if not value:
        raise ValueError("UI source-balanced sampling requires a source image id/path")
    return str(value)


def _stable_sampler_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _deterministic_permutation(
    values: Sequence[Any], seed: int, *namespace: object
) -> List[Any]:
    output = list(values)
    random.Random(_stable_sampler_seed(seed, *namespace)).shuffle(output)
    return output


def build_task_source_balanced_rotating_plan(
    records: Sequence[dict],
    negative_to_positive_ratio: float = 2.0,
) -> dict:
    """Index all legal records by task, polarity, and source image.

    The active pool remains the complete recipe.  An effective epoch uses the
    same number of draws for every task and approximately the requested
    negative:positive ratio, while choosing source images uniformly rather
    than weighting a page by how many strips it produced.
    """
    if negative_to_positive_ratio <= 0:
        raise ValueError("negative_to_positive_ratio must be positive")

    buckets: Dict[int, Dict[str, Dict[str, List[int]]]] = defaultdict(
        lambda: {
            "positive": defaultdict(list),
            "negative": defaultdict(list),
        }
    )
    passthrough: List[int] = []
    manual_indices = set()
    for index, record in enumerate(records):
        task = identify_ui_defect_task(record)
        if task is None:
            passthrough.append(index)
            continue
        polarity = "positive" if is_positive_ui_defect(record) else "negative"
        source_id = _ui_source_group_id(record)
        buckets[task[1]][polarity][source_id].append(index)
        if record.get("_ui5_crop_source") == "manual_gt_repair":
            if polarity != "positive":
                raise ValueError("manual_gt_repair record must be positive")
            manual_indices.add(index)

    if not buckets:
        raise ValueError("task_source_balanced_rotating requires UI records")
    if passthrough:
        raise ValueError(
            "task_source_balanced_rotating was enabled for a mixed dataset; "
            "move non-UI records to a separate recipe entry"
        )
    max_positive_sources = max(
        len(polarities["positive"]) for polarities in buckets.values()
    )
    max_negative_sources = max(
        len(polarities["negative"]) for polarities in buckets.values()
    )
    positive_slots = max(
        max_positive_sources,
        int(math.ceil(max_negative_sources / negative_to_positive_ratio)),
    )
    negative_slots = max(1, int(round(positive_slots * negative_to_positive_ratio)))
    while negative_slots < max_negative_sources:
        positive_slots += 1
        negative_slots = max(
            1, int(round(positive_slots * negative_to_positive_ratio))
        )

    planned_indices = {
        index
        for polarities in buckets.values()
        for source_groups in polarities.values()
        for values in source_groups.values()
        for index in values
    }
    expected_indices = set(range(len(records)))
    if planned_indices != expected_indices:
        missing = sorted(expected_indices - planned_indices)
        raise RuntimeError(
            "task_source_balanced_rotating dropped legal records from its active plan: "
            f"count={len(missing)}, first={missing[:20]}"
        )
    if not manual_indices.issubset(planned_indices):
        raise RuntimeError(
            "task_source_balanced_rotating dropped manual_gt_repair records: "
            f"{sorted(manual_indices - planned_indices)[:20]}"
        )

    normalized_buckets = {
        defect_type: {
            polarity: {
                source_id: tuple(sorted(values))
                for source_id, values in sorted(source_groups.items())
            }
            for polarity, source_groups in polarities.items()
        }
        for defect_type, polarities in sorted(buckets.items())
    }
    per_task_records = positive_slots + negative_slots
    task_slots = {
        task: {"positive": positive_slots if p["negative"] else per_task_records,
               "negative": negative_slots if p["positive"] else per_task_records}
        for task, p in normalized_buckets.items()
    }
    for task, p in normalized_buckets.items():
        if not p["positive"]: task_slots[task]["positive"] = 0
        if not p["negative"]: task_slots[task]["negative"] = 0
    return {
        "buckets": normalized_buckets,
        "slots_by_task": task_slots,
        "manual_indices": frozenset(manual_indices),
        "positive_slots_per_task": positive_slots,
        "negative_slots_per_task": negative_slots,
        "negative_to_positive_ratio": negative_slots / positive_slots,
        "per_task_records": per_task_records,
        "epoch_length": per_task_records * len(normalized_buckets),
        "source_groups_by_task": {
            defect_type: {
                polarity: len(source_groups)
                for polarity, source_groups in polarities.items()
            }
            for defect_type, polarities in normalized_buckets.items()
        },
        "records_by_task": {
            defect_type: {
                polarity: sum(len(values) for values in source_groups.values())
                for polarity, source_groups in polarities.items()
            }
            for defect_type, polarities in normalized_buckets.items()
        },
        "source_group_repeat_draws_per_epoch_by_task": {
            defect_type: {
                "positive": max(0, task_slots[defect_type]["positive"] - len(polarities["positive"])),
                "negative": max(0, task_slots[defect_type]["negative"] - len(polarities["negative"])),
            }
            for defect_type, polarities in normalized_buckets.items()
        },
    }


def _rotating_source_record(
    *,
    source_groups: Mapping[str, Sequence[int]],
    manual_indices: frozenset,
    global_position: int,
    seed: int,
    defect_type: int,
    polarity: str,
) -> int:
    source_ids = tuple(sorted(source_groups))
    source_cycle, source_offset = divmod(global_position, len(source_ids))
    source_order = _deterministic_permutation(
        source_ids, seed, "source", defect_type, polarity, source_cycle
    )
    source_id = source_order[source_offset]
    values = tuple(source_groups[source_id])

    # Every source appears once per source cycle, so source_cycle is also this
    # source's zero-based visit count.  Rotate all crops for that source before
    # any crop repeats.  Manual repair views are visited first in cycle zero.
    record_cycle, record_offset = divmod(source_cycle, len(values))
    if record_cycle == 0:
        required = [index for index in values if index in manual_indices]
        ordinary = [index for index in values if index not in manual_indices]
        record_order = [
            *sorted(required),
            *_deterministic_permutation(
                ordinary, seed, "record", defect_type, polarity, source_id, 0
            ),
        ]
    else:
        record_order = _deterministic_permutation(
            values,
            seed,
            "record",
            defect_type,
            polarity,
            source_id,
            record_cycle,
        )
    return int(record_order[record_offset])


def _proportionally_interleave(positive: Sequence[int], negative: Sequence[int]) -> List[int]:
    output: List[int] = []
    positive_index = 0
    negative_index = 0
    total = len(positive) + len(negative)
    for position in range(total):
        desired_positive = ((position + 1) * len(positive)) // max(total, 1)
        if desired_positive > positive_index:
            output.append(int(positive[positive_index]))
            positive_index += 1
        else:
            output.append(int(negative[negative_index]))
            negative_index += 1
    return output


def materialize_task_source_balanced_rotating_indices(
    plan: Mapping[str, Any],
    *,
    seed: int = 202603,
    epoch_index: int = 0,
) -> List[int]:
    """Materialize one deterministic source-balanced epoch from a reusable plan."""
    if epoch_index < 0:
        raise ValueError("epoch_index cannot be negative")
    positive_slots = int(plan["positive_slots_per_task"])
    negative_slots = int(plan["negative_slots_per_task"])
    manual_indices = plan["manual_indices"]
    task_streams: Dict[int, List[int]] = {}
    for defect_type, polarities in sorted(plan["buckets"].items()):
        slots = plan.get("slots_by_task", {}).get(defect_type, {})
        positive_slots = slots.get("positive", int(plan["positive_slots_per_task"]))
        negative_slots = slots.get("negative", int(plan["negative_slots_per_task"]))
        positive = [
            _rotating_source_record(
                source_groups=polarities["positive"],
                manual_indices=manual_indices,
                global_position=epoch_index * positive_slots + position,
                seed=seed,
                defect_type=int(defect_type),
                polarity="positive",
            )
            for position in range(positive_slots)
        ]
        negative = [
            _rotating_source_record(
                source_groups=polarities["negative"],
                manual_indices=manual_indices,
                global_position=epoch_index * negative_slots + position,
                seed=seed,
                defect_type=int(defect_type),
                polarity="negative",
            )
            for position in range(negative_slots)
        ]
        task_streams[int(defect_type)] = _proportionally_interleave(positive, negative)

    task_order = _deterministic_permutation(
        sorted(task_streams), seed, "task-order", epoch_index
    )
    per_task = int(plan["per_task_records"])
    result: List[int] = []
    for position in range(per_task):
        offset = position % len(task_order)
        rotated = task_order[offset:] + task_order[:offset]
        result.extend(task_streams[defect_type][position] for defect_type in rotated)
    if len(result) != int(plan["epoch_length"]):
        raise RuntimeError(
            "task_source_balanced_rotating epoch length mismatch: "
            f"{len(result)} != {plan['epoch_length']}"
        )
    return result


def build_task_source_balanced_rotating_indices(
    records: Sequence[dict],
    negative_to_positive_ratio: float = 2.0,
    seed: int = 202603,
    epoch_index: int = 0,
) -> List[int]:
    plan = build_task_source_balanced_rotating_plan(
        records,
        negative_to_positive_ratio=negative_to_positive_ratio,
    )
    return materialize_task_source_balanced_rotating_indices(
        plan, seed=seed, epoch_index=epoch_index
    )
