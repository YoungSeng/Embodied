"""UI-defect task parsing and deterministic class/label balancing."""

from collections import defaultdict
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from eaglevl.model.locany.relation_modules import UI_RELATION_PROMPT_SPECS


BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", re.IGNORECASE)

# defect_type, relation_family. Boundary is shared by overflow and cropping.
# The routing table itself is shared with trust_remote_code inference.
_TRAIN_TASK_NAMES = {
    "occlusion": "overlap",
    "text_ellipsis": "ellipsis",
    "content_missing": "missing",
}
TASK_SPECS = tuple(
    (
        _TRAIN_TASK_NAMES.get(spec.task_name, spec.task_name),
        spec.defect_type,
        spec.relation_family,
        spec.aliases,
    )
    for spec in UI_RELATION_PROMPT_SPECS
)


def _conversation_text(record: dict, role: str) -> str:
    values = [
        str(turn.get("value", ""))
        for turn in record.get("conversations", [])
        if turn.get("from") == role
    ]
    return "\n".join(values)


def identify_ui_defect_task(record: dict) -> Optional[Tuple[str, int, int]]:
    """Return (task_name, defect_type, relation_family), if this is a UI task."""
    explicit_type = record.get("defect_type")
    explicit_family = record.get("relation_family")
    if explicit_type is not None and explicit_family is not None:
        if isinstance(explicit_type, str) and not explicit_type.isdigit():
            for task_name, defect_type, _, _ in TASK_SPECS:
                if explicit_type == task_name:
                    return task_name, defect_type, int(explicit_family)
            raise ValueError(f"Unknown explicit UI defect_type: {explicit_type}")
        defect_type = int(explicit_type)
        task_name = TASK_SPECS[defect_type][0]
        return task_name, defect_type, int(explicit_family)

    prompt = _conversation_text(record, "human").lower()
    for task_name, defect_type, relation_family, aliases in TASK_SPECS:
        if any(alias.lower() in prompt for alias in aliases):
            return task_name, defect_type, relation_family
    return None


def is_positive_ui_defect(record: dict) -> bool:
    return bool(BOX_PATTERN.search(_conversation_text(record, "gpt")))


def extract_ui_defect_targets(record: dict, max_boxes: int = 8) -> Dict[str, torch.Tensor]:
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

    Sampling is with replacement for the minority side.  With the default 1:2
    effective positive/negative ratio this strongly oversamples text overflow
    while preserving the requested 88,020-record epoch size.
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

    expected_types = set(range(len(TASK_SPECS)))
    if set(buckets) != expected_types:
        missing = sorted(expected_types - set(buckets))
        raise ValueError(f"UI balancing requires all five tasks; missing defect types: {missing}")
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
        if count <= len(values):
            return rng.sample(values, count)
        return [rng.choice(values) for _ in range(count)]

    result: List[int] = []
    for defect_type in sorted(buckets):
        positive = buckets[defect_type]["positive"]
        negative = buckets[defect_type]["negative"]
        result.extend(sample_bucket(positive, positive_count))
        result.extend(sample_bucket(negative, negative_count))
    rng.shuffle(result)
    return result
