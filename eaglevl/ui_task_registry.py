"""Versioned, CPU-only UI task identities shared by data and remote checkpoints.

Model IDs are deliberately different from the historical scorer class IDs.
Only UI5 supports implicit prompt routing; new sources require an explicit ID.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

RELATION_FAMILIES = ("boundary", "pairwise", "text", "presence")
REGISTRY_VERSION = 1


@dataclass(frozen=True)
class UITask:
    task_id: int
    task_key: str
    source_dataset: str
    defect_type: str
    relation_family: str
    prompt_label: str
    view_policy: str
    diagnostic_name: str
    class_id: int
    aliases: tuple[str, ...] = ()
    source_version: str = "v1"
    train: str = ""
    test: str = ""

    @property
    def family_id(self):
        return RELATION_FAMILIES.index(self.relation_family)

    @property
    def prompt(self):
        return "Locate all the instances that match the following description: " + self.prompt_label + "."

    def to_dict(self):
        return asdict(self)


UI_TASKS = (
    UITask(0, "text_overflow", "ui5", "文字溢出容器", "boundary", "text overflow", "crops", "text_overflow", 2, ("text overflow", "文字溢出")),
    UITask(1, "cropping", "ui5", "元素被裁切", "boundary", "cropped element", "crops", "element_cropping", 1, ("cropped element", "element cropping", "元素裁切")),
    UITask(2, "occlusion", "ui5", "元素重叠", "pairwise", "overlapping elements", "crops", "element_overlap", 0, ("overlapping elements", "element overlap", "元素重叠")),
    UITask(3, "text_ellipsis", "ui5", "文字省略异常", "text", "abnormal text ellipsis", "crops", "text_ellipsis", 3, ("abnormal text ellipsis", "ellipsis anomaly", "省略异常")),
    UITask(4, "content_missing", "ui5", "内容未展示", "presence", "missing content", "full_image", "content_missing", 4, ("missing content", "content missing", "内容缺失")),
    UITask(5, "ui_alignment", "ui_alignment", "对齐异常", "pairwise", "misaligned elements", "full_image", "ui_alignment", 5),
    UITask(6, "change_line_illegal_v3", "change_line_illegal_v3", "换行规则不合理", "text", "illegal line breaks", "crops", "change_line_illegal_v3", 6),
    UITask(7, "synth_cropping", "clip-v2-20250826", "元素被裁切", "boundary", "cropped element", "crops", "synth_cropping", 7),
    UITask(8, "synth_occlusion", "occl-v2-20250827", "元素重叠", "pairwise", "overlapping elements", "crops", "synth_occlusion", 8),
    UITask(9, "synth_radius", "corn-v2-20250918", "圆角异常", "boundary", "abnormal corner radius", "crops", "synth_radius", 9),
    UITask(10, "synth_loneword", "lw-v1-20251117", "换行规则不合理", "text", "illegal line breaks", "crops", "synth_loneword", 10),
    UITask(11, "synth_large_margin", "lmar-v1-20251121", "元素间距过大", "pairwise", "excessive spacing between elements", "full_image", "synth_large_margin", 11),
    UITask(12, "synth_inner_margin", "apad-v2-20251202", "内间距异常", "boundary", "abnormal inner padding", "crops", "synth_inner_margin", 12),
    UITask(13, "synth_small_margin", "smar-v1-20251211", "元素间距过小", "pairwise", "insufficient spacing between elements", "crops", "synth_small_margin", 13),
)
UI5_TASKS = UI_TASKS[:5]
UI9_TASKS = UI_TASKS[5:]
TASK_BY_KEY = {task.task_key: task for task in UI_TASKS}
LEGACY_ALIASES = {"overlap": "occlusion", "ellipsis": "text_ellipsis", "missing": "content_missing"}


def get_task(value):
    if isinstance(value, int) or str(value).isdigit():
        index = int(value)
        if not 0 <= index < len(UI_TASKS):
            raise ValueError(f"Unknown UI task ID: {value}")
        return UI_TASKS[index]
    key = LEGACY_ALIASES.get(str(value), str(value))
    if key not in TASK_BY_KEY:
        raise ValueError(f"Unknown UI task key: {value}")
    return TASK_BY_KEY[key]


def validate_registry(value, expected_count=None):
    rows = value.get("tasks") if isinstance(value, dict) else value
    if not isinstance(rows, list) or len(rows) not in (5, 14):
        raise ValueError("UI registry must contain the 5 legacy or all 14 tasks")
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError("UI task count disagrees with checkpoint registry")
    for index, row in enumerate(rows):
        task = UI_TASKS[index]
        for key in ("task_id", "task_key", "relation_family", "view_policy", "prompt_label", "class_id"):
            if row.get(key) != getattr(task, key):
                raise ValueError(f"UI registry routing drift at {index}.{key}: {row.get(key)!r}")
        for key in ("source_dataset", "defect_type", "train", "test"):
            if key not in row:
                raise ValueError(f"UI registry missing {index}.{key}")
    return rows


def load_registry(path):
    return validate_registry(json.loads(Path(path).read_text(encoding="utf-8")))


def configure_task_registry(config, registry=None, num_tasks=None):
    registry = registry if registry is not None else getattr(config, "ui_task_registry", None)
    count = int(num_tasks or getattr(config, "ui_num_tasks", 5))
    rows = validate_registry(registry, count) if registry is not None else [t.to_dict() for t in UI_TASKS[:count]]
    validate_registry(rows, count)
    config.ui_num_tasks = count
    config.ui_task_registry = rows
    config.ui_task_registry_version = REGISTRY_VERSION
    return config
