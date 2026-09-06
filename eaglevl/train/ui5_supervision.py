"""Canonical train-only UI5 answers. Never infer crop polarity from the source image."""
import copy

FORMAT_VERSION = "crop_gt_answer_v3_1"
TASK_LABELS = {
    "occlusion": "overlapping elements", "cropping": "cropped element",
    "text_overflow": "text overflow", "text_ellipsis": "abnormal text ellipsis",
    "content_missing": "missing content",
}


def canonical_answer(task, boxes):
    task = str(task).removeprefix("ui_")
    if task not in TASK_LABELS or not isinstance(boxes, list):
        raise ValueError(f"unknown UI5 task or missing GT: {task}")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(x) is not int or not 0 <= x <= 1000 for x in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError(f"invalid norm1000 training GT; no coordinate repair: {box}")
    if not boxes:
        return "<box>none</box>"
    return f"<ref>{TASK_LABELS[task]}</ref>" + "".join(
        "<box>" + "".join(f"<{x}>" for x in box) + "</box>" for box in boxes)


def record_boxes(record):
    # Crop GT is authoritative even when union/sample GT is positive.
    key = ("_ui5_crop_gt_local_1000" if record.get("_ui5_record_kind") == "crop"
           else "_ui5_union_gt_1000")
    if key not in record:
        raise ValueError(f"training record lacks authoritative {key}: {sample_id(record)}")
    boxes = record[key]
    if "_ui5_positive" in record and record["_ui5_positive"] is not bool(boxes):
        raise ValueError(f"training polarity conflicts with own GT: {sample_id(record)}")
    return boxes


def sample_id(record):
    return str(record.get("_ui5_sample_id") or record.get("sample_id") or record.get("id") or "unknown")


def assistant_slot(record):
    messages = record.get("conversations", record.get("messages"))
    if not isinstance(messages, list):
        raise ValueError("training conversations/messages missing")
    found = [m for m in messages if m.get("from", m.get("role")) in {"gpt", "assistant"}]
    if len(found) != 1:
        raise ValueError("UI5 training requires exactly one assistant answer")
    message = found[0]
    key = "value" if "value" in message else "content"
    if not isinstance(message.get(key), str):
        raise ValueError("UI5 assistant answer must be a plain string")
    return message, key


def answer_for_record(record):
    return canonical_answer(record.get("_ui5_task", record.get("task")), record_boxes(record))


def corrected_record(record):
    expected = answer_for_record(record)
    message, key = assistant_slot(record)
    if message[key] == expected:
        return record, False
    result = copy.deepcopy(record)
    message, key = assistant_slot(result)
    message[key] = expected
    return result, True


def validate_answer(record):
    expected = answer_for_record(record)
    message, key = assistant_slot(record)
    if message[key] != expected:
        raise ValueError(f"noncanonical complete assistant answer: {sample_id(record)}: "
                         f"actual={message[key]!r}, expected={expected!r}")
    return expected
