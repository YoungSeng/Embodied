"""Pure, offline-recomputable metrics for the ten LocateAnything CPT tasks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence


BOX_RE = re.compile(
    r"<box>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*</box>",
    re.IGNORECASE,
)
PAIR_RE = re.compile(
    r"<ref>(.*?)</ref>\s*" + BOX_RE.pattern,
    re.IGNORECASE | re.DOTALL,
)
POINT_RE = re.compile(
    r"<\|point_start\|>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\)?\s*<\|point_end\|>",
    re.IGNORECASE,
)
TAP_POINT_RE = re.compile(
    r"(?:tap|click)\s*\(\s*position\s*=\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\)?\s*\)",
    re.IGNORECASE,
)
SPECIAL_END_RE = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*$")
BOX_NONE_RE = re.compile(r"<box>\s*(?:none|null|无)\s*</box>", re.IGNORECASE)
DEFAULT_IOU_THRESHOLD = 0.1


PRIMARY_METRIC_BY_TASK = {
    "ui_caption": "rouge_l",
    "agent_action": "action_hit50",
    "agent_grounding": "point_hit50",
    "ui_defect": "defect_macro_f1",
    "all_ui_elements": "box_f1",
    "single_grounding": "box_recall",
    "ocr": "ocr_f1",
    "referring_kg": "rouge_l",
    "referring": "rouge_l",
    "vqa": "vqa_accuracy",
}

UI_DEFECT_CLASSES = (
    "text_overflow",
    "text_ellipsis",
    "occlusion",
    "cropping",
    "content_missing",
)
UI_DEFECT_CLASS_DISPLAY = {
    "text_overflow": "文字溢出",
    "text_ellipsis": "文本省略",
    "occlusion": "元素遮挡/重叠",
    "cropping": "元素裁切",
    "content_missing": "内容缺失",
}
_UI_DEFECT_LABEL_ALIASES = {
    "text_overflow": {
        "text overflow",
        "文字溢出",
        "文本溢出",
        "文字溢出容器",
    },
    "text_ellipsis": {
        "text ellipsis",
        "text truncation error",
        "文本省略",
        "文字省略",
        "文字省略异常",
    },
    "occlusion": {
        "occlusion",
        "element overlap",
        "element occlusion",
        "ui element overlap",
        "元素遮挡",
        "元素重叠",
        "遮挡",
        "重叠",
    },
    "cropping": {
        "cropping",
        "cropped element",
        "element cropping",
        "ui element cropping",
        "元素裁切",
        "元素被裁切",
        "元素截断",
        "裁切",
    },
    "content_missing": {
        "content missing",
        "missing content",
        "content not displayed",
        "内容缺失",
        "内容未展示",
    },
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", SPECIAL_END_RE.sub("", value.strip())).strip()


def normalize_label(value: str) -> str:
    value = normalize_text(value).casefold()
    return re.sub(r"\s*\|\s*type\s*=.*$", "", value).strip()


def canonical_defect_label(value: str) -> str:
    """Map dataset/prompt aliases onto the fixed five UI-defect classes."""

    normalized = normalize_label(value)
    lookup = re.sub(r"[\s_-]+", " ", normalized).strip().casefold()
    for canonical, aliases in _UI_DEFECT_LABEL_ALIASES.items():
        if lookup == canonical.replace("_", " ") or lookup in {
            alias.casefold() for alias in aliases
        }:
            return canonical
    return normalized


def parse_label_type(value: str) -> tuple[str, str | None]:
    raw = normalize_text(value)
    match = re.match(r"^(.*?)\s*\|\s*type\s*=\s*(.*?)\s*$", raw, re.IGNORECASE)
    if match:
        return normalize_label(match.group(1)), normalize_text(match.group(2)).casefold()
    return normalize_label(raw), None


def char_f1(prediction: str, target: str) -> float:
    pred = Counter(char for char in normalize_text(prediction) if not char.isspace())
    gold = Counter(char for char in normalize_text(target) if not char.isspace())
    pred_total, gold_total = sum(pred.values()), sum(gold.values())
    if not pred_total or not gold_total:
        return float(pred_total == gold_total)
    common = sum((pred & gold).values())
    if not common:
        return 0.0
    precision, recall = common / pred_total, common / gold_total
    return 2.0 * precision * recall / (precision + recall)


def rouge_l(prediction: str, target: str) -> float:
    left, right = list(normalize_text(prediction)), list(normalize_text(target))
    if not left or not right:
        return float(not left and not right)
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if token == other
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(left), lcs / len(right)
    return 2.0 * precision * recall / (precision + recall) if lcs else 0.0


def _coordinate(value: str) -> float:
    return float(value)


def valid_point(point: Sequence[float]) -> bool:
    return len(point) == 2 and all(0.0 <= float(value) <= 1000.0 for value in point)


def valid_box(box: Sequence[float]) -> bool:
    return (
        len(box) == 4
        and all(0.0 <= float(value) <= 1000.0 for value in box)
        and float(box[0]) < float(box[2])
        and float(box[1]) < float(box[3])
    )


def parse_points(text: str) -> list[list[float]]:
    matches = [*POINT_RE.finditer(text), *TAP_POINT_RE.finditer(text)]
    matches.sort(key=lambda match: match.start())
    return [[_coordinate(match.group(1)), _coordinate(match.group(2))] for match in matches]


def parse_boxes(text: str) -> list[list[float]]:
    return [[_coordinate(value) for value in match.groups()] for match in BOX_RE.finditer(text)]


def parse_labeled_boxes(text: str) -> list[dict[str, Any]]:
    output = []
    paired_box_spans = []
    for match in PAIR_RE.finditer(text):
        label, element_type = parse_label_type(match.group(1))
        output.append(
            {
                "label": label,
                "type": element_type,
                "raw_label": normalize_text(match.group(1)),
                "box": [_coordinate(match.group(index)) for index in range(2, 6)],
            }
        )
        # The nested BOX_RE is the tail of the pair match.  Remember the pair
        # span so the fallback below does not add the same box twice.
        paired_box_spans.append(match.span())
    for match in BOX_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in paired_box_spans):
            continue
        output.append(
            {
                "label": "",
                "type": None,
                "raw_label": "",
                "box": [_coordinate(value) for value in match.groups()],
            }
        )
    return output


def box_center(box: Sequence[float]) -> list[float]:
    return [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0]


def point_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def maximum_weight_matching(weights: Sequence[Sequence[float]]) -> list[tuple[int, int, float]]:
    """Rectangular Hungarian assignment maximizing total weight, dependency-free."""
    rows = len(weights)
    columns = max((len(row) for row in weights), default=0)
    if not rows or not columns:
        return []
    transposed = rows > columns
    matrix = [list(map(float, row)) + [0.0] * (columns - len(row)) for row in weights]
    if transposed:
        matrix = [list(row) for row in zip(*matrix)]
        rows, columns = columns, rows

    maximum = max((max(row, default=0.0) for row in matrix), default=0.0)
    costs = [[maximum - value for value in row] for row in matrix]
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = float("inf"), 0
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(columns + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments = []
    original_rows = len(weights)
    original_columns = max((len(row) for row in weights), default=0)
    for column in range(1, columns + 1):
        row = p[column]
        if not row:
            continue
        left, right = (column - 1, row - 1) if transposed else (row - 1, column - 1)
        if left < original_rows and right < len(weights[left]) and right < original_columns:
            assignments.append((left, right, float(weights[left][right])))
    return sorted(assignments)


def one_to_one_boxes(
    prediction: str,
    target: str,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    label_aware: bool = False,
) -> dict[str, Any]:
    predicted = parse_labeled_boxes(prediction)
    gold = parse_labeled_boxes(target)
    weights = []
    for gold_item in gold:
        row = []
        for pred_item in predicted:
            compatible = (
                valid_box(gold_item["box"])
                and valid_box(pred_item["box"])
                and (not label_aware or gold_item["label"] == pred_item["label"])
            )
            row.append(box_iou(gold_item["box"], pred_item["box"]) if compatible else 0.0)
        weights.append(row)
    assignment = maximum_weight_matching(weights)
    matches = [item for item in assignment if item[2] >= iou_threshold]
    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(gold) - true_positive
    precision = true_positive / len(predicted) if predicted else float(not gold)
    recall = true_positive / len(gold) if gold else float(not predicted)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    label_exact = [
        float(gold[left]["label"] == predicted[right]["label"])
        for left, right, iou in assignment
        if iou >= iou_threshold
    ]
    label_char = [
        char_f1(predicted[right]["label"], gold[left]["label"])
        for left, right, iou in assignment
        if iou >= iou_threshold
    ]
    type_exact = [
        float(gold[left].get("type") == predicted[right].get("type"))
        for left, right, iou in assignment
        if iou >= iou_threshold and gold[left].get("type") is not None
    ]
    return {
        "gold_count": len(gold),
        "pred_count": len(predicted),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_iou_sum": sum(item[2] for item in matches),
        "matched_iou_mean": sum(item[2] for item in matches) / len(matches) if matches else None,
        "label_exact_sum": sum(label_exact),
        "label_char_f1_sum": sum(label_char),
        "location_match_count": len(label_exact),
        "type_exact_sum": sum(type_exact),
        "type_match_count": len(type_exact),
        "format_valid": (
            (
                bool(predicted)
                and len(predicted) == len(parse_boxes(prediction))
                and all(valid_box(item["box"]) for item in predicted)
            )
            or (not predicted and bool(BOX_NONE_RE.search(prediction)))
        ),
        "assignments": [
            {"gold_index": left, "pred_index": right, "iou": iou}
            for left, right, iou in assignment
        ],
    }


def parse_vqa_label(text: str) -> str | None:
    value = normalize_text(text).casefold()
    # Negative must be tested first because 不正确 contains 正确.
    if re.search(r"(?:不正确|错误|incorrect|false)", value):
        return "incorrect"
    if re.search(r"(?:正确|correct|true)", value):
        return "correct"
    return None


def parse_action_type(text: str) -> str | None:
    value = normalize_text(text).casefold()
    aliases = (
        ("tap", ("tap", "click", "点击")),
        ("swipe", ("swipe", "scroll", "滑动", "滚动")),
        ("type", ("type", "input", "输入")),
        ("back", ("back", "返回")),
        ("home", ("home", "主页")),
        ("wait", ("wait", "等待")),
    )
    for canonical, candidates in aliases:
        if any(candidate in value for candidate in candidates):
            return canonical
    return None


def _target_points(text: str) -> list[list[float]]:
    points = parse_points(text)
    return points or [box_center(box) for box in parse_boxes(text)]


def point_metrics(prediction: str, target: str) -> dict[str, Any]:
    predicted, gold = parse_points(prediction), _target_points(target)
    weights = [
        [
            -point_distance(left, right)
            if valid_point(left) and valid_point(right)
            else -1.0e12
            for right in predicted
        ]
        for left in gold
    ]
    assignment = maximum_weight_matching(weights)
    distances = [-weight for _, _, weight in assignment]
    best = min(distances) if distances else None
    return {
        "gold_point_count": len(gold),
        "pred_point_count": len(predicted),
        "point_format_valid": bool(predicted) and all(valid_point(point) for point in predicted),
        "point_distance": best,
        "point_hit25": float(best <= 25.0) if best is not None else 0.0,
        "point_hit50": float(best <= 50.0) if best is not None else 0.0,
        "point_hit100": float(best <= 100.0) if best is not None else 0.0,
    }


def defect_metrics(
    prediction: str,
    target: str,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    predicted, gold = parse_labeled_boxes(prediction), parse_labeled_boxes(target)
    for item in [*predicted, *gold]:
        item["label"] = canonical_defect_label(item["label"])
    observed_classes = {item["label"] for item in [*predicted, *gold] if item["label"]}
    classes = [label for label in UI_DEFECT_CLASSES if label in observed_classes]
    classes.extend(sorted(observed_classes.difference(UI_DEFECT_CLASSES)))
    per_class = {}
    image_per_class = {}
    confusion: Counter[str] = Counter()
    for label in classes:
        pred_text = "\n".join(
            f"<ref>{item['label']}</ref><box><{item['box'][0]}><{item['box'][1]}><{item['box'][2]}><{item['box'][3]}></box>"
            for item in predicted
            if item["label"] == label
        )
        gold_text = "\n".join(
            f"<ref>{item['label']}</ref><box><{item['box'][0]}><{item['box'][1]}><{item['box'][2]}><{item['box'][3]}></box>"
            for item in gold
            if item["label"] == label
        )
        per_class[label] = one_to_one_boxes(
            pred_text,
            gold_text,
            iou_threshold=iou_threshold,
            label_aware=True,
        )

    image_classes = [*UI_DEFECT_CLASSES]
    image_classes.extend(sorted(observed_classes.difference(UI_DEFECT_CLASSES)))
    for label in image_classes:
        gold_positive = any(item["label"] == label for item in gold)
        pred_positive = any(item["label"] == label for item in predicted)
        image_per_class[label] = {
            "tp": int(gold_positive and pred_positive),
            "fp": int(not gold_positive and pred_positive),
            "fn": int(gold_positive and not pred_positive),
            "tn": int(not gold_positive and not pred_positive),
            "gold_positive": int(gold_positive),
            "predicted_positive": int(pred_positive),
        }

    # Location matching without class compatibility exposes class confusion.
    weights = [
        [
            box_iou(g["box"], p["box"])
            if valid_box(g["box"]) and valid_box(p["box"])
            else 0.0
            for p in predicted
        ]
        for g in gold
    ]
    for left, right, iou in maximum_weight_matching(weights):
        if iou >= iou_threshold:
            confusion[f"{gold[left]['label']}->{predicted[right]['label']}"] += 1
    macro_values = [value["f1"] for value in per_class.values()]
    return {
        "iou_threshold": iou_threshold,
        "defect_macro_f1": sum(macro_values) / len(macro_values) if macro_values else 1.0,
        "defect_per_class": per_class,
        "defect_image_per_class": image_per_class,
        "defect_confusion": dict(sorted(confusion.items())),
        "defect_gold_count": len(gold),
        "defect_pred_count": len(predicted),
        "format_valid": (
            (
                bool(predicted)
                and len(predicted) == len(parse_boxes(prediction))
                and all(valid_box(item["box"]) for item in predicted)
            )
            or (not predicted and bool(BOX_NONE_RE.search(prediction)))
        ),
    }


def score_task(
    task: str,
    prediction: str,
    target: str,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    base = {
        "exact_match": float(normalize_text(prediction) == normalize_text(target)),
        "char_f1": char_f1(prediction, target),
        "rouge_l": rouge_l(prediction, target),
        "length_ratio": len(normalize_text(prediction)) / max(len(normalize_text(target)), 1),
    }
    if task == "vqa":
        pred_label, gold_label = parse_vqa_label(prediction), parse_vqa_label(target)
        base.update(
            {
                "parsed_vqa_prediction": pred_label,
                "parsed_vqa_target": gold_label,
                "vqa_accuracy": float(pred_label == gold_label) if pred_label is not None and gold_label is not None else 0.0,
                "vqa_invalid": float(pred_label is None),
                "vqa_confusion": f"{gold_label}->{pred_label}",
            }
        )
    elif task in {"agent_action", "agent_grounding"}:
        point = point_metrics(prediction, target)
        base.update(point)
        if task == "agent_action":
            predicted_action, target_action = parse_action_type(prediction), parse_action_type(target)
            action_accuracy = float(
                predicted_action == target_action
                and predicted_action is not None
                and target_action is not None
            )
            coordinate_required = target_action in {"tap", "swipe"}
            base.update(
                {
                    "predicted_action_type": predicted_action,
                    "target_action_type": target_action,
                    "action_type_accuracy": action_accuracy,
                    "action_hit50": action_accuracy * point["point_hit50"] if coordinate_required else action_accuracy,
                }
            )
    elif task == "ui_defect":
        base.update(
            defect_metrics(
                prediction,
                target,
                iou_threshold=iou_threshold,
            )
        )
    elif task in {"all_ui_elements", "ocr"}:
        location = one_to_one_boxes(
            prediction,
            target,
            iou_threshold=iou_threshold,
            label_aware=False,
        )
        label = one_to_one_boxes(
            prediction,
            target,
            iou_threshold=iou_threshold,
            label_aware=True,
        )
        prefix = "ocr" if task == "ocr" else "box"
        base.update(
            {
                "iou_threshold": iou_threshold,
                f"{prefix}_precision": label["precision"] if task == "ocr" else location["precision"],
                f"{prefix}_recall": label["recall"] if task == "ocr" else location["recall"],
                f"{prefix}_f1": label["f1"] if task == "ocr" else location["f1"],
                "location_metrics": location,
                "label_aware_metrics": label,
                "label_accuracy": (
                    location["label_exact_sum"] / location["location_match_count"]
                    if location["location_match_count"]
                    else None
                ),
                "type_accuracy": (
                    location["type_exact_sum"] / location["type_match_count"]
                    if location["type_match_count"]
                    else None
                ),
                "matched_label_char_f1": (
                    location["label_char_f1_sum"] / location["location_match_count"]
                    if location["location_match_count"]
                    else None
                ),
                "format_valid": location["format_valid"],
            }
        )
    elif task == "single_grounding":
        location = one_to_one_boxes(
            prediction,
            target,
            iou_threshold=iou_threshold,
            label_aware=False,
        )
        gold = parse_labeled_boxes(target)
        predicted = parse_labeled_boxes(prediction)
        weights = [
            [
                box_iou(g["box"], p["box"])
                if valid_box(g["box"]) and valid_box(p["box"])
                else 0.0
                for p in predicted
            ]
            for g in gold
        ]
        assignments = maximum_weight_matching(weights)
        best_ious = [weight for _, _, weight in assignments]
        base.update(
            {
                "iou_threshold": iou_threshold,
                "box_recall": location["recall"],
                "mean_iou": sum(best_ious) / len(gold) if gold else None,
                "box_iou_sum": sum(best_ious),
                "format_valid": location["format_valid"],
                "location_metrics": location,
            }
        )
    primary_name = PRIMARY_METRIC_BY_TASK.get(task, "char_f1")
    primary = base.get(primary_name)
    base["primary_name"] = primary_name
    base["primary_metric"] = float(primary) if isinstance(primary, (int, float)) else None
    return base


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return sum(numeric) / len(numeric) if numeric else None


def aggregate_scores(
    task: str,
    scores: Sequence[dict[str, Any]],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    scalar_keys = sorted(
        {
            key
            for score in scores
            for key, value in score.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    output = {key: _mean(score.get(key) for score in scores) for key in scalar_keys}
    output.update(
        {
            "task": task,
            "examples": len(scores),
            "inference_error_count": sum(
                int(bool(score.get("evaluation_error"))) for score in scores
            ),
            "primary_name": PRIMARY_METRIC_BY_TASK.get(task, "char_f1"),
            "primary_metric": _mean(score.get("primary_metric") for score in scores),
        }
    )
    if task in {"agent_action", "agent_grounding"}:
        distances = sorted(
            float(score["point_distance"])
            for score in scores
            if isinstance(score.get("point_distance"), (int, float))
        )
        output["point_mean_l2"] = _mean(distances)
        if distances:
            middle = len(distances) // 2
            output["point_median_l2"] = (
                distances[middle]
                if len(distances) % 2
                else (distances[middle - 1] + distances[middle]) / 2.0
            )
        else:
            output["point_median_l2"] = None
        if task == "agent_action":
            output["action_confusion"] = dict(
                sorted(
                    Counter(
                        f"{score.get('target_action_type')}->{score.get('predicted_action_type')}"
                        for score in scores
                    ).items()
                )
            )
    if task in {"all_ui_elements", "ocr"}:
        def micro(key: str) -> dict[str, float | int]:
            tp = sum(int(score[key]["tp"]) for score in scores)
            fp = sum(int(score[key]["fp"]) for score in scores)
            fn = sum(int(score[key]["fn"]) for score in scores)
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 1.0
            return {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }

        location = micro("location_metrics")
        label_aware = micro("label_aware_metrics")
        output["location_micro"] = location
        output["label_aware_micro"] = label_aware
        matched_iou_sum = sum(float(score["location_metrics"]["matched_iou_sum"]) for score in scores)
        matched = sum(int(score["location_metrics"]["tp"]) for score in scores)
        output["matched_iou_mean"] = matched_iou_sum / matched if matched else None
        label_exact_sum = sum(float(score["location_metrics"]["label_exact_sum"]) for score in scores)
        label_match_count = sum(int(score["location_metrics"]["location_match_count"]) for score in scores)
        output["label_accuracy"] = label_exact_sum / label_match_count if label_match_count else None
        type_exact_sum = sum(float(score["location_metrics"]["type_exact_sum"]) for score in scores)
        type_match_count = sum(int(score["location_metrics"]["type_match_count"]) for score in scores)
        output["type_accuracy"] = type_exact_sum / type_match_count if type_match_count else None
        output["primary_metric"] = (
            label_aware["f1"] if task == "ocr" else location["f1"]
        )
    if task == "single_grounding":
        gold = sum(int(score["location_metrics"]["gold_count"]) for score in scores)
        predictions = sum(int(score["location_metrics"]["pred_count"]) for score in scores)
        hits = sum(int(score["location_metrics"]["tp"]) for score in scores)
        iou_sum = sum(float(score.get("box_iou_sum", 0.0)) for score in scores)
        output.update(
            {
                "box_gold_count": gold,
                "box_pred_count": predictions,
                "iou_threshold": iou_threshold,
                "box_hits": hits,
                "box_recall": hits / gold if gold else float(predictions == 0),
                "mean_iou": iou_sum / gold if gold else None,
            }
        )
        output["primary_metric"] = output["box_recall"]
    if task == "vqa":
        output["confusion"] = dict(sorted(Counter(score.get("vqa_confusion") for score in scores).items()))
    if task == "ui_defect":
        per_class_box_counts: dict[str, Counter[str]] = defaultdict(Counter)
        per_class_image_counts: dict[str, Counter[str]] = defaultdict(Counter)
        confusion: Counter[str] = Counter()
        for score in scores:
            for label, metrics in score.get("defect_per_class", {}).items():
                for key in ("tp", "fp", "fn"):
                    per_class_box_counts[label][key] += int(metrics[key])
            for label, metrics in score.get("defect_image_per_class", {}).items():
                for key in ("tp", "fp", "fn", "tn"):
                    per_class_image_counts[label][key] += int(metrics[key])
            confusion.update(score.get("defect_confusion", {}))

        def prf(counts: Counter[str], *, include_tn: bool = False) -> dict[str, Any]:
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            tn = counts["tn"] if include_tn else 0
            active = tp + fp + fn
            if active:
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
            else:
                precision = recall = f1 = None
            result = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            if include_tn:
                total = tp + fp + fn + tn
                result.update(
                    tn=tn,
                    accuracy=(tp + tn) / total if total else None,
                    images=total,
                )
            return result

        observed = set(per_class_box_counts) | set(per_class_image_counts)
        classes = [*UI_DEFECT_CLASSES]
        classes.extend(sorted(observed.difference(UI_DEFECT_CLASSES)))
        per_class = {}
        for label in classes:
            bbox = prf(per_class_box_counts[label])
            image = prf(per_class_image_counts[label], include_tn=True)
            per_class[label] = {
                **bbox,
                "display_label": UI_DEFECT_CLASS_DISPLAY.get(label, label),
                "bbox": bbox,
                "image": image,
            }

        def macro(granularity: str) -> dict[str, float | None]:
            values = [
                per_class[label][granularity]
                for label in UI_DEFECT_CLASSES
                if per_class[label][granularity]["f1"] is not None
            ]
            return {
                metric: _mean(item[metric] for item in values)
                for metric in ("precision", "recall", "f1")
            }

        def micro(granularity: str) -> dict[str, Any]:
            counts: Counter[str] = Counter()
            for value in per_class.values():
                for key in ("tp", "fp", "fn", "tn"):
                    counts[key] += int(value[granularity].get(key) or 0)
            return prf(counts, include_tn=granularity == "image")

        bbox_macro = macro("bbox")
        image_macro = macro("image")
        bbox_micro = micro("bbox")
        image_micro = micro("image")
        output["per_class"] = per_class
        output["bbox_macro"] = bbox_macro
        output["image_macro"] = image_macro
        output["bbox_micro"] = bbox_micro
        output["image_micro"] = image_micro
        output["iou_threshold"] = iou_threshold
        output["defect_bbox_macro_f1"] = bbox_macro["f1"]
        output["defect_image_macro_f1"] = image_macro["f1"]
        output["defect_bbox_micro_f1"] = bbox_micro["f1"]
        output["defect_image_micro_f1"] = image_micro["f1"]
        output["defect_macro_f1"] = bbox_macro["f1"]
        output["confusion"] = dict(sorted(confusion.items()))
        output["primary_metric"] = bbox_macro["f1"]
    if task in {"all_ui_elements", "ocr"}:
        box_units = sum(
            int(score.get("location_metrics", {}).get("gold_count", 0))
            for score in scores
        )
        output["primary_weight"] = box_units or len(scores)
    elif task == "single_grounding":
        box_units = sum(
            int(score.get("location_metrics", {}).get("gold_count", 0))
            for score in scores
        )
        output["primary_weight"] = box_units or len(scores)
    elif task == "ui_defect":
        box_units = sum(int(score.get("defect_gold_count", 0)) for score in scores)
        output["primary_weight"] = box_units or len(scores)
    else:
        output["primary_weight"] = len(scores)

    # A model/processor exception is observable in the error artifact, but it
    # must never disappear from the denominator and inflate held-out scores.
    error_count = int(output["inference_error_count"])
    sample_primary = _mean(score.get("primary_metric") for score in scores)
    if (
        error_count
        and isinstance(output.get("primary_metric"), (int, float))
        and isinstance(sample_primary, (int, float))
    ):
        output["primary_metric_before_error_penalty"] = output["primary_metric"]
        output["primary_metric"] = min(float(output["primary_metric"]), sample_primary)
    return output


def task_macro_primary(per_task: dict[str, dict[str, Any]]) -> float | None:
    return _mean(value.get("primary_metric") for value in per_task.values())


def micro_primary(per_task: dict[str, dict[str, Any]]) -> float | None:
    """Weight primary metrics by evaluated sample or ground-truth box units."""
    numerator = 0.0
    denominator = 0.0
    for value in per_task.values():
        primary = value.get("primary_metric")
        weight = value.get("primary_weight")
        if not isinstance(primary, (int, float)) or not isinstance(weight, (int, float)):
            continue
        if not math.isfinite(float(primary)) or float(weight) <= 0:
            continue
        numerator += float(primary) * float(weight)
        denominator += float(weight)
    return numerator / denominator if denominator else None
