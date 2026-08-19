#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并 Qwen3-VL 单图推理结果、将相对坐标转换为原图真实像素坐标，并计算文字溢出检测指标。

Qwen3-VL 输出示例：
    0个问题

或：
    1个问题，<|object_ref_start|>文字溢出容器<|object_ref_end|>
    <|box_start|>(463,115),(522,146)<|box_end|>

默认将坐标视为 0~1000 的相对坐标：
    real_x = qwen_x / 1000 * image_width
    real_y = qwen_y / 1000 * image_height

输出 pred_ans：
{
    "has_issue": true,
    "issues": [
        {
            "type": "文字溢出容器",
            "bbox": [x1, y1, x2, y2],
            "bbox_type": "real"
        }
    ]
}
"""
from datetime import datetime
import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment


TASK_CONFIG = {
    "text_overflow": {
        "issue_name": "文字溢出容器",
        "gt_filename": "test_ui_text_overflow_wcnt_no_figma.jsonl",
        "colleague_filename": "test_ui_text_overflow_wcnt.jsonl",
        "swift_pattern": "test_ui_text_overflow_wcnt_no_figma*.jsonl",
    },
    "text_ellipsis": {
        "issue_name": "文字省略异常",
        "gt_filename": "test_ui_text_ellipsis_wcnt_no_figma.jsonl",
        "colleague_filename": "test_ui_text_ellipsis_wcnt.jsonl",
        "swift_pattern": "test_ui_text_ellipsis_wcnt_no_figma*.jsonl",
    },
    "occlusion": {
        "issue_name": "元素重叠",
        "gt_filename": "test_ui_occlusion_wcnt_no_figma.jsonl",
        "colleague_filename": "test_ui_occlusion_wcnt.jsonl",
        "swift_pattern": "test_ui_occlusion_wcnt_no_figma*.jsonl",
    },
    "cropping": {
        "issue_name": "元素被裁切",
        "gt_filename": "test_ui_cropping_wcnt_no_figma.jsonl",
        "colleague_filename": "test_ui_cropping_wcnt.jsonl",
        "swift_pattern": "test_ui_cropping_wcnt_no_figma*.jsonl",
    },
    "content_missing": {
        "issue_name": "内容未展示",
        "gt_filename": "test_ui_content_missing_wcnt_no_figma.jsonl",
        "colleague_filename": "test_ui_content_missing_wcnt.jsonl",
        "swift_pattern": "test_ui_content_missing_wcnt_no_figma*.jsonl",
    },
}

DEFAULT_TASK = "text_overflow"

BOX_PATTERN = re.compile(
    r"<\|box_start\|>\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"\s*[,，]\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"\s*<\|box_end\|>",
    flags=re.IGNORECASE,
)

# 兼容模型偶尔漏掉特殊 token、只输出 "(x1,y1),(x2,y2)" 的情况。
PLAIN_BOX_PATTERN = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"\s*[,，]\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\)"
)

COUNT_PATTERN = re.compile(r"(\d+)\s*个问题")


SCRIPT_VERSION = "2026-07-16-fix-images-list-v2"


def strip_file_uri(path: str) -> str:
    """将 file:///path/a.jpg 还原为本地绝对路径。"""
    if path.startswith("file://"):
        return path[7:]
    return path


def extract_image_path(payload: Any) -> str | None:
    """
    同时支持：
      {"image": "/path/a.jpg"}
      {"images": ["/path/a.jpg"]}
      {"images": [{"path": "/path/a.jpg"}]}
    """
    if not isinstance(payload, dict):
        return None

    image = payload.get("image")
    if isinstance(image, str) and image.strip():
        return strip_file_uri(image.strip())
    if isinstance(image, dict):
        path = image.get("path")
        if isinstance(path, str) and path.strip():
            return strip_file_uri(path.strip())

    images = payload.get("images")
    if isinstance(images, (list, tuple)) and images:
        first_image = images[0]
        if isinstance(first_image, str) and first_image.strip():
            return strip_file_uri(first_image.strip())
        if isinstance(first_image, dict):
            path = first_image.get("path")
            if isinstance(path, str) and path.strip():
                return strip_file_uri(path.strip())

    return None


def get_image_size(image_path: str) -> tuple[int, int]:
    """读取经过 EXIF 方向校正后的原图宽高。"""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def qwen_box_to_real(
    box: list[float],
    image_width: int,
    image_height: int,
    coord_base: float = 1000.0,
) -> list[int]:
    """
    将 Qwen3-VL 相对坐标 [x1,y1,x2,y2] 转换为原图真实像素坐标。

    注意：这里是 xyxy，不是 xywh。
    """
    if coord_base <= 0:
        raise ValueError("coord_base 必须大于 0")

    x1, y1, x2, y2 = box

    x1 = clip(x1 / coord_base * image_width, 0, image_width)
    x2 = clip(x2 / coord_base * image_width, 0, image_width)
    y1 = clip(y1 / coord_base * image_height, 0, image_height)
    y2 = clip(y2 / coord_base * image_height, 0, image_height)

    # 防止模型偶尔把左上角和右下角顺序写反。
    real_x1, real_x2 = sorted((int(round(x1)), int(round(x2))))
    real_y1, real_y2 = sorted((int(round(y1)), int(round(y2))))

    return [real_x1, real_y1, real_x2, real_y2]


def parse_qwen3vl_response(
    response: str,
    image_width: int,
    image_height: int,
    target_issue: str,
    coord_base: float = 1000.0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    解析 Qwen3-VL 文本输出。

    返回：
      pred_ans:
        - 合法输出时为标准字典；
        - 声称有问题但完全没有解析到框时为 None，让评测代码按非法输出惩罚。
      parse_meta:
        保存声明数量、解析框数量和解析状态，便于排查。
    """
    response = response.strip()

    count_match = COUNT_PATTERN.search(response)
    declared_count = int(count_match.group(1)) if count_match else None

    matches = BOX_PATTERN.findall(response)

    # 特殊 token 完全缺失时，才启用纯坐标兜底，避免重复解析。
    if not matches:
        matches = PLAIN_BOX_PATTERN.findall(response)

    relative_boxes: list[list[float]] = [
        [float(x1), float(y1), float(x2), float(y2)]
        for x1, y1, x2, y2 in matches
    ]

    real_boxes = [
        qwen_box_to_real(
            box=box,
            image_width=image_width,
            image_height=image_height,
            coord_base=coord_base,
        )
        for box in relative_boxes
    ]

    # 去除退化框和重复框。
    valid_real_boxes: list[list[int]] = []
    valid_relative_boxes: list[list[float]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for relative_box, real_box in zip(relative_boxes, real_boxes):
        x1, y1, x2, y2 = real_box
        if x2 <= x1 or y2 <= y1:
            continue

        key = tuple(real_box)
        if key in seen:
            continue

        seen.add(key)
        valid_relative_boxes.append(relative_box)
        valid_real_boxes.append(real_box)

    parsed_count = len(valid_real_boxes)

    if declared_count == 0 and parsed_count == 0:
        status = "ok_no_issue"
        pred_ans: dict[str, Any] | None = {
            "has_issue": False,
            "issues": [],
        }
    elif parsed_count > 0:
        status = (
            "ok"
            if declared_count is None or declared_count == parsed_count
            else "count_bbox_mismatch"
        )
        pred_ans = {
            "has_issue": True,
            "issues": [
                {
                    "type": target_issue,
                    "bbox": bbox,
                    "bbox_type": "real",
                }
                for bbox in valid_real_boxes
            ],
        }
    elif declared_count is not None and declared_count > 0:
        # 模型声称存在问题，却没有产生任何可解析坐标。
        # 不应伪装成“0 个问题”，否则图片级判断会被错误当作阴性。
        status = "positive_without_valid_bbox"
        pred_ans = None
    else:
        status = "unrecognized_output"
        pred_ans = None

    parse_meta = {
        "parse_status": status,
        "declared_count": declared_count,
        "parsed_bbox_count": parsed_count,
        "relative_bboxes": valid_relative_boxes,
        "image_width": image_width,
        "image_height": image_height,
        "coord_base": coord_base,
    }
    return pred_ans, parse_meta


def find_prediction_file(pred_dir: str, file_id: str) -> str | None:
    """
    优先匹配 file_id.json，同时兼容 _ok、_defect、_error 等后缀。
    """
    exact_path = os.path.join(pred_dir, f"{file_id}.json")
    if os.path.isfile(exact_path):
        return exact_path

    candidates = sorted(glob.glob(os.path.join(pred_dir, f"{file_id}*.json")))
    return candidates[0] if candidates else None


def get_prediction_image_path(raw_pred: Any) -> str | None:
    return extract_image_path(raw_pred)

def load_swift_jsonl_predictions(pred_jsonl: str) -> dict[str, dict[str, Any]]:
    """
    读取 swift infer 生成的汇总 JSONL，并按图片 ID 建立索引。

    每行格式示例：
    {
        "response": "0个问题",
        "images": [{"bytes": null, "path": "/path/153779.jpg"}],
        "objects": {...}
    }
    """
    predictions: dict[str, dict[str, Any]] = {}
    invalid_lines = 0
    duplicate_ids = 0

    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_lines += 1
                print(
                    f"⚠️ Swift JSONL 第 {line_number} 行解析失败：{exc}"
                )
                continue

            image_path = extract_image_path(sample)
            if not image_path:
                invalid_lines += 1
                print(
                    f"⚠️ Swift JSONL 第 {line_number} 行没有有效图片路径"
                )
                continue

            file_id = (
                os.path.splitext(os.path.basename(image_path))[0]
                .replace(":", "_")
            )

            if file_id in predictions:
                duplicate_ids += 1
                print(
                    f"⚠️ Swift JSONL 出现重复图片 ID，使用最后一条："
                    f"{file_id}"
                )

            predictions[file_id] = sample

    print(f">>> Swift JSONL 有效预测数：{len(predictions)}")
    print(f">>> Swift JSONL 无效行数：{invalid_lines}")
    print(f">>> Swift JSONL 重复 ID 数：{duplicate_ids}")

    return predictions


def merge_gt_and_swift_jsonl_preds(
    gt_path: str,
    pred_jsonl: str,
    output_path: str,
    coord_base: float,
    target_issue: str,
) -> dict[str, int]:
    """
    将 swift infer 汇总 JSONL 中的 response 按图片 ID 合并到 GT，
    并将 Qwen3-VL 的 0~1000 坐标转换成原图真实像素坐标。
    """
    predictions = load_swift_jsonl_predictions(pred_jsonl)

    total_lines = 0
    matched_files = 0
    missing_files = 0
    parse_errors = 0
    count_mismatches = 0

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True,
    )

    with open(gt_path, "r", encoding="utf-8") as f_gt, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f_out:
        for line_number, line in enumerate(f_gt, start=1):
            if not line.strip():
                continue

            total_lines += 1
            data = json.loads(line)

            gt_image_path = extract_image_path(data)

            if not gt_image_path:
                parse_errors += 1
                data["pred_ans"] = None
                data["pred_raw_response"] = ""
                data["pred_parse_info"] = {
                    "parse_status": "missing_gt_image_path",
                    "line_number": line_number,
                }
                f_out.write(
                    json.dumps(data, ensure_ascii=False) + "\n"
                )
                continue

            file_id = (
                os.path.splitext(os.path.basename(gt_image_path))[0]
                .replace(":", "_")
            )

            raw_pred = predictions.get(file_id)

            if raw_pred is None:
                missing_files += 1
                data["pred_ans"] = None
                data["pred_raw_response"] = ""
                data["pred_parse_info"] = {
                    "parse_status": "missing_prediction",
                    "expected_file_id": file_id,
                }
                f_out.write(
                    json.dumps(data, ensure_ascii=False) + "\n"
                )
                print(f"未找到 Swift JSONL 预测：{file_id}")
                continue

            try:
                response = raw_pred.get("response", "")
                if not isinstance(response, str):
                    response = str(response)

                # 优先使用 GT 图片路径。
                image_path = gt_image_path

                if not os.path.isfile(image_path):
                    pred_image_path = extract_image_path(raw_pred)

                    if (
                        pred_image_path
                        and os.path.isfile(pred_image_path)
                    ):
                        image_path = pred_image_path
                    else:
                        raise FileNotFoundError(
                            f"图片不存在：{gt_image_path}"
                        )

                image_width, image_height = get_image_size(image_path)

                pred_ans, parse_meta = parse_qwen3vl_response(
                    response=response,
                    image_width=image_width,
                    image_height=image_height,
                    coord_base=coord_base,
                    target_issue=target_issue,
                )

                parse_meta["source_mode"] = "swift_jsonl"
                parse_meta["source_jsonl"] = pred_jsonl
                parse_meta["resolved_image_path"] = image_path

                data["pred_ans"] = pred_ans
                data["pred_raw_response"] = response
                data["pred_parse_info"] = parse_meta

                parse_status = parse_meta["parse_status"]

                if parse_status in {
                    "positive_without_valid_bbox",
                    "unrecognized_output",
                }:
                    parse_errors += 1
                elif parse_status == "count_bbox_mismatch":
                    count_mismatches += 1

                matched_files += 1

            except Exception as exc:
                parse_errors += 1
                data["pred_ans"] = None
                data["pred_raw_response"] = raw_pred.get(
                    "response",
                    "",
                )
                data["pred_parse_info"] = {
                    "parse_status": "prediction_processing_error",
                    "source_jsonl": pred_jsonl,
                    "error": f"{type(exc).__name__}: {exc}",
                }

                print(f"解析 Swift 预测 {file_id} 时出错：{exc}")

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    stats = {
        "total_lines": total_lines,
        "matched_files": matched_files,
        "missing_files": missing_files,
        "parse_errors": parse_errors,
        "count_mismatches": count_mismatches,
    }

    print("\n--- Swift JSONL 合并完成 ---")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"已保存至：{output_path}")
    return stats


def standardize_yolo_pred(
    raw_pred: Any,
    target_issue: str,
    bbox_format: str = "xyxy",
) -> dict[str, Any] | None:
    """
    YOLO 原始格式：

    []

    或：

    [
        {
            "bbox_2d": [x1, y1, x2, y2],
            "label": "文字溢出"
        }
    ]

    转为统一 pred_ans：

    {
        "has_issue": true,
        "issues": [
            {
                "type": "文字溢出容器",
                "bbox": [x1,y1,x2,y2],
                "bbox_type": "real"
            }
        ]
    }
    """
    if not isinstance(raw_pred, list):
        return None

    standard_pred = {
        "has_issue": False,
        "issues": [],
    }

    for item in raw_pred:
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox_2d")
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(is_number(value) for value in bbox)
        ):
            continue

        x1, y1, value3, value4 = [float(value) for value in bbox]

        if bbox_format == "xywh":
            x2 = x1 + value3
            y2 = y1 + value4
        else:
            x2 = value3
            y2 = value4

        # 防止坐标顺序反了。
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        # 删除宽或高为 0 的无效框。
        if x2 <= x1 or y2 <= y1:
            continue

        label_mapping = {
            # 文字溢出
            "文字溢出": "文字溢出容器",
            "文字溢出容器": "文字溢出容器",
            "text overflow": "文字溢出容器",

            # 文本省略
            "文本省略": "文字省略异常",
            "文字省略": "文字省略异常",
            "文字省略异常": "文字省略异常",
            "text truncation error": "文字省略异常",

            # 元素重叠/遮挡
            "元素遮挡": "元素重叠",
            "元素重叠": "元素重叠",
            "element overlap": "元素重叠",
            "UI element overlap": "元素重叠",

            # 元素裁切
            "元素裁切": "元素被裁切",
            "元素被裁切": "元素被裁切",
            "element cropping": "元素被裁切",
            "UI element cropping": "元素被裁切",

            # 内容缺失
            "内容缺失": "内容未展示",
            "内容未展示": "内容未展示",
            "content missing": "内容未展示",
            "content not displayed": "内容未展示",
        }

        class_id_mapping = {
            0: "元素重叠",
            1: "元素被裁切",
            2: "文字溢出容器",
            3: "文字省略异常",
            4: "内容未展示",
        }

        class_id = item.get("class_id")

        if class_id in class_id_mapping:
            label = class_id_mapping[class_id]
        else:
            raw_label = item.get("label", target_issue)
            raw_label = str(raw_label).strip()
            label = label_mapping.get(raw_label, raw_label)

        # 每个任务只评测自己的目标类别。
        if label != target_issue:
            continue

        standard_pred["issues"].append(
            {
                "type": target_issue,
                "bbox": [x1, y1, x2, y2],
                "bbox_type": "real",
            }
        )

    standard_pred["has_issue"] = len(standard_pred["issues"]) > 0
    return standard_pred


def merge_gt_and_yolo_dir_preds(
    gt_path: str,
    pred_dir: str,
    output_path: str,
    bbox_format: str,
    target_issue: str,
) -> dict[str, int]:
    total_lines = 0
    matched_files = 0
    missing_files = 0
    parse_errors = 0

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True,
    )

    print("开始合并 YOLO 结果...")
    print(f"GT 文件: {gt_path}")
    print(f"预测目录: {pred_dir}")
    print(f"bbox 格式: {bbox_format}")

    with open(gt_path, "r", encoding="utf-8") as f_gt, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f_out:
        for line_number, line in enumerate(f_gt, start=1):
            if not line.strip():
                continue

            total_lines += 1
            data = json.loads(line)

            # 使用已有的统一图片路径读取函数，
            # 兼容 image 和 images 两种 GT。
            image_path = extract_image_path(data)

            if not image_path:
                parse_errors += 1
                data["pred_ans"] = None
                data["pred_parse_info"] = {
                    "parse_status": "missing_gt_image_path",
                    "line_number": line_number,
                }
                f_out.write(
                    json.dumps(data, ensure_ascii=False) + "\n"
                )
                continue

            file_id = (
                os.path.splitext(os.path.basename(image_path))[0]
                .replace(":", "_")
            )

            # 复用已有函数，兼容：
            # 155329.json
            # 155329_defect.json
            # 155329_ok.json
            pred_file_path = find_prediction_file(pred_dir, file_id)

            if pred_file_path is None:
                missing_files += 1

                # 不要 continue 丢掉 GT 样本。
                data["pred_ans"] = None
                data["pred_parse_info"] = {
                    "parse_status": "missing_prediction_file",
                    "expected_file_id": file_id,
                }

                f_out.write(
                    json.dumps(data, ensure_ascii=False) + "\n"
                )
                print(f"未找到 YOLO 预测：{file_id}")
                continue

            try:
                with open(pred_file_path, "r", encoding="utf-8") as f_pred:
                    raw_pred = json.load(f_pred)

                pred_ans = standardize_yolo_pred(
                    raw_pred=raw_pred,
                    bbox_format=bbox_format,
                    target_issue=target_issue,
                )

                data["pred_ans"] = pred_ans
                data["pred_parse_info"] = {
                    "parse_status": (
                        "ok"
                        if pred_ans is not None
                        else "invalid_yolo_prediction"
                    ),
                    "prediction_file": pred_file_path,
                    "bbox_format": bbox_format,
                }

                if pred_ans is None:
                    parse_errors += 1
                else:
                    matched_files += 1

            except Exception as exc:
                parse_errors += 1
                data["pred_ans"] = None
                data["pred_parse_info"] = {
                    "parse_status": "prediction_processing_error",
                    "prediction_file": pred_file_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"解析 YOLO 预测 {file_id} 时出错：{exc}")

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    stats = {
        "total_lines": total_lines,
        "matched_files": matched_files,
        "missing_files": missing_files,
        "parse_errors": parse_errors,
    }

    print("\n--- YOLO 合并完成 ---")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"已保存至：{output_path}")
    return stats


def merge_gt_and_qwen3vl_preds(
    gt_path: str,
    pred_dir: str,
    output_path: str,
    coord_base: float,
    target_issue: str,
) -> dict[str, int]:
    total_lines = 0
    matched_files = 0
    missing_files = 0
    parse_errors = 0
    count_mismatches = 0
    missing_gt_image_fields = 0
    missing_prediction_ids: list[str] = []

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(
        f"开始合并 Qwen3-VL 结果...\n"
        f"GT 文件: {gt_path}\n"
        f"预测目录: {pred_dir}\n"
        f"坐标基准: 0~{coord_base:g}\n"
        f"脚本版本: {SCRIPT_VERSION}"
    )

    with open(gt_path, "r", encoding="utf-8") as f_gt, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line_number, line in enumerate(f_gt, start=1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            data = json.loads(line)

            gt_image_path = extract_image_path(data)

            if total_lines <= 3:
                print(
                    f"[DEBUG GT {total_lines}] "
                    f"images={data.get('images')!r}, "
                    f"image={data.get('image')!r}, "
                    f"resolved_path={gt_image_path!r}"
                )

            if not gt_image_path:
                data["pred_ans"] = None
                data["pred_raw_response"] = ""
                data["pred_parse_info"] = {
                    "parse_status": "missing_gt_image_field",
                    "line_number": line_number,
                    "raw_images": data.get("images"),
                    "raw_image": data.get("image"),
                }
                missing_gt_image_fields += 1
                parse_errors += 1
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            filename_with_ext = os.path.basename(gt_image_path)
            file_id = os.path.splitext(filename_with_ext)[0].replace(":", "_")
            pred_file_path = find_prediction_file(pred_dir, file_id)

            if pred_file_path is None:
                # 不能 continue，否则该样本会从评测集消失，导致分数虚高。
                missing_files += 1
                missing_prediction_ids.append(file_id)
                data["pred_ans"] = None
                data["pred_raw_response"] = ""
                data["pred_parse_info"] = {
                    "parse_status": "missing_prediction_file",
                    "expected_file_id": file_id,
                }
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                print(f"未找到预测文件: {file_id}.json")
                continue

            try:
                with open(pred_file_path, "r", encoding="utf-8") as f_pred:
                    raw_pred_data = json.load(f_pred)

                if not isinstance(raw_pred_data, dict):
                    raise TypeError(
                        f"Qwen3-VL 预测文件顶层应为 dict，实际为 "
                        f"{type(raw_pred_data).__name__}"
                    )

                response = raw_pred_data.get("response", "")
                if not isinstance(response, str):
                    response = str(response)

                # 优先使用 GT 的原图路径；若路径已经迁移，则尝试预测结果中记录的路径。
                image_path = gt_image_path
                if not os.path.isfile(image_path):
                    pred_image_path = get_prediction_image_path(raw_pred_data)
                    if pred_image_path and os.path.isfile(pred_image_path):
                        image_path = pred_image_path
                    else:
                        raise FileNotFoundError(f"无法读取原图: {gt_image_path}")

                image_width, image_height = get_image_size(image_path)
                pred_ans, parse_meta = parse_qwen3vl_response(
                    response=response,
                    image_width=image_width,
                    image_height=image_height,
                    coord_base=coord_base,
                    target_issue=target_issue,
                )

                parse_meta["prediction_file"] = pred_file_path
                parse_meta["resolved_image_path"] = image_path

                data["pred_ans"] = pred_ans
                data["pred_raw_response"] = response
                data["pred_parse_info"] = parse_meta

                if parse_meta["parse_status"] in {
                    "positive_without_valid_bbox",
                    "unrecognized_output",
                }:
                    parse_errors += 1
                elif parse_meta["parse_status"] == "count_bbox_mismatch":
                    count_mismatches += 1

                matched_files += 1

            except Exception as exc:
                parse_errors += 1
                data["pred_ans"] = None
                data["pred_raw_response"] = ""
                data["pred_parse_info"] = {
                    "parse_status": "prediction_processing_error",
                    "prediction_file": pred_file_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"解析 {pred_file_path} 时出错: {exc}")

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    stats = {
        "total_lines": total_lines,
        "matched_files": matched_files,
        "missing_files": missing_files,
        "parse_errors": parse_errors,
        "count_mismatches": count_mismatches,
        "missing_gt_image_fields": missing_gt_image_fields,
    }

    print("\n--- 合并完成 ---")
    for key, value in stats.items():
        print(f"{key}: {value}")
    if missing_prediction_ids:
        print("\n缺失预测 ID：")
        for missing_id in missing_prediction_ids[:50]:
            print(f"  - {missing_id}")
        if len(missing_prediction_ids) > 50:
            print(f"  ... 其余 {len(missing_prediction_ids) - 50} 条省略")

    print(f"已保存至: {output_path}")
    return stats


# ==============================================================================
# 以下算分逻辑与用户现有 minimal_scoring.py 保持一致
# ==============================================================================

EN2ZH = {
    "element overlap": "元素重叠",
    "UI element overlap": "元素重叠",
    "text overflow": "文字溢出容器",
    "文字溢出": "文字溢出容器",
    "text truncation error": "文字省略异常",
    "content not displayed": "内容未展示",
    "element cropping": "元素被裁切",
    "UI element cropping": "元素被裁切",
}


def norm_issue(issue_type: Any) -> str | None:
    if issue_type is None:
        return None
    if not isinstance(issue_type, str):
        issue_type = str(issue_type)
    issue_type = issue_type.strip()
    return EN2ZH.get(issue_type, issue_type)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_bboxes(raw_bboxes: Any) -> list[list[float]]:
    if raw_bboxes is None or not isinstance(raw_bboxes, (list, tuple)):
        return []

    if len(raw_bboxes) == 4 and all(is_number(v) for v in raw_bboxes):
        return [list(raw_bboxes)]

    bboxes: list[list[float]] = []
    for bbox in raw_bboxes:
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(is_number(v) for v in bbox)
        ):
            bboxes.append(list(bbox))
    return bboxes


def filter_bboxes_by_issue(
    bboxes: list[list[float]],
    types: Any,
    target_issue: str,
) -> list[list[float]]:
    if not bboxes:
        return []
    if not types:
        return bboxes
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, (list, tuple)):
        return bboxes

    norm_types = [norm_issue(x) for x in types]
    if len(norm_types) == len(bboxes):
        return [
            bbox
            for issue_type, bbox in zip(norm_types, bboxes)
            if issue_type == target_issue
        ]
    if target_issue not in norm_types:
        return []
    return bboxes


def extract_bboxes_for_issue(
    payload: Any,
    target_issue: str,
) -> list[list[float]]:
    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("ground_truth"), dict):
        return extract_bboxes_for_issue(payload["ground_truth"], target_issue)

    if isinstance(payload.get("issues"), list):
        bboxes: list[list[float]] = []
        for issue in payload["issues"]:
            if not isinstance(issue, dict):
                continue
            issue_type = norm_issue(issue.get("type") or issue.get("issue_type"))
            if issue_type == target_issue:
                bboxes.extend(
                    normalize_bboxes(issue.get("bbox") or issue.get("bboxes"))
                )
        return bboxes

    refs = payload.get("ref") or []
    bboxes = normalize_bboxes(payload.get("bbox") or payload.get("bboxes"))

    if refs:
        norm_refs = [norm_issue(ref) for ref in refs]
        if len(norm_refs) == len(bboxes) + 1:
            return [
                bbox
                for issue_type, bbox in zip(norm_refs[1:], bboxes)
                if issue_type == target_issue
            ]
        return filter_bboxes_by_issue(bboxes, norm_refs, target_issue)

    types = payload.get("types", []) or payload.get("type", []) or []
    return filter_bboxes_by_issue(bboxes, types, target_issue)


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """计算 [x1,y1,x2,y2] 格式框的 IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def get_sample_image_id(sample: dict[str, Any]) -> str:
    images = sample.get("images")
    if isinstance(images, list) and images:
        first_image = images[0]
        if isinstance(first_image, dict) and isinstance(first_image.get("path"), str):
            return first_image["path"]
        if isinstance(first_image, str):
            return first_image

    image = sample.get("image")
    return image if isinstance(image, str) else "unknown"


def is_figma_sample(sample: dict[str, Any]) -> bool:
    image_id = get_sample_image_id(sample)
    basename = os.path.basename(image_id)
    return "figma" in image_id.lower() or ":" in basename


def get_gt_payload(sample: dict[str, Any]) -> Any:
    extra_info = (
        sample.get("extra_info")
        if isinstance(sample.get("extra_info"), dict)
        else {}
    )
    reward_model = (
        sample.get("reward_model")
        if isinstance(sample.get("reward_model"), dict)
        else {}
    )

    return (
        sample.get("answer")
        or sample.get("objects")
        or reward_model.get("ground_truth")
        or extra_info.get("answer")
        or {}
    )


def evaluate_merged_file(
    merged_path: str,
    target_issue: str,
    iou_thresh: float,
    include_figma: bool,
) -> dict[str, int]:
    metrics = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "img_tp": 0,
        "img_fp": 0,
        "img_fn": 0,
        "img_tn": 0,
        "count_match": 0,
        "total_samples": 0,
        "invalid_pred": 0,
    }

    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)

            if not include_figma and is_figma_sample(sample):
                continue

            gt_payload = get_gt_payload(sample)
            gt_bboxes = extract_bboxes_for_issue(gt_payload, target_issue)
            gt_count = len(gt_bboxes)
            metrics["total_samples"] += 1

            pred_ans = sample.get("pred_ans")
            pred_ans_valid = isinstance(pred_ans, dict)

            # 保持原评测逻辑：非法/缺失输出不能视为合法空预测。
            if not pred_ans_valid:
                metrics["invalid_pred"] += 1
                if gt_count > 0:
                    metrics["img_fn"] += 1
                    metrics["fn"] += gt_count
                else:
                    metrics["img_fp"] += 1
                    metrics["fp"] += 1
                continue

            pred_bboxes = extract_bboxes_for_issue(pred_ans, target_issue)
            pred_count = len(pred_bboxes)

            if gt_count == pred_count:
                metrics["count_match"] += 1

            if gt_count > 0 and pred_count > 0:
                metrics["img_tp"] += 1
            elif gt_count > 0 and pred_count == 0:
                metrics["img_fn"] += 1
            elif gt_count == 0 and pred_count > 0:
                metrics["img_fp"] += 1
            else:
                metrics["img_tn"] += 1

            if gt_count == 0:
                metrics["fp"] += pred_count
                continue

            if pred_count == 0:
                metrics["fn"] += gt_count
                continue

            iou_matrix = np.zeros((gt_count, pred_count), dtype=np.float64)
            for gt_index, gt_box in enumerate(gt_bboxes):
                for pred_index, pred_box in enumerate(pred_bboxes):
                    iou_matrix[gt_index, pred_index] = calculate_iou(
                        gt_box, pred_box
                    )

            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            matched_tp = sum(
                1
                for row_index, col_index in zip(row_indices, col_indices)
                if iou_matrix[row_index, col_index] >= iou_thresh
            )

            metrics["tp"] += matched_tp
            metrics["fp"] += pred_count - matched_tp
            metrics["fn"] += gt_count - matched_tp

    return metrics


def safe_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def build_metrics_summary(metrics: dict[str, int]) -> dict[str, Any]:
    bbox_precision, bbox_recall, bbox_f1 = safe_prf(
        metrics["tp"], metrics["fp"], metrics["fn"]
    )
    image_precision, image_recall, image_f1 = safe_prf(
        metrics["img_tp"], metrics["img_fp"], metrics["img_fn"]
    )

    total_images = (
        metrics["img_tp"]
        + metrics["img_fp"]
        + metrics["img_fn"]
        + metrics["img_tn"]
    )
    image_accuracy = (
        (metrics["img_tp"] + metrics["img_tn"]) / total_images
        if total_images > 0
        else 0.0
    )
    count_accuracy = (
        metrics["count_match"] / metrics["total_samples"]
        if metrics["total_samples"] > 0
        else 0.0
    )

    return {
        "bbox": {
            "precision": bbox_precision,
            "recall": bbox_recall,
            "f1": bbox_f1,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "count_accuracy": count_accuracy,
        },
        "image": {
            "precision": image_precision,
            "recall": image_recall,
            "f1": image_f1,
            "accuracy": image_accuracy,
            "tp": metrics["img_tp"],
            "fp": metrics["img_fp"],
            "fn": metrics["img_fn"],
            "tn": metrics["img_tn"],
        },
        "total_samples": metrics["total_samples"],
        "invalid_pred": metrics["invalid_pred"],
    }


def display_width(text: Any) -> int:
    width = 0
    for char in str(text):
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def pad_display(text: Any, width: int, align: str = "left") -> str:
    text = str(text)
    padding = max(0, width - display_width(text))
    return (" " * padding + text) if align == "right" else (text + " " * padding)


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    widths = [display_width(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], display_width(cell))

    def format_row(row: list[Any]) -> str:
        cells = [
            pad_display(cell, widths[index], "left" if index == 0 else "right")
            for index, cell in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator] + [format_row(r) for r in rows])


def print_evaluation(
    title: str,
    metrics: dict[str, int],
    output_stream: Any,
) -> None:
    summary = build_metrics_summary(metrics)

    headers = [
        "level",
        "prec",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "accuracy",
    ]
    rows = [
        [
            "bbox",
            f"{summary['bbox']['precision']:.4f}",
            f"{summary['bbox']['recall']:.4f}",
            f"{summary['bbox']['f1']:.4f}",
            summary["bbox"]["tp"],
            summary["bbox"]["fp"],
            summary["bbox"]["fn"],
            "",
            f"{summary['bbox']['count_accuracy']:.4f}",
        ],
        [
            "image",
            f"{summary['image']['precision']:.4f}",
            f"{summary['image']['recall']:.4f}",
            f"{summary['image']['f1']:.4f}",
            summary["image"]["tp"],
            summary["image"]["fp"],
            summary["image"]["fn"],
            summary["image"]["tn"],
            f"{summary['image']['accuracy']:.4f}",
        ],
    ]

    print(f"\n>>> {title}", file=output_stream)
    print(format_markdown_table(headers, rows), file=output_stream)
    print(
        f"total_samples={summary['total_samples']}, "
        f"invalid_pred={summary['invalid_pred']}",
        file=output_stream,
    )


def format_bbox_table(summary: dict[str, Any]) -> str:
    bbox = summary["bbox"]

    headers = [
        "prec",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "accuracy",
    ]

    rows = [[
        f"{bbox['precision']:.4f}",
        f"{bbox['recall']:.4f}",
        f"{bbox['f1']:.4f}",
        bbox["tp"],
        bbox["fp"],
        bbox["fn"],
        f"{bbox['count_accuracy']:.4f}",
    ]]

    return format_markdown_table(headers, rows)


def format_image_table(summary: dict[str, Any]) -> str:
    image = summary["image"]

    headers = [
        "prec",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "accuracy",
    ]

    rows = [[
        f"{image['precision']:.4f}",
        f"{image['recall']:.4f}",
        f"{image['f1']:.4f}",
        image["tp"],
        image["fp"],
        image["fn"],
        image["tn"],
        f"{image['accuracy']:.4f}",
    ]]

    return format_markdown_table(headers, rows)


def score_and_write_report(
    merged_path: str,
    report_path: str,
    iou_thresh: float,
    target_issue: str,
) -> dict[str, Any]:
    os.makedirs(
        os.path.dirname(os.path.abspath(report_path)),
        exist_ok=True,
    )

    # 输入测试集已经提前过滤掉 Figma，因此只计算这一套结果。
    metrics = evaluate_merged_file(
        merged_path=merged_path,
        target_issue=target_issue,
        iou_thresh=iou_thresh,
        include_figma=False,
    )

    summary = build_metrics_summary(metrics)

    def write_report(output_stream: Any) -> None:
        print(f"目标问题: {target_issue}", file=output_stream)
        print(f"合并结果: {merged_path}", file=output_stream)
        print(f"IoU 阈值: {iou_thresh}", file=output_stream)

        print("\n>>> Bbox 粒度", file=output_stream)
        print(format_bbox_table(summary), file=output_stream)

        print("\n>>> Image 粒度", file=output_stream)
        print(format_image_table(summary), file=output_stream)

        print(
            f"\ntotal_samples={summary['total_samples']}, "
            f"invalid_pred={summary['invalid_pred']}",
            file=output_stream,
        )

    with open(report_path, "w", encoding="utf-8") as report:
        write_report(report)

    write_report(sys.stdout)

    print(f"\n评测报告已保存至: {report_path}")

    # 返回结果，供五任务汇总使用。
    return summary


def resolve_task_jsonl(
    task_pred_dir: Path,
    filename: str | None = None,
    pattern: str | None = None,
) -> str:
    if filename:
        exact_path = task_pred_dir / filename
        if exact_path.is_file():
            return str(exact_path)

    if pattern:
        candidates = sorted(
            task_pred_dir.glob(pattern),
            key=lambda path: path.stat().st_mtime,
        )

        if candidates:
            if len(candidates) > 1:
                print(
                    f"⚠️ {task_pred_dir} 匹配到多个 JSONL，"
                    f"使用最新文件：{candidates[-1].name}"
                )
            return str(candidates[-1])

    raise FileNotFoundError(
        f"在 {task_pred_dir} 中没有找到对应预测 JSONL"
    )


def run_one_task(
    args: argparse.Namespace,
    task_key: str,
    gt_path: str,
    task_pred_source: str,
    task_output_dir: Path,
) -> dict[str, Any] | None:
    config = TASK_CONFIG[task_key]
    target_issue = config["issue_name"]

    task_output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = task_output_dir / f"{task_key}_merged.jsonl"
    report_path = task_output_dir / f"{task_key}_evaluation.txt"

    print("\n" + "=" * 80)
    print(f">>> 任务：{task_key}")
    print(f">>> 问题类型：{target_issue}")
    print(f">>> GT：{gt_path}")
    print(f">>> 预测来源：{task_pred_source}")
    print(f">>> 输出目录：{task_output_dir}")
    print("=" * 80)

    if args.input_mode == "dir":
        merge_gt_and_qwen3vl_preds(
            gt_path=gt_path,
            pred_dir=task_pred_source,
            output_path=str(merged_path),
            coord_base=args.coord_base,
            target_issue=target_issue,
        )
        score_input_path = str(merged_path)

    elif args.input_mode == "yolo_dir":
        merge_gt_and_yolo_dir_preds(
            gt_path=gt_path,
            pred_dir=task_pred_source,
            output_path=str(merged_path),
            bbox_format=args.yolo_bbox_format,
            target_issue=target_issue,
        )
        score_input_path = str(merged_path)

    elif args.input_mode == "colleague_jsonl":
        score_input_path = task_pred_source

    elif args.input_mode == "swift_jsonl":
        merge_gt_and_swift_jsonl_preds(
            gt_path=gt_path,
            pred_jsonl=task_pred_source,
            output_path=str(merged_path),
            coord_base=args.coord_base,
            target_issue=target_issue,
        )
        score_input_path = str(merged_path)

    else:
        raise ValueError(f"不支持的 input_mode：{args.input_mode}")

    if args.merge_only:
        return None

    return score_and_write_report(
        merged_path=score_input_path,
        report_path=str(report_path),
        iou_thresh=args.iou_thresh,
        target_issue=target_issue,
    )


def write_all_tasks_summary(
    task_summaries: dict[str, dict[str, Any]],
    output_path: str,
) -> None:
    bbox_headers = [
        "task",
        "prec",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "accuracy",
    ]

    image_headers = [
        "task",
        "prec",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "accuracy",
    ]

    bbox_rows = []
    image_rows = []

    for task_key in TASK_CONFIG:
        if task_key not in task_summaries:
            continue

        issue_name = TASK_CONFIG[task_key]["issue_name"]
        summary = task_summaries[task_key]

        bbox = summary["bbox"]
        image = summary["image"]

        bbox_rows.append([
            issue_name,
            f"{bbox['precision']:.4f}",
            f"{bbox['recall']:.4f}",
            f"{bbox['f1']:.4f}",
            bbox["tp"],
            bbox["fp"],
            bbox["fn"],
            f"{bbox['count_accuracy']:.4f}",
        ])

        image_rows.append([
            issue_name,
            f"{image['precision']:.4f}",
            f"{image['recall']:.4f}",
            f"{image['f1']:.4f}",
            image["tp"],
            image["fp"],
            image["fn"],
            image["tn"],
            f"{image['accuracy']:.4f}",
        ])

    summaries = list(task_summaries.values())

    if summaries:
        # 平均值只计算比例指标。
        # TP/FP/FN/TN 是数量，平均没有太强意义，因此平均行留空。
        bbox_rows.append([
            "五类平均",
            f"{np.mean([x['bbox']['precision'] for x in summaries]):.4f}",
            f"{np.mean([x['bbox']['recall'] for x in summaries]):.4f}",
            f"{np.mean([x['bbox']['f1'] for x in summaries]):.4f}",
            "",
            "",
            "",
            f"{np.mean([x['bbox']['count_accuracy'] for x in summaries]):.4f}",
        ])

        image_rows.append([
            "五类平均",
            f"{np.mean([x['image']['precision'] for x in summaries]):.4f}",
            f"{np.mean([x['image']['recall'] for x in summaries]):.4f}",
            f"{np.mean([x['image']['f1'] for x in summaries]):.4f}",
            "",
            "",
            "",
            "",
            f"{np.mean([x['image']['accuracy'] for x in summaries]):.4f}",
        ])

    bbox_table = format_markdown_table(bbox_headers, bbox_rows)
    image_table = format_markdown_table(image_headers, image_rows)

    def write_summary(output_stream: Any) -> None:
        print("\n" + "=" * 80, file=output_stream)
        print("五类任务汇总", file=output_stream)
        print("=" * 80, file=output_stream)

        print("\n>>> Bbox 粒度", file=output_stream)
        print(bbox_table, file=output_stream)

        print("\n>>> Image 粒度", file=output_stream)
        print(image_table, file=output_stream)

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as output_file:
        write_summary(output_file)

    write_summary(sys.stdout)

    def metric_group(name: str) -> dict[str, float]:
        return {
            "precision": float(np.mean([x[name]["precision"] for x in summaries])),
            "recall": float(np.mean([x[name]["recall"] for x in summaries])),
            "f1": float(np.mean([x[name]["f1"] for x in summaries])),
        }

    json_summary = {
        "schema_version": 1,
        "tasks": {
            task_key: {
                "issue_name": TASK_CONFIG[task_key]["issue_name"],
                "bbox": {
                    key: (float(value) if isinstance(value, (float, np.floating)) else int(value))
                    for key, value in summary["bbox"].items()
                },
                "image": {
                    key: (float(value) if isinstance(value, (float, np.floating)) else int(value))
                    for key, value in summary["image"].items()
                },
            }
            for task_key, summary in task_summaries.items()
        },
        "macro": {
            "bbox": metric_group("bbox") if summaries else {},
            "image": metric_group("image") if summaries else {},
        },
    }
    json_output_path = str(Path(output_path).with_suffix(".json"))
    with open(json_output_path, "w", encoding="utf-8") as json_output:
        json.dump(json_summary, json_output, ensure_ascii=False, indent=2)
        json_output.write("\n")

    print(f"\n五任务汇总报告已保存至: {output_path}")
    print(f"五任务机器可读指标已保存至: {json_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并 Qwen3-VL 文字溢出预测、转换坐标并计算指标"
    )
    parser.add_argument(
        "--gt_path",
        default=None,
        help="单任务模式的 GT JSONL",
    )

    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIG.keys()),
        default=DEFAULT_TASK,
        help="单任务模式下评测的任务",
    )

    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="一次运行全部五类任务",
    )

    parser.add_argument(
        "--gt_dir",
        default=None,
        help="all_tasks 模式下，五个 GT JSONL 所在目录",
    )

    parser.add_argument(
        "--pred_root",
        default=None,
        help=(
            "all_tasks 模式下的预测根目录。"
            "目录内部应包含 text_overflow、text_ellipsis、occlusion、"
            "cropping、content_missing 五个子目录"
        ),
    )

    parser.add_argument(
        "--output_root",
        default="./evaluation_runs",
        help="多任务合并结果和评测报告的根目录",
    )

    parser.add_argument(
        "--run_name",
        default=None,
        help="本次运行目录名称；不传则自动使用当前时间",
    )
    parser.add_argument(
        "--input_mode",
        choices=["dir", "yolo_dir", "colleague_jsonl", "swift_jsonl"],
        default="dir",
        help=(
            "dir：Qwen3-VL 逐图片 JSON 目录；"
            "yolo_dir：YOLO 逐图片 JSON 目录；"
            "colleague_jsonl：同事已经带 answer/pred_ans 的 JSONL；"
            "swift_jsonl：swift infer 生成的 response/objects/images JSONL"
        ),
    )

    parser.add_argument(
        "--pred_dir",
        default=None,
        help="input_mode=dir 时使用：单图预测 JSON 所在目录",
    )

    parser.add_argument(
        "--pred_jsonl",
        default=None,
        help=(
            "input_mode=colleague_jsonl 或 swift_jsonl 时使用："
            "对应的预测 JSONL 文件"
        ),
    )

    parser.add_argument(
        "--merged_path",
        default=None,
        help="dir 或 swift_jsonl 模式下生成的合并 JSONL",
    )
    parser.add_argument(
        "--report_path",
        default="./qwen3vl_text_overflow_evaluation.txt",
        help="评测报告路径",
    )
    parser.add_argument(
        "--coord_base",
        type=float,
        default=1000.0,
        help="Qwen3-VL 相对坐标基准，默认 1000",
    )
    parser.add_argument(
        "--yolo_bbox_format",
        choices=["xyxy", "xywh"],
        default="xyxy",
        help=(
            "YOLO JSON 中 bbox_2d 的格式。"
            "xyxy 表示 [x1,y1,x2,y2]；"
            "xywh 表示 [x,y,width,height]。默认 xyxy"
        ),
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.1,
        help="bbox 匹配 IoU 阈值，默认与原脚本一致为 0.1",
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="只合并和转换，不执行算分",
    )
    args = parser.parse_args()

    if args.all_tasks:
        if not args.gt_dir:
            parser.error("--all_tasks 时必须提供 --gt_dir")

        if not args.pred_root:
            parser.error("--all_tasks 时必须提供 --pred_root")

        run_name = (
            args.run_name
            or datetime.now().strftime("run_%Y%m%d-%H%M%S")
        )

        run_dir = Path(args.output_root) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

        print(f">>> 本次五任务输出目录：{run_dir}")
        task_summaries: dict[str, dict[str, Any]] = {}
        for task_key, config in TASK_CONFIG.items():
            gt_path = Path(args.gt_dir) / config["gt_filename"]

            if not gt_path.is_file():
                raise FileNotFoundError(f"GT 不存在：{gt_path}")

            # 每类预测放在自己的子目录，避免相同图片 ID 覆盖。
            task_pred_dir = Path(args.pred_root) / task_key

            if not task_pred_dir.is_dir():
                raise FileNotFoundError(
                    f"任务预测目录不存在：{task_pred_dir}"
                )

            if args.input_mode in {"dir", "yolo_dir"}:
                task_pred_source = str(task_pred_dir)

            elif args.input_mode == "colleague_jsonl":
                task_pred_source = resolve_task_jsonl(
                    task_pred_dir=task_pred_dir,
                    filename=config["colleague_filename"],
                    pattern="*.jsonl",
                )

            elif args.input_mode == "swift_jsonl":
                task_pred_source = resolve_task_jsonl(
                    task_pred_dir=task_pred_dir,
                    pattern=config["swift_pattern"],
                )

            else:
                raise ValueError(
                    f"不支持的 input_mode：{args.input_mode}"
                )

            task_summary = run_one_task(
                args=args,
                task_key=task_key,
                gt_path=str(gt_path),
                task_pred_source=task_pred_source,
                task_output_dir=run_dir / task_key,
            )

            if task_summary is not None:
                task_summaries[task_key] = task_summary

        if not args.merge_only:
            summary_path = run_dir / "all_tasks_evaluation.txt"

            write_all_tasks_summary(
                task_summaries=task_summaries,
                output_path=str(summary_path),
            )

        print(f"\n🎉 五类任务全部完成：{run_dir}")

    else:
        # 保留单任务能力。
        if not args.gt_path:
            parser.error("单任务模式必须提供 --gt_path")

        config = TASK_CONFIG[args.task]
        target_issue = config["issue_name"]

        if args.input_mode in {"dir", "yolo_dir"}:
            if not args.pred_dir:
                parser.error(
                    f"input_mode={args.input_mode} 时必须提供 --pred_dir"
                )
            task_pred_source = args.pred_dir

        else:
            if not args.pred_jsonl:
                parser.error(
                    f"input_mode={args.input_mode} 时必须提供 --pred_jsonl"
                )
            task_pred_source = args.pred_jsonl

        if args.merged_path:
            task_output_dir = Path(args.merged_path).parent
        elif args.report_path:
            task_output_dir = Path(args.report_path).parent
        else:
            task_output_dir = Path("./evaluation_single")

        run_one_task(
            args=args,
            task_key=args.task,
            gt_path=args.gt_path,
            task_pred_source=task_pred_source,
            task_output_dir=task_output_dir,
        )




if __name__ == "__main__":
    """
    python qwen3vl_merge_and_score_fixed.py \
    --gt_path "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/test_ui_text_overflow_wcnt_no_figma.jsonl" \
    --pred_dir "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/qwen3vl-v0-8b_sft_sicheng/v1-20260713-204440/checkpoint-28000/infer_results/image_list_results/" \
    --merged_path "./test_ui_text_overflow_wcnt.jsonl" \
    --report_path "./qwen3vl_text_overflow_evaluation.txt" \
    --coord_base 1000 \
    --iou_thresh 0.1 \
    --input_mode dir


    python qwen3vl_merge_and_score_fixed.py \
    --input_mode colleague_jsonl \
    --gt_path "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/test_ui_text_overflow_wcnt_no_figma.jsonl" \
    --pred_jsonl "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/qwen3vl-v0-8b_sft_sicheng/v1-20260713-204440/checkpoint-28000/infer_results/infer/test_ui_text_overflow_wcnt.jsonl" \
    --report_path "./colleague_evaluation.txt" \
    --iou_thresh 0.1


    python qwen3vl_merge_and_score_fixed.py \
    --input_mode swift_jsonl \
    --gt_path "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/test_ui_text_overflow_wcnt_no_figma.jsonl" \
    --pred_jsonl "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/test_ui_text_overflow_hf_generate_swift_template_441.jsonl" \
    --merged_path "./test_ui_text_overflow_swift_merged.jsonl" \
    --report_path "./qwen3vl_swift_evaluation.txt" \
    --coord_base 1000 \
    --iou_thresh 0.1

    # yolo
    python qwen3vl_merge_and_score_fixed_5tasks.py \
    --input_mode yolo_dir \
    --task cropping \
    --gt_path "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/test_ui_cropping_wcnt_no_figma.jsonl" \
    --pred_dir "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/yolo/jsonl_results_yolo_v5_single_cropping_20260718_174125/" \
    --merged_path "./test_ui_cropping_yolo_merged.jsonl" \
    --report_path "./yolo_26x_imgsz1920_v5_single_evaluation_cropping.txt" \
    --yolo_bbox_format xyxy \
    --iou_thresh 0.1

    python qwen3vl_merge_and_score_fixed_5tasks.py \
    --input_mode yolo_dir \
    --task occlusion \
    --gt_path "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/test_ui_occlusion_wcnt_no_figma.jsonl" \
    --pred_dir "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/yolo/jsonl_results_yolo_v5_single_occlusion_20260718_173737" \
    --merged_path "./test_ui_occlusion_yolo_merged.jsonl" \
    --report_path "./yolo_26x_imgsz1920_v5_single_evaluation_occlusion.txt" \
    --yolo_bbox_format xyxy \
    --iou_thresh 0.1

    python qwen3vl_merge_and_score_fixed_5tasks.py \
    --input_mode yolo_dir \
    --task content_missing \
    --gt_path "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/test_ui_content_missing_wcnt_no_figma.jsonl" \
    --pred_dir "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/yolo/jsonl_results_yolo_v5_single_content_missing_20260719_105338" \
    --merged_path "./test_ui_content_missing_yolo_merged.jsonl" \
    --report_path "./yolo_26x_imgsz1920_v5_single_evaluation_content_missing.txt" \
    --yolo_bbox_format xyxy \
    --iou_thresh 0.1

    python qwen3vl_merge_and_score_fixed_5tasks.py \
    --input_mode yolo_dir \
    --task text_ellipsis \
    --gt_path "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/test_ui_text_ellipsis_wcnt_no_figma.jsonl" \
    --pred_dir "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/yolo/jsonl_results_yolo_v5_single_text_ellipsis_20260719_105943" \
    --merged_path "./test_ui_text_ellipsis_yolo_merged.jsonl" \
    --report_path "./yolo_26x_imgsz1920_v5_single_evaluation_text_ellipsis.txt" \
    --yolo_bbox_format xyxy \
    --iou_thresh 0.1


    python qwen3vl_merge_and_score_fixed_5tasks.py \
    --input_mode yolo_dir \
    --task occlusion \
    --gt_path "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data/test_ui_occlusion_wcnt_no_figma.jsonl" \
    --pred_dir "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/yolo/jsonl_results_yolo_v10_single_occlusion_20260723_121535" \
    --merged_path "./test_ui_occlusion_yolo_merged_v10.jsonl" \
    --report_path "./yolo_26x_imgsz1920_v10_single_evaluation_occlusion.txt" \
    --yolo_bbox_format xyxy \
    --iou_thresh 0.1

    # 五任务运行示例
    # Qwen 单图目录
    python qwen3vl_merge_and_score_fixed.py \
    --all_tasks \
    --input_mode dir \
    --gt_dir "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data" \
    --pred_root "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/output/" \
    --output_root "./evaluation_runs" \
    --coord_base 1000 \
    --iou_thresh 0.1

    python qwen3vl_merge_and_score_fixed_5tasks.py \
    --all_tasks \
    --input_mode swift_jsonl \
    --gt_dir "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data" \
    --pred_root "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/output" \
    --output_root "./evaluation_runs" \
    --coord_base 1000 \
    --iou_thresh 0.1

    # YOLO
    python qwen3vl_merge_and_score_fixed_5tasks.py \
    --all_tasks \
    --input_mode yolo_dir \
    --gt_dir "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data" \
    --pred_root "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_models/locany-3b-ui5-h20-full-v3_h20x4-en/inference-checkpoint-2000-full" \
    --output_root "./evaluation_runs" \
    --yolo_bbox_format xyxy \
    --iou_thresh 0.1
    """


    main()
