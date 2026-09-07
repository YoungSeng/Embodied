"""Use the production parser, coordinate merge/NMS and threshold-aware matcher."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from run_ui5_train_rollout_worker import load_module, score_prediction, task_config
from ui5_lossless_tiling import merge_tile_predictions


def formal_modules():
    from scripts import inference_ui_defect_locany as parser
    scorer = load_module(ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py", "ui5_grpo_formal_scorer")
    return parser, scorer


def reward_from_boxes(scorer, gt, predictions, *, invalid=False, image_size=(1, 1), config=None):
    from eaglevl.train.ui5_grpo_core import GRPOConfig
    config = config or GRPOConfig()
    score = score_prediction(scorer, gt, predictions,
                             "parse_error" if invalid else "defect" if predictions else "ok",
                             config.iou_threshold, image_size)
    if invalid:
        box = image = iou = 0.0
        reward = -1.0
    elif not predictions and not gt:
        box = image = iou = reward = 1.0
    elif not predictions or not gt:
        box = image = iou = reward = 0.0
    else:
        tp, fp, fn = (score[key] for key in ("TP_box", "FP_box", "FN_box"))
        box = 2.0 * tp / (2 * tp + fp + fn)
        image = 1.0
        # Includes assigned pairs below IoU=0.1, unlike discrete TP_box.
        iou = 2.0 * sum(pair["iou"] for pair in score["matched_pairs"]) / (len(predictions) + len(gt))
        reward = config.reward_box * box + config.reward_image * image + config.reward_iou * iou
    return dict(reward=reward, F_box=box, C_img=image, S_iou=iou,
                invalid=invalid, pred_global=predictions, **score)


def score_trajectory(group, outputs, parser, scorer, config):
    if len(outputs) != len(group["views"]):
        raise ValueError("a trajectory must retain every crop, including invalid ones")
    width, height = group["width"], group["height"]
    pending, raw = [], []
    invalid = False
    task = task_config(parser, group["task"])
    for view, output in zip(group["views"], outputs):
        crop = view["crop_xyxy"]
        parsed = parser.parse_locateanything_answer(output["raw_output"])
        detections, local = parser.build_yolo_compatible_detections(
            parsed, task, crop[2] - crop[0], crop[3] - crop[1], None)
        invalid |= parsed.status == "parse_error"
        raw.append(dict(crop_id=view["crop_id"], raw_output=output["raw_output"],
                        parse_status=parsed.status, parse_warnings=parsed.warnings,
                        pred_local=local, crop_xyxy=crop))
        for detection in detections:
            pending.append(dict(bbox=detection["bbox_2d"], tile_bbox=crop,
                                label=detection["label"], class_id=detection["class_id"],
                                score=1.0, confidence=detection.get("confidence"),
                                source_tile_index=view["crop_index"]))
    merged = merge_tile_predictions(pending, image_size=(width, height), iou_threshold=config.nms_iou)
    predictions = []
    for item in merged:
        box = [int(round(x)) for x in item["bbox"]]
        box = [max(0, min(width, box[0])), max(0, min(height, box[1])),
               max(0, min(width, box[2])), max(0, min(height, box[3]))]
        if box[0] < box[2] and box[1] < box[3]:
            predictions.append(box)
    return dict(crop_outputs=raw, **reward_from_boxes(
        scorer, group["gt_global"], predictions, invalid=invalid,
        image_size=(width, height), config=config))
