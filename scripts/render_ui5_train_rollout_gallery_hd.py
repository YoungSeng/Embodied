#!/usr/bin/env python3
"""Rebuild a UI5 rollout snapshot as a high-resolution, explainable HTML gallery.

The renderer reads ``complete8.jsonl`` from one immutable snapshot, copies each
selected source image without resizing/re-encoding, and draws boxes as inline
SVG.  It is CPU-only and does not read the live rollout streams.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


MODELS = ("m31", "crop")
ROLLOUT_IDS = (0, 1, 2, 3)
BUCKETS = ("easy", "medium", "hard", "FP", "FN")
TASK_LABELS = {
    "occlusion": "元素遮挡",
    "cropping": "元素裁切",
    "text_overflow": "文字溢出",
    "text_ellipsis": "异常省略",
    "content_missing": "内容缺失",
}
ERROR_LABELS = {
    "TN": "正确判无缺陷",
    "FP_ONLY": "无标注框但模型多报",
    "FN_NO_PRED": "有标注框但模型未输出框",
    "EXACT_TP": "全部标注框命中且无多余框",
    "LOC_WRONG": "输出了框，但位置均未达到 IoU 阈值",
    "PARTIAL_MISS": "部分标注框漏检",
    "PARTIAL_EXTRA": "标注框均命中，但存在多余预测框",
    "PARTIAL_BOTH": "同时存在漏检和多余预测框",
    "PARSE_ERROR": "模型输出无法解析",
    "RUNTIME_ERROR": "推理运行错误",
    "INCOMPLETE": "结果不完整",
}
PRESENCE_LABELS = {
    "TP": "TP：有标注框，模型也输出了框（只说明有/无，不代表位置正确）",
    "TN": "TN：无标注框，模型也没有输出框",
    "FP": "FP：无标注框，但模型输出了框",
    "FN": "FN：有标注框，但模型没有输出框",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument(
        "--index-name",
        default="index_hd.html",
        help="HTML filename inside SNAPSHOT_ROOT/visualizations.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also refresh the published snapshot summary/manifest (off by default).",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: Any) -> str:
    text = str(value)
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in text)
    return cleaned or "unnamed"


def as_boxes(value: Any) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return boxes
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(number) for number in item)
        except (TypeError, ValueError):
            continue
        boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    return boxes


def all_rollouts(record: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    rollout_map = record.get("rollouts")
    if not isinstance(rollout_map, Mapping):
        return
    for model in MODELS:
        rows = rollout_map.get(model, [])
        if isinstance(rows, Sequence):
            for row in rows:
                if isinstance(row, Mapping):
                    yield row


def record_buckets(record: Mapping[str, Any]) -> set[str]:
    result = {str(record.get("difficulty"))}
    for rollout in all_rollouts(record):
        if rollout.get("image_confusion") == "FP" or int(rollout.get("FP_box") or 0) > 0:
            result.add("FP")
        if rollout.get("image_confusion") == "FN" or int(rollout.get("FN_box") or 0) > 0:
            result.add("FN")
    return result & set(BUCKETS)


def severity(record: Mapping[str, Any], bucket: str) -> tuple[Any, ...]:
    m31 = int(record.get("m31_correct_count") or 0)
    crop = int(record.get("crop_correct_count") or 0)
    if bucket == "medium":
        stable_disagreement = int({m31, crop} == {0, 4})
        return (-stable_disagreement, -abs(m31 - crop), int(record.get("total_correct_count") or 0))
    if bucket in {"FP", "FN"}:
        field = f"{bucket}_box"
        total = sum(int(row.get(field) or 0) for row in all_rollouts(record))
        presence = sum(row.get("image_confusion") == bucket for row in all_rollouts(record))
        return (-presence, -total)
    return (str(record.get("task")), str(record.get("record_id")))


def select_records(
    records: Sequence[Mapping[str, Any]], limit: int
) -> tuple[list[Mapping[str, Any]], dict[str, list[str]]]:
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    memberships: dict[str, list[str]] = {}
    used_images: dict[str, set[str]] = {bucket: set() for bucket in BUCKETS}
    for bucket in BUCKETS:
        candidates = [row for row in records if bucket in record_buckets(row)]
        candidates.sort(key=lambda row: (*severity(row, bucket), str(row.get("record_id"))))
        for row in candidates:
            if len(used_images[bucket]) >= limit:
                break
            image_id = str(row.get("source_image_id") or row.get("record_id"))
            if image_id in used_images[bucket]:
                continue
            used_images[bucket].add(image_id)
            record_id = str(row["record_id"])
            selected_by_id[record_id] = row
            memberships.setdefault(record_id, []).append(bucket)
    ordering = {bucket: index for index, bucket in enumerate(BUCKETS)}
    selected = sorted(
        selected_by_id.values(),
        key=lambda row: (
            min(ordering[bucket] for bucket in memberships[str(row["record_id"])]),
            str(row.get("task")),
            str(row["record_id"]),
        ),
    )
    return selected, memberships


def copy_source_image(
    record: Mapping[str, Any], bundle_root: Path, visual_root: Path
) -> tuple[str, int, int]:
    source = (bundle_root / str(record["image_relpath"])).resolve(strict=True)
    with Image.open(source) as opened:
        width, height = opened.size
        opened.verify()
    suffix = source.suffix.lower() if source.suffix else ".img"
    destination = (
        visual_root
        / "assets_hd"
        / "original"
        / f"{safe_name(record.get('source_image_id', record['record_id']))}{suffix}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or source.stat().st_size != destination.stat().st_size:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    relative = destination.relative_to(visual_root).as_posix()
    return relative, int(width), int(height)


def fmt_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def svg_overlay(
    record: Mapping[str, Any],
    rollout: Mapping[str, Any],
    width: int,
    height: int,
) -> str:
    gt = as_boxes(rollout.get("gt_global") or record.get("gt_global") or [])
    pred = as_boxes(rollout.get("pred_global") or [])
    pairs = rollout.get("matched_pairs") or []
    matched_gt = {
        int(pair["gt_index"])
        for pair in pairs
        if isinstance(pair, Mapping) and pair.get("is_tp") is True and "gt_index" in pair
    }
    matched_pred = {
        int(pair["pred_index"])
        for pair in pairs
        if isinstance(pair, Mapping) and pair.get("is_tp") is True and "pred_index" in pair
    }
    font_size = max(16.0, min(width, height) / 38.0)
    label_dy = max(18.0, font_size * 1.05)
    elements: list[str] = []

    crops = [
        as_boxes([item.get("crop_xyxy")])[0]
        for item in rollout.get("crop_outputs", [])
        if isinstance(item, Mapping) and as_boxes([item.get("crop_xyxy")])
    ]
    full_image_crop = bool(
        len(crops) == 1
        and abs(crops[0][0]) < 1
        and abs(crops[0][1]) < 1
        and abs(crops[0][2] - width) < 1
        and abs(crops[0][3] - height) < 1
    )
    if not full_image_crop:
        for index, (x1, y1, x2, y2) in enumerate(crops):
            elements.append(
                f"<rect class='box crop-box' x='{x1:g}' y='{y1:g}' "
                f"width='{x2-x1:g}' height='{y2-y1:g}'/>"
            )
            elements.append(
                f"<text class='box-label crop-label' x='{x1 + 4:g}' "
                f"y='{max(label_dy, y1 + label_dy):g}' font-size='{font_size:g}'>"
                f"C{index}</text>"
            )

    for index, (x1, y1, x2, y2) in enumerate(gt):
        missed = index not in matched_gt
        css_class = "gt-box fn-box" if missed else "gt-box"
        label = f"G{index} FN" if missed else f"G{index} GT"
        elements.append(
            f"<rect class='box {css_class}' x='{x1:g}' y='{y1:g}' "
            f"width='{x2-x1:g}' height='{y2-y1:g}'/>"
        )
        elements.append(
            f"<text class='box-label {'fn-label' if missed else 'gt-label'}' "
            f"x='{x1 + 4:g}' y='{max(label_dy, y1 + label_dy):g}' "
            f"font-size='{font_size:g}'>{label}</text>"
        )
    for index, (x1, y1, x2, y2) in enumerate(pred):
        matched = index in matched_pred
        css_class = "tp-box" if matched else "fp-box"
        label = f"P{index} TP" if matched else f"P{index} FP"
        y = min(height - 4, max(label_dy, y1 + 2 * label_dy))
        elements.append(
            f"<rect class='box {css_class}' x='{x1:g}' y='{y1:g}' "
            f"width='{x2-x1:g}' height='{y2-y1:g}'/>"
        )
        elements.append(
            f"<text class='box-label {'tp-label' if matched else 'fp-label'}' "
            f"x='{x1 + 4:g}' y='{y:g}' font-size='{font_size:g}'>{label}</text>"
        )
    return (
        f"<svg class='overlay' viewBox='0 0 {width} {height}' "
        f"preserveAspectRatio='none' aria-hidden='true'>{''.join(elements)}</svg>"
    )


def json_pre(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def matching_table(rollout: Mapping[str, Any]) -> str:
    pairs = rollout.get("matched_pairs") or []
    if not pairs:
        return "<p class='muted'>没有 Hungarian 配对记录。</p>"
    rows = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            continue
        passed = pair.get("is_tp") is True
        rows.append(
            "<tr>"
            f"<td>G{html.escape(str(pair.get('gt_index', '—')))}</td>"
            f"<td>P{html.escape(str(pair.get('pred_index', '—')))}</td>"
            f"<td>{fmt_number(pair.get('iou'))}</td>"
            f"<td><span class='badge {'ok' if passed else 'bad'}'>{'达标' if passed else '未达标'}</span></td>"
            f"<td>{fmt_number(pair.get('center_distance_px'), 1)}</td>"
            f"<td>{fmt_number(pair.get('pred_gt_area_ratio'), 2)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>GT</th><th>预测</th><th>IoU</th><th>匹配结果</th>"
        "<th>中心距(px)</th><th>预测/GT面积比</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def inference_path(record: Mapping[str, Any], model: str, rollout: Mapping[str, Any], width: int, height: int) -> str:
    if model == "m31":
        return "全图模型：原图直接输入"
    crops = [
        as_boxes([item.get("crop_xyxy")])[0]
        for item in rollout.get("crop_outputs", [])
        if isinstance(item, Mapping) and as_boxes([item.get("crop_xyxy")])
    ]
    full = bool(
        len(crops) == 1
        and abs(crops[0][0]) < 1
        and abs(crops[0][1]) < 1
        and abs(crops[0][2] - width) < 1
        and abs(crops[0][3] - height) < 1
    )
    if record.get("task") == "content_missing" and full:
        return f"Crop 路径：全图退化（1 tile，[0,0,{width},{height}]）"
    return f"Crop 路径：{len(crops)} 个裁剪区域，结果映射回全图后合并"


def panel_html(
    record: Mapping[str, Any],
    model: str,
    rollout: Mapping[str, Any] | None,
    image_relpath: str,
    width: int,
    height: int,
    iou_threshold: float,
) -> str:
    if rollout is None:
        return "<article class='panel missing'><h4>结果缺失</h4></article>"
    exact = rollout.get("exact_correct") is True
    image_confusion = str(rollout.get("image_confusion") or "—")
    error_type = str(rollout.get("error_type") or "INCOMPLETE")
    if ":" in error_type:
        error_type = error_type.split(":", 1)[1]
    gt = as_boxes(rollout.get("gt_global") or record.get("gt_global") or [])
    pred = as_boxes(rollout.get("pred_global") or [])
    parse_status = rollout.get("parse_status")
    route = inference_path(record, model, rollout, width, height)
    panel_classes = "panel correct" if exact else "panel incorrect"
    return f"""
<article class="{panel_classes}">
  <header class="panel-head">
    <div><strong>{model.upper()} · rollout {html.escape(str(rollout.get('rollout_id', '—')))}</strong>
      <span class="muted">seed={html.escape(str(rollout.get('seed', '—')))}</span></div>
    <span class="badge {'ok' if exact else 'bad'}">{'完全正确' if exact else '未完全正确'}</span>
  </header>
  <div class="route">{html.escape(route)}</div>
  <div class="metrics">
    <span class="badge presence-{html.escape(image_confusion.lower())}">存在性 {html.escape(image_confusion)}</span>
    <span>BBox TP={int(rollout.get('TP_box') or 0)}</span>
    <span>FP={int(rollout.get('FP_box') or 0)}</span>
    <span>FN={int(rollout.get('FN_box') or 0)}</span>
    <span>GT={len(gt)}</span><span>预测={len(pred)}</span>
  </div>
  <p class="meaning"><strong>{html.escape(PRESENCE_LABELS.get(image_confusion, '存在性结果缺失'))}</strong><br>
    定位结论：{html.escape(ERROR_LABELS.get(error_type, error_type))}</p>
  <button class="figure" type="button" onclick="openZoom(this)" aria-label="点击放大">
    <img loading="lazy" decoding="async" src="{html.escape(image_relpath)}"
      width="{width}" height="{height}" alt="{html.escape(str(record.get('record_id')))}">
    {svg_overlay(record, rollout, width, height)}
    <span class="zoom-hint">点击查看原分辨率标注</span>
  </button>
  <div class="panel-links"><a href="{html.escape(image_relpath)}" target="_blank" rel="noopener">打开无标注原图</a>
    <span>IoU 阈值={iou_threshold:g} · parse={html.escape(str(parse_status))}</span></div>
  <details><summary>框坐标与匹配明细</summary>
    {matching_table(rollout)}
    <pre>GT boxes: {html.escape(json.dumps(gt, ensure_ascii=False))}\nPred boxes: {html.escape(json.dumps(pred, ensure_ascii=False))}</pre>
  </details>
  <details><summary>模型原始输出</summary><pre>{json_pre(rollout.get('raw_output'))}</pre></details>
  {('<details><summary>Crop 子图输出与坐标变换</summary><pre>' + json_pre(rollout.get('crop_outputs')) + '</pre></details>') if model == 'crop' else ''}
</article>"""


def pattern_explanation(m31: int, crop: int) -> tuple[str, str]:
    if m31 == 0 and crop == 4:
        return "m31_0_crop_4", "M31 四次均未完全正确；Crop 四次均完全正确：稳定的 Crop 优势样本。"
    if m31 == 4 and crop == 0:
        return "m31_4_crop_0", "M31 四次均完全正确；Crop 四次均未完全正确：稳定的 M31 优势样本。"
    if 0 < m31 < 4 or 0 < crop < 4:
        return "within_model_mixed", "至少一个模型四次结果有对有错，存在模型内采样差异，可作为 GRPO 候选。"
    return "other", "两个数字分别表示各模型四次采样中“完全正确”的次数。"


def card_html(
    record: Mapping[str, Any],
    memberships: Sequence[str],
    image_relpath: str,
    width: int,
    height: int,
    iou_threshold: float,
) -> str:
    record_id = str(record["record_id"])
    task = str(record.get("task"))
    difficulty = str(record.get("difficulty"))
    m31 = int(record.get("m31_correct_count") or 0)
    crop = int(record.get("crop_correct_count") or 0)
    total = int(record.get("total_correct_count") or 0)
    pattern, explanation = pattern_explanation(m31, crop)
    rollout_map = record.get("rollouts") if isinstance(record.get("rollouts"), Mapping) else {}
    model_sections = []
    for model in MODELS:
        by_id = {
            int(row.get("rollout_id")): row
            for row in rollout_map.get(model, [])
            if isinstance(row, Mapping) and isinstance(row.get("rollout_id"), int)
        }
        panels = "".join(
            panel_html(
                record,
                model,
                by_id.get(rollout_id),
                image_relpath,
                width,
                height,
                iou_threshold,
            )
            for rollout_id in ROLLOUT_IDS
        )
        model_sections.append(
            f"<section class='model-block'><h3>{model.upper()}：4 次独立采样</h3>"
            f"<div class='panel-grid'>{panels}</div></section>"
        )
    bucket_badges = "".join(
        f"<span class='badge bucket'>{html.escape(bucket)}</span>" for bucket in memberships
    )
    grpo = []
    if record.get("grpo_ready_m31"):
        grpo.append("M31 GRPO-ready")
    if record.get("grpo_ready_crop"):
        grpo.append("Crop GRPO-ready")
    grpo_html = "".join(f"<span class='badge grpo'>{html.escape(item)}</span>" for item in grpo)
    return f"""
<article class="sample-card" id="{safe_name(record_id)}"
 data-difficulty="{html.escape(difficulty)}" data-task="{html.escape(task)}"
 data-buckets="{html.escape(' '.join(memberships))}" data-pattern="{html.escape(pattern)}"
 data-search="{html.escape(record_id + ' ' + str(record.get('source_image_id')))}">
 <header class="sample-head">
   <div><h2>{html.escape(TASK_LABELS.get(task, task))} <small>{html.escape(task)}</small></h2>
     <code>{html.escape(record_id)}</code> · image=<code>{html.escape(str(record.get('source_image_id')))}</code></div>
   <div>{bucket_badges}{grpo_html}</div>
 </header>
 <div class="scoreline"><strong>M31={m31}/4</strong><strong>Crop={crop}/4</strong>
   <strong>合计={total}/8 · {html.escape(difficulty)}</strong></div>
 <p class="pattern-note">{html.escape(explanation)} 这里的 x/4 是四次 rollout 中 exact_correct 的次数，不是框数或 IoU。</p>
 {''.join(model_sections)}
</article>"""


def refresh_snapshot_metadata(snapshot_root: Path, gallery: Mapping[str, Any]) -> None:
    summary_path = snapshot_root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["visualizations"] = dict(gallery)
        atomic_json(summary_path, summary)
    manifest_path = snapshot_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = []
        for path in sorted(snapshot_root.rglob("*")):
            if not path.is_file() or path.name in {"manifest.json", "_SUCCESS"}:
                continue
            row_count = None
            if path.suffix == ".jsonl":
                with path.open("rb") as handle:
                    row_count = sum(1 for line in handle if line.strip())
            files.append(
                {
                    "path": path.relative_to(snapshot_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "jsonl_records": row_count,
                }
            )
        manifest["files"] = files
        atomic_json(manifest_path, manifest)


def build_document(cards: str, counts: Mapping[str, int], tasks: Sequence[str], threshold: float) -> str:
    options = "".join(
        f"<option value='{html.escape(task)}'>{html.escape(TASK_LABELS.get(task, task))}</option>"
        for task in tasks
    )
    count_text = " · ".join(f"{bucket}={counts.get(bucket, 0)}" for bucket in BUCKETS)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UI5 八路 Rollout 高清审计</title>
<style>
:root{{--bg:#f4f6fa;--card:#fff;--ink:#172033;--muted:#64748b;--line:#dbe3ee;
--gt:#16a34a;--tp:#2563eb;--fp:#dc2626;--fn:#f97316;--crop:#eab308}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Noto Sans SC",sans-serif}}
.page{{max-width:1920px;margin:auto;padding:20px}} h1{{margin:.1em 0}} h2{{margin:0;font-size:21px}} h2 small{{font-weight:400;color:var(--muted);font-size:13px}}
h3{{margin:20px 0 9px}} code{{font-size:12px}} .muted{{color:var(--muted);font-size:12px}}
.guide,.toolbar,.sample-card{{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 10px #1e293b0a}}
.guide{{padding:16px;margin:14px 0}} .guide-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.guide-item{{padding:11px;border:1px solid var(--line);border-radius:9px;background:#fbfdff}}
.legend{{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}} .legend span{{display:inline-flex;align-items:center;gap:6px}}
.swatch{{width:18px;height:12px;border:3px solid;border-radius:2px}} .swatch.gt{{border-color:var(--gt)}} .swatch.tp{{border-color:var(--tp)}}
.swatch.fp{{border-color:var(--fp)}} .swatch.fn{{border-color:var(--fn);border-style:dashed}} .swatch.crop{{border-color:var(--crop);border-style:dashed}}
.toolbar{{position:sticky;top:0;z-index:20;padding:10px;margin:14px 0;display:flex;gap:9px;flex-wrap:wrap}}
.toolbar input,.toolbar select{{border:1px solid #cbd5e1;border-radius:7px;padding:7px 9px;background:#fff}}
.toolbar input{{min-width:260px}} .sample-card{{padding:16px;margin:18px 0}}
.sample-head{{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;flex-wrap:wrap}}
.badge{{display:inline-block;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px;background:#eef2f7}}
.badge.ok{{background:#dcfce7;color:#166534}} .badge.bad{{background:#fee2e2;color:#991b1b}}
.badge.bucket{{background:#e0e7ff;color:#3730a3}} .badge.grpo{{background:#fae8ff;color:#86198f}}
.scoreline{{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0;font-size:17px}} .pattern-note{{background:#eff6ff;border-left:4px solid #3b82f6;padding:8px 11px}}
.model-block{{border-top:1px solid var(--line);margin-top:14px}} .panel-grid{{display:grid;grid-template-columns:repeat(4,minmax(280px,1fr));gap:11px}}
.panel{{min-width:0;border:2px solid #cbd5e1;border-radius:9px;padding:9px;background:#fff}}
.panel.correct{{border-color:#86efac}} .panel.incorrect{{border-color:#fca5a5;background:#fffafa}}
.panel-head{{display:flex;justify-content:space-between;gap:8px;align-items:center}} .route{{font-size:12px;color:#334155;margin:6px 0}}
.metrics{{display:flex;flex-wrap:wrap;gap:5px 9px;font-size:12px}} .meaning{{font-size:12px;min-height:56px}}
.figure{{appearance:none;display:block;position:relative;width:100%;padding:0;border:0;background:#111;cursor:zoom-in;overflow:hidden}}
.figure img{{display:block;width:100%;height:auto}} .overlay{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
.box{{fill:none;stroke-width:3;vector-effect:non-scaling-stroke}} .gt-box{{stroke:var(--gt)}} .tp-box{{stroke:var(--tp)}}
.fp-box{{stroke:var(--fp);stroke-dasharray:9 5}} .fn-box{{stroke:var(--fn);stroke-width:5;stroke-dasharray:12 6}}
.crop-box{{stroke:var(--crop);stroke-width:3;stroke-dasharray:14 7}}
.box-label{{font-weight:800;paint-order:stroke;stroke:#fff;stroke-width:5px;stroke-linejoin:round}}
.gt-label{{fill:var(--gt)}} .tp-label{{fill:var(--tp)}} .fp-label{{fill:var(--fp)}} .fn-label{{fill:var(--fn)}} .crop-label{{fill:#a16207}}
.zoom-hint{{position:absolute;right:6px;bottom:6px;background:#000b;color:#fff;border-radius:5px;padding:3px 7px;font-size:11px}}
.panel-links{{display:flex;justify-content:space-between;gap:8px;font-size:11px;margin-top:5px}} details{{margin-top:6px;font-size:12px}}
summary{{cursor:pointer;color:#334155}} pre{{white-space:pre-wrap;word-break:break-word;max-height:360px;overflow:auto;background:#f8fafc;padding:8px;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:11px}} th,td{{border:1px solid #dbe3ee;padding:4px;text-align:left}}
dialog{{width:98vw;height:96vh;max-width:none;border:0;border-radius:10px;padding:8px}} dialog::backdrop{{background:#000b}}
.dialog-bar{{position:sticky;top:0;z-index:2;display:flex;justify-content:space-between;background:#fff;padding:7px;border-bottom:1px solid var(--line)}}
#zoomHost{{overflow:auto;height:calc(96vh - 55px);background:#1f2937;padding:12px}} .zoom-stage{{position:relative;width:max-content}}
.zoom-stage img{{display:block;width:auto;max-width:none;height:auto}} .zoom-stage .overlay{{position:absolute;inset:0;width:100%;height:100%}} .zoom-stage .zoom-hint{{display:none}}
.hidden{{display:none!important}}
@media(max-width:1300px){{.panel-grid{{grid-template-columns:repeat(2,minmax(260px,1fr))}}}}
@media(max-width:680px){{.page{{padding:8px}}.panel-grid{{grid-template-columns:1fr}}.toolbar{{position:static}}}}
</style></head><body><main class="page">
<h1>UI5 八路 Rollout 高清审计</h1>
<p>每个样本：M31 独立采样4次 + Crop独立采样4次。{html.escape(count_text)}</p>
<section class="guide">
 <h2>先看这里：数字和颜色分别表示什么</h2>
 <div class="legend"><span><i class="swatch gt"></i>绿色：GT标注框</span><span><i class="swatch tp"></i>蓝色：IoU≥{threshold:g} 的预测框（BBox TP）</span>
 <span><i class="swatch fp"></i>红色虚线：多余或位置不达标的预测框（BBox FP）</span><span><i class="swatch fn"></i>橙色虚线：没有被命中的GT（BBox FN）</span>
 <span><i class="swatch crop"></i>黄色虚线：Crop模型实际输入区域</span></div>
 <div class="guide-grid">
  <div class="guide-item"><strong>M31=0/4、Crop=4/4</strong><br>表示同一样本各推理4次，M31四次都未“完全正确”，Crop四次都完全正确。它不是框数，也不是IoU。</div>
  <div class="guide-item"><strong>图像存在性 TP/TN/FP/FN</strong><br>只判断“是否有缺陷框”：TP=GT有且预测有；TN=两者都无；FP=GT无但预测有；FN=GT有但预测无。位置全错仍可能是存在性TP。</div>
  <div class="guide-item"><strong>BBox TP/FP/FN</strong><br>按IoU≥{threshold:g}匹配：蓝框是TP，红框是FP，橙色GT是FN。LOC_WRONG会同时产生BBox FP和FN。</div>
  <div class="guide-item"><strong>Easy / Medium / Hard</strong><br>8/8完全正确=easy；1–7/8=medium；0/8=hard。“完全正确”要求所有GT均命中且无多余框，或GT和预测均为空。</div>
 </div>
</section>
<section class="toolbar">
 <input id="search" placeholder="搜索 record_id / image_id">
 <select id="difficulty"><option value="">全部难度</option><option>easy</option><option>medium</option><option>hard</option></select>
 <select id="task"><option value="">全部任务</option>{options}</select>
 <select id="bucket"><option value="">全部类别</option><option value="FP">任意 FP（存在性或框级）</option><option value="FN">任意 FN（存在性或框级）</option></select>
 <select id="pattern"><option value="">全部模型关系</option><option value="m31_0_crop_4">M31 0/4 → Crop 4/4</option><option value="m31_4_crop_0">M31 4/4 → Crop 0/4</option><option value="within_model_mixed">模型内有对有错</option></select>
</section>
<div id="cards">{cards}</div>
</main>
<dialog id="zoomDialog"><div class="dialog-bar"><strong>原分辨率标注视图</strong><button onclick="document.getElementById('zoomDialog').close()">关闭</button></div><div id="zoomHost"></div></dialog>
<script>
const cards=[...document.querySelectorAll('.sample-card')];
function applyFilters(){{
 const q=document.getElementById('search').value.trim().toLowerCase();
 const d=document.getElementById('difficulty').value;
 const t=document.getElementById('task').value;
 const b=document.getElementById('bucket').value;
 const p=document.getElementById('pattern').value;
 cards.forEach(card=>{{
  const show=(!q||card.dataset.search.toLowerCase().includes(q))&&(!d||card.dataset.difficulty===d)&&(!t||card.dataset.task===t)&&(!b||card.dataset.buckets.split(' ').includes(b))&&(!p||card.dataset.pattern===p);
  card.classList.toggle('hidden',!show);
 }});
}}
document.querySelectorAll('.toolbar input,.toolbar select').forEach(node=>node.addEventListener('input',applyFilters));
function openZoom(button){{
 const host=document.getElementById('zoomHost'); host.innerHTML='';
 const stage=document.createElement('div'); stage.className='zoom-stage';
 const image=button.querySelector('img').cloneNode(true); image.removeAttribute('loading');
 const overlay=button.querySelector('svg').cloneNode(true); stage.append(image,overlay); host.append(stage);
 document.getElementById('zoomDialog').showModal();
}}
document.getElementById('zoomDialog').addEventListener('click',e=>{{if(e.target===e.currentTarget)e.currentTarget.close();}});
</script></body></html>"""


def render(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_root = args.snapshot_root.expanduser().resolve(strict=True)
    bundle_root = args.bundle_root.expanduser().resolve(strict=True)
    if args.max_per_category <= 0:
        raise ValueError("--max-per-category must be positive")
    records_path = snapshot_root / "complete8.jsonl"
    if not records_path.is_file():
        records_path = snapshot_root / "samples.jsonl"
    records = read_jsonl(records_path)
    selected, memberships = select_records(records, args.max_per_category)
    visual_root = snapshot_root / "visualizations"
    cards: list[str] = []
    failures: list[dict[str, str]] = []
    copied_images: set[str] = set()
    for record in selected:
        try:
            image_relpath, width, height = copy_source_image(record, bundle_root, visual_root)
            copied_images.add(image_relpath)
            cards.append(
                card_html(
                    record,
                    memberships[str(record["record_id"])],
                    image_relpath,
                    width,
                    height,
                    args.iou_threshold,
                )
            )
        except Exception as exc:
            failures.append(
                {"record_id": str(record.get("record_id")), "error": f"{type(exc).__name__}: {exc}"}
            )
    counts = Counter(bucket for values in memberships.values() for bucket in values)
    document = build_document(
        "".join(cards), counts, sorted({str(row.get("task")) for row in selected}), args.iou_threshold
    )
    visual_root.mkdir(parents=True, exist_ok=True)
    index_path = visual_root / args.index_name
    atomic_text(index_path, document)
    result = {
        "renderer": "high_resolution_svg_v1",
        "index": f"visualizations/{args.index_name}",
        "source_records": str(records_path),
        "rendered_cards": len(cards),
        "copied_original_images": len(copied_images),
        "failures": failures,
        "counts": {bucket: counts[bucket] for bucket in BUCKETS},
        "max_per_category": args.max_per_category,
        "iou_threshold_displayed": args.iou_threshold,
        "image_assets_are_original_bytes": True,
        "boxes_are_inline_svg": True,
    }
    atomic_json(visual_root / "gallery_hd_summary.json", result)
    if args.refresh_manifest:
        refresh_snapshot_metadata(snapshot_root, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    render(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
