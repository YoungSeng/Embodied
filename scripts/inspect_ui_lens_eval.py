#!/usr/bin/env python3
"""Print full UI-Lens evaluation statistics and render converted GT for inspection.

Reads the saved UI5 JSONL, not a second conversion of the raw dataset. Requires
only Pillow. The HTML gallery works offline and uses full-resolution PNG pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_ui_lens_eval import TASKS, normalize_image_size


def read_converted(input_dir: Path):
    records_by_task, sizes = {}, {}
    for task, issue in TASKS.values():
        path = input_dir / f"test_ui_{task}_wcnt_no_figma.jsonl"
        records, seen = [], set()
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    images = row["images"]
                    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
                        raise ValueError("Expected exactly one image path")
                    image_path = Path(images[0]).expanduser()
                    if not image_path.is_absolute():
                        image_path = input_dir / image_path
                    image_path = image_path.resolve(strict=True)
                    if image_path in seen:
                        raise ValueError("Duplicate image within task")
                    seen.add(image_path)
                    if image_path not in sizes:
                        with Image.open(image_path) as image:
                            if image.getexif().get(274, 1) != 1:
                                raise ValueError("Nontrivial EXIF orientation")
                            sizes[image_path] = image.size
                    width, height = sizes[image_path]
                    boxes = row["answer"]["bbox"]
                    if not isinstance(boxes, list):
                        raise ValueError("answer.bbox must be a list")
                    for box in boxes:
                        if not isinstance(box, list) or len(box) != 4 or any(
                            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                            for v in box
                        ):
                            raise ValueError(f"Invalid pixel xyxy box: {box!r}")
                        x1, y1, x2, y2 = box
                        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                            raise ValueError(f"Out-of-bounds pixel xyxy box: {box!r}, size={width, height}")
                    if row["answer"]["types"] != [issue] * len(boxes):
                        raise ValueError("answer.types does not match the task and box count")
                    original = row["extra_info"]["original_infos"]
                    source_boxes = original["box_list"]
                    if not isinstance(source_boxes, list) or any(
                        not isinstance(box, list) or len(box) != 4 or any(
                            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                            for v in box
                        ) for box in source_boxes
                    ):
                        raise ValueError("Missing or invalid original xywh boxes")
                    if any(w <= 0 or h <= 0 or not all(math.isfinite(v) for v in (x + w, y + h))
                           for x, y, w, h in source_boxes):
                        raise ValueError("Original xywh boxes must have positive size and finite corners")
                    # Independent check against the saved raw coordinates, not
                    # the converter's convert_boxes helper.
                    expected = [[x, y, x + w, y + h] for x, y, w, h in source_boxes]
                    metadata = row["extra_info"].get("bbox_conversion", {})
                    if not isinstance(metadata, dict):
                        raise ValueError("bbox_conversion must be an object")
                    policy = metadata.get("boundary_policy", "error")
                    if policy not in ("error", "clip"):
                        raise ValueError(f"Unknown bbox boundary policy: {policy!r}")
                    corrections = []
                    if policy == "clip":
                        for box_index, (raw_box, corners) in enumerate(zip(source_boxes, expected)):
                            bounded = [max(0, min(width, corners[0])), max(0, min(height, corners[1])),
                                       max(0, min(width, corners[2])), max(0, min(height, corners[3]))]
                            if bounded != corners:
                                corrections.append({
                                    "box_index": box_index, "original_xywh": raw_box,
                                    "original_xyxy": corners, "clipped_xyxy": bounded,
                                    "max_clip_pixels": max(abs(a - b) for a, b in zip(corners, bounded)),
                                    "removed_area_ratio": 1 - ((bounded[2] - bounded[0]) / raw_box[2]) * ((bounded[3] - bounded[1]) / raw_box[3]),
                                })
                            expected[box_index] = bounded
                    conversion_ok = boxes == expected and normalize_image_size(original["image_size"]) == [width, height]
                    conversion_ok = conversion_ok and metadata.get("clipped_boxes", []) == corrections
                    records.append({
                        "task": task, "id": row.get("id"), "line": line_number,
                        "jsonl": str(path), "image": str(image_path), "image_size": [width, height],
                        "positive": bool(boxes), "boxes_xyxy": boxes,
                        "source_boxes_xywh": source_boxes, "conversion_ok": conversion_ok,
                        "bbox_boundary_policy": policy, "clipped_boxes": corrections,
                    })
                except (KeyError, ValueError, TypeError, OSError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not records:
            raise ValueError(f"No records in {path}")
        records_by_task[task] = records
    return records_by_task


def summarize(records_by_task):
    def counts(records):
        n = len(records)
        positive = sum(r["positive"] for r in records)
        boxes = sum(len(r["boxes_xyxy"]) for r in records)
        return {"samples": n, "positive": positive, "negative": n - positive,
                "positive_ratio": positive / n if n else 0.0, "boxes": boxes,
                "max_boxes_per_image": max((len(r["boxes_xyxy"]) for r in records), default=0),
                "clipped_images": sum(bool(r["clipped_boxes"]) for r in records),
                "clipped_boxes": sum(len(r["clipped_boxes"]) for r in records),
                "max_clip_pixels": max((c["max_clip_pixels"] for r in records for c in r["clipped_boxes"]), default=0),
                "max_removed_area_ratio": max((c["removed_area_ratio"] for r in records for c in r["clipped_boxes"]), default=0.0),
                "conversion_mismatches": sum(not r["conversion_ok"] for r in records)}

    all_records = [r for records in records_by_task.values() for r in records]
    return {"counting_unit": "image-task annotation record; unique images counted by resolved path",
            "bbox_boundary_policies": sorted({r["bbox_boundary_policy"] for r in all_records}),
            "unique_image_paths": len({r["image"] for r in all_records}),
            "tasks": {task: counts(records) for task, records in records_by_task.items()},
            "total_image_task_records": counts(all_records)}


def print_summary(summary):
    print("\n全量标注统计（不是可视化抽样统计）")
    print(f"{'task':<19} {'samples':>8} {'positive':>9} {'negative':>9} {'pos_%':>8} {'boxes':>8} {'max_box':>8} {'clip_img':>9} {'clip_box':>9} {'mismatch':>9}")
    for task, value in [*summary["tasks"].items(), ("TOTAL(image-task)", summary["total_image_task_records"])]:
        print(f"{task:<19} {value['samples']:>8} {value['positive']:>9} {value['negative']:>9} "
              f"{100 * value['positive_ratio']:>7.2f}% {value['boxes']:>8} "
              f"{value['max_boxes_per_image']:>8} {value['clipped_images']:>9} {value['clipped_boxes']:>9} {value['conversion_mismatches']:>9}")
    print(f"去重图片数（按路径）: {summary['unique_image_paths']}")
    print("positive/negative = 该任务有框/无框；同一图片可在多个任务中计数。")
    print("clip_img/clip_box = 发生边界裁剪的图片任务记录数/框数；裁剪保留每个 GT 框。")
    print("mismatch = 已保存框、尺寸或裁剪记录不符合声明的转换策略；合法 clip 不计为错误。")
    total = summary["total_image_task_records"]
    print(f"边界策略: {', '.join(summary['bbox_boundary_policies'])}; 最大边缘裁剪: {total['max_clip_pixels']} px; "
          f"单框最大面积移除比例: {total['max_removed_area_ratio']:.2%}")


def select_samples(records, limit, seed):
    if limit == 0 or limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    # Show mismatches, then boundary clips, before balancing remaining slots.
    selected = [r for r in records if not r["conversion_ok"]][:limit]
    chosen = {r["line"] for r in selected}
    selected.extend([r for r in records if r.get("clipped_boxes") and r["line"] not in chosen][:limit - len(selected)])
    chosen = {r["line"] for r in selected}
    pools = [[r for r in records if r["positive"] == positive and r["line"] not in chosen]
             for positive in (True, False)]
    for pool in pools:
        rng.shuffle(pool)
    while len(selected) < limit and any(pools):
        for pool in pools:
            if pool and len(selected) < limit:
                selected.append(pool.pop())
    return sorted(selected, key=lambda r: r["line"])


def load_font(size):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_pair(record, output_path):
    with Image.open(record["image"]) as opened:
        original = opened.convert("RGB")
    width, height = original.size
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    stroke = max(2, round(min(width, height) / 350))
    box_font = load_font(max(12, min(24, width // 40)))
    clipped_indices = {c["box_index"] for c in record.get("clipped_boxes", [])}
    for number, box in enumerate(record["boxes_xyxy"], 1):
        # Pixel xyxy edges may equal W/H; draw their visible border at W-1/H-1.
        display_box = [min(box[0], width - 1), min(box[1], height - 1),
                       min(box[2], width - 1), min(box[3], height - 1)]
        clipped = number - 1 in clipped_indices
        draw.rectangle(display_box, outline="#e08016" if clipped else "#00b86b", width=stroke)
        x, y = box[:2]
        draw.text((x + stroke + 1, y + stroke + 1), f"#{number}" + (" clipped" if clipped else ""), fill="#ffffff",
                  font=box_font, stroke_width=1, stroke_fill="#00351e")
    header, gap = 64, 16
    panel_width = max(width, 280)
    pair = Image.new("RGB", (panel_width * 2 + gap, height + header), "#edf1f5")
    pair.paste(original, (0, header))
    pair.paste(overlay, (panel_width + gap, header))
    heading = ImageDraw.Draw(pair)
    font = load_font(18)
    heading.text((10, 8), "Original image", fill="#172334", font=font)
    heading.text((10, 34), f"{width} x {height}", fill="#536277", font=load_font(14))
    heading.text((panel_width + gap + 10, 8), "Converted GT (pixel xyxy)", fill="#172334", font=font)
    status = "POSITIVE" if record["positive"] else "NEGATIVE"
    check = "OK" if record["conversion_ok"] else "MISMATCH"
    if clipped_indices:
        check += f" | CLIPPED={len(clipped_indices)}"
    heading.text((panel_width + gap + 10, 34), f"{status} | boxes={len(record['boxes_xyxy'])} | {check}",
                 fill="#116b45" if record["conversion_ok"] else "#bc2434", font=load_font(14))
    pair.save(output_path)


def build_gallery(summary, manifest):
    esc = lambda value: html.escape(str(value), quote=True)
    stats_rows = []
    for task, value in [*summary["tasks"].items(), ("合计（图片 × 任务）", summary["total_image_task_records"])]:
        cells = [task, value["samples"], value["positive"], value["negative"],
                 f"{value['positive_ratio']:.1%}", value["boxes"], value["clipped_images"],
                 value["clipped_boxes"], value["conversion_mismatches"]]
        stats_rows.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in cells) + "</tr>")
    cards = []
    for sample in manifest:
        polarity = "positive" if sample["positive"] else "negative"
        label = "正样本" if sample["positive"] else "负样本"
        check = "符合声明的转换策略" if sample["conversion_ok"] else "转换或裁剪记录不一致，请检查"
        check += f" · 策略 {sample['bbox_boundary_policy']} · 边界裁剪 {len(sample['clipped_boxes'])} 个框"
        corrections = {c["box_index"]: c for c in sample["clipped_boxes"]}
        coordinates = []
        for i in range(max(len(sample["source_boxes_xywh"]), len(sample["boxes_xyxy"]))):
            raw = sample["source_boxes_xywh"][i] if i < len(sample["source_boxes_xywh"]) else "缺失"
            converted = sample["boxes_xyxy"][i] if i < len(sample["boxes_xyxy"]) else "缺失"
            correction = corrections.get(i)
            change = (f"裁剪边缘最多 {correction['max_clip_pixels']} px；移除面积 {correction['removed_area_ratio']:.2%}"
                      if correction else "无")
            coordinates.append(f"<tr><td>#{i + 1}</td><td>{esc(raw)}</td><td>{esc(converted)}</td><td>{esc(change)}</td></tr>")
        coordinate_table = ("<table><tr><th>框</th><th>原始 xywh</th><th>转换后 xyxy</th><th>边界裁剪</th></tr>"
                            + "".join(coordinates) + "</table>") if coordinates else "<p>该任务没有 GT 框，保留为负样本。</p>"
        cards.append(f'''<article data-task="{esc(sample['task'])}" data-polarity="{polarity}" data-clipped="{'yes' if corrections else 'no'}">
<h2>{esc(sample['task'])} · {label} · {len(sample['boxes_xyxy'])} 个框</h2>
<p class="{'ok' if sample['conversion_ok'] else 'error'}">{check}</p>
<a href="{esc(sample['preview'])}" target="_blank" rel="noopener"><img loading="lazy" src="{esc(sample['preview'])}" alt="原图与转换后的 GT 框对照"></a>
<p class="path">{esc(sample['image'])}</p><p>JSONL 第 {sample['line']} 行 · 原图 {sample['image_size'][0]} × {sample['image_size'][1]}</p>
<details><summary>查看原始与转换后的坐标</summary>{coordinate_table}</details></article>''')
    options = "".join(f'<option value="{esc(task)}">{esc(task)}</option>' for task in summary["tasks"])
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UI-Lens GT 数据检查</title><style>
body{font:15px/1.6 system-ui,sans-serif;background:#f3f5f8;color:#172334;margin:0;padding:28px;max-width:1600px;margin:auto}
h1{margin-bottom:4px}h2{font-size:18px;margin:0}p{margin:8px 0}.muted{color:#536277}table{border-collapse:collapse;width:100%;background:white}
th,td{text-align:left;border-bottom:1px solid #e2e7ee;padding:9px}select{padding:9px;margin:8px;background:white;border:1px solid #c6d0de;border-radius:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,580px),1fr));gap:20px}article{background:white;padding:18px;border-radius:12px;border:1px solid #e2e7ee}
article[hidden]{display:none}img{width:100%;height:auto;display:block}.path{overflow-wrap:anywhere;color:#536277;font-size:12px}.ok{color:#116b45}.error{color:#bc2434;font-weight:700}
details{overflow:auto}summary{cursor:pointer}header{margin-bottom:22px}.table-scroll{overflow:auto}</style><header>
<h1>UI-Lens · 转换后 GT 检查</h1><p>左侧原图，右侧直接绘制已保存 JSONL 中的像素 xyxy 框。绿色为未裁剪框，橙色为边界裁剪框。点击图片查看完整分辨率。</p>
<p class="muted">统计覆盖全部标注；下方图片为抽样检查。正负标签仅针对当前任务，不代表整张图所有缺陷类别。</p>
''' + f'<p>去重图片数（按路径）：<b>{summary["unique_image_paths"]}</b> · 可视化记录：<b>{len(manifest)}</b></p>' + '''
<div class="table-scroll"><table><tr><th>任务</th><th>样本数</th><th>正样本</th><th>负样本</th><th>正样本比例</th><th>框数</th><th>裁剪图片</th><th>裁剪框数</th><th>转换不一致</th></tr>
''' + "".join(stats_rows) + '</table></div><label>任务<select id="task"><option value="">全部</option>' + options + '''</select></label>
<label>正负样本<select id="polarity"><option value="">全部</option><option value="positive">正样本</option><option value="negative">负样本</option></select></label>
<label>边界裁剪<select id="clipping"><option value="">全部</option><option value="yes">有裁剪</option><option value="no">无裁剪</option></select></label>
</header><main class="grid">''' + "".join(cards) + '''</main><script>
function filter(){const task=document.getElementById('task').value, polarity=document.getElementById('polarity').value, clipping=document.getElementById('clipping').value;
document.querySelectorAll('article').forEach(card=>{card.hidden=!!((task&&card.dataset.task!==task)||(polarity&&card.dataset.polarity!==polarity)||(clipping&&card.dataset.clipped!==clipping));});}
document.querySelectorAll('select').forEach(select=>select.addEventListener('change',filter));</script></html>'''


def inspect(input_dir, output_dir=None, samples_per_task=20, seed=42):
    if samples_per_task < 0:
        raise ValueError("samples_per_task must be >= 0 (0 means all)")
    input_dir = Path(input_dir).expanduser().resolve(strict=True)
    records_by_task = read_converted(input_dir)
    summary = summarize(records_by_task)
    print_summary(summary)
    if output_dir is None:
        return summary
    output_dir = Path(output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    for task, records in records_by_task.items():
        task_dir = output_dir / task
        task_dir.mkdir()
        task_seed = seed + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)
        for record in select_samples(records, samples_per_task, task_seed):
            name = f"{record['line']:06d}_{'positive' if record['positive'] else 'negative'}.png"
            render_pair(record, task_dir / name)
            manifest.append({**record, "preview": f"{task}/{name}"})
    summary["visualization"] = {"samples_per_task": samples_per_task, "seed": seed,
                                "rendered": len(manifest), "counts": {
                                    task: sum(m["task"] == task for m in manifest) for task in records_by_task}}
    (output_dir / "dataset_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "samples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "dataset_stats.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", *summary["total_image_task_records"]])
        writer.writeheader()
        for task, value in [*summary["tasks"].items(), ("TOTAL(image-task)", summary["total_image_task_records"])]:
            writer.writerow({"task": task, **value})
    (output_dir / "index.html").write_text(build_gallery(summary, manifest), encoding="utf-8")
    print(f"\n可视化入口: {output_dir / 'index.html'}")
    print(f"统计文件: {output_dir / 'dataset_stats.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Converted UI5 evaluation directory")
    parser.add_argument("--output-dir", type=Path, help="New directory; defaults to INPUT_DIR/inspection")
    parser.add_argument("--samples-per-task", type=int, default=20, help="Balanced positive/negative previews per task; 0 = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-only", action="store_true", help="Print full statistics without writing previews or files")
    args = parser.parse_args()
    if args.samples_per_task < 0:
        parser.error("--samples-per-task must be >= 0")
    if args.stats_only and args.output_dir:
        parser.error("--stats-only does not write files; omit --output-dir")
    summary = inspect(args.input_dir, None if args.stats_only else (args.output_dir or args.input_dir / "inspection"),
                      args.samples_per_task, args.seed)
    return int(summary["total_image_task_records"]["conversion_mismatches"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
