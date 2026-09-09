#!/usr/bin/env python3
"""Per-task screenshot duplicates, annotation disagreement and quality candidates.

Reuses images.jsonl from audit_ui_lens_duplicates.py, but recomputes within-task
edges so cross-task bridges and truncated CSVs cannot distort task counts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_ui_lens_duplicates import Progress, json_write, rgb_image, sha256_file
from locany_ui5_common import TASK_JSONL, TASK_ISSUE_NAMES

QUALITY_FLAGS = ("unreadable", "low_resolution", "near_blank", "low_detail")
CASE_NAMES = {
    "polarity_conflict": "像素相同、同任务正负标签不同",
    "bbox_conflict": "像素相同、同任务 bbox 不同",
    "invalid_bbox": "GT 框格式或边界待检查",
    "unreadable": "读取失败", "low_resolution": "低分辨率候选",
    "near_blank": "近纯色/空白候选", "low_detail": "低细节/疑似模糊候选",
    "exact_duplicate": "任务内部像素完全重复", "near_candidate": "任务内部近重复候选对",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def quality_metrics(path, min_short_side=256, blank_std=5.0, low_detail_variance=20.0):
    rgba, rgb = rgb_image(path)
    rgba.close()
    try:
        width, height = rgb.size
        gray = rgb.convert("L")
        gray.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
        std = float(pixels.std())
        if min(pixels.shape) >= 3:
            laplacian = (pixels[:-2, 1:-1] + pixels[2:, 1:-1] + pixels[1:-1, :-2]
                         + pixels[1:-1, 2:] - 4 * pixels[1:-1, 1:-1])
            variance = float(laplacian.var())
        else:
            variance = 0.0
        flags = []
        if min(width, height) < min_short_side:
            flags.append("low_resolution")
        if std < blank_std:
            flags.append("near_blank")
        elif variance < low_detail_variance:
            flags.append("low_detail")
        return {"width": width, "height": height, "gray_std": round(std, 6),
                "laplacian_variance": round(variance, 6), "flags": flags}
    finally:
        rgb.close()


def bbox_valid(boxes, size):
    if not isinstance(boxes, list):
        return False
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in box
        ):
            return False
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            return False
        if size and (x2 > size[0] or y2 > size[1]):
            return False
    return True


def bbox_signature(boxes):
    # Same task + identical pixels: compare box sets independent of list order.
    return tuple(sorted(tuple(float(v) for v in box) for box in boxes))


def classify_task(task, rows, fingerprints, quality, cross_tasks, threshold, aspect_tolerance,
                  pair_writer=None, max_pairs=100000, progress=None):
    exact = defaultdict(list)
    file_groups = defaultdict(list)
    for row in rows:
        path = row["path"]
        if path in fingerprints:
            exact[fingerprints[path]["pixel_sha256"]].append(row)
            file_groups[fingerprints[path]["file_sha256"]].append(row)
    exact = [group for group in exact.values() if len(group) > 1]
    file_groups = [group for group in file_groups.values() if len(group) > 1]
    cases = defaultdict(list)
    exact_paths, polarity_paths, bbox_paths = set(), set(), set()
    exact_ids = {}
    polarity_groups = bbox_groups = 0
    for index, group in enumerate(exact):
        paths = [row["path"] for row in group]
        exact_paths.update(paths)
        exact_ids.update({path: index for path in paths})
        cases["exact_duplicate"].append(paths)
        if len({row["positive"] for row in group}) > 1:
            polarity_groups += 1
            polarity_paths.update(paths)
            cases["polarity_conflict"].append(paths)
        if all(row["valid_bbox"] for row in group) and len({bbox_signature(row["bbox"]) for row in group}) > 1:
            bbox_groups += 1
            bbox_paths.update(paths)
            cases["bbox_conflict"].append(paths)
    eligible = [row for row in rows if row["path"] in fingerprints]
    near_paths, near_label_paths = set(), set()
    near_pairs = near_label_pairs = saved_pairs = 0
    edges = defaultdict(set)
    if progress:
        progress.begin(f"compare-{task}", len(eligible) * (len(eligible) - 1) // 2)
    for i, left in enumerate(eligible):
        a = fingerprints[left["path"]]
        for right in eligible[i+1:]:
            b = fingerprints[right["path"]]
            if a["pixel_sha256"] == b["pixel_sha256"]:
                continue
            ar, br = a["width"] / a["height"], b["width"] / b["height"]
            if abs(ar-br) / max(ar, br) > aspect_tolerance:
                continue
            distance = (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count()
            if distance > threshold:
                continue
            p, q = left["path"], right["path"]
            near_pairs += 1
            near_paths.update((p, q))
            edges[p].add(q)
            edges[q].add(p)
            different = left["positive"] != right["positive"]
            near_label_pairs += different
            if different:
                near_label_paths.update((p, q))
            if len(cases["near_candidate"]) < 100:
                cases["near_candidate"].append([p, q])
            if pair_writer is not None and saved_pairs < max_pairs:
                pair_writer.writerow({"task": task, "left_path": p, "right_path": q,
                                      "phash_distance": distance, "polarity_disagreement": different})
                saved_pairs += 1
        if progress:
            progress.done += len(eligible) - i - 1
    if progress:
        progress.emit()
    # Connected components only of this task's candidate edges (exact excluded).
    groups, visited = [], set()
    for path in sorted(edges):
        if path in visited:
            continue
        group, pending = [], [path]
        visited.add(path)
        while pending:
            item = pending.pop()
            group.append(item)
            for neighbour in edges[item] - visited:
                visited.add(neighbour)
                pending.append(neighbour)
        groups.append(sorted(group))
    counts = Counter()
    quality_paths = set()
    invalid_paths = {row["path"] for row in rows if not row["valid_bbox"]}
    for row in rows:
        path = row["path"]
        for flag in quality[path]["flags"]:
            counts[flag] += 1
            quality_paths.add(path)
            cases[flag].append([path])
        if path in invalid_paths:
            cases["invalid_bbox"].append([path])
    needs_review = exact_paths | near_paths | quality_paths | invalid_paths
    stats = {
        "task": task, "issue": TASK_ISSUE_NAMES[task], "rows": len(rows),
        "positive": sum(row["positive"] for row in rows), "negative": sum(not row["positive"] for row in rows),
        "duplicate_assessed_rows": len(eligible), "duplicate_unassessed_rows": len(rows)-len(eligible),
        "file_duplicate_groups": len(file_groups), "file_extra_copies": sum(len(g)-1 for g in file_groups),
        "pixel_duplicate_groups": len(exact), "pixel_duplicate_images": len(exact_paths),
        "pixel_extra_copies": sum(len(g)-1 for g in exact),
        "pixel_duplicate_positive": sum(row["positive"] for row in rows if row["path"] in exact_paths),
        "pixel_duplicate_negative": sum(not row["positive"] for row in rows if row["path"] in exact_paths),
        "pixel_extra_copy_rate": sum(len(g)-1 for g in exact) / len(rows) if rows else 0.0,
        "near_pairs_excluding_exact": near_pairs, "near_images_excluding_exact_pairs": len(near_paths),
        "near_connected_groups": len(groups), "exact_or_near_images": len(exact_paths | near_paths),
        "exact_polarity_conflict_groups": polarity_groups, "exact_polarity_conflict_images": len(polarity_paths),
        "exact_bbox_disagreement_groups": bbox_groups, "exact_bbox_disagreement_images": len(bbox_paths),
        "near_polarity_disagreement_pairs": near_label_pairs, "near_polarity_disagreement_images": len(near_label_paths),
        "cross_task_shared_content_images": sum(len(cross_tasks.get(fingerprints[row['path']]['pixel_sha256'], set())) > 1
                                                for row in eligible),
        **{flag: counts[flag] for flag in QUALITY_FLAGS},
        "quality_candidate_images": len(quality_paths),
        "quality_candidate_positive": sum(row["positive"] for row in rows if row["path"] in quality_paths),
        "quality_candidate_negative": sum(not row["positive"] for row in rows if row["path"] in quality_paths),
        "invalid_bbox_rows": len(invalid_paths), "clipped_gt_rows": sum(row["clipped_gt"] for row in rows),
        "review_union_images": len(needs_review), "near_csv_pairs_written": saved_pairs,
        "near_csv_pairs_omitted": near_pairs-saved_pairs,
    }
    details = [{"task": task, "path": row["path"], "positive": row["positive"],
                "bbox_count": len(row["bbox"]) if isinstance(row["bbox"], list) else None,
                "exact_group": exact_ids.get(row["path"]), "near_candidate": row["path"] in near_paths,
                "exact_polarity_conflict": row["path"] in polarity_paths,
                "exact_bbox_disagreement": row["path"] in bbox_paths,
                "invalid_bbox": not row["valid_bbox"], "clipped_gt": row["clipped_gt"],
                "quality_flags": quality[row["path"]]["flags"],
                "duplicate_assessed": row["path"] in fingerprints} for row in rows]
    return stats, {"exact_pixel_groups": [[row["path"] for row in group] for group in exact],
                   "near_components_excluding_exact_edges": groups}, dict(cases), details


def render(output, tasks, stats, cases, quality, max_previews, progress):
    assets = output / "assets"
    assets.mkdir()
    cards = []
    progress.begin("render", len(tasks))
    for task, rows in tasks.items():
        indexed = {row["path"]: row for row in rows}
        selected = []
        # Round-robin by issue type so a large duplicate cluster cannot hide
        # unreadable/blank/annotation-conflict examples in the preview budget.
        index = 0
        while len(selected) < max_previews:
            added = False
            for kind in CASE_NAMES:
                pool = cases[task].get(kind, [])
                if index < len(pool) and len(selected) < max_previews:
                    selected.append((kind, pool[index]))
                    added = True
            if not added:
                break
            index += 1
        cards.append(f'<section><h2>{html.escape(TASK_ISSUE_NAMES[task])}</h2><p>任务样本 {len(rows)} 张；'
                     f'任务内部精确重复额外副本 {stats[task]["pixel_extra_copies"]} 张；'
                     f'近重复涉及 {stats[task]["near_images_excluding_exact_pairs"]} 张；'
                     f'质量候选 {stats[task]["quality_candidate_images"]} 张。</p>')
        for kind, members in selected:
            figures = []
            for path in members[:3]:
                row = indexed[path]
                name = hashlib.sha256(f"{task}:{path}".encode()).hexdigest()[:20] + ".jpg"
                destination = assets / name
                image_html = "<p>图片无法读取</p>"
                if "unreadable" not in quality[path]["flags"]:
                    if not destination.exists():
                        rgba, rgb = rgb_image(Path(path))
                        rgba.close()
                        width, height = rgb.size
                        rgb.thumbnail((480, 960), Image.Resampling.LANCZOS)
                        if row["valid_bbox"]:
                            draw = ImageDraw.Draw(rgb)
                            for x1, y1, x2, y2 in row["bbox"]:
                                scaled = [min(rgb.width-1, round(x*rgb.width/width)) for x in (x1, x2)]
                                vertical = [min(rgb.height-1, round(y*rgb.height/height)) for y in (y1, y2)]
                                draw.rectangle((scaled[0], vertical[0], scaled[1], vertical[1]), outline="#ff6b00", width=2)
                        rgb.save(destination, quality=88)
                        rgb.close()
                    image_html = f'<a href="assets/{name}"><img loading="lazy" src="assets/{name}"></a>'
                figures.append(f'<figure>{image_html}<figcaption>{"正" if row["positive"] else "负"}样本 · '
                               f'{html.escape(path)}<br>{html.escape(json.dumps(quality[path], ensure_ascii=False))}</figcaption></figure>')
            cards.append(f'<article><h3>{CASE_NAMES[kind]}（{len(members)} 张，最多预览 3 张）</h3><div class="images">'
                         + "".join(figures) + '</div></article>')
        cards.append('</section>')
        progress.done += 1
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>UI5 分类数据审查</title>'
                '<style>body{font:16px system-ui;margin:30px;background:#f4f6fa;color:#192335}'
                'article{background:white;border-radius:10px;padding:16px;margin:16px 0}.images{display:flex;gap:18px;overflow:auto}'
                'figure{margin:0;min-width:180px;max-width:32%}img{max-width:100%;max-height:850px;object-fit:contain}'
                'figcaption{font-size:12px;overflow-wrap:anywhere}</style><h1>五任务重复与质量候选</h1>'
                '<p>橙色框为此任务原图 GT。质量标记是规则候选，空白/低细节也可能是合法缺陷。'
                '跨任务复用不计为任务内部重复。近重复预览只展示实际满足阈值的图对。</p>'
                '<p><a href="report.md">中文汇总</a> · <a href="task_summary.csv">分类统计 CSV</a> · '
                '<a href="summary.json">完整统计 JSON</a> · <a href="task_samples.jsonl">逐任务样本</a> · '
                '<a href="groups.json">完整分组</a> · <a href="near_pairs.csv">近重复对</a> · '
                '<a href="image_quality.jsonl">逐图质量</a></p>' + "".join(cards) + '</html>')
    (output / "index.html").write_text(document, encoding="utf-8")
    progress.emit()


def analyze(audit_dir, eval_dir, output, *, threshold=None, aspect_tolerance=None, min_short_side=256,
            blank_std=5.0, low_detail_variance=20.0, max_pairs=100000, max_previews=12, interval=5):
    audit_dir, eval_dir = audit_dir.expanduser().resolve(strict=True), eval_dir.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory")
    audit_summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    threshold = audit_summary["phash_threshold"] if threshold is None else threshold
    aspect_tolerance = audit_summary["aspect_tolerance"] if aspect_tolerance is None else aspect_tolerance
    if not 0 <= threshold <= 64 or not 0 <= aspect_tolerance <= 1 or min_short_side <= 0:
        raise ValueError("Invalid pHash threshold, aspect tolerance or minimum size")
    if any(not math.isfinite(x) or x < 0 for x in (blank_std, low_detail_variance)) or interval <= 0 or not math.isfinite(interval):
        raise ValueError("Quality thresholds must be finite/nonnegative; interval must be finite/positive")
    if max_pairs < 0 or max_previews < 0:
        raise ValueError("Report limits cannot be negative")
    cached = {str(Path(row["path"]).expanduser().resolve()): row for row in read_jsonl(audit_dir / "images.jsonl")}
    tasks, sources, all_paths = {}, {}, set()
    for task, filename in TASK_JSONL.items():
        source = eval_dir / filename
        sources[task] = {"path": str(source), "sha256": sha256_file(source)}
        rows, seen = [], set()
        for line_number, row in enumerate(read_jsonl(source), 1):
            images = row.get("images")
            if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
                raise ValueError(f"{source}:{line_number}: expected one image path")
            path = Path(images[0]).expanduser()
            if not path.is_absolute():
                path = eval_dir / path
            path = str(path.resolve())
            if path in seen:
                raise ValueError(f"Repeated image path within task: {task}, {path}")
            seen.add(path)
            if not isinstance(row.get("answer"), dict) or not isinstance(row["answer"].get("bbox"), list):
                raise ValueError(f"{source}:{line_number}: expected converted answer.bbox list")
            boxes = row["answer"]["bbox"]
            rows.append({"path": path, "bbox": boxes, "positive": bool(boxes),
                         "clipped_gt": bool(row.get("extra_info", {}).get("bbox_conversion", {}).get("clipped_boxes", []))})
            all_paths.add(path)
        tasks[task] = rows
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "status.json", {"status": "running"})
    fingerprints, quality, cache_issues = {}, {}, []
    with Progress(interval) as progress:
        progress.begin("quality-and-cache-check", len(all_paths))
        for path in sorted(all_paths):
            progress.current = path
            try:
                digest = sha256_file(Path(path))
                item = quality_metrics(Path(path), min_short_side, blank_std, low_detail_variance)
                if path not in cached:
                    cache_issues.append({"path": path, "reason": "not_in_previous_audit"})
                elif digest != cached[path]["file_sha256"]:
                    cache_issues.append({"path": path, "reason": "image_changed_since_previous_audit"})
                else:
                    fingerprints[path] = cached[path]
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                item = {"flags": ["unreadable"], "error": f"{type(exc).__name__}: {exc}"}
                cache_issues.append({"path": path, "reason": "unreadable"})
            quality[path] = item
            progress.done += 1
        progress.emit()
        cross_tasks = defaultdict(set)
        for task, rows in tasks.items():
            for row in rows:
                q = quality[row["path"]]
                size = (q["width"], q["height"]) if "width" in q else None
                row["valid_bbox"] = bbox_valid(row["bbox"], size)
                if row["path"] in fingerprints:
                    cross_tasks[fingerprints[row["path"]]["pixel_sha256"]].add(task)
        task_stats, groups, all_cases, details = {}, {}, {}, []
        with (output / "near_pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["task", "left_path", "right_path", "phash_distance", "polarity_disagreement"])
            writer.writeheader()
            for task, rows in tasks.items():
                stats, task_groups, cases, samples = classify_task(
                    task, rows, fingerprints, quality, cross_tasks, threshold, aspect_tolerance,
                    writer, max_pairs, progress)
                task_stats[task], groups[task], all_cases[task] = stats, task_groups, cases
                details.extend(samples)
        summary = {"status": "complete_with_unassessed_images" if cache_issues else "complete",
                   "audit_dir": str(audit_dir), "eval_dir": str(eval_dir), "task_sources": sources,
                   "previous_images_sha256": sha256_file(audit_dir / "images.jsonl"),
                   "settings": {"phash_threshold": threshold, "aspect_tolerance": aspect_tolerance,
                                "min_short_side": min_short_side, "blank_gray_std_threshold": blank_std,
                                "low_detail_laplacian_variance_threshold": low_detail_variance,
                                "quality_downscale_long_side": 1024, "max_near_csv_pairs_per_task": max_pairs},
                   "unique_image_paths": len(all_paths), "image_task_records": len(details),
                   "duplicate_unassessed_unique_paths": len(cache_issues), "cache_issues": cache_issues,
                   "quality_candidate_unique_paths": sum(bool(v["flags"]) for v in quality.values()),
                   "tasks": task_stats}
        json_write(output / "summary.json", summary)
        json_write(output / "groups.json", groups)
        with (output / "task_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(task_stats.values()))))
            writer.writeheader()
            writer.writerows(task_stats.values())
        for filename, records in (("task_samples.jsonl", details),
                                  ("image_quality.jsonl", [{"path": path, **q} for path, q in quality.items()])):
            (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        text = "# 五任务截图重复与质量检查\n\n重复均在各任务内部计算，跨任务内容复用单列。质量结果为规则候选。\n\n"
        text += "## 任务内部重复\n\n|任务|样本|正|负|像素重复组|涉及图片|额外副本|近重复对|近重复涉及图片|重复/近重复并集|未完成重复检查|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        for s in task_stats.values():
            text += "|" + "|".join(str(s[k]) for k in ("issue", "rows", "positive", "negative", "pixel_duplicate_groups", "pixel_duplicate_images", "pixel_extra_copies", "near_pairs_excluding_exact", "near_images_excluding_exact_pairs", "exact_or_near_images", "duplicate_unassessed_rows")) + "|\n"
        text += "\n## 图像质量候选\n\n|任务|读取失败|低分辨率|近纯色/空白|低细节/疑似模糊|质量候选并集|其中正样本|其中负样本|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
        for s in task_stats.values():
            text += "|" + "|".join(str(s[k]) for k in ("issue", *QUALITY_FLAGS, "quality_candidate_images", "quality_candidate_positive", "quality_candidate_negative")) + "|\n"
        text += "\n## 标注复查与跨任务复用\n\n|任务|同像素正负冲突组|同像素 bbox 不同组|近重复正负不同对|GT 框异常样本|已审计 GT clip 样本|跨任务共享内容图片|\n|---|---:|---:|---:|---:|---:|---:|\n"
        for s in task_stats.values():
            text += "|" + "|".join(str(s[k]) for k in ("issue", "exact_polarity_conflict_groups", "exact_bbox_disagreement_groups", "near_polarity_disagreement_pairs", "invalid_bbox_rows", "clipped_gt_rows", "cross_task_shared_content_images")) + "|\n"
        text += ("\n## 口径\n\n额外副本为每个精确重复组的 n−1，不指定保留哪张。近重复涉及图片可能同时属于精确重复组，"
                 "请看并集，不能直接相加。近重复边只在当前任务内重算，不借助其他任务串联；CSV 截断不影响计数。"
                 "跨任务共享内容不属于任务内部重复。每任务分母是该任务 JSONL 的样本数，五类相加是图像×任务记录数。\n\n"
                 "同像素正负冲突属于同任务需优先复查的标注问题；bbox 不同也可能是标注范围差异，两项会重叠。"
                 "近重复正负不同可能是合法缺陷变体。GT clip 表示之前转换器已审计的越界修正，不自动计为质量问题。\n\n"
                 f"质量规则：原图短边 < {min_short_side}px；缩至长边最多 1024（不放大）后的灰度标准差 < {blank_std} 标近纯色；"
                 f"其余图片四邻域 Laplacian 方差 < {low_detail_variance} 标低细节。质量项可与低分辨率重叠，按并集统计。"
                 "这些阈值未在 UI-Lens 标定；简洁页面和内容未展示正样本也可能触发，不能直接判为坏样本。\n\n"
                 f"哈希阈值 {threshold}/64，宽高比相对差 ≤ {aspect_tolerance:.1%}。哈希记录缺失、文件变化或不可读的图片"
                 "不使用旧哈希计算重复，详见 summary.json 的 cache_issues。页面标注和缩略图仅供审核；脚本不删改数据。\n")
        (output / "report.md").write_text(text, encoding="utf-8")
        render(output, tasks, task_stats, all_cases, quality, max_previews, progress)
        json_write(output / "status.json", {"status": summary["status"]})
    print(text, flush=True)
    print(f"Report: {output / 'index.html'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phash-threshold", type=int, default=None)
    parser.add_argument("--aspect-tolerance", type=float, default=None)
    parser.add_argument("--min-short-side", type=int, default=256)
    parser.add_argument("--blank-std", type=float, default=5.0)
    parser.add_argument("--low-detail-variance", type=float, default=20.0)
    parser.add_argument("--max-pairs-per-task", type=int, default=100000)
    parser.add_argument("--max-previews-per-task", type=int, default=12)
    parser.add_argument("--progress-interval-seconds", type=float, default=5)
    args = parser.parse_args()
    try:
        summary = analyze(args.audit_dir, args.eval_dir, args.output_dir, threshold=args.phash_threshold,
                          aspect_tolerance=args.aspect_tolerance, min_short_side=args.min_short_side,
                          blank_std=args.blank_std, low_detail_variance=args.low_detail_variance,
                          max_pairs=args.max_pairs_per_task, max_previews=args.max_previews_per_task,
                          interval=args.progress_interval_seconds)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    if summary["duplicate_unassessed_unique_paths"]:
        parser.exit(2, "Report generated with incomplete duplicate coverage; inspect cache_issues in summary.json.\n")


if __name__ == "__main__":
    main()
