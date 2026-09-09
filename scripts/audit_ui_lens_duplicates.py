#!/usr/bin/env python3
"""Read-only screenshot duplicate audit: SHA256, RGBA pixel hashes, pHash and dHash.

Uses Pillow/numpy on CPU. Similarity is candidate evidence, never a deletion rule.
The report includes all groups/counts even when CSV/HTML presentation is capped.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import html
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageChops, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
# Orthonormal DCT-II: pHash retains the top-left 8x8 coefficients of a 32x32 image.
_k = np.arange(32, dtype=np.float64)[:, None]
_x = np.arange(32, dtype=np.float64)[None, :]
DCT = np.cos(np.pi * (2 * _x + 1) * _k / 64) * np.sqrt(2 / 32)
DCT[0] /= np.sqrt(2)


class Progress:
    def __init__(self, interval):
        self.interval = interval
        self.stop = threading.Event()
        self.stage, self.done, self.total, self.current = "starting", 0, 0, ""
        self.start = time.monotonic()

    def begin(self, stage, total):
        self.stage, self.done, self.total = stage, 0, total
        self.current = ""
        self.start = time.monotonic()
        self.emit()

    def emit(self):
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed else 0
        eta = "0s" if self.done >= self.total else (f"{(self.total-self.done)/rate:.0f}s" if rate else "--")
        print(f"[{self.stage}] {self.done}/{self.total} elapsed={elapsed:.0f}s "
              f"rate={rate:.1f}/s ETA={eta} current={self.current}", flush=True)

    def __enter__(self):
        def heartbeat():
            while not self.stop.wait(self.interval):
                self.emit()
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bits_to_int(values):
    result = 0
    for value in values.flat:
        result = (result << 1) | bool(value)
    return result


def rgb_image(path):
    with Image.open(path) as opened:
        rgba = ImageOps.exif_transpose(opened).convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    return rgba, rgb


def fingerprint(path):
    file_hash = sha256_file(path)
    rgba, rgb = rgb_image(path)
    try:
        width, height = rgba.size
        pixel = hashlib.sha256(f"RGBA:{width}:{height}:".encode() + rgba.tobytes()).hexdigest()
        gray = rgb.convert("L")
        small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        coeff = (DCT @ small @ DCT.T)[:8, :8]
        # Suppress round-off noise on constant/very simple synthetic pages.
        coeff[np.abs(coeff) < 1e-8] = 0
        phash = bits_to_int(coeff > np.median(coeff.flat[1:]))
        horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
        dhash = bits_to_int(horizontal[:, 1:] > horizontal[:, :-1])
        color = np.asarray(rgb.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.int16)
        return {"width": width, "height": height, "file_sha256": file_hash,
                "pixel_sha256": pixel, "phash": f"{phash:016x}", "dhash": f"{dhash:016x}"}, (phash, dhash, color)
    finally:
        rgba.close()
        rgb.close()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def app_value(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value.strip() if isinstance(value, str) else ""


def annotation_index(eval_dir):
    indexed = defaultdict(lambda: {"apps": set(), "task_positive": {}})
    if eval_dir is None:
        return indexed
    from locany_ui5_common import TASK_JSONL
    for task, filename in TASK_JSONL.items():
        for line in (eval_dir / filename).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            image = Path(row["images"][0]).expanduser()
            if not image.is_absolute():
                image = eval_dir / image
            info = indexed[str(image.resolve())]
            name = app_value(row.get("extra_info", {}).get("original_infos", {}).get("app_name"))
            if name:
                info["apps"].add(name)
            if task in info["task_positive"]:
                raise ValueError(f"Repeated image-task annotation: {image}, {task}")
            info["task_positive"][task] = bool(row["answer"]["bbox"])
    return indexed


def groups_by(records, field):
    groups = defaultdict(list)
    for record in records:
        groups[record[field]].append(record["id"])
    return sorted((group for group in groups.values() if len(group) > 1), key=lambda group: (-len(group), group[0]))


def render_report(output, records, summary, exact_groups, previews, max_groups, progress):
    assets = output / "assets"
    assets.mkdir()
    thumbnails = {}

    def thumbnail(index):
        if index not in thumbnails:
            rgba, rgb = rgb_image(Path(records[index]["path"]))
            rgba.close()
            rgb.thumbnail((480, 960), Image.Resampling.LANCZOS)
            rgb.save(assets / f"{index}.jpg", quality=85)
            rgb.close()
            thumbnails[index] = f"assets/{index}.jpg"
        return thumbnails[index]

    def panel(index):
        row = records[index]
        return (f'<figure><a href="{thumbnail(index)}"><img loading="lazy" src="{thumbnail(index)}"></a>'
                f'<figcaption>#{index} · {row["width"]}×{row["height"]} · {html.escape(", ".join(row["apps"]))}'
                f'<br>{html.escape(row["relative_path"])}</figcaption></figure>')

    sections = []
    total = min(len(exact_groups), max_groups) + len(previews)
    progress.begin("render", total)
    for group in exact_groups[:max_groups]:
        sections.append(f'<article><h3>像素完全一致：{len(group)} 张</h3><div class="pair">'
                        + "".join(panel(i) for i in group[:6]) + '</div></article>')
        progress.done += 1
    for pair in previews:
        left, right = pair["left_id"], pair["right_id"]
        thumbnail(left)
        thumbnail(right)
        with Image.open(output / thumbnails[left]) as a, Image.open(output / thumbnails[right]) as b:
            diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB").resize(a.size))
            diff = diff.point(lambda value: min(255, value * 4))
            diff_name = f"assets/diff-{left}-{right}.png"
            diff.save(output / diff_name)
        sections.append(f'<article><h3>近重复候选 #{left} / #{right} · pHash={pair["phash_distance"]}/64 · '
                        f'dHash={pair["dhash_distance"]}/64 · RGB MAE={pair["rgb_mae_32"]}</h3>'
                        f'<p>正负标签不同的共同任务：{html.escape(pair["label_disagreement_tasks"] or "无/无标注")}</p>'
                        '<div class="pair">' + panel(left) + panel(right)
                        + f'<figure><img loading="lazy" src="{diff_name}"><figcaption>缩略图对齐后差异 ×4（含缩放/JPEG误差）</figcaption></figure></div></article>')
        progress.done += 1
    links = " ".join(f'<a href="{name}">{name}</a>' for name in (
        "summary.json", "report.md", "images.jsonl", "pairs.csv", "exact_groups.json", "near_groups.json", "errors.json"))
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>UI-Lens 近重复检查</title>'
                '<style>body{font:16px system-ui;margin:32px;background:#f4f6fa;color:#192335}'
                'article{background:white;padding:18px;margin:20px 0;border-radius:12px}.pair{display:flex;gap:16px;overflow:auto}'
                'figure{margin:0;min-width:180px;max-width:32%}img{max-width:100%;max-height:850px;object-fit:contain}'
                'figcaption{overflow-wrap:anywhere;font-size:13px}a{margin-right:10px}pre{white-space:pre-wrap}</style>'
                '<h1>UI-Lens 截图近重复检查</h1><p>仅供审核，不删除图片。视觉相似可能正是缺陷变体；'
                '相似组为连通分量，不保证组内任意两张都相似。预览按较小 pHash/dHash 距离排序，并非随机抽样。</p>'
                f'<p>{links}</p><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>'
                + "".join(sections) + '</html>')
    (output / "index.html").write_text(document, encoding="utf-8")
    progress.emit()


def audit(image_dir, output, *, eval_dir=None, threshold=8, aspect_tolerance=0.03,
          max_pairs=100000, max_preview_pairs=60, max_preview_groups=20, interval=5):
    if not 0 <= threshold <= 64 or not 0 <= aspect_tolerance <= 1:
        raise ValueError("pHash threshold must be 0..64; aspect tolerance must be 0..1")
    if min(max_pairs, max_preview_pairs, max_preview_groups) < 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError("report limits must be nonnegative and progress interval must be positive")
    image_dir = image_dir.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new report directory")
    paths = sorted({path.resolve() for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS})
    if not paths:
        raise ValueError(f"No supported images under {image_dir}")
    metadata = annotation_index(eval_dir.expanduser().resolve(strict=True) if eval_dir else None)
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "status.json", {"status": "running", "image_dir": str(image_dir)})
    start = time.monotonic()
    records, features, errors = [], [], []
    with Progress(interval) as progress:
        progress.begin("fingerprint", len(paths))
        for path in paths:
            progress.current = str(path)
            try:
                info, feature = fingerprint(path)
                meta = metadata.get(str(path), {"apps": set(), "task_positive": {}})
                try:
                    relative = str(path.relative_to(image_dir))
                except ValueError:
                    relative = str(path)
                records.append({"id": len(records), "path": str(path), "relative_path": relative,
                                **info, "apps": sorted(meta["apps"]), "task_positive": meta["task_positive"]})
                features.append(feature)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            progress.done += 1
        progress.emit()
        json_write(output / "errors.json", errors)
        with (output / "images.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        file_groups, pixel_groups = groups_by(records, "file_sha256"), groups_by(records, "pixel_sha256")
        json_write(output / "exact_groups.json", {"identical_file_groups": file_groups, "identical_pixel_groups": pixel_groups})
        parent = list(range(len(records)))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            parent[root(right)] = root(left)

        # Exact pixel matches are grouped independently from perceptual thresholds.
        for group in pixel_groups:
            for member in group[1:]:
                union(group[0], member)
        total_pairs = len(records) * (len(records) - 1) // 2
        progress.begin("compare", total_pairs)
        progress.current = "all image pairs; exact + perceptual"
        histogram = Counter()
        candidate_count = saved_pairs = exact_pair_count = aspect_rejected = label_disagreements = 0
        preview_heap = []
        columns = ["left_id", "right_id", "left_path", "right_path", "kind", "phash_distance", "dhash_distance",
                   "rgb_mae_32", "aspect_delta", "left_apps", "right_apps", "label_disagreement_tasks"]
        with (output / "pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for i, left in enumerate(records):
                for j in range(i + 1, len(records)):
                    right = records[j]
                    a, b = left["width"] / left["height"], right["width"] / right["height"]
                    aspect_delta = abs(a - b) / max(a, b)
                    is_exact = left["pixel_sha256"] == right["pixel_sha256"]
                    if not is_exact and aspect_delta > aspect_tolerance:
                        aspect_rejected += 1
                        continue
                    ph = (features[i][0] ^ features[j][0]).bit_count()
                    if not is_exact:
                        histogram[ph] += 1
                    if not is_exact and ph > threshold:
                        continue
                    candidate_count += not is_exact
                    exact_pair_count += is_exact
                    union(i, j)
                    dh = (features[i][1] ^ features[j][1]).bit_count()
                    tasks = sorted(t for t in left["task_positive"].keys() & right["task_positive"].keys()
                                   if left["task_positive"][t] != right["task_positive"][t])
                    label_disagreements += bool(tasks)
                    row = dict(zip(columns, [i, j, left["path"], right["path"],
                        "identical_file" if left["file_sha256"] == right["file_sha256"] else ("identical_pixels" if is_exact else "near_candidate"),
                        ph, dh, round(float(np.abs(features[i][2] - features[j][2]).mean()), 3),
                        round(aspect_delta, 6), ";".join(left["apps"]), ";".join(right["apps"]), ";".join(tasks)]))
                    if saved_pairs < max_pairs:
                        writer.writerow(row)
                        saved_pairs += 1
                    if not is_exact and max_preview_pairs:
                        item = (-ph, -dh, -i, -j, row)
                        if len(preview_heap) < max_preview_pairs:
                            heapq.heappush(preview_heap, item)
                        elif item[:4] > preview_heap[0][:4]:
                            heapq.heapreplace(preview_heap, item)
                progress.done += len(records) - i - 1
        progress.emit()
        components = defaultdict(list)
        for index in range(len(records)):
            components[root(index)].append(index)
        near_groups = sorted((group for group in components.values() if len(group) > 1), key=lambda group: (-len(group), group[0]))
        json_write(output / "near_groups.json", {"definition": "connected components of exact-pixel and candidate edges; not cliques", "groups": near_groups})
        thresholds = sorted({0, 2, 4, 6, 8, 10, 12, threshold})
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "image_dir": str(image_dir),
            "eval_dir": str(eval_dir) if eval_dir else None, "discovered_images": len(paths),
            "decoded_images": len(records), "unreadable_images": len(errors),
            "images_with_task_metadata": sum(bool(row["task_positive"]) for row in records),
            "images_by_app": dict(Counter(app for row in records for app in row["apps"])),
            "identical_file_groups": len(file_groups), "identical_file_extra_copies": sum(len(g)-1 for g in file_groups),
            "identical_pixel_groups": len(pixel_groups), "identical_pixel_extra_copies": sum(len(g)-1 for g in pixel_groups),
            "exact_pixel_pairs": exact_pair_count, "near_candidate_pairs_excluding_exact": candidate_count,
            "images_in_exact_or_near_groups": sum(map(len, near_groups)), "connected_groups": len(near_groups),
            "largest_connected_group": max(map(len, near_groups), default=0),
            "candidate_pairs_with_task_label_disagreement": label_disagreements,
            "phash_bits": 64, "phash_threshold": threshold, "aspect_tolerance": aspect_tolerance,
            "pairs_compared": total_pairs, "aspect_rejected_nonexact_pairs": aspect_rejected,
            "near_candidate_counts_by_threshold_excluding_exact": {str(t): sum(n for d, n in histogram.items() if d <= t) for t in thresholds},
            "csv_pairs_written": saved_pairs, "csv_pairs_omitted": candidate_count + exact_pair_count - saved_pairs,
            "csv_order": "image path pair order; capped CSV is not a ranked or random sample",
            "preview_pairs": len(preview_heap), "preview_exact_groups": min(len(pixel_groups), max_preview_groups),
            "hash_and_compare_seconds": round(time.monotonic() - start, 2),
            "status": "complete_with_errors" if errors else "complete",
        }
        json_write(output / "summary.json", summary)
        report = (f"# UI 截图近重复检查\n\n成功解码 {len(records)} / {len(paths)} 张，读取失败 {len(errors)} 张。\n\n"
                  f"- 文件完全一致：{len(file_groups)} 组，额外副本 {summary['identical_file_extra_copies']} 张。\n"
                  f"- EXIF 方向校正后的 RGBA 像素完全一致：{len(pixel_groups)} 组，额外副本 {summary['identical_pixel_extra_copies']} 张（包含文件完全一致）。\n"
                  f"- 排除像素一致后的视觉近似候选：{candidate_count} 对。pHash ≤ {threshold}/64，宽高比相对差 ≤ {aspect_tolerance:.1%}。\n"
                  f"- 精确/近似边构成 {len(near_groups)} 个连通组，涉及 {summary['images_in_exact_or_near_groups']} 张图片。连通组不保证两两相似。\n"
                  f"- 候选/精确对中，共同任务正负标签不同：{label_disagreements} 对。差异可能是合法缺陷变体，不能自动合并。\n"
                  f"- CSV 保存 {saved_pairs} 对，省略 {summary['csv_pairs_omitted']} 对；汇总和分组使用全部比较结果。\n\n"
                  "## 阈值敏感性（已排除精确像素对，保留宽高比限制）\n\n| pHash 最大距离 | 候选对数 |\n|---|---:|\n"
                  + "".join(f"| {t} | {summary['near_candidate_counts_by_threshold_excluding_exact'][str(t)]} |\n" for t in thresholds)
                  + "\n## 如何解读\n\npHash/dHash 衡量整体外观，不判断两个 UI 的语义是否相同。大面积空白、相同导航栏可能导致误报；"
                  "滚动、裁剪、弹窗或局部缺陷也可能造成漏报。阈值 8 是候选筛选起点，未经此数据集人工标定。"
                  "dHash 和 32×32 RGB 平均绝对差只辅助审核，不额外过滤。哈希不使用 GT；标签差异只用于展示。"
                  "报告不修改或删除源图。index.html 为抽样预览，完整成员 ID 在 groups JSON 中，路径在 images.jsonl 中。\n")
        (output / "report.md").write_text(report, encoding="utf-8")
        previews = [item[4] for item in sorted(preview_heap, key=lambda item: tuple(-v for v in item[:4]))]
        render_report(output, records, summary, pixel_groups, previews, max_preview_groups, progress)
        json_write(output / "status.json", {"status": summary["status"], "elapsed_seconds": round(time.monotonic() - start, 2)})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {output / 'index.html'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path, help="UI_lens root; scans single_UIs_cn only")
    source.add_argument("--image-dir", type=Path, help="Explicit screenshot directory, searched recursively")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, help="Optional converted UI5 directory for app/task label annotations")
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--aspect-tolerance", type=float, default=0.03)
    parser.add_argument("--max-pairs", type=int, default=100000, help="CSV row cap; all comparisons still count in groups/summary")
    parser.add_argument("--max-preview-pairs", type=int, default=60)
    parser.add_argument("--max-preview-groups", type=int, default=20)
    parser.add_argument("--progress-interval-seconds", type=float, default=5)
    args = parser.parse_args()
    try:
        summary = audit(args.image_dir or args.dataset_root / "single_UIs_cn", args.output_dir,
                        eval_dir=args.eval_dir, threshold=args.phash_threshold, aspect_tolerance=args.aspect_tolerance,
                        max_pairs=args.max_pairs, max_preview_pairs=args.max_preview_pairs,
                        max_preview_groups=args.max_preview_groups, interval=args.progress_interval_seconds)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    if summary["unreadable_images"]:
        parser.exit(2, "Report generated with unreadable images; see errors.json.\n")


if __name__ == "__main__":
    main()
