#!/usr/bin/env python3
"""Render task-aware UI5 crop failures from immutable on-disk detections.

This tool never imports or runs PP-OCRv5/OmniParser.  It reads the selected
candidate geometry and merged detector JSONL, then creates one four-panel audit
image per failed GT bbox.  Existing manual labels are preserved across resume.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_CONFIG = "TA_CTX015_H050"
ROOT_CAUSES = (
    "text_detector_miss",
    "icon_detector_miss",
    "gt_spans_multiple_components",
    "context_too_small",
    "task_linking_rule_mismatch",
    "annotation_suspect",
    "other",
)
FAILURE_TYPES = ("partial_intersection", "uncovered")
COMPENSATION_BUCKETS = (
    "small_0_16px",
    "medium_17_64px",
    "large_over_64px",
)
EXPECTED_FAILURES_BY_TASK = {
    "ui_occlusion": {"partial_intersection": 27, "uncovered": 19},
    "ui_cropping": {"partial_intersection": 23, "uncovered": 0},
    "ui_text_overflow": {"partial_intersection": 2, "uncovered": 1},
    "ui_text_ellipsis": {"partial_intersection": 35, "uncovered": 0},
    "ui_content_missing": {"partial_intersection": 0, "uncovered": 0},
}
EXPECTED_FAILURES_BY_DENSITY = {
    "sparse": {"partial_intersection": 35, "uncovered": 3},
    "medium": {"partial_intersection": 49, "uncovered": 17},
    "dense": {"partial_intersection": 3, "uncovered": 0},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-audit-name", default="crop_audit_v3")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-failures", type=int, default=107)
    parser.add_argument("--expected-partial", type=int, default=87)
    parser.add_argument("--expected-uncovered", type=int, default=20)
    parser.add_argument("--panel-max-width", type=int, default=700)
    parser.add_argument("--panel-max-height", type=int, default=1400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Preserve images and rebuild galleries/diagnosis after manual JSONL edits",
    )
    parser.add_argument(
        "--require-manual-review",
        action="store_true",
        help="Fail unless all failures have a valid manual_root_cause",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def index_unique(
    rows: Sequence[Mapping[str, Any]], key: str, source: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise ValueError(f"duplicate {key} in {source}: {value}")
        indexed[value] = row
    return indexed


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    image.save(temporary, format="PNG", compress_level=1)
    os.replace(temporary, path)


def valid_png(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                return False
            image.verify()
        return True
    except (OSError, SyntaxError, ValueError):
        return False


def failure_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["config"]),
        str(row["sample_id"]),
        int(row["gt_index"]),
    )


def safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return token or "unknown"


def rect_intersects(left: Sequence[int], right: Sequence[int]) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def display_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    return min(1.0, max_width / max(1, width), max_height / max(1, height))


def scaled_box(box: Sequence[int], scale: float) -> tuple[int, int, int, int]:
    return tuple(round(int(value) * scale) for value in box)  # type: ignore[return-value]


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_labeled_boxes(
    image: Image.Image,
    detections: Sequence[Mapping[str, Any]],
    *,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    draw = ImageDraw.Draw(image)
    line = max(2, round(3 * scale))
    font = load_font(max(10, round(14 * min(1.0, max(scale, 0.7)))))
    for index, detection in enumerate(detections, 1):
        box = scaled_box(detection["bbox"], scale)
        draw.rectangle(box, outline=color, width=line)
        score = detection.get("score", detection.get("confidence"))
        label = f"{index}" if score is None else f"{index}:{float(score):.2f}"
        draw.text((box[0] + 2, max(0, box[1] - 14)), label, fill=color, font=font)


def draw_gt_and_crops(
    image: Image.Image,
    failure: Mapping[str, Any],
    crop_boxes: Sequence[Sequence[int]],
    *,
    scale: float,
) -> None:
    draw = ImageDraw.Draw(image)
    blue = (35, 105, 235)
    yellow = (255, 196, 0)
    red = (225, 20, 35)
    font = load_font(max(11, round(15 * min(1.0, max(scale, 0.7)))))
    intersecting_ids = {int(value) for value in failure.get("intersecting_crop_ids", [])}
    for index, crop in enumerate(crop_boxes, 1):
        box = scaled_box(crop, scale)
        is_partial = index in intersecting_ids
        draw.rectangle(
            box,
            outline=yellow if is_partial else blue,
            width=max(4 if is_partial else 2, round((5 if is_partial else 3) * scale)),
        )
        draw.text(
            (box[0] + 2, box[1] + 2),
            f"crop {index}",
            fill=yellow if is_partial else blue,
            font=font,
        )
    gt = scaled_box(failure["gt_bbox"], scale)
    draw.rectangle(gt, outline=red, width=max(5, round(7 * scale)))
    draw.text((gt[0] + 3, gt[1] + 3), "GT", fill=red, font=font)
    if failure["failure_type"] == "uncovered":
        draw.text(
            (gt[0] + 3, min(image.height - 20, gt[3] + 4)),
            "NO CROP INTERSECTION",
            fill=red,
            font=font,
        )
        return
    compensation = failure.get("required_compensation_px") or {}
    labels = {
        "left": (max(0, gt[0] - 80), (gt[1] + gt[3]) // 2),
        "top": ((gt[0] + gt[2]) // 2, max(0, gt[1] - 18)),
        "right": (min(image.width - 75, gt[2] + 4), (gt[1] + gt[3]) // 2),
        "bottom": ((gt[0] + gt[2]) // 2, min(image.height - 18, gt[3] + 4)),
    }
    abbreviations = {"left": "L", "top": "T", "right": "R", "bottom": "B"}
    for direction in ("left", "top", "right", "bottom"):
        value = int(compensation.get(direction, 0))
        draw.text(
            labels[direction],
            f"{abbreviations[direction]} +{value}px",
            fill=red,
            font=font,
        )


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_four_panel(
    *,
    source_path: Path,
    failure: Mapping[str, Any],
    text_detections: Sequence[Mapping[str, Any]],
    icon_detections: Sequence[Mapping[str, Any]],
    crop_boxes: Sequence[Sequence[int]],
    output_path: Path,
    panel_max_width: int,
    panel_max_height: int,
    expected_size: tuple[int, int],
) -> None:
    with Image.open(source_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    try:
        width, height = original.size
        if (width, height) != expected_size:
            raise ValueError(
                f"source image dimensions changed: expected {expected_size}, "
                f"found {(width, height)} at {source_path}"
            )
        scale = display_scale(width, height, panel_max_width, panel_max_height)
        display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        base = original.resize(display_size, Image.Resampling.LANCZOS)
        panels = [base.copy() for _ in range(4)]
        draw_labeled_boxes(panels[1], text_detections, color=(0, 150, 75), scale=scale)
        draw_labeled_boxes(panels[2], icon_detections, color=(232, 125, 20), scale=scale)
        draw_gt_and_crops(panels[3], failure, crop_boxes, scale=scale)
        gap = 12
        title_height = 30
        canvas_width = display_size[0] * 4 + gap * 5
        header_font = load_font(18)
        body_font = load_font(14)
        top = (
            f"{failure['task']} | {failure['failure_type']} | "
            f"{failure['density']} | {failure['sample_id']} | "
            f"{failure['image_id']} | gt{failure['gt_index']}"
        )
        compensation = failure.get("required_compensation_px")
        footer = (
            f"gt_bbox={failure['gt_bbox']} | intersecting_crop_ids="
            f"{failure.get('intersecting_crop_ids', [])} | required_compensation_px="
            f"{compensation} | required_max_single_side_px="
            f"{failure.get('required_max_single_side_px')} | candidate={failure['config']}"
        )
        measure = ImageDraw.Draw(base)
        header_lines = wrap_text(
            measure, top, header_font, max(1, canvas_width - gap * 2)
        )
        footer_lines = wrap_text(
            measure, footer, body_font, max(1, canvas_width - gap * 2)
        )
        header_height = 12 + max(1, len(header_lines)) * 24
        footer_height = 14 + max(1, len(footer_lines)) * 20
        canvas_height = header_height + title_height + display_size[1] + footer_height
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)
        header_y = 8
        for line in header_lines:
            draw.text((gap, header_y), line, fill=(15, 15, 15), font=header_font)
            header_y += 24
        titles = ("Original", "Text detections", "Icon detections", "GT and crop result")
        for index, (panel, title) in enumerate(zip(panels, titles)):
            left = gap + index * (display_size[0] + gap)
            draw.text((left, header_height + 5), title, fill=(20, 20, 20), font=body_font)
            canvas.paste(panel, (left, header_height + title_height))
        footer_y = header_height + title_height + display_size[1] + 8
        for line in footer_lines:
            draw.text((gap, footer_y), line, fill=(40, 40, 40), font=body_font)
            footer_y += 20
        try:
            atomic_save_png(canvas, output_path)
        finally:
            for panel in panels:
                panel.close()
            base.close()
            canvas.close()
    finally:
        original.close()


def relative_image_path(record: Mapping[str, Any], gallery_dir: Path) -> str:
    return Path(str(record["visualization_4panel"])).resolve().relative_to(
        gallery_dir.parent.resolve()
    ).as_posix()


def html_cards(records: Sequence[Mapping[str, Any]], gallery_dir: Path) -> str:
    cards = []
    for row in records:
        image_path = "../" + relative_image_path(row, gallery_dir)
        cause = str(row.get("manual_root_cause", "")) or "unreviewed"
        note = str(row.get("manual_note", ""))
        cards.append(
            "<article class='card' "
            f"data-task='{html.escape(str(row['task']))}' "
            f"data-density='{html.escape(str(row['density']))}' "
            f"data-failure='{html.escape(str(row['failure_type']))}' "
            f"data-cause='{html.escape(cause)}'>"
            f"<a href='{html.escape(image_path)}'><img loading='lazy' src='{html.escape(image_path)}'></a>"
            f"<div><b>{html.escape(str(row['task']))}</b> · {html.escape(str(row['density']))} · "
            f"{html.escape(str(row['failure_type']))}</div>"
            f"<div>{html.escape(str(row['sample_id']))} · gt{int(row['gt_index'])}</div>"
            f"<div>cause: {html.escape(cause)}</div>"
            f"<div>{html.escape(note)}</div></article>"
        )
    return "\n".join(cards)


def page_html(
    title: str,
    records: Sequence[Mapping[str, Any]],
    gallery_dir: Path,
    *,
    filters: bool,
    introduction: str = "",
) -> str:
    controls = ""
    script = ""
    if filters:
        options = []
        for field, values in (
            ("task", sorted({str(row["task"]) for row in records})),
            ("density", sorted({str(row["density"]) for row in records})),
            ("failure", sorted({str(row["failure_type"]) for row in records})),
            (
                "cause",
                sorted({str(row.get("manual_root_cause") or "unreviewed") for row in records}),
            ),
        ):
            opts = "".join(
                f"<option value='{html.escape(value)}'>{html.escape(value)}</option>"
                for value in values
            )
            options.append(
                f"<label>{field}: <select id='{field}'><option value=''>ALL</option>{opts}</select></label>"
            )
        controls = "<div class='controls'>" + "".join(options) + "</div>"
        script = """
<script>
const fields = ['task','density','failure','cause'];
function applyFilters() {
  document.querySelectorAll('.card').forEach(card => {
    card.hidden = fields.some(field => {
      const selected = document.getElementById(field).value;
      return selected && card.dataset[field] !== selected;
    });
  });
}
fields.forEach(field => document.getElementById(field).addEventListener('change', applyFilters));
</script>"""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#202124;background:#f5f7fa}}
.controls{{position:sticky;top:0;background:white;padding:12px;z-index:2;border:1px solid #ccd3dc}}
label{{margin-right:18px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:16px;margin-top:16px}}
.card{{background:white;border:1px solid #ccd3dc;border-radius:8px;padding:10px;box-shadow:0 1px 3px #ccd3dc}}
.card img{{width:100%;height:auto;display:block;margin-bottom:8px}}
table{{border-collapse:collapse;background:white}}th,td{{border:1px solid #ccd3dc;padding:6px 10px;text-align:left}}th{{background:#1f4e78;color:white}}
</style></head><body><h1>{html.escape(title)}</h1>{introduction}{controls}
<div class="grid">{html_cards(records, gallery_dir)}</div>{script}</body></html>"""


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def representative_partial_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["failure_type"] == "partial_intersection":
            grouped[(str(row["task"]), str(row.get("compensation_bucket")))].append(row)
    selected: list[Mapping[str, Any]] = []
    seen = set()
    for key in sorted(grouped):
        members = sorted(
            grouped[key], key=lambda row: float(row.get("required_max_single_side_px") or 0)
        )
        values = [float(row.get("required_max_single_side_px") or 0) for row in members]
        targets = (
            values[round((len(values) - 1) * 0.50)],
            values[round((len(values) - 1) * 0.90)],
            values[-1],
        )
        for target in targets:
            chosen = min(
                members,
                key=lambda row: (
                    abs(float(row.get("required_max_single_side_px") or 0) - target),
                    str(row["sample_id"]),
                    int(row["gt_index"]),
                ),
            )
            key_value = failure_key(chosen)
            if key_value not in seen:
                selected.append(chosen)
                seen.add(key_value)
    return selected


def diagnosis_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "unreviewed") for row in rows).items()))


def render_diagnosis_summary(
    rows: Sequence[Mapping[str, Any]], gallery_dir: Path
) -> tuple[dict[str, Any], str]:
    reviewed = [row for row in rows if row.get("manual_root_cause")]
    diagnosis = {
        "total_failures": len(rows),
        "reviewed": len(reviewed),
        "unreviewed": len(rows) - len(reviewed),
        "by_task": diagnosis_counts(rows, "task"),
        "by_density": diagnosis_counts(rows, "density"),
        "by_failure_type": diagnosis_counts(rows, "failure_type"),
        "by_root_cause": diagnosis_counts(rows, "manual_root_cause"),
    }
    decision_path = gallery_dir.parent / "failure_diagnosis_decision.json"
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    else:
        decision = {
            "decision": "pending_manual_review",
            "manual_note": "Fill all manual_root_cause fields, then decide whether to stop geometry search.",
        }
    diagnosis["manual_decision"] = decision
    tables = []
    for title, values in (
        ("Task", diagnosis["by_task"]),
        ("Density", diagnosis["by_density"]),
        ("Failure type", diagnosis["by_failure_type"]),
        ("Root cause", diagnosis["by_root_cause"]),
    ):
        body = "".join(
            f"<tr><td>{html.escape(str(key))}</td><td>{value}</td></tr>"
            for key, value in values.items()
        )
        tables.append(f"<h2>{html.escape(title)}</h2><table><tr><th>Value</th><th>Count</th></tr>{body}</table>")
    representatives = []
    by_cause: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in reviewed:
        by_cause[str(row["manual_root_cause"])].append(row)
    for cause in sorted(by_cause):
        members = sorted(
            by_cause[cause],
            key=lambda row: (
                -float(row.get("required_max_single_side_px") or 0),
                str(row["sample_id"]),
            ),
        )
        representatives.extend(members[:3])
    intro = (
        f"<p>Reviewed {len(reviewed)}/{len(rows)}; unreviewed {len(rows) - len(reviewed)}.</p>"
        f"<p>Decision: {html.escape(str(decision.get('decision', 'pending_manual_review')))} — "
        f"{html.escape(str(decision.get('manual_note', '')))}</p>"
        + "".join(tables)
        + "<h2>Representative cases by root cause (up to 3 each)</h2>"
    )
    return diagnosis, page_html(
        "UI5 crop failure diagnosis",
        representatives,
        gallery_dir,
        filters=False,
        introduction=intro,
    )


def validate_expected_failures(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_name: str,
    expected_failures: int | None,
    expected_partial: int | None,
    expected_uncovered: int | None,
    expected_by_task: Mapping[str, Mapping[str, int]] | None = None,
    expected_by_density: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    if expected_failures is not None and len(rows) != expected_failures:
        raise ValueError(f"expected {expected_failures} failures, found {len(rows)}")
    configs = {str(row.get("config")) for row in rows}
    if rows and configs != {config_name}:
        raise ValueError(f"failure configs must be exactly {{{config_name}}}, found {configs}")
    counts = Counter(str(row.get("failure_type")) for row in rows)
    if set(counts) - set(FAILURE_TYPES):
        raise ValueError(f"unsupported failure types: {dict(counts)}")
    if expected_partial is not None and counts["partial_intersection"] != expected_partial:
        raise ValueError(
            f"expected {expected_partial} partial_intersection, found {counts['partial_intersection']}"
        )
    if expected_uncovered is not None and counts["uncovered"] != expected_uncovered:
        raise ValueError(f"expected {expected_uncovered} uncovered, found {counts['uncovered']}")
    for field, expected in (
        ("task", expected_by_task),
        ("detection_density", expected_by_density),
    ):
        if expected is None:
            continue
        actual = {
            key: {
                failure_type: sum(
                    str(row.get(field)) == key
                    and str(row.get("failure_type")) == failure_type
                    for row in rows
                )
                for failure_type in FAILURE_TYPES
            }
            for key in expected
        }
        normalized_expected = {
            str(key): {
                failure_type: int(values.get(failure_type, 0))
                for failure_type in FAILURE_TYPES
            }
            for key, values in expected.items()
        }
        if actual != normalized_expected:
            raise ValueError(
                f"failure distribution by {field} differs: "
                f"expected={normalized_expected}, actual={actual}"
            )


def render_failure_visualizations(
    *,
    output_dir: Path,
    crop_audit_name: str,
    config_name: str,
    expected_failures: int | None,
    expected_partial: int | None,
    expected_uncovered: int | None,
    expected_by_task: Mapping[str, Mapping[str, int]] | None = None,
    expected_by_density: Mapping[str, Mapping[str, int]] | None = None,
    panel_max_width: int = 700,
    panel_max_height: int = 1400,
    resume: bool = True,
    summary_only: bool = False,
    require_manual_review: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve(strict=True)
    audit_dir = output_dir / crop_audit_name
    failures_path = audit_dir / "gt_failures.jsonl"
    visualized_path = audit_dir / "gt_failures_visualized.jsonl"
    failures = read_jsonl(failures_path)
    validate_expected_failures(
        failures,
        config_name=config_name,
        expected_failures=expected_failures,
        expected_partial=expected_partial,
        expected_uncovered=expected_uncovered,
        expected_by_task=expected_by_task,
        expected_by_density=expected_by_density,
    )
    unique = index_unique(
        read_jsonl(output_dir / "manifest" / "unique_images.jsonl"),
        "image_id",
        "unique_images.jsonl",
    )
    samples = index_unique(
        read_jsonl(output_dir / "manifest" / "task_samples.jsonl"),
        "sample_id",
        "task_samples.jsonl",
    )
    detections = index_unique(
        read_jsonl(output_dir / "detections" / "merged" / "detections.jsonl"),
        "image_id",
        "detections.jsonl",
    )
    geometry_rows = [
        row
        for shard in sorted((audit_dir / f"candidate_{config_name}" / "geometry").glob("shard_*.jsonl"))
        for row in read_jsonl(shard)
    ]
    geometry = {str(row["image_id"]): row for row in geometry_rows}
    if len(geometry) != len(geometry_rows):
        raise ValueError("duplicate image_id in selected candidate geometry")
    existing_manual: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    if visualized_path.is_file():
        for row in read_jsonl(visualized_path):
            existing_manual[failure_key(row)] = row
    visual_root = audit_dir / "failure_visualizations"
    gallery_dir = visual_root / "gallery"
    output_rows = []
    rendered = 0
    reused = 0
    for failure in failures:
        image_id = str(failure["image_id"])
        sample_id = str(failure["sample_id"])
        if image_id not in unique or image_id not in detections or image_id not in geometry:
            raise KeyError(f"failure cannot resolve image/detection/geometry: {image_id}")
        if sample_id not in samples:
            raise KeyError(f"failure cannot resolve task sample: {sample_id}")
        sample = samples[sample_id]
        if str(sample["image_id"]) != image_id or str(sample["task"]) != str(failure["task"]):
            raise ValueError(f"failure/sample identity mismatch: {sample_id}")
        image_geometry = geometry[image_id]
        sample_results = {
            str(row["sample_id"]): row for row in image_geometry["sample_results"]
        }
        if sample_id not in sample_results:
            raise KeyError(f"failure cannot resolve task-aware crop boxes: {sample_id}")
        sample_geometry = sample_results[sample_id]
        crop_boxes = sample_geometry["crop_boxes"]
        gt_index = int(failure["gt_index"])
        if not 0 <= gt_index < len(sample["gt_boxes"]):
            raise IndexError(f"invalid gt_index for {sample_id}: {gt_index}")
        if list(sample["gt_boxes"][gt_index]) != list(failure["gt_bbox"]):
            raise ValueError(f"failure GT does not match task sample: {sample_id} gt{gt_index}")
        intersecting = [
            index + 1
            for index, crop in enumerate(crop_boxes)
            if rect_intersects(crop, failure["gt_bbox"])
        ]
        if intersecting != [int(value) for value in failure.get("intersecting_crop_ids", [])]:
            raise ValueError(f"intersecting crop ids disagree for {sample_id} gt{gt_index}")
        detection = detections[image_id]
        expected_size = (int(unique[image_id]["width"]), int(unique[image_id]["height"]))
        detection_size = (int(detection["width"]), int(detection["height"]))
        if detection_size != expected_size:
            raise ValueError(
                f"manifest/detection dimensions disagree for {image_id}: "
                f"{expected_size} != {detection_size}"
            )
        text_rows = detection.get("text_detections", [])
        icon_rows = detection.get("icon_detections", [])
        density = str(
            failure.get("detection_density")
            or sample_geometry.get("detail", {}).get("detection_density")
            or "unknown"
        )
        filename = (
            f"{safe_token(failure['task'])}__{safe_token(density)}__"
            f"{safe_token(failure['failure_type'])}__{safe_token(sample_id)}__gt{gt_index}.png"
        )
        output_path = (
            visual_root / str(failure["task"]) / str(failure["failure_type"]) / filename
        )
        manual = existing_manual.get(failure_key(failure), {})
        root_cause = str(manual.get("manual_root_cause", ""))
        if root_cause and root_cause not in ROOT_CAUSES:
            raise ValueError(f"invalid manual_root_cause for {sample_id} gt{gt_index}: {root_cause}")
        row = {
            **failure,
            "density": density,
            "visualization_4panel": str(output_path.resolve()),
            "text_detection_count": len(text_rows),
            "icon_detection_count": len(icon_rows),
            "crop_count_for_task": len(crop_boxes),
            "manual_root_cause": root_cause,
            "manual_note": str(manual.get("manual_note", "")),
        }
        if summary_only:
            if not valid_png(output_path):
                raise FileNotFoundError(f"summary-only requires existing visualization: {output_path}")
            reused += 1
        elif resume and valid_png(output_path):
            reused += 1
        else:
            source_path = Path(str(unique[image_id]["image_path"])).resolve(strict=True)
            render_four_panel(
                source_path=source_path,
                failure=row,
                text_detections=text_rows,
                icon_detections=icon_rows,
                crop_boxes=crop_boxes,
                output_path=output_path,
                panel_max_width=panel_max_width,
                panel_max_height=panel_max_height,
                expected_size=expected_size,
            )
            rendered += 1
        if not valid_png(output_path):
            raise RuntimeError(f"four-panel visualization is invalid: {output_path}")
        output_rows.append(row)
    if require_manual_review:
        missing = [row for row in output_rows if not row["manual_root_cause"]]
        if missing:
            raise ValueError(f"manual review is incomplete: {len(missing)} failures remain")
        other_without_note = [
            row
            for row in output_rows
            if row["manual_root_cause"] == "other"
            and not str(row.get("manual_note", "")).strip()
        ]
        if other_without_note:
            raise ValueError(
                "manual review is incomplete: "
                f"{len(other_without_note)} failures use 'other' without manual_note"
            )
    atomic_write_jsonl(visualized_path, output_rows)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        gallery_dir / "index.html",
        page_html("UI5 crop failures: all", output_rows, gallery_dir, filters=True),
    )
    uncovered = [row for row in output_rows if row["failure_type"] == "uncovered"]
    atomic_write_text(
        gallery_dir / "uncovered_all.html",
        page_html("UI5 crop failures: all uncovered", uncovered, gallery_dir, filters=True),
    )
    representative = representative_partial_rows(output_rows)
    atomic_write_text(
        gallery_dir / "representative_partial.html",
        page_html(
            "UI5 crop failures: representative partial",
            representative,
            gallery_dir,
            filters=True,
            introduction=(
                "<p>Per task and compensation bucket, cases nearest p50, p90 and maximum are shown.</p>"
            ),
        ),
    )
    diagnosis, diagnosis_html = render_diagnosis_summary(output_rows, gallery_dir)
    atomic_write_json(audit_dir / "failure_diagnosis_summary.json", diagnosis)
    atomic_write_text(gallery_dir / "diagnosis_summary.html", diagnosis_html)
    return {
        "config": config_name,
        "failures": len(output_rows),
        "partial_intersection": sum(
            row["failure_type"] == "partial_intersection" for row in output_rows
        ),
        "uncovered": len(uncovered),
        "rendered": rendered,
        "reused": reused,
        "manual_reviewed": diagnosis["reviewed"],
        "manual_unreviewed": diagnosis["unreviewed"],
        "visualized_jsonl": str(visualized_path.resolve()),
        "gallery_index": str((gallery_dir / "index.html").resolve()),
        "diagnosis_summary": str((gallery_dir / "diagnosis_summary.html").resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    strict_current = (
        args.config == DEFAULT_CONFIG
        and args.expected_failures == 107
        and args.expected_partial == 87
        and args.expected_uncovered == 20
    )
    result = render_failure_visualizations(
        output_dir=args.output_dir,
        crop_audit_name=args.crop_audit_name,
        config_name=args.config,
        expected_failures=args.expected_failures,
        expected_partial=args.expected_partial,
        expected_uncovered=args.expected_uncovered,
        expected_by_task=EXPECTED_FAILURES_BY_TASK if strict_current else None,
        expected_by_density=(
            EXPECTED_FAILURES_BY_DENSITY if strict_current else None
        ),
        panel_max_width=args.panel_max_width,
        panel_max_height=args.panel_max_height,
        resume=args.resume,
        summary_only=args.summary_only,
        require_manual_review=args.require_manual_review,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
