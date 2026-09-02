#!/usr/bin/env python3
"""Render deterministic UI5 rollout audit composites and an HTML index."""
from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "gt": (20, 180, 70),
    "matched": (30, 100, 230),
    "unmatched": (225, 45, 45),
    "crop": (240, 190, 20),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--panel-long-side", type=int, default=640)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {path}:{line_no}")
                rows.append(value)
    return rows


def draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], color: tuple[int, int, int], width: int) -> None:
    draw.rectangle(tuple(round(float(value)) for value in box), outline=color, width=width)


def render_panel(
    source: Image.Image,
    gt: Sequence[Sequence[float]],
    rollout: Mapping[str, Any],
    long_side: int,
) -> Image.Image:
    canvas = source.copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 350))
    for crop in rollout.get("crop_boundaries", []):
        draw_box(draw, crop, COLORS["crop"], line_width)
    for box in gt:
        draw_box(draw, box, COLORS["gt"], line_width)
    matched = {
        int(pair["pred_index"])
        for pair in rollout.get("matched_pairs", [])
        if pair.get("is_tp")
    }
    for pred_index, box in enumerate(rollout.get("pred_global", [])):
        draw_box(
            draw,
            box,
            COLORS["matched"] if pred_index in matched else COLORS["unmatched"],
            line_width,
        )
    scale = min(1.0, long_side / max(canvas.size))
    if scale < 1:
        canvas.thumbnail(
            (max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale))),
            Image.Resampling.LANCZOS,
        )
    banner_height = 28
    panel = Image.new("RGB", (canvas.width, canvas.height + banner_height), "white")
    panel.paste(canvas, (0, banner_height))
    label = (
        f"rollout {rollout['rollout_id']} | {rollout['error_type']} | "
        f"exact={bool(rollout['exact_correct'])}"
    )
    ImageDraw.Draw(panel).text((6, 7), label, fill=(20, 20, 20), font=ImageFont.load_default())
    canvas.close()
    return panel


def render(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve(strict=True)
    bundle = args.bundle_root.expanduser().resolve(strict=True)
    selection_path = output_root / "reports" / "gallery_selection.jsonl"
    selections = read_jsonl(selection_path)
    visual_root = output_root / "visualizations"
    assets = visual_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    cards = []
    failures = []
    for row in selections:
        key = (str(row["model_id"]), str(row["task"]), str(row["category"]))
        counts[key] += 1
        if counts[key] > 10:
            raise RuntimeError(f"gallery cap exceeded for {key}")
        source_path = bundle / str(row["image_relpath"])
        try:
            with Image.open(source_path) as opened:
                source = opened.convert("RGB")
            panels = [
                render_panel(source, row.get("gt_global", []), rollout, args.panel_long_side)
                for rollout in sorted(row["rollouts"], key=lambda item: int(item["rollout_id"]))
            ]
            source.close()
            width = sum(panel.width for panel in panels) + 12 * (len(panels) - 1)
            height = max(panel.height for panel in panels)
            composite = Image.new("RGB", (width, height), (238, 242, 247))
            x = 0
            for panel in panels:
                composite.paste(panel, (x, 0))
                x += panel.width + 12
                panel.close()
            relative = Path("assets") / key[0] / key[1] / key[2] / f"{row['record_id']}.jpg"
            destination = visual_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            composite.save(temporary, format="JPEG", quality=90, subsampling=0)
            composite.close()
            os.replace(temporary, destination)
            cards.append({**row, "visual_relpath": relative.as_posix()})
        except Exception as exc:
            failures.append(
                {
                    "record_id": row.get("record_id"),
                    "image": str(source_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    sections = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for card in cards:
        grouped.setdefault(
            (str(card["model_id"]), str(card["task"]), str(card["category"])), []
        ).append(card)
    for key in sorted(grouped):
        model, task, category = key
        items = []
        for card in grouped[key]:
            items.append(
                "<article class='card'>"
                f"<img loading='lazy' src='{html.escape(card['visual_relpath'])}' "
                f"alt='{html.escape(str(card['record_id']))}'>"
                f"<div><code>{html.escape(str(card['record_id']))}</code> "
                f"image=<code>{html.escape(str(card['source_image_id']))}</code> "
                f"correct={int(card['correct_count'])}/4</div>"
                "</article>"
            )
        sections.append(
            f"<section><h2>{html.escape(model)} / {html.escape(task)} / "
            f"{html.escape(category)} ({len(items)})</h2>"
            f"<div class='grid'>{''.join(items)}</div></section>"
        )
    failure_html = ""
    if failures:
        failure_html = (
            f"<section><h2>Render failures ({len(failures)})</h2><pre>"
            + html.escape(json.dumps(failures, ensure_ascii=False, indent=2))
            + "</pre></section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UI5 train rollout audit gallery</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f8fb;color:#172033}}
h1{{margin-bottom:8px}} h2{{margin-top:36px;border-bottom:1px solid #ccd6e3;padding-bottom:8px}}
.legend span{{margin-right:18px}} .swatch{{display:inline-block;width:14px;height:14px;margin-right:5px;vertical-align:-2px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}}
.card{{background:white;border:1px solid #dbe2ea;border-radius:8px;padding:10px;box-shadow:0 1px 3px #0001}}
.card img{{width:100%;height:auto;display:block;margin-bottom:8px}} code{{font-size:12px}}
pre{{white-space:pre-wrap;background:white;padding:12px;border:1px solid #ddd}}
</style></head><body>
<h1>UI5 train dual-model 4+4 rollout audit</h1>
<p>Four answers per model. The combined 4+4 view is cross-model consistency, not pass@8.</p>
<div class="legend">
<span><i class="swatch" style="background:#14b446"></i>GT</span>
<span><i class="swatch" style="background:#1e64e6"></i>matched prediction</span>
<span><i class="swatch" style="background:#e12d2d"></i>unmatched prediction</span>
<span><i class="swatch" style="background:#f0be14"></i>crop boundary</span>
</div>
{''.join(sections)}{failure_html}
</body></html>"""
    index_path = visual_root / "index.html"
    temporary = index_path.with_name(f".{index_path.name}.tmp-{os.getpid()}")
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, index_path)
    summary = {
        "index": str(index_path),
        "rendered": len(cards),
        "failures": failures,
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
    }
    (visual_root / "gallery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    render(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
