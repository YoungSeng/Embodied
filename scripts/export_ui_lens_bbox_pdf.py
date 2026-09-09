#!/usr/bin/env python3
"""Export UI-Lens screenshots with vector GT rectangles as separate paper PDFs.

Defaults select positive annotations for 788.png and 819.png from the converted
UI5 data. Multiple positive tasks are reported instead of silently merging GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
from locany_ui5_common import TASK_JSONL, TASK_ISSUE_NAMES


def select_samples(eval_dir, image_names, task=None):
    matches = {name: [] for name in image_names}
    for key, filename in TASK_JSONL.items():
        if task is not None and key != task:
            continue
        source = eval_dir / filename
        with source.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                images = row.get("images", [])
                if len(images) != 1:
                    continue
                name = Path(images[0]).name
                if name not in matches:
                    continue
                boxes = row["answer"]["bbox"]
                if not isinstance(boxes, list):
                    raise ValueError(f"{source}:{line_number}: answer.bbox must be a list")
                print(f"{name}: task={key} ({TASK_ISSUE_NAMES[key]}), "
                      f"{'positive' if boxes else 'negative'}, boxes={len(boxes)}", flush=True)
                if boxes:
                    path = Path(images[0]).expanduser()
                    if not path.is_absolute():
                        path = eval_dir / path
                    matches[name].append({"image_name": name, "image": str(path.resolve()),
                                          "task": key, "task_name": TASK_ISSUE_NAMES[key],
                                          "bbox_xyxy": boxes, "annotation_file": str(source),
                                          "annotation_line": line_number})
    selected = []
    for name, candidates in matches.items():
        if not candidates:
            raise ValueError(f"No positive annotation found for {name}; check --eval-dir and --task")
        if len(candidates) != 1:
            tasks = [item["task"] for item in candidates]
            raise ValueError(f"{name} has multiple positive annotations: {tasks}; use --task to select one task")
        selected.append(candidates[0])
    return selected


def validate_boxes(boxes, width, height):
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box
        ):
            raise ValueError(f"Invalid pixel xyxy box: {box!r}")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Out-of-bounds box {box} for {width}x{height}; use the validated converted GT")


def pdf_rect(box, width, height, scale, line_width):
    # Clamp the stroke centre only at the page edge, keeping the whole stroke
    # visible. GT in the manifest remains the exact original pixel coordinates.
    half = line_width / 2
    x1, y1, x2, y2 = box
    left = max(half, min(width * scale - half, x1 * scale))
    right = max(half, min(width * scale - half, x2 * scale))
    bottom = max(half, min(height * scale - half, (height-y2) * scale))
    top = max(half, min(height * scale - half, (height-y1) * scale))
    return left, bottom, right-left, top-bottom


def export(eval_dir, output, image_names=("788.png", "819.png"), *, task=None, dpi=144, line_width=1.5):
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    if not math.isfinite(dpi) or dpi <= 0 or not math.isfinite(line_width) or line_width <= 0:
        raise ValueError("dpi and line width must be finite and positive")
    names = list(image_names)
    if not names or any(Path(name).name != name for name in names) or len({Path(n).stem for n in names}) != len(names):
        raise ValueError("Specify unique image basenames, with distinct stems")
    eval_dir = eval_dir.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    samples = select_samples(eval_dir, names, task)
    destinations = [output / "export_manifest.json"]
    for sample in samples:
        stem = Path(sample["image_name"]).stem
        sample["pdf"] = str(output / f"{stem}_bbox.pdf")
        sample["preview"] = str(output / f"{stem}_bbox_preview.png")
        destinations.extend((Path(sample["pdf"]), Path(sample["preview"])))
    occupied = [str(path) for path in destinations if path.exists()]
    if occupied:
        raise FileExistsError(f"Output files already exist: {occupied}; use a new --output-dir")
    prepared = []
    try:
        # Validate both original images and all selected coordinates before writing.
        for sample in samples:
            path = Path(sample["image"])
            with Image.open(path) as opened:
                if opened.getexif().get(274, 1) != 1:
                    raise ValueError(f"Nontrivial EXIF orientation: {path}; reconcile GT orientation first")
                rgba = opened.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            rgba.close()
            prepared.append(rgb)
            width, height = rgb.size
            validate_boxes(sample["bbox_xyxy"], width, height)
            if line_width >= min(width, height) * 72 / dpi:
                raise ValueError("Line width is larger than the PDF page")
            sample.update(width=width, height=height, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        output.mkdir(parents=True, exist_ok=True)
        scale = 72 / dpi
        for sample, rgb in zip(samples, prepared):
            width, height = rgb.size
            page = Canvas(sample["pdf"], pagesize=(width*scale, height*scale), pageCompression=1)
            page.setTitle(f"UI-Lens {sample['image_name']} - {sample['task']}")
            page.drawImage(ImageReader(rgb), 0, 0, width=width*scale, height=height*scale)
            page.setStrokeColorRGB(1, 0, 0)
            page.setLineWidth(line_width)
            page.setLineJoin(0)
            preview = rgb.copy()
            draw = ImageDraw.Draw(preview)
            for box in sample["bbox_xyxy"]:
                page.rect(*pdf_rect(box, width, height, scale, line_width), stroke=1, fill=0)
                # Preview uses the same page-edge-adjusted stroke geometry.
                x, y, w, h = pdf_rect(box, width, height, scale, line_width)
                stroke = max(1, round(line_width / scale))
                half = stroke / 2
                draw.rectangle((max(0, round(x/scale-half)), max(0, round(height-(y+h)/scale-half)),
                                min(width-1, round((x+w)/scale+half)), min(height-1, round(height-y/scale+half))),
                               outline="red", width=stroke)
            page.showPage()
            page.save()
            preview.save(sample["preview"])
            preview.close()
            sample.update(page_points=[width*scale, height*scale], dpi=dpi, line_width_points=line_width)
            print(f"SAVED PDF: {sample['pdf']}\nPREVIEW: {sample['preview']}", flush=True)
        (output / "export_manifest.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    finally:
        for image in prepared:
            image.close()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("/mnt/bn/intelligent-service-yg/dataset/UI_lens_ui5_eval_clip_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/bn/intelligent-service-yg/dataset/UI_lens_paper_figures"))
    parser.add_argument("--image-names", nargs="+", default=["788.png", "819.png"])
    parser.add_argument("--task", choices=list(TASK_JSONL), default=None)
    parser.add_argument("--dpi", type=float, default=144, help="Page sizing only; original embedded image pixels are not downsampled")
    parser.add_argument("--line-width", type=float, default=1.5, help="Red vector stroke width in PDF points")
    args = parser.parse_args()
    try:
        export(args.eval_dir, args.output_dir, args.image_names, task=args.task, dpi=args.dpi, line_width=args.line_width)
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}. Install with: python -m pip install reportlab\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
