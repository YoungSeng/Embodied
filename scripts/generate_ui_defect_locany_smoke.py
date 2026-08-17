#!/usr/bin/env python3
"""Generate ten deterministic, synthetic UI-defect smoke samples.

The dataset has five tasks. Each task contains one positive and one negative
sample. Images are intentionally small so a single RTX 4090 can exercise the
same LocateAnything data/forward/backward path with a short SDPA context.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 640
HEIGHT = 400
BACKGROUND = "#EEF2F7"
SURFACE = "#FFFFFF"
TEXT = "#18212F"
MUTED = "#778196"
PRIMARY = "#4F6BED"
BORDER = "#D6DCE8"
WARNING = "#F59E0B"

README_TEXT = """# UI Defect Smoke Samples

This directory contains 10 fully synthetic samples for testing the LocateAnything UI-defect training pipeline. It contains one positive and one negative example for each class:

| annotation file | label |
| --- | --- |
| `ui_occlusion_train.jsonl` | `overlapping elements` |
| `ui_cropping_train.jsonl` | `cropped element` |
| `ui_text_overflow_train.jsonl` | `text overflow` |
| `ui_text_ellipsis_train.jsonl` | `abnormal text ellipsis` |
| `ui_content_missing_train.jsonl` | `missing content` |

Use `recipe/ui_defect_5class_train.json` as `META_PATH`. All paths are relative to the `Embodied` project root.

The images are generated locally by `scripts/generate_ui_defect_locany_smoke.py`; they do not contain internal screenshots or user data. This dataset is only for a 1–2 step pipeline smoke test and must not be used for model evaluation or reported results.
"""


@dataclass(frozen=True)
class Task:
    name: str
    label: str
    draw_positive: Callable[[ImageDraw.ImageDraw, ImageFont.ImageFont, ImageFont.ImageFont], tuple[int, int, int, int]]
    draw_negative: Callable[[ImageDraw.ImageDraw, ImageFont.ImageFont, ImageFont.ImageFont], None]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/samples/ui_defect_locany_smoke",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated smoke dataset.",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = SURFACE,
    outline: str = BORDER,
    radius: int = 12,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def common_frame(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
    title: str,
) -> None:
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=BACKGROUND)
    draw.rectangle((0, 0, 118, HEIGHT), fill="#172033")
    draw.text((24, 24), "NOVA", fill="#FFFFFF", font=title_font)
    for idx, item in enumerate(("Home", "Orders", "Messages", "Settings")):
        y = 92 + idx * 48
        if item == title:
            draw.rounded_rectangle((12, y - 8, 106, y + 28), radius=8, fill="#2E3B55")
        draw.text((26, y), item, fill="#DDE4F2", font=font)
    draw.rectangle((118, 0, WIDTH, 62), fill=SURFACE)
    draw.text((146, 20), title, fill=TEXT, font=title_font)
    draw.ellipse((574, 15, 606, 47), fill="#8EA2FF")


def button(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: str = PRIMARY,
) -> None:
    draw.rounded_rectangle(box, radius=9, fill=fill)
    text_box = draw.textbbox((0, 0), text, font=font)
    tw = text_box[2] - text_box[0]
    th = text_box[3] - text_box[1]
    x1, y1, x2, y2 = box
    draw.text(
        ((x1 + x2 - tw) / 2, (y1 + y2 - th) / 2 - 2),
        text,
        fill="#FFFFFF",
        font=font,
    )


def draw_occlusion_positive(draw, font, title_font):
    common_frame(draw, font, title_font, "Orders")
    rounded(draw, (155, 96, 510, 286))
    draw.text((180, 120), "Order #2048", fill=TEXT, font=title_font)
    draw.text((180, 162), "Shipping address", fill=MUTED, font=font)
    draw.text((180, 192), "2-8-1 Marunouchi, Tokyo", fill=TEXT, font=font)
    button(draw, (180, 232, 310, 270), "Confirm", font)
    # A help panel incorrectly covers the right side of the order card/button.
    rounded(draw, (280, 184, 594, 330), fill="#FFF7DF", outline="#F4C96B")
    draw.text((306, 208), "Need help?", fill="#7C5310", font=title_font)
    draw.text((306, 246), "Chat with support", fill="#7C5310", font=font)
    return (280, 184, 510, 286)


def draw_occlusion_negative(draw, font, title_font):
    common_frame(draw, font, title_font, "Orders")
    rounded(draw, (155, 96, 430, 286))
    draw.text((180, 120), "Order #2049", fill=TEXT, font=title_font)
    draw.text((180, 162), "Shipping address", fill=MUTED, font=font)
    draw.text((180, 192), "1-4-2 Ginza, Tokyo", fill=TEXT, font=font)
    button(draw, (180, 232, 310, 270), "Confirm", font)
    rounded(draw, (450, 96, 610, 214), fill="#FFF7DF", outline="#F4C96B")
    draw.text((470, 118), "Need help?", fill="#7C5310", font=font)
    draw.text((470, 154), "Open chat", fill="#7C5310", font=font)


def draw_cropping_positive(draw, font, title_font):
    common_frame(draw, font, title_font, "Messages")
    rounded(draw, (155, 96, 485, 310))
    draw.text((180, 120), "Inbox", fill=TEXT, font=title_font)
    for idx, name in enumerate(("Design team", "Research group", "Alex")):
        y = 166 + idx * 46
        draw.ellipse((180, y, 206, y + 26), fill="#AFC0FF")
        draw.text((220, y + 4), name, fill=TEXT, font=font)
    # Notification card extends beyond the right edge and is visibly clipped.
    rounded(draw, (520, 112, 700, 238), fill="#E8EEFF", outline="#9CAFED")
    draw.text((542, 136), "New message", fill=TEXT, font=title_font)
    draw.text((542, 177), "Open details", fill=PRIMARY, font=font)
    return (520, 112, 639, 238)


def draw_cropping_negative(draw, font, title_font):
    common_frame(draw, font, title_font, "Messages")
    rounded(draw, (155, 96, 418, 310))
    draw.text((180, 120), "Inbox", fill=TEXT, font=title_font)
    for idx, name in enumerate(("Design team", "Research group", "Alex")):
        y = 166 + idx * 46
        draw.ellipse((180, y, 206, y + 26), fill="#AFC0FF")
        draw.text((220, y + 4), name, fill=TEXT, font=font)
    rounded(draw, (438, 112, 610, 238), fill="#E8EEFF", outline="#9CAFED")
    draw.text((460, 136), "New message", fill=TEXT, font=title_font)
    draw.text((460, 177), "Open details", fill=PRIMARY, font=font)


def draw_overflow_positive(draw, font, title_font):
    common_frame(draw, font, title_font, "Settings")
    rounded(draw, (155, 96, 610, 310))
    draw.text((180, 122), "Account profile", fill=TEXT, font=title_font)
    draw.text((180, 167), "Display name", fill=MUTED, font=font)
    rounded(draw, (180, 198, 452, 244), radius=8)
    # Text is rendered after the input and intentionally continues outside it.
    draw.text(
        (194, 212),
        "Taylor Chen - Multimodal Interaction Researcher",
        fill=TEXT,
        font=font,
    )
    draw.text((180, 270), "Visible to your team", fill=MUTED, font=font)
    return (180, 198, 590, 244)


def draw_overflow_negative(draw, font, title_font):
    common_frame(draw, font, title_font, "Settings")
    rounded(draw, (155, 96, 610, 310))
    draw.text((180, 122), "Account profile", fill=TEXT, font=title_font)
    draw.text((180, 167), "Display name", fill=MUTED, font=font)
    rounded(draw, (180, 198, 452, 244), radius=8)
    draw.text((194, 212), "Taylor Chen", fill=TEXT, font=font)
    draw.text((180, 270), "Visible to your team", fill=MUTED, font=font)


def draw_ellipsis_positive(draw, font, title_font):
    common_frame(draw, font, title_font, "Home")
    rounded(draw, (155, 96, 610, 310))
    draw.text((180, 120), "Quick actions", fill=TEXT, font=title_font)
    rounded(draw, (180, 168, 450, 224), fill="#F7F9FC")
    # The short label is truncated even though most of the control is empty.
    draw.text((200, 186), "Create new...", fill=TEXT, font=font)
    draw.text((180, 260), "Recent activity", fill=MUTED, font=font)
    return (180, 168, 450, 224)


def draw_ellipsis_negative(draw, font, title_font):
    common_frame(draw, font, title_font, "Home")
    rounded(draw, (155, 96, 610, 310))
    draw.text((180, 120), "Quick actions", fill=TEXT, font=title_font)
    rounded(draw, (180, 168, 450, 224), fill="#F7F9FC")
    draw.text((200, 186), "Create new project", fill=TEXT, font=font)
    draw.text((180, 260), "Recent activity", fill=MUTED, font=font)


def draw_missing_positive(draw, font, title_font):
    common_frame(draw, font, title_font, "Home")
    draw.text((155, 94), "Recommended for you", fill=TEXT, font=title_font)
    rounded(draw, (155, 132, 330, 296))
    draw.rectangle((170, 148, 315, 226), fill="#DDE7FF")
    draw.text((170, 248), "Interface design", fill=TEXT, font=font)
    rounded(draw, (350, 132, 525, 296))
    # The second card has an empty media/content region.
    draw.rectangle((365, 148, 510, 226), fill="#F8FAFD", outline=BORDER, width=2)
    draw.line((390, 190, 485, 190), fill=BORDER, width=2)
    draw.text((365, 248), "", fill=TEXT, font=font)
    return (350, 132, 525, 296)


def draw_missing_negative(draw, font, title_font):
    common_frame(draw, font, title_font, "Home")
    draw.text((155, 94), "Recommended for you", fill=TEXT, font=title_font)
    rounded(draw, (155, 132, 330, 296))
    draw.rectangle((170, 148, 315, 226), fill="#DDE7FF")
    draw.text((170, 248), "Interface design", fill=TEXT, font=font)
    rounded(draw, (350, 132, 525, 296))
    draw.rectangle((365, 148, 510, 226), fill="#FFE7C2")
    draw.ellipse((414, 162, 460, 208), fill=WARNING)
    draw.text((365, 248), "Model evaluation", fill=TEXT, font=font)


TASKS = (
    Task("ui_occlusion", "overlapping elements", draw_occlusion_positive, draw_occlusion_negative),
    Task("ui_cropping", "cropped element", draw_cropping_positive, draw_cropping_negative),
    Task("ui_text_overflow", "text overflow", draw_overflow_positive, draw_overflow_negative),
    Task("ui_text_ellipsis", "abnormal text ellipsis", draw_ellipsis_positive, draw_ellipsis_negative),
    Task("ui_content_missing", "missing content", draw_missing_positive, draw_missing_negative),
)


def normalize_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        round(x1 / WIDTH * 1000),
        round(y1 / HEIGHT * 1000),
        round(x2 / WIDTH * 1000),
        round(y2 / HEIGHT * 1000),
    )


def make_record(label: str, image_name: str, box: tuple[int, int, int, int] | None) -> dict:
    prompt = f"Locate all the instances that match the following description: {label}."
    if box is None:
        answer = "<box>none</box>"
    else:
        x1, y1, x2, y2 = normalize_box(box)
        answer = f"<ref>{label}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "image": image_name,
    }


def jsonl_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "samples" / "ui_defect_locany_smoke"
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise SystemExit(
                f"Output directory is not empty: {output_dir}\n"
                "Pass --force only when you intend to replace the generated samples."
            )
        shutil.rmtree(output_dir)

    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    recipe_dir = output_dir / "recipe"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    recipe_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(16)
    title_font = load_font(19, bold=True)
    annotation_paths: list[str] = []
    manifest_samples: list[dict] = []

    try:
        output_rel = output_dir.relative_to(project_root).as_posix()
    except ValueError:
        output_rel = output_dir.as_posix()

    for task in TASKS:
        annotation_path = annotation_dir / f"{task.name}_train.jsonl"
        records: list[dict] = []

        for kind in ("positive", "negative"):
            image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
            draw = ImageDraw.Draw(image)
            if kind == "positive":
                box = task.draw_positive(draw, font, title_font)
            else:
                task.draw_negative(draw, font, title_font)
                box = None

            image_name = f"{task.name}_{kind}.png"
            image.save(image_dir / image_name, format="PNG", optimize=True)
            record = make_record(task.label, image_name, box)
            records.append(record)
            manifest_samples.append(
                {
                    "task": task.name,
                    "label": task.label,
                    "kind": kind,
                    "image": image_name,
                    "normalized_box": list(normalize_box(box)) if box else None,
                }
            )

        annotation_path.write_text(
            "".join(jsonl_line(record) for record in records),
            encoding="utf-8",
        )
        annotation_paths.append(
            f"{output_rel}/annotations/{task.name}_train.jsonl"
        )

    recipe = {
        "ui_defect_5class_smoke": {
            "annotation": annotation_paths,
            "root": f"{output_rel}/images",
            "repeat_time": 1.0,
            "data_augment": False,
        }
    }
    recipe_path = recipe_dir / "ui_defect_5class_train.json"
    recipe_path.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "name": "LocateAnything UI defect smoke samples",
        "provenance": "fully synthetic; generated by scripts/generate_ui_defect_locany_smoke.py",
        "intended_use": "pipeline smoke testing only; not model evaluation",
        "image_size": [WIDTH, HEIGHT],
        "num_samples": len(manifest_samples),
        "samples": manifest_samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(README_TEXT, encoding="utf-8")

    print(f"Generated {len(manifest_samples)} samples under: {output_dir}")
    print(f"Recipe: {recipe_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
