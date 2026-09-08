#!/usr/bin/env python3
"""Select application subsets from converted UI-Lens UI5 evaluation JSONL.

Selection uses only extra_info.original_infos.app_name, never GT positivity,
image appearance, or filename substrings. Images and annotations are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from locany_ui5_common import TASK_JSONL


def app_name(value: object) -> str:
    # Some exports wrap scalar metadata in a singleton list.
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("app_name must be a nonempty string or a singleton string list")
    return unicodedata.normalize("NFKC", value).strip().casefold()


def subset(input_dir: Path, output_dir: Path | None = None,
           apps: tuple[str, ...] = ("douyin", "抖音"), *, list_apps: bool = False) -> dict:
    input_dir = input_dir.expanduser().resolve(strict=True)
    if output_dir is not None:
        output_dir = output_dir.expanduser().absolute()
        if list_apps:
            raise ValueError("--list-apps cannot be combined with --output-dir")
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}; choose a new directory")
    selected_apps = {app_name(value) for value in apps}
    all_apps, selected_paths, all_paths = Counter(), set(), set()
    records, selected_manifest, clipping_audit, issues = {}, [], [], []
    report = {
        "input_dir": str(input_dir), "selection_field": "extra_info.original_infos.app_name",
        "selected_app_names": sorted(selected_apps), "coordinate_format": "pixel_xyxy",
        "selection_uses_gt": False, "images_copied": False, "tasks": {},
    }
    for task, filename in TASK_JSONL.items():
        path = input_dir / filename
        raw = path.read_bytes()
        rows, task_apps = [], Counter()
        stats = {"input_rows": 0, "rows": 0, "positive": 0, "negative": 0,
                 "boxes": 0, "clipped_images": 0, "clipped_boxes": 0,
                 "missing_or_invalid_app_name": 0, "input_sha256": hashlib.sha256(raw).hexdigest()}
        for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                stats["input_rows"] += 1
                try:
                    name = app_name(row.get("extra_info", {}).get("original_infos", {}).get("app_name"))
                except ValueError:
                    stats["missing_or_invalid_app_name"] += 1
                    issues.append(f"{filename}:{line_number}: missing/invalid original_infos.app_name")
                    continue
                task_apps[name] += 1
                all_apps[name] += 1
                if list_apps or name not in selected_apps:
                    continue
                images = row["images"]
                if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
                    raise ValueError("expected one image path in images")
                image = Path(images[0]).expanduser()
                if not image.is_absolute():
                    raise ValueError("converted image paths must be absolute before moving JSONL into a subset directory")
                if not image.is_file():
                    raise FileNotFoundError(f"selected image is missing: {image}")
                image_key = str(image.resolve())
                if (task, image_key) in all_paths:
                    raise ValueError("duplicate selected image within task")
                all_paths.add((task, image_key))
                selected_paths.add(image_key)
                boxes, types = row["answer"]["bbox"], row["answer"]["types"]
                if not isinstance(boxes, list) or not isinstance(types, list) or len(boxes) != len(types):
                    raise ValueError("expected parallel converted answer.bbox and answer.types lists")
                corrections = row["extra_info"].get("bbox_conversion", {}).get("clipped_boxes", [])
                stats["rows"] += 1
                stats["positive"] += bool(boxes)
                stats["negative"] += not boxes
                stats["boxes"] += len(boxes)
                stats["clipped_images"] += bool(corrections)
                stats["clipped_boxes"] += len(corrections)
                # Preserve the complete original converted record, including IDs,
                # original annotations, coordinates, clipping audit and metadata.
                rows.append(line)
                selected_manifest.append({"task": task, "input_file": str(path),
                                          "input_line": line_number, "id": row.get("id"),
                                          "app_name": name, "image": images[0]})
                if corrections:
                    extra = row["extra_info"]
                    clipping_audit.append({"task": task, "image": images[0],
                        "source_file": extra.get("source_file"), "source_line": extra.get("source_line"),
                        "image_size": extra["original_infos"].get("image_size"),
                        "clipped_boxes": corrections})
            except (KeyError, ValueError, TypeError, OSError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        stats["available_app_names"] = dict(sorted(task_apps.items()))
        stats["excluded_rows"] = stats["input_rows"] - stats["rows"]
        records[task] = rows
        report["tasks"][task] = stats
        print(f"{task}: input={stats['input_rows']} selected={stats['rows']} "
              f"positive={stats['positive']} negative={stats['negative']} boxes={stats['boxes']} "
              f"invalid_app={stats['missing_or_invalid_app_name']}", flush=True)
    report["available_app_names"] = dict(sorted(all_apps.items()))
    report["unique_images_by_path"] = len(selected_paths)
    report["image_task_records"] = sum(s["rows"] for s in report["tasks"].values())
    report["empty_tasks"] = [task for task, stats in report["tasks"].items() if not stats["rows"]]
    report["invalid_app_records"] = len(issues)
    print("Available app names (image-task records): " + json.dumps(report["available_app_names"], ensure_ascii=False))
    if list_apps:
        return report
    if issues:
        raise ValueError(f"Cannot certify a complete app subset: {len(issues)} records lack valid app metadata. "
                         f"Examples: {issues[:5]}; inspect the original annotations before filtering.")
    if not report["image_task_records"]:
        raise ValueError(f"No matches for {sorted(selected_apps)}; inspect --list-apps and specify exact --app-name values")
    if output_dir is not None:
        # All files are parsed and selection checked before creating any output.
        output_dir.mkdir(parents=True, exist_ok=False)
        for task, rows in records.items():
            (output_dir / TASK_JSONL[task]).write_text("".join(line + "\n" for line in rows), encoding="utf-8")
        for filename, rows in (("selection_manifest.jsonl", selected_manifest),
                               ("bbox_clipping_audit.jsonl", clipping_audit)):
            (output_dir / filename).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        (output_dir / "subset_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {report['image_task_records']} image-task records / "
          f"{report['unique_images_by_path']} unique image paths; empty tasks={report['empty_tasks']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Omit to preview selection and statistics without writing")
    parser.add_argument("--app-name", action="append", help="Exact app name, repeat for aliases; defaults: douyin and 抖音")
    parser.add_argument("--list-apps", action="store_true", help="Inventory original app names without selecting or writing")
    args = parser.parse_args()
    try:
        subset(args.input_dir, args.output_dir, tuple(args.app_name or ("douyin", "抖音")), list_apps=args.list_apps)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
