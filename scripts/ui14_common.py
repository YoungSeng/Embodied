"""CPU-only paths, serialization and source-image identities for UI14."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from eaglevl.ui_task_registry import UI_TASKS, UI5_TASKS, UI9_TASKS, load_registry, get_task

WORKSPACE = "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace"
UI9_DATA_ROOT = "/mnt/bn/intelligent-service-yg/dataset/gui/ui9_datasets_v1"
DATA_ROOT = WORKSPACE + "/gui_data/ui14_cpt9000_v1"
CLUSTER_PROJECT = WORKSPACE + "/code/Eagle_LocateUI5_v4/Embodied-ui14-cpt9000"
INIT_CHECKPOINT = WORKSPACE + "/gui_models/locany-3b-ui-cpt-v4-v3-h20x2-formal-segmented-eval/checkpoint-9000"
UI5_AUDIT = WORKSPACE + "/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_crop_audit_20260825/crop_audit_v4_gt_repair"
UI5_RECIPE = UI5_AUDIT + "/training_recipes/ui_defect_5class_train_crop_only.json"
SCAN_NAME = "horizontal_scan_v5_raw_detector_edge_aligned"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{number}") from exc


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def image_identity(path):
    from PIL import Image, ImageOps
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        # Decoded content catches re-encoded files and cross-source path copies.
        value = hashlib.sha256(f"{width}x{height}:RGB:".encode() + image.tobytes()).hexdigest()
    return value, width, height


def paths_for(root, task, split):
    root = Path(root)
    return {
        "normalized": root / "normalized" / task / f"{split}.jsonl",
        "detector_input": root / "detector_inputs" / task / f"{split}.jsonl",
        "detector_inputs": root / "detector_inputs" / task / f"{split}.json",
        "cache": root / "cache" / task / split,
        "derived": root / "derived" / task / f"{split}.jsonl",
        "crop_images": root / "crop_images" / task / split,
    }
