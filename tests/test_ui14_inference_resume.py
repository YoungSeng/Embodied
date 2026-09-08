"""Exercise the actual CPU manifest/pending functions without importing CUDA."""
import argparse
import ast
from collections import defaultdict
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from eaglevl.ui_task_registry import UI_TASKS

ROOT = Path(__file__).resolve().parents[1]


def cpu_inference_functions():
    source = ROOT / "scripts/inference_ui_defect_locany.py"
    names = {
        "TaskConfig", "TaskWork", "parse_optional_bool", "parse_args",
        "extract_image_paths_from_sample", "get_image_paths", "legacy_output_stem",
        "build_output_stems", "result_candidates", "result_already_exists", "prepare_work",
        "atomic_write_json", "build_manifest", "manifest_identity",
        "collect_existing_task_artifacts", "clear_existing_task_artifacts", "check_and_write_manifest",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = dict(globals(), PROMPT_TEMPLATE="Locate all the instances that match the following description: {label}.")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
    return namespace


class InferenceResumeTests(unittest.TestCase):
    def test_less_concurrency_reuses_results_and_retries_only_oom_image(self):
        functions = cpu_inference_functions()
        task = next(t for t in UI_TASKS if t.task_key == "synth_loneword")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest = root / "evaluation_manifest.json"
            manifest.write_text('{}', encoding="utf-8")
            images = [root / f"image-{index}.png" for index in range(3)]
            for path in images:
                path.write_bytes(b"CPU fixture; image decoding must not be required for resume")
            records = root / "test.jsonl"
            records.write_text("".join(json.dumps({"image": str(p)}) + "\n" for p in images), encoding="utf-8")
            functions["TASK_CONFIGS"] = [functions["TaskConfig"](task.task_key, str(records), task.class_id, task.task_key, task.prompt_label)]
            argv = ["inference", "--checkpoint", str(root / "checkpoint-1000"), "--input-dir", str(root),
                    "--output-dir", str(root / "pred"), "--eval-manifest", str(manifest), "--tasks", task.task_key,
                    "--attn-implementation", "sdpa", "--vision-attn-implementation", "flash_attention_2",
                    "--cuda-visible-devices", "3", "--save-raw-answer", "--inference-crop-mode", "detector_scan",
                    "--detector-crop-manifest", str(root / "detector_scan_crops.jsonl")]
            with mock.patch.object(sys, "argv", argv):
                args = functions["parse_args"]()
            args.input_dir, args.output_dir = Path(args.input_dir), Path(args.output_dir)
            args.detector_crop_manifest_digest = "fixture-geometry-digest"
            args.workers_per_gpu = 2
            work = functions["prepare_work"](args)[0]
            functions["check_and_write_manifest"](args)
            completed = []
            # A parsed illegal output is a completed scored prediction; an OOM
            # error sidecar alone must remain pending and must not become negative.
            for index, suffix in ((0, ""), (1, "_parse_error")):
                for subdirectory in ("", "gate", "raw"):
                    path = work.output_dir / subdirectory / f"image-{index}{suffix}.json"
                    functions["atomic_write_json"](path, [])
                    completed.append(path)
            functions["atomic_write_json"](work.output_dir / "errors/image-2_error.json", {"error": "CUDA out of memory"})
            saved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in completed}
            before = functions["manifest_identity"](functions["build_manifest"](args))
            args.workers_per_gpu = 1
            args.cuda_visible_devices = "0"
            after = functions["manifest_identity"](functions["build_manifest"](args))
            self.assertEqual(before, after)
            functions["check_and_write_manifest"](args)
            resumed = functions["prepare_work"](args)[0]
            self.assertEqual(resumed.skipped_existing, 2)
            self.assertEqual(resumed.pending_paths, [str(images[2])])
            self.assertEqual(saved, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in completed})
            functions["atomic_write_json"](work.output_dir / "image-2.json", [])
            self.assertEqual(functions["prepare_work"](args)[0].pending_paths, [])
            for field, value in (("checkpoint", str(root / "checkpoint-2000")), ("detector_crop_manifest_digest", "changed")):
                with self.subTest(field=field):
                    previous = getattr(args, field)
                    setattr(args, field, value)
                    with self.assertRaisesRegex(RuntimeError, "checkpoint/生成参数"):
                        functions["check_and_write_manifest"](args)
                    setattr(args, field, previous)


if __name__ == "__main__":
    unittest.main()
