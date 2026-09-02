from __future__ import annotations

import hashlib
import itertools
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import aggregate_ui5_train_rollouts as aggregate
import preflight_ui5_train_rollouts as preflight
import prepare_ui5_train_rollout_bundle as prepare
import render_ui5_train_rollout_gallery as gallery
from run_ui5_train_rollout_worker import load_module, score_prediction


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class UI5TrainRolloutTest(unittest.TestCase):
    def build_bundle(self, root: Path) -> Path:
        full = root / "full"
        audit = root / "audit"
        crop = audit / "crop"
        output = root / "bundle"
        full.mkdir(parents=True)
        image_path = root / "source.png"
        Image.new("RGB", (100, 100), "white").save(image_path)
        image_id = "img_test"
        labels = {
            "occlusion": "overlapping elements",
            "cropping": "cropped element",
            "text_overflow": "text overflow",
            "text_ellipsis": "abnormal text ellipsis",
            "content_missing": "missing content",
        }
        task_samples = []
        task_aware = []
        for task, label in labels.items():
            source_path = full / f"ui_{task}_train.jsonl"
            original = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Locate all the instances that match the following description: {label}.",
                    },
                    {
                        "from": "gpt",
                        "value": f"<ref>{label}</ref><box><100><400><300><600></box>",
                    },
                ],
                "image": str(image_path),
            }
            write_jsonl(source_path, [original])
            sample_id = f"sample_{task}"
            task_samples.append(
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "task": f"ui_{task}",
                    "width": 100,
                    "height": 100,
                    "gt_boxes": [[10, 40, 30, 60]],
                    "gt_boxes_1000": [[100, 400, 300, 600]],
                    "source_records": [
                        {"source_file": str(source_path), "line_no": 1}
                    ],
                    "same_task_polarity_conflict": False,
                }
            )
            task_aware.append(
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "task": f"ui_{task}",
                    "base_tiles": [[0, 0, 100, 50], [0, 50, 100, 100]],
                    "final_tiles": [[0, 0, 100, 100]],
                    "removed_gt_crossing_seams": [50],
                }
            )
        write_jsonl(
            audit / "manifest" / "unique_images.jsonl",
            [
                {
                    "image_id": image_id,
                    "content_id": "bytes",
                    "image_path": str(image_path),
                    "canonical_paths": [str(image_path)],
                    "basename": image_path.name,
                    "width": 100,
                    "height": 100,
                    "tasks": [f"ui_{task}" for task in labels],
                }
            ],
        )
        write_jsonl(
            audit / "manifest" / "task_samples.jsonl",
            task_samples,
        )
        crop.mkdir(parents=True)
        (crop / "base_scan_plans.json").write_text(
            json.dumps(
                {
                    image_id: {
                        "tiles": [[0, 0, 100, 50], [0, 50, 100, 100]],
                        "horizontal_seams": [50],
                    }
                }
            ),
            encoding="utf-8",
        )
        # These training-only fields exist in the source audit but must not
        # reach the portable rollout plan.
        write_jsonl(
            crop / "task_aware_manifest.jsonl",
            task_aware,
        )
        detector_path = audit / "detections" / "merged" / "detections.jsonl"
        write_jsonl(detector_path, [{"image_id": image_id}])
        digest = hashlib.blake2b(detector_path.read_bytes(), digest_size=16).hexdigest()
        (crop / "summary.json").write_text(
            json.dumps(
                {"input_state": {"detections_digest": "blake2b128:" + digest}}
            ),
            encoding="utf-8",
        )
        summary = prepare.build(
            SimpleNamespace(
                full_data=full,
                audit_root=audit,
                crop_root=crop,
                output_dir=output,
            )
        )
        self.assertEqual(summary["pipeline_coverage_failures"], 4)
        portable = json.loads((output / "base_scan_plans.json").read_text())
        self.assertIn("base_tiles", portable[image_id])
        self.assertNotIn("final_tiles", portable[image_id])
        self.assertFalse(any((output / "images").glob("*__y*.png")))
        return output

    def test_bundle_preflight_scoring_aggregate_and_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.build_bundle(root)
            summary, status = preflight.run(
                SimpleNamespace(
                    bundle_root=bundle,
                    diagnostics_dir=root / "diagnostics",
                    m31_checkpoint=root / "missing_m31",
                    crop_checkpoint=root / "missing_crop",
                    processor_candidate=[root / "missing_processor"],
                    m31_repo=PROJECT_ROOT,
                    crop_repo=PROJECT_ROOT,
                    require_runtime=False,
                )
            )
            self.assertEqual(status, 0)
            self.assertTrue(summary["bundle"]["complete"])
            self.assertTrue((root / "diagnostics" / "nastk_copy_commands.sh").is_file())

            # The Codex desktop's small CPU runtime omits SciPy.  Install a
            # tiny exact assignment stub only in this unit-test process; the
            # production worker still imports the formal scorer's SciPy
            # linear_sum_assignment and the H20 preflight requires SciPy.
            try:
                import scipy.optimize  # noqa: F401
            except ImportError:
                scipy_module = types.ModuleType("scipy")
                optimize_module = types.ModuleType("scipy.optimize")

                def linear_sum_assignment(cost: np.ndarray):
                    rows, cols = cost.shape
                    if rows <= cols:
                        best = min(
                            itertools.permutations(range(cols), rows),
                            key=lambda chosen: sum(cost[row, col] for row, col in enumerate(chosen)),
                        )
                        return np.arange(rows), np.array(best)
                    chosen_rows = min(
                        itertools.permutations(range(rows), cols),
                        key=lambda chosen: sum(cost[row, col] for col, row in enumerate(chosen)),
                    )
                    return np.array(chosen_rows), np.arange(cols)

                optimize_module.linear_sum_assignment = linear_sum_assignment
                scipy_module.optimize = optimize_module
                sys.modules["scipy"] = scipy_module
                sys.modules["scipy.optimize"] = optimize_module

            scorer = load_module(
                PROJECT_ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py",
                "test_ui5_formal_scorer",
            )
            wrong = score_prediction(
                scorer,
                [[10, 10, 30, 30]],
                [[70, 70, 90, 90]],
                "defect",
                0.1,
                (100, 100),
            )
            self.assertEqual(wrong["image_confusion"], "TP")
            self.assertEqual(wrong["error_type"], "LOC_WRONG")
            self.assertEqual((wrong["TP_box"], wrong["FP_box"], wrong["FN_box"]), (0, 1, 1))

            output = root / "rollouts"
            output.mkdir()
            sample = json.loads(
                (bundle / "manifest" / "task_samples.jsonl").read_text().splitlines()[0]
            )
            for model in ("m31", "crop"):
                for rollout in range(4):
                    correct = rollout < (2 if model == "m31" else 1)
                    pred = sample["gt_global"] if correct else [[70, 70, 90, 90]]
                    score = score_prediction(
                        scorer, sample["gt_global"], pred, "defect", 0.1, (100, 100)
                    )
                    crop_outputs = []
                    if model == "crop":
                        crop_outputs = [
                            {
                                "crop_id": crop_id,
                                "crop_xyxy": [0, 0, 100, 50] if index == 0 else [0, 50, 100, 100],
                                "gt_local": [],
                                "raw_output": "<box>none</box>",
                                "parse_status": "ok",
                                "exact_correct": True,
                            }
                            for index, crop_id in enumerate(sample["crop_ids"])
                        ]
                    raw = {
                        "model_id": model,
                        "checkpoint": f"/{model}/checkpoint",
                        "git_commit": "deadbeef",
                        "baseline_git_commit": "5d7a313" if model == "m31" else "945ce39",
                        "rollout_id": rollout,
                        "seed": 100 + rollout,
                        "generation_config": {"mode": "hybrid", "do_sample": True},
                        "record_id": sample["record_id"],
                        "sample_id": sample["sample_id"],
                        "source_image_id": sample["source_image_id"],
                        "image_id": sample["source_image_id"],
                        "image_relpath": sample["image_relpath"],
                        "image_size": {"width": 100, "height": 100},
                        "task": sample["task"],
                        "source_records": sample["source_records"],
                        "original_training_record": sample["original_training_record"],
                        "prompt": sample["prompt"],
                        "gt_global": sample["gt_global"],
                        "pred_global": pred,
                        "parse_status": "defect",
                        "latency_seconds": 1.0,
                        "crop_outputs": crop_outputs,
                        "pipeline_coverage_failure": True,
                        "annotation_anomaly": False,
                        "coordinate_transform_anomaly": False,
                        **score,
                    }
                    write_jsonl(
                        output / "raw" / model / f"rollout_{rollout}" / "part-00000.jsonl",
                        [raw],
                    )
                    write_jsonl(
                        output / "progress" / model / f"rollout_{rollout}.jsonl",
                        [
                            {
                                "status": "completed",
                                "completed": 1,
                                "total": 1,
                                "elapsed_seconds": 1.0,
                                "throughput_samples_per_second": 1.0,
                                "remaining_seconds": 0.0,
                                "estimated_completion": None,
                                "errors": 0,
                            }
                        ],
                    )
            analysis = aggregate.run(
                SimpleNamespace(output_root=output, bundle_root=bundle, repo_root=PROJECT_ROOT)
            )
            self.assertEqual(analysis["common_image_task_intersection"], 1)
            self.assertTrue((output / "reports" / "ui5_train_rollout_analysis.xlsx").is_file())
            rendered = gallery.render(
                SimpleNamespace(output_root=output, bundle_root=bundle, panel_long_side=160)
            )
            self.assertGreater(rendered["rendered"], 0)
            self.assertTrue((output / "visualizations" / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
