from __future__ import annotations

import json
import hashlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_ui5_curriculum_evaluation as evaluation  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_frozen_and_curriculum(
    root: Path,
    hard_groups: int,
    *,
    hard_sample_ids: list[str] | None = None,
    hard_groups_path: Path | None = None,
) -> tuple[Path, Path]:
    frozen = root / "frozen"
    frozen.mkdir(parents=True)
    complete8 = frozen / "complete8.jsonl"
    difficulty = frozen / "sample_difficulty.jsonl"
    summary_path = frozen / "summary.json"
    sample_ids = hard_sample_ids or [f"sample-{index}" for index in range(hard_groups)]
    if len(sample_ids) != hard_groups:
        raise ValueError("hard_sample_ids length differs from hard_groups")
    selection_rows = [
        {
            "sample_id": sample_id,
            "crop_correct_count": 0,
            "grpo_source_eligible": True,
            "pipeline_coverage_failure": False,
            "annotation_anomaly": False,
            "coordinate_transform_anomaly": False,
        }
        for sample_id in sample_ids
    ]
    write_jsonl(complete8, selection_rows)
    write_jsonl(difficulty, selection_rows)
    sources: list[dict] = []
    source_set_sha256 = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    hard_ids = sorted(row["sample_id"] for row in selection_rows)
    hard_ids_digest = evaluation._canonical_json_sha256(hard_ids)
    summary = {
        "schema_version": 1,
        "source_set_sha256": source_set_sha256,
        "unique_complete8_samples": hard_groups,
        "formal_eligible_groups": hard_groups,
        "formal_crop_hard_groups": hard_groups,
        "formal_crop_hard_sample_ids": hard_ids,
        "formal_crop_hard_sample_ids_sha256": hard_ids_digest,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    inventoried = []
    for path in (complete8, difficulty, summary_path):
        inventoried.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "jsonl_records": (
                    len(path.read_text(encoding="utf-8").splitlines())
                    if path.suffix == ".jsonl"
                    else None
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact_type": "ui5_frozen_rollout_selection",
        "status": "complete",
        "immutable": True,
        "success_marker": "_SUCCESS",
        "training_input_policy": "resolve_once_at_run_start_no_hot_reload",
        "technical_policy": "complete8_and_error_free_routes_only",
        "source_set_sha256": source_set_sha256,
        "sources": sources,
        "files": inventoried,
    }
    (frozen / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (frozen / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    curriculum_dir = root / "curriculum"
    curriculum_dir.mkdir()
    curriculum_path = curriculum_dir / "curriculum_manifest.json"
    curriculum = {
        "schema_version": 1,
        "expected_hard_groups": hard_groups,
        "hard_groups": hard_groups,
        "inputs": {
            "rollout_difficulty_sha256": _sha256(complete8),
            "frozen_selection_summary": {
                "path": str(summary_path.resolve()),
                "sha256": _sha256(summary_path),
                "formal_crop_hard_groups": hard_groups,
                "formal_crop_hard_sample_ids_sha256": hard_ids_digest,
                "membership_source": "explicit_summary_ids",
                "authoritative_complete8_path": str(complete8.resolve()),
                "authoritative_complete8_sha256": _sha256(complete8),
            },
        },
    }
    curriculum["identity_digest"] = evaluation._canonical_json_sha256(curriculum)
    curriculum_path.write_text(json.dumps(curriculum), encoding="utf-8")
    inventoried_hard_groups = hard_groups_path or complete8
    (curriculum_dir / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "complete": True,
                "identity_digest": curriculum["identity_digest"],
                "files": {
                    "hard_groups.jsonl": {
                        "bytes": inventoried_hard_groups.stat().st_size,
                        "sha256": _sha256(inventoried_hard_groups),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return frozen, curriculum_path


def frozen_crop_rollouts() -> list[dict]:
    return [
        {
            "model_id": "crop",
            "rollout_id": rollout_id,
            "seed": seed,
            "exact_correct": False,
        }
        for rollout_id, seed in enumerate(evaluation.FORMAL_ROLLOUT_SEEDS)
    ]


def load_inference_module(module_name: str):
    """Import the inference worker without requiring the GPU test environment."""

    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.float16 = object()
    torch_stub.float32 = object()
    torch_stub.Tensor = object
    torch_stub.manual_seed = lambda seed: None
    torch_stub.is_tensor = lambda value: False
    torch_stub.equal = lambda left, right: False
    torch_stub.inference_mode = lambda: (lambda function: function)
    torch_stub.cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: None,
        device_count=lambda: 0,
        empty_cache=lambda: None,
    )
    transformers_stub = types.ModuleType("transformers")
    for name in ("AutoConfig", "AutoModel", "AutoProcessor", "AutoTokenizer"):
        setattr(transformers_stub, name, type(name, (), {}))
    relation_stub = types.ModuleType("eaglevl.model.locany.relation_modules")
    relation_stub.UI_RELATION_PROMPT_SPECS = tuple(
        SimpleNamespace(task_name=task, prompt_label=task.replace("_", " "))
        for task in evaluation.TASKS
    )
    spec = importlib.util.spec_from_file_location(
        module_name, SCRIPTS / "inference_ui_defect_locany.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "torch": torch_stub,
            "transformers": transformers_stub,
            "eaglevl.model.locany.relation_modules": relation_stub,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class CurriculumEvaluationTopologyTest(unittest.TestCase):
    def make_args(self, root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            python="python",
            worker_script=root / "worker.py",
            scorer_script=root / "scorer.py",
            checkpoint=root / "checkpoint",
            processor_path=root / "processor",
            input_dir=root / "input",
            output_dir=root / "evaluation" / "step-000200",
            rollout_bundle_root=root / "bundle",
            dtype="bf16",
            attn_implementation="sdpa",
            vision_attn_implementation="flash_attention_2",
            generation_mode="hybrid",
            max_new_tokens=4096,
            n_future_tokens=6,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.1,
            seed=42,
            greedy=False,
            relation_gate_mode="observe",
            relation_gate_threshold=None,
            inference_crop_mode="detector_scan",
            detector_crop_manifest=root / "detector.jsonl",
            tile_max_count=10,
            tile_target_long_side=1600,
            tile_overlap_ratio=0.10,
            tile_nms_iou=0.50,
            evaluator_iou_threshold=0.10,
            max_images_per_task=0,
            overwrite=False,
        )

    def test_worker_specs_use_fixed_two_three_mapping_and_exact_ui_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            hard_counts = {task: index for index, task in enumerate(evaluation.TASKS)}
            specs = evaluation.build_worker_specs(
                args,
                ("0", "1"),
                hard_counts,
                root / "resolved.jsonl",
                root / "identity.json",
            )
            self.assertEqual(len(specs), 5)
            self.assertEqual(
                {spec.task: spec.physical_gpu for spec in specs},
                {
                    "occlusion": "0",
                    "cropping": "0",
                    "text_overflow": "1",
                    "text_ellipsis": "1",
                    "content_missing": "1",
                },
            )
            for spec in specs:
                command = list(spec.command)
                self.assertEqual(spec.output_dir.name, f"ui_{spec.task}")
                self.assertEqual(
                    command[command.index("--single-task-output-dir") + 1],
                    str(spec.output_dir),
                )
                self.assertEqual(command[command.index("--device") + 1], "cuda:0")
                self.assertEqual(command[command.index("--checkpoint") + 1], str(args.checkpoint))
                self.assertEqual(
                    command[command.index("--processor-path") + 1], str(args.processor_path)
                )
                seed_index = command.index("--hard-rollout-seeds")
                self.assertEqual(
                    command[seed_index + 1 : seed_index + 5],
                    [str(seed) for seed in evaluation.FORMAL_ROLLOUT_SEEDS],
                )
                for explicit in (
                    "--dtype",
                    "--attn-implementation",
                    "--vision-attn-implementation",
                    "--generation-mode",
                    "--max-new-tokens",
                    "--n-future-tokens",
                    "--temperature",
                    "--top-p",
                    "--top-k",
                    "--repetition-penalty",
                    "--tile-nms-iou",
                    "--rollout-scorer-script",
                    "--evaluation-identity-file",
                ):
                    self.assertIn(explicit, command)

    def test_curriculum_identity_rejects_wrong_frozen_summary_or_id_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen, curriculum_path = make_frozen_and_curriculum(root, 3)
            selection = evaluation.resolve_frozen_selection(frozen)
            original = json.loads(curriculum_path.read_text(encoding="utf-8"))
            success_path = curriculum_path.parent / "_SUCCESS.json"

            for field, message in (
                ("sha256", "summary SHA-256"),
                ("formal_crop_hard_sample_ids_sha256", "hard-ID digest"),
            ):
                with self.subTest(field=field):
                    changed = json.loads(json.dumps(original))
                    changed["inputs"]["frozen_selection_summary"][field] = "0" * 64
                    changed.pop("identity_digest", None)
                    changed["identity_digest"] = evaluation._canonical_json_sha256(
                        changed
                    )
                    curriculum_path.write_text(json.dumps(changed), encoding="utf-8")
                    success_path.write_text(
                        json.dumps(
                            {
                                "complete": True,
                                "identity_digest": changed["identity_digest"],
                            }
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        evaluation.curriculum_evaluation_identity(
                            curriculum_path,
                            selection=selection,
                            expected_hard_groups=3,
                        )

    def test_identity_guard_rejects_matcher_change_after_worker_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            processor = root / "processor"
            bundle = root / "bundle"
            checkpoint.mkdir()
            processor.mkdir()
            bundle.mkdir()
            (checkpoint / "weights.bin").write_bytes(b"weights")
            (processor / "tokenizer.json").write_text("{}", encoding="utf-8")

            worker = root / "worker.py"
            orchestrator = root / "orchestrator.py"
            crop = root / "tiling.py"
            matcher = root / "matching.py"
            scorer = root / "scorer.py"
            hard = root / "hard.jsonl"
            resolved = root / "resolved.jsonl"
            detector = root / "detector.jsonl"
            plans = bundle / "base_scan_plans.json"
            curriculum = root / "curriculum.json"
            curriculum_success = root / "curriculum_success.json"
            selection_files = {
                name: root / f"selection_{name}"
                for name in ("manifest", "summary", "complete8", "success")
            }
            eval_gt = root / "eval_gt.jsonl"
            for path in (
                worker,
                orchestrator,
                crop,
                matcher,
                scorer,
                hard,
                resolved,
                detector,
                plans,
                curriculum,
                curriculum_success,
                eval_gt,
                *selection_files.values(),
            ):
                path.write_text(path.name, encoding="utf-8")

            args = SimpleNamespace(
                checkpoint=checkpoint,
                processor_path=processor,
                worker_script=worker,
                scorer_script=scorer,
                hard_groups_jsonl=hard,
                rollout_bundle_root=bundle,
                detector_crop_manifest=detector,
            )
            identity = {
                "candidate": evaluation.directory_inventory(checkpoint),
                "processor": evaluation.directory_inventory(processor),
                "orchestrator": {
                    "path": str(orchestrator),
                    "sha256": evaluation.file_sha256(orchestrator),
                },
                "worker_script": {"sha256": evaluation.file_sha256(worker)},
                "crop_and_merge": {
                    "implementation_path": str(crop),
                    "implementation_sha256": evaluation.file_sha256(crop),
                    "detector_crop_manifest_sha256": evaluation.file_sha256(detector),
                },
                "evaluator": {
                    "sha256": evaluation.file_sha256(scorer),
                    "matching_implementation_path": str(matcher),
                    "matching_implementation_sha256": evaluation.file_sha256(matcher),
                },
                "hard_rollout": {
                    "source_sha256": evaluation.file_sha256(hard),
                    "resolved_source": str(resolved),
                    "resolved_source_sha256": evaluation.file_sha256(resolved),
                    "base_scan_plans_sha256": evaluation.file_sha256(plans),
                },
                "curriculum": {
                    "path": str(curriculum),
                    "sha256": evaluation.file_sha256(curriculum),
                    "success_path": str(curriculum_success),
                    "success_sha256": evaluation.file_sha256(curriculum_success),
                },
                "frozen_selection": {
                    **{
                        f"{name}_path": str(path)
                        for name, path in selection_files.items()
                    },
                    **{
                        f"{name}_sha256": evaluation.file_sha256(path)
                        for name, path in selection_files.items()
                    },
                },
                "evaluation_inputs": {
                    "occlusion": {
                        "path": str(eval_gt),
                        "sha256": evaluation.file_sha256(eval_gt),
                    }
                },
            }

            evaluation.validate_identity_unchanged(args, identity)
            matcher.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "metric matching implementation"):
                evaluation.validate_identity_unchanged(args, identity)
            matcher.write_text(matcher.name, encoding="utf-8")
            orchestrator.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "evaluation orchestrator"):
                evaluation.validate_identity_unchanged(args, identity)

    def test_launcher_starts_all_five_before_first_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = [
                evaluation.WorkerSpec(
                    task=task,
                    physical_gpu="0" if index < 2 else "1",
                    output_dir=root / f"ui_{task}",
                    summary_path=root / "_worker_summaries" / f"{task}.json",
                    log_path=root / "_worker_logs" / f"{task}.log",
                    command=("fake", "--tasks", task),
                )
                for index, task in enumerate(evaluation.TASKS)
            ]
            processes: list[object] = []
            environments: list[dict[str, str]] = []

            class FakeProcess:
                def __init__(self, command, **kwargs):
                    self.command = command
                    self.pid = 1000 + len(processes)
                    processes.append(self)
                    environments.append(kwargs["env"])

                def wait(self):
                    if len(processes) != 5:
                        raise AssertionError("a worker was awaited before all five were launched")
                    return 0

                def poll(self):
                    if len(processes) != 5:
                        raise AssertionError("a worker was polled before all five were launched")
                    return 0

                def terminate(self):
                    return None

            results = evaluation.launch_workers(
                specs,
                project_root=root,
                popen_factory=FakeProcess,
            )
            self.assertEqual(len(results), 5)
            self.assertEqual(
                [env["CUDA_VISIBLE_DEVICES"] for env in environments].count("0"), 2
            )
            self.assertEqual(
                [env["CUDA_VISIBLE_DEVICES"] for env in environments].count("1"), 3
            )
            self.assertEqual(len({row["pid"] for row in results}), 5)

    def test_launcher_heartbeat_reports_all_worker_states_without_serial_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = [
                evaluation.WorkerSpec(
                    task=task,
                    physical_gpu="0" if index < 2 else "1",
                    output_dir=root / f"ui_{task}",
                    summary_path=root / "_worker_summaries" / f"{task}.json",
                    log_path=root / "_worker_logs" / f"{task}.log",
                    command=("fake", "--tasks", task),
                )
                for index, task in enumerate(evaluation.TASKS)
            ]

            class SlowFakeProcess:
                instances: list["SlowFakeProcess"] = []

                def __init__(self, command, **kwargs):
                    self.command = command
                    self.pid = 2000 + len(self.instances)
                    self.poll_count = 0
                    self.instances.append(self)

                def poll(self):
                    self.poll_count += 1
                    return None if self.poll_count < 3 else 0

                def wait(self):
                    return 0

                def terminate(self):
                    return None

            output = io.StringIO()
            with redirect_stdout(output):
                results = evaluation.launch_workers(
                    specs,
                    project_root=root,
                    popen_factory=SlowFakeProcess,
                    # The CLI enforces a positive production interval.  Use a
                    # deliberately elapsed deadline in this direct unit call
                    # so the assertion is independent of Windows timer ticks.
                    heartbeat_seconds=-1.0,
                )
            heartbeat_lines = [
                line
                for line in output.getvalue().splitlines()
                if line.startswith("[UI5 HEARTBEAT] ")
            ]
            self.assertGreaterEqual(len(heartbeat_lines), 1)
            heartbeat = json.loads(heartbeat_lines[0].split(" ", 2)[2])
            self.assertEqual(heartbeat["event"], "ui5_worker_heartbeat")
            self.assertEqual(len(heartbeat["workers"]), 5)
            self.assertEqual({row["state"] for row in heartbeat["workers"]}, {"running"})
            self.assertEqual([row["task"] for row in results], list(evaluation.TASKS))


class HardGroupValidationTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> tuple[Path, Path]:
        bundle = root / "bundle"
        image = bundle / "images" / "sample.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"fake-image")
        (bundle / "base_scan_plans.json").write_text(
            json.dumps(
                {
                    "image-1": {
                        "image_id": "image-1",
                        "width": 100,
                        "height": 80,
                        "base_tiles": [[0, 0, 100, 40], [0, 40, 100, 80]],
                        "geometry_digest": "geometry",
                        "gt_used": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        return bundle, image

    def test_resolves_only_authoritative_crop_zero_of_four_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, _ = self.make_bundle(root)
            manifest = root / "hard.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "record_id": "record-1",
                        "sample_id": "sample-1",
                        "source_image_id": "image-1",
                        "task": "occlusion",
                        "image_relpath": "images/sample.png",
                        "prompt": "Locate occlusion.",
                        "gt_global": [[1, 2, 3, 4]],
                        "positive": True,
                        "crop_correct_count": 0,
                        "crop_complete4": True,
                    }
                ],
            )
            rows, counts, plan_path = evaluation.resolve_hard_groups(manifest, bundle, 1)
            self.assertEqual(counts["occlusion"], 1)
            self.assertEqual(rows[0]["_base_tiles"], [[0, 0, 100, 40], [0, 40, 100, 80]])
            self.assertEqual(Path(rows[0]["_resolved_image_path"]).name, "sample.png")
            self.assertEqual(plan_path, bundle / "base_scan_plans.json")

            bad = dict(rows[0])
            for key in list(bad):
                if key.startswith("_"):
                    bad.pop(key)
            bad["crop_correct_count"] = 1
            write_jsonl(manifest, [bad])
            with self.assertRaisesRegex(ValueError, "not a crop 0/4"):
                evaluation.resolve_hard_groups(manifest, bundle, 1)

    def test_hard_transition_reuses_exact_frozen_crop_rollout_seeds(self) -> None:
        rows = [
            {
                "record_id": "hard-1",
                "rollouts": {
                    "crop": [
                        {
                            "model_id": "crop",
                            "rollout_id": rollout_id,
                            "seed": seed,
                        }
                        for rollout_id, seed in enumerate(
                            evaluation.FORMAL_ROLLOUT_SEEDS
                        )
                    ]
                },
            },
            {
                "record_id": "hard-2",
                "rollouts": {
                    "crop": [
                        {
                            "model_id": "crop",
                            "rollout_id": rollout_id,
                            "seed": seed,
                        }
                        for rollout_id, seed in enumerate(
                            evaluation.FORMAL_ROLLOUT_SEEDS
                        )
                    ]
                },
            },
        ]
        self.assertEqual(
            evaluation.paired_crop_rollout_seeds(rows),
            evaluation.FORMAL_ROLLOUT_SEEDS,
        )

        rows[1]["rollouts"]["crop"][3]["seed"] = 45
        with self.assertRaisesRegex(ValueError, "different crop rollout seeds"):
            evaluation.paired_crop_rollout_seeds(rows)

    def test_hard_transition_rejects_unpaired_seed_identity(self) -> None:
        rows = [
            {
                "record_id": "hard-1",
                "rollouts": {
                    "crop": [
                        {
                            "model_id": "crop",
                            "rollout_id": rollout_id,
                            "seed": 42 + rollout_id,
                        }
                        for rollout_id in range(4)
                    ]
                },
            }
        ]
        with self.assertRaisesRegex(ValueError, "formal baseline identity"):
            evaluation.paired_crop_rollout_seeds(rows)

    def test_content_missing_never_requires_a_crop_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, _ = self.make_bundle(root)
            manifest = root / "hard.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "record_id": "record-global",
                        "sample_id": "sample-global",
                        "task": "content_missing",
                        "image_relpath": "images/sample.png",
                        "prompt": "Locate missing content.",
                        "gt_global": [],
                        "positive": False,
                        "crop_correct_count": 0,
                        "crop_complete4": True,
                        "rollouts": {"crop": frozen_crop_rollouts()},
                    }
                ],
            )
            rows, _, _ = evaluation.resolve_hard_groups(manifest, bundle, 1)
            self.assertIsNone(rows[0]["_base_tiles"])
            self.assertIsNone(rows[0]["_base_plan_width"])

    def test_content_missing_rollout4_reuses_direct_full_image_inferencer(self) -> None:
        module = load_inference_module("ui5_curriculum_inference_direct_test")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "content.png"
            Image.new("RGB", (31, 17), "white").save(image_path)
            with Image.open(image_path) as check_image:
                self.assertEqual(check_image.size, (31, 17))
            self.assertIs(module.Image, Image)
            args = SimpleNamespace(
                hard_groups_jsonl=str(root / "resolved.jsonl"),
                rollout_scorer_script=str(root / "scorer.py"),
                hard_rollout_output_dir=str(root / "rollout4"),
                hard_rollout_seeds=list(evaluation.FORMAL_ROLLOUT_SEEDS),
                evaluation_identity_digest="identity",
                inference_crop_mode="detector_scan",
                compat_confidence=None,
                hard_rollout_iou_threshold=0.1,
                tile_max_count=10,
                tile_target_long_side=1600,
                tile_overlap_ratio=0.1,
                tile_nms_iou=0.5,
            )
            rows = [
                {
                    "record_id": "hard-content",
                    "sample_id": "hard-content",
                    "source_image_id": None,
                    "task": "content_missing",
                    "image_relpath": "images/content.png",
                    "_resolved_image_path": str(image_path),
                    "prompt": "Locate missing content.",
                    "gt_global": [],
                    "rollouts": {"crop": frozen_crop_rollouts()},
                    "_base_tiles": None,
                    "_base_plan_digest": None,
                }
            ]
            class FakeInferencer:
                def __init__(self) -> None:
                    self.calls: list[tuple[tuple[int, int], str]] = []
                    self.last_ui_diagnostics = {"available": False}

                def predict(self, *, image, question):
                    self.calls.append((image.size, question))
                    return "<box>none</box>"

            inferencer = FakeInferencer()

            score = {
                "matched_pairs": [],
                "TP_box": 0,
                "FP_box": 0,
                "FN_box": 0,
                "image_confusion": "TN",
                "error_type": "TN",
                "exact_correct": True,
            }
            with mock.patch.object(
                module, "_load_python_module", return_value=object()
            ), mock.patch.object(
                module, "predict_with_lossless_tiles"
            ) as tiled, mock.patch.object(
                module, "_score_hard_prediction", return_value=score
            ):
                summary = module.run_hard_rollouts(
                    args,
                    inferencer,
                    module.TASK_BY_NAME["content_missing"],
                    rows,
                )
            tiled.assert_not_called()
            self.assertEqual(len(inferencer.calls), 4)
            self.assertEqual(
                inferencer.calls,
                [((31, 17), "Locate missing content.")] * 4,
            )
            self.assertEqual(summary["attempt_count"], 4)
            self.assertEqual(summary["groups_improved"], 1)
            self.assertEqual(summary["baseline_correct_count"], 0)
            self.assertEqual(summary["candidate_correct_count"], 4)
            self.assertEqual(
                summary["comparison"],
                "paired_frozen_crop_baseline_to_candidate",
            )
            for rollout_id in range(4):
                rollout_path = root / "rollout4" / f"rollout_{rollout_id}.jsonl"
                self.assertTrue(rollout_path.is_file())
                row = json.loads(rollout_path.read_text(encoding="utf-8"))
                self.assertEqual(row["crop_mode"], "content_missing_direct_full_image")
                self.assertEqual(row["tiles"], [])
                self.assertEqual(row["tile_count"], 1)
                self.assertFalse(row["baseline_exact_correct"])
                self.assertTrue(row["candidate_exact_correct"])
                self.assertEqual(row["attempt_transition"], "incorrect_to_correct")
            group = json.loads(
                (root / "rollout4" / "groups.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(group["baseline_correct_count"], 0)
            self.assertEqual(group["candidate_correct_count"], 4)
            self.assertTrue(
                all(
                    item["transition"] == "incorrect_to_correct"
                    and item["baseline"]["exact_correct"] is False
                    and item["candidate"]["exact_correct"] is True
                    for item in group["rollout_results"]
                )
            )

    def test_content_missing_main_ui5_pass_bypasses_detector_tiles(self) -> None:
        module = load_inference_module("ui5_curriculum_inference_main_direct_test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "content.png"
            Image.new("RGB", (31, 17), "white").save(image_path)
            output_dir = root / "ui_content_missing"
            work = module.TaskWork(
                config=module.TASK_BY_NAME["content_missing"],
                jsonl_path=root / "input.jsonl",
                output_dir=output_dir,
                image_paths=[str(image_path)],
                output_stems={str(image_path): "content"},
                pending_paths=[str(image_path)],
                skipped_existing=0,
            )
            args = SimpleNamespace(
                overwrite=False,
                seed=42,
                inference_crop_mode="detector_scan",
                detector_scan_index={},
                compat_confidence=None,
                print_raw_answer=False,
                tag_filename=False,
                save_raw_answer=False,
                save_visualization=False,
                relation_gate_mode="observe",
                relation_gate_threshold=None,
                fail_fast=True,
            )

            class FakeInferencer:
                def __init__(self) -> None:
                    self.calls: list[tuple[tuple[int, int], str]] = []
                    self.last_ui_diagnostics = {"available": False}

                def predict(self, *, image, question):
                    self.calls.append((image.size, question))
                    return "unparseable model response"

            inferencer = FakeInferencer()
            with mock.patch.object(module, "predict_with_lossless_tiles") as tiled:
                stats = module.run_one_task(args, inferencer, work)
            tiled.assert_not_called()
            self.assertEqual(
                inferencer.calls,
                [((31, 17), module.TASK_BY_NAME["content_missing"].prompt)],
            )
            self.assertEqual(stats["processed"], 1)
            self.assertEqual(stats["ok"], 0)
            self.assertEqual(stats["parse_error"], 1)
            self.assertEqual(stats["inference_error"], 0)
            prediction = output_dir / "content_parse_error.json"
            self.assertTrue(prediction.is_file())
            self.assertIsNone(json.loads(prediction.read_text(encoding="utf-8")))

    def test_hard_rollout_parse_errors_are_scored_not_runtime_failures(self) -> None:
        module = load_inference_module("ui5_curriculum_inference_parse_test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "content.png"
            Image.new("RGB", (31, 17), "white").save(image_path)
            args = SimpleNamespace(
                hard_groups_jsonl=str(root / "resolved.jsonl"),
                rollout_scorer_script=str(root / "scorer.py"),
                hard_rollout_output_dir=str(root / "rollout4"),
                hard_rollout_seeds=list(evaluation.FORMAL_ROLLOUT_SEEDS),
                evaluation_identity_digest="identity",
                inference_crop_mode="detector_scan",
                compat_confidence=None,
                hard_rollout_iou_threshold=0.1,
                tile_max_count=10,
                tile_target_long_side=1600,
                tile_overlap_ratio=0.1,
                tile_nms_iou=0.5,
            )
            rows = [
                {
                    "record_id": "hard-content",
                    "sample_id": "hard-content",
                    "source_image_id": None,
                    "task": "content_missing",
                    "image_relpath": "images/content.png",
                    "_resolved_image_path": str(image_path),
                    "prompt": "Locate missing content.",
                    "gt_global": [],
                    "rollouts": {"crop": frozen_crop_rollouts()},
                    "_base_tiles": None,
                    "_base_plan_digest": None,
                }
            ]

            class ParseErrorInferencer:
                last_ui_diagnostics = {"available": False}

                def predict(self, *, image, question):
                    return "unparseable model response"

            with mock.patch.object(
                module, "_load_python_module", return_value=object()
            ), mock.patch.object(module, "predict_with_lossless_tiles") as tiled:
                summary = module.run_hard_rollouts(
                    args,
                    ParseErrorInferencer(),
                    module.TASK_BY_NAME["content_missing"],
                    rows,
                )
            tiled.assert_not_called()
            self.assertEqual(summary["parse_error_count"], 4)
            self.assertEqual(summary["runtime_error_count"], 0)
            self.assertEqual(summary["error_count"], 0)
            self.assertEqual(summary["groups_still_hard"], 1)
            for rollout_id in range(4):
                row = json.loads(
                    (root / "rollout4" / f"rollout_{rollout_id}.jsonl").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(row["parse_status"], "parse_error")
                self.assertFalse(row["exact_correct"])
                self.assertIsNone(row["runtime_error"])
                self.assertEqual(
                    row["attempt_transition"], "incorrect_to_incorrect"
                )


class EvaluationBarrierAndMetricsTest(unittest.TestCase):
    def test_any_worker_failure_blocks_the_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            processor = root / "processor"
            input_dir = root / "input"
            bundle = root / "bundle"
            for directory in (checkpoint, processor, input_dir, bundle / "images"):
                directory.mkdir(parents=True, exist_ok=True)
            worker = root / "worker.py"
            scorer = root / "scorer.py"
            worker.write_text("# fake\n", encoding="utf-8")
            scorer.write_text("# fake\n", encoding="utf-8")
            image = bundle / "images" / "sample.png"
            image.write_bytes(b"fake")
            (bundle / "base_scan_plans.json").write_text("{}", encoding="utf-8")
            eval_image = root / "eval.png"
            eval_image.write_bytes(b"fake")
            for task in evaluation.TASKS:
                write_jsonl(
                    input_dir / evaluation.TASK_GT_FILE[task],
                    [{"image": str(eval_image)}],
                )
            hard = root / "hard.jsonl"
            write_jsonl(
                hard,
                [
                    {
                        "record_id": "hard-global",
                        "sample_id": "hard-global",
                        "task": "content_missing",
                        "image_relpath": "images/sample.png",
                        "prompt": "Locate missing content.",
                        "gt_global": [],
                        "positive": False,
                        "crop_correct_count": 0,
                        "crop_complete4": True,
                        "rollouts": {"crop": frozen_crop_rollouts()},
                    }
                ],
            )
            frozen, curriculum = make_frozen_and_curriculum(
                root,
                1,
                hard_sample_ids=["hard-global"],
                hard_groups_path=hard,
            )
            args = evaluation.parse_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--processor-path",
                    str(processor),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(root / "evaluation" / "step-000200"),
                    "--hard-groups-jsonl",
                    str(hard),
                    "--rollout-bundle-root",
                    str(bundle),
                    "--curriculum-manifest",
                    str(curriculum),
                    "--frozen-selection",
                    str(frozen),
                    "--expected-hard-groups",
                    "1",
                    "--inference-crop-mode",
                    "full_image",
                    "--worker-script",
                    str(worker),
                    "--scorer-script",
                    str(scorer),
                    "--project-root",
                    str(root),
                    "--fake-worker",
                ]
            )
            results = [
                {
                    "task": task,
                    "physical_gpu": "0" if index < 2 else "1",
                    "return_code": 17 if task == "cropping" else 0,
                    "log_path": str(root / f"{task}.log"),
                }
                for index, task in enumerate(evaluation.TASKS)
            ]
            with mock.patch.object(evaluation, "launch_workers", return_value=results), mock.patch.object(
                evaluation, "run_scorer"
            ) as scorer_call:
                with self.assertRaisesRegex(RuntimeError, "scoring is blocked"):
                    evaluation.run(args)
                scorer_call.assert_not_called()

    def test_enriches_macro_with_pooled_micro_and_joint(self) -> None:
        tasks = {
            task: {
                "image": {
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                    "accuracy": 0.5,
                    "tp": 1,
                    "fp": 1,
                    "fn": 1,
                    "tn": 1,
                },
                "bbox": {
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                    "count_accuracy": 0.5,
                    "tp": 2,
                    "fp": 1,
                    "fn": 1,
                },
            }
            for task in evaluation.TASKS
        }
        result = evaluation.enrich_metrics(
            {"schema_version": 1, "tasks": tasks, "macro": {"image": {"f1": 0.4}, "bbox": {"f1": 0.6}}}
        )
        self.assertAlmostEqual(result["overall"]["joint_score"], 0.5)
        self.assertEqual(result["micro"]["image"]["tp"], 5)
        self.assertEqual(result["micro"]["bbox"]["tp"], 10)

        args = SimpleNamespace(step=200, total_steps=1200)
        status = evaluation.evaluation_status_payload(
            args=args,
            metrics=result,
            evaluation_seconds=12.5,
            hard_summary={
                "group_count": 114,
                "groups_improved": 9,
                "groups_still_hard": 105,
                "parse_error_count": 2,
                "runtime_error_count": 0,
            },
        )
        self.assertEqual(status["phase"], 1)
        self.assertEqual(status["curriculum_target"]["hard_ratio"], 0.60)
        self.assertEqual(tuple(status["tasks"]), tuple(f"ui_{task}" for task in evaluation.TASKS))
        output = io.StringIO()
        with redirect_stdout(output):
            evaluation.print_evaluation_status(status)
        self.assertEqual(output.getvalue().count("[UI5 TASK METRICS]"), 5)
        self.assertIn("samples=4 invalid=0", output.getvalue())
        self.assertIn(
            "[UI5 HARD TRANSITION] step=200 groups=114 improved=9 "
            "still_hard=105 parse_errors=2 runtime_errors=0",
            output.getvalue(),
        )
        self.assertIn("samples=20 invalid=0", output.getvalue())
        structured = next(
            line.split(" ", 2)[2]
            for line in output.getvalue().splitlines()
            if line.startswith("[CURRICULUM STATUS] ")
        )
        self.assertEqual(json.loads(structured), status)

    def test_process_level_fake_worker_and_scorer_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            processor = root / "processor"
            input_dir = root / "input"
            bundle = root / "bundle"
            for directory in (checkpoint, processor, input_dir, bundle / "images"):
                directory.mkdir(parents=True, exist_ok=True)
            eval_image = root / "eval.png"
            eval_image.write_bytes(b"fake")
            for task in evaluation.TASKS:
                write_jsonl(
                    input_dir / evaluation.TASK_GT_FILE[task],
                    [{"image": str(eval_image)}],
                )
            plans = {}
            hard_rows = []
            for task in evaluation.TASKS:
                image = bundle / "images" / f"{task}.png"
                image.write_bytes(b"fake")
                source_id = f"image-{task}"
                if task != "content_missing":
                    plans[source_id] = {
                        "width": 1,
                        "height": 1,
                        "base_tiles": [[0, 0, 1, 1]],
                        "gt_used": False,
                    }
                hard_rows.append(
                    {
                        "record_id": f"record-{task}",
                        "sample_id": f"sample-{task}",
                        "source_image_id": source_id,
                        "task": task,
                        "image_relpath": f"images/{task}.png",
                        "prompt": f"Locate {task}.",
                        "gt_global": [],
                        "positive": False,
                        "crop_correct_count": 0,
                        "crop_complete4": True,
                        "rollouts": {"crop": frozen_crop_rollouts()},
                    }
                )
            (bundle / "base_scan_plans.json").write_text(
                json.dumps(plans), encoding="utf-8"
            )
            hard = root / "hard.jsonl"
            write_jsonl(hard, hard_rows)
            frozen, curriculum = make_frozen_and_curriculum(
                root,
                5,
                hard_sample_ids=[str(row["sample_id"]) for row in hard_rows],
                hard_groups_path=hard,
            )
            worker = root / "fake_worker.py"
            worker.write_text(
                """
import argparse, hashlib, json, time
from pathlib import Path
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--output-dir'); p.add_argument('--single-task-output-dir')
p.add_argument('--summary-path'); p.add_argument('--tasks'); p.add_argument('--hard-groups-jsonl')
p.add_argument('--hard-rollout-output-dir'); p.add_argument('--expected-hard-task-count', type=int)
p.add_argument('--evaluation-identity-file')
p.add_argument('--hard-rollout-seeds', nargs=4, type=int)
a,_=p.parse_known_args()
root=Path(a.output_dir); marks=root/'_fake_started'; marks.mkdir(parents=True, exist_ok=True)
(marks/a.tasks).write_text('started', encoding='utf-8')
deadline=time.time()+10
while len(list(marks.iterdir())) < 5:
    if time.time() > deadline: raise SystemExit('workers were not concurrent')
    time.sleep(0.02)
task_out=Path(a.single_task_output_dir); task_out.mkdir(parents=True, exist_ok=True)
(task_out/'eval.json').write_text('[]\\n', encoding='utf-8')
hard=[json.loads(line) for line in Path(a.hard_groups_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
hard=[row for row in hard if row['task']==a.tasks]
rollout=Path(a.hard_rollout_output_dir); rollout.mkdir(parents=True, exist_ok=True)
digest=hashlib.sha256(Path(a.evaluation_identity_file).read_bytes()).hexdigest()
for rid in range(4):
    rows=[{**row,'rollout_id':rid,'rollout_seed':a.hard_rollout_seeds[rid],'evaluation_identity_digest':digest,'runtime_error':None,'parse_status':'ok','exact_correct':False} for row in hard]
    (rollout/f'rollout_{rid}.jsonl').write_text(''.join(json.dumps(row)+'\\n' for row in rows), encoding='utf-8')
groups=[{'record_id':row['record_id'],'sample_id':row['sample_id'],'task':a.tasks,'candidate_correct_count':0,'transition':'still_hard'} for row in hard]
(rollout/'groups.jsonl').write_text(''.join(json.dumps(row)+'\\n' for row in groups), encoding='utf-8')
hard_summary={'evaluation_identity_digest':digest,'task':a.tasks,'group_count':len(hard),'attempt_count':len(hard)*4,'error_count':0,'runtime_error_count':0,'parse_error_count':0}
(rollout/'rollout4_summary.json').write_text(json.dumps(hard_summary), encoding='utf-8')
summary={'evaluation_identity_digest':digest,'tasks':[{'task':a.tasks,'dataset_images':1,'processed':1,'skipped_existing':0,'inference_error':0}], 'hard_rollout':hard_summary}
Path(a.summary_path).parent.mkdir(parents=True, exist_ok=True)
Path(a.summary_path).write_text(json.dumps(summary), encoding='utf-8')
""".lstrip(),
                encoding="utf-8",
            )
            scorer = root / "fake_scorer.py"
            scorer.write_text(
                """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--pred_root'); p.add_argument('--output_root'); p.add_argument('--run_name'); p.add_argument('--all_tasks', action='store_true'); p.add_argument('--input_mode'); p.add_argument('--gt_dir'); p.add_argument('--yolo_bbox_format'); p.add_argument('--iou_thresh')
a=p.parse_args()
tasks=('occlusion','cropping','text_overflow','text_ellipsis','content_missing')
sample_ids={}
for task in tasks:
    files=list((Path(a.pred_root)/task).glob('*.json'))
    assert len(files) == 1
    sample_ids[task]=[files[0].stem]
metric={'precision':1.0,'recall':1.0,'f1':1.0,'tp':1,'fp':0,'fn':0}
image={**metric,'tn':0,'accuracy':1.0}; bbox={**metric,'count_accuracy':1.0}
payload={'schema_version':1,'tasks':{task:{'image':image,'bbox':bbox,'total_samples':1,'scored_sample_count':1,'scored_sample_ids':sample_ids[task],'skipped_sample_count':0,'skipped_sample_ids':[]} for task in tasks},'macro':{'image':{'precision':1.0,'recall':1.0,'f1':1.0},'bbox':{'precision':1.0,'recall':1.0,'f1':1.0}}}
out=Path(a.output_root)/a.run_name; out.mkdir(parents=True)
(out/'all_tasks_evaluation.json').write_text(json.dumps(payload), encoding='utf-8')
""".lstrip(),
                encoding="utf-8",
            )
            output = root / "evaluation" / "step-000200"
            command = [
                sys.executable,
                str(SCRIPTS / "run_ui5_curriculum_evaluation.py"),
                "--checkpoint",
                str(checkpoint),
                "--processor-path",
                str(processor),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output),
                "--hard-groups-jsonl",
                str(hard),
                "--rollout-bundle-root",
                str(bundle),
                "--curriculum-manifest",
                str(curriculum),
                "--frozen-selection",
                str(frozen),
                "--expected-hard-groups",
                "5",
                "--inference-crop-mode",
                "full_image",
                "--worker-script",
                str(worker),
                "--scorer-script",
                str(scorer),
                "--project-root",
                str(root),
                "--python",
                sys.executable,
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(len(list((output / "_fake_started").iterdir())), 5)
            for task in evaluation.TASKS:
                self.assertTrue((output / f"ui_{task}" / "eval.json").is_file())
            status = json.loads((output / "evaluation_status.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "ui5_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(status["success"])
            self.assertEqual(status["metrics_sha256"], _sha256(output / "ui5_metrics.json"))
            self.assertEqual(metrics["overall"]["joint_score"], 1.0)
            transitions = (output / "hard_transition.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(transitions), 5)
            hard_summary = json.loads(
                (output / "hard_rollout4_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(hard_summary["group_count"], 5)

            verified = subprocess.run(
                [*command, "--verify-existing-identity"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                verified.returncode, 0, verified.stdout + verified.stderr
            )
            self.assertIn("[UI5 EVALUATION REUSE VERIFIED]", verified.stdout)

            # Relation-gate settings change worker behavior and therefore form
            # part of the durable evaluation identity.
            rejected_gate_config = subprocess.run(
                [
                    *command,
                    "--verify-existing-identity",
                    "--relation-gate-mode",
                    "hard",
                    "--relation-gate-threshold",
                    "0.25",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(rejected_gate_config.returncode, 0)
            self.assertIn(
                "evaluation identity differs", rejected_gate_config.stderr
            )

            # A durable evaluation is reusable only while the exact metrics
            # bytes named by evaluation_status.json remain unchanged.
            metrics_path = output / "ui5_metrics.json"
            original_metrics = metrics_path.read_bytes()
            metrics_path.write_text('{"tampered":true}\n', encoding="utf-8")
            rejected_metrics = subprocess.run(
                [*command, "--verify-existing-identity"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(rejected_metrics.returncode, 0)
            self.assertIn("metrics SHA-256 differs", rejected_metrics.stderr)
            completed_status = json.loads(
                (output / "evaluation_status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(completed_status["success"])
            self.assertEqual(completed_status["status"], "completed")
            metrics_path.write_bytes(original_metrics)

            # Same path/size is not enough: checkpoint content changes must
            # invalidate reuse, while the already completed status remains intact.
            (checkpoint / "weights.bin").write_bytes(b"changed")
            rejected = subprocess.run(
                [*command, "--verify-existing-identity"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            completed_status = json.loads(
                (output / "evaluation_status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(completed_status["success"])
            self.assertEqual(completed_status["status"], "completed")


if __name__ == "__main__":
    unittest.main()
