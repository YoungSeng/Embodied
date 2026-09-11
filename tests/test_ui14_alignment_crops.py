"""CPU checks for the isolated alignment-crops run and production decoder comparison."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ui14_common import read_json, read_jsonl, write_json, write_jsonl, file_digest, UI_TASKS, paths_for
from ui14_alignment_crops_common import PROFILE, PROJECT, DATA, OUTPUT, default_roots, protect_context
from ui14_alignment_data import prepare_frozen_data
from ui14_alignment_context import parse_args
from ui14_alignment_eval import prepare_comparisons, run_comparison
from ui14_verification import signature, verification_session
from ui14_alignment_crops_data import finalize_crops
from tests import test_ui14_alignment_context as comparison_tests
from tests.test_ui14_inference_workers import fixture
import run_ui5_parallel_inference as parallel


class CropAlignmentTests(unittest.TestCase):
    def test_frozen_available_crop_pipeline_reuses_parent_and_keeps_original_test(self):
        from tests.test_ui14_neg11 import Neg11Tests
        from tests import test_ui14_pipeline as fixtures
        from ui14_cache_prepare import publish_prepared
        from eaglevl.ui_task_registry import task_from_spec, configure_task_registry
        from types import SimpleNamespace
        from PIL import Image
        Neg11Tests.setUpClass()
        try:
            case = Neg11Tests("test_full_composite_cpu_path_and_incremental_cache")
            with contextlib.redirect_stdout(io.StringIO()): case.test_full_composite_cpu_path_and_incremental_cache()
            parent, root, output = case.root / "new", case.root / "crops-alignment", case.root / "fresh-run"
            before = {p: (p.read_bytes(), signature(p)) for p in parent.rglob("*") if p.is_file()}
            parent_eval, parent_recipe = read_json(parent / "evaluation_manifest.json"), read_json(parent / "training_recipe.json")
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(Image, "open", side_effect=AssertionError("source image decoded during freeze")):
                with verification_session(root): prepared = prepare_frozen_data(parent, root, output, profile=PROFILE)
            self.assertFalse(prepared["ready"])
            self.assertEqual(prepared["negative_quota_policy"], "available")
            self.assertEqual(prepared["one_to_one_shortfall"], 2)
            self.assertEqual((root / "negative_selection.jsonl").read_bytes(), (parent / "negative_selection.jsonl").read_bytes())
            task = task_from_spec(read_json(root / "task_registry.json")["tasks"][5])
            self.assertEqual(task.view_policy, "crops")
            # Fixed raw detector outputs are a CPU fixture, never called a GPU run.
            # All geometry, PNGs, crop labels, ready markers and checks are real.
            with contextlib.redirect_stdout(io.StringIO()), verification_session(root):
                for split in ("train", "test"):
                    paths = paths_for(root, task.task_key, split)
                    local_inputs = paths["cache"] / "task_input_manifest.json"
                    write_json(local_inputs, {task.task_key: str(paths["detector_input"])})
                    fixture_paths = {**paths, "detector_inputs": local_inputs}
                    with mock.patch.object(fixtures, "paths_for", return_value=fixture_paths):
                        fixtures.detector_fixture(root, task, split)
                    config = read_json(paths["cache"] / "detections/detector_config.json")
                    publish_prepared(paths, prepared["normalization_id"], config, 1)
                    from prepare_ui14_sft import crop_annotations
                    crop_annotations(root, task, split, list(read_jsonl(paths["normalized"])))
                # The inherited UI5 audit is a fixture-only marker. Validate
                # every actual UI9/added alignment cache with the real checker.
                from ui5_eval_detector_cache import validate_eval_detector_cache as actual_cache_check
                def validate_cache(cache, *a, **kw):
                    return {} if Path(cache) == case.ui5cache else actual_cache_check(cache, *a, **kw)
                with mock.patch("ui5_eval_detector_cache.validate_eval_detector_cache", side_effect=validate_cache):
                    first = finalize_crops(root, output)
            crop_bytes = {p: (p.read_bytes(), signature(p)) for p in (root / "crop_images").rglob("*.png")}
            self.assertTrue(crop_bytes)
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(Image, "open", side_effect=AssertionError("repeat finalize decoded image")):
                with verification_session(root): second = finalize_crops(root, output)
                with verification_session(root): resumed = prepare_frozen_data(parent, root, output, profile=PROFILE)
            self.assertTrue(first["ready"] and second["ready"] and resumed["ready"])
            self.assertEqual(first["negative_counts"], prepared["negative_counts"])
            current, recipe = read_json(root / "evaluation_manifest.json"), read_json(root / "training_recipe.json")
            self.assertEqual(current["eval_set_id"], parent_eval["eval_set_id"])
            self.assertEqual(current["normalization_id"], parent_eval["normalization_id"])
            for t in UI_TASKS:
                if t.task_id != 5: self.assertEqual(recipe[t.task_key], parent_recipe[t.task_key])
                self.assertEqual(current["tasks"][t.task_id]["test"], parent_eval["tasks"][t.task_id]["test"])
            for row in read_jsonl(recipe["ui_alignment"]["annotation"]):
                self.assertNotEqual(row["image"], row["source_image"])
                self.assertEqual(row["_ui5_record_kind"], "crop")
                self.assertEqual(row["view_policy"], "crops")
            self.assertEqual(current["tasks"][4]["view_policy"], "full_image")
            self.assertEqual(first["alignment_crop_check"]["sampling"]["negative_to_positive_ratio"], 2)
            config = configure_task_registry(SimpleNamespace(), read_json(root / "task_registry.json"), 14)
            self.assertEqual(config.ui_task_registry[5]["view_policy"], "crops")
            self.assertEqual(before, {p: (p.read_bytes(), signature(p)) for p in before})
            self.assertEqual(crop_bytes, {p: (p.read_bytes(), signature(p)) for p in crop_bytes})
        finally:
            Neg11Tests.tearDownClass()

    def test_only_alignment_can_switch_view_and_worker_uses_saved_crop_policy(self):
        from eaglevl.ui_task_registry import validate_registry, task_from_spec
        specs = [t.to_dict() for t in UI_TASKS]
        validate_registry(specs, 14)
        specs[5]["view_policy"] = "crops"
        validate_registry(specs, 14)
        self.assertEqual(task_from_spec(specs[5]).view_policy, "crops")
        from prepare_ui14_detector_crops import selected_crop_jobs
        self.assertEqual([(t.task_key, s) for t, s in selected_crop_jobs(specs, ["ui_alignment"])],
                         [("ui_alignment", "train"), ("ui_alignment", "test")])
        self.assertEqual(len(selected_crop_jobs(specs)), 16)
        specs[4]["view_policy"] = "crops"
        with self.assertRaises(ValueError): validate_registry(specs, 14)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = fixture(root)
            document = read_json(root / "evaluation_manifest.json")
            document["tasks"][5]["view_policy"] = "crops"
            write_json(root / "evaluation_manifest.json", document)
            with mock.patch.object(sys, "argv", argv): args = parallel.parse_args()
            command = parallel.build_command(args, "ui_alignment", "0", root / "summary.json")
            self.assertEqual(command[command.index("--inference-crop-mode") + 1], "detector_scan")
            self.assertIn("--detector-crop-manifest", command)
            command = parallel.build_command(args, "content_missing", "0", root / "summary.json")
            self.assertEqual(command[command.index("--inference-crop-mode") + 1], "full_image")
            self.assertNotIn("--detector-crop-manifest", command)

    def test_old_generic_and_alignment_variables_cannot_select_new_run(self):
        env = {"UI14_DATA_ROOT": "old-neg", "UI14_ALIGNMENT_DATA_ROOT": "old-context"}
        self.assertEqual(default_roots(env)[0], DATA)
        with mock.patch.dict(os.environ, env):
            args = parse_args(["prepare"], alignment_crops_run=True)
        self.assertEqual((args.data_root, args.output_dir, args.profile), (DATA, OUTPUT, PROFILE))
        from ui14_alignment_common import DATA as OLD_DATA
        with self.assertRaisesRegex(ValueError, "read-only source"): protect_context(OLD_DATA, OUTPUT)

    def test_formal_yaml_is_four_a800_cpt9000_with_production_decoder_and_exclusive_tasks(self):
        from submit_locany_ui5 import parse_args as submit_args, render_job
        from ui14_checks import validate_formal_yaml
        import yaml
        args = submit_args(["--profile", PROFILE, "--machine", "a800", "--gpus", "4", "--render-only"])
        rendered, runtime = render_job(args)
        validate_formal_yaml(rendered, runtime, config_path=args.config)
        self.assertEqual(runtime["PROJECT_ROOT"], PROJECT)
        self.assertEqual(runtime["OUTPUT_DIR"], OUTPUT)
        self.assertEqual(runtime["UI_EVAL_ANSWER_GRAMMAR"], "legacy")
        self.assertEqual(runtime["EVAL_FAIL_POLICY"], "stop")
        self.assertEqual(runtime["EVAL_EXCLUSIVE_GPU_TASKS"].split(), ["synth_loneword", "change_line_illegal_v3", "synth_inner_margin"])
        self.assertEqual(yaml.safe_load(rendered)["jobRunParams"]["envsList"]["EVAL_EXCLUSIVE_GPU_TASKS"], runtime["EVAL_EXCLUSIVE_GPU_TASKS"])
        for key, value in dict(MAX_STEPS=16000, INIT_CPT_STEP=9000, EVAL_AT_START=1, EVAL_INFERENCE_WORKERS_PER_GPU=2,
                               EVAL_INTERVAL_STEPS=1000, SAMPLE_LOG_INTERVAL=100).items(): self.assertEqual(int(runtime[key]), value)

    def test_independent_comparison_selects_legacy_even_with_stale_grammar_env(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            old, manifest = comparison_tests.SnapshotTests().fixture(root)
            old_before = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
            structured = prepare_comparisons(old, manifest, ["ui_alignment"], [2000], root / "new", grammar="ui14_answer_v1")
            snapshots = {p: signature(p) for p in (root / "new/model_snapshots").rglob("*") if p.is_file() and p.name != "snapshot.json"}
            legacy = prepare_comparisons(old, manifest, ["ui_alignment"], [2000], root / "new")
            self.assertNotEqual(legacy, structured)
            self.assertEqual(snapshots, {p: signature(p) for p in snapshots})
            with mock.patch.dict(os.environ, {"UI_EVAL_ANSWER_GRAMMAR": "ui14_answer_v1"}), \
                    mock.patch("ui14_alignment_eval.subprocess.run") as child, \
                    mock.patch("run_ui14_eval.score_ui9", return_value=({"image": {"f1": .1}}, {})):
                run_comparison(legacy[0])
            command = child.call_args.args[0]
            self.assertEqual(command[command.index("--ui-answer-grammar")+1], "legacy")
            self.assertEqual(child.call_args.kwargs["env"]["UI_EVAL_ANSWER_GRAMMAR"], "legacy")
            self.assertEqual(old_before, {p: p.read_bytes() for p in old_before})

    def test_three_oom_tasks_reserve_physical_cards_and_other_tasks_stay_parallel(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            argv = fixture(root)[:-1]  # Use the production default exclusive set.
            guard, barrier = threading.Lock(), threading.Barrier(5, timeout=10)
            active = {g: set() for g in "0123"}
            arrivals = []
            exclusive = {"synth_loneword", "change_line_illegal_v3", "synth_inner_margin"}
            first_wave = []
            def child(command, **kwargs):
                task, gpu = command[command.index("--tasks")+1], kwargs["env"]["CUDA_VISIBLE_DEVICES"]
                with guard:
                    active[gpu].add(task)
                    arrivals.append(task)
                    first = len(arrivals) <= 5
                    if first: first_wave.append((task, gpu))
                    if active[gpu] & exclusive: self.assertEqual(len(active[gpu]), 1)
                    self.assertLessEqual(len(active[gpu]), 2)
                if first: barrier.wait()
                with guard: active[gpu].remove(task)
                write_json(root / "pred" / task / "sample.json", {"task": task})
                return subprocess.CompletedProcess(command, 0)
            with mock.patch.object(sys, "argv", argv), mock.patch.object(parallel.subprocess, "run", side_effect=child):
                self.assertEqual(parallel.main(), 0)
            status = read_json(root / "pred/parallel_inference_status.json")
            self.assertEqual(len(arrivals), 14)
            self.assertEqual(len({g for t, g in first_wave if t in exclusive}), 3)
            for t, row in status["tasks"].items(): self.assertEqual(row["gpu_slots_reserved"], 2 if t in exclusive else 1)


if __name__ == "__main__": unittest.main()
