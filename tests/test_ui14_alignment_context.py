"""CPU integration for frozen neg11 data, model-only snapshots and new profile."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ui14_common import read_json, write_json, write_jsonl, file_digest, UI_TASKS, INIT_CHECKPOINT
from ui14_alignment_common import PROFILE, PROJECT, DATA, OUTPUT, NEG_OUTPUT
from ui14_alignment_data import prepare_frozen_data
from ui14_alignment_eval import snapshot_checkpoint, prepare_comparisons, run_comparison
from ui14_verification import signature, verification_session
from eaglevl.ui_answer_grammar import VERSION, decode_contract


class FrozenDataTests(unittest.TestCase):
    def test_real_composite_available_data_can_be_frozen_without_images_or_parent_writes(self):
        # Build the existing 14-task CPU integration fixture, including real PNGs,
        # detector/cache bindings and a synth task with a real fixture deficit.
        from tests.test_ui14_neg11 import Neg11Tests
        from PIL import Image
        Neg11Tests.setUpClass()
        try:
            case = Neg11Tests("test_full_composite_cpu_path_and_incremental_cache")
            with contextlib.redirect_stdout(io.StringIO()):
                case.test_full_composite_cpu_path_and_incremental_cache()
            parent, root = case.root / "new", case.root / "alignment"
            before = {p: (file_digest(p), signature(p)) for p in parent.rglob("*") if p.is_file()}
            old_eval = read_json(parent / "evaluation_manifest.json")
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    Image, "open", side_effect=AssertionError("Frozen parent image decoded")):
                with verification_session(root):
                    first = prepare_frozen_data(parent, root, case.root / "new-output")
                with verification_session(root):
                    second = prepare_frozen_data(parent, root, case.root / "new-output")
            self.assertTrue(first["ready"])
            self.assertEqual(second["manifest_copy_counts"]["copied"], 0)
            self.assertEqual(second["negative_quota_policy"], "available")
            self.assertEqual(second["one_to_one_shortfall"], 2)
            self.assertEqual((root / "training_recipe.json").read_bytes(), (parent / "training_recipe.json").read_bytes())
            self.assertEqual((root / "negative_selection.jsonl").read_bytes(), (parent / "negative_selection.jsonl").read_bytes())
            new_eval = read_json(root / "evaluation_manifest.json")
            self.assertEqual(new_eval["eval_set_id"], old_eval["eval_set_id"])
            self.assertEqual(new_eval["normalization_id"], old_eval["normalization_id"])
            for old, new in zip(old_eval["tasks"], new_eval["tasks"]):
                self.assertTrue(all(new[k] == v for k, v in old.items()))
                if old["task_id"] >= 5 and old["view_policy"] == "crops":
                    self.assertTrue(new["detector_input"].startswith(str(parent)))
            self.assertEqual(before, {p: (file_digest(p), signature(p)) for p in before})
        finally:
            Neg11Tests.tearDownClass()


class SnapshotTests(unittest.TestCase):
    def fixture(self, root):
        old = root / "old-run"
        manifest = root / "old-data/evaluation_manifest.json"
        specs = [{**t.to_dict(), "test": str(root / f"test-{t.task_id}.jsonl"),
                  "normalization_id": "old-normalization"} for t in UI_TASKS]
        write_jsonl(specs[5]["test"], [])
        write_json(manifest, dict(tasks=specs, normalization_id="old-normalization", eval_set_id="old-test"))
        for step in (2000, 4000):
            model = old / f"checkpoint-{step}"
            write_json(model / "config.json", dict(ui_num_tasks=14, ui_task_registry=specs))
            (model / "model.safetensors").write_bytes(b"CPU fixture, not real tensors")
            write_json(model / "trainer_state.json", dict(global_step=step))
            write_json(old / "evaluation" / f"ui14-step-{step}.json",
                       dict(status="success", identity=dict(manifest_digest=file_digest(manifest))))
        return old, manifest

    def test_snapshot_and_prediction_reuse_are_bound_and_old_weights_are_never_modified(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            old, manifest = self.fixture(root)
            before = {p: (p.read_bytes(), signature(p)) for p in old.rglob("*") if p.is_file()}
            targets = prepare_comparisons(old, manifest, ["ui_alignment"], [2000, 4000], root / "new-data")
            self.assertEqual(targets, prepare_comparisons(old, manifest, ["ui_alignment"], [2000, 4000], root / "new-data"))
            binding = read_json(Path(targets[0]) / "binding.json")
            model = Path(binding["model_snapshot"])
            self.assertNotEqual(model, old / "checkpoint-2000")
            self.assertFalse((model / "trainer_state.json").exists())
            self.assertEqual(file_digest(model / "model.safetensors"), file_digest(old / "checkpoint-2000/model.safetensors"))
            with mock.patch("ui14_alignment_eval.subprocess.run") as launched, mock.patch(
                    "run_ui14_eval.score_ui9", return_value=({"image": {"f1": .1}}, {})):
                done = run_comparison(targets[0])
            self.assertEqual(done["status"], "success")
            command = launched.call_args.args[0]
            self.assertIn(str(model), command)
            self.assertEqual(launched.call_args.kwargs["env"]["UI_EVAL_ANSWER_GRAMMAR"], "legacy")
            with mock.patch("ui14_alignment_eval.subprocess.run", side_effect=AssertionError("repeat inference")):
                self.assertEqual(run_comparison(targets[0]), done)
            with mock.patch("ui14_alignment_eval.decode_contract", return_value={"changed": True}):
                with self.assertRaisesRegex(ValueError, "Decoder code changed"):
                    run_comparison(targets[0])
                changed = prepare_comparisons(old, manifest, ["ui_alignment"], [2000], root / "new-data")
                self.assertNotEqual(changed[0], targets[0])
            self.assertEqual(before, {p: (p.read_bytes(), signature(p)) for p in before})
            (old / "checkpoint-2000/model.safetensors").write_bytes(b"changed source weights")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                snapshot_checkpoint(old / "checkpoint-2000", model)

    def test_interrupted_snapshot_retains_other_completed_files(self):
        import ui14_alignment_eval as driver
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            old, _ = self.fixture(root)
            target = root / "copy"
            first = snapshot_checkpoint(old / "checkpoint-2000", target)
            signature_before = signature(target / "model.safetensors")
            # Simulate interruption after publishing the first file on a rerun.
            original = driver.write_json
            def interrupt(path, value):
                original(path, value)
                raise InterruptedError("fixture interruption")
            with mock.patch.object(driver, "write_json", side_effect=interrupt):
                with self.assertRaises(InterruptedError): snapshot_checkpoint(old / "checkpoint-2000", target)
            self.assertIn("model.safetensors", read_json(target / "snapshot.json")["files"])
            last = snapshot_checkpoint(old / "checkpoint-2000", target)
            self.assertEqual(last["weight_id"], first["weight_id"])
            self.assertEqual(signature(target / "model.safetensors"), signature_before)


class AlignmentProfileTests(unittest.TestCase):
    def test_four_a800_formal_profile_unchanged_optimizer_data_and_isolated_paths(self):
        from submit_locany_ui5 import parse_args, render_job
        from ui14_checks import validate_formal_yaml
        args = parse_args(["--profile", PROFILE, "--machine", "a800", "--gpus", "4",
                           "--resource-group", "aiai_locate", "--ui14-data-root", DATA, "--render-only"])
        yaml, runtime = render_job(args)
        validate_formal_yaml(yaml, runtime, config_path=args.config)
        self.assertEqual(runtime["PROJECT_ROOT"], PROJECT)
        self.assertEqual(runtime["OUTPUT_DIR"], OUTPUT)
        self.assertNotEqual(runtime["OUTPUT_DIR"], NEG_OUTPUT)
        self.assertEqual(runtime["UI_EVAL_ANSWER_GRAMMAR"], VERSION)
        self.assertEqual(runtime["INIT_CHECKPOINT"], INIT_CHECKPOINT)
        self.assertEqual(float(runtime["UI_NEGATIVE_TO_POSITIVE_RATIO"]), 2.)
        for field, value in dict(MAX_STEPS=16000, SAMPLE_LOG_INTERVAL=100, EVAL_INTERVAL_STEPS=1000,
                                 EVAL_INFERENCE_WORKERS_PER_GPU=2, EVAL_AT_START=1).items():
            self.assertEqual(int(runtime[field]), value)


if __name__ == "__main__": unittest.main()
