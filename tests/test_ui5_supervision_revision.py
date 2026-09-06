import contextlib
import ast
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from eaglevl.train.ui5_supervision import canonical_answer, corrected_record, validate_answer
from scripts import ui5_curriculum_text_revision as revision
from tests import test_ui5_curriculum_recipe as fixtures


class AnswerContractTests(unittest.TestCase):
    def record(self, boxes, task="occlusion", kind="crop"):
        return {"_ui5_sample_id": "source-positive-crop-negative", "_ui5_task": task,
                "_ui5_record_kind": kind, "_ui5_positive": bool(boxes),
                "_ui5_crop_gt_local_1000": boxes, "_ui5_union_gt_1000": boxes,
                "_ui5_sample_gt_global": [[1, 1, 20, 20]], "image": "/unchanged/image.png",
                "conversations": [{"from": "human", "value": "unchanged prompt"},
                    {"from": "gpt", "value": "<ref>overlapping elements</ref><box>none</box>"}]}

    def test_original_positive_crop_negative_uses_only_current_crop_gt(self):
        record = self.record([])
        record["_ui5_union_gt_1000"] = [[1, 1, 20, 20]]
        before = copy.deepcopy(record)
        fixed, changed = corrected_record(record)
        self.assertTrue(changed)
        self.assertEqual(validate_answer(fixed), "<box>none</box>")
        self.assertEqual(record, before)
        fixed["conversations"][-1]["value"] = before["conversations"][-1]["value"]
        self.assertEqual(fixed, before)  # All IDs, GT, image, prompts unchanged.

    def test_native_negative_and_content_missing_have_same_exact_contract(self):
        for kind, task in (("crop", "cropping"), ("full_image", "content_missing"), ("global_view", "content_missing")):
            record = self.record([], task, kind)
            record["conversations"][-1]["value"] = "<box>none</box>"
            self.assertFalse(corrected_record(record)[1])
            self.assertEqual(validate_answer(record), "<box>none</box>")

    def test_multi_box_positive_has_task_ref_and_all_boxes_in_original_order(self):
        boxes = [[1, 2, 10, 20], [100, 200, 300, 400]]
        expected = "<ref>text overflow</ref><box><1><2><10><20></box><box><100><200><300><400></box>"
        self.assertEqual(canonical_answer("ui_text_overflow", boxes), expected)
        record = self.record(boxes, "text_overflow")
        self.assertEqual(validate_answer(corrected_record(record)[0]), expected)

    def test_full_structure_rejects_wrong_ref_missing_box_trailing_eos_or_mixed_none(self):
        record = self.record([[1, 2, 10, 20]])
        correct = canonical_answer("occlusion", [[1, 2, 10, 20]])
        for bad in (correct.replace("overlapping elements", "wrong task"), correct + "<|im_end|>",
                    correct + "<box>none</box>", "<ref>overlapping elements</ref>"):
            record["conversations"][-1]["value"] = bad
            with self.assertRaisesRegex(ValueError, "noncanonical complete"):
                validate_answer(record)
        negative = self.record([])
        with self.assertRaisesRegex(ValueError, "noncanonical complete"):
            validate_answer(negative)


class TextPublicationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        fixture = fixtures.CurriculumRecipeTests()
        bundle, difficulty = fixture._fixture(self.root)
        self.source = self.root / "curriculum"
        with contextlib.redirect_stdout(io.StringIO()):
            fixtures.curriculum_recipe.build(fixture._args(bundle, difficulty, self.source))
        # Reproduce existing erroneous answers without touching any non-answer field.
        success = json.loads((self.source / "_SUCCESS.json").read_text())
        self.modified_counts = {}
        for pool in revision.POOLS:
            path = self.source / f"{pool}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.modified_counts[pool] = 0
            for record in records:
                if not revision.record_boxes(record):
                    message, key = revision.assistant_slot(record)
                    message[key] = "<ref>old full image prefix</ref>" + message[key]
                    self.modified_counts[pool] += 1
            fixtures.write_jsonl(path, records)
            success["files"][path.name] = revision.metadata(path)
        (self.source / "_SUCCESS.json").write_text(json.dumps(success))

    def publish(self):
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(fixtures.curriculum_recipe.Image, "open", side_effect=AssertionError("no PNG reads")):
            return revision.publish_revision(self.source)

    def test_answer_only_revision_real_recipe_loading_and_checkpoint_identity(self):
        before = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        destination, manifest = self.publish()
        self.assertNotEqual(destination, self.source)
        self.assertEqual(list(destination.rglob("*.png")), [])
        # Actual builder/trainer recipe-resolution helper reads the NEW annotation files.
        loaded = fixtures.curriculum_recipe._base_records(destination / revision.RECIPE)
        for record in loaded:
            validate_answer(record)
        # Actual production JSONL loader + path resolver, not just a copied parser.
        from eaglevl.train.dataset_sampling import resolve_recipe_entry_paths
        trainer_path = fixtures.PROJECT_ROOT / "eaglevl/train/locany_finetune_magi_stream.py"
        loader_class = next(n for n in ast.parse(trainer_path.read_text(encoding="utf-8")).body
                            if isinstance(n, ast.ClassDef) and n.name == "LazyJsonlLoader")
        namespace = {"List": list, "os": os, "json": json, "logger": mock.Mock()}
        exec(compile(ast.Module(body=[loader_class], type_ignores=[]), str(trainer_path), "exec"), namespace)
        real_loaded_count = 0
        recipe_path = destination / revision.RECIPE
        for entry in json.loads(recipe_path.read_text()).values():
            resolved = resolve_recipe_entry_paths(entry, recipe_path)
            loader = namespace["LazyJsonlLoader"](resolved["annotation"])
            try:
                self.assertEqual(Path(loader.paths[0]).parent, destination)
                for i in range(len(loader)):
                    validate_answer(loader[i])
                real_loaded_count += len(loader)
            finally:
                loader.__del__()
        self.assertEqual(real_loaded_count, len(loaded))
        self.assertEqual(len(loaded), sum(p["training_records"] for p in manifest["pools"].values()))
        for pool in revision.POOLS:
            old = [json.loads(l) for l in (self.source / f"{pool}.jsonl").read_text().splitlines()]
            new = [json.loads(l) for l in (destination / f"{pool}.jsonl").read_text().splitlines()]
            self.assertEqual(new, [corrected_record(r)[0] for r in old])
        report = json.loads((destination / "supervision_format.json").read_text())
        for pool in report["pools"]:
            self.assertEqual(pool["ref_negative_before"], self.modified_counts[pool["pool"]])
            self.assertEqual(pool["modified_answers"], self.modified_counts[pool["pool"]])
            self.assertEqual(pool["ref_negative_after"], 0)
            self.assertTrue(pool["modified_sample_ids"])
        identity = fixtures.curriculum_artifact_identity(destination / revision.RECIPE, fixtures.CurriculumRecipeTests._schedule())
        self.assertEqual(identity["identity_digest"], manifest["identity_digest"])
        self.assertEqual(before, {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*") if p.is_file()})
        self.assertEqual(self.publish()[0], destination)

    def test_full_audit_checks_gt_not_tag_counts(self):
        with self.assertRaisesRegex(ValueError, "noncanonical complete"):
            revision.supervision_audit(self.source)
        destination, _ = self.publish()
        self.assertTrue(all(row["full_answer_validated"] for row in revision.supervision_audit(destination)))

    def test_old_recipe_redirect_or_changed_text_cannot_pass_resume_identity(self):
        destination, _ = self.publish()
        recipe_path = destination / revision.RECIPE
        recipe = json.loads(recipe_path.read_text())
        next(iter(recipe.values()))["annotation"] = [str(self.source / "hard.jsonl")]
        recipe_path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            revision.verify_revision(destination)

    def test_formal_binding_rejects_old_text_and_wrong_hash(self):
        with self.assertRaisesRegex(ValueError, "corrected training-text revision"):
            revision.verify_revision(self.source)
        destination, _ = self.publish()
        with self.assertRaisesRegex(ValueError, "hash differs from formal YAML"):
            revision.verify_revision(destination, expected_recipe_sha="0" * 64)

    def test_runtime_negative_audit_does_not_advance_rng_or_sampler(self):
        import random
        import numpy as np
        import torch
        from eaglevl.train.ui5_token_contract import audit_training_negative_pools
        destination, _ = self.publish()
        datasets, calls = [], []
        for pool in revision.POOLS:
            records = [json.loads(line) for line in (destination / f"{pool}.jsonl").read_text().splitlines()]
            def materialize(index, *, audit_negative, pool=pool):
                self.assertTrue(audit_negative)
                calls.append((pool, index))
                random.random()
                np.random.random()
                torch.rand(1)
                return {"negative_supervision_audit": {"valid": True, "ar_labels": [2, 4, 3, 10],
                        "mtp_blocks": [[2, 4, 3, 9, 9, 9], [10, 9, 9, 9, 9, 9]]}}
            datasets.append(SimpleNamespace(curriculum_pool=pool, ds_name=pool, lazy_loader=records,
                            active_indices=list(range(len(records))), _materialize_logical_index=materialize))
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with contextlib.redirect_stdout(io.StringIO()):
            report = audit_training_negative_pools(SimpleNamespace(datasets=datasets), self.root / "run", 0,
                                                   destination / revision.RECIPE)
        self.assertEqual(len(calls), 3)
        self.assertFalse(report["sampler_advanced"])
        self.assertEqual(random.getstate(), rng[0])
        np.testing.assert_equal(np.random.get_state(), rng[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng[2]))


if __name__ == "__main__":
    unittest.main()
