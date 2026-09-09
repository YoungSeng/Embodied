"""CPU evidence fixtures; these are not claimed to be cluster raw answers."""
import ast
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ui14_common import read_json, read_jsonl, write_json, write_jsonl, file_digest
import qwen3vl_merge_and_score_fixed_5tasks as scorer
from ui14_alignment_audit import audit_step, audit_runs
from PIL import Image


def evidence_fixture(root, steps=(2000, 4000)):
    old, target = root / "old", root / "old-data/test.jsonl"
    gt, merged, forms = [], [], []
    valid = "<ref>对齐异常</ref><box><10><10><100><100></box>"
    # Positive invalid, negative invalid, positive invalid, negative empty,
    # runtime failure, legal FP, miss, unmatched, TP, TN, missing raw.
    cases = [
        (True, None, "<ref>对齐异常</ref></ref><box><1><2><3><4></box>", "parse_error"),
        (False, None, "<box>none<10></box>", "parse_error"),
        (True, None, "<box><1><2></box>", "parse_error"),
        (False, None, "", "parse_error"),
        (True, None, None, "runtime"),
        (False, [[1, 1, 10, 10]], valid, "defect"),
        (True, [], "<box>none</box>", "ok"),
        (True, [[50, 50, 60, 60]], valid, "defect"),
        (True, [[1, 1, 10, 10]], valid, "defect"),
        (False, [], "<box>none</box>", "ok"),
        (False, None, None, "parse_error"),
    ]
    for i, (positive, boxes, raw, status) in enumerate(cases):
        path = root / f"image-{i}.png"
        Image.new("RGB", (100, 100), (i * 10, 100, 50)).save(path)
        actual = [[1, 1, 10, 10]] if positive else []
        gt.append(dict(source_image_id=str(i), source_image=str(path),
                       boxes_px=actual, width=100, height=100))
        merged.append(dict(image_id=str(i), image=str(path),
                           objects=dict(bbox=actual, type="ui_alignment"),
                           pred_ans=dict(bbox=boxes, type="ui_alignment") if boxes is not None else None))
        forms.append((path, raw, status, boxes))
    write_jsonl(target, gt)
    manifest = root / "old-data/evaluation_manifest.json"
    write_json(manifest, dict(eval_set_id="frozen-old-test", normalization_id="old-normalization",
        tasks=[dict(task_key="ui_alignment", task_id=5, test=str(target))]))
    for step in steps:
        score = old / "evaluation/raw" / f"attempt-{step}"
        prediction = old / f"inference-checkpoint-{step}-ui14/ui_alignment"
        write_jsonl(score / "ui_alignment.merged.jsonl", merged)
        metrics = scorer.build_metrics_summary(scorer.evaluate_samples(merged, "ui_alignment", .1, True))
        write_json(score / "ui14_metrics.json", dict(tasks={"ui_alignment": metrics}))
        write_json(old / "evaluation" / f"ui14-step-{step}.json", dict(status="success",
            identity=dict(manifest_digest=file_digest(manifest)), evaluation_run_dir=str(score)))
        for i, (path, raw, status, boxes) in enumerate(forms):
            if status == "runtime":
                write_json(prediction / "errors" / f"{i}.json", dict(image_path=str(path), error="GPU worker exception"))
                continue
            write_json(prediction / "gate" / f"{i}.json", dict(image_path=str(path),
                       prediction_status=status, final_boxes_pixel_xyxy=boxes or []))
            if raw is not None:
                write_json(prediction / "raw" / f"{i}.json", dict(raw_answer=raw))
        # Stale error sidecar alongside a successful prediction must not count twice.
        write_json(prediction / "errors/8.json", dict(image_path=str(forms[8][0]), error="old failed attempt"))
    return old, manifest, merged


class AlignmentAuditTests(unittest.TestCase):
    def test_original_scorer_attribution_readonly_missing_raw_and_reuse(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            old, manifest, _ = evidence_fixture(root)
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in old.rglob("*") if p.is_file()}
            results = audit_runs(old, manifest, ["ui_alignment"], [2000, 4000], root / "new-audit")
            result = results[0]
            self.assertEqual(result["status"], "complete", result["reconciliation_errors"])
            self.assertEqual(result["metrics"]["image"], dict(tp=2, fp=4, fn=4, tn=1,
                precision=2/6, recall=2/6, f1=2/6, accuracy=3/11))
            self.assertEqual({k: result["metrics"]["bbox"][k] for k in ("tp", "fp", "fn")},
                             dict(tp=1, fp=5, fn=5))
            self.assertEqual(result["scorer_invalid_by_gt"], dict(positive=3, negative=3))
            self.assertEqual(result["invalid_scoring_contribution"]["img_fp"], 3)
            self.assertEqual(result["invalid_scoring_contribution"]["img_fn"], 3)
            self.assertEqual(result["csv_row"]["missing_raw"], 2)
            self.assertEqual(sum(result["error_types"].values()), 11)
            self.assertEqual(result["error_types"]["runtime_failure"], 1)
            folder = Path(result["directory"])
            page = (folder / "samples.html").read_text(encoding="utf-8")
            self.assertIn("<details><summary>", page)
            self.assertNotIn("<details open", page)
            self.assertEqual(len(list(read_jsonl(folder / "missing_raw.jsonl"))), 2)
            with mock.patch.object(scorer, "evaluate_samples", side_effect=AssertionError("repeat audit rescored")):
                again = audit_step(old, manifest, "ui_alignment", 2000, root / "new-audit")
            self.assertEqual(again["audit_id"], result["audit_id"])
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
            raw = old / "inference-checkpoint-2000-ui14/ui_alignment/raw/0.json"
            write_json(raw, dict(raw_answer="<box>none<20></box>"))
            refreshed = audit_step(old, manifest, "ui_alignment", 2000, root / "new-audit")
            self.assertNotEqual(refreshed["audit_id"], result["audit_id"])
            write_json(manifest, {**read_json(manifest), "eval_set_id": "another-test"})
            with self.assertRaisesRegex(ValueError, "frozen manifest"):
                audit_step(old, manifest, "ui_alignment", 2000, root / "new-audit")

    def test_refactored_scoring_is_identical_to_verified_baseline(self):
        baseline = subprocess.check_output(["git", "show",
            "0379209bd6089e4deeed473d3108f1f41c8d97bf:qwen3vl_merge_and_score_fixed_5tasks.py"],
            cwd=ROOT).decode("utf-8")
        original = next(n for n in ast.parse(baseline).body
                        if isinstance(n, ast.FunctionDef) and n.name == "evaluate_merged_file")
        namespace = dict(vars(scorer))
        exec(compile(ast.Module(body=[original], type_ignores=[]), "baseline-scorer", "exec"), namespace)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, merged = evidence_fixture(root, steps=())
            path = root / "merged.jsonl"
            for records in (merged, [], merged + [dict(merged[0], image="Figma/example.png")]):
                write_jsonl(path, records)
                for include in (True, False):
                    self.assertEqual(namespace["evaluate_merged_file"](str(path), "ui_alignment", .1, include),
                                     scorer.evaluate_merged_file(str(path), "ui_alignment", .1, include))


if __name__ == "__main__": unittest.main()
