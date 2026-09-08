"""CPU policy checks: sampling 1:1 does not invent independent negatives."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from ui14_common import read_json, read_jsonl
from ui14_negative_quota import image_quota_counts, quota_policy
from ui14_local_candidates import discover_local_candidates
from ui14_neg11 import parse_args


class NegativeQuotaTests(unittest.TestCase):
    def test_actual_counts_zero_negatives_and_strict_compatibility(self):
        counts={"positive_images":3,"negative_images":1,"total_images":4}
        ext={"targets":{"task/train":{"positive_images":3,"existing_negative_images":0,"selected_images":1}}}
        self.assertEqual(quota_policy(ext),"strict")
        with self.assertRaisesRegex(ValueError,"not 1:1"):
            image_quota_counts(ext,"task/train",counts)
        ext["negative_quota_policy"]="available"
        result=image_quota_counts(ext,"task/train",counts)
        self.assertEqual(result["one_to_one_shortfall"],2)
        self.assertAlmostEqual(result["negative_to_positive_ratio"],1/3)
        # A shortage is allowed; dropping an already selected image is not.
        with self.assertRaisesRegex(ValueError,"counts changed"):
            image_quota_counts(ext,"task/train",{**counts,"negative_images":0})
        ext["targets"]["task/train"]["selected_images"]=0
        result=image_quota_counts(ext,"task/train",{**counts,"negative_images":0,"total_images":3})
        self.assertEqual(result["negative_to_positive_ratio"],0)
        self.assertEqual(result["one_to_one_shortfall"],3)

    def test_cli_defaults_to_available_and_strict_remains_explicit(self):
        with mock.patch.dict("os.environ",{},clear=True):
            self.assertEqual(parse_args(["normalize"]).negative_quota_policy,"available")
        self.assertEqual(parse_args(["normalize","--negative-quota-policy","strict"]).negative_quota_policy,"strict")

    def test_local_filename_discovery_does_not_read_or_label_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=SimpleNamespace(task_key="synth_cropping")
            paths=[root/"local_imgs"/name for name in ("design_12:34_local.jpeg","design_56:78_local.jpeg","unknown.png")]
            # Paths deliberately do not exist: this operation must not open images.
            resolver=SimpleNamespace(index=lambda _: (paths,{}))
            rows=[{"task_key":task.task_key,"source_page_id":"design:12:34","source_record_id":"p1",
                   "split":"train","boxes_px":[[1,2,3,4]]}]
            result=discover_local_candidates(root,resolver,[task],rows,{(task.task_key,str(paths[1]))})
            self.assertEqual(result[task.task_key]["files"],3)
            self.assertEqual(result[task.task_key]["same_task_page_matched"],1)
            observed=list(read_jsonl(root/"local_candidate_discovery/candidates.jsonl"))
            self.assertEqual(observed[0]["evidence_status"],"unconfirmed_local_candidate")
            self.assertEqual(observed[0]["observed_parent_splits"],["train"])
            self.assertFalse(any(r["admitted_by_discovery"] for r in observed))
            self.assertEqual(read_json(root/"local_candidate_discovery/summary.json")["label_changes"],0)


if __name__=="__main__":unittest.main()
