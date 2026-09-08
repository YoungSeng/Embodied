"""CPU-only raw audit: observations never become negative eligibility."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from PIL import Image
from ui14_common import write_json,write_jsonl,read_json,read_jsonl,image_identity,paths_for
from ui14_neg11_data import SYNTH,eligible_evidence
from ui14_neg11 import parse_args,run
import ui14_neg11_raw_audit as audit


class RawAuditTests(unittest.TestCase):
    def fixture(self,root):
        parent=root/"parent";source=root/"source";data=root/"new"
        source.mkdir();data.mkdir()
        images=[]
        for i in range(6):
            image=source/f"image-{i}.png"
            Image.new("RGB",(64,96),(20+i*20,12,30)).save(image)
            images.append(str(image))
        alias=source/"alias.png";alias.write_bytes(Path(images[0]).read_bytes())
        def row(task,split,index,main,raw,local=None):
            identity,w,h=image_identity(main)
            metadata={"ScreenShotURL":main,"RawImgURL":raw}
            if local:metadata["LocalImgURL"]=local
            return {"task_key":task,"split":split,"source_record_id":str(index),
                "source_dataset":task,"source_version":"fixture","source_image":main,
                "source_image_id":identity,"width":w,"height":h,"boxes_px":[[1,2,10,20]],
                "source_page_id":"fixture-page","source_metadata":metadata}
        values={
            ("synth_cropping","train"):[
                row("synth_cropping","train",0,images[2],images[0],images[0]),
                row("synth_cropping","train",1,images[3],str(alias)),
                row("synth_cropping","train",2,images[3],str(source/"absent.png"))],
            ("synth_cropping","test"):[row("synth_cropping","test",0,images[4],images[1])],
            ("synth_occlusion","test"):[row("synth_occlusion","test",0,images[5],images[0])],
            ("synth_radius","train"):[row("synth_radius","train",0,images[2],images[2])]}
        for task in SYNTH:
            for split in ("train","test"):
                write_jsonl(paths_for(parent,task.task_key,split)["normalized"],
                            values.get((task.task_key,split),[]))
        write_json(data/"inventory_summary.json",{"gap":83407,"inventory_id":"frozen fixture"})
        write_jsonl(data/"negative_selection.proposed.jsonl",[{
            "task_key":"synth_cropping","split":"train","source_image_id":image_identity(images[0])[0]}])
        write_json(data/"negative_page_assignments.json",{"assignments":{"old":"train"}})
        options=parse_args(["audit-raw","--parent-root",str(parent),"--source-root",str(source),
            "--data-root",str(data),"--old-output",str(root/"old-model"),"--output-dir",str(root/"new-model")])
        return options,parent,source,data

    def test_dedup_pair_comparisons_conflicts_and_no_selection_changes(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            args,parent,source,data=self.fixture(Path(tmp))
            protected={p:(p.read_bytes(),p.stat().st_mtime_ns) for folder in (parent,source,data)
                       for p in folder.rglob("*") if p.is_file()}
            with mock.patch.object(audit,"parent_binding",return_value={
                "parent_normalization_id":"parent-v1","repair_run_id":"repair-v2"}):
                result=run(args)
                first=result["tasks"]["synth_cropping/train"]
                self.assertEqual(first["positive_records"],3)
                self.assertEqual(first["positive_images"],2)
                self.assertEqual(first["raw_reference_occurrences"],3)
                self.assertEqual(first["missing_reference_occurrences"],1)
                self.assertEqual(first["raw_unique_images"],1)
                self.assertEqual(first["raw_matches_local_images"],1)
                self.assertEqual(first["raw_already_selected_images"],1)
                self.assertEqual(first["raw_observed_cross_split_images"],1)
                self.assertEqual(result["tasks"]["synth_cropping/test"]["raw_unconfirmed_images"],1)
                self.assertEqual(result["tasks"]["synth_radius/train"]["raw_matches_defect_images"],1)
                self.assertEqual(result["label_changes"],0)
                self.assertEqual(result["selection_changes"],0)
                self.assertEqual(len(list(read_jsonl(data/"raw_reference_audit/pairs.jsonl"))),4)
                page=(data/"raw_reference_audit/samples.html").read_text(encoding="utf-8")
                self.assertIn("Raw（语义待确认）",page)
                self.assertIn("未加入负样本",page)
                self.assertNotIn("type=\"checkbox\" checked",page)
                with mock.patch.object(Image,"open",side_effect=AssertionError("audit reread image")):
                    again=run(args)
                self.assertEqual(result["tasks"],again["tasks"])
            self.assertEqual(protected,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in protected})
            self.assertEqual(read_json(data/"stage_summaries/audit-raw.json")["status"],"complete")
            self.assertIsNone(eligible_evidence({},"raw","synth_cropping"))

    def test_cli_exposes_cpu_audit_without_a_trust_raw_switch(self):
        self.assertEqual(parse_args(["audit-raw"]).stage,"audit-raw")
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):
            parse_args(["inventory","--trust-raw"])


if __name__=="__main__":unittest.main()
