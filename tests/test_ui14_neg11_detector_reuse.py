"""Real CPU images/JSON; fake detector verifies only absent content is inferred."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from PIL import Image
import prepare_ui5_eval_detector_crops as entry
import run_ui5_crop_audit as audit
import ui14_neg11_cache as cache
from ui14_common import read_json,read_jsonl,write_json,write_jsonl


class DetectorReuseTests(unittest.TestCase):
    def test_cross_task_partial_shard_preserves_members_and_skips_cached_image(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); train=root/"train"; test=root/"test"
            args=entry.parse_args(["--output-dir",str(test),"--parser-root",str(root),"--resume"])
            config=audit.detector_config(args)
            rows=[]
            for i in range(2):
                image=root/f"{i}.png";Image.new("RGB",(12,24),(i,0,0)).save(image)
                rows.append({"image_id":f"eval_{i}","content_id":f"full-file-content-{i}",
                    "width":12,"height":24,"image_path":str(image),"image_paths":[str(image)]})
            for folder,values in ((train,rows[:1]),(test,rows)):
                write_json(folder/"detections/detector_config.json",config)
                write_jsonl(folder/"manifest/unique_images.jsonl",values)
                write_jsonl(folder/"manifest/shards/shard_00000.jsonl",values)
            source=train/"detections/text/shard_00000.jsonl"
            donor={"image_id":"eval_0","width":12,"height":24,"image":rows[0]["image_path"],
                   "text_detections":[{"bbox":[2,3,5,9]}],"inference_ms":123.}
            write_jsonl(source,[donor])
            write_json(source.with_suffix(".done.json"),{"stage":"text","count":1,
                "image_id_digest":audit.digest_ids(["eval_0"])})
            protected={p:p.read_bytes() for p in train.rglob("*") if p.is_file()}
            fake_task=SimpleNamespace(task_key="crop",view_policy="crops")
            with mock.patch.object(cache,"UI9_TASKS",(fake_task,)),mock.patch.object(cache,"paths_for",
                    side_effect=lambda root,task,split:{"cache":Path(root)/split}):
                summary=cache.seed_cross_task_detections(root)
            self.assertEqual(summary[f"{root.name}/test/text"]["cross_task_seed_images"],1)
            shard=test/"manifest/shards/shard_00000.jsonl";members=shard.read_bytes()
            seen=[]
            class Detector:
                def __init__(self,*a,**kw):pass
                def predict(self,image,**kw):
                    seen.append(image.getpixel((0,0))[0]);return [{"bbox":[1,2,4,8]}]
            args.detector_stage="text";args.worker_index=0;args.worker_count=1
            original_open=Image.open
            def guarded_open(path,*a,**kw):
                if str(path)==rows[0]["image_path"]:raise AssertionError("cached image was decoded")
                return original_open(path,*a,**kw)
            with mock.patch.object(audit,"load_parser_module",return_value=SimpleNamespace(PaddleTextDetector=Detector)),\
                 mock.patch.object(Image,"open",side_effect=guarded_open):
                audit.run_detector_worker(args)
            self.assertEqual(seen,[1]);self.assertEqual(shard.read_bytes(),members)
            outputs=list(read_jsonl(test/"detections/text/shard_00000.jsonl"))
            self.assertEqual([r["image_id"] for r in outputs],["eval_0","eval_1"])
            self.assertEqual(outputs[0]["text_detections"],donor["text_detections"])
            self.assertEqual(outputs[0]["inference_ms"],0)
            self.assertEqual(protected,{p:p.read_bytes() for p in protected})
            with mock.patch.object(audit,"load_parser_module",side_effect=AssertionError("model loaded")),\
                 mock.patch.object(Image,"open",side_effect=AssertionError("image opened")):
                audit.run_detector_worker(args)
            altered={**config,"version":"different"}
            with self.assertRaisesRegex(ValueError,"configuration"):
                cache.load_detector_reuse(test,"text",shard,altered)


if __name__=="__main__":unittest.main()
