"""CPU integration: isolated parent, real image evidence, stable shards and PNGs."""
import contextlib
from collections import Counter
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT))
from PIL import Image
from ui14_common import *
import ui14_neg11_data as data
import ui14_neg11_finalize as final
from ui14_verification import verification_session
from ui14_cache_prepare import ImageInfoJournal, publish_prepared, validate_prepared
from ui14_neg11_cache import import_parent_cache, cache_status
from tests.test_ui14_pipeline import source_fixture, detector_fixture, make_record
import prepare_ui14_sft as prepare
import prepare_ui5_eval_detector_crops as detector
from eaglevl.train.ui_defect_data import negative_kind, recipe_sampling_ratio, build_task_source_balanced_rotating_plan, materialize_task_source_balanced_rotating_indices


def make_parent(root):
    source=root/"source"; parent=root/"parent"; source_fixture(source)
    for task in UI9_TASKS:
        for i,split in enumerate(("train","test")):
            path=source/task.task_key/f"{split}.jsonl"; rows=list(read_jsonl(path)); raw=rows[0]
            image=Path(raw["images"][0]); Image.new("RGB",(750,1600),(task.task_id*13,20+i*30,50)).save(image)
            if task.task_id>=7:
                normal=source/task.task_key/"sample_imgs/_folders/version/local_imgs"/f"normal-{split}.png"
                normal.parent.mkdir(parents=True,exist_ok=True)
                Image.new("RGB",(750,1600),(task.task_id*13,120+i*30,100)).save(normal)
                raw.update(LocalImgURL=str(normal),RawImgURL=str(normal))
            write_jsonl(path,rows)
    from locany_ui5_common import TASK_JSONL
    audit=root/"audit/crop_audit_v4_gt_repair"; recipe=audit/"training_recipes/old.json"; entries={}; samples=[]
    for task in UI5_TASKS:
        image=root/f"ui5-{task.task_id}.png"; Image.new("RGB",(375,800),"green").save(image)
        row=make_record(task.task_id,True,f"ui5-{task.task_id}"); row["image"]=str(image)
        row["_ui5_crop_source"]="manual_gt_repair" if task.task_id<4 else "full_image"
        row.pop("source_image");row.pop("_ui5_source_image")
        samples.append(dict(sample_id=row["_ui5_sample_id"],image_id=row["_ui5_image_id"],canonical_path=str(image)))
        train=root/f"oldtrain-{task.task_id}.jsonl";write_jsonl(train,[row]);entries[task.task_key]={"annotation":str(train),"root":str(root)}
        write_jsonl(root/"test"/TASK_JSONL[task.task_key],[{"images":[str(image)],"objects":{"bbox":[]}}])
    write_json(recipe,entries);write_jsonl(audit.parent/"manifest/task_samples.jsonl",samples)
    ui5cache=root/"ui5cache";write_json(ui5cache/SCAN_NAME/"eval_detector_cache_ready.json",{"fixture":True})
    init=root/"checkpoint-9000";write_json(init/"config.json",{"model_type":"locateanything"});(init/"model.safetensors").write_bytes(b"CPU fixture only")
    args=SimpleNamespace(ui9_data_root=source,output_dir=parent,ui5_recipe=recipe,ui5_test_dir=root/"test",ui5_cache=ui5cache,init_checkpoint=init)
    prepare.normalize(args)
    for task in UI9_TASKS:
        if task.view_policy!="crops":continue
        for split in ("train","test"):
            detector_fixture(parent,task,split)
            prepare.crop_annotations(parent,task,split,list(read_jsonl(paths_for(parent,task.task_key,split)["normalized"])))
    from ui5_eval_detector_cache import validate_eval_detector_cache as actual
    def validate(cache,*a,**kw): return {} if Path(cache)==ui5cache else actual(cache,*a,**kw)
    with mock.patch("run_ui5_crop_audit.validate_training_ready_marker",return_value={"crop_train_mode":"crop_only"}),\
         mock.patch("ui5_eval_detector_cache.validate_eval_detector_cache",side_effect=validate):
        prepare.finalize(args)
    return source,parent,init,ui5cache


class Neg11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()): cls.source,cls.parent,cls.init,cls.ui5cache=make_parent(cls.root)
        cls.parent_bytes={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in cls.parent.rglob("*") if p.is_file()}

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def options(self,name):
        return SimpleNamespace(data_root=self.root/name,parent_root=self.parent,source_root=self.source,
                               init_checkpoint=self.init,ui5_cache=self.ui5cache,full_verify=False)

    def normalize_new(self,args,missing_negatives=0):
        data.seed_evidence(args.data_root,args.parent_root)
        with verification_session(args.data_root):
            report=data.inventory(args)
            self.assertEqual(report["gap"],missing_negatives)
            self.assertEqual(report["selected_count"],14-missing_negatives)
            data.normalize(args)

    def test_full_composite_cpu_path_and_incremental_cache(self):
        args=self.options("new");root=args.data_root
        with contextlib.redirect_stdout(io.StringIO()):
            evidence=data.eligible_evidence
            with mock.patch.object(data,"eligible_evidence",side_effect=lambda raw,role,task:
                                   None if task=="synth_radius" else evidence(raw,role,task)):
                self.normalize_new(args,missing_negatives=2)
            with verification_session(root):
                summary=import_parent_cache(root,self.parent)
                self.assertGreater(summary["parent_pngs_referenced"],0)
                journal=ImageInfoJournal(root/"cache_preparation/image_info.jsonl",2)
                for task in UI9_TASKS:
                    if task.view_policy!="crops":continue
                    for split in ("train","test"):
                        p=paths_for(root,task.task_key,split);old=paths_for(self.parent,task.task_key,split)
                        old_shards={p.name:p.read_bytes() for p in (old["cache"]/"manifest/shards").glob("*.jsonl")}
                        opts=detector.parse_args(["--stage","prepare","--input-dir",str(p["detector_input"].parent),
                            "--task-input-manifest",str(p["detector_inputs"]),"--data-split",split,"--output-dir",str(p["cache"]),
                            "--parser-root",str(root),"--cache-scope","full_test" if split=="test" else "full_train",
                            "--expected-unique-images",str(2 if task.task_id>=7 and task.task_key!="synth_radius" else 1),"--resume","--no-skip-figma","--visualization-samples","1"])
                        config=read_json(p["cache"]/"detections/detector_config.json")
                        with mock.patch.object(detector,"detector_config",return_value=config):
                            unique=detector.prepare_manifest(opts,image_info_loader=journal.load,allow_selection_refresh=True)
                        for name,content in old_shards.items():self.assertEqual((p["cache"]/"manifest/shards"/name).read_bytes(),content)
                        config=read_json(p["cache"]/"detections/detector_config.json")
                        publish_prepared(p,read_json(root/"source_snapshot.json")["normalization_id"],config,len(unique))
                        self.assertEqual(validate_prepared(p,read_json(root/"source_snapshot.json")["normalization_id"],config),len(unique))
                        from run_ui5_crop_audit import completed_shard_valid
                        for stage in ("text","icon"):
                            for name in old_shards:
                                self.assertTrue(completed_shard_valid(p["cache"]/"manifest/shards"/name,
                                    p["cache"]/"detections"/stage/name,p["cache"]/"detections"/stage/Path(name).with_suffix(".done.json"),stage))
                            for shard in (p["cache"]/"manifest/shards").glob("*.jsonl"):
                                if shard.name in old_shards:continue
                                added=list(read_jsonl(shard))
                                write_jsonl(p["cache"]/"detections"/stage/shard.name,[{"image_id":r["image_id"]} for r in added])
                                write_json(p["cache"]/"detections"/stage/(shard.stem+".done.json"),{"stage":stage,"count":len(added),
                                    "image_id_digest":detector.digest_ids(r["image_id"] for r in added)})
                            stage_path=p["cache"]/"detections"/stage/"stage_summary.json"
                            write_json(stage_path,{**read_json(stage_path),"images":len(unique)})
                        previous={r["image_id"]:r for r in read_jsonl(old["cache"]/"detections/merged/detections.jsonl")}
                        # Only new images receive synthetic detector results in this CPU fixture.
                        merged=[previous.get(r["image_id"],{**r,"text_detections":[],"icon_detections":[]}) for r in unique]
                        write_jsonl(p["cache"]/"detections/merged/detections.jsonl",merged)
                        with mock.patch.object(detector,"generate_detector_scan_plan",wraps=detector.generate_detector_scan_plan) as plan:
                            detector.build_scan_crops(opts)
                            self.assertEqual(plan.call_count,1 if task.task_id>=7 and task.task_key!="synth_radius" else 0)
                        from ui14_crop_materialization import materialize_split
                        normalized=list(read_jsonl(p["normalized"]))
                        derived=materialize_split(root,task,split,normalized)
                        old_paths={r["image"] for r in read_jsonl(old["derived"])}
                        self.assertTrue(old_paths.issubset({r["image"] for r in derived}))
                        with mock.patch.object(Image,"open",side_effect=AssertionError("completed crop reopened")):
                            self.assertEqual(materialize_split(root,task,split,normalized),derived)
                from ui5_eval_detector_cache import validate_eval_detector_cache as actual
                def validate(cache,*a,**kw):return {} if Path(cache)==self.ui5cache else actual(cache,*a,**kw)
                with mock.patch("ui5_eval_detector_cache.validate_eval_detector_cache",side_effect=validate):
                    final.finalize(args)
                    with mock.patch.object(Image,"open",side_effect=AssertionError("repeat finalize image read")):
                        final.finalize(args)
                recipe=read_json(root/"training_recipe.json")
                for task in UI_TASKS:
                    self.assertEqual(recipe_sampling_ratio(recipe[task.task_key]),1. if task.task_id>=7 else 2.)
                for key,r in read_json(root/"negative_image_counts.json").items():
                    self.assertEqual((r["positive_images"],r["negative_images"]),(1,0) if key.startswith("synth_radius/") else (1,1))
                report=read_json(root/"cpu_check_report.json")
                self.assertTrue(report["ready"])
                self.assertEqual(report["negative_quota_policy"],"available")
                self.assertEqual(report["one_to_one_shortfall"],2)
                spec=read_json(root/"evaluation_manifest.json")["tasks"][9]
                self.assertEqual((spec["positive_count"],spec["negative_count"]),(1,0))
                before={p:p.read_bytes() for p in (root/"training_recipe.json",root/"task_registry.json",root/"evaluation_manifest.json")}
                with mock.patch.object(Image,"open",side_effect=AssertionError("repeat normalize image read")):
                    data.normalize(args)
                self.assertEqual(before,{p:p.read_bytes() for p in before})
                stats=read_json(root/"sampling_stats.json")
                for task in data.SYNTH:
                    if stats[task.task_key]["both_labels_available"]:
                        self.assertEqual(stats[task.task_key]["sampled_positive"],stats[task.task_key]["sampled_negative"])
        for p,(content,mtime) in self.parent_bytes.items():
            self.assertEqual(p.read_bytes(),content);self.assertEqual(p.stat().st_mtime_ns,mtime)

    def test_normal_main_image_empty_boxes_and_frozen_page(self):
        args=self.options("labels")
        with contextlib.redirect_stdout(io.StringIO()):self.normalize_new(args)
        selected=list(read_jsonl(args.data_root/"negative_selection.jsonl"))
        self.assertEqual(len({(r["task_key"],r["split"],r["source_image_id"]) for r in selected}),14)
        for task in data.SYNTH:
            for split in ("train","test"):
                rows=list(read_jsonl(paths_for(args.data_root,task.task_key,split)["normalized"]))
                self.assertEqual(rows[1]["boxes_px"],[]);self.assertFalse(rows[1]["is_positive"])
                self.assertNotEqual(rows[0]["source_image"],rows[1]["source_image"])
                self.assertEqual(rows[0]["source_page_id"],rows[1]["source_page_id"])
                from ui14_annotations import training_record
                record=training_record(rows[1],task,rows[1]["source_image"],[],750,1600)
                self.assertEqual(record["conversations"][1]["value"],"<box>none</box>")
                self.assertEqual(negative_kind(record),"clean_source_image")

    def test_no_raw_folder_or_other_task_negative_label_is_evidence(self):
        self.assertIsNone(data.eligible_evidence({"is_positive":False},"raw","synth_radius"))
        self.assertIsNone(data.eligible_evidence({"negative_evidence":{"clean_tasks":["synth_cropping"],"basis":"manual","provenance":"audit"}},"pool","synth_radius"))
        self.assertIsNotNone(data.eligible_evidence({"raw_is_pre_synthesis":True},"raw","synth_radius"))

    def test_cross_split_content_alias_is_rejected_and_deficit_blocks_normalize(self):
        args=self.options("conflict");data.seed_evidence(args.data_root,args.parent_root)
        args.negative_quota_policy="strict"
        original=data.ReferenceResolver.resolve
        def alias(resolver,value,task):
            if task=="synth_radius" and "normal-train" in str(value):
                return original(resolver,str(value).replace("normal-train","normal-test"),task)
            return original(resolver,value,task)
        with contextlib.redirect_stdout(io.StringIO()),verification_session(args.data_root),\
             mock.patch.object(data.ReferenceResolver,"resolve",alias):
            result=data.inventory(args)
            self.assertGreater(result["gap"],0)
            with self.assertRaisesRegex(ValueError,"quota not met"):data.normalize(args)
            self.assertFalse((args.data_root/"negative_extension_manifest.json").exists())

    def test_duplicate_normal_references_count_as_one_and_selection_resumes(self):
        args=self.options("dedup");data.seed_evidence(args.data_root,args.parent_root)
        original=data.image_slots
        def duplicate(raw):
            rows=list(original(raw));return iter(rows+rows)
        with contextlib.redirect_stdout(io.StringIO()),verification_session(args.data_root),mock.patch.object(data,"image_slots",duplicate):
            first=data.inventory(args)
            self.assertEqual(first["candidate_count"],14)
            self.assertEqual(first["selected_count"],14)
            with mock.patch.object(Image,"open",side_effect=AssertionError("inventory reread image")):
                second=data.inventory(args)
            self.assertEqual(first,second)

    def test_failed_legacy_inventory_reuses_selection_without_image_scan(self):
        args=self.options("legacy-gap");args.negative_quota_policy="strict"
        data.seed_evidence(args.data_root,args.parent_root)
        original=data.eligible_evidence
        with contextlib.redirect_stdout(io.StringIO()),verification_session(args.data_root):
            with mock.patch.object(data,"eligible_evidence",side_effect=lambda raw,role,task:
                                   None if task=="synth_radius" else original(raw,role,task)):
                inv=data.inventory(args)
            for key in ("negative_quota_policy","gap_blocks_normalize","sampling_negative_to_positive_ratio",
                        "evaluation_balance","zero_negative_splits","inventory_id"):
                inv.pop(key,None)
            inv["inventory_id"]=digest(inv)
            write_json(args.data_root/"inventory_summary.json",inv)
            protected=[args.data_root/name for name in ("negative_selection.proposed.jsonl","negative_page_assignments.json")]
            before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in protected}
            args.negative_quota_policy="available"
            with mock.patch.object(Image,"open",side_effect=AssertionError("migration reread image")),\
                 mock.patch.object(data,"inventory",side_effect=AssertionError("migration rescanned inventory")):
                data.normalize(args)
                snap=data.validate_extension(args.data_root)
                self.assertEqual(snap["negative_quota_policy"],"available")
                migrated=read_json(args.data_root/"inventory_summary.json")
                self.assertEqual(migrated["gap"],2)
                self.assertFalse(migrated["gap_blocks_normalize"])
                self.assertEqual(migrated["parent_inventory_id"],inv["inventory_id"])
                self.assertEqual(read_json(args.data_root/"inventory_history"/(inv["inventory_id"]+".json")),inv)
                # An explicit different CLI default cannot mutate a frozen set.
                args.negative_quota_policy="strict"
                data.normalize(args)
                self.assertEqual(data.validate_extension(args.data_root),snap)
            self.assertEqual(before,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in protected})

    def test_completed_legacy_strict_snapshot_survives_new_default(self):
        args=self.options("legacy-complete");args.negative_quota_policy="strict"
        data.seed_evidence(args.data_root,args.parent_root)
        with contextlib.redirect_stdout(io.StringIO()),verification_session(args.data_root):
            inv=data.inventory(args)
            for key in ("negative_quota_policy","gap_blocks_normalize","sampling_negative_to_positive_ratio",
                        "evaluation_balance","zero_negative_splits","inventory_id"):
                inv.pop(key,None)
            inv["inventory_id"]=digest(inv)
            write_json(args.data_root/"inventory_summary.json",inv)
            data.normalize(args)
            snap=data.validate_extension(args.data_root)
            self.assertNotIn("negative_quota_policy",snap)
            args.negative_quota_policy="available"
            with mock.patch.object(Image,"open",side_effect=AssertionError("legacy resume image read")):
                data.normalize(args)
                self.assertEqual(data.validate_extension(args.data_root),snap)

    def test_isolation_rejects_parent_or_training_output_writes(self):
        with self.assertRaises(ValueError):data.assert_isolated(self.parent,self.parent,self.source)
        with self.assertRaises(ValueError):data.assert_isolated(self.parent/"new",self.parent,self.source)

    def test_evidenced_pool_can_live_in_new_root_without_changing_source(self):
        args=self.options("new-pool");data.seed_evidence(args.data_root,args.parent_root)
        task=get_task("synth_radius")
        positive=next(read_jsonl(paths_for(self.parent,task.task_key,"train")["normalized"]))
        raw=positive["source_metadata"]
        normal=next(value for _,_,value,role in data.image_slots(raw) if role=="normal")
        path=data.ReferenceResolver(self.source).resolve(normal,task.task_key)
        pool=args.data_root/"normal_pools"/task.task_key/"verified.jsonl"
        write_jsonl(pool,[{"image":str(path),"source_dataset":task.source_dataset,
            "source_page_id":positive["source_page_id"],"negative_evidence":{
                "clean_tasks":[task.task_key],"basis":"source normal export","provenance":"fixture-pair"}}])
        with contextlib.redirect_stdout(io.StringIO()),verification_session(args.data_root):
            report=data.inventory(args)
            self.assertEqual(report["gap"],0);self.assertEqual(report["candidate_count"],14)
            self.assertEqual(report["files"][str(pool)],file_digest(pool))
            data.normalize(args)
        self.assertIn(str(pool),read_json(args.data_root/"negative_extension_manifest.json")["pool_files"])
        self.assertEqual(self.parent_bytes,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.parent_bytes})

    def test_ratio_source_balance_and_background_kind(self):
        records=[make_record(7,True,"p")]+[make_record(7,False,"b",str(i)) for i in range(8)]+[
            {**make_record(7,False,"clean",str(i)),"clean_source_image":True} for i in range(20)]
        plan=build_task_source_balanced_rotating_plan(records,recipe_sampling_ratio({"negative_to_positive_ratio":1.0}))
        draws=materialize_task_source_balanced_rotating_indices(plan,seed=42)
        counts=Counter(negative_kind(records[i]) for i in draws)
        self.assertEqual(counts["positive"],counts["background_crop"]+counts["clean_source_image"])
        self.assertLessEqual(abs(counts["background_crop"]-counts["clean_source_image"]),1)


if __name__=="__main__":unittest.main()
