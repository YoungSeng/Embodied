"""Assemble and check the composite dataset without repairing/redecoding its parent."""
from collections import Counter
from pathlib import Path
from ui14_common import *
from ui14_neg11_data import SYNTH, save_if_changed, validate_extension
from ui14_annotations import training_record, answer, crop_boxes
from ui14_verification import current_checks
from ui14_progress import track
from eaglevl.train.ui_defect_data import (identify_ui_defect_task, is_positive_ui_defect, negative_kind,
    recipe_sampling_ratio, build_task_source_balanced_rotating_plan, materialize_task_source_balanced_rotating_indices)


def image_counts(rows):
    positives={r["source_image_id"] for r in rows if r["boxes_px"]}
    negatives={r["source_image_id"] for r in rows if not r["boxes_px"]}-positives
    return {"positive_images":len(positives),"negative_images":len(negatives),"total_images":len(positives|negatives)}


def finalize(args):
    root,parent=Path(args.data_root),Path(args.parent_root)
    snapshot=validate_extension(root)
    parent_recipe=read_json(parent/"training_recipe.json")
    parent_eval=read_json(parent/"evaluation_manifest.json")
    registry=read_json(root/"task_registry.json")
    recipes={}; evaluation=[]; statistics={}; originals={}
    for task in track(UI_TASKS,"neg11 组装 recipe/14 项评测",unit="任务"):
        spec={**registry["tasks"][task.task_id]}
        if task.task_id<7:
            # UI5 and the two annotated tasks retain their exact parent streams.
            recipes[task.task_key]={**parent_recipe[task.task_key]}
            old_spec=parent_eval["tasks"][task.task_id]
            evaluation.append({**old_spec,"normalization_id":snapshot["normalization_id"],
                "cache":str(paths_for(root,task.task_key,"test")["cache"]) if task.task_id>=5 and task.view_policy=="crops" else old_spec["cache"]})
            spec.update(train=recipes[task.task_key]["annotation"],test=old_spec["test"],normalization_id=snapshot["normalization_id"])
            registry["tasks"][task.task_id]=spec
            continue
        for split in ("train","test"):
            p=paths_for(root,task.task_key,split); rows=list(read_jsonl(p["normalized"]))
            if task.view_policy=="crops":
                from ui14_crop_materialization import load_completed
                derived=load_completed(root,task,split,rows)
            else:
                derived=[training_record(r,task,r["source_image"],r["boxes_px"],r["width"],r["height"]) for r in rows]
                save_if_changed(p["derived"],derived,True)
            counts=image_counts(rows)
            if counts["positive_images"]!=counts["negative_images"]:
                raise ValueError(f"Independent image quota not 1:1: {task.task_key}/{split}: {counts}")
            counts.update(derived_records=len(derived),derived_positive=sum(is_positive_ui_defect(r) for r in derived),
                          derived_negative=sum(not is_positive_ui_defect(r) for r in derived),
                          negative_kinds=dict(Counter(negative_kind(r) for r in derived)))
            originals[f"{task.task_key}/{split}"]=counts
            if split=="train":
                train=derived
                recipes[task.task_key]={**parent_recipe[task.task_key],"annotation":str(p["derived"]),"length":len(derived),
                    "negative_to_positive_ratio":1.0,"negative_pool_balance":"source_image","extension_id":snapshot["extension_id"]}
            else:
                by_image={}
                for row in rows:
                    key=row["source_image_id"]
                    if key not in by_image:
                        by_image[key]={k:v for k,v in row.items() if k!="source_metadata"}
                        by_image[key].update(source_record_ids=[],boxes_px=[])
                    by_image[key]["source_record_ids"].append(row["source_record_id"])
                    for box in row["boxes_px"]:
                        if box not in by_image[key]["boxes_px"]: by_image[key]["boxes_px"].append(box)
                test=root/"evaluation_inputs"/f"{task.task_key}.jsonl"
                save_if_changed(test,list(by_image.values()),True)
                spec.update(train=recipes[task.task_key]["annotation"],test=str(test),normalization_id=snapshot["normalization_id"])
                evaluation.append({**spec,"split":"test","cache":str(p["cache"]) if task.view_policy=="crops" else None,
                    "skip_figma":False,"scan_name":SCAN_NAME,"expected_records":len(by_image),
                    "positive_count":counts["positive_images"],"negative_count":counts["negative_images"],
                    "data_sha256":file_digest(test),"parent_positive_test":parent_eval["tasks"][task.task_id]["test"]})
        registry["tasks"][task.task_id]=spec
    # The exact ratio used by the real dataset loader also drives this simulation.
    for task in UI_TASKS:
        entry=recipes[task.task_key]; train=list(read_jsonl(entry["annotation"]))
        ratio=recipe_sampling_ratio(entry,2.0)
        plan=build_task_source_balanced_rotating_plan(train,negative_to_positive_ratio=ratio)
        draws=materialize_task_source_balanced_rotating_indices(plan,seed=42)
        statistics[task.task_key]={"sampling_probability":1/14,"negative_to_positive_ratio":ratio,
            "source_images":len({r["source_image_id"] for r in train}),"train_records":len(train),
            "derived_positive":sum(is_positive_ui_defect(r) for r in train),
            "derived_negative":sum(not is_positive_ui_defect(r) for r in train),"epoch_draws":len(draws),
            "sampled_positive":sum(is_positive_ui_defect(train[i]) for i in draws),
            "sampled_negative":sum(not is_positive_ui_defect(train[i]) for i in draws),
            "sampled_negative_kinds":dict(Counter(negative_kind(train[i]) for i in draws)),
            "image_counts":originals.get(f"{task.task_key}/train")}
    eval_id=digest({"normalization_id":snapshot["normalization_id"],
                    "test_files":{r["task_key"]:file_digest(r["test"]) for r in evaluation},
                    "scoring":"UI5 no_figma; UI9 all; IoU=.1; illegal outputs penalized"})
    for spec in evaluation: spec["eval_set_id"]=eval_id
    save_if_changed(root/"training_recipe.json",recipes)
    save_if_changed(root/"task_registry.json",registry)
    save_if_changed(root/"evaluation_manifest.json",{"schema_version":3,"tasks":evaluation,"eval_set_id":eval_id,
        "repair_run_id":snapshot["repair_run_id"],"normalization_id":snapshot["normalization_id"],
        "parent_normalization_id":snapshot["parent_normalization_id"],"extension_id":snapshot["extension_id"]})
    save_if_changed(root/"sampling_stats.json",statistics)
    save_if_changed(root/"negative_image_counts.json",originals)
    return check(args)


def check(args):
    root,parent=Path(args.data_root),Path(args.parent_root)
    snap=validate_extension(root); checks=current_checks()
    write_json(root/"cpu_check_report.json",{"ready":False,"normalization_complete":True,
        "normalization_id":snap["normalization_id"],"stage":"neg11_final_cpu_check"})
    recipe=read_json(root/"training_recipe.json"); evaluation=read_json(root/"evaluation_manifest.json")
    registry=read_json(root/"task_registry.json")
    from eaglevl.ui_task_registry import validate_registry
    validate_registry(registry["tasks"],14); validate_registry(evaluation["tasks"],14)
    if set(recipe)!={t.task_key for t in UI_TASKS}: raise ValueError("Recipe must contain exactly 14 tasks")
    selected=list(read_jsonl(root/"negative_selection.jsonl"))
    selected_by={(r["task_key"],r["split"],r["source_image_id"]):r for r in selected}
    bindings={}; external={}; task_results={}
    parent_recipe=read_json(parent/"training_recipe.json"); parent_eval=read_json(parent/"evaluation_manifest.json")
    for task in track(UI_TASKS,"CPU neg11 路由/标签/配额/缓存连接",unit="任务"):
        spec=evaluation["tasks"][task.task_id]; entry=recipe[task.task_key]
        if task.task_id<7:
            if entry!=parent_recipe[task.task_key] or spec["test"]!=parent_eval["tasks"][task.task_id]["test"]:
                raise ValueError("Non-synthetic source train/test/sampling changed")
            for name in (entry["annotation"],spec["test"]): external[name]=file_digest(name)
        else:
            if recipe_sampling_ratio(entry)!=1.0: raise ValueError("Synthetic sampler ratio must be 1")
            for split in ("train","test"):
                p=paths_for(root,task.task_key,split); rows=list(read_jsonl(p["normalized"]))
                old=list(read_jsonl(paths_for(parent,task.task_key,split)["normalized"]))
                if len(rows)<len(old): raise ValueError("Parent records were removed")
                for current,previous in zip(rows,old):
                    if any(current.get(k)!=v for k,v in previous.items() if k!="normalization_id"):
                        raise ValueError("Frozen parent annotation was modified")
                counts=image_counts(rows)
                if counts["positive_images"]!=counts["negative_images"]: raise ValueError("Original image quota must be 1:1")
                by_record={r["source_record_id"]:r for r in rows}
                checks.prefetch_images(r["source_image"] for r in rows)
                for r in rows:
                    if image_identity(r["source_image"])!=(r["source_image_id"],r["width"],r["height"]): raise ValueError("Source image changed")
                    if r["split"]!=split or r["task_id"]!=task.task_id or r["normalization_id"]!=snap["normalization_id"]:
                        raise ValueError("Normalization route/version/split changed")
                    if r.get("clean_source_image") and not r.get("parent_record"):
                        c=selected_by[(task.task_key,split,r["source_image_id"])]
                        if r["source_image"]!=c["source_image"] or r["image"]!=c["source_image"] or r["boxes_px"] or r["is_positive"] is not False:
                            raise ValueError("Negative main image/bbox is invalid")
                        if not r["negative_evidence"]: raise ValueError("Negative evidence missing")
                if task.view_policy=="crops":
                    from ui14_crop_materialization import load_completed
                    derived=load_completed(root,task,split,rows,full=getattr(args,"full_verify",False))
                    from prepare_ui14_sft import validate_task_cache
                    validate_task_cache(root,task,split,sum(1 for _ in read_jsonl(p["cache"]/"manifest/unique_images.jsonl")))
                    for name in ("ui14_crop_complete.json","ui14_label_cache_ready.json","crop_index/images.jsonl",
                                 SCAN_NAME+"/detector_scan_crops.jsonl",SCAN_NAME+"/eval_detector_cache_ready.json"):
                        f=p["cache"]/name; bindings[str(f.relative_to(root))]=file_digest(f)
                else: derived=list(read_jsonl(p["derived"]))
                for r in derived:
                    source=by_record[r["source_record_id"]]; boxes=source["boxes_px"]
                    if r.get("crop_box"): boxes,_=crop_boxes(boxes,r["crop_box"])
                    if identify_ui_defect_task(r)[1]!=task.task_id or r["split"]!=split: raise ValueError("Derived routing changed")
                    if r["conversations"]!=[{"from":"human","value":"<image>\n"+task.prompt},
                                           {"from":"gpt","value":answer(boxes,r["width"],r["height"],task.prompt_label)}]:
                        raise ValueError("Derived answer/coordinates changed")
                    if r.get("crop_image_sha256") and file_digest(r["image"])!=r["crop_image_sha256"]:
                        raise ValueError("Crop bytes changed")
                for name in ("normalized","derived","detector_input","detector_inputs"):
                    bindings[str(p[name].relative_to(root))]=file_digest(p[name])
            if len(list(read_jsonl(spec["test"])))!=spec["expected_records"]: raise ValueError("Test count changed")
            if file_digest(spec["test"])!=spec["data_sha256"]: raise ValueError("Test digest changed")
            bindings[str(Path(spec["test"]).relative_to(root))]=file_digest(spec["test"])
        task_results[task.task_key]="pass"
    from run_ui14_eval import validate_evaluation_manifest
    validate_evaluation_manifest(root/"evaluation_manifest.json")
    from ui14_checks import validate_initial_checkpoint, render_formal_yaml
    initialization=validate_initial_checkpoint(Path(args.init_checkpoint))
    yaml_path,runtime_path=render_formal_yaml(root,profile="m32-cpt9000-ui14-neg11-v1")
    checks.export_images(root/"verification/image_evidence.jsonl")
    for name in ("negative_extension_manifest.json","negative_selection.jsonl","negative_page_assignments.json",
                 "inventory_summary.json","negative_split_overlap.json","source_snapshot.json","normalization_stats.json","task_registry.json",
                 "training_recipe.json","evaluation_manifest.json","negative_image_counts.json","sampling_stats.json",
                 "verification/image_evidence.jsonl",yaml_path.name,runtime_path.name): bindings[name]=file_digest(root/name)
    external[str(Path(args.init_checkpoint)/"config.json")]=file_digest(Path(args.init_checkpoint)/"config.json")
    report={"ready":True,"cpu_only":True,"gpu_loaded":False,"registry_count":14,"evaluation_count":14,"tasks":task_results,
        "repair_run_id":snap["repair_run_id"],"normalization_id":snap["normalization_id"],"extension_id":snap["extension_id"],
        "parent_normalization_id":snap["parent_normalization_id"],"eval_set_id":evaluation["eval_set_id"],
        "init_checkpoint":str(args.init_checkpoint),"init_cpt_step":9000,"sft_start_step":0,"initialization":initialization,
        "artifact_digests":bindings,"external_digests":external,"verification":checks.counts,
        "negative_counts":read_json(root/"negative_image_counts.json"),"errors":[]}
    report["split_overlap"]=read_json(root/"negative_split_overlap.json")
    write_json(root/"cpu_check_report.json",report)
    return report
