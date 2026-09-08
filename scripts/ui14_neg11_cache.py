"""Import immutable cache evidence into private manifests; append new shards."""
from pathlib import Path
from ui14_common import *
from ui14_neg11_data import independent_copy, save_if_changed
from ui14_progress import track
from ui14_verification import checksum


def import_parent_cache(root, parent):
    root,parent=Path(root),Path(parent)
    summary={"files_copied":0,"files_already_imported":0,"parent_pngs_referenced":0,"tasks":{}}
    for task in UI9_TASKS:
        if task.view_policy!="crops": continue
        for split in ("train","test"):
            old,new=paths_for(parent,task.task_key,split),paths_for(root,task.task_key,split)
            receipt=new["cache"]/"neg11_parent_import.json"
            if receipt.exists():
                summary["tasks"][f"{task.task_key}/{split}"]=read_json(receipt)
                continue
            if not (old["cache"]/"ui14_crop_complete.json").exists():
                raise ValueError(f"Parent crop cache is incomplete: {old['cache']}")
            copied=0; identities={}
            # Private JSON, shard/done, geometry, and per-image label index copies.
            # PNG locations inside the index continue pointing to old immutable PNGs.
            for src in track(sorted(old["cache"].rglob("*")),f"{task.task_key}/{split} 导入父缓存清单",unit="文件"):
                if not src.is_file() or src.suffix not in (".json", ".jsonl", ".csv", ".html"): continue
                relative=src.relative_to(old["cache"])
                if any(p in ("_worker_logs","progress") for p in relative.parts) or src.name=="run_status.json": continue
                target=new["cache"]/relative
                copied+=independent_copy(src,target)
                identities[str(src)]=file_digest(src)
            for key in ("derived",):
                for src in (old[key],old[key].with_suffix(".coverage.json")):
                    independent_copy(src,new[key] if src==old[key] else new[key].with_suffix(".coverage.json"))
            pngs={r["image"] for r in read_jsonl(old["derived"])}
            value={"parent_cache":str(old["cache"]),"parent_files":identities,
                   "private_files_copied":copied,"parent_pngs_referenced":len(pngs),
                   "mutable_hardlinks":False,"geometry_policy":"content+dimensions+detector configuration",
                   "label_policy":"task+split+labels+normalization identity"}
            write_json(receipt,value)
            summary["tasks"][f"{task.task_key}/{split}"]=value
            summary["files_copied"]+=copied
    summary["parent_pngs_referenced"]=sum(v["parent_pngs_referenced"] for v in summary["tasks"].values())
    write_json(root/"cache_import_summary.json",summary)
    return summary


def cache_status(root):
    """Counts describe complete shard memberships, never crop/record aliases."""
    result={}
    for task in UI9_TASKS:
        if task.view_policy!="crops": continue
        for split in ("train","test"):
            p=paths_for(root,task.task_key,split); cache=p["cache"]
            info={"text":{},"icon":{}}
            old_ids=set()
            imported=cache/"neg11_parent_import.json"
            if imported.exists():
                old_ids={r["image_id"] for r in read_jsonl(Path(read_json(imported)["parent_cache"])/"manifest/unique_images.jsonl")}
            for stage in ("text","icon"):
                complete=set(); required=set(); cross_reused=set()
                for shard in sorted((cache/"manifest/shards").glob("shard_*.jsonl")):
                    ids=[r["image_id"] for r in read_jsonl(shard)]; required.update(ids)
                    dest=cache/"detections"/stage/shard.name
                    done=dest.with_suffix(".done.json")
                    from run_ui5_crop_audit import completed_shard_valid
                    if completed_shard_valid(shard,dest,done,stage):
                        got=[r["image_id"] for r in read_jsonl(dest)]
                        if set(got)==set(ids): complete.update(ids)
                        cross_reused.update(r["image_id"] for r in read_jsonl(dest) if r.get("reused_from_cache"))
                info[stage]={"total":len(required),"completed":len(complete),"pending":len(required-complete),
                             "parent_reused":len(complete&old_ids),"new_completed":len(complete-old_ids),
                             "cross_task_reused":len(complete&cross_reused),
                             "new_inferred_completed":len(complete-old_ids-cross_reused)}
            result[f"{task.task_key}/{split}"]=info
            for stage,value in info.items():
                print(f"[neg11 cache] {task.task_key}/{split} {stage}: {value['completed']}/{value['total']} "
                      f"parent_reused={value['parent_reused']} new_completed={value['new_completed']} "
                      f"pending={value['pending']} output={cache}",flush=True)
    write_json(Path(root)/"cache_reuse_summary.json",result)
    return result


def seed_cross_task_detections(root, ui5_cache=None):
    """CPU-only: match complete detections by full byte ID, dimensions and config.

    The pool comes from private imported manifests and the immutable UI5 cache.
    Labels, source-image RGB IDs and defect/normal pairing never enter this key.
    Seed rows let a partially cached shard infer only its missing images without
    changing the prepared shard's membership or order.
    """
    from run_ui5_crop_audit import completed_shard_valid, digest_ids
    caches=[paths_for(root,t.task_key,s)["cache"] for t in UI9_TASKS
            if t.view_policy=="crops" for s in ("train","test")]
    donors=caches+([Path(ui5_cache)] if ui5_cache else [])
    pool={}
    for cache in track(donors,"CPU 跨任务检测内容索引",unit="cache"):
        config_path=cache/"detections/detector_config.json"
        unique_path=cache/"manifest/unique_images.jsonl"
        if not config_path.is_file() or not unique_path.is_file(): continue
        config_id=checksum(read_json(config_path))
        images={r["image_id"]:r for r in read_jsonl(unique_path)}
        for stage in ("text","icon"):
            for shard in sorted((cache/"manifest/shards").glob("shard_*.jsonl")):
                output=cache/"detections"/stage/shard.name
                if not completed_shard_valid(shard,output,output.with_suffix(".done.json"),stage): continue
                provenance={"output":str(output),"sha256":file_digest(output)}
                for row in read_jsonl(output):
                    meta=images[row["image_id"]]
                    if (not meta.get("content_id") or not isinstance(row.get(stage+"_detections"),list)
                            or (row.get("width"),row.get("height"))!=(meta["width"],meta["height"])): continue
                    key=(config_id,stage,meta["content_id"],meta["width"],meta["height"])
                    pool.setdefault(key,{"detection":row,"source":provenance,"content_id":meta["content_id"]})
    result={}
    for cache in caches:
        config_id=checksum(read_json(cache/"detections/detector_config.json"))
        for stage in ("text","icon"):
            reused=0;complete=0
            for shard in sorted((cache/"manifest/shards").glob("shard_*.jsonl")):
                output=cache/"detections"/stage/shard.name
                if completed_shard_valid(shard,output,output.with_suffix(".done.json"),stage): continue
                rows=list(read_jsonl(shard));seeds={}
                for row in rows:
                    key=(config_id,stage,row["content_id"],row["width"],row["height"])
                    if key in pool: seeds[row["image_id"]]=pool[key]
                payload={"schema_version":1,"stage":stage,"config_digest":config_id,
                         "shard_sha256":file_digest(shard),"rows":seeds}
                payload["digest"]=checksum(payload)
                save_if_changed(cache/"detections/reuse"/stage/(shard.stem+".json"),payload)
                reused+=len(seeds)
                if seeds and len(seeds)==len(rows):
                    values=load_detector_reuse(cache,stage,shard,read_json(cache/"detections/detector_config.json"))
                    write_jsonl(output,[values[r["image_id"]] for r in rows])
                    write_json(output.with_suffix(".done.json"),{"stage":stage,"count":len(rows),
                        "image_id_digest":digest_ids(r["image_id"] for r in rows),"input_shard":str(shard)})
                    complete+=1
            result[f"{cache.parent.name}/{cache.name}/{stage}"]={"cross_task_seed_images":reused,"completed_shards":complete}
    write_json(Path(root)/"cross_task_detector_reuse.json",result)
    return result


def load_detector_reuse(cache, stage, shard, config):
    """GPU handoff reads JSON only; no image lookup/hash or parent mutation."""
    path=Path(cache)/"detections/reuse"/stage/(Path(shard).stem+".json")
    if not path.is_file(): return {}
    value=read_json(path); claimed=value.pop("digest")
    if checksum(value)!=claimed or value["config_digest"]!=checksum(config):
        raise ValueError("Detector reuse configuration/evidence changed")
    if value.get("stage")!=stage: raise ValueError("Detector reuse stage mismatch")
    if file_digest(shard)!=value["shard_sha256"]: raise ValueError("Detector reuse shard changed; run cache-prepare")
    rows={r["image_id"]:r for r in read_jsonl(shard)}; result={}
    for image_id,seed in value["rows"].items():
        row=rows[image_id]; detection=seed["detection"]
        if (seed["content_id"]!=row["content_id"] or
                (detection["width"],detection["height"])!=(row["width"],row["height"])):
            raise ValueError("Detector reuse image identity/dimensions changed")
        result[image_id]={**detection,"image_id":image_id,"image":row["image_path"],
                          "reused_from_cache":seed["source"],"inference_ms":0.0}
    return result
