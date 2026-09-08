"""Inspect raw references without admitting negatives or changing frozen splits."""
from collections import Counter, defaultdict
from pathlib import Path
import html

from ui14_common import read_jsonl, digest, image_identity, paths_for
from ui14_neg11_data import SYNTH, SEED, ReferenceResolver, parent_binding, save_if_changed
from ui14_progress import track
from ui14_verification import current_checks
from ui9_source_parser import image_slots


def audit_raw(args):
    root, parent, source = Path(args.data_root), Path(args.parent_root), Path(args.source_root)
    binding = parent_binding(parent)
    resolver = ReferenceResolver(source)
    entries = {}; missing = []; totals = {}; identities_to_splits = defaultdict(set)
    # Read normalized metadata, never infer clean labels from names/empty GT.
    for task in SYNTH:
        for split in ("train", "test"):
            rows = list(read_jsonl(paths_for(parent, task.task_key, split)["normalized"]))
            key = f"{task.task_key}/{split}"
            positive = [r for r in rows if r["boxes_px"]]
            totals[key] = {"positive_records":len(positive),
                          "positive_images":len({r["source_image_id"] for r in positive}),
                          "raw_reference_occurrences":0, "missing_reference_occurrences":0}
            for row in track(positive, key+" raw 引用审计", unit="记录"):
                raw = row.get("source_metadata", {})
                slots = list(image_slots(raw)); local = []
                for _, _, value, role in slots:
                    if role != "normal": continue
                    try: local.append(str(resolver.resolve(value, task.task_key)))
                    except (OSError, ValueError): pass
                for _, _, value, role in slots:
                    if role != "raw": continue
                    totals[key]["raw_reference_occurrences"] += 1
                    try: path = str(resolver.resolve(value, task.task_key))
                    except (OSError, ValueError) as exc:
                        totals[key]["missing_reference_occurrences"] += 1
                        missing.append({"task_key":task.task_key,"split":split,
                            "source_record_id":row["source_record_id"],"reference":value,"error":str(exc)})
                        continue
                    item = entries.setdefault((task.task_key,split,path), {
                        "task_key":task.task_key,"split":split,"raw_image":path,
                        "source_dataset":row["source_dataset"],"source_version":row["source_version"],
                        "raw_references":set(),"paired_positive_ids":set(),"paired_image_ids":set(),
                        "local_images":set(),"source_pages":set(),"examples":[]})
                    item["raw_references"].add(str(value))
                    item["paired_positive_ids"].add(row["source_record_id"])
                    item["paired_image_ids"].add(row["source_image_id"])
                    item["local_images"].update(local)
                    if row.get("source_page_id"): item["source_pages"].add(row["source_page_id"])
                    if len(item["examples"]) < 3:
                        item["examples"].append({"source_record_id":row["source_record_id"],
                            "image":row["source_image"],"boxes_px":row["boxes_px"],
                            "width":row["width"],"height":row["height"],"local_images":sorted(set(local)),
                            "source_metadata":raw})
    checks = current_checks()
    if checks is None: raise RuntimeError("audit-raw requires a CPU verification session")
    checks.prefetch_images([p for e in entries.values() for p in [e["raw_image"],*e["local_images"]]])
    unique = {}
    for item in track(entries.values(), "raw 内容身份去重（已验证项只 stat）", unit="路径"):
        identity,w,h = image_identity(item["raw_image"])
        local_ids = {image_identity(p)[0] for p in item["local_images"]}
        item.update(raw_image_id=identity,width=w,height=h,
                    equals_paired_defect=identity in item["paired_image_ids"],
                    equals_declared_local=identity in local_ids)
        key = (item["task_key"],item["split"],identity)
        identities_to_splits[identity].add(item["split"])
        item["raw_image_paths"] = {item["raw_image"]}
        if key in unique:
            existing = unique[key]
            for field in ("raw_references","paired_positive_ids","paired_image_ids",
                          "local_images","source_pages","raw_image_paths"):
                existing[field].update(item[field])
            for field in ("equals_paired_defect","equals_declared_local"):
                existing[field] |= item[field]
            existing["examples"] = (existing["examples"] + item["examples"])[:3]
        else: unique[key] = item
    selected = set()
    selection = root/"negative_selection.proposed.jsonl"
    if selection.is_file():
        selected = {(r["task_key"],r["split"],r["source_image_id"]) for r in read_jsonl(selection)}
    audit_rows = []
    for key,item in sorted(unique.items()):
        item["observed_raw_cross_split"] = len(identities_to_splits[item["raw_image_id"]]) > 1
        item["already_selected"] = key in selected
        item["clean_status"] = ("matches_paired_defect" if item["equals_paired_defect"] else
            "matches_declared_local_reference" if item["equals_declared_local"] else "unconfirmed")
        for field,value in list(item.items()):
            if isinstance(value,set): item[field] = sorted(value)
        audit_rows.append(item)
    for key,total in totals.items():
        task,split = key.split("/")
        values = [r for r in audit_rows if (r["task_key"],r["split"]) == (task,split)]
        total.update(raw_unique_images=len(values),
            raw_matches_defect_images=sum(r["equals_paired_defect"] for r in values),
            raw_matches_local_images=sum(r["equals_declared_local"] for r in values),
            raw_unconfirmed_images=sum(r["clean_status"]=="unconfirmed" for r in values),
            raw_observed_cross_split_images=sum(r["observed_raw_cross_split"] for r in values),
            raw_already_selected_images=sum(r["already_selected"] for r in values),
            unselected_raw_without_observed_conflict=sum(not (r["equals_paired_defect"] or
                r["observed_raw_cross_split"] or r["already_selected"]) for r in values),
            directory_counts=dict(Counter(str(Path(r["raw_image"]).parent) for r in values).most_common(8)))
        print(f"[audit-raw] {key}: raw_unique={total['raw_unique_images']} "
              f"unconfirmed={total['raw_unconfirmed_images']} same_defect={total['raw_matches_defect_images']} "
              f"same_local={total['raw_matches_local_images']} cross_split={total['raw_observed_cross_split_images']} "
              f"already_selected={total['raw_already_selected_images']}",flush=True)
    output = root/"raw_reference_audit"
    # Large association index omits raw JSON; bounded samples preserve the complete
    # source fields for checking dataset generation provenance.
    samples = []
    for task in SYNTH:
        samples.extend(sorted((r for r in audit_rows if r["task_key"]==task.task_key),
                       key=lambda r:digest([SEED,r["task_key"],r["raw_image_id"],r["split"]]))[:10])
    save_if_changed(output/"pairs.jsonl",[
        {**r,"examples":[{k:v for k,v in e.items() if k!="source_metadata"} for e in r["examples"]]}
        for r in audit_rows],True)
    save_if_changed(output/"source_metadata_examples.jsonl",samples,True)
    save_if_changed(output/"missing_references.jsonl",missing,True)
    summary = {"status":"complete","parent_normalization_id":binding["parent_normalization_id"],
        "repair_run_id":binding["repair_run_id"],"tasks":totals,"samples":len(samples),
        "sample_seed":SEED,"output_path":str(output),"image_verification":dict(checks.counts),
        "label_changes":0,"selection_changes":0,"raw_policy":"unconfirmed; not admitted as negatives",
        "split_scope":"observed raw references across seven synthetic tasks; inventory still enforces all 14 tasks",
        "count_scope":"raw occurrences are references; unique counts are RGB-content images per task/split",
        "next_action":"Verify original generator semantics or task-clean provenance; audit counts are not negative quotas."}
    save_if_changed(output/"summary.json",summary)
    write_gallery(output/"samples.html",samples)
    print(f"[audit-raw] no labels/selections changed; summary={output/'summary.json'}; samples={output/'samples.html'}",flush=True)
    return summary


def write_gallery(path, samples):
    parts = ['<!doctype html><meta charset="utf-8"><title>UI14 raw reference audit</title>',
        '<style>body{font:16px sans-serif}article{border:1px solid #ccc;margin:24px;padding:16px}'
        '.pair{display:flex;align-items:flex-start;gap:12px}figure{width:32%;margin:0}.frame{position:relative}'
        'img{width:100%}.gt{display:none;position:absolute;inset:0;width:100%;height:100%}'
        'article>input:checked~.pair .gt{display:block}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
        '<h1>Raw 引用审计：未自动确认正常，也未加入负样本</h1>'
        '<p>仅凭视觉样例不能证明整个来源无缺陷；应核对生成代码或可追溯来源说明。</p>']
    for index,row in enumerate(samples):
        example = row["examples"][0]
        parts.append('<article><h2>'+html.escape(row["task_key"]+" / "+row["split"])+'</h2>')
        parts.append(f'<label for="gt-{index}">显示异常图 GT</label><input id="gt-{index}" type="checkbox"><div class="pair">')
        views = [("Raw（语义待确认）",row["raw_image"],False),
                 ("ScreenShot（原缺陷标注）",example["image"],True)]
        if example["local_images"]: views.append(("LocalImgURL（来源声明的正常引用）",example["local_images"][0],False))
        for title,image,gt in views:
            parts.append('<figure><figcaption>'+html.escape(title)+'</figcaption><div class="frame"><img loading="lazy" src="'+html.escape(Path(image).as_uri(),quote=True)+'">')
            if gt:
                parts.append(f'<svg class="gt" viewBox="0 0 {example["width"]} {example["height"]}">')
                for x1,y1,x2,y2 in example["boxes_px"]:
                    parts.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="red" stroke-width="3"/>')
                parts.append('</svg>')
            parts.append('</div></figure>')
        parts.append('</div><pre>'+html.escape(str({k:row[k] for k in (
            "source_dataset","source_version","clean_status","observed_raw_cross_split",
            "already_selected","raw_references","paired_positive_ids","source_pages")}))+'</pre></article>')
    Path(path).write_text("\n".join(parts),encoding="utf-8")
