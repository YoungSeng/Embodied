"""Read-only parent snapshot plus independently bound task-specific clean images.

No detector or model is imported here. Missing evidence or quota fails closed.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import html
import os
import shutil
import sqlite3
from urllib.parse import unquote, urlparse

from ui14_common import *
from ui14_verification import current_checks, signature
from ui14_progress import phase, track
from ui9_source_parser import Resolver, image_slots, page_key

PARENT_COMMIT = "e06add6b0b3c05b3f66384cf93331a0c94c076e8"
NEG_DATA = WORKSPACE + "/gui_data/ui14_cpt9000_neg11_v1"
NEG_PROJECT = WORKSPACE + "/code/Eagle_LocateUI5_v4/Embodied-ui14-neg11"
NEG_OUTPUT = WORKSPACE + "/gui_models/locany-m32-cpt9000-ui14-neg11-a800x4-v1"
OLD_OUTPUT = WORKSPACE + "/gui_models/locany-m32-cpt9000-ui14-a800x4-repair-v2"
SYNTH = tuple(t for t in UI_TASKS if t.task_id >= 7)
SEED = 42
REFERENCE_TEST_COUNTS = dict(zip((t.task_key for t in SYNTH),(1664,5907,692,767,460,37,73)))


def independent_copy(source, target):
    """Atomic, private copy; never link a writable manifest to its parent."""
    source, target = Path(source), Path(target)
    if target.exists(): return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f".import-{os.getpid()}")
    shutil.copy2(source, temp)
    os.replace(temp, target)
    return True


def assert_isolated(root, parent, source, old_output=OLD_OUTPUT, output=NEG_OUTPUT):
    root, parent, source, old_output, output = map(lambda p: Path(p).resolve(), (root, parent, source, old_output, output))
    for write in (root, output):
        for readonly in (parent, source, old_output, Path(CLUSTER_PROJECT).resolve()):
            if write == readonly or readonly in write.parents or write in readonly.parents:
                raise ValueError(f"New writable directory overlaps read-only input: {write} / {readonly}")
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Data and training output directories must be separate")
    # Symlinked directories would allow child writes to escape the new root.
    if root.exists() and any(p.is_symlink() for p in root.rglob("*") if p.is_dir()):
        raise ValueError("New data directory must not contain directory symlinks")


def seed_evidence(root, parent):
    for relative in ("verification/files.jsonl", "cache_preparation/image_info.jsonl"):
        source = Path(parent)/relative
        if source.is_file(): independent_copy(source, Path(root)/relative)


def parent_binding(parent):
    """Validate frozen parent JSON evidence without opening its writable journals."""
    parent = Path(parent).resolve(strict=True)
    snap = read_json(parent/"source_snapshot.json")
    stats = read_json(parent/"normalization_stats.json")
    report = read_json(parent/"cpu_check_report.json")
    if (not stats.get("complete") or not report.get("ready")
            or stats["normalization_id"] != snap["normalization_id"]
            or report["normalization_id"] != snap["normalization_id"]
            or digest({k:v for k,v in snap.items() if k != "normalization_id"}) != snap["normalization_id"]):
        raise ValueError("Parent normalization/finalize is incomplete or unbound")
    bound = {str(parent/name): value for name,value in report["artifact_digests"].items()}
    bound.update(snap["source_files"])
    for name in ("source_snapshot.json", "normalization_stats.json", "cpu_check_report.json"):
        bound[str(parent/name)] = file_digest(parent/name)
    for path, expected in track(bound.items(), "只读父版本摘要", unit="文件"):
        if file_digest(path) != expected: raise ValueError(f"Parent input changed: {path}")
    return {"parent_root": str(parent), "parent_commit": PARENT_COMMIT,
            "parent_normalization_id": snap["normalization_id"],
            "repair_run_id": snap["repair_run_id"], "files": bound}


def save_if_changed(path, rows, jsonl=False):
    if Path(path).is_file():
        previous = list(read_jsonl(path)) if jsonl else read_json(path)
        if previous == rows: return
    (write_jsonl if jsonl else write_json)(path, rows)


class ReferenceResolver:
    """Use exact copied paths/ledger mappings, then unique folder suffix matches.

    A filename-only match is accepted only when unique within that source;
    this locates an already evidenced reference, never labels a folder of images.
    """
    def __init__(self, source):
        self.source = Path(source); self.resolver = Resolver([])
        self.by_task, self.ledger = {}, {}
        work = self.source/".work"
        for db in sorted(work.glob("*.sqlite*")) + sorted(work.glob("*.db")):
            try:
                with sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True) as conn:
                    columns = {r[1] for r in conn.execute("PRAGMA table_info(files)")}
                    if {"src", "dst"} <= columns:
                        for src,dst in conn.execute("SELECT src,dst FROM files"):
                            self.ledger.setdefault(str(src), set()).add(str(dst))
            except sqlite3.DatabaseError:
                continue
        for path in (work/"folder_copy_manifest.json", work/"copy_manifest.json", work/"copy_report.json"):
            if path.is_file():
                def visit(v):
                    if isinstance(v, dict):
                        a,b = v.get("src", v.get("source")), v.get("dst", v.get("destination"))
                        if isinstance(a,str) and isinstance(b,str): self.ledger.setdefault(a,set()).add(b)
                        for child in v.values(): visit(child)
                    elif isinstance(v,list):
                        for child in v: visit(child)
                visit(read_json(path))

    def index(self, task):
        if task not in self.by_task:
            files = sorted(p.resolve() for p in (self.source/task/"sample_imgs").rglob("*")
                           if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} and p.is_file())
            by_name = defaultdict(list)
            for p in files: by_name[p.name].append(p)
            self.by_task[task] = (files, by_name)
        return self.by_task[task]

    def resolve(self, value, task):
        base = self.source/task
        path, found = self.resolver.resolve(value, str(base))
        if found: return path.resolve()
        mapped = [Path(p).resolve() for p in self.ledger.get(str(value), ()) if Path(p).is_file()]
        if len(set(mapped)) == 1: return mapped[0]
        _, index = self.index(task)
        raw = unquote(urlparse(str(value)).path).replace("\\", "/")
        candidates = index.get(Path(raw).name, []) + index.get(Path(raw).name.replace(":", "_"), [])
        candidates = list(dict.fromkeys(candidates))
        for depth in range(min(5, len(Path(raw).parts)), 1, -1):
            suffix = "/".join(Path(raw).parts[-depth:])
            exact = [p for p in candidates if p.as_posix().endswith(suffix)]
            if len(exact) == 1: return exact[0]
        if len(candidates) == 1: return candidates[0]
        raise ValueError(f"Reference missing or ambiguous ({len(candidates)} matches): {task}: {value}")


def source_rows(parent):
    for task in UI9_TASKS:
        for split in ("train", "test"):
            yield task, split, list(read_jsonl(paths_for(parent, task.task_key, split)["normalized"]))


def group_keys(row):
    keys = ["image:" + row["source_image_id"]]
    if row.get("source_page_id"): keys.append("page:" + row["source_page_id"])
    keys.extend("reference:"+p for p in row.get("raw_parent_paths",[]))
    return keys


def eligible_evidence(raw, role, task):
    if role == "normal":
        return {"kind": "paired_normal_reference", "field": "LocalImgURL",
                "claim": "source export identifies the corresponding normal image"}
    explicit = raw.get("negative_evidence", {})
    if (isinstance(explicit,dict) and task in explicit.get("clean_tasks", [])
            and explicit.get("basis") and explicit.get("provenance")):
        return {"kind": "explicit_task_clean", **explicit}
    if role == "raw" and raw.get("raw_is_pre_synthesis") is True:
        return {"kind": "paired_pre_synthesis", "field": "RawImgURL",
                "claim": "source explicitly declares raw_is_pre_synthesis=true"}
    return None


def inventory(args):
    root, parent, source = Path(args.data_root), Path(args.parent_root), Path(args.source_root)
    binding = parent_binding(parent)
    if (root/"negative_extension_manifest.json").is_file() and (root/"normalization_stats.json").is_file():
        validate_extension(root)
        previous=read_json(root/"inventory_summary.json")
        print(f"[inventory] reused frozen selection={previous['selected_count']} gap={previous['gap']}",flush=True)
        return previous
    checks = current_checks(); resolver = ReferenceResolver(source)
    frozen = defaultdict(set); candidates = []; rejected = []; targets = {}; old_rows = []
    old_ui5_rows = []
    for task, split, rows in source_rows(parent):
        old_rows.extend(rows)
        if task in SYNTH:
            positives = {r["source_image_id"] for r in rows if r["boxes_px"]}
            negatives = {r["source_image_id"] for r in rows if not r["boxes_px"]} - positives
            targets[f"{task.task_key}/{split}"] = {"positive_images": len(positives), "existing_negative_images": len(negatives),
                "required_new": max(0,len(positives)-len(negatives)), "existing_image_ids": sorted(positives|negatives)}
            if split == "test":
                targets[f"{task.task_key}/{split}"].update(reference_positive_images=REFERENCE_TEST_COUNTS[task.task_key],
                    matches_reference=len(positives)==REFERENCE_TEST_COUNTS[task.task_key])
        for row in rows:
            for key in group_keys(row): frozen[key].add(split)
    # UI5 train carries original-image identity; its frozen no_figma test also
    # participates in leakage checks, even when it has no Figma metadata.
    registry = read_json(parent/"task_registry.json")["tasks"]
    from qwen3vl_merge_and_score_fixed_5tasks import extract_image_path, is_figma_sample
    for spec in registry[:5]:
        for split in ("train", "test"):
            for row in read_jsonl(spec[split]):
                if split == "test" and is_figma_sample(row): continue
                image = row.get("source_image") or extract_image_path(row)
                if not image: raise ValueError("Parent UI5 record has no source image")
                image = str((Path(spec[split]).parent/Path(image)).resolve())
                identity = row.get("source_image_id")
                if not identity: identity = image_identity(image)[0]
                frozen["image:"+identity].add(split)
                pg = row.get("source_page_id") or page_key(row)
                if pg: frozen["page:"+pg].add(split)
                old_ui5_rows.append({"source_image_id":identity,"source_page_id":pg,
                    "task_key":spec["task_key"],"split":split})
    for task in track(SYNTH, "盘点合成来源的正常/原图引用", unit="任务"):
        resolver.index(task.task_key)
        for split in ("train", "test"):
            normalized = list(read_jsonl(paths_for(parent,task.task_key,split)["normalized"]))
            source_path = source/task.task_key/f"{split}.jsonl"
            raw_lines = list(read_jsonl(source_path))
            by_id = {str(r.get("source_record_id",r.get("id",r.get("ID",i)))): r for i,r in enumerate(raw_lines)}
            for positive in track(normalized, f"{task.task_key}/{split} 引用与证据", unit="记录"):
                if not positive["boxes_px"]: continue
                raw = positive.get("source_metadata") or by_id.get(positive["source_record_id"])
                if not isinstance(raw,dict): raise ValueError("Cannot recover paired source metadata")
                raw_parents=[]
                for _,_,reference,role in image_slots(raw):
                    if role!="raw": continue
                    try: raw_parents.append(str(resolver.resolve(reference,task.task_key)))
                    except (OSError,ValueError): pass
                for _,_,value,role in image_slots(raw):
                    if role not in ("normal", "raw"): continue
                    evidence = eligible_evidence(raw, role, task.task_key)
                    try: path = resolver.resolve(value, task.task_key)
                    except (OSError, ValueError) as exc:
                        rejected.append({"task_key":task.task_key,"split":split,"reference":value,"reason":str(exc)}); continue
                    # All references inherit their original positive's split,
                    # including unevidenced raw references; they cannot leak later.
                    frozen["reference:"+str(path)].add(split)
                    if not evidence:
                        rejected.append({"task_key":task.task_key,"split":split,"reference":str(path),
                                         "reason":"raw reference lacks explicit pre-synthesis/task-clean evidence"}); continue
                    candidates.append({"task_key": task.task_key, "source_dataset": positive["source_dataset"],
                        "source_version":positive["source_version"], "requested_split":split, "source_image":str(path),
                        "source_page_id":positive.get("source_page_id"), "negative_evidence":evidence,
                        "paired_positive_ids":[positive["source_record_id"]], "paired_positive_images":[positive["source_image"]],
                        "paired_positive_image_ids":[positive["source_image_id"]], "source_metadata":raw,
                        "paired_positive_boxes_px":positive["boxes_px"],"paired_width":positive["width"],"paired_height":positive["height"],
                        "source_split":raw.get("split"), "source_jsonl":str(source_path),
                        "raw_parent_paths":sorted(set(raw_parents)),
                        "selection_priority":0 if role=="normal" else 1, "original_reference":value})
    # A pool needs task-specific clean evidence; neither another task's negative
    # label nor a raw folder name establishes this. Support explicit local pools.
    pools = sorted({p for task in SYNTH for pattern in ("normal_pool*.jsonl","negative_pool*.jsonl")
                    for p in (source/task.task_key).rglob(pattern)
                    if not any(part.lower() in ("backup","backups","quarantine",".work") for part in p.relative_to(source).parts)})
    pool_sources={p:p.relative_to(source).parts[0] for p in pools}
    for task in SYNTH:
        for p in sorted((root/"normal_pools"/task.task_key).glob("*.jsonl")):
            pools.append(p);pool_sources[p]=task.task_key
    for pool in pools:
        binding["files"][str(pool)] = file_digest(pool)
        for raw in read_jsonl(pool):
            evidence = raw.get("negative_evidence", {})
            for task in SYNTH:
                if not eligible_evidence(raw,"pool",task.task_key): continue
                path = resolver.resolve(raw["image"], pool_sources[pool])
                candidates.append({"task_key":task.task_key,"source_dataset":raw["source_dataset"],
                    "source_version":raw.get("source_version","pool"),"requested_split":None,"source_image":str(path),
                    "source_page_id":page_key(raw) or raw.get("source_page_id"),"negative_evidence":evidence,
                    "paired_positive_ids":[],"paired_positive_images":[],"paired_positive_image_ids":[],
                    "source_metadata":raw,"source_jsonl":str(pool),"source_split":raw.get("split"),
                    "selection_priority":2 if raw["source_dataset"]==task.source_dataset else 3})
    # Reference aliases of EVERY old source also constrain new content, even
    # when those references are not eligible negatives themselves.
    old_references=[]
    for row in old_rows:
        raw=row.get("source_metadata",{})
        for _,_,value,role in image_slots(raw):
            if role not in ("normal","raw"): continue
            try: path=str(resolver.resolve(value,row["task_key"]))
            except (OSError,ValueError): continue
            old_references.append((path,row["split"]))
    checks.prefetch_images([c["source_image"] for c in candidates]+[p for p,_ in old_references])
    for path,split in old_references:
        frozen["image:"+image_identity(path)[0]].add(split)
        frozen["reference:"+path].add(split)
    # Resolve content aliases BEFORE assigning any new page. All seven tasks
    # share one union-find: content, page and raw-parent associations are transitive.
    links = {}
    def find(key):
        links.setdefault(key,key)
        while key != links[key]: links[key] = links[links[key]]; key=links[key]
        return key
    def union(keys):
        roots = sorted({find(k) for k in keys})
        for key in roots[1:]: links[key]=roots[0]
    for row in old_rows+old_ui5_rows: union(group_keys(row))
    for key in frozen: find(key)
    for c in candidates:
        identity,w,h = image_identity(c["source_image"])
        c.update(source_image_id=identity,width=w,height=h,selection_seed=SEED)
        c["group_keys"] = group_keys(c)+["reference:"+c["source_image"]]
        union(c["group_keys"])
    memberships = defaultdict(set)
    for key,splits in frozen.items(): memberships[find(key)].update(splits)
    frozen_map_path = root/"negative_page_assignments.json"
    previous = read_json(frozen_map_path).get("assignments",{}) if frozen_map_path.exists() else {}
    for key,split in previous.items():
        if key in links: memberships[find(key)].add(split)
    assignment = dict(previous)
    unique = {}
    for c in candidates:
        group = find(c["group_keys"][0]); sides = memberships[group]
        if len(sides)>1:
            rejected.append({"task_key":c["task_key"],"source_image":c["source_image"],"reason":"frozen cross-split page/content conflict"}); continue
        split = next(iter(sides)) if sides else assignment.get(group)
        if split is None: split = "test" if int(digest([SEED,group])[:12],16)%10==0 else "train"
        if group in previous and previous[group]!=split: raise ValueError("Frozen negative page assignment changed")
        assignment[group]=split
        for key in c["group_keys"]: assignment[key]=split
        if c["requested_split"] and split!=c["requested_split"]:
            rejected.append({"task_key":c["task_key"],"source_image":c["source_image"],"reason":"paired page belongs to opposite split"}); continue
        c.update(split=split,page_group=group)
        target=targets[f"{c['task_key']}/{split}"]
        if c["source_image_id"] in target["existing_image_ids"]:
            rejected.append({"task_key":c["task_key"],"source_image":c["source_image"],"reason":"already present in parent task/split"}); continue
        key=(c["task_key"],split,c["source_image_id"])
        if key in unique:
            old=unique[key]
            for name in ("paired_positive_ids","paired_positive_images","paired_positive_image_ids"):
                old[name]=sorted(set(old[name]+c[name]))
            if c["selection_priority"]<old["selection_priority"]:
                old["negative_evidence"]=c["negative_evidence"]; old["selection_priority"]=c["selection_priority"]
        else: unique[key]=c
    selected=[]
    for key,target in targets.items():
        task,split=key.split("/")
        available=sorted((c for (t,s,_),c in unique.items() if (t,s)==(task,split)),
                         key=lambda c:(c["selection_priority"],digest([SEED,task,split,c["source_image_id"]])))
        chosen=available[:target["required_new"]]; selected.extend(chosen)
        target.update(candidate_images=len(available),selected_images=len(chosen),gap=target["required_new"]-len(chosen),
                      selected_sources=dict(Counter(c["source_dataset"] for c in chosen)))
        target["selected_source_fraction"]={s:n/len(chosen) for s,n in target["selected_sources"].items()} if chosen else {}
        target.pop("existing_image_ids")
    save_if_changed(frozen_map_path,{"seed":SEED,"assignments":assignment})
    save_if_changed(root/"negative_candidates.jsonl",list(unique.values()),True)
    save_if_changed(root/"negative_rejections.jsonl",rejected,True)
    save_if_changed(root/"negative_selection.proposed.jsonl",selected,True)
    selection_digest=file_digest(root/"negative_selection.proposed.jsonl")
    payload={"schema_version":1,**binding,"seed":SEED,"targets":targets,
        "proposed_selection_sha256":selection_digest,"page_assignments_sha256":file_digest(frozen_map_path),
        "directory_images":{t.task_key:len(resolver.index(t.task_key)[0]) for t in SYNTH},
        "rejected_count":len(rejected),"candidate_count":len(unique),"selected_count":len(selected),
        "gap":sum(t["gap"] for t in targets.values()),
        "existing_conflicting_groups":sum(len(s)>1 for s in memberships.values()),
        "new_cross_split_groups":0,"selection_basis":"source evidence, frozen page/content, seed=42; no model outputs"}
    payload["inventory_id"]=digest(payload)
    save_if_changed(root/"inventory_summary.json",payload)
    from prepare_ui14_sft import image_overlaps
    old_uses=old_rows+old_ui5_rows
    combined=image_overlaps(old_uses+selected)
    added_ids={c["source_image_id"] for c in selected}
    introduced=[r for r in combined["train_test"] if r["source_image_id"] in added_ids]
    if introduced: raise ValueError("Selected normal images introduced train/test overlap")
    overlap={"existing":image_overlaps(old_uses),"combined":combined,
             "new_cross_split_images":len(introduced),"new_cross_split_page_groups":0,
             "old_records_with_page":sum(bool(r.get("source_page_id")) for r in old_uses),
             "old_records_without_page":sum(not r.get("source_page_id") for r in old_uses),
             "page_policy":"Known Figma/raw/content groups frozen; unavailable page IDs are reported, not inferred."}
    save_if_changed(root/"negative_split_overlap.json",overlap)
    for key,t in targets.items():
        print(f"[inventory] {key}: positive={t['positive_images']} candidates={t['candidate_images']} "
              f"selected={t['selected_images']} gap={t['gap']}",flush=True)
    gallery(root,selected)
    return payload


def gallery(root,selected):
    parts=['<!doctype html><meta charset="utf-8"><title>UI14 normal pairs</title>',
           '<style>body{font:16px sans-serif}article{margin:24px;border:1px solid #ccc;padding:12px}.frame{display:inline-block;position:relative;width:42%;vertical-align:top}.frame img{width:100%}.overlay{position:absolute;inset:0;width:100%;height:100%;display:none}input:checked~.pair .overlay{display:block}pre{white-space:pre-wrap}</style>',
           '<h1>正常/异常配对（GT 默认隐藏）</h1>']
    for task in SYNTH:
        for c in [r for r in selected if r["task_key"]==task.task_key][:10]:
            raw=c["source_metadata"]
            parts.append('<article><h2>'+html.escape(task.task_key+' / '+c['split'])+'</h2>')
            parts.append('<label>显示 GT <input type="checkbox" onclick="this.parentNode.nextElementSibling.checked=this.checked"></label><input type="checkbox" hidden><div class="pair">')
            for i,image in enumerate([c["source_image"],*c["paired_positive_images"][:1]]):
                parts.append('<div class="frame"><img loading="lazy" src="'+html.escape(Path(image).as_uri(),quote=True)+'">')
                if i:
                    parts.append(f'<svg class="overlay" viewBox="0 0 {int(c["paired_width"])} {int(c["paired_height"])}">')
                    for x1,y1,x2,y2 in c.get("paired_positive_boxes_px",[]):
                        parts.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="red" stroke-width="3"/>')
                    parts.append('</svg>')
                parts.append('</div>')
            parts.append('</div>')
            parts.append('<pre>'+html.escape(json.dumps({k:c[k] for k in ("source_dataset","negative_evidence","paired_positive_ids","source_page_id")},ensure_ascii=False,indent=2))+'</pre>')
            parts.append('<details><summary>显示原始 GT（正常图无框）</summary><pre>'+html.escape(json.dumps(raw.get("Objects"),ensure_ascii=False,indent=2))+'</pre></details></article>')
    (Path(root)/"normal_pairs.html").write_text('\n'.join(parts),encoding="utf-8")


def normalize(args):
    root,parent=Path(args.data_root),Path(args.parent_root)
    inv=read_json(root/"inventory_summary.json")
    if digest({k:v for k,v in inv.items() if k!="inventory_id"})!=inv["inventory_id"]:
        raise ValueError("Inventory identity changed")
    if file_digest(root/"negative_page_assignments.json")!=inv["page_assignments_sha256"]:
        raise ValueError("Inventory page assignments changed")
    parent_now=parent_binding(parent)
    if parent_now["parent_normalization_id"]!=inv["parent_normalization_id"]: raise ValueError("Parent version changed")
    if inv["gap"]: raise ValueError(f"Independent negative quota not met: gap={inv['gap']}; inspect inventory and evidence pools")
    for path,h in inv["files"].items():
        if file_digest(path)!=h: raise ValueError(f"Inventory input changed: {path}")
    proposed=root/"negative_selection.proposed.jsonl"
    if file_digest(proposed)!=inv["proposed_selection_sha256"]: raise ValueError("Inventory selection changed")
    selected=list(read_jsonl(proposed)); checks=current_checks()
    checks.prefetch_images(r["source_image"] for r in selected)
    for row in selected:
        if image_identity(row["source_image"])!=(row["source_image_id"],row["width"],row["height"]):
            raise ValueError("Selected normal image changed; rerun inventory before selection is frozen")
    marker=root/"negative_extension_manifest.json"
    body={"schema_version":1,"kind":"ui14_negative_extension",**parent_now,"seed":SEED,
          "inventory_id":inv["inventory_id"],"selection_sha256":inv["proposed_selection_sha256"],
          "page_assignments_sha256":inv["page_assignments_sha256"],"targets":inv["targets"],
          "selection_file":str(root/"negative_selection.jsonl")}
    # Pool exports are extension inputs, not fictitious entries in the old
    # repair's after counts. Preserve their original inventory digests, too.
    body["pool_files"]={p:h for p,h in inv["files"].items() if p not in parent_now["files"]}
    body["extension_id"]=digest(body)
    if marker.exists() and read_json(marker).get("extension_id")!=body["extension_id"]:
        raise ValueError("Extension selection already frozen; use a new independent data root")
    if marker.exists() and (root/"normalization_stats.json").is_file():
        try:
            validate_extension(root)
        except (ValueError,OSError,KeyError):
            pass  # Rebuild missing/incomplete outputs, never mutate the selection.
        else:
            count=sum(s["normalized_records"] for s in read_json(root/"normalization_stats.json")["tasks"].values())
            print(f"[normalize] reused={count} rebuilt=0; completed composite artifacts unchanged",flush=True)
            return {"reused_records":count,"rebuilt_records":0,"normalization_complete":True}
    save_if_changed(root/"negative_selection.jsonl",selected,True)
    save_if_changed(marker,body)
    parent_snap=read_json(parent/"source_snapshot.json")
    snapshot={**parent_snap,"kind":"ui14_negative_extension","parent_normalization_id":parent_snap["normalization_id"],
              "parent_snapshot_sha256":file_digest(parent/"source_snapshot.json"),"extension_id":body["extension_id"]}
    snapshot.pop("normalization_id"); snapshot["normalization_id"]=digest(snapshot)
    save_if_changed(root/"source_snapshot.json",snapshot)
    registry_doc=read_json(parent/"task_registry.json")
    registry_doc.update(normalization_id=snapshot["normalization_id"],parent_normalization_id=body["parent_normalization_id"],extension_id=body["extension_id"])
    artifacts={}; stats={}; all_rows=[]
    for task,split,old in source_rows(parent):
        rows=[]
        for row in old:
            rows.append({**row,"parent_normalization_id":row["normalization_id"],"normalization_id":snapshot["normalization_id"],
                         "is_positive":bool(row["boxes_px"]),"parent_record":True})
            if task.task_id>=7 and not row["boxes_px"]:
                rows[-1].update(clean_source_image=True,negative_kind="clean_source_image")
        for c in selected:
            if (c["task_key"],c["split"])!=(task.task_key,split): continue
            rows.append({**c,"source_record_id":"neg11:"+c["source_image_id"],"task_id":task.task_id,
                "image":c["source_image"],"boxes_px":[],"is_positive":False,"clean_source_image":True,
                "negative_kind":"clean_source_image","crop_id":"full","view_policy":task.view_policy,
                "normalization_id":snapshot["normalization_id"],"parent_normalization_id":body["parent_normalization_id"],
                "repair_run_id":body["repair_run_id"],"extension_id":body["extension_id"],"selected_gt":[],
                "scale_xy":None,"coordinate_basis":"explicit task-clean normal image; no bbox projection"})
        p=paths_for(root,task.task_key,split)
        save_if_changed(p["normalized"],rows,True)
        save_if_changed(p["detector_input"],[{"image":r["source_image"]} for r in rows],True)
        save_if_changed(p["detector_inputs"],{task.task_key:str(p["detector_input"])})
        for key in ("normalized","detector_input","detector_inputs"): artifacts[str(p[key].relative_to(root))]=file_digest(p[key])
        stats[f"{task.task_key}/{split}"]={"records":len(rows),"normalized_records":len(rows),"failed_records":0,
            "reused_records":len(old),"new_records":len(rows)-len(old)}
        all_rows.extend(rows)
        print(f"[normalize] {task.task_key}/{split}: reused={len(old)} added={len(rows)-len(old)} failed=0",flush=True)
    save_if_changed(root/"task_registry.json",registry_doc)
    receipts={p:{"sha256":h,"stat":signature(p)} for p,h in {**body["files"],**body["pool_files"],
        **{str(root/n):h for n,h in artifacts.items()}, str(root/"negative_selection.jsonl"):body["selection_sha256"],
        str(root/"negative_page_assignments.json"):body["page_assignments_sha256"]}.items()}
    save_if_changed(root/"normalization_stats.json",{"complete":True,"normalization_id":snapshot["normalization_id"],
        "parent_stats":str(parent/"normalization_stats.json"),"parent_stats_sha256":file_digest(parent/"normalization_stats.json"),
        "tasks":stats,"artifact_digests":artifacts,"file_receipts":receipts})
    from ui14_repair import page_statistics
    from prepare_ui14_sft import image_overlaps
    save_if_changed(root/"ui9_page_split.json",page_statistics(all_rows))
    save_if_changed(root/"ui9_image_overlap.json",image_overlaps(all_rows))
    report={"ready":False,"normalization_complete":True,"normalization_id":snapshot["normalization_id"],
        "repair_run_id":body["repair_run_id"],"extension_id":body["extension_id"],"parent_version":body["parent_normalization_id"],
        "normalization_resume":stats,"cpu_only":True,"gpu_loaded":False}
    # The unchanged, complete path returned above preserves finalize. Rebuilt
    # artifacts require a new final check, even when their labels are identical.
    write_json(root/"cpu_check_report.json",report)
    return report


def validate_extension(root):
    root=Path(root); ext=read_json(root/"negative_extension_manifest.json"); snap=read_json(root/"source_snapshot.json")
    stats=read_json(root/"normalization_stats.json")
    def matches(path,expected):
        receipt=stats.get("file_receipts",{}).get(str(path))
        if receipt and receipt["sha256"]==expected and receipt["stat"]==signature(path): return True
        return file_digest(path)==expected
    if digest({k:v for k,v in ext.items() if k!="extension_id"})!=ext["extension_id"]: raise ValueError("Extension identity changed")
    if digest({k:v for k,v in snap.items() if k!="normalization_id"})!=snap["normalization_id"]: raise ValueError("Composite snapshot changed")
    if snap["extension_id"]!=ext["extension_id"]: raise ValueError("Wrong extension snapshot")
    if snap["parent_normalization_id"]!=ext["parent_normalization_id"]:
        raise ValueError("Wrong parent normalization")
    required={str(paths_for(root,t.task_key,s)[k].relative_to(root))
              for t in UI9_TASKS for s in ("train","test")
              for k in ("normalized","detector_input","detector_inputs")}
    if set(stats.get("artifact_digests",{}))!=required:
        raise ValueError("Composite normalization requires all 18 split outputs and detector inputs")
    if set(stats.get("tasks",{}))!={f"{t.task_key}/{s}" for t in UI9_TASKS for s in ("train","test")}:
        raise ValueError("Composite normalization task/split statistics incomplete")
    if any(s.get("failed_records")!=0 or s.get("normalized_records")!=s.get("records")
           for s in stats["tasks"].values()):
        raise ValueError("Composite normalization contains failed/incomplete splits")
    if not matches(ext["selection_file"],ext["selection_sha256"]): raise ValueError("Frozen negative selection changed")
    if not matches(root/"negative_page_assignments.json",ext["page_assignments_sha256"]): raise ValueError("Frozen page map changed")
    for path,h in {**ext["files"],**ext.get("pool_files",{})}.items():
        if not matches(path,h): raise ValueError(f"Read-only parent/pool changed: {path}")
    if not stats.get("complete") or stats["normalization_id"]!=snap["normalization_id"]: raise ValueError("Composite normalization incomplete")
    for name,h in stats["artifact_digests"].items():
        if not matches(root/name,h): raise ValueError(f"Extension normalized artifact changed: {name}")
    return snap
