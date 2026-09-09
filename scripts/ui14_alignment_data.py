"""Create independent manifests around a frozen, read-only neg11 dataset."""
from pathlib import Path
import os
import shutil
from ui14_common import read_json, read_jsonl, write_json, file_digest, paths_for, UI_TASKS
from ui14_alignment_common import PROFILE, NEG_OUTPUT, OLD_OUTPUT, independent_output
from ui14_progress import track


def training_answer_contract(recipe):
    from ui14_annotations import answer
    from eaglevl.ui_task_registry import get_task
    task = get_task("ui_alignment")
    path = Path(recipe[task.task_key]["annotation"])
    result = {"task_key": task.task_key, "train": str(path), "sha256": file_digest(path),
              "positive_records": 0, "negative_records": 0, "examples": []}
    for row in read_jsonl(path):
        expected = answer(row["boxes_px"], row["width"], row["height"], task.prompt_label)
        messages = [{"from": "human", "value": "<image>\n" + task.prompt},
                    {"from": "gpt", "value": expected}]
        if row.get("conversations") != messages or row.get("task_id") != task.task_id:
            raise ValueError(f"Frozen alignment training label/prompt disagrees with answer grammar: {row.get('source_record_id')}")
        positive = bool(row["boxes_px"])
        field = "positive_records" if positive else "negative_records"
        result[field] += 1
        if result[field] <= 3:
            result["examples"].append({"positive": positive, "answer": expected,
                                       "source_record_id": row.get("source_record_id")})
    result["negative_template"] = answer([], 1, 1, task.prompt_label)
    return result


def copy_verified(source, target, expected):
    source, target = Path(source), Path(target)
    if file_digest(source) != expected: raise ValueError(f"Frozen parent artifact changed: {source}")
    if target.is_file() and file_digest(target) == expected: return "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    if file_digest(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Source changed while copying: {source}")
    os.replace(temporary, target)
    return "copied"


def prepare_frozen_data(parent, root, output):
    parent = Path(parent).resolve(strict=True)
    root = independent_output(root, parent, NEG_OUTPUT, OLD_OUTPUT, output)
    independent_output(output, root, parent, NEG_OUTPUT, OLD_OUTPUT)
    source_report_path = parent / "cpu_check_report.json"
    source_report = read_json(source_report_path)
    snapshot = read_json(parent / "source_snapshot.json")
    extension = read_json(parent / "negative_extension_manifest.json")
    evaluation = read_json(parent / "evaluation_manifest.json")
    from ui14_negative_quota import quota_policy
    if quota_policy(extension) != "available": raise ValueError("Alignment experiment requires the already frozen available selection")
    if (not source_report.get("ready") or source_report.get("registry_count") != 14
            or source_report.get("evaluation_count") != 14
            or source_report.get("normalization_id") != snapshot["normalization_id"]):
        raise ValueError("Frozen neg11 parent is not fully prepared and checked")
    binding = {"parent": str(parent), "normalization_id": snapshot["normalization_id"],
               "selection_sha256": extension["selection_sha256"], "eval_set_id": evaluation["eval_set_id"],
               "source_report_sha256": file_digest(source_report_path), "negative_quota_policy": "available"}
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "frozen_parent.json"
    if marker.is_file() and read_json(marker) != binding:
        raise ValueError("This alignment directory is already bound to a different frozen selection/test set")
    write_json(marker, binding)
    if any(Path(output).glob("checkpoint-*")):
        if not (root / "cpu_check_report.json").is_file(): raise ValueError("Training output exists without a complete frozen data report")
        from ui14_profile import validate_prepared_profile
        validate_prepared_profile({"UI14_DATA_ROOT": str(root), "INIT_CHECKPOINT": source_report["init_checkpoint"]})
        return read_json(root / "cpu_check_report.json")
    counts = {"reused": 0, "copied": 0}
    normalization = read_json(parent / "normalization_stats.json")
    bound_files = {**source_report["artifact_digests"], **normalization["artifact_digests"]}
    for name, expected in track(bound_files.items(), "冻结 neg11 清单与只读缓存引用", unit="文件"):
        source, target = (parent / name).resolve(), (root / name).resolve()
        if parent not in source.parents or root not in target.parents:
            raise ValueError(f"Artifact escapes its data directory: {name}")
        # Formal config and test index are independently rebuilt below.
        if name in ("formal_job.yaml", "formal_runtime.json", "evaluation_manifest.json"): continue
        counts[copy_verified(source, target, expected)] += 1
    # Journals are mutable: independent copies only, never symlinks/hardlinks.
    journal = parent / "verification/files.jsonl"
    if journal.is_file() and not (root / "verification/files.jsonl").exists():
        copy_verified(journal, root / "verification/files.jsonl", file_digest(journal))
    # A preceding audit may already have created the new journal. Import the
    # digest-bound image evidence into that journal without opening the parent
    # journal for writing (Journal itself is intentionally a writable class).
    from ui14_verification import current_checks
    checks = current_checks()
    if checks:
        for row in read_jsonl(root / "verification/image_evidence.jsonl"):
            key = row["key"]
            value = {k: v for k, v in row.items() if k != "key"}
            if checks.journal.rows.get(key) != value:
                checks.journal.put(key, value)
    for spec in evaluation["tasks"]:
        if spec["task_id"] >= 5 and spec["view_policy"] == "crops":
            spec["detector_input"] = str(paths_for(parent, spec["task_key"], "test")["detector_input"])
    evaluation["frozen_parent_manifest"] = str(parent / "evaluation_manifest.json")
    evaluation["frozen_parent_manifest_sha256"] = file_digest(parent / "evaluation_manifest.json")
    # IDs, test membership, records, and per-task sampler ratios are untouched.
    write_json(root / "evaluation_manifest.json", evaluation)
    from ui14_repair import validate_normalization
    current = validate_normalization(root)
    if current["normalization_id"] != binding["normalization_id"]: raise ValueError("Frozen normalization changed")
    label_contract = training_answer_contract(read_json(root / "training_recipe.json"))
    write_json(root / "alignment_answer_contract.json", label_contract)
    from ui14_checks import render_formal_yaml
    render_formal_yaml(root, profile=PROFILE)
    artifacts = {name: file_digest(root / name) for name in bound_files}
    artifacts["frozen_parent.json"] = file_digest(marker)
    artifacts["alignment_answer_contract.json"] = file_digest(root / "alignment_answer_contract.json")
    external = dict(source_report.get("external_digests", {}))
    external.update({str(parent / name): h for name, h in bound_files.items()})
    external[str(source_report_path)] = binding["source_report_sha256"]
    result = {**source_report, "artifact_digests": artifacts, "external_digests": external,
              "profile": PROFILE, "frozen_parent": binding, "manifest_copy_counts": counts,
              "data_policy": "frozen available; immutable annotations/images/cache referenced read-only",
              "source_report": str(source_report_path), "ready": True}
    result["training_answer_contract"] = label_contract
    write_json(root / "cpu_check_report.json", result)
    from ui14_profile import validate_prepared_profile
    try:
        validate_prepared_profile({"UI14_DATA_ROOT": str(root), "INIT_CHECKPOINT": source_report["init_checkpoint"]})
    except BaseException:
        result["ready"] = False
        write_json(root / "cpu_check_report.json", result)
        raise
    print(f"[alignment prepare] {counts}; frozen normalization={binding['normalization_id']}; eval_set={binding['eval_set_id']}", flush=True)
    return result
