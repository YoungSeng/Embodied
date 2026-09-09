"""Read-only, original-scorer attribution for historical UI9 task predictions."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import html
import math
import base64
import mimetypes
from pathlib import Path
from ui14_common import read_json, read_jsonl, write_json, write_jsonl, file_digest, digest
from ui14_progress import track
from eaglevl.ui_task_registry import get_task
from eaglevl.ui_answer_grammar import raw_format_issue

HISTORICAL_REFERENCE = {
    2000: {"image_f1": .1897, "bbox_f1": .1391, "tp": 24, "fp": 99, "fn": 106, "parse_error": 21},
    4000: {"image_f1": .1561, "bbox_f1": .0957, "tp": 64, "fp": 626, "fn": 66, "parse_error": 284},
}


def image_key(value):
    return str(Path(str(value)).resolve())


def indexed_files(folder):
    result = {}
    for path in sorted(Path(folder).glob("*.json")):
        row = read_json(path)
        key = row.get("image_path") or row.get("source_image")
        if not key: continue
        key = image_key(key)
        if key in result: raise ValueError(f"Duplicate task/image artifacts: {path}")
        result[key] = (path, row)
    return result


def classify_record(answer, gate_status, runtime_failure, scored, positive):
    syntax = raw_format_issue(answer)
    if runtime_failure: return "runtime_failure", syntax
    if syntax is not None: return syntax, syntax
    if scored["invalid_pred"]: return "bbox_parse_failure", None
    if not positive and scored["img_fp"]: return "legal_negative_false_positive", None
    if positive and scored["img_fn"]: return "positive_miss", None
    if positive and (scored["fn"] or scored["fp"]): return "legal_bbox_unmatched", None
    return "correct", None


def render_samples(records, path, *, limit=5):
    """Stable samples per category; original images only, GT hidden in details."""
    groups = defaultdict(list)
    for row in records: groups[row["error_type"]].append(row)
    cards = []
    for kind, rows in sorted(groups.items()):
        for row in sorted(rows, key=lambda r: digest(r["source_image_id"]))[:limit]:
            width, height = row["width"], row["height"]
            source = Path(row["source_image"])
            mime = mimetypes.guess_type(source.name)[0] or "image/png"
            src = f"data:{mime};base64," + base64.b64encode(source.read_bytes()).decode("ascii")
            def svg(boxes, color):
                rectangles = "".join(f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" fill="none" stroke="{color}" stroke-width="3"/>' for b in boxes)
                return f'<svg viewBox="0 0 {width} {height}"><image href="{html.escape(src, quote=True)}" width="{width}" height="{height}"/>{rectangles}</svg>'
            cards.append(f'<article><h3>{html.escape(kind)} · {html.escape(row["source_image_id"])}</h3>'
                + svg(row["prediction_boxes_px"], "#ef4444")
                + '<details><summary>点击显示 GT</summary>' + svg(row["gt_boxes_px"], "#22c55e") + '</details>'
                + '<pre>' + html.escape(row["raw_answer"] if row["raw_answer"] is not None else "[原始答案缺失]") + '</pre>'
                + '<p>' + html.escape(row["source_image"]) + '</p></article>')
    page = '<!doctype html><meta charset="utf-8"><title>UI alignment audit</title><style>body{font:16px sans-serif;margin:24px;background:#eef2f6}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}article{background:white;padding:18px;overflow:auto}svg{width:100%;max-height:640px}pre{white-space:pre-wrap}p{overflow-wrap:anywhere}</style><h1>原预测审计 · 红框为预测，GT 点击显示</h1><main>' + "".join(cards) + '</main>'
    Path(path).write_text(page, encoding="utf-8")


def audit_step(old_run, manifest, task_key, step, output, *, examples=5):
    from qwen3vl_merge_and_score_fixed_5tasks import evaluate_samples, evaluate_merged_file, build_metrics_summary
    task = get_task(task_key)
    if task.task_id < 5: raise ValueError("This per-image UI9 audit uses explicit task IDs; UI5 retains audit-errors legacy scoring")
    old_run, manifest, output = Path(old_run).resolve(), Path(manifest).resolve(), Path(output).resolve()
    if output == old_run or old_run in output.parents or output in old_run.parents:
        raise ValueError("Audit output must be independent of the old run")
    state_path = old_run / "evaluation" / f"ui14-step-{step}.json"
    state = read_json(state_path)
    if state.get("status") != "success": raise ValueError(f"Old evaluation is not complete: {state_path}")
    if state.get("identity", {}).get("manifest_digest") != file_digest(manifest):
        raise ValueError("The supplied old test manifest differs from this evaluation's frozen manifest")
    doc = read_json(manifest)
    spec = next(s for s in doc["tasks"] if s["task_key"] == task_key)
    gt_path = Path(spec["test"])
    source = list(read_jsonl(gt_path))
    prediction = old_run / f"inference-checkpoint-{step}-ui14" / task_key
    score_dir = Path(state["evaluation_run_dir"])
    merged_path, metrics_path = score_dir / f"{task_key}.merged.jsonl", score_dir / "ui14_metrics.json"
    # The completed scorer's merged file is authoritative for scoring. Gate/raw
    # evidence explains it; missing raw text is never guessed or repaired.
    merged_rows = list(read_jsonl(merged_path))
    merged = {str(r["image_id"]): r for r in merged_rows}
    if len(merged) != len(merged_rows): raise ValueError("Duplicate image IDs in old merged result")
    source_by_id = {str(r["source_image_id"]): r for r in source}
    if len(source_by_id) != len(source) or set(source_by_id) != set(merged):
        raise ValueError("Old GT and merged result do not contain the same unique image set")
    gates, errors = indexed_files(prediction / "gate"), indexed_files(prediction / "errors")
    raw_by_image = indexed_files(prediction / "raw")
    raw_files = sorted((prediction / "raw").glob("*.json"))
    files = [state_path, manifest, gt_path, merged_path, metrics_path,
             *[v[0] for v in gates.values()], *[v[0] for v in errors.values()], *raw_files]
    provenance = {str(p): file_digest(p) for p in track(files, f"{task_key}/step-{step} 审计输入摘要", unit="文件")}
    from ui14_verification import signature
    image_files = {image_key(r["source_image"]): signature(r["source_image"])
                   for r in track(source, f"{task_key}/step-{step} 原图属性（不解码）", unit="图片")}
    contract = {"schema_version": 1, "step": step, "task_key": task_key, "sources": provenance,
                "scorer_sha256": file_digest(Path(__file__).resolve().parents[1] / "qwen3vl_merge_and_score_fixed_5tasks.py"),
                "audit_sha256": file_digest(__file__), "examples": examples,
                "format_classifier_sha256": file_digest(Path(__file__).resolve().parents[1] / "eaglevl/ui_answer_grammar.py"),
                "image_files": image_files,
                "eval_set_id": doc.get("eval_set_id"), "normalization_id": doc.get("normalization_id")}
    identity = digest(contract)
    destination = output / task_key / f"step-{step}" / identity
    summary_path = destination / "summary.json"
    if summary_path.is_file():
        previous = read_json(summary_path)
        if previous.get("audit_id") == identity and all(file_digest(destination / name) == h for name, h in previous["artifacts"].items()):
            print(f"[audit-errors] reused {task_key}/step-{step}: {previous['metrics']['total_samples']} images | {destination}", flush=True)
            return previous
    destination.mkdir(parents=True, exist_ok=True)
    records, kinds, invalid_gt, format_gt, missing_raw = [], Counter(), Counter(), Counter(), []
    totals, invalid_contribution = Counter(), Counter()
    reconciliation = []
    for image_id, gt in track(source_by_id.items(), f"{task_key}/step-{step} 原评分归因", unit="原图"):
        sample = merged[image_id]
        boxes = gt["boxes_px"]
        if sample.get("objects", {}).get("bbox") != boxes:
            raise ValueError(f"Merged GT differs from frozen test: {image_id}")
        scored = evaluate_samples([sample], task_key, .1, include_figma=True)
        totals.update(scored)
        positive = bool(boxes)
        key = image_key(gt["source_image"])
        gate_path, gate = gates.get(key, (None, {}))
        raw_path = raw_by_image.get(key, (None,))[0]
        if raw_path is None and gate_path: raw_path = prediction / "raw" / gate_path.name
        raw = read_json(raw_path) if raw_path and raw_path.is_file() else {}
        answer = raw.get("raw_answer", raw.get("answer"))
        if not isinstance(answer, str): answer = None
        if answer is None: missing_raw.append(image_id)
        status = gate.get("prediction_status")
        runtime_failure = (key in errors and not gate) or status in ("error", "runtime_error", "failed")
        expected_prediction = ({"bbox": gate["final_boxes_pixel_xyxy"], "type": task_key}
                               if status in ("ok", "defect") else None)
        if gate and expected_prediction != sample.get("pred_ans"):
            reconciliation.append(f"Gate/merged prediction differs: {image_id}")
        kind, syntax = classify_record(answer, status, runtime_failure, scored, positive)
        kinds[kind] += 1
        if syntax and syntax != "missing_raw_evidence": format_gt["positive" if positive else "negative"] += 1
        if scored["invalid_pred"]:
            invalid_gt["positive" if positive else "negative"] += 1
            invalid_contribution.update({k: scored[k] for k in ("img_tp", "img_fp", "img_fn", "img_tn", "tp", "fp", "fn")})
        pred = sample.get("pred_ans")
        records.append({"step": step, "task_key": task_key, "source_image_id": image_id,
            "source_image": key, "width": gt["width"], "height": gt["height"],
            "positive": positive, "gt_boxes_px": boxes,
            "prediction_boxes_px": pred.get("bbox", []) if isinstance(pred, dict) else [],
            "raw_answer": answer, "raw_path": str(raw_path) if raw_path else None,
            "gate_path": str(gate_path) if gate_path else None, "gate_status": status,
            "runtime_error": errors.get(key, (None, {}))[1] if runtime_failure else None,
            "error_type": kind, "raw_format_issue": syntax, "scorer_invalid": bool(scored["invalid_pred"]),
            "scoring": build_metrics_summary(scored)})
    official = evaluate_merged_file(str(merged_path), task_key, .1, include_figma=True)
    if dict(totals) != official: raise AssertionError("Per-image contributions do not reproduce the original scorer")
    metrics = build_metrics_summary(official)
    saved_metrics = read_json(metrics_path)["tasks"][task_key]
    for granularity in ("image", "bbox"):
        for field in ("tp", "fp", "fn", "precision", "recall", "f1") + (("tn",) if granularity == "image" else ()):
            actual, previous = metrics[granularity][field], saved_metrics[granularity].get(field)
            if previous is None or not math.isclose(actual, previous, abs_tol=1e-6, rel_tol=0):
                reconciliation.append(f"Saved {granularity}.{field}: {previous!r}; recomputed: {actual!r}")
    if saved_metrics.get("invalid_pred") != official["invalid_pred"]:
        reconciliation.append("Saved invalid_pred differs from recomputed original scorer")
    write_jsonl(destination / "images.jsonl", records)
    write_jsonl(destination / "missing_raw.jsonl", [r for r in records if r["raw_answer"] is None])
    render_samples(records, destination / "samples.html", limit=examples)
    fields = {"task_key": task_key, "step": step, "images": metrics["total_samples"],
              "positive_images": sum(bool(r["positive"]) for r in records),
              "negative_images": sum(not r["positive"] for r in records),
              "image_f1": metrics["image"]["f1"], "bbox_f1": metrics["bbox"]["f1"],
              **{f"image_{k}": metrics["image"][k] for k in ("tp", "fp", "fn", "tn")},
              **{f"bbox_{k}": metrics["bbox"][k] for k in ("tp", "fp", "fn")},
              "scorer_invalid": official["invalid_pred"], "parse_error": sum(r["gate_status"] == "parse_error" for r in records),
              "missing_raw": len(missing_raw), **{f"error_{k}": v for k, v in kinds.items()}}
    with (destination / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields)); writer.writeheader(); writer.writerow(fields)
    reference = HISTORICAL_REFERENCE.get(step) if task_key == "ui_alignment" else None
    observed = {"image_f1": metrics["image"]["f1"], "bbox_f1": metrics["bbox"]["f1"],
                **{k: metrics["image"][k] for k in ("tp", "fp", "fn")}, "parse_error": fields["parse_error"]}
    result = {"audit_id": identity, "status": "complete" if not reconciliation else "mismatch",
              "contract": contract, "directory": str(destination), "csv_row": fields, "metrics": metrics,
              "error_types": dict(kinds), "scorer_invalid_by_gt": dict(invalid_gt),
              "invalid_scoring_contribution": dict(invalid_contribution), "raw_format_issue_by_gt": dict(format_gt),
              "reconciliation_errors": reconciliation, "historical_reference": reference,
              "historical_reference_matches": {k: math.isclose(observed[k], v, abs_tol=5e-5, rel_tol=0) for k, v in reference.items()} if reference else None,
              "artifacts": {name: file_digest(destination / name) for name in ("images.jsonl", "missing_raw.jsonl", "samples.html", "summary.csv")}}
    write_json(summary_path, result)
    print(f"[audit-errors] built {task_key}/step-{step}: {len(records)} images; missing_raw={len(missing_raw)}; status={result['status']} | {destination}", flush=True)
    return result


def audit_runs(old_run, manifest, task_keys, steps, output, *, examples=5):
    results = [audit_step(old_run, manifest, task, step, output, examples=examples) for task in task_keys for step in steps]
    output = Path(output)
    rows = [r["csv_row"] for r in results]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write_json(output / "latest.json", {"status": "complete" if all(r["status"] == "complete" for r in results) else "mismatch", "results": results})
    return results
