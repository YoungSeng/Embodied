#!/usr/bin/env python3
"""Read-only raw-output diagnosis. NEVER produces or repairs predictions."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tarfile

TASKS = ("occlusion", "cropping", "text_overflow", "text_ellipsis", "content_missing")


def classify_output(answer, status, trace=None, runtime_error=None):
    if runtime_error:
        return "runtime_error"
    if answer is None:
        return "missing_raw"
    if not isinstance(answer, str):
        raise ValueError("raw generated answer must be text, not a prediction object")
    if status != "parse_error":
        return "valid"
    trace = trace or {}
    if trace.get("termination_reason") == "length_limit" or (
            trace.get("termination_reason") is None and trace.get("length_limit_reached") is True):
        return "length_truncated"
    content = re.sub(r"<\|(?:im_end|endoftext)\|>|<null>", "", answer, flags=re.I).strip()
    if not content:
        return "empty_answer"
    if re.search(r"<ref>(?:(?!</ref>|<box>).)*</box>", content, flags=re.S):
        return "ref_box_tag_mix"
    opened = len(re.findall(r"<box>", content, flags=re.I))
    closed = len(re.findall(r"</box>", content, flags=re.I))
    if opened != closed:
        return "box_unclosed"
    if re.search(r"<ref>.*?</ref>", content, flags=re.I | re.S) and opened == 0:
        return "ref_only"
    if "<ref>" in content and opened == 0:
        return "ref_unclosed"
    if opened:
        return "malformed_box"
    return "other_invalid"


def audit_evaluation(directory: Path):
    directory = directory.resolve(strict=True)
    records, examples, summaries = [], [], {}
    for task in TASKS:
        summary_path = directory / "_worker_summaries" / f"{task}.json"
        if not summary_path.is_file():
            raise ValueError(f"worker summary missing; do not infer counts from F1: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries[task] = {"path": str(summary_path), "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()}
        counters = Counter()
        paths = sorted((directory / f"ui_{task}" / "raw").glob("*.json"))
        expected = int(summary["totals"]["dataset_images"])
        seen = set()
        for index, path in enumerate(paths, 1):
            raw = json.loads(path.read_text(encoding="utf-8"))
            image = raw.get("image_path")
            if not image or image in seen:
                raise ValueError(f"raw output has duplicate/missing image identity: {path}")
            seen.add(image)
            status = raw["parse"]["status"]
            counters["images_observed"] += 1
            counters["image_invalid"] += int(status == "parse_error")
            tiles = (raw.get("inference_crop") or {}).get("tiles") or [{
                "answer": raw.get("raw_answer"), "status": status, "gate": raw.get("gate", {})}]
            for tile_index, tile in enumerate(tiles):
                trace = (tile.get("gate") or {}).get("decode_trace", {})
                reason = classify_output(tile.get("answer"), tile["status"], trace, tile.get("runtime_error"))
                counters["tiles_observed"] += 1
                counters["tile_invalid"] += int(tile["status"] == "parse_error")
                counters["reason:" + reason] += 1
                if reason != "valid" and counters["reason:" + reason] <= 4:
                    examples.append({"task": task, "reason": reason, "raw_path": str(path),
                                     "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "tile_index": tile_index, "raw_generated_text": tile.get("answer"),
                                     "decode_trace": trace})
            if index % 500 == 0:
                print(f"[RAW AUDIT] task={task} completed={index}/{len(paths)}", flush=True)
        if len(paths) > expected:
            raise ValueError(f"more raw images than worker input images: {task}")
        counters["missing_raw_images"] = expected - len(paths)
        counters["runtime_errors"] = int(summary["totals"].get("inference_error", 0))
        # Missing/runtime results remain missing/runtime, NEVER empty negatives.
        records.append({"task": f"ui_{task}", "expected_images": expected,
                        "summary_parse_errors": summary["totals"].get("parse_error"),
                        "length_truncation_evidence": "trace_only; unknown for old raw without trace",
                        **dict(counters)})
        print(f"[OUTPUT VALIDITY] task={task} images={len(paths)}/{expected} "
              f"invalid_images={counters['image_invalid']} invalid_tiles={counters['tile_invalid']} "
              f"runtime={counters['runtime_errors']} categories=" + json.dumps(
                  {k[7:]: v for k, v in counters.items() if k.startswith('reason:')}, sort_keys=True), flush=True)
    return {"schema_version": 1, "evaluation_dir": str(directory), "scope": "UI5 held-out diagnostic only",
            "scoring_changed": False, "rows": records, "examples": examples,
            "worker_summaries": summaries,
            "root_cause_status": "observed output categories; causal attribution requires paired decoding evidence"}


def audit_archive(path: Path):
    """Read an evidence export WITHOUT extraction or treating unexported raw as failures."""
    stats, answers, examples, summaries, metrics, identities = {}, {}, {}, {}, {}, {}
    with tarfile.open(path, "r:gz") as archive:
        names = set()
        for member in archive:
            if not member.isfile():
                continue
            if member.name in names:
                raise ValueError(f"duplicate evidence member: {member.name}")
            names.add(member.name)
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[0] != "evaluation" or parts[1] not in {"step-000000", "step-000200"}:
                continue
            if member.size > 64 * 1024 * 1024 or not member.name.endswith(".json"):
                continue
            value = json.load(archive.extractfile(member))
            step = int(parts[1][5:])
            if parts[2] == "ui5_metrics.json":
                metrics[step] = {"overall": value["overall"], "micro": value["micro"],
                                 "macro": value["macro"], "invalid": sum(t["invalid_pred"] for t in value["tasks"].values())}
            elif parts[2] == "evaluation_manifest.json":
                identities[step] = value
            elif parts[2] == "_worker_summaries":
                summaries[step, Path(parts[3]).stem] = value["totals"]
            elif len(parts) == 5 and parts[3] == "raw":
                key = step, parts[2].removeprefix("ui_")
                counts = stats.setdefault(key, Counter())
                counts["exported_images"] += 1
                counts["exported_invalid_images"] += value["parse"]["status"] == "parse_error"
                tiles = (value.get("inference_crop") or {}).get("tiles") or [{
                    "answer": value.get("raw_answer"), "status": value["parse"]["status"]}]
                for tile in tiles:
                    counts["exported_tiles"] += 1
                    if tile["status"] != "parse_error":
                        continue
                    answer = tile.get("answer")
                    counts["exported_invalid_tiles"] += 1
                    counts["reason:" + classify_output(answer, tile["status"])] += 1
                    answers.setdefault(key, Counter())[answer] += 1
                    examples.setdefault((key, answer), member.name)
    rows = []
    for step in (0, 200):
        for task in TASKS:
            key = step, task
            summary, counts = summaries[key], stats.get(key, Counter())
            rows.append({"step": step, "task": task, **dict(counts), "worker_totals": summary,
                         "invalid_export_complete": counts["exported_invalid_images"] == summary["parse_error"],
                         "unexported_images": summary["dataset_images"] - counts["exported_images"],
                         "top_invalid_texts": [{"text": answer, "count": count, "example": examples[key, answer]}
                                               for answer, count in answers.get(key, Counter()).most_common(12)]})
    return {"archive": str(path), "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "export_policy": "subset: missing exported raw is NOT a runtime failure or empty prediction",
            "rows": rows, "metrics": metrics, "evaluation_identities": identities,
            "length_truncation": "unknown: historical raw lacks token-decision trace",
            "token_id_mismatch": "not established: model/processor tokenizer files absent from archive"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evaluation-dir", type=Path)
    source.add_argument("--evidence-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_archive(args.evidence_archive) if args.evidence_archive else audit_evaluation(args.evaluation_dir)
    # Explicit new report only. Never overwrite an older diagnostic or raw.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report["rows"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
