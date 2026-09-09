"""Copy completed audit artifacts out of an old directory without modifying it.

The original audit implementation and identity contract deliberately stay
unchanged. audit_runs subsequently verifies current input/code digests and
reuses an imported result only if the complete identity still matches.
"""
from pathlib import Path
from ui14_common import read_json, write_json, file_digest, digest
from ui14_alignment_common import independent_output
from ui14_alignment_data import copy_verified
from ui14_progress import track

ARTIFACTS = {"images.jsonl", "missing_raw.jsonl", "samples.html", "summary.csv"}


def import_completed_audits(source, destination, tasks, steps):
    source = Path(source).resolve(strict=True)
    destination = independent_output(destination, source)
    result = {"source": str(source), "destination": str(destination),
              "imported": 0, "reused": 0, "skipped": [], "audits": []}
    paths = [path for task in tasks for step in steps
             for path in sorted((source / task / f"step-{step}").glob("*/summary.json"))]
    for summary_path in track(paths, "只读接续已完成审计", unit="审计"):
        try:
            source_folder = summary_path.parent.resolve()
            if source not in source_folder.parents:
                raise ValueError("Audit directory escapes the supplied read-only source")
            previous = read_json(summary_path)
            identity, contract = previous["audit_id"], previous["contract"]
            if (previous.get("status") != "complete" or identity != source_folder.name
                    or identity != digest(contract) or len(identity) != 64
                    or any(ch not in "0123456789abcdef" for ch in identity)):
                raise ValueError("Incomplete audit or invalid audit identity")
            relative = Path(contract["task_key"]) / f"step-{contract['step']}" / identity
            if (contract["task_key"] not in tasks or contract["step"] not in steps
                    or source_folder.relative_to(source) != relative
                    or set(previous["artifacts"]) != ARTIFACTS):
                raise ValueError("Audit task/step/artifact set differs from its directory")
            for name, expected in previous["artifacts"].items():
                path = (source_folder / name).resolve(strict=True)
                if source_folder not in path.parents or file_digest(path) != expected:
                    raise ValueError(f"Incomplete/changed audit artifact: {name}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["skipped"].append({"summary": str(summary_path), "reason": str(exc)})
            print(f"[audit import] skipped {summary_path}: {exc}", flush=True)
            continue
        target = (destination / relative).resolve()
        if destination not in target.parents:
            raise ValueError(f"Audit import target escapes new output: {target}")
        for name, expected in previous["artifacts"].items():
            copy_verified(source_folder / name, target / name, expected)
        # summary.json is published only after all four copied artifacts pass.
        imported = {**previous, "directory": str(target), "reused_from": str(summary_path)}
        target_summary = target / "summary.json"
        unchanged = target_summary.is_file() and read_json(target_summary) == imported
        if not unchanged: write_json(target_summary, imported)
        result["reused" if unchanged else "imported"] += 1
        result["audits"].append(str(target_summary))
        print(f"[audit import] {'reused' if unchanged else 'imported'} "
              f"{contract['task_key']}/step-{contract['step']} -> {target}", flush=True)
    print(f"[audit import] imported={result['imported']} reused={result['reused']} "
          f"skipped={len(result['skipped'])}; current source identities will be verified by audit-errors", flush=True)
    return result
