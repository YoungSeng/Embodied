"""Discover unreferenced page-local files without manufacturing clean labels."""
from collections import Counter, defaultdict
import re

from ui14_common import write_json, write_jsonl


def discover_local_candidates(root, resolver, tasks, parent_rows, declared_local_paths):
    pages = defaultdict(list)
    for row in parent_rows:
        if row.get("boxes_px") and row.get("source_page_id"):
            pages[row["task_key"], row["source_page_id"]].append(row)
    observations, totals = [], {}
    for task in tasks:
        counts = Counter()
        for path in resolver.index(task.task_key)[0]:
            if path.parent.name != "local_imgs":
                continue
            match = re.fullmatch(r"(.+)_([0-9]+[:_][0-9]+)_local", path.stem)
            page = match[1] + ":" + match[2].replace("_", ":") if match else None
            paired = pages.get((task.task_key, page), [])
            declared = (task.task_key, str(path)) in declared_local_paths
            state = "declared_local_reference" if declared else "unconfirmed_local_candidate"
            counts.update(files=1, page_name_recognized=int(page is not None),
                          same_task_page_matched=int(bool(paired)))
            counts[state] += 1
            observations.append({"task_key": task.task_key, "image": str(path),
                "source_page_id": page, "evidence_status": state,
                "paired_positive_records": len(paired),
                "paired_record_examples": [r["source_record_id"] for r in paired[:3]],
                "observed_parent_splits": sorted({r["split"] for r in paired}),
                "content_identity_checked": False, "final_split_check_complete": False,
                "admitted_by_discovery": False})
        totals[task.task_key] = dict(counts)
    destination = root / "local_candidate_discovery"
    write_jsonl(destination / "candidates.jsonl", observations)
    write_json(destination / "summary.json", {"tasks": totals,
        "scope": "file paths and page names only; no image decode or new labels",
        "label_changes": 0, "selection_changes": 0})
    return totals
