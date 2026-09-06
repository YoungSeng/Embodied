"""CPU-only split commits and migration of pre-resume UI14 normalization outputs.

Never inspect screenshot paths here: reuse trusts the source JSONL and output
digests from the same normalization snapshot. EXIF=0 acceptance is compatible
with every previously successful row and does not change that snapshot ID.
"""
from collections import defaultdict
from pathlib import Path

from ui14_common import digest, file_digest, paths_for, read_json, read_jsonl, write_json, write_jsonl
from ui14_progress import track

SPLIT_SCHEMA_VERSION = 1  # Bump for future changes incompatible with successful normalized rows.
OUTPUT_KEYS = ("normalized", "detector_input", "detector_inputs")


def split_state_paths(root, task_key, split):
    folder = Path(root) / "normalization_splits" / task_key
    return folder / f"{split}.state.json", folder / f"{split}.issues.jsonl"


def optional_json(path):
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def successful_stats(stats, source, split):
    expected = source[f"{split}_records"]
    return (expected > 0 and stats.get("failed_records") == 0
            and stats.get("records") == stats.get("normalized_records") == expected
            and stats.get("manifest_entry") == source)


def save_split_state(root, task, split, snapshot, stats, issues, errors, artifacts=None):
    """Publish the commit marker last, after labels, inputs, audit details and hashes."""
    root = Path(root)
    paths = paths_for(root, task.task_key, split)
    marker, issue_path = split_state_paths(root, task.task_key, split)
    write_jsonl(issue_path, issues)
    if artifacts is None:
        artifacts = {str(paths[k].relative_to(root)): file_digest(paths[k]) for k in OUTPUT_KEYS}
    value = {"schema_version": SPLIT_SCHEMA_VERSION, "normalization_id": snapshot["normalization_id"],
             "repair_run_id": snapshot["repair_run_id"], "task_key": task.task_key, "split": split,
             "complete": not errors and successful_stats(stats, snapshot["sources"][task.task_key], split),
             "stats": stats, "errors": errors, "issue_records": len(issues),
             "artifact_digests": {**artifacts, str(issue_path.relative_to(root)): file_digest(issue_path)}}
    write_json(marker, {**value, "state_digest": digest(value)})
    return artifacts


class SplitResume:
    def __init__(self, root, snapshot):
        self.root, self.snapshot = Path(root), snapshot
        self.previous = optional_json(self.root / "normalization_stats.json")
        saved_snapshot = optional_json(self.root / "source_snapshot.json")
        # Exact snapshot comparison includes its digest, all source files, parser
        # provenance and split contracts. Global complete/errors do not veto a split.
        self.legacy_compatible = (saved_snapshot == snapshot
            and self.previous.get("normalization_id") == snapshot["normalization_id"]
            and self.previous.get("source_files") == snapshot["source_files"]
            and self.previous.get("parser") == snapshot["parser"])
        self.legacy_issues = None
        self.legacy_issue_error = None
        self.actions = {}

    def _read_legacy_issues(self, task_key, split, stats):
        if self.legacy_issues is None:
            self.legacy_issues = defaultdict(list)
            try:
                for row in read_jsonl(self.root / "parser_compatibility_issues.jsonl"):
                    self.legacy_issues[row["task_key"], row["split"]].append(row)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.legacy_issue_error = str(exc)
                self.legacy_issues.clear()
        counts = stats["parser_comparison"]
        if self.legacy_issue_error and (counts.get("legacy_consumer_failure_records", 0)
                                        or counts.get("parse_result_difference_records", 0)):
            raise ValueError("Legacy parser audit details unavailable: " + self.legacy_issue_error)
        return self.legacy_issues[task_key, split]

    def load(self, task, split):
        """Return validated rows/stats/details/hashes, or a reason to rebuild this split."""
        root, snapshot = self.root, self.snapshot
        paths = paths_for(root, task.task_key, split)
        marker, issue_path = split_state_paths(root, task.task_key, split)
        source = snapshot["sources"][task.task_key]
        src = str(Path(snapshot["source_root"]) / task.task_key / f"{split}.jsonl")
        using_marker = marker.is_file()
        try:
            if using_marker:
                state = optional_json(marker)
                payload = {k: v for k, v in state.items() if k != "state_digest"}
                if (state.get("schema_version") != SPLIT_SCHEMA_VERSION or state.get("state_digest") != digest(payload)
                    or state.get("normalization_id") != snapshot["normalization_id"]
                    or state.get("repair_run_id") != snapshot["repair_run_id"]
                    or state.get("task_key") != task.task_key or state.get("split") != split
                    or not state.get("complete") or state.get("errors")):
                    raise ValueError("Split commit is incomplete, invalid or belongs to another snapshot")
                stats, recorded = state["stats"], state["artifact_digests"]
            else:
                if not self.legacy_compatible:
                    raise ValueError("No compatible previous normalization snapshot")
                stats = self.previous.get("tasks", {}).get(f"{task.task_key}/{split}", {})
                recorded = self.previous.get("artifact_digests", {})
            if not successful_stats(stats, source, split):
                raise ValueError("Previous split failed or has incomplete record counts")
            if (stats.get("input_sha256") != snapshot["source_files"][src]
                or stats.get("repair_counts") != next(r for r in snapshot["repair_summary"]["datasets"]
                                                      if r["dataset"] == task.task_key)):
                raise ValueError("Source digest or repair counts changed")
            artifacts = {}
            for key in OUTPUT_KEYS:
                name = str(paths[key].relative_to(root))
                actual = file_digest(paths[key])
                if recorded.get(name) != actual:
                    raise ValueError(f"Output digest mismatch: {name}")
                artifacts[name] = actual
            records = list(read_jsonl(paths["normalized"]))
            if len(records) != stats["normalized_records"]:
                raise ValueError("Normalized row count changed")
            for row in track(records, f"{task.task_key}/{split} 续跑绑定检查（不读原图）", unit="记录"):
                if (row["normalization_id"] != snapshot["normalization_id"] or row["repair_run_id"] != snapshot["repair_run_id"]
                    or row["source_jsonl_sha256"] != stats["input_sha256"] or row["source_jsonl"] != src
                    or row["task_key"] != task.task_key or row["task_id"] != task.task_id
                    or row["split"] != split or row["view_policy"] != task.view_policy):
                    raise ValueError("Normalized row source/task/split binding changed")
            if using_marker:
                if recorded.get(str(issue_path.relative_to(root))) != file_digest(issue_path):
                    raise ValueError("Split parser audit details changed")
                issues = list(read_jsonl(issue_path))
                if len(issues) != state["issue_records"]:
                    raise ValueError("Split parser audit row count changed")
            else:
                issues = self._read_legacy_issues(task.task_key, split, stats)
            for issue in issues:
                if (issue["task_key"], issue["split"], issue["source_jsonl"]) != (task.task_key, split, src):
                    raise ValueError("Parser audit details belong to another source")
            if not using_marker:
                # Migrating a legacy successful split writes only a small commit
                # and audit shard. Existing normalized/detector inputs stay untouched.
                save_split_state(root, task, split, snapshot, stats, issues, [], artifacts)
            return (records, stats, issues, artifacts), "split commit" if using_marker else "legacy successful split"
        except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as exc:
            return None, str(exc)

    def begin_rebuild(self, task, split):
        marker, _ = split_state_paths(self.root, task.task_key, split)
        # An interrupted rebuild must never fall back to a previous success marker.
        write_json(marker, {"complete": False, "normalization_id": self.snapshot["normalization_id"],
                            "task_key": task.task_key, "split": split, "status": "building"})

    def record(self, task, split, action, stats, reason):
        self.actions[f"{task.task_key}/{split}"] = {
            "action": action, "records": stats["records"], "normalized_records": stats["normalized_records"],
            "failed_records": stats["failed_records"], "reason": reason}
        print(f"[normalize {action}] {task.task_key}/{split}: records={stats['records']}, "
              f"normalized={stats['normalized_records']}, failed={stats['failed_records']} | {reason}", flush=True)

    def summary(self):
        result = {"splits": dict(self.actions)}
        for action in ("reused", "rebuilt"):
            rows = [s for s in self.actions.values() if s["action"] == action]
            result[action + "_splits"] = len(rows)
            result[action + "_records"] = sum(s["records"] for s in rows)
        result["normalized_records"] = sum(s["normalized_records"] for s in self.actions.values())
        result["failed_records"] = sum(s["failed_records"] for s in self.actions.values())
        return result
