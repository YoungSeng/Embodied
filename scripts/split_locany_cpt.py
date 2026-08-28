#!/usr/bin/env python3
"""Create deterministic, image-grouped train/validation CPT recipes.

The splitter never assigns rows independently.  Every row sharing the same
image content digest is assigned as one group, including rows from different
CPT tasks.  The manifest is the stable source of truth for later training,
coverage reporting, and held-out evaluation.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import tempfile
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


DEFAULT_SEED = 20260826
DEFAULT_VAL_FRACTION = 0.02
DEFAULT_VAL_FAST_PER_TASK = 200

_UNSUPPORTED_FSYNC_ERRNOS = {
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def best_effort_fsync(handle: Any, path: Path) -> None:
    """Durably flush when supported, but tolerate ByteNAS ENOSYS/ENOTSUP."""

    try:
        os.fsync(handle.fileno())
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
        warnings.warn(
            f"filesystem does not support fsync for {path} ({exc}); "
            "continuing with close + atomic replace"
        )


def stable_hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_unit_interval(*values: object) -> float:
    return int(stable_hash(*values)[:16], 16) / float(1 << 64)


def stable_hash64(value: object) -> int:
    """Match the compact hash stored in runtime packed-sample metadata."""
    return int.from_bytes(
        hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def iter_image_values(value: Any) -> Iterator[str]:
    if isinstance(value, str) and value:
        yield value
    elif isinstance(value, dict):
        candidate = value.get("path") or value.get("image") or value.get("file")
        if isinstance(candidate, str) and candidate:
            yield candidate
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_image_values(item)


def record_image_values(record: dict[str, Any]) -> list[str]:
    for key in ("image", "images", "original_images", "image_list"):
        values = list(iter_image_values(record.get(key)))
        if values:
            return values
    return []


def resolve_record_images(record: dict[str, Any], root: Path | None) -> list[Path]:
    output = []
    for value in record_image_values(record):
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        if not path.is_absolute() and root is not None:
            path = root / path
        output.append(path.resolve())
    if not output:
        raise ValueError("record has no image/images/original_images path")
    return output


class ImageHashCache:
    """Persistent SHA-256 cache keyed by canonical path and file stat."""

    def __init__(self, path: Path):
        self.path = path
        self.values: dict[str, dict[str, Any]] = {}
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.values = loaded
        self.dirty = False

    def digest(self, path: Path) -> str:
        stat = path.stat()
        key = path.as_posix()
        cached = self.values.get(key)
        if (
            isinstance(cached, dict)
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
        ):
            return str(cached["sha256"])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self.values[key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": value,
        }
        self.dirty = True
        return value

    def save(self) -> None:
        if not self.dirty:
            return
        atomic_write_json(self.path, self.values, sort_keys=True)
        self.dirty = False


def normalized_path_digest(paths: Iterable[Path]) -> str:
    return stable_hash("paths", *(sorted(path.as_posix().casefold() for path in paths)))


def image_group_id(
    paths: list[Path], cache: ImageHashCache, mode: str = "sha256"
) -> tuple[str, list[str], list[str]]:
    normalized_paths = sorted(path.as_posix() for path in paths)
    if mode == "path":
        content_digests = [stable_hash("path", path.casefold()) for path in normalized_paths]
    elif mode == "sha256":
        content_digests = sorted(cache.digest(Path(path)) for path in normalized_paths)
    else:
        raise ValueError(f"unsupported group id mode: {mode}")
    return (
        stable_hash("image-group-v1", *content_digests),
        normalized_paths,
        content_digests,
    )


class ImageIdentityUnion:
    """Deterministic union-find for records that share any image identity."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def connect(self, values: Iterable[str]) -> None:
        members = sorted(set(str(value) for value in values))
        if not members:
            raise ValueError("record has no image identities")
        root = self.find(members[0])
        for value in members[1:]:
            other = self.find(value)
            if root == other:
                continue
            # A lexical root keeps the result independent of annotation order.
            root, child = sorted((root, other))
            self.parent[child] = root

    def group_ids(self) -> dict[str, str]:
        components: dict[str, set[str]] = defaultdict(set)
        for value in sorted(self.parent):
            components[self.find(value)].add(value)
        output = {}
        for members in components.values():
            group_id = stable_hash("image-group-v1", *sorted(members))
            for value in members:
                output[value] = group_id
        return output


def connected_image_group_id(
    paths: list[Path],
    cache: ImageHashCache,
    identity_to_group: Mapping[str, str],
    mode: str = "sha256",
) -> tuple[str, list[str], list[str]]:
    _, normalized_paths, content_digests = image_group_id(paths, cache, mode)
    group_ids = {identity_to_group[digest] for digest in content_digests}
    if len(group_ids) != 1:
        raise RuntimeError(
            "image identity connected-component mismatch: "
            f"paths={normalized_paths}, groups={sorted(group_ids)}"
        )
    return next(iter(group_ids)), normalized_paths, content_digests


def _answer(record: dict[str, Any]) -> str:
    for turn in reversed(record.get("conversations", [])):
        if str(turn.get("from", turn.get("role", ""))).lower() in {"gpt", "assistant"}:
            value = turn.get("value", turn.get("content", ""))
            return value if isinstance(value, str) else str(value)
    return ""


def record_strata(task: str, record: dict[str, Any]) -> set[str]:
    """Return small label strata that must remain represented in validation."""
    answer = _answer(record).strip().lower()
    if task == "vqa":
        if "不正确" in answer or "错误" in answer or "false" in answer:
            return {"vqa:incorrect"}
        if "正确" in answer or "true" in answer:
            return {"vqa:correct"}
        return {"vqa:unknown"}
    if task == "ui_defect":
        import re

        labels = {
            re.sub(r"\s+", " ", value).strip().casefold()
            for value in re.findall(r"<ref>(.*?)</ref>", answer, flags=re.DOTALL)
            if value.strip()
        }
        return {f"ui_defect:{label}" for label in labels} or {"ui_defect:none"}
    return {f"task:{task}"}


def _resolve_path(value: str, recipe_path: Path, relative: bool) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((recipe_path.parent if relative else Path.cwd()) / path).resolve()


@dataclass(frozen=True)
class RecipeSource:
    recipe_name: str
    task: str
    annotation: Path
    root: Path | None
    meta: dict[str, Any]


def load_recipe_sources(recipe_path: Path) -> tuple[OrderedDict[str, dict[str, Any]], list[RecipeSource]]:
    recipe_path = recipe_path.expanduser().resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    if not isinstance(recipe, dict) or not recipe:
        raise ValueError("recipe must be a non-empty object")
    sources: list[RecipeSource] = []
    for recipe_name, raw_meta in recipe.items():
        if not isinstance(raw_meta, dict):
            raise ValueError(f"{recipe_name}: recipe metadata is not an object")
        meta = dict(raw_meta)
        task = str(meta.get("cpt_task") or recipe_name.removeprefix("locany_cpt_"))
        relative = bool(meta.get("paths_relative_to_meta", False))
        annotations = meta.get("annotation", [])
        if isinstance(annotations, str):
            annotations = [annotations]
        root_value = str(meta.get("root", "")).strip()
        root = _resolve_path(root_value, recipe_path, relative) if root_value else None
        for value in annotations:
            sources.append(
                RecipeSource(
                    recipe_name=recipe_name,
                    task=task,
                    annotation=_resolve_path(str(value), recipe_path, relative),
                    root=root,
                    meta=meta,
                )
            )
    return OrderedDict((str(k), dict(v)) for k, v in recipe.items()), sources


def iter_source_records(source: RecipeSource) -> Iterator[tuple[int, dict[str, Any]]]:
    with source.annotation.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source.annotation}:{line_number}: row is not an object")
            yield line_number, value


def source_record_id(record: Mapping[str, Any]) -> str | None:
    existing = record.get(
        "cpt_source_record_id",
        record.get("cpt_record_id", record.get("id")),
    )
    if existing is not None and str(existing).strip():
        return str(existing).strip()
    return None


def base_record_id(
    source: RecipeSource, line_number: int, record: Mapping[str, Any]
) -> str:
    existing = source_record_id(record)
    if existing is not None:
        return f"{source.task}:{existing}"
    return f"{source.task}:{source.annotation.name}:{line_number}"


def record_source_locator(
    source: RecipeSource, line_number: int, record: Mapping[str, Any]
) -> tuple[str, int]:
    """Return a cluster-portable source row locator for ID disambiguation."""

    raw_source = str(record.get("cpt_source") or source.annotation.name)
    normalized_source = raw_source.replace("\\", "/").strip()
    raw_line = record.get("cpt_source_line", line_number)
    try:
        source_line = int(raw_line)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid cpt_source_line={raw_line!r} for {source.annotation}:{line_number}"
        ) from exc
    return normalized_source, source_line


def stable_record_id(
    source: RecipeSource,
    line_number: int,
    record: Mapping[str, Any],
    *,
    duplicate_base_ids: set[str] | frozenset[str] = frozenset(),
) -> str:
    """Keep unique business IDs readable and deterministically disambiguate duplicates."""

    base = base_record_id(source, line_number, record)
    if base not in duplicate_base_ids:
        return base
    source_name, source_line = record_source_locator(source, line_number, record)
    suffix = stable_hash(
        "duplicate-record-id-v1",
        source.task,
        source_name,
        source_line,
    )[:16]
    return f"{base}:dup-{suffix}"


@dataclass
class GroupInfo:
    tasks: set[str] = field(default_factory=set)
    strata: set[str] = field(default_factory=set)
    rows: int = 0
    normalized_paths: set[str] = field(default_factory=set)
    content_digests: set[str] = field(default_factory=set)


def choose_validation_groups(
    groups: dict[str, GroupInfo], seed: int, val_fraction: float
) -> set[str]:
    selected = {
        group_id
        for group_id in groups
        if stable_unit_interval(seed, "split", group_id) < val_fraction
    }

    task_groups: dict[str, set[str]] = defaultdict(set)
    stratum_groups: dict[str, set[str]] = defaultdict(set)
    for group_id, info in groups.items():
        for task in info.tasks:
            task_groups[task].add(group_id)
        for stratum in info.strata:
            stratum_groups[stratum].add(group_id)

    # Small tasks/classes must not silently receive an empty validation split.
    # A singleton cannot populate both train and validation, so it is reported
    # in the summary instead of being duplicated across the boundary.
    for values in (*task_groups.values(), *stratum_groups.values()):
        if len(values) >= 2 and selected.isdisjoint(values):
            selected.add(min(values, key=lambda value: stable_hash(seed, "promote", value)))

    # Preserve at least one training group for every task when possible.
    for task, values in task_groups.items():
        if len(values) >= 2 and values.issubset(selected):
            selected.remove(max(values, key=lambda value: stable_hash(seed, "demote", task, value)))
    return selected


def atomic_write_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        handle.write("\n")
        handle.flush()
        best_effort_fsync(handle, temporary)
    os.replace(temporary, path)


def _recipe_for_split(
    original: OrderedDict[str, dict[str, Any]],
    split: str,
    task_stats: Mapping[str, Mapping[str, int]] | None = None,
) -> OrderedDict[str, dict[str, Any]]:
    output: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for recipe_name, raw_meta in original.items():
        task = str(raw_meta.get("cpt_task") or recipe_name.removeprefix("locany_cpt_"))
        meta = dict(raw_meta)
        meta.update(
            {
                "annotation": [f"../{split}/{task}.jsonl"],
                "paths_relative_to_meta": True,
                "cpt_task": task,
                "cpt_split": "heldout" if split in {"val", "val_fast"} else "train",
            }
        )
        if task_stats is not None and task in task_stats:
            meta["dataset_rows"] = int(task_stats[task].get(f"{split}_rows", 0))
            meta["dataset_groups"] = int(task_stats[task].get(f"{split}_groups", 0))
        # Image paths inside normalized records are already absolute unless a
        # portable bundle requested a media root.  Preserve the original root.
        output[recipe_name] = meta
    return output


def _fast_subset(
    val_path: Path, limit: int, seed: int, task: str
) -> list[dict[str, Any]]:
    # The full held-out pool is only ~2%, so retaining one task at a time is
    # bounded and lets VQA/UI-defect preserve deterministic label strata.
    records: list[tuple[int, dict[str, Any]]] = []
    with val_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            score = int(
                stable_hash(
                    seed,
                    "val-fast",
                    record.get("cpt_group_id"),
                    record.get("cpt_record_id"),
                )[:16],
                16,
            )
            records.append((score, record))
    records.sort(key=lambda item: (item[0], str(item[1].get("cpt_record_id"))))
    if len(records) <= limit or task not in {"vqa", "ui_defect"}:
        return [record for _, record in records[:limit]]

    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in records:
        strata = sorted(record_strata(task, item[1]))
        # UI-defect can be multi-label. Treat the stable label combination as
        # its stratum so rare combinations are not silently discarded.
        bucket = "|".join(strata) if strata else f"{task}:unknown"
        buckets[bucket].append(item)
    total = len(records)
    quotas = {
        bucket: max(1, int(limit * len(values) / total))
        for bucket, values in buckets.items()
    }
    while sum(quotas.values()) > limit:
        candidates = [bucket for bucket, quota in quotas.items() if quota > 1]
        if not candidates:
            break
        bucket = max(
            candidates,
            key=lambda value: (quotas[value] / len(buckets[value]), value),
        )
        quotas[bucket] -= 1
    while sum(quotas.values()) < limit:
        candidates = [
            bucket
            for bucket, values in buckets.items()
            if quotas[bucket] < len(values)
        ]
        if not candidates:
            break
        bucket = max(
            candidates,
            key=lambda value: (
                len(buckets[value]) * limit / total - quotas[value],
                value,
            ),
        )
        quotas[bucket] += 1
    selected = [
        item
        for bucket, values in sorted(buckets.items())
        for item in values[: quotas[bucket]]
    ]
    selected.sort(key=lambda item: (item[0], str(item[1].get("cpt_record_id"))))
    return [record for _, record in selected[:limit]]


def split_recipe(
    recipe_path: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    val_fast_per_task: int = DEFAULT_VAL_FAST_PER_TASK,
    group_id_mode: str = "sha256",
    train_recipe_name: str = "locany_cpt_train.json",
    progress_every: int = 1000,
) -> dict[str, Any]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if val_fast_per_task <= 0:
        raise ValueError("val_fast_per_task must be positive")

    recipe_path = recipe_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    diagnostics_dir = output_dir / "diagnostics"
    cache = ImageHashCache(diagnostics_dir / "image_hash_cache.json")
    original_recipe, sources = load_recipe_sources(recipe_path)

    split_started = time.time()

    def report_progress(
        phase: str,
        rows: int,
        phase_started: float,
        *,
        done: bool = False,
    ) -> None:
        elapsed = max(time.time() - phase_started, 1.0e-9)
        state = "DONE" if done else "PROGRESS"
        print(
            f"[split] phase={phase} state={state} rows={rows:,} "
            f"rows_per_second={rows / elapsed:.1f} "
            f"phase_seconds={elapsed:.1f} total_seconds={time.time() - split_started:.1f}",
            flush=True,
        )

    identity_union = ImageIdentityUnion()
    base_record_id_counts: Counter[str] = Counter()
    path_duplicate_candidates: dict[tuple[str, int], set[str]] = defaultdict(set)
    phase = "hash_images_and_connect_groups"
    phase_started = time.time()
    phase_rows = 0
    print(
        f"[split] phase={phase} state=START mode={group_id_mode} "
        f"cached_images={len(cache.values):,}",
        flush=True,
    )
    try:
        for source in sources:
            for line_number, record in iter_source_records(source):
                paths = resolve_record_images(record, source.root)
                if group_id_mode == "path":
                    for path in paths:
                        stat = path.stat()
                        path_duplicate_candidates[
                            (path.name.casefold(), int(stat.st_size))
                        ].add(path.as_posix())
                _, _, content_digests = image_group_id(paths, cache, group_id_mode)
                identity_union.connect(content_digests)
                base_record_id_counts[
                    base_record_id(source, line_number, record)
                ] += 1
                phase_rows += 1
                if progress_every and phase_rows % progress_every == 0:
                    report_progress(phase, phase_rows, phase_started)
    finally:
        # Preserve expensive content hashes even when a user interrupts the
        # first full NAS scan. OVERWRITE=1 can then rebuild annotations while
        # reusing the cache instead of re-reading every image byte.
        cache.save()
    report_progress(phase, phase_rows, phase_started, done=True)
    identity_to_group = identity_union.group_ids()
    duplicate_base_ids = {
        record_id
        for record_id, count in base_record_id_counts.items()
        if count > 1
    }
    duplicate_source_record_rows = sum(
        base_record_id_counts[record_id] for record_id in duplicate_base_ids
    )
    if duplicate_base_ids:
        print(
            "[split] duplicate source record IDs will be deterministically "
            f"disambiguated: ids={len(duplicate_base_ids):,} "
            f"rows={duplicate_source_record_rows:,} "
            f"examples={sorted(duplicate_base_ids)[:5]}",
            flush=True,
        )

    # Aggregate tasks and label strata only after shared-image connected
    # components are complete. This prevents a multi-image record [A, B] from
    # being split away from single-image records A or B.
    groups: dict[str, GroupInfo] = {}
    phase = "aggregate_groups_and_strata"
    phase_started = time.time()
    phase_rows = 0
    print(f"[split] phase={phase} state=START", flush=True)
    for source in sources:
        for line_number, record in iter_source_records(source):
            paths = resolve_record_images(record, source.root)
            group_id, normalized_paths, content_digests = connected_image_group_id(
                paths, cache, identity_to_group, group_id_mode
            )
            info = groups.setdefault(group_id, GroupInfo())
            info.tasks.add(source.task)
            info.strata.update(record_strata(source.task, record))
            info.rows += 1
            info.normalized_paths.update(normalized_paths)
            info.content_digests.update(content_digests)
            phase_rows += 1
            if progress_every and phase_rows % progress_every == 0:
                report_progress(phase, phase_rows, phase_started)
    report_progress(phase, phase_rows, phase_started, done=True)
    path_duplicate_suspects = [
        {
            "basename": basename,
            "size": size,
            "paths": sorted(paths),
        }
        for (basename, size), paths in sorted(path_duplicate_candidates.items())
        if len(paths) > 1
    ]
    if group_id_mode == "path":
        atomic_write_json(
            diagnostics_dir / "path_duplicate_suspects.json",
            {
                "schema_version": 1,
                "note": "same-basename/same-size candidates require manual or sampled content-hash review",
                "suspects": path_duplicate_suspects,
            },
        )

    validation_groups = choose_validation_groups(groups, seed, val_fraction)
    split_paths: dict[tuple[str, str], Path] = {}
    handles: dict[tuple[str, str], Any] = {}
    manifest_path = diagnostics_dir / "split_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_handle = manifest_path.open("w", encoding="utf-8")
    task_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "train_rows": 0,
            "val_rows": 0,
            "train_groups": set(),
            "val_groups": set(),
            "train_labels": Counter(),
            "val_labels": Counter(),
        }
    )
    phase = "write_train_val_and_manifest"
    phase_started = time.time()
    phase_rows = 0
    final_record_ids: set[str] = set()
    duplicate_record_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    print(f"[split] phase={phase} state=START", flush=True)
    try:
        for source in sources:
            for line_number, record in iter_source_records(source):
                paths = resolve_record_images(record, source.root)
                group_id, normalized_paths, content_digests = connected_image_group_id(
                    paths, cache, identity_to_group, group_id_mode
                )
                split = "val" if group_id in validation_groups else "train"
                base_id = base_record_id(source, line_number, record)
                record_id = stable_record_id(
                    source,
                    line_number,
                    record,
                    duplicate_base_ids=duplicate_base_ids,
                )
                if record_id in final_record_ids:
                    raise ValueError(
                        "duplicate record_id remains after source-row disambiguation: "
                        f"record_id={record_id!r}, annotation={source.annotation}, "
                        f"line={line_number}; check whether the same annotation source "
                        "was listed more than once"
                    )
                final_record_ids.add(record_id)
                original_record_id = source_record_id(record)
                strata = sorted(record_strata(source.task, record))
                output_record = dict(record)
                output_record.update(
                    {
                        "cpt_task": source.task,
                        "cpt_record_id": record_id,
                        "cpt_source_record_id": original_record_id,
                        "cpt_group_id": group_id,
                        "cpt_split": "heldout" if split == "val" else "train",
                    }
                )
                key = split, source.task
                if key not in handles:
                    path = output_dir / split / f"{source.task}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    split_paths[key] = path
                    handles[key] = path.open("w", encoding="utf-8")
                handles[key].write(json.dumps(output_record, ensure_ascii=False, separators=(",", ":")) + "\n")

                values = task_summary[source.task]
                values[f"{split}_rows"] += 1
                values[f"{split}_groups"].add(group_id)
                values[f"{split}_labels"].update(strata)
                manifest = {
                    "manifest_version": 1,
                    "grouping_algorithm": "shared_image_connected_components_v1",
                    "seed": seed,
                    "group_id_mode": group_id_mode,
                    "group_id": group_id,
                    "split": "heldout" if split == "val" else "train",
                    "task": source.task,
                    "record_id": record_id,
                    "source_record_id": original_record_id,
                    "record_id_hash": stable_hash64(record_id),
                    "group_id_hash": stable_hash64(group_id),
                    "source": str(record.get("cpt_source") or source.annotation),
                    "line": int(record.get("cpt_source_line") or line_number),
                    "annotation": str(source.annotation),
                    "annotation_line": line_number,
                    "image": normalized_paths,
                    "image_sha256": content_digests,
                    "strata": strata,
                }
                manifest_handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
                if base_id in duplicate_base_ids:
                    duplicate_record_details[base_id].append(
                        {
                            "record_id": record_id,
                            "task": source.task,
                            "source_record_id": original_record_id,
                            "source": manifest["source"],
                            "line": manifest["line"],
                            "annotation": manifest["annotation"],
                            "annotation_line": manifest["annotation_line"],
                        }
                    )
                phase_rows += 1
                if progress_every and phase_rows % progress_every == 0:
                    report_progress(phase, phase_rows, phase_started)
    finally:
        for handle in handles.values():
            handle.close()
        manifest_handle.close()
    report_progress(phase, phase_rows, phase_started, done=True)
    cache.save()
    duplicate_report_path = diagnostics_dir / "duplicate_record_ids.json"
    atomic_write_json(
        duplicate_report_path,
        {
            "schema_version": 1,
            "policy": "preserve unique task:source_id values; append a stable source-row hash only to duplicates",
            "duplicate_source_record_id_count": len(duplicate_base_ids),
            "duplicate_source_record_row_count": duplicate_source_record_rows,
            "duplicates": dict(sorted(duplicate_record_details.items())),
        },
        sort_keys=True,
    )

    # Ensure empty task files exist so recipes are schema-stable and validation
    # can produce a clear empty-split error rather than FileNotFoundError.
    tasks = [str(meta.get("cpt_task") or name.removeprefix("locany_cpt_")) for name, meta in original_recipe.items()]
    for task in tasks:
        for split in ("train", "val"):
            path = output_dir / split / f"{task}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    val_fast_dir = output_dir / "val_fast"
    val_fast_counts = {}
    val_fast_groups = {}
    print("[split] phase=build_val_fast state=START", flush=True)
    for task in tasks:
        records = _fast_subset(
            output_dir / "val" / f"{task}.jsonl",
            val_fast_per_task,
            seed,
            task,
        )
        path = val_fast_dir / f"{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        val_fast_counts[task] = len(records)
        val_fast_groups[task] = len(
            {str(record.get("cpt_group_id")) for record in records}
        )
        print(
            f"[split] phase=build_val_fast task={task} rows={len(records):,}",
            flush=True,
        )

    recipe_task_stats = {
        task: {
            "train_rows": int(task_summary[task]["train_rows"]),
            "train_groups": len(task_summary[task]["train_groups"]),
            "val_rows": int(task_summary[task]["val_rows"]),
            "val_groups": len(task_summary[task]["val_groups"]),
            "val_fast_rows": int(val_fast_counts[task]),
            "val_fast_groups": int(val_fast_groups[task]),
        }
        for task in tasks
    }

    recipe_dir = output_dir / "recipe"
    atomic_write_json(
        recipe_dir / train_recipe_name,
        _recipe_for_split(original_recipe, "train", recipe_task_stats),
    )
    if train_recipe_name != "locany_cpt_train.json":
        atomic_write_json(
            recipe_dir / "locany_cpt_train.json",
            _recipe_for_split(original_recipe, "train", recipe_task_stats),
        )
    atomic_write_json(
        recipe_dir / "locany_cpt_val.json",
        _recipe_for_split(original_recipe, "val", recipe_task_stats),
    )
    atomic_write_json(
        recipe_dir / "locany_cpt_val_fast.json",
        _recipe_for_split(original_recipe, "val_fast", recipe_task_stats),
    )

    summary_tasks = {}
    for task in tasks:
        values = task_summary[task]
        total_rows = values["train_rows"] + values["val_rows"]
        all_groups = values["train_groups"] | values["val_groups"]
        summary_tasks[task] = {
            "train_rows": values["train_rows"],
            "val_rows": values["val_rows"],
            "val_fast_rows": val_fast_counts[task],
            "train_groups": len(values["train_groups"]),
            "val_groups": len(values["val_groups"]),
            "total_groups": len(all_groups),
            "val_row_fraction": values["val_rows"] / total_rows if total_rows else None,
            "train_label_distribution": dict(sorted(values["train_labels"].items())),
            "val_label_distribution": dict(sorted(values["val_labels"].items())),
            "singleton_group_warning": len(all_groups) < 2,
        }
    summary = {
        "schema_version": 1,
        "seed": seed,
        "val_fraction_target": val_fraction,
        "group_id_mode": group_id_mode,
        "grouping_algorithm": "shared_image_connected_components_v1",
        "manifest": str(manifest_path),
        "train_recipe": str(recipe_dir / train_recipe_name),
        "val_recipe": str(recipe_dir / "locany_cpt_val.json"),
        "val_fast_recipe": str(recipe_dir / "locany_cpt_val_fast.json"),
        "total_groups": len(groups),
        "train_groups": len(groups) - len(validation_groups),
        "val_groups": len(validation_groups),
        "group_intersection": 0,
        "path_duplicate_suspect_count": len(path_duplicate_suspects),
        "duplicate_source_record_id_count": len(duplicate_base_ids),
        "duplicate_source_record_row_count": duplicate_source_record_rows,
        "duplicate_record_id_report": str(duplicate_report_path),
        "tasks": summary_tasks,
    }
    atomic_write_json(diagnostics_dir / "split_summary.json", summary)
    print(
        f"[split] phase=complete state=DONE total_rows={phase_rows:,} "
        f"groups={len(groups):,} total_seconds={time.time() - split_started:.1f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True, help="unsplit normalized CPT recipe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--val-fast-per-task", type=int, default=DEFAULT_VAL_FAST_PER_TASK)
    parser.add_argument("--group-id-mode", choices=("sha256", "path"), default="sha256")
    parser.add_argument("--train-recipe-name", default="locany_cpt_train.json")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = split_recipe(
        args.recipe,
        args.output_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        val_fast_per_task=args.val_fast_per_task,
        group_id_mode=args.group_id_mode,
        train_recipe_name=args.train_recipe_name,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
