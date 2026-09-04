"""Deterministic UI5 curriculum and learning-rate schedule primitives.

The formal UI5 run is split into short training processes for evaluation, but
the curriculum is defined on the *optimizer* global-step axis.  This module is
kept independent from Transformers/PyTorch so the schedule and its persisted
state can be validated before a model is loaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CURRICULUM_STATE_VERSION = 1
CURRICULUM_POOLS = ("hard", "matched_anchor", "global_replay")
GROUP_CYCLE_STATE_VERSION = 1
GROUP_CYCLE_POLICY = "uniform_group_epoch_shuffle_group_seeded_cyclic_view_v1"
_POOL_ALIASES = {
    "hard": "hard",
    "hard_matched": "hard",
    "hardmatched": "hard",
    "matched_anchor": "matched_anchor",
    "matchedanchor": "matched_anchor",
    "anchor": "matched_anchor",
    "global_replay": "global_replay",
    "globalreplay": "global_replay",
    "replay": "global_replay",
}


def should_export_model_at_training_end(*, segment_mode: bool) -> bool:
    """Segment checkpoints are the sole model artifact in an eval-interleaved run."""

    return not bool(segment_mode)


def should_write_training_done_marker(*, segment_mode: bool) -> bool:
    """Only a non-segmented training process may declare itself fully done."""

    return not bool(segment_mode)


def canonical_curriculum_pool(value: str) -> str:
    """Return the persisted pool name, accepting a few input-only aliases."""

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _POOL_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unknown curriculum_pool={value!r}; expected one of "
            f"{', '.join(CURRICULUM_POOLS)}"
        )
    return canonical


class CurriculumGroupCycle:
    """Deterministic group-uniform sampling with a cyclic view per group.

    One iterator draw always selects exactly one sample group.  Every group is
    visited once per shuffled group epoch, independent of how many crop views
    it owns.  A group's successive visits advance one position in its own view
    cycle.  ``seed`` and the scalar ``global_idx`` therefore fully determine
    both the next group and the next view, which keeps the state compact and
    exactly reconstructible across DataLoader worker restarts.
    """

    def __init__(self, group_views: Mapping[str, Sequence[str]]) -> None:
        if not isinstance(group_views, Mapping) or not group_views:
            raise ValueError("group_views must be a non-empty mapping")
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_group_id, raw_views in group_views.items():
            group_id = str(raw_group_id).strip()
            if not group_id:
                raise ValueError("curriculum group id must not be empty")
            if not isinstance(raw_views, Sequence) or isinstance(
                raw_views, (str, bytes)
            ):
                raise ValueError(f"group {group_id!r} views must be a sequence")
            views = tuple(str(view).strip() for view in raw_views)
            if not views or any(not view for view in views):
                raise ValueError(f"group {group_id!r} must contain non-empty views")
            if len(set(views)) != len(views):
                raise ValueError(f"group {group_id!r} contains duplicate views")
            normalized[group_id] = views
        self._group_views = {
            group_id: normalized[group_id] for group_id in sorted(normalized)
        }
        self._group_ids = tuple(self._group_views)
        identity = {
            "schema_version": GROUP_CYCLE_STATE_VERSION,
            "policy": GROUP_CYCLE_POLICY,
            "groups": [
                {"group_id": group_id, "views": list(self._group_views[group_id])}
                for group_id in self._group_ids
            ],
        }
        encoded = json.dumps(
            identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._identity = identity
        self._fingerprint = hashlib.sha256(encoded).hexdigest()
        self._order_cache: dict[tuple[int, int], tuple[str, ...]] = {}
        self._offset_cache: dict[tuple[int, str], int] = {}

    @property
    def group_count(self) -> int:
        return len(self._group_ids)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def identity(self) -> dict[str, Any]:
        return copy.deepcopy(self._identity)

    def _group_order(self, *, seed: int, epoch_index: int) -> tuple[str, ...]:
        key = (int(seed), int(epoch_index))
        cached = self._order_cache.get(key)
        if cached is None:
            order = list(self._group_ids)
            random.Random(key[0] + key[1] * 999983).shuffle(order)
            cached = tuple(order)
            self._order_cache[key] = cached
        return cached

    def _initial_view_offset(
        self, *, seed: int, group_id: str, view_count: int
    ) -> int:
        key = (int(seed), str(group_id))
        cached = self._offset_cache.get(key)
        if cached is not None:
            return cached
        encoded = json.dumps(
            [GROUP_CYCLE_POLICY, key[0], key[1]],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % int(
            view_count
        )
        self._offset_cache[key] = offset
        return offset

    def draw_at(self, global_idx: int, *, seed: int) -> dict[str, Any]:
        draw_index = int(global_idx)
        if draw_index < 0 or isinstance(global_idx, bool):
            raise ValueError("group draw global_idx must be a non-negative integer")
        try:
            if float(global_idx) != draw_index:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "group draw global_idx must be a non-negative integer"
            ) from exc
        epoch_index, epoch_position = divmod(draw_index, self.group_count)
        group_id = self._group_order(
            seed=int(seed), epoch_index=epoch_index
        )[epoch_position]
        views = self._group_views[group_id]
        view_index = (
            self._initial_view_offset(
                seed=int(seed), group_id=group_id, view_count=len(views)
            )
            + epoch_index
        ) % len(views)
        return {
            "global_idx": draw_index,
            "epoch_index": epoch_index,
            "epoch_position": epoch_position,
            "group_id": group_id,
            "group_draw_index": epoch_index,
            "view_index": view_index,
            "view_id": views[view_index],
        }

    def iterator_state(self, *, seed: int, global_idx: int) -> dict[str, Any]:
        return {
            "schema_version": GROUP_CYCLE_STATE_VERSION,
            "policy": GROUP_CYCLE_POLICY,
            "fingerprint": self.fingerprint,
            "seed": int(seed),
            "global_idx": int(global_idx),
            "group_count": self.group_count,
            "next_draw": self.draw_at(global_idx, seed=int(seed)),
        }

    def validate_iterator_state(
        self, state: Mapping[str, Any], *, seed: int, global_idx: int
    ) -> None:
        expected = self.iterator_state(seed=int(seed), global_idx=int(global_idx))
        if dict(state) != expected:
            raise RuntimeError(
                "curriculum group iterator state changed across resume: "
                f"saved={dict(state)}, expected={expected}"
            )


class DeferredSampleLocations:
    """Per-dataset FIFO queues for sampled-but-not-yet-trained locations."""

    def __init__(
        self,
        locations: Sequence[Sequence[int]],
        *,
        dataset_count: int,
        iterator_states: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        count = int(dataset_count)
        if count <= 0:
            raise ValueError("dataset_count must be positive")
        if not isinstance(locations, Sequence) or isinstance(locations, (str, bytes)):
            raise ValueError("deferred sample locations must be a sequence")
        if iterator_states is not None and len(iterator_states) != count:
            raise ValueError("iterator state count does not match dataset_count")
        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for position, raw in enumerate(locations):
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
                raise ValueError(f"invalid deferred sample location at index {position}")
            values: list[int] = []
            for value in raw:
                if isinstance(value, bool):
                    raise ValueError(
                        f"invalid deferred sample location at index {position}"
                    )
                try:
                    integer = int(value)
                    if integer < 0 or float(value) != integer:
                        raise ValueError
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid deferred sample location at index {position}"
                    ) from exc
                values.append(integer)
            location = (values[0], values[1])
            if location[0] >= count:
                raise ValueError(
                    f"deferred sample dataset index is out of range: {location}"
                )
            if location in seen:
                raise ValueError(f"duplicate deferred sample location: {location}")
            if iterator_states is not None:
                cursor = int(iterator_states[location[0]].get("global_idx", -1))
                if location[1] >= cursor:
                    raise ValueError(
                        "deferred sample location is not behind its iterator cursor: "
                        f"location={location}, cursor={cursor}"
                    )
            seen.add(location)
            normalized.append(location)
        self._queues = [deque() for _ in range(count)]
        for dataset_index, global_index in sorted(normalized):
            self._queues[dataset_index].append((dataset_index, global_index))

    def __len__(self) -> int:
        return sum(len(queue) for queue in self._queues)

    def pop_for_dataset(self, dataset_index: int) -> tuple[int, int] | None:
        index = int(dataset_index)
        if index < 0 or index >= len(self._queues):
            raise IndexError(f"dataset index out of range: {dataset_index}")
        return self._queues[index].popleft() if self._queues[index] else None

    def to_list(self) -> list[tuple[int, int]]:
        return [location for queue in self._queues for location in queue]

    def counts(self) -> list[int]:
        return [len(queue) for queue in self._queues]


def curriculum_pool_draw_counts(
    worker_states: Mapping[str, Any], dataset_pools: Sequence[str]
) -> dict[str, int]:
    """Count cumulative sampler selections by curriculum pool for one rank.

    New checkpoints persist ``dataset_sampler_draws`` separately from iterator
    cursors.  That distinction matters when a phase-boundary packing backlog is
    returned: choosing a pool consumes one sampler draw but reuses an already
    materialized iterator location.  Version-7 checkpoints have no backlog and
    safely fall back to their per-dataset iterator cursors.
    """

    if not isinstance(worker_states, Mapping):
        raise ValueError("worker_states must be a mapping")
    canonical_pools = tuple(canonical_curriculum_pool(pool) for pool in dataset_pools)
    if not canonical_pools:
        raise ValueError("dataset_pools must not be empty")
    counts = {pool: 0 for pool in CURRICULUM_POOLS}
    for worker_key, raw_worker in worker_states.items():
        if not isinstance(raw_worker, Mapping):
            raise ValueError(f"{worker_key}.worker_state must be a mapping")
        iterator_states = raw_worker.get("iterator_states")
        if not isinstance(iterator_states, Sequence) or isinstance(
            iterator_states, (str, bytes)
        ):
            raise ValueError(f"{worker_key}.iterator_states must be a sequence")
        if len(iterator_states) != len(canonical_pools):
            raise ValueError(
                f"{worker_key}.iterator_states length {len(iterator_states)} does not "
                f"match dataset_pools length {len(canonical_pools)}"
            )
        sampler_draws = raw_worker.get("dataset_sampler_draws")
        if sampler_draws is None:
            raw_counts = []
            for dataset_index, iterator_state in enumerate(iterator_states):
                if not isinstance(iterator_state, Mapping):
                    raise ValueError(
                        f"{worker_key}.iterator_states[{dataset_index}] must be a mapping"
                    )
                raw_counts.append(iterator_state.get("global_idx"))
        else:
            if not isinstance(sampler_draws, Sequence) or isinstance(
                sampler_draws, (str, bytes)
            ):
                raise ValueError(f"{worker_key}.dataset_sampler_draws must be a sequence")
            if len(sampler_draws) != len(canonical_pools):
                raise ValueError(
                    f"{worker_key}.dataset_sampler_draws length {len(sampler_draws)} "
                    f"does not match dataset_pools length {len(canonical_pools)}"
                )
            raw_counts = list(sampler_draws)
        for dataset_index, (pool, raw_index) in enumerate(
            zip(canonical_pools, raw_counts)
        ):
            if isinstance(raw_index, bool):
                raise ValueError(
                    f"{worker_key}.dataset_sampler_draws[{dataset_index}] "
                    "must be a non-negative integer"
                )
            try:
                global_index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{worker_key}.dataset_sampler_draws[{dataset_index}] "
                    "must be a non-negative integer"
                ) from exc
            try:
                numeric_index = float(raw_index)
            except (TypeError, ValueError):
                numeric_index = float(global_index)
            if global_index < 0 or numeric_index != global_index:
                raise ValueError(
                    f"{worker_key}.dataset_sampler_draws[{dataset_index}] "
                    "must be a non-negative integer"
                )
            counts[pool] += global_index
    return counts


def _parse_csv(name: str, raw: str) -> tuple[float, ...]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"{name} contains an empty value: {raw!r}")
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-number: {item!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CONTINUITY_IMPLEMENTATION_FILES = (
    "eaglevl/train/arguments.py",
    "eaglevl/train/locany_finetune_magi_stream.py",
    "eaglevl/train/ui5_checkpoint_utils.py",
    "eaglevl/train/ui5_curriculum.py",
    "eaglevl/model/locany/modeling_locateanything.py",
    "eaglevl/model/locany/relation_modules.py",
    "eaglevl/model/locany/ui_relation_setup.py",
)


def _continuity_value(value: Any) -> Any:
    """Convert an argument value to a stable JSON scalar/container."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {
            str(key): _continuity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_continuity_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _continuity_value(enum_value)
    return str(value)


def _continuity_fields(arguments: Any, names: Sequence[str]) -> dict[str, Any]:
    return {
        name: _continuity_value(getattr(arguments, name, None))
        for name in names
    }


def _continuity_implementation_sha256() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[2]
    identities: dict[str, str] = {}
    for relative in _CONTINUITY_IMPLEMENTATION_FILES:
        path = source_root / relative
        if not path.is_file():
            raise RuntimeError(
                f"Training continuity implementation file is missing: {path}"
            )
        identities[relative] = _sha256_file(path)
    return identities


def curriculum_artifact_identity(
    recipe_path: str | Path, schedule: "UI5CurriculumSchedule"
) -> dict[str, Any]:
    """Validate the curriculum builder's durable manifest and file hashes."""

    recipe = Path(recipe_path).expanduser().resolve()
    manifest_path = recipe.parent / "curriculum_manifest.json"
    success_path = recipe.parent / "_SUCCESS.json"
    for path in (recipe, manifest_path, success_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Missing curriculum artifact: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid curriculum artifact manifest: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(success, dict):
        raise RuntimeError("Curriculum manifests must contain JSON objects")
    if success.get("complete") is not True:
        raise RuntimeError("Curriculum _SUCCESS.json is not marked complete")
    if success.get("identity_digest") != manifest.get("identity_digest"):
        raise RuntimeError("Curriculum identity digest differs between manifests")
    manifest_without_identity = dict(manifest)
    declared_identity = str(manifest_without_identity.pop("identity_digest", ""))
    computed_identity = hashlib.sha256(
        json.dumps(
            manifest_without_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not declared_identity or computed_identity != declared_identity:
        raise RuntimeError("Curriculum manifest identity digest is invalid")
    expected = schedule.expected_hard_groups
    if expected is not None:
        actual = int(manifest.get("hard_groups", -1))
        declared = int(manifest.get("expected_hard_groups", -1))
        if actual != expected or declared != expected:
            raise RuntimeError(
                "Curriculum hard-group count mismatch: "
                f"actual={actual}, declared={declared}, expected={expected}"
            )
    artifact_schema = int(manifest.get("schema_version", 1))
    if artifact_schema >= 3:
        if int(success.get("schema_version", -1)) != artifact_schema:
            raise RuntimeError(
                "Curriculum schema version differs between manifest and success marker"
            )
        if int(manifest.get("matched_anchor_groups", -1)) != int(
            manifest.get("hard_groups", -2)
        ):
            raise RuntimeError(
                "Curriculum matched-anchor count differs from hard-group count"
            )
        expected_view_policy = {
            "hard": "all_gt_free_detector_scan_base_tiles",
            "matched_anchor": "all_gt_free_detector_scan_base_tiles",
            "content_missing": "full_image_global_view",
            "global_replay": (
                "all_gt_free_detector_scan_base_tiles_except_content_missing_full_image"
                if artifact_schema >= 4
                else "full_image_retention"
            ),
            "tile_selection_uses_gt": False,
            "partial_gt_allowed": False,
        }
        if manifest.get("training_view_policy") != expected_view_policy:
            raise RuntimeError("Curriculum training-view policy is invalid")
        pools = manifest.get("pools")
        if not isinstance(pools, Mapping) or set(pools) != set(CURRICULUM_POOLS):
            raise RuntimeError("Curriculum pool inventory is invalid")
        for pool in ("hard", "matched_anchor"):
            state = pools.get(pool)
            if not isinstance(state, Mapping):
                raise RuntimeError(f"Curriculum {pool} inventory is invalid")
            crop_records = int(state.get("crop_training_records", -1))
            global_records = int(state.get("content_missing_global_records", -1))
            training_records = int(state.get("training_records", -1))
            if crop_records <= 0 or global_records < 0 or training_records != (
                crop_records + global_records
            ):
                raise RuntimeError(f"Curriculum {pool} training views are invalid")
        replay = pools.get("global_replay")
        if artifact_schema >= 4:
            if not isinstance(replay, Mapping) or any(
                (
                    int(replay.get("crop_training_records", 0)) <= 0,
                    int(replay.get("content_missing_global_records", -1)) != 0,
                    int(replay.get("retention_full_image_records", 0)) <= 0,
                    int(replay.get("training_records", -1))
                    != int(replay.get("crop_training_records", -2))
                    + int(replay.get("retention_full_image_records", -3)),
                )
            ):
                raise RuntimeError(
                    "Curriculum global-replay crop/full-image views are invalid"
                )
        elif not isinstance(replay, Mapping) or any(
            (
                int(replay.get("crop_training_records", -1)) != 0,
                int(replay.get("content_missing_global_records", -1)) != 0,
                int(replay.get("retention_full_image_records", -1))
                != int(replay.get("training_records", -2)),
                int(replay.get("training_records", 0)) <= 0,
            )
        ):
            raise RuntimeError("Curriculum global-replay retention views are invalid")
        try:
            recipe_payload = json.loads(recipe.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Curriculum recipe JSON is invalid: {exc}") from exc
        expected_recipe_views = {
            "hard": (True, False),
            "matched_anchor": (True, False),
            "global_replay": (
                (True, True) if artifact_schema >= 4 else (False, True)
            ),
        }
        observed_recipe_views = {}
        if not isinstance(recipe_payload, Mapping):
            raise RuntimeError("Curriculum recipe is not an object")
        for entry in recipe_payload.values():
            if not isinstance(entry, Mapping):
                raise RuntimeError("Curriculum recipe entry is not an object")
            pool = canonical_curriculum_pool(str(entry.get("curriculum_pool") or ""))
            if pool in observed_recipe_views:
                raise RuntimeError(f"Curriculum recipe duplicates pool {pool}")
            observed_recipe_views[pool] = (
                entry.get("ui5_crop_recipe") is True,
                entry.get("ui5_retention_recipe") is True,
            )
        if observed_recipe_views != expected_recipe_views:
            raise RuntimeError("Curriculum recipe training-view flags are invalid")
    expected_recipe_hash = str(success.get("recipe_sha256", ""))
    actual_recipe_hash = _sha256_file(recipe)
    if not expected_recipe_hash or expected_recipe_hash != actual_recipe_hash:
        raise RuntimeError("Curriculum recipe hash does not match _SUCCESS.json")
    declared_files = success.get("files")
    if not isinstance(declared_files, dict) or not declared_files:
        raise RuntimeError("Curriculum _SUCCESS.json has no file hash inventory")
    crop_assets: list[Any] | None = None
    if artifact_schema >= 3:
        raw_crop_assets = manifest.get("crop_assets")
        if not isinstance(raw_crop_assets, list) or not raw_crop_assets:
            raise RuntimeError("Curriculum crop-asset inventory is empty")
        crop_assets = raw_crop_assets
        asset_names = {
            str(row.get("relative_path") or "")
            for row in crop_assets
            if isinstance(row, Mapping)
        }
        expected_file_names = {
            "ui5_crop_rollout4_curriculum.json",
            "hard.jsonl",
            "matched_anchor.jsonl",
            "global_replay.jsonl",
            "hard_groups.jsonl",
            "matched_anchor_groups.jsonl",
            "crop_assets.jsonl",
            *asset_names,
        }
        if (
            len(asset_names) != len(crop_assets)
            or "" in asset_names
            or set(declared_files) != expected_file_names
        ):
            raise RuntimeError("Curriculum durable file inventory is invalid")
    verified_files = {}
    for name, raw_metadata in sorted(declared_files.items()):
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Curriculum inventory path is unsafe: {name}")
        path = recipe.parent / relative
        if not path.is_file():
            raise RuntimeError(f"Curriculum inventory file is missing: {path}")
        if artifact_schema >= 3:
            if not isinstance(raw_metadata, Mapping):
                raise RuntimeError(
                    f"Curriculum inventory metadata is invalid: {path}"
                )
            expected_hash = str(raw_metadata.get("sha256") or "")
            expected_bytes = raw_metadata.get("bytes")
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
                or path.stat().st_size != expected_bytes
            ):
                raise RuntimeError(
                    f"Curriculum artifact byte count mismatch: {path}"
                )
        else:
            expected_hash = str(raw_metadata)
        actual_hash = _sha256_file(path)
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimeError(f"Curriculum artifact hash mismatch: {path}")
        verified_files[str(name)] = actual_hash
    if artifact_schema >= 3:
        assert crop_assets is not None
        crop_pools = CURRICULUM_POOLS if artifact_schema >= 4 else (
            "hard",
            "matched_anchor",
        )
        expected_crop_records = sum(
            int(pools[pool]["crop_training_records"]) for pool in crop_pools
        )
        if len(crop_assets) != expected_crop_records:
            raise RuntimeError(
                "Curriculum crop-asset count differs from crop training records"
            )
        seen_assets = set()
        for row in crop_assets:
            if not isinstance(row, Mapping):
                raise RuntimeError("Curriculum crop-asset row is invalid")
            relative = str(row.get("relative_path") or "")
            if not relative or relative in seen_assets or relative not in declared_files:
                raise RuntimeError("Curriculum crop-asset path inventory is invalid")
            seen_assets.add(relative)
            metadata = declared_files[relative]
            if not isinstance(metadata, Mapping) or any(
                (
                    metadata.get("bytes") != row.get("bytes"),
                    metadata.get("sha256") != row.get("sha256"),
                )
            ):
                raise RuntimeError(
                    f"Curriculum crop-asset manifest differs from success inventory: {relative}"
                )
    return {
        "identity_digest": str(manifest.get("identity_digest", "")),
        "recipe_sha256": actual_recipe_hash,
        "hard_groups": int(manifest.get("hard_groups", -1)),
        "matched_anchor_groups": int(manifest.get("matched_anchor_groups", -1)),
        "verified_files": verified_files,
    }


def training_continuity_config(
    training_args: Any,
    schedule: "UI5CurriculumSchedule",
    *,
    model_args: Any = None,
    data_args: Any = None,
) -> dict[str, Any]:
    """Return all training semantics that must not drift between segments."""

    deepspeed = getattr(training_args, "deepspeed", None)
    if isinstance(deepspeed, (str, Path)) and Path(deepspeed).is_file():
        deepspeed_identity: Any = {
            "path": str(Path(deepspeed).expanduser().resolve()),
            "sha256": _sha256_file(Path(deepspeed).expanduser().resolve()),
        }
    elif isinstance(deepspeed, Mapping):
        deepspeed_identity = {
            "config_digest": hashlib.sha256(
                json.dumps(
                    deepspeed,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        }
    else:
        deepspeed_identity = str(deepspeed) if deepspeed else None
    training_semantics = _continuity_fields(
        training_args,
        (
            "seed",
            "data_seed",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "dataloader_num_workers",
            "dataloader_prefetch_factor",
            "dataloader_persistent_workers",
            "dataloader_drop_last",
            "dataloader_pin_memory",
            "learning_rate",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "max_grad_norm",
            "optim",
            "lr_scheduler_type",
            "warmup_ratio",
            "warmup_steps",
            "gradient_checkpointing",
            "bf16",
            "fp16",
            "tf32",
            "max_steps",
            "lr_scale",
        ),
    )
    # Preserve historical defaults for lightweight callers while binding every
    # concrete value supplied by Transformers in the formal trainer.
    training_semantics.update(
        {
            "seed": int(getattr(training_args, "seed", 42)),
            "per_device_train_batch_size": int(
                getattr(training_args, "per_device_train_batch_size", 1)
            ),
            "gradient_accumulation_steps": int(
                getattr(training_args, "gradient_accumulation_steps", 1)
            ),
            "dataloader_num_workers": int(
                getattr(training_args, "dataloader_num_workers", 0)
            ),
            "learning_rate": float(getattr(training_args, "learning_rate")),
            "weight_decay": float(getattr(training_args, "weight_decay", 0.0)),
            "adam_beta1": float(getattr(training_args, "adam_beta1", 0.9)),
            "adam_beta2": float(getattr(training_args, "adam_beta2", 0.999)),
            "adam_epsilon": float(
                getattr(training_args, "adam_epsilon", 1.0e-8)
            ),
            "max_grad_norm": float(getattr(training_args, "max_grad_norm", 1.0)),
            "optimizer": str(getattr(training_args, "optim", "adamw_torch")),
            "bf16": bool(getattr(training_args, "bf16", False)),
            "fp16": bool(getattr(training_args, "fp16", False)),
            "max_steps": int(
                getattr(training_args, "max_steps", schedule.total_steps)
            ),
        }
    )
    training_semantics.pop("optim", None)
    return {
        "training": training_semantics,
        "model_semantics": _continuity_fields(
            model_args,
            (
                "model_name_or_path",
                "vision_path",
                "llm_path",
                "mlp_path",
                "processor_config_path",
                "preprocessor_config_path",
                "chat_template_path",
                "freeze_llm",
                "freeze_backbone",
                "freeze_mlp",
                "unfreeze_vit_layers",
                "vision_select_layer",
                "use_backbone_lora",
                "use_llm_lora",
                "unfreeze_lm_head",
                "grad_checkpoint",
                "freeze_backbones",
                "lr_scale",
                "use_fp8",
                "mlp_connector_layers",
                "block_size",
                "causal_attn",
                "attn_implementation",
                "expected_mask_repeat_times",
                "enable_ui_relation",
                "relation_detail_hidden_size",
                "relation_num_slots",
                "relation_adapter_bottleneck",
                "relation_detail_layers",
                "relation_gate_loss_weight",
                "relation_slot_gate_loss_weight",
                "relation_attention_loss_weight",
                "relation_gate_threshold",
                "relation_focal_beta",
                "relation_focal_gamma",
            ),
        ),
        "data_semantics": _continuity_fields(
            data_args,
            (
                "max_seq_length",
                "meta_path",
                "neftune_alpha",
                "n_frames",
                "sequence_parallel_degree",
                "ring_sequence_parallel_degree",
                "sample_length_div",
                "use_online_packing",
                "video_total_pixels",
                "max_frames",
                "target_fps",
                "max_num_tokens_per_sample",
                "max_num_tokens",
                "packing_buffer_size",
                "auto_thinking_handler",
                "balance_ui_defects",
                "ui_records_per_class",
                "ui_negative_to_positive_ratio",
                "ui_sampling_mode",
            ),
        ),
        "schedule_fingerprint": schedule.fingerprint,
        "schedule_total_steps": int(schedule.total_steps),
        "deepspeed": deepspeed_identity,
        "implementation_sha256": _continuity_implementation_sha256(),
    }


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    first_optimizer_step: int
    last_optimizer_step: int
    pool_weights: tuple[float, float, float]
    llm_lr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "first_optimizer_step": self.first_optimizer_step,
            "last_optimizer_step": self.last_optimizer_step,
            "pool_weights": {
                pool: self.pool_weights[index]
                for index, pool in enumerate(CURRICULUM_POOLS)
            },
            "llm_lr": self.llm_lr,
        }


class UI5CurriculumSchedule:
    """Validated equal-width curriculum stages over optimizer global steps."""

    def __init__(
        self,
        *,
        total_steps: int,
        hard_ratios: Sequence[float],
        matched_anchor_ratios: Sequence[float],
        global_replay_ratios: Sequence[float],
        llm_lrs: Sequence[float],
        expected_hard_groups: int | None = None,
    ) -> None:
        self.total_steps = int(total_steps)
        if self.total_steps <= 0:
            raise ValueError("TOTAL_STEPS must be positive")

        columns = tuple(
            tuple(float(value) for value in values)
            for values in (
                hard_ratios,
                matched_anchor_ratios,
                global_replay_ratios,
                llm_lrs,
            )
        )
        stage_count = len(columns[0])
        if stage_count <= 0 or any(len(values) != stage_count for values in columns):
            raise ValueError(
                "HARD_RATIOS, ANCHOR_RATIOS, GLOBAL_REPLAY_RATIOS, and "
                "LLM_LRS must have the same non-zero length"
            )
        if self.total_steps % stage_count:
            raise ValueError(
                "TOTAL_STEPS must be divisible by the number of curriculum stages "
                f"for unambiguous boundaries: total={self.total_steps}, stages={stage_count}"
            )
        for name, values in zip(
            ("HARD_RATIOS", "ANCHOR_RATIOS", "GLOBAL_REPLAY_RATIOS"),
            columns[:3],
        ):
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(f"{name} values must be in [0, 1]")
        if any(value <= 0.0 or not math.isfinite(value) for value in columns[3]):
            raise ValueError("LLM_LRS values must be finite and positive")
        for index, ratios in enumerate(zip(*columns[:3])):
            if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"Curriculum ratios at stage {index} must sum to 1; got {sum(ratios):.12g}"
                )

        self.expected_hard_groups = (
            None if expected_hard_groups is None else int(expected_hard_groups)
        )
        if self.expected_hard_groups is not None and self.expected_hard_groups <= 0:
            raise ValueError("EXPECTED_HARD_GROUPS must be positive when provided")

        width = self.total_steps // stage_count
        self.stages = tuple(
            CurriculumStage(
                index=index,
                first_optimizer_step=index * width + 1,
                last_optimizer_step=(index + 1) * width,
                pool_weights=(columns[0][index], columns[1][index], columns[2][index]),
                llm_lr=columns[3][index],
            )
            for index in range(stage_count)
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        *,
        default_total_steps: int,
    ) -> "UI5CurriculumSchedule | None":
        mode = str(environment.get("CURRICULUM_MODE", "none")).strip().lower()
        if mode in {"", "none", "off", "disabled"}:
            return None
        if mode != "scheduled":
            raise ValueError("CURRICULUM_MODE must be 'scheduled' or 'none'")
        total_steps = int(environment.get("TOTAL_STEPS", default_total_steps))
        expected_text = str(environment.get("EXPECTED_HARD_GROUPS", "")).strip()
        return cls(
            total_steps=total_steps,
            hard_ratios=_parse_csv(
                "HARD_RATIOS", environment.get("HARD_RATIOS", "0.60,0.45,0.30")
            ),
            matched_anchor_ratios=_parse_csv(
                "ANCHOR_RATIOS", environment.get("ANCHOR_RATIOS", "0.25,0.35,0.30")
            ),
            global_replay_ratios=_parse_csv(
                "GLOBAL_REPLAY_RATIOS",
                environment.get("GLOBAL_REPLAY_RATIOS", "0.15,0.20,0.40"),
            ),
            llm_lrs=_parse_csv(
                "LLM_LRS", environment.get("LLM_LRS", "1e-6,7e-7,5e-7")
            ),
            expected_hard_groups=(int(expected_text) if expected_text else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "scheduled",
            "optimizer_step_semantics": "one_based_inclusive",
            "total_steps": self.total_steps,
            "expected_hard_groups": self.expected_hard_groups,
            "pool_names": list(CURRICULUM_POOLS),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def stage_for_optimizer_step(self, optimizer_step: int) -> CurriculumStage:
        step = int(optimizer_step)
        if step < 1 or step > self.total_steps:
            raise ValueError(
                f"optimizer global step must be in [1, {self.total_steps}], got {step}"
            )
        width = self.stages[0].last_optimizer_step
        return self.stages[min((step - 1) // width, len(self.stages) - 1)]

    def stage_after_completed_step(self, completed_global_step: int) -> CurriculumStage:
        completed = int(completed_global_step)
        if completed < 0 or completed > self.total_steps:
            raise ValueError(
                f"completed global step must be in [0, {self.total_steps}], got {completed}"
            )
        next_step = min(completed + 1, self.total_steps)
        return self.stage_for_optimizer_step(next_step)

    def sampling_stage_at_checkpoint(self, completed_global_step: int) -> CurriculumStage:
        """Stage whose samples produced the final completed update in a checkpoint."""

        completed = int(completed_global_step)
        if completed == 0:
            return self.stages[0]
        return self.stage_for_optimizer_step(completed)

    def validate_segment(self, completed_global_step: int, target_global_step: int) -> None:
        """Reject a process that would cross a stage boundary with prefetched data."""

        completed = int(completed_global_step)
        target = int(target_global_step)
        if target <= completed:
            raise ValueError(
                f"segment target must exceed completed step: completed={completed}, target={target}"
            )
        if target > self.total_steps:
            raise ValueError(
                f"segment target exceeds TOTAL_STEPS={self.total_steps}: target={target}"
            )
        first = self.stage_after_completed_step(completed)
        last = self.stage_for_optimizer_step(target)
        if first.index != last.index:
            raise ValueError(
                "A scheduled-curriculum training process may not cross a curriculum "
                f"boundary: completed={completed}, target={target}, "
                f"first_stage={first.index}, last_stage={last.index}. "
                "End the process at the boundary and resume from its checkpoint."
            )

    def effective_dataset_weights(
        self,
        *,
        dataset_pools: Sequence[str],
        base_weights: Sequence[float],
        completed_global_step: int,
    ) -> list[float]:
        if len(dataset_pools) != len(base_weights) or not dataset_pools:
            raise ValueError("dataset_pools and base_weights must have the same non-zero length")
        canonical_pools = [canonical_curriculum_pool(pool) for pool in dataset_pools]
        weights = [float(weight) for weight in base_weights]
        if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
            raise ValueError("dataset base weights must be finite and non-negative")
        stage = self.stage_after_completed_step(completed_global_step)
        pool_totals = {
            pool: sum(
                weight
                for candidate, weight in zip(canonical_pools, weights)
                if candidate == pool
            )
            for pool in CURRICULUM_POOLS
        }
        for pool, target_weight in zip(CURRICULUM_POOLS, stage.pool_weights):
            if target_weight > 0.0 and pool_totals[pool] <= 0.0:
                raise ValueError(
                    f"Curriculum stage {stage.index} assigns weight to empty pool {pool!r}"
                )
        effective = [
            stage.pool_weights[CURRICULUM_POOLS.index(pool)]
            * weight
            / pool_totals[pool]
            if pool_totals[pool] > 0.0
            else 0.0
            for pool, weight in zip(canonical_pools, weights)
        ]
        if not math.isclose(sum(effective), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise RuntimeError(f"effective curriculum weights do not sum to 1: {effective}")
        return effective

    def lr_for_completed_steps(self, completed_steps: int) -> float:
        """LR for the next optimizer update after ``completed_steps`` updates."""

        return self.stage_after_completed_step(completed_steps).llm_lr

    def lr_multiplier_for_completed_steps(
        self, completed_steps: int, *, reference_lr: float
    ) -> float:
        reference = float(reference_lr)
        if reference <= 0.0 or not math.isfinite(reference):
            raise ValueError("reference_lr must be finite and positive")
        return self.lr_for_completed_steps(completed_steps) / reference

    def sampler_state(
        self, *, completed_global_step: int, sampling_stage_index: int
    ) -> dict[str, Any]:
        completed = int(completed_global_step)
        sampling_stage = self.sampling_stage_at_checkpoint(completed)
        if int(sampling_stage_index) != sampling_stage.index:
            raise RuntimeError(
                "Sampler stage does not match the optimizer step being checkpointed: "
                f"sampler={sampling_stage_index}, expected={sampling_stage.index}, "
                f"global_step={completed}"
            )
        next_stage = self.stage_after_completed_step(completed)
        return {
            "schema_version": CURRICULUM_STATE_VERSION,
            "schedule": self.to_dict(),
            "schedule_fingerprint": self.fingerprint,
            "completed_global_step": completed,
            "sampling_stage_index": sampling_stage.index,
            "sampling_pool_weights": sampling_stage.to_dict()["pool_weights"],
            "next_optimizer_step": min(completed + 1, self.total_steps),
            "next_stage_index": next_stage.index,
            "next_pool_weights": next_stage.to_dict()["pool_weights"],
        }

    def validate_sampler_state(
        self, state: Mapping[str, Any], *, expected_global_step: int
    ) -> None:
        if int(state.get("schema_version", -1)) != CURRICULUM_STATE_VERSION:
            raise RuntimeError(
                "Unsupported curriculum sampler state version: "
                f"{state.get('schema_version')!r}"
            )
        if state.get("schedule_fingerprint") != self.fingerprint:
            raise RuntimeError(
                "Curriculum schedule changed across resume: "
                f"saved={state.get('schedule_fingerprint')}, current={self.fingerprint}"
            )
        if state.get("schedule") != self.to_dict():
            raise RuntimeError("Curriculum schedule payload changed across resume")
        completed = int(state.get("completed_global_step", -1))
        if completed != int(expected_global_step):
            raise RuntimeError(
                "Curriculum global step does not match trainer_state.json: "
                f"curriculum={completed}, trainer={expected_global_step}"
            )
        expected_sampling_stage = self.sampling_stage_at_checkpoint(completed).index
        if int(state.get("sampling_stage_index", -1)) != expected_sampling_stage:
            raise RuntimeError(
                "Persisted curriculum sampling stage is inconsistent with global step: "
                f"saved={state.get('sampling_stage_index')}, "
                f"expected={expected_sampling_stage}"
            )


def prepare_worker_states_for_resume(
    worker_states: Mapping[str, Mapping[str, Any]],
    *,
    saved_sampling_stage: int,
    resume_sampling_stage: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Return old-stage pending samples without changing durable cursors.

    Pending packing entries cannot stay directly in the new-stage buffer because
    they were selected with the previous pool ratios.  They also cannot simply
    be cleared while retaining advanced iterator cursors: that permanently loses
    sampled-but-untrained records.  Instead, transition them into per-dataset
    deferred FIFO queues.  The new-stage pool sampler must select that dataset
    before its oldest deferred location is replayed, so phase ratios and record
    identity are both preserved.
    """

    prepared = copy.deepcopy(dict(worker_states))
    report = {
        "workers": len(prepared),
        "current_batch_samples": 0,
        "buffer_samples": 0,
        "already_deferred_samples": 0,
        "deferred_samples": 0,
    }
    if int(saved_sampling_stage) == int(resume_sampling_stage):
        return prepared, report
    for worker_key, state in prepared.items():
        if not isinstance(state, dict):
            raise ValueError(f"{worker_key}.worker_state must be a mapping")
        iterator_states = state.get("iterator_states")
        if not isinstance(iterator_states, Sequence) or isinstance(
            iterator_states, (str, bytes)
        ) or not iterator_states:
            raise ValueError(f"{worker_key}.iterator_states must be a non-empty sequence")
        current = list(state.get("current_batch_locations", []))
        buffered = list(state.get("buffer_locations", []))
        already_deferred = list(state.get("deferred_locations", []))
        report["current_batch_samples"] += len(current)
        report["buffer_samples"] += len(buffered)
        report["already_deferred_samples"] += len(already_deferred)
        queue = DeferredSampleLocations(
            already_deferred + current + buffered,
            dataset_count=len(iterator_states),
            iterator_states=iterator_states,
        )
        state["current_batch_locations"] = []
        state["buffer_locations"] = []
        state["deferred_locations"] = queue.to_list()
        report["deferred_samples"] += len(queue)
    return prepared, report
