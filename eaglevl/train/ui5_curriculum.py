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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CURRICULUM_STATE_VERSION = 1
CURRICULUM_POOLS = ("hard", "matched_anchor", "global_replay")
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


def curriculum_pool_draw_counts(
    worker_states: Mapping[str, Any], dataset_pools: Sequence[str]
) -> dict[str, int]:
    """Count cumulative iterator draws by curriculum pool for one rank.

    ``DeterministicIterator.global_idx`` is the durable per-dataset draw cursor
    persisted in every worker snapshot.  Summing those cursors is therefore a
    resume-stable count of actual sampler draws (including repeated epochs and
    samples subsequently rejected for length), unlike an estimate from target
    ratios.  The trainer all-reduces this rank-local result before reporting it.
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
        for dataset_index, (pool, iterator_state) in enumerate(
            zip(canonical_pools, iterator_states)
        ):
            if not isinstance(iterator_state, Mapping):
                raise ValueError(
                    f"{worker_key}.iterator_states[{dataset_index}] must be a mapping"
                )
            raw_index = iterator_state.get("global_idx")
            if isinstance(raw_index, bool):
                raise ValueError(
                    f"{worker_key}.iterator_states[{dataset_index}].global_idx "
                    "must be a non-negative integer"
                )
            try:
                global_index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{worker_key}.iterator_states[{dataset_index}].global_idx "
                    "must be a non-negative integer"
                ) from exc
            try:
                numeric_index = float(raw_index)
            except (TypeError, ValueError):
                numeric_index = float(global_index)
            if global_index < 0 or numeric_index != global_index:
                raise ValueError(
                    f"{worker_key}.iterator_states[{dataset_index}].global_idx "
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
    expected_recipe_hash = str(success.get("recipe_sha256", ""))
    actual_recipe_hash = _sha256_file(recipe)
    if not expected_recipe_hash or expected_recipe_hash != actual_recipe_hash:
        raise RuntimeError("Curriculum recipe hash does not match _SUCCESS.json")
    declared_files = success.get("files")
    if not isinstance(declared_files, dict) or not declared_files:
        raise RuntimeError("Curriculum _SUCCESS.json has no file hash inventory")
    verified_files = {}
    for name, expected_hash in sorted(declared_files.items()):
        path = recipe.parent / str(name)
        if not path.is_file():
            raise RuntimeError(f"Curriculum inventory file is missing: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != str(expected_hash):
            raise RuntimeError(f"Curriculum artifact hash mismatch: {path}")
        verified_files[str(name)] = actual_hash
    return {
        "identity_digest": str(manifest.get("identity_digest", "")),
        "recipe_sha256": actual_recipe_hash,
        "hard_groups": int(manifest.get("hard_groups", -1)),
        "matched_anchor_groups": int(manifest.get("matched_anchor_groups", -1)),
        "verified_files": verified_files,
    }


def training_continuity_config(
    training_args: Any, schedule: "UI5CurriculumSchedule"
) -> dict[str, Any]:
    """Return optimizer semantics that must not drift between segments."""

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
    return {
        "seed": int(getattr(training_args, "seed", 42)),
        "data_seed": getattr(training_args, "data_seed", None),
        "per_device_train_batch_size": int(
            getattr(training_args, "per_device_train_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            getattr(training_args, "gradient_accumulation_steps", 1)
        ),
        "dataloader_num_workers": int(
            getattr(training_args, "dataloader_num_workers", 0)
        ),
        "dataloader_prefetch_factor": getattr(
            training_args, "dataloader_prefetch_factor", None
        ),
        "dataloader_persistent_workers": bool(
            getattr(training_args, "dataloader_persistent_workers", False)
        ),
        "learning_rate": float(getattr(training_args, "learning_rate")),
        "weight_decay": float(getattr(training_args, "weight_decay", 0.0)),
        "adam_beta1": float(getattr(training_args, "adam_beta1", 0.9)),
        "adam_beta2": float(getattr(training_args, "adam_beta2", 0.999)),
        "adam_epsilon": float(getattr(training_args, "adam_epsilon", 1.0e-8)),
        "max_grad_norm": float(getattr(training_args, "max_grad_norm", 1.0)),
        "optimizer": str(getattr(training_args, "optim", "adamw_torch")),
        "bf16": bool(getattr(training_args, "bf16", False)),
        "fp16": bool(getattr(training_args, "fp16", False)),
        "max_steps": int(getattr(training_args, "max_steps", schedule.total_steps)),
        "schedule_fingerprint": schedule.fingerprint,
        "schedule_total_steps": int(schedule.total_steps),
        "deepspeed": deepspeed_identity,
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
    """Copy worker states and discard old-stage pending samples at a transition.

    Iterator positions and the sampler RNG remain restored exactly.  Only
    already-selected, not-yet-trained packing entries are discarded; otherwise
    stage-0 samples could be used for optimizer step 401 (or stage-1 samples for
    step 801).
    """

    prepared = copy.deepcopy(dict(worker_states))
    report = {"workers": len(prepared), "current_batch_samples": 0, "buffer_samples": 0}
    if int(saved_sampling_stage) == int(resume_sampling_stage):
        return prepared, report
    for state in prepared.values():
        current = list(state.get("current_batch_locations", []))
        buffered = list(state.get("buffer_locations", []))
        report["current_batch_samples"] += len(current)
        report["buffer_samples"] += len(buffered)
        state["current_batch_locations"] = []
        state["buffer_locations"] = []
    return prepared, report
