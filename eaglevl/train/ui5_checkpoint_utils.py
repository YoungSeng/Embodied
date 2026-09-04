"""Shared validation primitives for UI5 evaluation and resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
DEEPSPEED_ZERO_OPTIMIZER_PATTERN = re.compile(
    r"^(?:(?:bf16|fp16)_)?zero_pp_rank_(\d+)_mp_rank_(\d+)_optim_states\.pt$"
)


def atomic_save_with_fsync(save_callable, obj: Any, target: str | Path, *args, **kwargs):
    """Publish a torch-style binary only after a durable non-empty temp write."""

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        result = save_callable(obj, temporary, *args, **kwargs)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size <= 0:
            raise RuntimeError(f"atomic save produced an empty file: {temporary}")
        os.replace(temporary, destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _trainer_state(path: Path) -> dict[str, Any]:
    state_path = path / "trainer_state.json"
    if not _nonempty(state_path):
        raise ValueError(f"Missing or empty trainer_state.json under {path}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid trainer_state.json under {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"trainer_state.json must contain an object under {path}")
    return state


def checkpoint_step(path: Path) -> int:
    """Resolve a step from ``checkpoint-N`` or an arbitrary rolling directory."""

    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is not None:
        return int(match.group(1))
    state = _trainer_state(path)
    try:
        step = int(state["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"trainer_state.json under {path} has no valid global_step"
        ) from exc
    if step < 0:
        raise ValueError(f"trainer_state global_step must be non-negative, got {step}")
    return step


def list_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    if not output_dir.is_dir():
        return []
    result = []
    for path in output_dir.iterdir():
        if path.is_dir() and CHECKPOINT_PATTERN.fullmatch(path.name):
            result.append((checkpoint_step(path), path.resolve()))
    return sorted(result)


def list_training_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Exclude the model-only checkpoint-0 from training-resume candidates."""

    return [(step, path) for step, path in list_checkpoints(output_dir) if step > 0]


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def has_model_weights(checkpoint: Path) -> bool:
    if _nonempty(checkpoint / "model.safetensors") or _nonempty(
        checkpoint / "pytorch_model.bin"
    ):
        return True
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = checkpoint / index_name
        if not _nonempty(index_path):
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(index.get("weight_map", {}).values())
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if not shard_names:
            return False
        return all(_nonempty(checkpoint / name) for name in shard_names)
    return False


def validate_checkpoint(
    checkpoint: Path,
    *,
    mode: str,
    expected_ranks: int | None = None,
    strict: bool = False,
    scaler_required: bool | None = None,
    expected_curriculum_fingerprint: str | None = None,
    require_completion_marker: bool | None = None,
) -> dict[str, Any]:
    if mode not in {"eval", "resume"}:
        raise ValueError(f"Unsupported checkpoint validation mode: {mode}")
    checkpoint = checkpoint.expanduser().resolve()
    if expected_ranks is not None and int(expected_ranks) <= 0:
        raise ValueError("expected_ranks must be positive")
    if require_completion_marker is None:
        require_completion_marker = bool(strict)
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"strict": bool(strict)}
    if not checkpoint.is_dir():
        errors.append("checkpoint directory does not exist")
    else:
        named_match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        try:
            step = checkpoint_step(checkpoint)
            details["global_step"] = step
        except ValueError as exc:
            step = None
            if mode == "resume":
                errors.append(f"cannot resolve resume checkpoint global step: {exc}")
        if not _nonempty(checkpoint / "config.json"):
            errors.append("missing or empty config.json")
        if not has_model_weights(checkpoint):
            errors.append("missing or empty model weights")
        if mode == "resume":
            training_args = checkpoint / "training_args.bin"
            if not _nonempty(training_args):
                errors.append("missing or empty training_args.bin")

            trainer_state = checkpoint / "trainer_state.json"
            state: dict[str, Any] | None = None
            try:
                state = _trainer_state(checkpoint)
                state_step = int(state.get("global_step", -1))
                if named_match is not None and state_step != int(named_match.group(1)):
                    errors.append(
                        "trainer_state global_step does not match checkpoint directory"
                    )
                if strict and state_step <= 0:
                    errors.append("strict resume requires a positive trainer global_step")
            except (ValueError, TypeError) as exc:
                errors.append(f"invalid trainer_state.json: {exc}")

            optimizer_files = [
                path
                for path in checkpoint.glob("global_step*/**/*optim_states.pt")
                if _nonempty(path)
            ]
            optimizer_pt = checkpoint / "optimizer.pt"
            if not _nonempty(optimizer_pt) and not optimizer_files:
                errors.append("missing optimizer/DeepSpeed optimizer state")
            if (
                expected_ranks
                and not _nonempty(optimizer_pt)
                and len(optimizer_files) < expected_ranks
            ):
                errors.append(
                    "DeepSpeed optimizer state count is smaller than expected: "
                    f"found={len(optimizer_files)}, expected={expected_ranks}"
                )
            zero_optimizer_shards: list[tuple[Path, int, int]] = []
            if strict and optimizer_files and not _nonempty(optimizer_pt):
                for optimizer_path in optimizer_files:
                    match = DEEPSPEED_ZERO_OPTIMIZER_PATTERN.fullmatch(
                        optimizer_path.name
                    )
                    if match is not None:
                        zero_optimizer_shards.append(
                            (optimizer_path, int(match.group(1)), int(match.group(2)))
                        )
                if expected_ranks:
                    actual_dp_ranks = {
                        dp_rank for _, dp_rank, _ in zero_optimizer_shards
                    }
                    expected_dp_ranks = set(range(expected_ranks))
                    if len(zero_optimizer_shards) != expected_ranks:
                        errors.append(
                            "DeepSpeed ZeRO optimizer shard count does not equal "
                            f"expected ranks: found={len(zero_optimizer_shards)}, "
                            f"expected={expected_ranks}"
                        )
                    if actual_dp_ranks != expected_dp_ranks:
                        errors.append(
                            "DeepSpeed ZeRO optimizer shard dp ranks are incomplete: "
                            f"found={sorted(actual_dp_ranks)}, "
                            f"expected={sorted(expected_dp_ranks)}"
                        )
                    actual_mp_ranks = {
                        mp_rank for _, _, mp_rank in zero_optimizer_shards
                    }
                    if actual_mp_ranks and actual_mp_ranks != {0}:
                        errors.append(
                            "UI5 H20x2 expects DeepSpeed optimizer mp_rank 00: "
                            f"found={sorted(actual_mp_ranks)}"
                        )

            model_state_files = [
                path
                for path in checkpoint.glob("global_step*/**/*model_states.pt")
                if _nonempty(path)
            ]
            if optimizer_files and not model_state_files:
                errors.append("missing DeepSpeed model state")
            if strict and step is not None and (optimizer_files or model_state_files):
                state_tags = {
                    part
                    for path in (*optimizer_files, *model_state_files)
                    for part in path.relative_to(checkpoint).parts
                    if re.fullmatch(r"global_step\d+", part)
                }
                expected_tag = f"global_step{step}"
                if state_tags != {expected_tag}:
                    errors.append(
                        "DeepSpeed state tag does not match trainer global_step: "
                        f"found={sorted(state_tags)}, expected={expected_tag}"
                    )
                latest_tag = checkpoint / "latest"
                if _nonempty(latest_tag):
                    try:
                        if latest_tag.read_text(encoding="utf-8").strip() != expected_tag:
                            errors.append(
                                "DeepSpeed latest tag does not match trainer global_step"
                            )
                    except OSError as exc:
                        errors.append(f"cannot read DeepSpeed latest tag: {exc}")

            scheduler_pt = checkpoint / "scheduler.pt"
            if strict and not _nonempty(scheduler_pt) and not model_state_files:
                errors.append("missing scheduler/DeepSpeed scheduler state")

            rank_states = [
                path
                for path in checkpoint.glob("dataloader_state_rank*.pt")
                if _nonempty(path)
            ]
            legacy_rank_states = [
                path for path in checkpoint.glob("*.pth") if _nonempty(path)
            ]
            if expected_ranks and max(len(rank_states), len(legacy_rank_states)) < expected_ranks:
                errors.append(
                    "rank state count is smaller than expected: "
                    f"dataloader={len(rank_states)}, legacy={len(legacy_rank_states)}, "
                    f"expected={expected_ranks}"
                )

            rng_files = [
                path for path in checkpoint.glob("rng_state*.pth") if _nonempty(path)
            ]
            if not rng_files:
                message = "missing RNG state; resume would not be bitwise reproducible"
                (errors if strict else warnings).append(message)
            elif strict and expected_ranks and len(rng_files) != expected_ranks:
                errors.append(
                    "RNG state count does not equal expected ranks: "
                    f"found={len(rng_files)}, expected={expected_ranks}"
                )
            if strict and expected_ranks:
                expected_rng_names = (
                    {"rng_state.pth"}
                    if expected_ranks == 1
                    else {
                        f"rng_state_{rank}.pth" for rank in range(expected_ranks)
                    }
                )
                actual_rng_names = {path.name for path in rng_files}
                if actual_rng_names != expected_rng_names:
                    errors.append(
                        "RNG state filenames do not match process ranks: "
                        f"found={sorted(actual_rng_names)}, "
                        f"expected={sorted(expected_rng_names)}"
                    )

            continuity_path = checkpoint / "continuity_state.json"
            continuity: dict[str, Any] | None = None
            if _nonempty(continuity_path):
                try:
                    continuity = json.loads(continuity_path.read_text(encoding="utf-8"))
                    if not isinstance(continuity, dict):
                        raise ValueError("root must be an object")
                    if step is not None and int(continuity.get("global_step", -1)) != step:
                        errors.append(
                            "continuity_state global_step does not match trainer state"
                        )
                    if strict and step is not None:
                        source_step = int(
                            continuity.get("source_global_step", -1)
                        )
                        segment_target_step = int(
                            continuity.get("segment_target_global_step", -1)
                        )
                        if source_step < 0 or source_step >= step:
                            errors.append(
                                "continuity_state source_global_step is not a valid "
                                f"segment start: source={source_step}, checkpoint={step}"
                            )
                        if segment_target_step != step:
                            errors.append(
                                "continuity_state segment_target_global_step does not "
                                f"match trainer state: target={segment_target_step}, "
                                f"checkpoint={step}"
                            )
                    if strict and step is not None and int(
                        continuity.get("target_total_steps", -1)
                    ) < step:
                        errors.append(
                            "continuity_state target_total_steps is behind global_step"
                        )
                    if expected_ranks and int(
                        continuity.get("world_size", -1)
                    ) != int(expected_ranks):
                        errors.append(
                            "continuity_state world_size does not match expected ranks"
                        )
                    if strict:
                        if continuity.get("curriculum_mode") != "scheduled":
                            errors.append(
                                "strict UI5 resume requires curriculum_mode=scheduled"
                            )
                        if int(continuity.get("dataloader_state_version", -1)) < 8:
                            errors.append(
                                "continuity_state requires dataloader state version >= 8"
                            )
                        training_config = continuity.get(
                            "training_continuity_config"
                        )
                        if not isinstance(training_config, dict):
                            errors.append(
                                "continuity_state lacks training_continuity_config"
                            )
                        else:
                            actual_training_digest = hashlib.sha256(
                                json.dumps(
                                    training_config,
                                    ensure_ascii=True,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()
                            if actual_training_digest != continuity.get(
                                "training_continuity_config_digest"
                            ):
                                errors.append(
                                    "continuity_state training config digest is invalid"
                                )
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid continuity_state.json: {exc}")
            elif strict:
                errors.append("missing or empty continuity_state.json")

            if strict:
                for rng_path in rng_files:
                    try:
                        import torch

                        rng_state = torch.load(
                            rng_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                        if not isinstance(rng_state, dict):
                            raise ValueError("root must be a mapping")
                        required_rng_keys = {"python", "numpy", "cpu"}
                        missing_rng_keys = required_rng_keys - set(rng_state)
                        if missing_rng_keys:
                            raise ValueError(
                                f"missing RNG keys {sorted(missing_rng_keys)}"
                            )
                        if (
                            continuity is not None
                            and continuity.get("cuda_rng_required") is True
                            and "cuda" not in rng_state
                        ):
                            raise ValueError("missing CUDA RNG state")
                    except Exception as exc:
                        errors.append(
                            f"invalid {rng_path.name}: {type(exc).__name__}: {exc}"
                        )

            inferred_scaler_required = scaler_required
            if inferred_scaler_required is None and continuity is not None:
                scaler_info = continuity.get("gradient_scaler", {})
                if isinstance(scaler_info, dict):
                    inferred_scaler_required = bool(scaler_info.get("applicable", False))
            if strict and continuity is not None and scaler_required is not None:
                manifest_scaler_required = bool(
                    continuity.get("gradient_scaler", {}).get("applicable", False)
                    if isinstance(continuity.get("gradient_scaler"), dict)
                    else False
                )
                if manifest_scaler_required != bool(scaler_required):
                    errors.append(
                        "gradient scaler applicability changed across resume"
                    )
            scaler_pt = checkpoint / "scaler.pt"
            if (
                strict
                and inferred_scaler_required
                and not _nonempty(scaler_pt)
                and not optimizer_files
            ):
                errors.append("missing gradient scaler state for FP16 resume")

            curriculum_fingerprints: set[str] = set()
            stream_config_digests: set[str] = set()
            if strict:
                if expected_ranks and len(rank_states) != expected_ranks:
                    errors.append(
                        "dataloader state count does not equal expected ranks: "
                        f"found={len(rank_states)}, expected={expected_ranks}"
                    )
                if expected_ranks:
                    expected_rank_state_names = {
                        f"dataloader_state_rank{rank}.pt"
                        for rank in range(expected_ranks)
                    }
                    actual_rank_state_names = {path.name for path in rank_states}
                    if actual_rank_state_names != expected_rank_state_names:
                        errors.append(
                            "dataloader state filenames do not match process ranks: "
                            f"found={sorted(actual_rank_state_names)}, "
                            f"expected={sorted(expected_rank_state_names)}"
                        )
                for rank_state_path in rank_states:
                    try:
                        import torch

                        rank_state = torch.load(
                            rank_state_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                        if not isinstance(rank_state, dict):
                            raise ValueError("root must be a mapping")
                        from eaglevl.train.ui5_curriculum import (
                            CurriculumGroupCycle,
                            DeferredSampleLocations,
                        )
                        if int(rank_state.get("version", -1)) < 8:
                            raise ValueError(
                                f"state version {rank_state.get('version')!r} is older than 8"
                            )
                        worker_states = rank_state.get("worker_states")
                        if not isinstance(worker_states, dict) or not worker_states:
                            raise ValueError("missing worker_states")
                        saved_num_workers = int(rank_state.get("num_workers", -1))
                        if saved_num_workers != len(worker_states):
                            raise ValueError(
                                "num_workers does not match worker_states: "
                                f"num_workers={saved_num_workers}, states={len(worker_states)}"
                            )
                        for worker_key, worker_state in worker_states.items():
                            if not isinstance(worker_state, dict):
                                raise ValueError(f"invalid worker state {worker_key}")
                            if "sample_rng_state" not in worker_state:
                                raise ValueError(
                                    f"worker state {worker_key} lacks sample_rng_state"
                                )
                            if not isinstance(worker_state.get("iterator_states"), list):
                                raise ValueError(
                                    f"worker state {worker_key} lacks iterator_states"
                                )
                            iterator_states = worker_state["iterator_states"]
                            sampler_draws = worker_state.get("dataset_sampler_draws")
                            if (
                                not isinstance(sampler_draws, list)
                                or len(sampler_draws) != len(iterator_states)
                                or any(
                                    isinstance(value, bool)
                                    or not isinstance(value, int)
                                    or value < 0
                                    for value in sampler_draws
                                )
                            ):
                                raise ValueError(
                                    f"worker state {worker_key} has invalid dataset_sampler_draws"
                                )
                            deferred = worker_state.get("deferred_locations")
                            if not isinstance(deferred, list):
                                raise ValueError(
                                    f"worker state {worker_key} lacks deferred_locations"
                                )
                            DeferredSampleLocations(
                                deferred
                                + list(worker_state.get("current_batch_locations", []))
                                + list(worker_state.get("buffer_locations", [])),
                                dataset_count=len(iterator_states),
                                iterator_states=iterator_states,
                            )
                            for dataset_index, iterator_state in enumerate(iterator_states):
                                if not isinstance(iterator_state, dict):
                                    raise ValueError(
                                        f"worker state {worker_key} iterator {dataset_index} "
                                        "is not a mapping"
                                    )
                                group_state = iterator_state.get(
                                    "curriculum_group_cycle"
                                )
                                if not isinstance(group_state, dict):
                                    raise ValueError(
                                        f"worker state {worker_key} iterator {dataset_index} "
                                        "lacks curriculum_group_cycle"
                                    )
                                if (
                                    int(group_state.get("seed", -1))
                                    != int(iterator_state.get("seed", -2))
                                    or int(group_state.get("global_idx", -1))
                                    != int(iterator_state.get("global_idx", -2))
                                ):
                                    raise ValueError(
                                        f"worker state {worker_key} iterator {dataset_index} "
                                        "group cursor does not match iterator cursor"
                                    )
                        stream_config = rank_state.get("stream_resume_config")
                        if not isinstance(stream_config, dict):
                            raise ValueError("missing stream_resume_config")
                        dataset_configs = stream_config.get("datasets")
                        if not isinstance(dataset_configs, list) or not dataset_configs:
                            raise ValueError("stream_resume_config lacks datasets")
                        group_cycles = []
                        for dataset_index, dataset_config in enumerate(dataset_configs):
                            if not isinstance(dataset_config, dict) or dataset_config.get(
                                "sampling_unit"
                            ) != "sample_group":
                                raise ValueError(
                                    f"stream dataset {dataset_index} is not sample-group sampled"
                                )
                            identity = dataset_config.get("curriculum_group_identity")
                            if not isinstance(identity, dict):
                                raise ValueError(
                                    f"stream dataset {dataset_index} lacks group identity"
                                )
                            groups = identity.get("groups")
                            if not isinstance(groups, list) or not groups:
                                raise ValueError(
                                    f"stream dataset {dataset_index} has invalid group identity"
                                )
                            cycle = CurriculumGroupCycle(
                                {
                                    str(group["group_id"]): tuple(group["views"])
                                    for group in groups
                                }
                            )
                            if cycle.identity != identity:
                                raise ValueError(
                                    f"stream dataset {dataset_index} group identity mismatch"
                                )
                            group_cycles.append(cycle)
                        if len(group_cycles) != len(dataset_configs):
                            raise ValueError("stream group cycle inventory is incomplete")
                        for worker_key, worker_state in worker_states.items():
                            iterator_states = worker_state["iterator_states"]
                            if len(iterator_states) != len(group_cycles):
                                raise ValueError(
                                    f"worker state {worker_key} iterator/dataset count mismatch"
                                )
                            for dataset_index, (cycle, iterator_state) in enumerate(
                                zip(group_cycles, iterator_states)
                            ):
                                try:
                                    cycle.validate_iterator_state(
                                        iterator_state["curriculum_group_cycle"],
                                        seed=int(iterator_state["seed"]),
                                        global_idx=int(iterator_state["global_idx"]),
                                    )
                                except BaseException as exc:
                                    raise ValueError(
                                        f"worker state {worker_key} iterator {dataset_index} "
                                        f"has invalid group cursor: {exc}"
                                    ) from exc
                        stream_config_digests.add(
                            hashlib.sha256(
                                json.dumps(
                                    stream_config,
                                    ensure_ascii=True,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()
                        )
                        sampler = rank_state.get("curriculum_sampler")
                        if not isinstance(sampler, dict):
                            raise ValueError("missing curriculum_sampler")
                        if step is not None and int(
                            sampler.get("completed_global_step", -1)
                        ) != step:
                            raise ValueError(
                                "curriculum completed_global_step does not match trainer state"
                            )
                        fingerprint = str(sampler.get("schedule_fingerprint", ""))
                        if not fingerprint:
                            raise ValueError("missing curriculum schedule_fingerprint")
                        schedule_payload = sampler.get("schedule")
                        if not isinstance(schedule_payload, dict):
                            raise ValueError("missing curriculum schedule payload")
                        computed_fingerprint = hashlib.sha256(
                            json.dumps(
                                schedule_payload,
                                ensure_ascii=True,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        if computed_fingerprint != fingerprint:
                            raise ValueError(
                                "curriculum schedule payload fingerprint mismatch"
                            )
                        curriculum_fingerprints.add(fingerprint)
                    except BaseException as exc:
                        errors.append(
                            f"invalid {rank_state_path.name}: {type(exc).__name__}: {exc}"
                        )
                if len(curriculum_fingerprints) > 1:
                    errors.append("curriculum schedule differs between rank states")
                if len(stream_config_digests) > 1:
                    errors.append("stream resume configuration differs between rank states")
                actual_fingerprint = (
                    next(iter(curriculum_fingerprints))
                    if len(curriculum_fingerprints) == 1
                    else None
                )
                manifest_fingerprint = (
                    continuity.get("curriculum_schedule_fingerprint")
                    if continuity is not None
                    else None
                )
                if actual_fingerprint and manifest_fingerprint != actual_fingerprint:
                    errors.append(
                        "continuity_state curriculum fingerprint does not match rank states"
                    )
                actual_stream_digest = (
                    next(iter(stream_config_digests))
                    if len(stream_config_digests) == 1
                    else None
                )
                manifest_stream_digest = (
                    continuity.get("stream_resume_config_digest")
                    if continuity is not None
                    else None
                )
                if actual_stream_digest and manifest_stream_digest != actual_stream_digest:
                    errors.append(
                        "continuity_state stream config digest does not match rank states"
                    )
                if (
                    expected_curriculum_fingerprint is not None
                    and actual_fingerprint != expected_curriculum_fingerprint
                ):
                    errors.append(
                        "checkpoint curriculum schedule does not match the requested schedule"
                    )

            completion = checkpoint / "checkpoint_complete.json"
            if not _nonempty(completion):
                message = "missing checkpoint_complete.json (legacy or interrupted save)"
                (errors if require_completion_marker else warnings).append(message)
            else:
                try:
                    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
                    if step is not None and int(
                        completion_payload.get("global_step", -1)
                    ) != step:
                        errors.append(
                            "checkpoint_complete global_step does not match trainer state"
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid checkpoint_complete.json: {exc}")
            details.update(
                {
                    "optimizer_state_files": len(optimizer_files)
                    + int(_nonempty(optimizer_pt)),
                    "deepspeed_zero_optimizer_dp_ranks": sorted(
                        {dp_rank for _, dp_rank, _ in zero_optimizer_shards}
                    ),
                    "deepspeed_model_state_files": len(model_state_files),
                    "dataloader_state_files": len(rank_states),
                    "legacy_rank_state_files": len(legacy_rank_states),
                    "rng_state_files": len(rng_files),
                    "scheduler_state_files": int(_nonempty(scheduler_pt))
                    + int(bool(model_state_files)),
                    "scaler_state_files": int(_nonempty(scaler_pt))
                    + int(bool(optimizer_files) and bool(inferred_scaler_required)),
                    "curriculum_schedule_fingerprints": sorted(
                        curriculum_fingerprints
                    ),
                    "stream_resume_config_digests": sorted(stream_config_digests),
                    "continuity_manifest": continuity,
                    "training_args_size": (
                        training_args.stat().st_size if training_args.is_file() else 0
                    ),
                }
            )
    return {
        "checkpoint": str(checkpoint),
        "mode": mode,
        "expected_ranks": expected_ranks,
        "strict": bool(strict),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def _remove_tree_or_link(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated_resume_step(
    checkpoint: Path,
    *,
    expected_ranks: int | None,
    strict: bool,
    label: str,
) -> tuple[int, dict[str, Any]]:
    """Validate one recovery participant and return its durable global step."""

    if checkpoint.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {checkpoint}")
    if not checkpoint.is_dir():
        raise RuntimeError(f"{label} is not a checkpoint directory: {checkpoint}")
    report = validate_checkpoint(
        checkpoint,
        mode="resume",
        expected_ranks=expected_ranks,
        strict=strict,
        require_completion_marker=strict,
    )
    if not report["valid"]:
        raise RuntimeError(
            f"invalid {label} checkpoint {checkpoint}: "
            + "; ".join(str(error) for error in report["errors"])
        )
    try:
        step = int(report["details"]["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"validated {label} checkpoint has no global_step: {checkpoint}"
        ) from exc
    return step, report


def recover_atomic_promotion(
    destination: Path,
    *,
    expected_ranks: int | None = None,
    strict: bool = True,
    expected_step_delta: int | None = None,
) -> dict[str, Any]:
    """Recover one interrupted :func:`atomic_promote_checkpoint` transaction.

    Only transaction directories produced by this module are considered:
    ``.<destination>.staging-<uuid4 hex>`` and
    ``.<destination>.backup-<uuid4 hex>`` in the destination's parent.  Every
    participant is validated before mutation, all artifacts must carry the
    same single transaction id, and the newer checkpoint must have the
    requested step relationship to the older one.  States which cannot be
    reached by the promotion rename sequence are rejected rather than guessed
    through.
    """

    destination = destination.expanduser().resolve(strict=False)
    if not destination.name or destination.parent == destination:
        raise ValueError("rolling destination must be a named checkpoint directory")
    if expected_ranks is not None and int(expected_ranks) <= 0:
        raise ValueError("expected_ranks must be positive")
    if expected_step_delta is not None and int(expected_step_delta) <= 0:
        raise ValueError("expected_step_delta must be positive")

    parent = destination.parent
    if not parent.exists():
        return {
            "destination": str(destination),
            "recovered": False,
            "action": "no_transaction",
            "transaction_id": None,
            "global_step": None,
            "validation": None,
        }
    if not parent.is_dir():
        raise RuntimeError(f"rolling checkpoint parent is not a directory: {parent}")

    escaped_name = re.escape(destination.name)
    exact_pattern = re.compile(
        rf"^\.{escaped_name}\.(staging|backup)-([0-9a-f]{{32}})$"
    )
    related_prefixes = (
        f".{destination.name}.staging-",
        f".{destination.name}.backup-",
    )
    artifacts: dict[str, tuple[str, Path]] = {}
    malformed: list[str] = []
    for path in parent.iterdir():
        match = exact_pattern.fullmatch(path.name)
        if match is None:
            if path.name.startswith(related_prefixes):
                malformed.append(path.name)
            continue
        role, transaction_id = match.groups()
        if role in artifacts:
            raise RuntimeError(
                f"multiple rolling {role} artifacts are ambiguous: "
                f"{artifacts[role][1]}, {path}"
            )
        artifacts[role] = (transaction_id, path)
    if malformed:
        raise RuntimeError(
            "malformed rolling transaction artifacts require manual audit: "
            + ", ".join(sorted(malformed))
        )

    transaction_ids = {transaction_id for transaction_id, _ in artifacts.values()}
    if len(transaction_ids) > 1:
        raise RuntimeError(
            "rolling transaction artifacts have multiple transaction ids: "
            + ", ".join(sorted(transaction_ids))
        )

    destination_exists = destination.exists() or destination.is_symlink()
    if not artifacts:
        destination_report = None
        destination_step = None
        if destination_exists:
            destination_step, destination_report = _validated_resume_step(
                destination,
                expected_ranks=expected_ranks,
                strict=strict,
                label="rolling destination",
            )
        return {
            "destination": str(destination),
            "recovered": False,
            "action": "no_transaction",
            "transaction_id": None,
            "global_step": destination_step,
            "validation": destination_report,
        }

    transaction_id = next(iter(transaction_ids))
    staging = artifacts.get("staging", (None, None))[1]
    backup = artifacts.get("backup", (None, None))[1]
    has_staging = staging is not None
    has_backup = backup is not None

    # The three mutation phases produce exactly D+S, S+B, then D+B.  S by
    # itself is the first-ever promotion (there was no previous destination).
    # B alone and D+S+B cannot be produced by the rename sequence.
    state = (destination_exists, has_staging, has_backup)
    actions = {
        (True, True, False): "commit_staged_over_destination",
        (False, True, True): "commit_staged_after_backup",
        (True, False, True): "remove_committed_backup",
        (False, True, False): "commit_initial_staging",
    }
    action = actions.get(state)
    if action is None:
        raise RuntimeError(
            "invalid rolling transaction state; refusing recovery: "
            f"destination={destination_exists}, staging={has_staging}, "
            f"backup={has_backup}, transaction={transaction_id}"
        )

    steps: dict[str, int] = {}
    reports: dict[str, dict[str, Any]] = {}
    if destination_exists:
        steps["destination"], reports["destination"] = _validated_resume_step(
            destination,
            expected_ranks=expected_ranks,
            strict=strict,
            label="rolling destination",
        )
    if staging is not None:
        steps["staging"], reports["staging"] = _validated_resume_step(
            staging,
            expected_ranks=expected_ranks,
            strict=strict,
            label="rolling staging",
        )
    if backup is not None:
        steps["backup"], reports["backup"] = _validated_resume_step(
            backup,
            expected_ranks=expected_ranks,
            strict=strict,
            label="rolling backup",
        )

    if has_staging:
        newer_step = steps["staging"]
        older_step = steps.get("backup", steps.get("destination"))
    else:
        newer_step = steps["destination"]
        older_step = steps["backup"]
    if older_step is None:
        if expected_step_delta is not None and newer_step != int(expected_step_delta):
            raise RuntimeError(
                "initial rolling staging step does not match the expected first "
                f"boundary: found={newer_step}, expected={expected_step_delta}"
            )
    elif expected_step_delta is not None:
        expected_newer_step = older_step + int(expected_step_delta)
        if newer_step != expected_newer_step:
            raise RuntimeError(
                "rolling transaction global steps are not consecutive: "
                f"older={older_step}, newer={newer_step}, "
                f"expected_newer={expected_newer_step}"
            )
    elif newer_step <= older_step:
        raise RuntimeError(
            "rolling transaction staging is not newer than the previous "
            f"checkpoint: older={older_step}, newer={newer_step}"
        )

    if action == "commit_staged_over_destination":
        assert staging is not None and backup is None
        backup = destination.with_name(
            f".{destination.name}.backup-{transaction_id}"
        )
        os.replace(destination, backup)
        _fsync_directory(parent)
        os.replace(staging, destination)
        _fsync_directory(parent)
    elif action in {"commit_staged_after_backup", "commit_initial_staging"}:
        assert staging is not None
        os.replace(staging, destination)
        _fsync_directory(parent)

    final_step, final_report = _validated_resume_step(
        destination,
        expected_ranks=expected_ranks,
        strict=strict,
        label="recovered rolling destination",
    )
    if final_step != newer_step:
        raise RuntimeError(
            "recovered rolling checkpoint changed global_step unexpectedly: "
            f"found={final_step}, expected={newer_step}"
        )
    if backup is not None and (backup.exists() or backup.is_symlink()):
        _remove_tree_or_link(backup)
        _fsync_directory(parent)

    return {
        "destination": str(destination),
        "recovered": True,
        "action": action,
        "transaction_id": transaction_id,
        "global_step": final_step,
        "validation": final_report,
    }


def atomic_promote_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_ranks: int | None = None,
    strict: bool = True,
    move_source: bool = False,
) -> dict[str, Any]:
    """Validate and transactionally replace a rolling checkpoint directory.

    The staging directory lives beside ``destination``, so the final rename is
    on one filesystem.  ``move_source`` avoids a second full checkpoint copy on
    network storage; on failure, the source and previous rolling checkpoint are
    restored before the exception is re-raised.
    """

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve(strict=False)
    if source == destination:
        raise ValueError("source and destination checkpoint paths must differ")
    if not destination.name or destination.parent == destination:
        raise ValueError("rolling destination must be a named checkpoint directory")
    if not source.is_dir():
        raise FileNotFoundError(f"source checkpoint does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if move_source and source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError(
            "--move-source requires source and rolling destination on one filesystem"
        )
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("rolling destination must not be inside the source checkpoint")
    try:
        source.relative_to(destination)
    except ValueError:
        pass
    else:
        raise ValueError("rolling destination must not contain the source checkpoint")

    source_report = validate_checkpoint(
        source,
        mode="resume",
        expected_ranks=expected_ranks,
        strict=strict,
        require_completion_marker=strict,
    )
    if not source_report["valid"]:
        raise RuntimeError(
            "Refusing to promote an invalid checkpoint: "
            + "; ".join(source_report["errors"])
        )
    step = int(source_report["details"]["global_step"])
    transaction = uuid.uuid4().hex
    staging = destination.with_name(f".{destination.name}.staging-{transaction}")
    backup = destination.with_name(f".{destination.name}.backup-{transaction}")
    committed = False
    previous_saved = False
    try:
        if move_source:
            os.replace(source, staging)
            _fsync_directory(source.parent)
            _fsync_directory(destination.parent)
        else:
            shutil.copytree(source, staging, copy_function=shutil.copy2)

        staging_report = validate_checkpoint(
            staging,
            mode="resume",
            expected_ranks=expected_ranks,
            strict=strict,
            require_completion_marker=strict,
        )
        if not staging_report["valid"]:
            raise RuntimeError(
                "Staged rolling checkpoint failed validation: "
                + "; ".join(staging_report["errors"])
            )

        if destination.exists() or destination.is_symlink():
            os.replace(destination, backup)
            previous_saved = True
            _fsync_directory(destination.parent)
        os.replace(staging, destination)
        committed = True
        _fsync_directory(destination.parent)

        destination_report = validate_checkpoint(
            destination,
            mode="resume",
            expected_ranks=expected_ranks,
            strict=strict,
            require_completion_marker=strict,
        )
        if not destination_report["valid"]:
            raise RuntimeError(
                "Promoted rolling checkpoint failed validation: "
                + "; ".join(destination_report["errors"])
            )
        if previous_saved:
            _remove_tree_or_link(backup)
            previous_saved = False
        _fsync_directory(destination.parent)
        return {
            "source": str(source),
            "destination": str(destination),
            "global_step": step,
            "moved_source": bool(move_source),
            "validation": destination_report,
        }
    except BaseException:
        if committed and (destination.exists() or destination.is_symlink()):
            os.replace(destination, staging)
            committed = False
        if previous_saved and (backup.exists() or backup.is_symlink()):
            os.replace(backup, destination)
            previous_saved = False
        if move_source and staging.exists() and not source.exists():
            os.replace(staging, source)
        elif not move_source and (staging.exists() or staging.is_symlink()):
            _remove_tree_or_link(staging)
        _fsync_directory(destination.parent)
        raise
    finally:
        if backup.exists() or backup.is_symlink():
            _remove_tree_or_link(backup)


def safe_remove_checkpoint(path: Path, output_dir: Path) -> None:
    resolved = path.resolve()
    root = output_dir.resolve()
    if resolved.parent != root or CHECKPOINT_PATTERN.fullmatch(resolved.name) is None:
        raise ValueError(
            f"Refusing to remove checkpoint outside output directory: {resolved}"
        )
    shutil.rmtree(resolved)
