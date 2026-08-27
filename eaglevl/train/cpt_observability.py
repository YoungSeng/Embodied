"""CPT-only accounting helpers; JSON/JSONL remains the source of truth."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


CPT_TASKS = (
    "ui_caption",
    "agent_action",
    "agent_grounding",
    "ui_defect",
    "all_ui_elements",
    "single_grounding",
    "ocr",
    "referring_kg",
    "referring",
    "vqa",
)
CPT_TASK_TO_ID = {name: index for index, name in enumerate(CPT_TASKS)}
CPT_ID_TO_TASK = dict(enumerate(CPT_TASKS))


def stable_hash64(value: object) -> int:
    # Keep values in signed int64 range so they can be moved through tensors.
    return int.from_bytes(
        hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def supervision_kinds(
    labels: Sequence[int], pre_mtp_length: int, ignore_index: int = -100
) -> list[int]:
    return [
        0 if int(label) == ignore_index else (1 if index < pre_mtp_length else 2)
        for index, label in enumerate(labels)
    ]


def sample_length_metadata(
    labels: Sequence[int],
    *,
    pre_mtp_length: int,
    vision_tokens: int,
    ignore_index: int = -100,
) -> dict[str, int]:
    kinds = supervision_kinds(labels, pre_mtp_length, ignore_index)
    main = kinds.count(1)
    mtp = kinds.count(2)
    return {
        "raw_text_tokens": int(pre_mtp_length) - int(vision_tokens),
        "vision_tokens": int(vision_tokens),
        "pre_mtp_seq_len": int(pre_mtp_length),
        "post_mtp_seq_len": len(labels),
        "main_supervised_tokens": main,
        "mtp_supervised_tokens": mtp,
        "total_supervised_tokens": main + mtp,
    }


def empty_task_counter() -> dict[str, Any]:
    return {
        "attempted_samples": 0,
        "accepted_samples": 0,
        "oversize_skipped_samples": 0,
        "oversize_pre_mtp_samples": 0,
        "oversize_mtp_expansion_samples": 0,
        "trained_samples": 0,
        "packed_batches": 0,
        "raw_text_tokens": 0,
        "vision_tokens": 0,
        "pre_mtp_tokens": 0,
        "post_mtp_tokens": 0,
        "main_supervised_tokens": 0,
        "mtp_supervised_tokens": 0,
        "total_supervised_tokens": 0,
        "packed_tokens": 0,
        "main_loss_sum": 0.0,
        "main_loss_tokens": 0,
        "mtp_loss_sum": 0.0,
        "mtp_loss_tokens": 0,
        "post_mtp_length_histogram": {},
        "oversize_post_mtp_length_histogram": {},
        "unique_record_hashes": set(),
        "unique_group_hashes": set(),
        "oversize_record_hashes": set(),
        "oversize_group_hashes": set(),
    }


def serializable_counter(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    for key in (
        "unique_record_hashes",
        "unique_group_hashes",
        "oversize_record_hashes",
        "oversize_group_hashes",
    ):
        output[key] = sorted(int(item) for item in value.get(key, ()))
    for key in ("post_mtp_length_histogram", "oversize_post_mtp_length_histogram"):
        output[key] = {str(k): int(v) for k, v in value.get(key, {}).items()}
    return output


def restore_counter(value: Mapping[str, Any]) -> dict[str, Any]:
    output = empty_task_counter()
    output.update(value)
    for key in (
        "unique_record_hashes",
        "unique_group_hashes",
        "oversize_record_hashes",
        "oversize_group_hashes",
    ):
        output[key] = {int(item) for item in value.get(key, ())}
    for key in ("post_mtp_length_histogram", "oversize_post_mtp_length_histogram"):
        output[key] = {int(k): int(v) for k, v in value.get(key, {}).items()}
    return output


def add_sample_to_counter(
    counter: dict[str, Any],
    metadata: Mapping[str, Any],
    *,
    outcome: str,
) -> None:
    if outcome not in {"attempted", "accepted", "oversize", "trained"}:
        raise ValueError(f"unknown CPT sample outcome: {outcome}")
    if outcome == "attempted":
        counter["attempted_samples"] += 1
        return
    main_tokens = int(metadata["main_supervised_tokens"])
    mtp_tokens = int(metadata["mtp_supervised_tokens"])
    total_tokens = int(metadata["total_supervised_tokens"])
    if main_tokens + mtp_tokens != total_tokens:
        raise RuntimeError(
            "CPT supervised-token accounting mismatch: "
            f"main={main_tokens}, mtp={mtp_tokens}, total={total_tokens}"
        )
    if outcome == "oversize":
        counter["oversize_skipped_samples"] += 1
        if int(metadata["pre_mtp_seq_len"]) > int(metadata["post_mtp_seq_len"]):
            raise ValueError("pre-MTP length cannot exceed post-MTP length")
        if bool(metadata.get("pre_mtp_oversize", False)):
            counter["oversize_pre_mtp_samples"] += 1
        else:
            counter["oversize_mtp_expansion_samples"] += 1
        length = int(metadata["post_mtp_seq_len"])
        histogram = counter["oversize_post_mtp_length_histogram"]
        histogram[length] = histogram.get(length, 0) + 1
        counter["oversize_record_hashes"].add(int(metadata["record_hash"]))
        counter["oversize_group_hashes"].add(int(metadata["group_hash"]))
        return
    sample_key = "accepted_samples" if outcome == "accepted" else "trained_samples"
    counter[sample_key] += 1
    if outcome == "accepted":
        length = int(metadata["post_mtp_seq_len"])
        histogram = counter["post_mtp_length_histogram"]
        histogram[length] = histogram.get(length, 0) + 1
    if outcome == "trained":
        for source, destination in (
            ("raw_text_tokens", "raw_text_tokens"),
            ("vision_tokens", "vision_tokens"),
            ("pre_mtp_seq_len", "pre_mtp_tokens"),
            ("post_mtp_seq_len", "post_mtp_tokens"),
            ("main_supervised_tokens", "main_supervised_tokens"),
            ("mtp_supervised_tokens", "mtp_supervised_tokens"),
            ("total_supervised_tokens", "total_supervised_tokens"),
        ):
            counter[destination] += int(metadata[source])
        counter["unique_record_hashes"].add(int(metadata["record_hash"]))
        counter["unique_group_hashes"].add(int(metadata["group_hash"]))


def validate_attempted_identity(counter: Mapping[str, Any]) -> None:
    attempted = int(counter.get("attempted_samples", 0))
    accepted = int(counter.get("accepted_samples", 0))
    skipped = int(counter.get("oversize_skipped_samples", 0))
    if attempted != accepted + skipped:
        raise RuntimeError(
            "CPT sample accounting mismatch: "
            f"attempted={attempted}, accepted={accepted}, oversize_skipped={skipped}"
        )


def histogram_quantile(histogram: Mapping[int | str, int], quantile: float) -> int | None:
    values = sorted((int(key), int(count)) for key, count in histogram.items() if int(count) > 0)
    total = sum(count for _, count in values)
    if not total:
        return None
    target = max(1, math.ceil(total * quantile))
    cumulative = 0
    for value, count in values:
        cumulative += count
        if cumulative >= target:
            return value
    return values[-1][0]


def aggregate_token_losses(
    token_losses,
    labels,
    task_token_ids,
    supervision_kind,
    *,
    ignore_index: int = -100,
) -> dict[int, dict[str, float | int]]:
    """Aggregate shifted per-token CE on device and transfer only tiny sums."""
    import torch

    losses = token_losses.detach().float().reshape(-1)
    shifted_labels = labels[..., 1:].detach().reshape(-1)
    shifted_tasks = task_token_ids[..., 1:].detach().long().reshape(-1)
    shifted_kinds = supervision_kind[..., 1:].detach().long().reshape(-1)
    if not (
        losses.numel()
        == shifted_labels.numel()
        == shifted_tasks.numel()
        == shifted_kinds.numel()
    ):
        raise ValueError(
            "CPT token diagnostic shape mismatch: "
            f"losses={losses.numel()}, labels={shifted_labels.numel()}, "
            f"tasks={shifted_tasks.numel()}, kinds={shifted_kinds.numel()}"
        )
    supervised = shifted_labels.ne(ignore_index)
    kind_is_supervised = shifted_kinds.eq(1) | shifted_kinds.eq(2)
    task_is_valid = shifted_tasks.ge(0) & shifted_tasks.lt(len(CPT_TASKS))
    alignment_errors = torch.stack(
        (
            (supervised & ~kind_is_supervised).any(),
            (~supervised & shifted_kinds.ne(0)).any(),
            (supervised & ~task_is_valid).any(),
            (supervised & ~torch.isfinite(losses)).any(),
        )
    ).cpu().tolist()
    if any(alignment_errors):
        raise RuntimeError(
            "CPT shifted token metadata is invalid: "
            f"supervised_without_kind={alignment_errors[0]}, "
            f"kind_without_label={alignment_errors[1]}, "
            f"supervised_without_task={alignment_errors[2]}, "
            f"nonfinite_token_loss={alignment_errors[3]}"
        )
    valid = supervised & kind_is_supervised & task_is_valid
    slots = shifted_tasks[valid] * 2 + (shifted_kinds[valid] - 1)
    sums = torch.zeros(
        len(CPT_TASKS) * 2,
        dtype=losses.dtype,
        device=losses.device,
    )
    sums.scatter_add_(0, slots, losses[valid])
    counts = torch.bincount(slots, minlength=len(CPT_TASKS) * 2)
    packed = torch.stack((sums, counts.to(sums.dtype)), dim=1).cpu().tolist()
    output: dict[int, dict[str, float | int]] = {}
    for task_id in range(len(CPT_TASKS)):
        main_sum, main_count = packed[task_id * 2]
        mtp_sum, mtp_count = packed[task_id * 2 + 1]
        if main_count or mtp_count:
            output[task_id] = {
                "main_loss_sum": float(main_sum),
                "main_loss_tokens": int(main_count),
                "mtp_loss_sum": float(mtp_sum),
                "mtp_loss_tokens": int(mtp_count),
            }
    return output


def reference_token_loss_aggregation(
    losses: Sequence[float],
    labels: Sequence[int],
    task_ids: Sequence[int],
    kinds: Sequence[int],
    *,
    ignore_index: int = -100,
) -> dict[int, dict[str, float | int]]:
    """Dependency-free reference for unit tests of packed causal shifting."""
    if len(labels) != len(task_ids) or len(labels) != len(kinds):
        raise ValueError("labels/task_ids/kinds must have equal unshifted lengths")
    if len(losses) != max(len(labels) - 1, 0):
        raise ValueError("losses must match causal-shifted label length")
    output: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {
            "main_loss_sum": 0.0,
            "main_loss_tokens": 0,
            "mtp_loss_sum": 0.0,
            "mtp_loss_tokens": 0,
        }
    )
    for loss, label, task_id, kind in zip(
        losses, labels[1:], task_ids[1:], kinds[1:]
    ):
        supervised = int(label) != ignore_index
        if (supervised and int(kind) not in {1, 2}) or (
            not supervised and int(kind) != 0
        ):
            raise RuntimeError("reference CPT label/supervision-kind alignment is invalid")
        if supervised and not 0 <= int(task_id) < len(CPT_TASKS):
            raise RuntimeError("reference CPT supervised token has invalid task id")
        if not supervised:
            continue
        prefix = "main" if int(kind) == 1 else "mtp"
        output[int(task_id)][f"{prefix}_loss_sum"] += float(loss)
        output[int(task_id)][f"{prefix}_loss_tokens"] += 1
    return dict(output)


def merge_task_counters(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    merged = empty_task_counter()
    numeric_keys = [
        key
        for key, value in merged.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    for raw in values:
        counter = restore_counter(raw)
        for key in numeric_keys:
            merged[key] += counter[key]
        for key in (
            "unique_record_hashes",
            "unique_group_hashes",
            "oversize_record_hashes",
            "oversize_group_hashes",
        ):
            merged[key].update(counter[key])
        for key in ("post_mtp_length_histogram", "oversize_post_mtp_length_histogram"):
            for length, count in counter[key].items():
                merged[key][length] = merged[key].get(length, 0) + count
    return merged


def summarize_task_counter(
    task: str,
    counter: Mapping[str, Any],
    *,
    dataset_rows: int,
    dataset_groups: int | None = None,
) -> dict[str, Any]:
    value = restore_counter(counter)
    validate_attempted_identity(value)
    attempted = value["attempted_samples"]
    trained = value["trained_samples"]
    unique_records = len(value["unique_record_hashes"])
    unique_groups = len(value["unique_group_hashes"])
    main_tokens, mtp_tokens = value["main_supervised_tokens"], value["mtp_supervised_tokens"]
    return {
        "task": task,
        **{key: value[key] for key in (
            "attempted_samples", "accepted_samples", "oversize_skipped_samples", "trained_samples",
            "oversize_pre_mtp_samples", "oversize_mtp_expansion_samples",
            "packed_batches", "raw_text_tokens", "vision_tokens", "pre_mtp_tokens", "post_mtp_tokens",
            "main_supervised_tokens", "mtp_supervised_tokens", "total_supervised_tokens", "packed_tokens",
            "main_loss_sum", "main_loss_tokens", "mtp_loss_sum", "mtp_loss_tokens",
        )},
        "oversize_skip_rate": value["oversize_skipped_samples"] / attempted if attempted else 0.0,
        "avg_post_mtp_length": value["post_mtp_tokens"] / trained if trained else None,
        "p50_post_mtp_length": histogram_quantile(value["post_mtp_length_histogram"], 0.50),
        "p95_post_mtp_length": histogram_quantile(value["post_mtp_length_histogram"], 0.95),
        "p99_post_mtp_length": histogram_quantile(value["post_mtp_length_histogram"], 0.99),
        "max_post_mtp_length": histogram_quantile(value["post_mtp_length_histogram"], 1.0),
        "oversize_p95_post_mtp_length": histogram_quantile(value["oversize_post_mtp_length_histogram"], 0.95),
        "oversize_p99_post_mtp_length": histogram_quantile(value["oversize_post_mtp_length_histogram"], 0.99),
        "oversize_max_post_mtp_length": histogram_quantile(value["oversize_post_mtp_length_histogram"], 1.0),
        "unique_record_count": unique_records,
        "unique_group_count": unique_groups,
        "unique_oversize_record_count": len(value["oversize_record_hashes"]),
        "unique_oversize_group_count": len(value["oversize_group_hashes"]),
        "row_coverage": unique_records / dataset_rows if dataset_rows else None,
        "group_coverage": unique_groups / dataset_groups if dataset_groups else None,
        "effective_epoch": trained / dataset_rows if dataset_rows else None,
        "repeat_factor": trained / max(unique_records, 1),
        "avg_tokens_per_sample": (main_tokens + mtp_tokens) / trained if trained else None,
        "train_main_token_ce": value["main_loss_sum"] / value["main_loss_tokens"] if value["main_loss_tokens"] else None,
        "train_mtp_token_ce": value["mtp_loss_sum"] / value["mtp_loss_tokens"] if value["mtp_loss_tokens"] else None,
        "train_total_token_ce": (
            (value["main_loss_sum"] + value["mtp_loss_sum"])
            / (value["main_loss_tokens"] + value["mtp_loss_tokens"])
            if value["main_loss_tokens"] + value["mtp_loss_tokens"]
            else None
        ),
    }
