#!/usr/bin/env python3
"""Create Excel-ready curriculum diagnostics from one completed eval node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaglevl.train.ui5_curriculum_profiles import curriculum_phases


TASKS = (
    "occlusion",
    "cropping",
    "text_overflow",
    "text_ellipsis",
    "content_missing",
)
TASK_DIRS = {task: f"ui_{task}" for task in TASKS}
RATIOS = (
    (0.60, 0.25, 0.15, 1.0e-6),
    (0.45, 0.35, 0.20, 7.0e-7),
    (0.30, 0.30, 0.40, 5.0e-7),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--curriculum-dir", type=Path, required=True)
    parser.add_argument("--trainer-state", type=Path)
    parser.add_argument("--total-steps", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not an object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _phase_for_step(step: int, total_steps: int) -> tuple[int, tuple[float, ...]]:
    RATIOS = curriculum_phases()
    if total_steps <= 0 or total_steps % len(RATIOS):
        raise ValueError("total steps must be positive and divisible by three")
    if step < 0 or step > total_steps:
        raise ValueError(f"step must be in [0,{total_steps}], got {step}")
    width = total_steps // len(RATIOS)
    optimizer_step = max(1, step)
    index = min((optimizer_step - 1) // width, len(RATIOS) - 1)
    return index, RATIOS[index]


def train_curve_rows(
    *, step: int, total_steps: int, trainer_state: Path | None
) -> list[dict[str, Any]]:
    history: list[Mapping[str, Any]] = []
    if trainer_state is not None:
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
        if int(state.get("global_step", -1)) != step:
            raise ValueError(
                f"trainer_state global_step does not match eval step: "
                f"{state.get('global_step')} != {step}"
            )
        raw_history = state.get("log_history") or []
        if not isinstance(raw_history, list):
            raise ValueError("trainer_state.log_history must be a list")
        history = [row for row in raw_history if isinstance(row, Mapping)]

    by_step: dict[int, dict[str, Any]] = {}
    for raw in history:
        if raw.get("step") is None:
            continue
        row_step = int(raw["step"])
        if row_step < 0 or row_step > step:
            continue
        if not any(
            key in raw
            for key in ("loss", "train_loss", "learning_rate", "lm_loss", "loss_lm")
        ):
            continue
        phase, ratios = _phase_for_step(row_step, total_steps)
        by_step[row_step] = {
            **dict(raw),
            "step": row_step,
            "phase": phase + 1,
            "learning_rate": raw.get("learning_rate", ratios[3]),
            "loss_total": raw.get("loss", raw.get("train_loss")),
            "loss_lm": raw.get("lm_loss", raw.get("loss_lm")),
            "hard_ratio": ratios[0],
            "anchor_ratio": ratios[1],
            "global_replay_ratio": ratios[2],
            "hard_samples": raw.get("curriculum_hard_samples"),
            "anchor_samples": raw.get("curriculum_anchor_samples"),
            "global_replay_samples": raw.get("curriculum_global_replay_samples"),
        }
        counts = [raw.get(key) for key in ("curriculum_hard_samples", "curriculum_anchor_samples",
                                           "curriculum_global_replay_samples")]
        total = sum(counts) if all(isinstance(n, (float, int)) for n in counts) else 0
        for name, count in zip(("hard", "anchor", "global_replay"), counts):
            by_step[row_step][f"actual_{name}_ratio_cumulative"] = count / total if total else None
    if step == 0 and 0 not in by_step:
        _, ratios = _phase_for_step(0, total_steps)
        by_step[0] = {
            "step": 0,
            "phase": "baseline",
            "learning_rate": ratios[3],
            "loss_total": None,
            "loss_lm": None,
            "hard_ratio": ratios[0],
            "anchor_ratio": ratios[1],
            "global_replay_ratio": ratios[2],
            "hard_samples": None,
            "anchor_samples": None,
            "global_replay_samples": None,
        }
    previous_counts, previous_step = [0, 0, 0], 0
    for key in sorted(by_step):
        row = by_step[key]
        current_counts = [row.get(name + "_samples") for name in ("hard", "anchor", "global_replay")]
        if all(isinstance(n, (int, float)) for n in current_counts):
            deltas = [now - old for now, old in zip(current_counts, previous_counts)]
            denominator = sum(deltas) if min(deltas) >= 0 else 0
            row["sampling_window_start_step"] = previous_step
            for name, delta in zip(("hard", "anchor", "global_replay"), deltas):
                row[f"actual_{name}_ratio_window"] = delta / denominator if denominator else None
            previous_counts, previous_step = current_counts, key
    return [by_step[key] for key in sorted(by_step)]


def _correct(row: Mapping[str, Any]) -> bool:
    value = row.get("exact_correct", row.get("reward"))
    if not isinstance(value, bool):
        raise ValueError(
            f"hard rollout {row.get('record_id')} lacks boolean exact_correct/reward"
        )
    if row.get("runtime_error"):
        raise RuntimeError(f"hard rollout has runtime error: {row.get('record_id')}")
    # A parse error is a completed model prediction and is scored as incorrect;
    # only execution/runtime failures make the diagnostic inputs incomplete.
    if row.get("parse_status") == "parse_error":
        return False
    return value


def hard_transition_rows(*, step: int, evaluation_dir: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for task in TASKS:
        per_group: dict[str, int] = defaultdict(int)
        expected_ids: set[str] | None = None
        for rollout_id in range(4):
            path = (
                evaluation_dir
                / TASK_DIRS[task]
                / "rollout4"
                / f"rollout_{rollout_id}.jsonl"
            )
            if not path.is_file():
                raise FileNotFoundError(f"hard rollout output is missing: {path}")
            rows = _read_jsonl(path)
            ids = {
                str(row.get("record_id") or row.get("sample_id") or "") for row in rows
            }
            if "" in ids or len(ids) != len(rows):
                raise ValueError(f"hard rollout ids are empty/duplicate: {path}")
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError(f"hard rollout group set changed across seeds: {task}")
            correct = sum(_correct(row) for row in rows)
            for row in rows:
                per_group[str(row.get("record_id") or row.get("sample_id"))] += int(
                    _correct(row)
                )
            result.append(
                {
                    "step": step,
                    "task": f"ui_{task}",
                    "group": "frozen_0_of_4",
                    "rollout_id": rollout_id,
                    "samples": len(rows),
                    "correct": correct,
                    "accuracy": correct / len(rows) if rows else 0.0,
                    "transition": "baseline_0_of_4_to_candidate_rollout",
                }
            )
        distribution = Counter(per_group.values())
        for correct_count in range(5):
            samples = distribution[correct_count]
            result.append(
                {
                    "step": step,
                    "task": f"ui_{task}",
                    "group": "frozen_0_of_4",
                    "rollout_id": -1,
                    "samples": samples,
                    "correct": samples * correct_count,
                    "accuracy": correct_count / 4.0,
                    "transition": f"0/4 -> {correct_count}/4",
                }
            )
    return result


def anchor_data_coverage_rows(*, step: int, curriculum_dir: Path) -> list[dict[str, Any]]:
    groups_path = curriculum_dir / "matched_anchor_groups.jsonl"
    records_path = curriculum_dir / "matched_anchor.jsonl"
    groups = _read_jsonl(groups_path)
    records = _read_jsonl(records_path)

    def row_task(row: Mapping[str, Any], *, label: str) -> str:
        candidates: set[str] = set()
        for key in ("_ui5_task", "task"):
            raw = str(row.get(key) or "").strip()
            if not raw:
                continue
            task = raw[3:] if raw.startswith("ui_") else raw
            if task not in TASKS:
                raise ValueError(f"unknown {label} task: {raw!r}")
            candidates.add(task)
        if not candidates:
            raise ValueError(f"{label} lacks a UI5 task")
        if len(candidates) != 1:
            raise ValueError(
                f"conflicting {label} task fields: {sorted(candidates)}"
            )
        return next(iter(candidates))

    def row_sample_id(row: Mapping[str, Any], *, label: str) -> str:
        value = (
            row.get("_ui5_sample_id")
            or row.get("sample_id")
            or row.get("record_id")
        )
        sample_id = str(value or "").strip()
        if not sample_id:
            raise ValueError(f"{label} lacks a stable sample_id")
        return sample_id

    expected: dict[str, set[str]] = defaultdict(set)
    present: dict[str, set[str]] = defaultdict(set)
    expected_owner: dict[str, str] = {}
    for index, row in enumerate(groups):
        label = f"matched_anchor_groups[{index}]"
        task = row_task(row, label=label)
        sample_id = row_sample_id(row, label=label)
        if sample_id in expected_owner:
            raise ValueError(
                "duplicate matched-anchor sample_id: "
                f"{sample_id} (tasks {expected_owner[sample_id]} and {task})"
            )
        expected_owner[sample_id] = task
        expected[task].add(sample_id)

    record_owner: dict[str, str] = {}
    for index, row in enumerate(records):
        label = f"matched_anchor[{index}]"
        task = row_task(row, label=label)
        sample_id = row_sample_id(row, label=label)
        previous_task = record_owner.get(sample_id)
        if previous_task is not None and previous_task != task:
            raise ValueError(
                "matched-anchor record sample_id belongs to multiple tasks: "
                f"{sample_id} ({previous_task} and {task})"
            )
        expected_task = expected_owner.get(sample_id)
        if expected_task is not None and expected_task != task:
            raise ValueError(
                "matched-anchor record task conflicts with its expected group: "
                f"{sample_id} (expected {expected_task}, found {task})"
            )
        record_owner[sample_id] = task
        present[task].add(sample_id)

    result = []
    for task in TASKS:
        total = len(expected[task])
        retained = len(expected[task] & present[task])
        score = retained / total if total else None
        result.append(
            {
                "step": step,
                "task": f"ui_{task}",
                "scope": "train manifest coverage; NOT inference retention",
                "pool": "matched_anchor", "expected_groups": total,
                "covered_groups": retained, "data_coverage": score,
            }
        )
    return result


def anchor_retention_rows(*, step: int, evaluation_dir: Path) -> list[dict[str, Any]]:
    baseline = evaluation_dir if step == 0 else evaluation_dir.parent / "step-000000"
    result = []
    for task in TASKS:
        suffix = Path(f"ui_{task}") / "anchor_inference" / "predictions.jsonl"
        current_rows, baseline_rows = _read_jsonl(evaluation_dir / suffix), _read_jsonl(baseline / suffix)
        current = {r["record_id"]: r for r in current_rows}
        previous = {r["record_id"]: r for r in baseline_rows}
        if len(current) != len(current_rows) or len(previous) != len(baseline_rows) or set(current) != set(previous):
            raise ValueError("anchor retention requires exactly the same fixed group IDs as step 0")
        if any(r["sample_seed"] != previous[key]["sample_seed"] for key, r in current.items()):
            raise ValueError("anchor seeds differ from this run step 0")
        baseline_correct = sum(_correct(r) for r in previous.values())
        correct = sum(_correct(r) for r in current.values())
        kept = sum(_correct(r) and _correct(previous[key]) for key, r in current.items())
        total = len(current)
        row = {"step": step, "task": f"ui_{task}", "scope": "train/mining anchor inference",
               "samples": total, "baseline_correct": baseline_correct, "current_correct": correct,
               "baseline_score": baseline_correct / total if total else None,
               "current_score": correct / total if total else None,
               "delta": (correct - baseline_correct) / total if total else None,
               "baseline_correct_retained": kept,
               "retention_rate": kept / baseline_correct if baseline_correct else None,
               "image_invalid": sum(r["parse_status"] == "parse_error" for r in current.values()),
               "inference_seconds": sum(r["latency_seconds"] for r in current.values())}
        for prefix, items in (("baseline", previous.values()), ("current", current.values())):
            tp, fp, fn = [sum(r[key] for r in items) for key in ("TP_box", "FP_box", "FN_box")]
            row.update({f"{prefix}_bbox_tp": tp, f"{prefix}_bbox_fp": fp, f"{prefix}_bbox_fn": fn,
                        f"{prefix}_bbox_precision": tp / (tp + fp) if tp + fp else 0.0,
                        f"{prefix}_bbox_recall": tp / (tp + fn) if tp + fn else 0.0,
                        f"{prefix}_bbox_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0})
            if any(r.get("image_confusion") not in {"TP", "FP", "FN", "TN"} for r in items):
                raise ValueError("anchor result lacks actual image confusion classification")
            tp, fp, fn, tn = [sum(r["image_confusion"] == key for r in items) for key in ("TP", "FP", "FN", "TN")]
            row.update({f"{prefix}_image_tp": tp, f"{prefix}_image_fp": fp,
                        f"{prefix}_image_fn": fn, f"{prefix}_image_tn": tn,
                        f"{prefix}_image_precision": tp / (tp + fp) if tp + fp else 0.,
                        f"{prefix}_image_recall": tp / (tp + fn) if tp + fn else 0.,
                        f"{prefix}_image_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.})
        result.append(row)
    return result


def run(args: argparse.Namespace) -> dict[str, str]:
    if args.step < 0:
        raise ValueError("--step must be non-negative")
    evaluation_dir = args.evaluation_dir.expanduser().resolve(strict=True)
    curriculum_dir = args.curriculum_dir.expanduser().resolve(strict=True)
    trainer_state = (
        args.trainer_state.expanduser().resolve(strict=True)
        if args.trainer_state is not None
        else None
    )
    if args.step > 0 and trainer_state is None:
        raise ValueError("nonzero evaluation steps require --trainer-state")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation_dir / "diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train_curve": {
            "train_curve": train_curve_rows(
                step=args.step,
                total_steps=args.total_steps,
                trainer_state=trainer_state,
            )
        },
        "hard_transition": {
            "hard_transition": hard_transition_rows(
                step=args.step, evaluation_dir=evaluation_dir
            )
        },
        "anchor_retention": {
            "anchor_retention": (anchor_retention_rows(step=args.step, evaluation_dir=evaluation_dir)
                                 if os.environ.get("UI5_CURRICULUM_PROFILE") == "global_replay_v3" else [])
        },
    }
    extra = {"data_coverage": anchor_data_coverage_rows(step=args.step, curriculum_dir=curriculum_dir)}
    for row in payloads["hard_transition"]["hard_transition"]:
        row["scope"] = "train/mining frozen 0/4, NOT held-out UI5"
        row["baseline_source"] = "frozen mining rollouts; not the new decoder step-0"
    if os.environ.get("UI5_CURRICULUM_PROFILE") == "global_replay_v3":
        from scripts.ui5_output_validity import audit_evaluation
        audit = audit_evaluation(evaluation_dir)
        _atomic_json(output_dir / "raw_output_audit.json", audit)
        extra["output_validity"] = [{"step": args.step, **r} for r in audit["rows"]]
        identity = json.loads((evaluation_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
        extra["provenance"] = [{"step": args.step, "code_sha": os.environ.get("CODE_REVISION"),
                                 "curriculum_profile": identity["curriculum_profile"],
                                 "evaluation_identity_path": str(evaluation_dir / "evaluation_manifest.json"),
                                 "evaluation_identity_sha256": hashlib.sha256(
                                     (evaluation_dir / "evaluation_manifest.json").read_bytes()).hexdigest(),
                                 "generation": identity["generation"],
                                 "frozen_selection_sha256": identity["frozen_selection"]["summary_sha256"],
                                 "loss_missing_policy": "null, never synthetic zero"}]
        run_root = evaluation_dir.parent.parent
        preparation = json.loads((run_root / "diagnostics/v3_preparation.json").read_text(encoding="utf-8"))
        extra["supervision_format"] = [{"step": args.step, **row} for row in preparation["supervision_format"]["pools"]]
        extra["training_text_identity"] = [{"step": args.step, **preparation["training_text"]}]
        mtp_reports = sorted((run_root / "diagnostics").glob("negative_mtp_audit_rank*.json"))
        extra["negative_mtp_supervision"] = []
        for path in mtp_reports:
            mtp = json.loads(path.read_text(encoding="utf-8"))
            if (mtp["recipe_sha256"] != preparation["training_text"]["recipe_sha256"]
                    or mtp["recipe_path"] != preparation["training_text"]["recipe_path"]
                    or {r["pool"] for r in mtp["pools"]} != {"hard", "matched_anchor", "global_replay"}
                    or any(r.get("valid") is not True for r in mtp["pools"])):
                raise ValueError("real MTP audit differs from the new training text or has invalid labels")
            extra["negative_mtp_supervision"].extend({"step": args.step, "report": str(path), **row} for row in mtp["pools"])
        if args.step > 0 and {p.name for p in mtp_reports} != {"negative_mtp_audit_rank0.json", "negative_mtp_audit_rank1.json"}:
            raise ValueError("v3 training requires both H20 ranks' real processor/MTP negative supervision audits")
        extra["data_coverage"].extend({"step": args.step, **row} for row in preparation["pool_coverage"])
        comparison = json.loads((run_root / "diagnostics/decoder_comparison.json").read_text(encoding="utf-8"))
        if comparison.get("complete") is not True:
            raise ValueError("formal decoder comparison is not complete")
        extra["decoder_comparison"] = [
            {"step": args.step, "model": cell["model"], "mode": cell["mode"],
             "decoder_policy": cell["decoder_policy"],
             "scope": "fixed held-out diagnostic subset; NOT full UI5 or training data",
             **cell["metrics"]["overall"],
             "image_invalid": sum(r.get("image_invalid", 0) for r in cell["raw_audit"]["rows"]),
             "output": cell["output"]} for cell in comparison["cells"]]
        extra["inference_fix_gain"] = [{"step": args.step, **row} for row in comparison["inference_fix_gain"]]
        current_metrics = json.loads((evaluation_dir / "ui5_metrics.json").read_text())["overall"]
        base_dir = evaluation_dir.parent / "step-000000"
        base_metrics = json.loads((base_dir / "ui5_metrics.json").read_text())["overall"]
        extra["training_vs_baseline"] = [
            {"step": args.step, "metric": key, "baseline_step": 0,
             "baseline_score": base_metrics[key], "current_score": current_metrics[key],
             "training_delta": current_metrics[key] - base_metrics[key],
             "decoder_policy": "boundary_v3",
             "interpretation": "training delta versus new FULL step-0; decoder gain reported separately on paired subset"}
            for key in ("image_macro_f1", "bbox_macro_f1", "joint_score")]
    payloads["extra_diagnostics"] = extra
    outputs: dict[str, str] = {}
    for name, payload in payloads.items():
        path = output_dir / f"{name}.json"
        _atomic_json(path, payload)
        outputs[name] = str(path)
    _atomic_json(output_dir / "diagnostics_complete.json", outputs)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[curriculum-diagnostics:error] {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
