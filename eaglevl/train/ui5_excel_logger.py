"""Atomic two-sheet Excel diagnostics for LocateAnything UI5 training/eval."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRAIN_TASKS = (
    "text_overflow",
    "text_ellipsis",
    "element_overlap",
    "element_cropping",
    "content_missing",
)

TRAIN_BASE_COLUMNS = (
    "step",
    "segment_epoch",
    "global_epoch",
    "gpu_num",
    "max_num_tokens",
    "learning_rate",
    "gate_loss_weight",
    "slot_gate_loss_weight",
    "attention_loss_weight",
    "gate_threshold",
    "focal_beta",
    "focal_gamma",
    "relation_num_slots",
    "loss_total",
    "loss_total_min",
    "loss_total_max",
    "loss_lm",
    "loss_gate",
    "loss_image_gate",
    "loss_slot_gate",
    "loss_attention",
    "weighted_gate_loss",
    "weighted_slot_gate_loss",
    "weighted_attention_loss",
    "loss_lm_contribution",
    "loss_image_gate_contribution",
    "loss_slot_gate_contribution",
    "loss_attention_contribution",
    "loss_reconstructed",
    "loss_reconstruction_error",
    "attention_active_batch_rate",
    "grad_norm",
    "grad_norm_max",
    "samples",
    "positive_samples",
    "negative_samples",
    "peak_gpu_memory_mb",
)
TRAIN_TASK_SUFFIXES = (
    "samples",
    "positive",
    "negative",
    "gate_loss",
    "attention_loss",
    "p_defect_pos",
    "p_defect_neg",
    "gate_precision",
    "gate_recall",
    "gate_f1",
    "gate_pr_auc",
    "slot_gate_loss",
    "slot_positive",
    "slot_negative",
    "detail_weight_l5",
    "detail_weight_l15",
    "detail_weight_l26",
)
TRAIN_MODULE_COLUMNS = (
    "detail_layer5_norm",
    "detail_layer15_norm",
    "detail_layer26_norm",
    "detail_layer5_abs_max",
    "detail_layer15_abs_max",
    "detail_layer26_abs_max",
    "detail_layer5_saturation_fraction",
    "detail_layer15_saturation_fraction",
    "detail_layer26_saturation_fraction",
    "detail_norm_ratio",
    "detail_fused_norm",
    "relation_context_norm",
    "relation_gate_prob_mean",
    "pbd_delta_norm",
    "pbd_active_positions",
    "relation_grad_norm",
    # Kept as the backwards-compatible aggregate of image/slot Gate gradients.
    # Older workbooks contain this column, while newer runs also expose the two
    # components below.  Retaining it lets name-based schema migration preserve
    # completed evaluations instead of rejecting the legacy header.
    "gate_grad_norm",
    "image_gate_grad_norm",
    "slot_gate_grad_norm",
    "pbd_grad_norm",
    "relation_grad_seen_steps",
    "image_gate_grad_seen_steps",
    "slot_gate_grad_seen_steps",
    "pbd_grad_seen_steps",
    "relation_absolute_update_norm",
    "relation_relative_update_norm",
    "relation_changed_element_count",
    "image_gate_absolute_update_norm",
    "image_gate_relative_update_norm",
    "image_gate_changed_element_count",
    "slot_gate_absolute_update_norm",
    "slot_gate_relative_update_norm",
    "slot_gate_changed_element_count",
    "pbd_absolute_update_norm",
    "pbd_relative_update_norm",
    "pbd_changed_element_count",
)
TRAIN_COLUMNS = (
    *TRAIN_BASE_COLUMNS,
    *(
        f"{task}_{suffix}"
        for task in TRAIN_TASKS
        for suffix in TRAIN_TASK_SUFFIXES
    ),
    *TRAIN_MODULE_COLUMNS,
)

EVAL_COLUMNS = (
    "step",
    "checkpoint",
    "evaluation_split",
    "cache_scope",
    "development_test_reuse",
    "git_sha",
    "git_dirty",
    "recipe_digest",
    "cache_digest",
    "task",
    "granularity",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "tn",
    "accuracy",
    "predicted_positive",
    "gate_positive",
    "gate_filtered",
    "p_defect_pos",
    "p_defect_neg",
    "parse_error",
    "f1_change_from_previous",
    "best_f1_so_far",
    "best_step_so_far",
    "raw_precision",
    "raw_recall",
    "raw_f1",
    "raw_predicted_positive",
    "selected_gate_threshold",
    "gated_precision",
    "gated_recall",
    "gated_f1",
    "gated_predicted_positive",
    "gate_filter_rate",
    "bbox_metrics_genuinely_rescored",
    "gate_metric_status",
)

SHEET_TRAIN = "train_100steps"
SHEET_EVAL = "eval_1000steps"
EXPECTED_SHEETS = (SHEET_TRAIN, SHEET_EVAL)


def _openpyxl():
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError as exc:
        raise RuntimeError(
            "UI5 Excel diagnostics require openpyxl>=3.1 in the training environment"
        ) from exc
    return openpyxl, Alignment, Font, PatternFill


def _value(value: Any) -> Any:
    """Convert tensor/numpy scalar-like values without importing those packages."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
            if isinstance(scalar, float) and not math.isfinite(scalar):
                return None
            return scalar
        except (ValueError, RuntimeError):
            pass
    return value


def _flatten_train_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: _value(value) for key, value in metrics.items() if key != "tasks"}
    tasks = metrics.get("tasks", {})
    if isinstance(tasks, Mapping):
        for task, values in tasks.items():
            if task not in TRAIN_TASKS or not isinstance(values, Mapping):
                continue
            for suffix, value in values.items():
                row[f"{task}_{suffix}"] = _value(value)
    return row


@dataclass
class UI5ExcelLogger:
    """Append-only, resume-safe writer for the requested two diagnostic sheets."""

    path: Path | str

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _new_workbook(self):
        openpyxl, Alignment, Font, PatternFill = _openpyxl()
        workbook = openpyxl.Workbook()
        train = workbook.active
        train.title = SHEET_TRAIN
        evaluation = workbook.create_sheet(SHEET_EVAL)
        train.append(list(TRAIN_COLUMNS))
        evaluation.append(list(EVAL_COLUMNS))
        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(color="FFFFFF", bold=True)
        for sheet in (train, evaluation):
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.row_dimensions[1].height = 28
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for index, heading in enumerate(sheet[1], start=1):
                sheet.column_dimensions[heading.column_letter].width = min(
                    max(12, len(str(heading.value)) + 2), 30
                )
        return workbook

    def _load_or_create(self):
        openpyxl, _, _, _ = _openpyxl()
        if not self.path.is_file():
            return self._new_workbook()
        workbook = openpyxl.load_workbook(self.path)
        if tuple(workbook.sheetnames) != EXPECTED_SHEETS:
            workbook.close()
            raise ValueError(
                f"Expected exactly sheets {EXPECTED_SHEETS}, found {workbook.sheetnames}"
            )
        for sheet_name, expected in (
            (SHEET_TRAIN, TRAIN_COLUMNS),
            (SHEET_EVAL, EVAL_COLUMNS),
        ):
            sheet = workbook[sheet_name]
            current = tuple(cell.value for cell in sheet[1])
            if current != expected:
                legacy_columns = {"epoch"} if sheet_name == SHEET_TRAIN else set()
                if not set(current).issubset(set(expected) | legacy_columns):
                    workbook.close()
                    raise ValueError(
                        f"Cannot migrate {sheet_name} header: current={current}"
                    )
                rows = [
                    dict(zip(current, values))
                    for values in sheet.iter_rows(min_row=2, values_only=True)
                ]
                if sheet_name == SHEET_TRAIN and "epoch" in current:
                    for row in rows:
                        legacy_epoch = row.get("epoch")
                        row.setdefault("segment_epoch", legacy_epoch)
                        row.setdefault("global_epoch", legacy_epoch)
                sheet.delete_rows(1, sheet.max_row)
                sheet.append(list(expected))
                for row in rows:
                    sheet.append([row.get(column) for column in expected])
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
        return workbook

    def _atomic_save(self, workbook) -> None:
        openpyxl, _, _, _ = _openpyxl()
        temporary = self.path.with_name(
            f".{self.path.stem}.tmp-{os.getpid()}{self.path.suffix}"
        )
        try:
            workbook.save(temporary)
            workbook.close()
            verification = openpyxl.load_workbook(temporary, read_only=True)
            try:
                if tuple(verification.sheetnames) != EXPECTED_SHEETS:
                    raise ValueError(
                        f"Temporary workbook has unexpected sheets: {verification.sheetnames}"
                    )
                if tuple(
                    cell.value for cell in next(verification[SHEET_TRAIN].iter_rows())
                ) != TRAIN_COLUMNS:
                    raise ValueError("Temporary workbook train header verification failed")
                if tuple(
                    cell.value for cell in next(verification[SHEET_EVAL].iter_rows())
                ) != EVAL_COLUMNS:
                    raise ValueError("Temporary workbook eval header verification failed")
            finally:
                verification.close()
            os.replace(temporary, self.path)
        finally:
            try:
                workbook.close()
            except Exception:
                pass
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _steps(sheet) -> set[int]:
        return {
            int(value)
            for (value,) in sheet.iter_rows(min_row=2, max_col=1, values_only=True)
            if value is not None
        }

    def has_train_step(self, step: int) -> bool:
        if not self.path.is_file():
            return False
        workbook = self._load_or_create()
        try:
            return int(step) in self._steps(workbook[SHEET_TRAIN])
        finally:
            workbook.close()

    def has_eval_step(self, step: int) -> bool:
        if not self.path.is_file():
            return False
        workbook = self._load_or_create()
        try:
            rows = [
                row
                for row in workbook[SHEET_EVAL].iter_rows(
                    min_row=2, values_only=True
                )
                if row[0] is not None and int(row[0]) == int(step)
            ]
            return len(rows) == 14
        finally:
            workbook.close()

    def latest_train_global_epoch(self) -> float:
        """Return the last cumulative epoch recorded before a new segment starts."""
        if not self.path.is_file():
            return 0.0
        workbook = self._load_or_create()
        try:
            sheet = workbook[SHEET_TRAIN]
            headers = [cell.value for cell in sheet[1]]
            rows = [
                dict(zip(headers, values))
                for values in sheet.iter_rows(min_row=2, values_only=True)
            ]
            valid = [
                row
                for row in rows
                if row.get("step") is not None and row.get("global_epoch") is not None
            ]
            if not valid:
                return 0.0
            latest = max(valid, key=lambda row: int(row["step"]))
            return float(latest["global_epoch"])
        finally:
            workbook.close()

    def update_train(self, step: int, window_metrics: Mapping[str, Any]) -> bool:
        step = int(step)
        if step <= 0 or step % 100:
            raise ValueError(f"train_100steps accepts positive multiples of 100, got {step}")
        workbook = self._load_or_create()
        sheet = workbook[SHEET_TRAIN]
        if step in self._steps(sheet):
            workbook.close()
            return False
        values = _flatten_train_metrics(window_metrics)
        values["step"] = step
        sheet.append([values.get(column) for column in TRAIN_COLUMNS])
        sheet.auto_filter.ref = sheet.dimensions
        self._atomic_save(workbook)
        return True

    @staticmethod
    def _decorate_eval_rows(existing_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
        for row in rows:
            history = [
                old
                for old in existing_rows
                if old.get("task") == row.get("task")
                and old.get("granularity") == row.get("granularity")
                and int(old.get("step", -1)) < int(row["step"])
                and old.get("f1") is not None
            ]
            previous = max(history, key=lambda item: int(item["step"]), default=None)
            row["f1_change_from_previous"] = (
                float(row["f1"]) - float(previous["f1"])
                if previous is not None and row.get("f1") is not None
                else None
            )
            candidates = [*history, row]
            valid = [item for item in candidates if item.get("f1") is not None]
            best = max(valid, key=lambda item: float(item["f1"]), default=None)
            row["best_f1_so_far"] = best.get("f1") if best else None
            row["best_step_so_far"] = best.get("step") if best else None

    def append_eval(self, step: int, task_metrics: Sequence[Mapping[str, Any]]) -> bool:
        step = int(step)
        rows = [dict(row) for row in task_metrics]
        for row in rows:
            row["step"] = step
        expected_pairs = {
            (task, granularity)
            for task in (*TRAIN_TASKS, "five_task_macro", "five_task_micro")
            for granularity in ("image", "bbox")
        }
        actual_pairs = {(row.get("task"), row.get("granularity")) for row in rows}
        if len(rows) != 14 or actual_pairs != expected_pairs:
            raise ValueError(
                "eval_1000steps requires exactly five tasks plus macro/micro, each at image/bbox; "
                f"found={sorted(actual_pairs)}"
            )

        workbook = self._load_or_create()
        sheet = workbook[SHEET_EVAL]
        headers = [cell.value for cell in sheet[1]]
        existing = [dict(zip(headers, values)) for values in sheet.iter_rows(min_row=2, values_only=True)]
        if sum(int(row.get("step", -1)) == step for row in existing) == 14:
            workbook.close()
            return False

        retained = [row for row in existing if int(row.get("step", -1)) != step]
        if len(retained) != len(existing):
            sheet.delete_rows(2, sheet.max_row - 1)
            for row in retained:
                sheet.append([row.get(column) for column in EVAL_COLUMNS])

        self._decorate_eval_rows(retained, rows)
        for row in rows:
            sheet.append([_value(row.get(column)) for column in EVAL_COLUMNS])
        sheet.auto_filter.ref = sheet.dimensions
        self._atomic_save(workbook)
        return True


def build_eval_rows(
    *,
    step: int,
    checkpoint: str,
    metrics: Mapping[str, Any],
    gate_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    audit_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert scorer and Gate summaries into task, macro, and micro rows.

    Macro rows contain only task-averaged metrics.  Micro rows contain summed
    confusion counts and metrics recomputed from those counts.  BBox Gate
    metrics never fall back to image-Gate metrics.
    """

    scorer_to_diagnostic = {
        "text_overflow": "text_overflow",
        "text_ellipsis": "text_ellipsis",
        "occlusion": "element_overlap",
        "cropping": "element_cropping",
        "content_missing": "content_missing",
    }
    gate_metrics = gate_metrics or {}
    audit_context = dict(audit_context or {})
    context_columns = {
        key: audit_context.get(key)
        for key in (
            "evaluation_split",
            "cache_scope",
            "development_test_reuse",
            "git_sha",
            "git_dirty",
            "recipe_digest",
            "cache_digest",
        )
    }

    def mean(values: Sequence[Any]) -> float | None:
        numeric = [float(value) for value in values if value is not None]
        return sum(numeric) / len(numeric) if numeric else None

    def sum_present(source_rows: Sequence[Mapping[str, Any]], name: str) -> int | None:
        values = [row.get(name) for row in source_rows]
        present = [int(value) for value in values if value is not None]
        return sum(present) if present else None

    def metrics_from_counts(
        *,
        tp: int,
        fp: int,
        fn: int,
        tn: int | None,
        granularity: str,
    ) -> dict[str, Any]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if granularity == "image":
            denominator = tp + fp + fn + int(tn or 0)
            accuracy = (tp + int(tn or 0)) / denominator if denominator else 0.0
        else:
            denominator = tp + fp + fn
            accuracy = tp / denominator if denominator else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def aggregate_gate_status(source_rows: Sequence[Mapping[str, Any]]) -> str:
        statuses = {str(row.get("gate_metric_status")) for row in source_rows}
        if "not_rescored" in statuses:
            return "not_rescored"
        if len(statuses) == 1:
            return next(iter(statuses))
        return "mixed"

    rows: list[dict[str, Any]] = []
    for scorer_task, diagnostic_task in scorer_to_diagnostic.items():
        task_values = metrics.get("tasks", {}).get(scorer_task, {})
        gate = gate_metrics.get(scorer_task, {})
        for granularity in ("image", "bbox"):
            values = task_values.get(granularity, {})
            rescored = gate.get("gated_metrics_by_granularity", {}).get(
                granularity, {}
            )
            threshold_value = gate.get("selected_gate_threshold")
            threshold = (
                float(threshold_value) if threshold_value is not None else 0.0
            )
            bbox_genuinely_rescored = bool(
                gate.get("gated_metrics_by_granularity", {}).get("bbox", {})
            )
            if rescored:
                gated = dict(rescored)
                gate_metric_status = "genuinely_rescored"
            elif threshold <= 0.0:
                # A disabled Gate is exactly the raw scorer result.  This is
                # especially important for BBox, where image Gate metrics must
                # never be copied into the BBox row.
                gated = dict(values)
                gate_metric_status = "raw_equivalent_threshold_zero"
            elif granularity == "bbox":
                gated = {}
                gate_metric_status = "not_rescored"
            else:
                gated = {
                    "precision": gate.get("gated_precision"),
                    "recall": gate.get("gated_recall"),
                    "f1": gate.get("gated_f1"),
                    "tp": gate.get("gated_tp"),
                    "fp": gate.get("gated_fp"),
                    "fn": gate.get("gated_fn"),
                    "tn": gate.get("gated_tn"),
                    "predicted_positive": gate.get("gated_predicted_positive"),
                }
                gate_metric_status = "image_gate_sidecar_rescored"
            tp = values.get("tp")
            fp = values.get("fp")
            gated_tp = gated.get("tp")
            gated_fp = gated.get("fp")
            gated_fn = gated.get("fn")
            gated_tn = gated.get("tn") if granularity == "image" else None
            gated_predicted_positive = (
                int(gated_tp) + int(gated_fp)
                if gated_tp is not None and gated_fp is not None
                else gated.get("predicted_positive")
            )
            raw_predicted_positive = (
                int(tp) + int(fp)
                if tp is not None and fp is not None
                else None
            )
            rows.append(
                {
                    **context_columns,
                    "step": step,
                    "checkpoint": checkpoint,
                    "task": diagnostic_task,
                    "granularity": granularity,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1": values.get("f1"),
                    "tp": tp,
                    "fp": fp,
                    "fn": values.get("fn"),
                    "tn": values.get("tn") if granularity == "image" else None,
                    "accuracy": values.get(
                        "accuracy" if granularity == "image" else "count_accuracy"
                    ),
                    "predicted_positive": (
                        raw_predicted_positive
                    ),
                    "gate_positive": gate.get("gate_positive"),
                    "gate_filtered": gate.get("gate_filtered"),
                    "p_defect_pos": gate.get("p_defect_pos"),
                    "p_defect_neg": gate.get("p_defect_neg"),
                    "parse_error": gate.get("parse_error"),
                    "raw_precision": values.get("precision"),
                    "raw_recall": values.get("recall"),
                    "raw_f1": values.get("f1"),
                    "raw_predicted_positive": (
                        raw_predicted_positive
                    ),
                    "selected_gate_threshold": threshold,
                    "gated_precision": gated.get("precision"),
                    "gated_recall": gated.get("recall"),
                    "gated_f1": gated.get("f1"),
                    "gated_predicted_positive": gated_predicted_positive,
                    "gate_filter_rate": (
                        1.0 - int(gated_predicted_positive) / max(1, int(raw_predicted_positive))
                        if gated_predicted_positive is not None
                        and raw_predicted_positive is not None
                        else None
                    ),
                    "bbox_metrics_genuinely_rescored": (
                        bbox_genuinely_rescored if granularity == "bbox" else None
                    ),
                    "gate_metric_status": gate_metric_status,
                    "_gated_tp": gated_tp,
                    "_gated_fp": gated_fp,
                    "_gated_fn": gated_fn,
                    "_gated_tn": gated_tn,
                }
            )

    for granularity in ("image", "bbox"):
        values = metrics.get("macro", {}).get(granularity, {})
        source_rows = [
            row
            for row in rows
            if row["granularity"] == granularity
        ]
        positive_count = sum(
            int(gate_metrics.get(task, {}).get("positive_count", 0))
            for task in scorer_to_diagnostic
        )
        negative_count = sum(
            int(gate_metrics.get(task, {}).get("negative_count", 0))
            for task in scorer_to_diagnostic
        )
        p_defect_pos = (
            sum(
                float(gate_metrics.get(task, {}).get("p_defect_pos") or 0.0)
                * int(gate_metrics.get(task, {}).get("positive_count", 0))
                for task in scorer_to_diagnostic
            )
            / positive_count
            if positive_count
            else None
        )
        p_defect_neg = (
            sum(
                float(gate_metrics.get(task, {}).get("p_defect_neg") or 0.0)
                * int(gate_metrics.get(task, {}).get("negative_count", 0))
                for task in scorer_to_diagnostic
            )
            / negative_count
            if negative_count
            else None
        )
        all_gated = all(row.get("gated_f1") is not None for row in source_rows)
        macro_row = {
            **context_columns,
            "step": step,
            "checkpoint": checkpoint,
            "task": "five_task_macro",
            "granularity": granularity,
            "precision": values.get("precision", mean([row.get("precision") for row in source_rows])),
            "recall": values.get("recall", mean([row.get("recall") for row in source_rows])),
            "f1": values.get("f1", mean([row.get("f1") for row in source_rows])),
            # Macro is a mean-of-tasks row.  Counts intentionally remain empty.
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
            "accuracy": mean([row.get("accuracy") for row in source_rows]),
            "predicted_positive": None,
            "gate_positive": None,
            "gate_filtered": None,
            "p_defect_pos": mean([row.get("p_defect_pos") for row in source_rows]),
            "p_defect_neg": mean([row.get("p_defect_neg") for row in source_rows]),
            "parse_error": None,
            "raw_precision": mean([row.get("raw_precision") for row in source_rows]),
            "raw_recall": mean([row.get("raw_recall") for row in source_rows]),
            "raw_f1": mean([row.get("raw_f1") for row in source_rows]),
            "raw_predicted_positive": None,
            "selected_gate_threshold": mean([row.get("selected_gate_threshold") for row in source_rows]),
            "gated_precision": mean([row.get("gated_precision") for row in source_rows]) if all_gated else None,
            "gated_recall": mean([row.get("gated_recall") for row in source_rows]) if all_gated else None,
            "gated_f1": mean([row.get("gated_f1") for row in source_rows]) if all_gated else None,
            "gated_predicted_positive": None,
            "gate_filter_rate": mean([row.get("gate_filter_rate") for row in source_rows]) if all_gated else None,
            "bbox_metrics_genuinely_rescored": (
                all(bool(row.get("bbox_metrics_genuinely_rescored")) for row in source_rows)
                if granularity == "bbox"
                else None
            ),
            "gate_metric_status": aggregate_gate_status(source_rows),
        }
        rows.append(macro_row)

        raw_tp = int(sum_present(source_rows, "tp") or 0)
        raw_fp = int(sum_present(source_rows, "fp") or 0)
        raw_fn = int(sum_present(source_rows, "fn") or 0)
        raw_tn = (
            int(sum_present(source_rows, "tn") or 0)
            if granularity == "image"
            else None
        )
        raw_micro = metrics_from_counts(
            tp=raw_tp, fp=raw_fp, fn=raw_fn, tn=raw_tn, granularity=granularity
        )
        all_gated_counts = all(
            row.get("_gated_tp") is not None
            and row.get("_gated_fp") is not None
            and row.get("_gated_fn") is not None
            for row in source_rows
        )
        gated_micro: dict[str, Any] | None = None
        gated_tp = gated_fp = gated_fn = gated_tn = None
        if all_gated_counts:
            gated_tp = int(sum_present(source_rows, "_gated_tp") or 0)
            gated_fp = int(sum_present(source_rows, "_gated_fp") or 0)
            gated_fn = int(sum_present(source_rows, "_gated_fn") or 0)
            gated_tn = (
                int(sum_present(source_rows, "_gated_tn") or 0)
                if granularity == "image"
                else None
            )
            gated_micro = metrics_from_counts(
                tp=gated_tp,
                fp=gated_fp,
                fn=gated_fn,
                tn=gated_tn,
                granularity=granularity,
            )
        rows.append(
            {
                **context_columns,
                "step": step,
                "checkpoint": checkpoint,
                "task": "five_task_micro",
                "granularity": granularity,
                **raw_micro,
                "tp": raw_tp,
                "fp": raw_fp,
                "fn": raw_fn,
                "tn": raw_tn,
                "predicted_positive": raw_tp + raw_fp,
                "gate_positive": sum(
                    int(gate_metrics.get(task, {}).get("gate_positive", 0))
                    for task in scorer_to_diagnostic
                ),
                "gate_filtered": sum(
                    int(gate_metrics.get(task, {}).get("gate_filtered", 0))
                    for task in scorer_to_diagnostic
                ),
                "p_defect_pos": p_defect_pos,
                "p_defect_neg": p_defect_neg,
                "parse_error": sum(
                    int(gate_metrics.get(task, {}).get("parse_error", 0))
                    for task in scorer_to_diagnostic
                ),
                "raw_precision": raw_micro["precision"],
                "raw_recall": raw_micro["recall"],
                "raw_f1": raw_micro["f1"],
                "raw_predicted_positive": raw_tp + raw_fp,
                "selected_gate_threshold": mean([row.get("selected_gate_threshold") for row in source_rows]),
                "gated_precision": gated_micro["precision"] if gated_micro else None,
                "gated_recall": gated_micro["recall"] if gated_micro else None,
                "gated_f1": gated_micro["f1"] if gated_micro else None,
                "gated_predicted_positive": (
                    int(gated_tp) + int(gated_fp) if gated_micro else None
                ),
                "gate_filter_rate": (
                    1.0 - (int(gated_tp) + int(gated_fp)) / max(1, raw_tp + raw_fp)
                    if gated_micro
                    else None
                ),
                "bbox_metrics_genuinely_rescored": (
                    all(bool(row.get("bbox_metrics_genuinely_rescored")) for row in source_rows)
                    if granularity == "bbox"
                    else None
                ),
                "gate_metric_status": aggregate_gate_status(source_rows),
            }
        )
    return rows
