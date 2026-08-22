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
    "epoch",
    "gpu_num",
    "max_num_tokens",
    "learning_rate",
    "gate_loss_weight",
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
    "loss_attention",
    "weighted_gate_loss",
    "weighted_attention_loss",
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
)
TRAIN_MODULE_COLUMNS = (
    "detail_layer5_norm",
    "detail_layer15_norm",
    "detail_layer26_norm",
    "detail_fused_norm",
    "relation_context_norm",
    "relation_gate_prob_mean",
    "pbd_delta_norm",
    "relation_grad_norm",
    "gate_grad_norm",
    "pbd_grad_norm",
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
            return len(rows) == 12
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
            for task in (*TRAIN_TASKS, "five_task_macro")
            for granularity in ("image", "bbox")
        }
        actual_pairs = {(row.get("task"), row.get("granularity")) for row in rows}
        if len(rows) != 12 or actual_pairs != expected_pairs:
            raise ValueError(
                "eval_1000steps requires exactly five tasks plus macro, each at image/bbox; "
                f"found={sorted(actual_pairs)}"
            )

        workbook = self._load_or_create()
        sheet = workbook[SHEET_EVAL]
        headers = [cell.value for cell in sheet[1]]
        existing = [dict(zip(headers, values)) for values in sheet.iter_rows(min_row=2, values_only=True)]
        if sum(int(row.get("step", -1)) == step for row in existing) == 12:
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
) -> list[dict[str, Any]]:
    """Convert scorer JSON plus gate summaries into the 12 requested rows."""

    scorer_to_diagnostic = {
        "text_overflow": "text_overflow",
        "text_ellipsis": "text_ellipsis",
        "occlusion": "element_overlap",
        "cropping": "element_cropping",
        "content_missing": "content_missing",
    }
    gate_metrics = gate_metrics or {}
    rows: list[dict[str, Any]] = []
    for scorer_task, diagnostic_task in scorer_to_diagnostic.items():
        task_values = metrics.get("tasks", {}).get(scorer_task, {})
        gate = gate_metrics.get(scorer_task, {})
        for granularity in ("image", "bbox"):
            values = task_values.get(granularity, {})
            tp = values.get("tp")
            fp = values.get("fp")
            rows.append(
                {
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
                        int(tp) + int(fp)
                        if tp is not None and fp is not None
                        else None
                    ),
                    "gate_positive": gate.get("gate_positive"),
                    "gate_filtered": gate.get("gate_filtered"),
                    "p_defect_pos": gate.get("p_defect_pos"),
                    "p_defect_neg": gate.get("p_defect_neg"),
                    "parse_error": gate.get("parse_error"),
                }
            )

    for granularity in ("image", "bbox"):
        values = metrics.get("macro", {}).get(granularity, {})
        source_rows = [
            row
            for row in rows
            if row["granularity"] == granularity
        ]
        summed = lambda name: sum(
            int(row[name]) for row in source_rows if row.get(name) is not None
        )
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
        rows.append(
            {
                "step": step,
                "checkpoint": checkpoint,
                "task": "five_task_macro",
                "granularity": granularity,
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1": values.get("f1"),
                "tp": summed("tp"),
                "fp": summed("fp"),
                "fn": summed("fn"),
                "tn": summed("tn") if granularity == "image" else None,
                "accuracy": (
                    sum(
                        float(row["accuracy"])
                        for row in source_rows
                        if row.get("accuracy") is not None
                    )
                    / max(1, sum(row.get("accuracy") is not None for row in source_rows))
                ),
                "predicted_positive": summed("predicted_positive"),
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
            }
        )
    return rows
