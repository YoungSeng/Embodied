"""Formal UI5 metrics, v3 joint selection, and complete GRPO Excel tables."""
from pathlib import Path
import json
import math
import os
import shutil

from eaglevl.train import ui5_curriculum_artifacts as artifacts
from eaglevl.train.ui5_grpo_checkpoint import validate_checkpoint
from scripts.build_ui5_grpo_mixed_manifest import read_json, write_json


def detail_tables(state):
    result = {"image_detail": [], "bbox_detail": []}
    baseline = state["evaluations"][0]["metrics"] if state["evaluations"] else None
    for evaluation in state["evaluations"]:
        metrics = evaluation["metrics"]
        for kind in ("image", "bbox"):
            for scope in (*artifacts.TASKS, "macro", "micro"):
                if scope in artifacts.TASKS:
                    current = metrics["tasks"][scope][kind]
                    initial = baseline["tasks"][scope][kind]
                    health = metrics["tasks"][scope]
                else:
                    current, initial = metrics[scope][kind], baseline[scope][kind]
                    health = metrics["overall"]
                row = dict(step=evaluation["step"], task=scope, decoder="boundary_v3/hybrid",
                           checkpoint=evaluation["candidate_checkpoint"],
                           total_samples=health["total_samples"], invalid=health["invalid_predictions"],
                           invalid_rate=health["invalid_prediction_rate"])
                for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
                    row[key] = current.get(key)
                    old, value = initial.get(key), current.get(key)
                    row["delta_step0_" + key] = value - old if value is not None and old is not None else None
                row["tn_availability"] = "available" if row["tn"] is not None else "not provided by evaluator for this scope"
                result[kind + "_detail"].append(row)
    return result


def write_workbook(path, tables):
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    expected = {}
    for name, rows in tables.items():
        columns = list(dict.fromkeys(key for row in rows for key in row)) or ["step"]
        sheet = workbook.create_sheet(name)
        sheet.append(columns)
        cells = [[artifacts._excel_value(row.get(key)) for key in columns] for row in rows]
        for row in cells:
            sheet.append(row)
        expected[name] = [columns, *cells]
        sheet.freeze_panes = "C2" if name in {"image_detail", "bbox_detail"} else "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.font = Font(name="Arial", size=10, color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sheet.row_dimensions[1].height = 32
        for column in sheet.columns:
            width = min(64, max(12, max(len(str(c.value or "")) for c in column) + 2))
            sheet.column_dimensions[column[0].column_letter].width = width
            for cell in column[1:]:
                cell.font = Font(name="Arial", size=10)
                cell.alignment = Alignment(vertical="center", wrap_text=isinstance(cell.value, str))
                if isinstance(cell.value, float):
                    cell.number_format = "0.00E+00" if column[0].value == "learning_rate" else "0.000000"
                elif isinstance(cell.value, int) and not isinstance(cell.value, bool):
                    cell.number_format = "#,##0"
                lines = max(1, math.ceil(len(str(cell.value or "")) / max(8, width - 2)))
                sheet.row_dimensions[cell.row].height = max(sheet.row_dimensions[cell.row].height or 18, lines * 13 + 4)
    temporary = path.with_suffix(".tmp.xlsx")
    workbook.save(temporary)
    workbook.close()
    verify = openpyxl.load_workbook(temporary, read_only=True, data_only=False)
    try:
        if verify.sheetnames != list(tables) or verify.sheetnames[-2:] != ["image_detail", "bbox_detail"]:
            raise ValueError("GRPO workbook sheet order changed")
        for name, rows in expected.items():
            actual = list(verify[name].values)
            if len(actual) != len(rows):
                raise ValueError(f"incomplete workbook table: {name}")
            for observed, expected_row in zip(actual, rows):
                if any(not artifacts._same_excel_value(a, b) for a, b in zip(observed, expected_row)):
                    raise ValueError(f"workbook values changed: {name}")
            if any(cell.data_type in {"f", "e"} for row in verify[name] for cell in row):
                raise ValueError(f"unexpected formula/error in GRPO workbook: {name}")
    finally:
        verify.close()
    os.replace(temporary, path)


def publish_best_link(alias, target):
    temporary = alias.with_name("." + alias.name + ".tmp")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target, target_is_directory=True)
    os.replace(temporary, alias)


def preserve_best_aliases(output, result):
    """Keep the three v3 winners (ties favor earlier steps) and full latest."""
    root = (output / "checkpoints").resolve()
    root.mkdir(parents=True, exist_ok=True)
    retained = set()
    for key in ("best_image", "best_bbox", "best_joint"):
        record = result[key]
        target = Path(record["checkpoint_path"]).resolve(strict=True)
        if target != (output / "initial_model").resolve() and target.parent != root:
            raise ValueError("best checkpoint points outside this run")
        retained.add(target)
        alias = root / key.replace("_", "-")
        publish_best_link(alias, target)
        write_json(output / (key.replace("_", "-") + ".json"), record)
    state_path = output / "checkpoints.json"
    state = read_json(state_path)
    for row in state["evaluations"]:
        if not row.get("checkpoint_preserved"):
            continue
        path = Path(row["checkpoint_path"]).resolve()
        if path in retained:
            continue
        if path.parent != root or path.name != f"step-{row['step']:06d}":
            raise ValueError("superseded checkpoint is outside the owned directory")
        if path.exists():
            if not (path / "ui5_preserved_checkpoint.json").is_file():
                raise ValueError("refusing to prune an unowned checkpoint directory")
            shutil.rmtree(path)
        row.update(checkpoint_preserved=False, checkpoint_path="", checkpoint_pruned=True)
    write_json(state_path, state)


def refresh_workbook(run):
    output = Path(run["output_dir"])
    state = artifacts.load_checkpoints_state(output / "checkpoints.json")
    train = [read_json(path) for path in sorted((output / "metrics").glob("step*.json"))]
    diagnostics = [row for path in sorted((output / "diagnostics").glob("train_ar_step*.json"))
                   for row in read_json(path)["rows"]]
    manifest = read_json(Path(run["mixed_dir"]) / "manifest.json")
    samples = [read_json(path) for path in sorted((output / "diagnostics").glob("sampling_step*.json"))]
    actual_sampling = [dict(step=sample["step"], stratum=key, draws=count,
                            fraction=count / sample["total_draws"])
                       for sample in samples for key, count in sorted(sample["counts"].items())]
    identity_rows = flatten_identity(run)
    revision = output / "runtime_code_revision.json"
    if revision.is_file():
        identity_rows.extend(flatten_identity(read_json(revision), "runtime_revision."))
    tables = dict(train_curve=train, ui5_overall=artifacts._overall_rows(state),
                  ui5_by_task=artifacts._by_task_rows(state), checkpoints=artifacts._checkpoint_rows(state),
                  mixed_pool=manifest["sampling"], actual_sampling=actual_sampling,
                  train_ar_diagnostic=diagnostics,
                  run_identity=identity_rows,
                  **detail_tables(state))
    write_workbook(output / "diagnostics/ui5_grpo_training_evaluation.xlsx", tables)


def flatten_identity(value, prefix=""):
    rows = []
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            rows.extend(flatten_identity(item, name + "."))
        else:
            rows.append(dict(key=name, value=item))
    return rows


def register_evaluation(run, step, candidate, metrics, seconds):
    output = Path(run["output_dir"])
    if step:
        validate_checkpoint(candidate, run["identity"])
    # The v3 normalizer and best-image/bbox/joint rules are reused. The GRPO
    # resume contract above replaces its SFT dataloader validation.
    result = artifacts.update_curriculum_artifacts(
        step=step, scorer_metrics=metrics, candidate_checkpoint=candidate,
        checkpoints_json=output / "checkpoints.json",
        workbook_path=output / "diagnostics/.curriculum_metrics.xlsx",
        formal_checkpoint_root=output / "checkpoints", resume_from=str(candidate),
        evaluation_seconds=seconds, validate_candidate=False, expected_ranks=2)
    preserve_best_aliases(output, result)
    refresh_workbook(run)
    return result
