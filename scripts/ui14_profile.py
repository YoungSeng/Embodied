"""Formal cluster profile layered on the existing UI5 pipeline."""
from pathlib import Path
from ui14_common import *


def profile_environment(*, project_root=None, data_root=None):
    root = str(data_root or DATA_ROOT)
    project = str(project_root or CLUSTER_PROJECT)
    env = read_json(PROJECT_ROOT / "configs" / "ui14_cpt9000_formal.json")["environment"]
    env.update(PROJECT_ROOT=project, BASE_MODEL=INIT_CHECKPOINT, MODEL_PATH=INIT_CHECKPOINT,
        INIT_CHECKPOINT=INIT_CHECKPOINT, UI14_DATA_ROOT=root,
        UI_TASK_REGISTRY=root + "/task_registry.json", UI_EVAL_MANIFEST=root + "/evaluation_manifest.json",
        UI14_CHECK_REPORT=root + "/cpu_check_report.json", META_PATH=root + "/training_recipe.json",
        UI5_CROP_META_PATH=root + "/training_recipe.json", UI5_CROP_AUDIT_DIR=UI5_AUDIT,
        EVAL_INPUT_DIR=WORKSPACE + "/data",
        EVAL_DETECTOR_CACHE=WORKSPACE + "/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5",
        DEEPSPEED_CONFIG=project + "/deepspeed_configs/zero_stage2_two_lr_config.json",
        OUTPUT_DIR=WORKSPACE + "/gui_models/locany-m32-cpt9000-ui14-a800x4-repair-v2")
    return env


def validate_prepared_profile(runtime):
    # Full checks run explicitly before rendering; launch validates the digest-bound report.
    root = Path(runtime["UI14_DATA_ROOT"])
    report = read_json(root / "cpu_check_report.json")
    if not report.get("ready") or report.get("registry_count") != 14 or report.get("evaluation_count") != 14:
        raise RuntimeError("UI14 CPU check is incomplete or failed")
    if not report.get("repair_run_id") or not report.get("normalization_id"):
        raise RuntimeError("UI14 CPU report is not bound to the repaired data batch")
    if runtime.get("INIT_CHECKPOINT") and Path(runtime["INIT_CHECKPOINT"]) != Path(report["init_checkpoint"]):
        raise RuntimeError("CPU check validated a different CPT initialization checkpoint")
    for name, expected in report.get("artifact_digests", {}).items():
        if file_digest(root / name) != expected: raise RuntimeError(f"Prepared UI14 artifact changed: {name}")
    if not report.get("artifact_digests"): raise RuntimeError("CPU check has no artifact digests")
    for path, expected in report.get("external_digests", {}).items():
        if file_digest(path) != expected: raise RuntimeError(f"UI14 source/cache changed since CPU check: {path}")
    from ui14_repair import validate_normalization
    snapshot = validate_normalization(root)
    if snapshot["normalization_id"] != report["normalization_id"]:
        raise RuntimeError("UI14 CPU report belongs to another repair batch")
    validate_run_data_binding(runtime, snapshot)


def validate_run_data_binding(runtime, snapshot, *, create=False):
    """A resumed SFT must retain the data batch that created its optimizer state."""
    if not runtime.get("OUTPUT_DIR"):
        return
    output = Path(runtime["OUTPUT_DIR"])
    marker = output / "ui14_training_data.json"
    expected = {"normalization_id": snapshot["normalization_id"], "repair_run_id": snapshot["repair_run_id"],
                "data_root": str(Path(runtime["UI14_DATA_ROOT"]).resolve()), "init_cpt_step": 9000}
    if marker.is_file():
        if read_json(marker) != expected:
            raise RuntimeError("SFT output directory belongs to another repair batch; use a fresh output directory")
    else:
        if output.exists() and any(output.glob("checkpoint-*")):
            raise RuntimeError("Existing SFT checkpoints have no repair data binding; refusing to resume them")
        if create:
            write_json(marker, expected)
