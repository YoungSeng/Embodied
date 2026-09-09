"""CPU-frozen model snapshots and independent UI alignment re-evaluation."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
from ui14_common import read_json, read_jsonl, write_json, digest, file_digest, PROJECT_ROOT
from ui14_progress import track, file_activity
from ui14_verification import signature
from ui14_alignment_common import independent_output
from eaglevl.ui_answer_grammar import VERSION, decode_contract


def snapshot_checkpoint(source, target):
    """Copy immutable weights once on CPU; never patch/write the old run.

    Each file is atomic and independently resumable. Hash while copying to
    avoid a second read of a model shard. Runtime validates the saved file
    attributes; it never starts a model copy on allocated evaluation GPUs.
    """
    source, target = Path(source).resolve(strict=True), Path(target).resolve()
    independent_output(target, source)
    from eaglevl.train.ui5_checkpoint_utils import has_model_weights
    from eaglevl.ui_task_registry import validate_registry
    config = read_json(source / "config.json")
    validate_registry(config.get("ui_task_registry"), 14)
    if config.get("ui_num_tasks") != 14 or not has_model_weights(source): raise ValueError("Expected complete 14-task model weights")
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "snapshot.json"
    previous = read_json(marker) if marker.is_file() else {"source": str(source), "files": {}}
    if previous["source"] != str(source): raise ValueError("Snapshot directory belongs to another checkpoint")
    # Trainer/optimizer/scheduler states are not inference inputs.
    files = [p for p in sorted(source.iterdir()) if p.is_file() and
             (p.suffix in (".json", ".py", ".safetensors", ".model", ".txt", ".jinja")
              or p.name.startswith("pytorch_model") and p.suffix == ".bin")
             and p.name not in ("trainer_state.json",)]
    names = {p.name for p in files}
    if previous.get("complete") and (names != set(previous["files"]) or any(
            signature(source / n) != row["source_stat"] for n, row in previous["files"].items())):
        raise ValueError("Completed source checkpoint changed; use a new comparison data directory")
    # Keep all previously published files in intermediate receipts so a second
    # interruption does not lose the other completed shards.
    receipt, reused = {"source": str(source), "files": dict(previous["files"])}, 0
    for file in track(files, f"CPU 模型快照 {source.name}", unit="文件"):
        state = signature(file)
        destination = target / file.name
        old = previous["files"].get(file.name)
        if old and state == old["source_stat"] and destination.is_file() and signature(destination) == old["snapshot_stat"]:
            receipt["files"][file.name] = old; reused += 1
        else:
            temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
            h = hashlib.sha256()
            with file.open("rb") as src, temporary.open("wb") as dst, file_activity(file, state[0], "字节") as progress:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk); h.update(chunk); progress.advance(len(chunk))
                dst.flush(); os.fsync(dst.fileno())
            if signature(file) != state: raise ValueError(f"Source checkpoint changed while copying: {file}")
            os.replace(temporary, destination)
            receipt["files"][file.name] = {"source_stat": state, "snapshot_stat": signature(destination), "sha256": h.hexdigest()}
        write_json(marker, receipt)
    receipt["files"] = {name: receipt["files"][name] for name in sorted(names)}
    receipt["weight_id"] = digest({name: r["sha256"] for name, r in receipt["files"].items()})
    receipt["complete"] = True
    write_json(marker, receipt)
    print(f"[eval-prepare] {source.name}: reused={reused}, copied={len(files)-reused}; {target}", flush=True)
    return receipt


def prepare_comparisons(old_run, manifest, tasks, steps, root):
    old_run, root = Path(old_run).resolve(), Path(root).resolve()
    independent_output(root, old_run, Path(manifest).parent)
    document = read_json(manifest)
    prepared = []
    for step in steps:
        state = read_json(old_run / "evaluation" / f"ui14-step-{step}.json")
        if state.get("status") != "success" or state["identity"]["manifest_digest"] != file_digest(manifest):
            raise ValueError("Comparison requires this checkpoint's completed evaluation and corresponding old test")
        source = old_run / f"checkpoint-{step}"
        model = root / "model_snapshots" / f"checkpoint-{step}"
        snapshot = snapshot_checkpoint(source, model)
        saved_tasks = read_json(model / "config.json")["ui_task_registry"]
        for spec in document["tasks"]:
            saved = saved_tasks[spec["task_id"]]
            for key in ("task_id", "task_key", "relation_family", "view_policy", "test", "normalization_id"):
                if saved.get(key) != spec.get(key): raise ValueError(f"Old checkpoint/test registry mismatch: {spec['task_key']}.{key}")
        task_files = {spec["task_key"]: {"path": spec["test"], "sha256": file_digest(spec["test"]), "stat": signature(spec["test"])}
                      for spec in document["tasks"] if spec["task_key"] in tasks}
        if set(task_files) != set(tasks): raise ValueError("Unknown task in old test manifest")
        image_files = {str(Path(row["source_image"]).resolve()): signature(row["source_image"])
                       for task_file in task_files.values() for row in read_jsonl(task_file["path"])}
        if any(next(s for s in document["tasks"] if s["task_key"] == t)["view_policy"] != "full_image" for t in tasks):
            raise ValueError("This independent alignment comparison expects the original full-image task policy")
        binding = {"step": step, "old_run": str(old_run), "weight_id": snapshot["weight_id"],
                   "model_snapshot": str(model), "source_manifest": str(Path(manifest).resolve()),
                   "manifest_sha256": file_digest(manifest), "tasks": list(tasks), "task_files": task_files,
                   "image_files": image_files,
                   "decoder_contract": decode_contract(VERSION), "eval_set_id": document.get("eval_set_id"),
                   "snapshot_record_sha256": file_digest(model / "snapshot.json"),
                   "prediction_reuse": "only this exact weight/test/prompt/view/decode identity; no old prediction import"}
        comparison_id = digest(binding)
        destination = root / "comparisons" / f"step-{step}" / comparison_id
        destination.mkdir(parents=True, exist_ok=True)
        # Independent list; the frozen test records/images themselves are read-only.
        data = Path(manifest).read_bytes()
        (destination / "evaluation_manifest.json").write_bytes(data)
        write_json(destination / "binding.json", {**binding, "comparison_id": comparison_id})
        prepared.append(str(destination))
    write_json(root / "comparison_plan.json", {"complete": True, "directories": prepared})
    return prepared


def run_comparison(destination, *, gpus="0"):
    destination = Path(destination)
    binding = read_json(destination / "binding.json")
    if binding["decoder_contract"] != decode_contract(VERSION):
        raise ValueError("Decoder code changed; run CPU eval-prepare to create a new prediction identity")
    model = Path(binding["model_snapshot"])
    if file_digest(model / "snapshot.json") != binding["snapshot_record_sha256"]: raise ValueError("Model snapshot receipt changed")
    snapshot = read_json(model / "snapshot.json")
    for name, row in snapshot["files"].items():
        if signature(model / name) != row["snapshot_stat"]: raise ValueError(f"Model snapshot changed: {name}")
    manifest = destination / "evaluation_manifest.json"
    if file_digest(manifest) != binding["manifest_sha256"]: raise ValueError("Comparison test manifest changed")
    for row in binding["task_files"].values():
        if signature(row["path"]) != row["stat"]: raise ValueError("Old test attributes changed; refresh CPU comparison preparation")
    for name, state in binding["image_files"].items():
        if signature(name) != state: raise ValueError("Old test image changed; refresh CPU comparison preparation")
    completed = destination / "complete.json"
    if completed.is_file():
        result = read_json(completed)
        if result.get("comparison_id") == binding["comparison_id"] and all(file_digest(destination / n) == h for n, h in result["artifacts"].items()):
            print(f"[eval-existing] reused step={binding['step']}: {destination}", flush=True)
            return result
    prediction = destination / "predictions"
    command = [sys.executable, str(PROJECT_ROOT / "scripts/run_ui5_parallel_inference.py"),
               "--checkpoint", str(model), "--processor-path", str(model), "--eval-manifest", str(manifest),
               "--input-dir", str(destination), "--output-dir", str(prediction), "--gpu-devices", gpus,
               "--workers-per-gpu", "2", "--attn-implementation", "sdpa",
               "--inference-script", str(PROJECT_ROOT / "scripts/inference_ui_defect_locany.py"),
               "--tasks", *binding["tasks"], "--save-raw-answer"]
    env = {**os.environ, "UI_EVAL_ANSWER_GRAMMAR": VERSION, "PYTHONUNBUFFERED": "1"}
    subprocess.run(command, env=env, check=True)
    from run_ui14_eval import score_ui9
    metrics = {}
    for spec in read_json(manifest)["tasks"]:
        if spec["task_key"] in binding["tasks"]:
            metrics[spec["task_key"]], _ = score_ui9(spec, prediction, destination)
    write_json(destination / "metrics.json", {"comparison_id": binding["comparison_id"], "step": binding["step"],
        "eval_set_id": binding["eval_set_id"], "manifest_sha256": binding["manifest_sha256"],
        "weight_id": binding["weight_id"], "decoder_contract": binding["decoder_contract"], "tasks": metrics})
    # Inference cannot mutate even the independent model copy: do not publish
    # success for an unexpected weight/config change.
    for name, row in snapshot["files"].items():
        if signature(model / name) != row["snapshot_stat"]: raise ValueError(f"Inference modified snapshot: {name}")
    result = {"comparison_id": binding["comparison_id"], "status": "success", "tasks": list(metrics),
              "artifacts": {"metrics.json": file_digest(destination / "metrics.json")}}
    write_json(completed, result)
    return result
