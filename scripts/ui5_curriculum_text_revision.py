"""Publish an immutable answer-only revision; never open or generate crop PNGs."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from eaglevl.train.ui5_supervision import (
    FORMAT_VERSION, answer_for_record, assistant_slot, corrected_record,
    record_boxes, sample_id, validate_answer,
)

POOLS = ("hard", "matched_anchor", "global_replay")
RECIPE = "ui5_crop_rollout4_curriculum.json"
SIDECARS = ("hard_groups.jsonl", "matched_anchor_groups.jsonl", "crop_assets.jsonl")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, payload):
    # Publication directory is exclusive; _SUCCESS is always written last.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def metadata(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def read_publication(source):
    manifest = json.loads((source / "curriculum_manifest.json").read_text(encoding="utf-8"))
    success = json.loads((source / "_SUCCESS.json").read_text(encoding="utf-8"))
    payload = dict(manifest)
    identity = payload.pop("identity_digest", None)
    if (digest(payload) != identity or success.get("identity_digest") != identity
            or success.get("complete") is not True):
        raise ValueError("text revision source publication identity is invalid")
    for name in (RECIPE, *SIDECARS, *(f"{pool}.jsonl" for pool in POOLS)):
        if metadata(source / name) != success.get("files", {}).get(name):
            raise ValueError(f"text publication hash differs: {source / name}")
    if sha(source / RECIPE) != success.get("recipe_sha256"):
        raise ValueError("text publication recipe hash differs")
    return manifest, success


def resolved_recipe(source):
    recipe = json.loads((source / RECIPE).read_text(encoding="utf-8"))
    pools = set()
    for entry in recipe.values():
        pool = entry.get("curriculum_pool")
        if pool not in POOLS or pool in pools:
            raise ValueError("text revision requires exactly the three existing pools")
        pools.add(pool)
        paths = entry["annotation"]
        paths = paths if isinstance(paths, list) else [paths]
        resolved = [Path(p) if Path(p).is_absolute() else source / p for p in paths]
        if len(resolved) != 1 or resolved[0].resolve() != (source / f"{pool}.jsonl").resolve():
            raise ValueError(f"recipe does not actually read the published pool: {pool}")
    if pools != set(POOLS):
        raise ValueError("recipe lacks a training pool")
    return recipe


def supervision_audit(directory):
    """Compare each complete answer to independent current-view GT, via the real recipe."""
    resolved_recipe(directory)
    reports = []
    for pool in POOLS:
        total = negative = 0
        examples = []
        with (directory / f"{pool}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                validate_answer(record)
                total += 1
                if not record_boxes(record):
                    negative += 1
                    if len(examples) < 3:
                        examples.append({"sample_id": sample_id(record), "crop_id": record.get("_ui5_crop_id"),
                                         "assistant_text": "<box>none</box>"})
        if not negative:
            raise ValueError(f"training pool lacks negative supervision: {pool}")
        reports.append({"pool": pool, "source": str(directory / f"{pool}.jsonl"),
                        "assistant_targets": total, "negative_box_none_targets": negative,
                        "ref_negative_targets": 0, "negative_examples": examples,
                        "full_answer_validated": True})
    return reports


def verify_revision(directory, *, expected_recipe_sha=None, expected_identity=None):
    directory = Path(directory).resolve(strict=True)
    manifest, success = read_publication(directory)
    revision = manifest.get("text_revision", {})
    if revision.get("format_version") != FORMAT_VERSION:
        raise ValueError("v3 requires a published corrected training-text revision")
    resolved_recipe(directory)
    if (manifest.get("training_text") != {"recipe_sha256": sha(directory / RECIPE),
            "annotations": {p: metadata(directory / f"{p}.jsonl") for p in POOLS}}
            or Path(manifest["outputs"]["recipe"]).resolve() != directory / RECIPE):
        raise ValueError("manifest training text paths/hashes differ from actual recipe")
    if expected_recipe_sha and sha(directory / RECIPE) != expected_recipe_sha:
        raise ValueError("new training recipe hash differs from formal YAML")
    if expected_identity and manifest["identity_digest"] != expected_identity:
        raise ValueError("new training text identity differs from formal YAML")
    if metadata(directory / "supervision_format.json") != success["files"].get("supervision_format.json"):
        raise ValueError("supervision format audit hash differs")
    source = Path(revision["source_directory"]).resolve(strict=True)
    if source == directory or sha(source / "curriculum_manifest.json") != revision["source_manifest_sha256"]:
        raise ValueError("text revision source manifest changed")
    if sha(source / "_SUCCESS.json") != revision["source_success_sha256"]:
        raise ValueError("text revision source publication changed")
    original = json.loads((source / "curriculum_manifest.json").read_text(encoding="utf-8"))
    asset_root = Path(original.get("outputs", {}).get("crop_asset_root", str(source))).resolve()
    if Path(manifest["outputs"]["crop_asset_root"]).resolve() != asset_root:
        raise ValueError("text revision PNG root differs from its source")
    if manifest["crop_assets"] != original["crop_assets"] or manifest["pools"] != original["pools"]:
        raise ValueError("text revision changed crop assets or sample counts")
    for name in SIDECARS:
        if metadata(directory / name) != metadata(source / name):
            raise ValueError(f"text revision changed immutable group/asset inventory: {name}")
    return manifest


def publish_revision(source):
    source = Path(source).resolve(strict=True)
    manifest, success = read_publication(source)
    recipe = resolved_recipe(source)
    identity = {"format_version": FORMAT_VERSION, "source_directory": str(source),
                "source_manifest_sha256": sha(source / "curriculum_manifest.json"),
                "source_success_sha256": sha(source / "_SUCCESS.json"),
                "implementation_sha256": sha(Path(__file__).resolve()),
                "answer_contract_sha256": sha(Path(__file__).resolve().parents[1] / "eaglevl/train/ui5_supervision.py")}
    destination = source.with_name(source.name + "-text-v3-1-" + digest(identity)[:12])
    if destination.exists():
        verified = verify_revision(destination)
        if verified["text_revision"] != identity:
            raise ValueError("existing text revision has another identity")
        supervision_audit(destination)
        print(f"[TEXT REUSE] {destination} PNG_generation=0", flush=True)
        return destination, verified
    destination.mkdir(exist_ok=False)
    reports = []
    for pool in POOLS:
        started = time.monotonic()
        expected_records = manifest["pools"][pool]["training_records"]
        report = {"pool": pool, "records": 0, "negative_targets": 0, "ref_negative_before": 0,
                  "ref_negative_after": 0, "modified_answers": 0, "source_positive_crop_negative": 0,
                  "modified_sample_ids": [], "modified_examples": [], "format_version": FORMAT_VERSION}
        filename = f"{pool}.jsonl"
        with (source / filename).open("rb") as reader, (destination / filename).open("xb") as writer:
            for line in reader:
                if not line.strip():
                    writer.write(line)
                    continue
                record = json.loads(line)
                answer, key = assistant_slot(record)
                negative = not record_boxes(record)
                report["records"] += 1
                report["negative_targets"] += int(negative)
                report["ref_negative_before"] += int(negative and "<ref>" in answer[key].lower())
                report["source_positive_crop_negative"] += int(negative and record.get("_ui5_record_kind") == "crop"
                                                               and bool(record.get("_ui5_sample_gt_global")))
                fixed, changed = corrected_record(record)
                validate_answer(fixed)
                fixed_message, fixed_key = assistant_slot(fixed)
                report["ref_negative_after"] += int(negative and "<ref>" in fixed_message[fixed_key].lower())
                report["modified_answers"] += int(changed)
                if changed and len(report["modified_examples"]) < 5:
                    report["modified_sample_ids"].append(sample_id(record))
                    report["modified_examples"].append({"sample_id": sample_id(record), "crop_id": record.get("_ui5_crop_id"),
                                                        "before": answer[key], "after": answer_for_record(fixed)})
                writer.write((json.dumps(fixed, ensure_ascii=False) + "\n").encode() if changed else line)
                if report["records"] % 10000 == 0:
                    elapsed = time.monotonic() - started
                    eta = elapsed * max(0, expected_records - report["records"]) / report["records"]
                    print(f"[TEXT PROGRESS] pool={pool} records={report['records']}/{expected_records} "
                          f"percent={100 * report['records'] / expected_records:.1f}% elapsed={elapsed:.0f}s "
                          f"eta={eta:.0f}s changed={report['modified_answers']}", flush=True)
            writer.flush()
            os.fsync(writer.fileno())
        if report["records"] != manifest["pools"][pool]["training_records"]:
            raise ValueError(f"source record count differs from frozen curriculum manifest: {pool}")
        report.update({"source": str(source / filename), "source_sha256": sha(source / filename),
                       "annotation": str(destination / filename), "annotation_sha256": sha(destination / filename)})
        reports.append(report)
        print(f"[TEXT FIX] pool={pool} records={report['records']} ref_negative={report['ref_negative_before']}->0 "
              f"modified={report['modified_answers']} PNG_generation=0", flush=True)
    for entry in recipe.values():
        old_root = Path(entry.get("root") or ".")
        # Image fields and their original resolution do not change.
        if not old_root.is_absolute():
            old_root = source / old_root if entry.get("paths_relative_to_meta") else Path.cwd() / old_root
        entry["root"] = str(old_root.resolve())
        entry["annotation"] = [str(destination / f"{entry['curriculum_pool']}.jsonl")]
        entry["paths_relative_to_meta"] = True
    write_json(destination / RECIPE, recipe)
    for name in SIDECARS:
        shutil.copyfile(source / name, destination / name)
    write_json(destination / "supervision_format.json", {"format_version": FORMAT_VERSION, "pools": reports,
                                                         "scope": "train-only answer text; unchanged IDs, GT, counts and PNGs"})
    revised = copy.deepcopy(manifest)
    revised.pop("identity_digest")
    revised["text_revision"] = identity
    revised["answer_format_version"] = FORMAT_VERSION
    revised["outputs"].update({"recipe": str(destination / RECIPE),
        "hard_groups": str(destination / "hard_groups.jsonl"),
        "matched_anchor_groups": str(destination / "matched_anchor_groups.jsonl"),
        "crop_assets_manifest": str(destination / "crop_assets.jsonl"),
        "crop_asset_root": manifest.get("outputs", {}).get("crop_asset_root", str(source)),
        "supervision_format": str(destination / "supervision_format.json")})
    revised["training_text"] = {"recipe_sha256": sha(destination / RECIPE),
                               "annotations": {p: metadata(destination / f"{p}.jsonl") for p in POOLS}}
    revised["identity_digest"] = digest(revised)
    write_json(destination / "curriculum_manifest.json", revised)
    new_success = copy.deepcopy(success)
    for name in (RECIPE, *SIDECARS, "supervision_format.json", *(f"{p}.jsonl" for p in POOLS)):
        new_success["files"][name] = metadata(destination / name)
    new_success.update(identity_digest=revised["identity_digest"], recipe_sha256=sha(destination / RECIPE))
    supervision_audit(destination)
    # Detect source mutation during conversion before declaring success.
    read_publication(source)
    if (sha(source / "curriculum_manifest.json") != identity["source_manifest_sha256"]
            or sha(source / "_SUCCESS.json") != identity["source_success_sha256"]):
        raise ValueError("source publication changed during text conversion")
    write_json(destination / "_SUCCESS.json", new_success)
    verify_revision(destination)
    return destination, revised
