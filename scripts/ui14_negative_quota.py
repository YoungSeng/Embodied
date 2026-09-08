"""Separate independent-image availability from the training sampling ratio."""
from copy import deepcopy
from pathlib import Path

from ui14_common import digest, read_json, write_json

POLICIES = ("available", "strict")


def quota_policy(document):
    # Checkpoints/data frozen before this option retain their strict contract.
    value = document.get("negative_quota_policy", "strict")
    if value not in POLICIES:
        raise ValueError(f"Unknown negative quota policy: {value}")
    return value


def inventory_policy_fields(targets, policy):
    quota_policy({"negative_quota_policy": policy})
    gap = sum(t["gap"] for t in targets.values())
    return {"negative_quota_policy": policy,
            "gap_blocks_normalize": policy == "strict" and gap > 0,
            "sampling_negative_to_positive_ratio": 1.0,
            "evaluation_balance": "observed_counts",
            "zero_negative_splits": [k for k, t in targets.items()
                                     if t["existing_negative_images"] + t["selected_images"] == 0]}


def reuse_inventory_policy(root, inventory, policy):
    """Rebind an unfrozen, verified selection without rescanning its candidates."""
    root = Path(root)
    if (root / "negative_extension_manifest.json").exists():
        return inventory  # Never relabel an already frozen evaluation set.
    quota_policy({"negative_quota_policy": policy})
    if quota_policy(inventory) == policy:
        return inventory
    previous_id = inventory["inventory_id"]
    if digest({k: v for k, v in inventory.items() if k != "inventory_id"}) != previous_id:
        raise ValueError("Cannot migrate an unbound inventory")
    write_json(root / "inventory_history" / (previous_id + ".json"), inventory)
    updated = deepcopy(inventory)
    updated.update(inventory_policy_fields(updated["targets"], policy),
                   parent_inventory_id=previous_id,
                   policy_change="reuse verified proposed selection; labels and page map unchanged")
    updated.pop("inventory_id")
    updated["inventory_id"] = digest(updated)
    write_json(root / "inventory_summary.json", updated)
    print(f"[normalize] reused inventory selection={updated['selected_count']} "
          f"policy={policy} gap={updated['gap']} (独立原图缺口，仅统计); "
          "candidate rescan=0; selection/page map unchanged", flush=True)
    return updated


def image_quota_counts(extension, key, counts):
    """Missing selected images still fail even when an availability gap is allowed."""
    policy = quota_policy(extension)
    target = extension["targets"][key]
    expected = (target["positive_images"],
                target["existing_negative_images"] + target["selected_images"])
    observed = (counts["positive_images"], counts["negative_images"])
    if observed != expected:
        raise ValueError(f"Bound positive/negative image counts changed: {key}: {observed} != {expected}")
    if policy == "strict" and observed[0] != observed[1]:
        raise ValueError(f"Independent image quota not 1:1: {key}: {counts}")
    return {**counts, "negative_quota_policy": policy,
            "negative_to_positive_ratio": observed[1] / observed[0] if observed[0] else None,
            "one_to_one_shortfall": max(0, observed[0] - observed[1]),
            "one_to_one_met": observed[0] == observed[1]}


def extension_quota(root, snapshot):
    extension = read_json(Path(root) / "negative_extension_manifest.json")
    if quota_policy(extension) != quota_policy(snapshot):
        raise ValueError("Snapshot/extension negative quota policy differs")
    return extension
