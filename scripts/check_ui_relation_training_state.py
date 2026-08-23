#!/usr/bin/env python3
"""Compare UI Relation/Gate/PBD parameter updates between two checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


GROUPS = ("relation", "image_gate", "slot_gate", "pbd")


def parameter_group(name: str) -> str | None:
    if "relation_pbd" in name:
        return "pbd"
    if "relation_pyramid.image_gate_heads" in name:
        return "image_gate"
    if "relation_pyramid.gate_heads" in name:
        return "slot_gate"
    if "relation_pyramid" in name:
        return "relation"
    return None


def checkpoint_files(checkpoint: Path) -> dict[str, Path]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = checkpoint / index_name
        if index.is_file():
            value = json.loads(index.read_text(encoding="utf-8"))
            return {
                key: checkpoint / filename
                for key, filename in value.get("weight_map", {}).items()
                if parameter_group(key) is not None
            }
    direct = list(checkpoint.glob("model*.safetensors")) + list(
        checkpoint.glob("pytorch_model*.bin")
    )
    if not direct:
        raise FileNotFoundError(f"No model weights under {checkpoint}")
    mapping: dict[str, Path] = {}
    for path in direct:
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if parameter_group(key) is not None:
                        mapping[key] = path
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            for key in state:
                if parameter_group(key) is not None:
                    mapping[key] = path
    return mapping


def load_tensor(path: Path, key: str) -> torch.Tensor:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state[key]


def compare(before: Path, after: Path) -> dict[str, Any]:
    before_files = checkpoint_files(before)
    after_files = checkpoint_files(after)
    if set(before_files) != set(after_files):
        raise RuntimeError(
            "UI checkpoint structures differ: "
            f"missing_after={sorted(set(before_files) - set(after_files))}, "
            f"new_after={sorted(set(after_files) - set(before_files))}"
        )
    accum = {
        group: {"delta_sq": 0.0, "base_sq": 0.0, "changed": 0, "elements": 0}
        for group in GROUPS
    }
    for key in sorted(before_files):
        lhs = load_tensor(before_files[key], key).float()
        rhs = load_tensor(after_files[key], key).float()
        if lhs.shape != rhs.shape:
            raise RuntimeError(f"Shape mismatch for {key}: {lhs.shape} != {rhs.shape}")
        delta = rhs - lhs
        group = parameter_group(key)
        accum[group]["delta_sq"] += float(delta.square().sum().item())
        accum[group]["base_sq"] += float(lhs.square().sum().item())
        accum[group]["changed"] += int(delta.ne(0).sum().item())
        accum[group]["elements"] += delta.numel()
    result = {"schema_version": 1, "before": str(before), "after": str(after), "groups": {}}
    for group, values in accum.items():
        absolute = math.sqrt(values["delta_sq"])
        result["groups"][group] = {
            "absolute_update_norm": absolute,
            "relative_update_norm": absolute / (math.sqrt(values["base_sq"]) + 1.0e-12),
            "changed_element_count": values["changed"],
            "element_count": values["elements"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-0", type=Path, required=True)
    parser.add_argument("--checkpoint-n", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-txt", type=Path, default=None)
    args = parser.parse_args()
    report = compare(args.checkpoint_0.resolve(), args.checkpoint_n.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    lines = ["group\tabsolute_update_norm\trelative_update_norm\tchanged/total"]
    for group in GROUPS:
        values = report["groups"][group]
        lines.append(
            f"{group}\t{values['absolute_update_norm']:.9g}\t"
            f"{values['relative_update_norm']:.9g}\t"
            f"{values['changed_element_count']}/{values['element_count']}"
        )
    text = "\n".join(lines) + "\n"
    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text(text, encoding="utf-8")
    print(payload, end="")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
