"""Dataset sampling helpers shared by LocateAnything training recipes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


def resolve_dataset_sampling_weight(
    meta: Mapping[str, Any], dataset_length: int
) -> float:
    """Return a positive stream-sampling weight for one recipe entry.

    ``sampling_weight`` is an explicit task probability weight.  It is useful
    for CPT mixtures where dataset size and desired task frequency are not the
    same thing.  Recipes without it keep LocateAnything's legacy behavior.
    """

    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive")

    explicit = meta.get("sampling_weight")
    if explicit is not None:
        weight = float(explicit)
    else:
        repeat_time = float(meta.get("repeat_time", 1.0))
        weight = repeat_time * dataset_length if repeat_time >= 1.0 else dataset_length

    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"sampling weight must be finite and positive, got {weight!r}")
    return weight


def resolve_recipe_entry_paths(
    meta: Mapping[str, Any], meta_path: str | Path
) -> dict[str, Any]:
    """Resolve opt-in recipe-relative annotation and media-root paths."""

    resolved = dict(meta)
    if not bool(resolved.get("paths_relative_to_meta", False)):
        return resolved
    base = Path(meta_path).expanduser().resolve().parent

    annotations = resolved.get("annotation")
    if annotations is not None:
        is_list = isinstance(annotations, (list, tuple))
        values = list(annotations) if is_list else [annotations]
        normalized = []
        for value in values:
            path = Path(str(value))
            normalized.append(str(path if path.is_absolute() else (base / path).resolve()))
        resolved["annotation"] = normalized if is_list else normalized[0]

    root_value = str(resolved.get("root", ""))
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            resolved["root"] = str((base / root).resolve())
    return resolved
