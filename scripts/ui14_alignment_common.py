"""Independent alignment-context experiment paths; no data-policy changes."""
from pathlib import Path
import os
from ui14_common import WORKSPACE

PROFILE = "m32-cpt9000-ui14-alignment-context-v1"
PROJECT = WORKSPACE + "/code/Eagle_LocateUI5_v4/Embodied-ui14-alignment-context"
DATA = WORKSPACE + "/gui_data/ui14_alignment_context_v1"
OUTPUT = WORKSPACE + "/gui_models/locany-m32-cpt9000-ui14-alignment-context-a800x4-v1"
NEG_DATA = WORKSPACE + "/gui_data/ui14_cpt9000_neg11_v1"
NEG_OUTPUT = WORKSPACE + "/gui_models/locany-m32-cpt9000-ui14-neg11-a800x4-v1"
OLD_OUTPUT = WORKSPACE + "/gui_models/locany-m32-cpt9000-ui14-a800x4-repair-v2"
OLD_MANIFEST = WORKSPACE + "/gui_data/ui14_cpt9000_repair_v2/evaluation_manifest.json"


def independent_output(destination, *sources):
    destination = Path(destination).resolve()
    for source in sources:
        source = Path(source).resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError(f"Output overlaps read-only source: {destination} / {source}")
    return destination


def alignment_default_roots(env=None):
    """Old experiments' generic UI14 variables cannot select this output."""
    env = os.environ if env is None else env
    workspace = env.get("WORKSPACE", WORKSPACE)
    return (env.get("UI14_ALIGNMENT_DATA_ROOT") or workspace + "/gui_data/ui14_alignment_context_v1",
            env.get("UI14_ALIGNMENT_PARENT_DATA_ROOT") or workspace + "/gui_data/ui14_cpt9000_neg11_v1")


def alignment_destination(destination, parent, old_run=OLD_OUTPUT, output=OUTPUT):
    root = independent_output(destination, parent, old_run, output, NEG_DATA,
                              Path(OLD_MANIFEST).parent, NEG_OUTPUT, OLD_OUTPUT)
    if any((root / name).is_file() for name in ("negative_extension_manifest.json", "source_snapshot.json")) and not (root / "frozen_parent.json").is_file():
        raise ValueError(f"Refusing to write into an existing UI14 dataset: {root}; "
                         "use a new alignment --data-root")
    return root


def require_neg11_parent(parent):
    """Read-only preflight, before logs/locks/journals create an output."""
    parent = Path(parent).resolve()
    required = ("cpu_check_report.json", "source_snapshot.json", "negative_extension_manifest.json",
                "normalization_stats.json", "evaluation_manifest.json")
    missing = [name for name in required if not (parent / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Alignment parent must be the completed neg11 available dataset. "
            f"parent={parent}; missing={', '.join(missing)}. "
            f"repair-v2 is only the historical evaluation input. "
            f"Set --parent-root or UI14_ALIGNMENT_PARENT_DATA_ROOT (default: {NEG_DATA}); "
            "UI14_PARENT_DATA_ROOT is not used by this entry.")
    return parent
