"""Independent alignment-context experiment paths; no data-policy changes."""
from pathlib import Path
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
