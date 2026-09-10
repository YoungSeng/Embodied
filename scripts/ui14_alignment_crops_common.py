"""Paths for the independent neg11/alignment-crops/production-decoder experiment."""
from pathlib import Path
import os
from ui14_common import WORKSPACE
from ui14_alignment_common import DATA as CONTEXT_DATA, OUTPUT as CONTEXT_OUTPUT, independent_output

PROFILE = "m32-cpt9000-ui14-neg11-alignment-crops-v1"
PROJECT = WORKSPACE + "/code/Eagle_LocateUI5_v4/Embodied-ui14-neg11-alignment-crops"
DATA = WORKSPACE + "/gui_data/ui14_neg11_alignment_crops_v1"
OUTPUT = WORKSPACE + "/gui_models/locany-m32-cpt9000-ui14-neg11-alignment-crops-a800x4-v1"


def default_roots(env=None):
    env = os.environ if env is None else env
    workspace = env.get("WORKSPACE", WORKSPACE)
    return (env.get("UI14_ALIGNMENT_CROPS_DATA_ROOT") or workspace + "/gui_data/ui14_neg11_alignment_crops_v1",
            env.get("UI14_ALIGNMENT_CROPS_PARENT_DATA_ROOT") or workspace + "/gui_data/ui14_cpt9000_neg11_v1")


def protect_context(root, output):
    for target in (root, output):
        independent_output(target, CONTEXT_DATA, CONTEXT_OUTPUT)
    return Path(root)
