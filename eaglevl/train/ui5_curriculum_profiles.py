"""Named experiment profiles; ratios are sample-group draws, never tile weights."""
import os

PROFILES = {
    "scheduled_v2": ((.60, .25, .15, 1e-6), (.45, .35, .20, 7e-7), (.30, .30, .40, 5e-7)),
    "global_replay_v3": ((.20, .20, .60, 1e-6), (.25, .25, .50, 7e-7), (.30, .30, .40, 5e-7)),
}


def curriculum_phases(name=None):
    name = name or os.environ.get("UI5_CURRICULUM_PROFILE", "scheduled_v2")
    if name not in PROFILES:
        raise ValueError(f"Unknown UI5 curriculum profile: {name}")
    return PROFILES[name]


def profile_env(name):
    phases = curriculum_phases(name)
    return {"UI5_CURRICULUM_PROFILE": name,
            **{key: ",".join(f"{row[index]:.2f}" for row in phases)
               for index, key in enumerate(("HARD_RATIOS", "ANCHOR_RATIOS", "GLOBAL_REPLAY_RATIOS"))},
            "LLM_LRS": "1e-6,7e-7,5e-7"}
