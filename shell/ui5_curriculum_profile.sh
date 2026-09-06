#!/usr/bin/env bash
# Source before resolving ratio defaults. Keep v2 usable; v3 must be explicit.
UI5_CURRICULUM_PROFILE="${UI5_CURRICULUM_PROFILE:-scheduled_v2}"
case "${UI5_CURRICULUM_PROFILE}" in
  scheduled_v2)
    PROFILE_HARD_RATIOS=0.60,0.45,0.30
    PROFILE_ANCHOR_RATIOS=0.25,0.35,0.30
    PROFILE_GLOBAL_REPLAY_RATIOS=0.15,0.20,0.40
    ;;
  global_replay_v3)
    PROFILE_HARD_RATIOS=0.20,0.25,0.30
    PROFILE_ANCHOR_RATIOS=0.20,0.25,0.30
    PROFILE_GLOBAL_REPLAY_RATIOS=0.60,0.50,0.40
    ;;
  *) echo "Unknown UI5_CURRICULUM_PROFILE=${UI5_CURRICULUM_PROFILE}" >&2; exit 20 ;;
esac
export UI5_CURRICULUM_PROFILE
