#!/usr/bin/env bash
set -Eeuo pipefail
# Run from any imported checkout of codex/ui5-crop-grpo-mixed-v1. No data leaves
# the internal host. The submitter switches to the recorded conda interpreter.
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_ROOT=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Embodied-ui5-crop-grpo-mixed-v1
REVISION="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor b29590f88c4b4a102c742f8410e1c6751b0859d2 "${REVISION}"
if [[ ! -d "${TARGET_ROOT}" ]]; then
  git -C "${SOURCE_ROOT}" worktree add --detach "${TARGET_ROOT}" "${REVISION}"
fi
test "$(git -C "${TARGET_ROOT}" rev-parse HEAD)" = "${REVISION}"
cd "${TARGET_ROOT}"
exec python scripts/ui5_grpo_bootstrap.py --submit "$@"
