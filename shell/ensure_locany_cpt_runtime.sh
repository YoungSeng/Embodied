#!/usr/bin/env bash
set -Eeuo pipefail

# Run once before CPT training/evaluation starts. The Merlin image contains an
# OpenCV wheel that may need libGL.so.1 from the task container's OS packages.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_DIR="${ENV_DIR:?ENV_DIR must point to the LocateAnything Python environment}"
INSTALL_SYSTEM_RUNTIME_DEPS="${INSTALL_SYSTEM_RUNTIME_DEPS:-1}"

preflight=(
  "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/preflight_locany_runtime.py"
  --processor-path "${ENV_DIR}"
  --skip-processor
)

preflight_code=0
if "${preflight[@]}"; then
  echo "CPT_RUNTIME_PREFLIGHT=PASSED"
  exit 0
else
  preflight_code=$?
fi

if (( preflight_code != 42 )); then
  echo "ERROR: CPT runtime preflight failed with exit code ${preflight_code}; not attempting apt install" >&2
  exit "${preflight_code}"
fi
if [[ "${INSTALL_SYSTEM_RUNTIME_DEPS}" != "1" ]]; then
  echo "ERROR: libGL.so.1 is missing and INSTALL_SYSTEM_RUNTIME_DEPS=${INSTALL_SYSTEM_RUNTIME_DEPS}" >&2
  echo "Set INSTALL_SYSTEM_RUNTIME_DEPS=1 or install libgl1 libglib2.0-0 in this task container." >&2
  exit 32
fi
if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: libGL.so.1 is missing, but apt-get is unavailable in this task container" >&2
  exit 32
fi

command_prefix=()
if (( EUID != 0 )); then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: libGL.so.1 is missing; task user is not root and sudo is unavailable" >&2
    exit 32
  fi
  command_prefix=(sudo -n)
fi

echo "[CPT RUNTIME] Installing libgl1 and libglib2.0-0 in the current task container"
"${command_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update
"${command_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  --no-install-recommends libgl1 libglib2.0-0
if command -v ldconfig >/dev/null 2>&1; then
  "${command_prefix[@]}" ldconfig
fi

if "${preflight[@]}"; then
  echo "CPT_RUNTIME_PREFLIGHT=PASSED_AFTER_INSTALL"
else
  preflight_code=$?
  echo "ERROR: system dependency installation completed, but cv2 still fails to import" >&2
  exit "${preflight_code}"
fi
