#!/usr/bin/env bash

locany_report_bash_error() {
  local exit_code="$1"
  local source_file="$2"
  local line_number="$3"
  local failed_command="$4"
  local frame

  trap - ERR
  {
    echo
    printf '[LOCANY FATAL] script=%s line=%s exit_code=%s\n' \
      "${source_file}" "${line_number}" "${exit_code}"
    printf '[LOCANY FATAL] command: %s\n' "${failed_command}"
    printf '[LOCANY FATAL] stack:'
    if (( ${#FUNCNAME[@]} == 1 )); then
      printf ' <top-level>'
    else
      for ((frame = 1; frame < ${#FUNCNAME[@]}; frame++)); do
        printf ' %s(%s:%s)' \
          "${FUNCNAME[frame]}" \
          "${BASH_SOURCE[frame]:-?}" \
          "${BASH_LINENO[frame - 1]:-?}"
      done
    fi
    echo
  } >&2
  exit "${exit_code}"
}

trap 'locany_report_bash_error "$?" "${BASH_SOURCE[0]:-$0}" "$LINENO" "$BASH_COMMAND"' ERR
