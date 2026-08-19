#!/usr/bin/env bash

locany_print_bash_stack() {
  local frame
  printf '[LOCANY FATAL] stack:'
  if (( ${#FUNCNAME[@]} <= 2 )); then
    printf ' <top-level>'
  else
    for ((frame = 2; frame < ${#FUNCNAME[@]}; frame++)); do
      printf ' %s(%s:%s)' \
        "${FUNCNAME[frame]}" \
        "${BASH_SOURCE[frame]:-?}" \
        "${BASH_LINENO[frame - 1]:-?}"
    done
  fi
  echo
}

locany_print_log_locations() {
  if [[ -n "${PIPELINE_LOG:-}" ]]; then
    printf '[LOCANY FATAL] combined_log=%s\n' "${PIPELINE_LOG}"
  fi
  if [[ -n "${PIPELINE_TRACE_LOG:-}" ]]; then
    printf '[LOCANY FATAL] command_trace=%s\n' "${PIPELINE_TRACE_LOG}"
  fi
}

locany_report_bash_error() {
  local exit_code="$1"
  local source_file="$2"
  local line_number="$3"
  local failed_command="$4"

  trap - ERR
  set +x
  set +e
  {
    echo
    printf '[LOCANY FATAL] script=%s line=%s exit_code=%s\n' \
      "${source_file}" "${line_number}" "${exit_code}"
    printf '[LOCANY FATAL] command: %s\n' "${failed_command}"
    locany_print_bash_stack
    locany_print_log_locations
  } >&2
  exit "${exit_code}"
}

locany_die() {
  local exit_code="$1"
  local message="$2"
  local source_file="${BASH_SOURCE[1]:-$0}"
  local line_number="${BASH_LINENO[0]:-?}"

  trap - ERR
  set +x
  set +e
  {
    echo
    printf '[LOCANY FATAL] %s\n' "${message}"
    printf '[LOCANY FATAL] script=%s line=%s exit_code=%s\n' \
      "${source_file}" "${line_number}" "${exit_code}"
    locany_print_bash_stack
    locany_print_log_locations
  } >&2
  exit "${exit_code}"
}

trap 'locany_report_bash_error "$?" "${BASH_SOURCE[0]:-$0}" "$LINENO" "$BASH_COMMAND"' ERR
