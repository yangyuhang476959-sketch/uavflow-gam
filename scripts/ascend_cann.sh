#!/usr/bin/env bash
# Shared deterministic CANN environment discovery. Source; do not execute.

uavflow_select_cann_env() {
  local search_root="${ASCEND_SEARCH_ROOT:-/usr/local/Ascend}"
  local candidate=""
  if [[ -n "${CANN_ROOT:-}" ]]; then
    if [[ -f "${CANN_ROOT}/set_env.sh" ]]; then
      candidate="${CANN_ROOT}/set_env.sh"
    elif [[ -f "${CANN_ROOT}/bin/set_env.sh" ]]; then
      candidate="${CANN_ROOT}/bin/set_env.sh"
    else
      echo "CANN_ROOT=${CANN_ROOT} has no set_env.sh" >&2
      return 2
    fi
  else
    local -a candidates=()
    mapfile -t candidates < <(find "${search_root}" -maxdepth 3 -type f -name set_env.sh -print 2>/dev/null | sort)
    if (( ${#candidates[@]} == 0 )); then
      echo "No CANN set_env.sh found under ${search_root}; export CANN_ROOT" >&2
      return 2
    fi
    if (( ${#candidates[@]} > 1 )); then
      echo "Multiple CANN installations found; export CANN_ROOT explicitly:" >&2
      printf '  %s\n' "${candidates[@]}" >&2
      return 2
    fi
    candidate="${candidates[0]}"
  fi
  printf '%s\n' "${candidate}"
}
