#!/usr/bin/env bash
# Shared deterministic CANN environment discovery. Source; do not execute.

uavflow_select_cann_env() {
  local candidate=""
  if [[ -n "${CANN_ROOT:-}" ]]; then
    if [[ -f "${CANN_ROOT}/set_env.sh" ]]; then
      candidate="${CANN_ROOT}/set_env.sh"
    elif [[ -f "${CANN_ROOT}/bin/set_env.sh" ]]; then
      candidate="${CANN_ROOT}/bin/set_env.sh"
    else
      local -a explicit_candidates=()
      while IFS= read -r path; do
        explicit_candidates+=("$(readlink -f "${path}")")
      done < <(find "${CANN_ROOT}" -maxdepth 5 -type f -name set_env.sh -print 2>/dev/null)
      mapfile -t explicit_candidates < <(printf '%s\n' "${explicit_candidates[@]}" | sed '/^$/d' | sort -u)
      if (( ${#explicit_candidates[@]} != 1 )); then
        echo "CANN_ROOT=${CANN_ROOT} must contain exactly one set_env.sh; found ${#explicit_candidates[@]}" >&2
        printf '  %s\n' "${explicit_candidates[@]}" >&2
        return 4
      fi
      candidate="${explicit_candidates[0]}"
    fi
  else
    local -a search_roots=(/usr/local/Ascend "${HOME}/Ascend")
    if [[ -n "${ASCEND_SEARCH_ROOT:-}" ]]; then
      IFS=':' read -r -a configured_roots <<< "${ASCEND_SEARCH_ROOT}"
      search_roots+=("${configured_roots[@]}")
    fi
    local -a candidates=()
    local root
    for root in "${search_roots[@]}"; do
      [[ -d "${root}" ]] || continue
      while IFS= read -r path; do
        candidates+=("$(readlink -f "${path}")")
      done < <(find "${root}" -maxdepth 5 -type f -name set_env.sh -print 2>/dev/null)
    done
    mapfile -t candidates < <(printf '%s\n' "${candidates[@]}" | sed '/^$/d' | sort -u)
    if (( ${#candidates[@]} == 0 )); then
      echo "No CANN Toolkit set_env.sh found under /usr/local/Ascend, ${HOME}/Ascend, or ASCEND_SEARCH_ROOT" >&2
      return 2
    fi
    if (( ${#candidates[@]} > 1 )); then
      echo "Multiple CANN installations found; export CANN_ROOT explicitly:" >&2
      printf '  %s\n' "${candidates[@]}" >&2
      return 3
    fi
    candidate="${candidates[0]}"
  fi
  printf '%s\n' "${candidate}"
}
