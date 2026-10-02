#!/usr/bin/env bash
# Install/select CANN Toolkit 9.0.0 + matching 910B ops without touching Driver/Firmware.
set -Eeuo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BOOTSTRAP_PY="${ROOT}/scripts/ascend_bootstrap.py"
ASCEND_STACK_MODE="${ASCEND_STACK_MODE:-reference}"
CANN_USER_ROOT="${CANN_USER_ROOT:-${ROOT}/.ascend/cann-9.0.0}"
CANN_PACKAGE_CACHE="${CANN_PACKAGE_CACHE:-${ROOT}/.ascend/packages:${HOME}/.cache/uavflow-ascend}"
RUNTIME_ENV="${ASCEND_RUNTIME_ENV:-${ROOT}/.ascend/runtime.env}"

case "$(uname -m)" in
  aarch64|arm64) ARCH=aarch64 ;;
  x86_64|amd64) ARCH=x86_64 ;;
  *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 2 ;;
esac
[[ "${ASCEND_STACK_MODE}" == reference || "${ASCEND_STACK_MODE}" == vendor ]] || {
  echo "ASCEND_STACK_MODE must be reference or vendor" >&2; exit 2; }
command -v npu-smi >/dev/null || {
  echo 'npu-smi is unavailable. Ask the administrator to install compatible Driver/Firmware.' >&2; exit 2; }

# shellcheck disable=SC1091
source "${ROOT}/scripts/ascend_cann.sh"

cann_json_for_env() {
  local env_file="$1" search_root
  search_root="$(dirname "${env_file}")"
  python3 "${BOOTSTRAP_PY}" cann-version "${search_root}" --arch "${ARCH}" 2>/dev/null
}

download_explicit_url() {
  local url="$1" output="$2"
  [[ "${url}" == https://* ]] || { echo "Only explicit HTTPS package URLs are accepted: ${url}" >&2; return 2; }
  command -v curl >/dev/null || { echo 'curl is required for explicit package downloads.' >&2; return 2; }
  mkdir -p "$(dirname "${output}")"
  curl --fail --location --retry 3 --output "${output}.part" "${url}"
  mv "${output}.part" "${output}"
}

find_package() {
  local variable="$1" expected="$2" explicit_url="$3" value="" cache
  value="${!variable:-}"
  if [[ -n "${value}" ]]; then printf '%s\n' "${value}"; return; fi
  local -a matches=()
  IFS=':' read -r -a caches <<< "${CANN_PACKAGE_CACHE}"
  for cache in "${caches[@]}"; do
    [[ -f "${cache}/${expected}" ]] && matches+=("$(readlink -f "${cache}/${expected}")")
  done
  mapfile -t matches < <(printf '%s\n' "${matches[@]}" | sed '/^$/d' | sort -u)
  (( ${#matches[@]} <= 1 )) || {
    echo "Multiple ${expected} files found; set ${variable} explicitly: ${matches[*]}" >&2; return 2; }
  if (( ${#matches[@]} == 1 )); then printf '%s\n' "${matches[0]}"; return; fi
  if [[ -n "${explicit_url}" ]]; then
    local destination="${ROOT}/.ascend/packages/${expected}"
    download_explicit_url "${explicit_url}" "${destination}" >&2
    printf '%s\n' "${destination}"
    return
  fi
  return 1
}

verify_checksum_if_given() {
  local package="$1" expected="$2" label="$3" actual
  [[ -n "${expected}" ]] || return 0
  actual="$(sha256sum "${package}" | awk '{print $1}')"
  [[ "${actual}" == "${expected,,}" ]] || {
    echo "${label} SHA-256 mismatch: expected ${expected}, got ${actual}" >&2; return 2; }
}

install_reference_cann() {
  local toolkit_name ops_name toolkit ops
  toolkit_name="$(python3 "${BOOTSTRAP_PY}" cann-installer-name --arch "${ARCH}")"
  ops_name="$(python3 "${BOOTSTRAP_PY}" cann-ops-installer-name --arch "${ARCH}")"
  toolkit="$(find_package CANN_TOOLKIT_INSTALLER "${toolkit_name}" "${CANN_TOOLKIT_URL:-}")" || {
    echo "Download the official Huawei CANN 9.0.0 package ${toolkit_name} from https://www.hiascend.com/cann/download" >&2
    echo "Then set CANN_TOOLKIT_INSTALLER or place it under one of: ${CANN_PACKAGE_CACHE}" >&2; return 2; }
  ops="$(find_package CANN_OPS_INSTALLER "${ops_name}" "${CANN_OPS_URL:-}")" || {
    echo "Download the matching official Atlas A2/910B package ${ops_name} from https://www.hiascend.com/cann/download" >&2
    echo "Then set CANN_OPS_INSTALLER or place it under one of: ${CANN_PACKAGE_CACHE}" >&2; return 2; }
  python3 - <<PY
import sys
from pathlib import Path
sys.path.insert(0, '${ROOT}/scripts')
from ascend_bootstrap import validate_cann_installer, validate_cann_ops_installer
validate_cann_installer(Path('${toolkit}'), arch='${ARCH}')
validate_cann_ops_installer(Path('${ops}'), arch='${ARCH}')
PY
  verify_checksum_if_given "${toolkit}" "${CANN_TOOLKIT_SHA256:-}" Toolkit
  verify_checksum_if_given "${ops}" "${CANN_OPS_SHA256:-}" Ops
  mkdir -p "${CANN_USER_ROOT}"
  chmod u+x "${toolkit}" "${ops}"
  echo "Installing CANN Toolkit and 910B ops into ${CANN_USER_ROOT}; administrator installations are untouched."
  "${toolkit}" --quiet --install --install-path="${CANN_USER_ROOT}" ${CANN_TOOLKIT_INSTALLER_ARGS:-}
  "${ops}" --quiet --install --install-path="${CANN_USER_ROOT}" ${CANN_OPS_INSTALLER_ARGS:-}
}

select_reference_env() {
  local env_file json version
  local -a exact=()
  # Project-local reference installation is deterministic and wins when valid.
  if [[ -d "${CANN_USER_ROOT}" ]]; then
    while IFS= read -r env_file; do
      json="$(cann_json_for_env "${env_file}" || true)"
      [[ -n "${json}" ]] || continue
      version="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["version"])' "${json}")"
      [[ "${version}" == 9.0.0 ]] && exact+=("${env_file}")
    done < <(find "${CANN_USER_ROOT}" -maxdepth 6 -type f -name set_env.sh -print 2>/dev/null)
  fi
  if (( ${#exact[@]} == 0 )) && [[ -n "${CANN_ROOT:-}" ]]; then
    local explicit
    explicit="$(uavflow_select_cann_env)" || true
    if [[ -n "${explicit}" ]]; then
      json="$(cann_json_for_env "${explicit}" || true)"
      [[ -n "${json}" ]] && version="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["version"])' "${json}")"
      [[ "${version:-}" == 9.0.0 ]] && exact+=("${explicit}")
    fi
  fi
  if (( ${#exact[@]} == 0 )); then
    while IFS= read -r env_file; do
      json="$(cann_json_for_env "${env_file}" || true)"
      [[ -n "${json}" ]] || continue
      version="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["version"])' "${json}")"
      [[ "${version}" == 9.0.0 ]] && exact+=("${env_file}")
    done < <(uavflow_list_cann_envs)
  fi
  mapfile -t exact < <(printf '%s\n' "${exact[@]}" | sed '/^$/d' | xargs -r -n1 readlink -f | sort -u)
  (( ${#exact[@]} <= 1 )) || {
    echo 'Multiple distinct CANN 9.0.0 installations found; set CANN_ROOT explicitly:' >&2
    printf '  %s\n' "${exact[@]}" >&2; return 3; }
  (( ${#exact[@]} == 1 )) && { printf '%s\n' "${exact[0]}"; return; }
  return 1
}

echo '== Ascend Driver/Firmware inventory (read-only) =='
npu-smi info
for info in /usr/local/Ascend/driver/version.info /usr/local/Ascend/firmware/version.info; do
  [[ -f "${info}" ]] && { echo "-- ${info}"; cat "${info}"; }
done

SELECTED_SOURCE=existing
if [[ "${ASCEND_STACK_MODE}" == vendor ]]; then
  CANN_ENV_FILE="$(uavflow_select_cann_env)"
else
  set +e
  CANN_ENV_FILE="$(select_reference_env)"
  selection_status=$?
  set -e
  (( selection_status != 3 )) || exit 3
  if [[ -z "${CANN_ENV_FILE}" ]]; then
    echo 'No reusable CANN 9.0.0 installation found; preserving any administrator CANN and installing project-local reference CANN.'
    install_reference_cann
    CANN_ENV_FILE="$(CANN_ROOT="${CANN_USER_ROOT}" uavflow_select_cann_env)"
    [[ -n "${CANN_ENV_FILE}" ]] || { echo 'CANN installation produced no set_env.sh' >&2; exit 2; }
    SELECTED_SOURCE=project-installed
  fi
fi
CANN_ENV_FILE="$(readlink -f "${CANN_ENV_FILE}")"
CANN_ROOT_SELECTED="$(dirname "${CANN_ENV_FILE}")"
CANN_JSON="$(cann_json_for_env "${CANN_ENV_FILE}")" || {
  echo 'Selected CANN lacks valid official Toolkit metadata.' >&2; exit 2; }
CANN_VERSION_DETECTED="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["version"])' "${CANN_JSON}")"
if [[ "${ASCEND_STACK_MODE}" == reference && "${CANN_VERSION_DETECTED}" != 9.0.0 ]]; then
  echo "Reference mode requires CANN 9.0.0; detected ${CANN_VERSION_DETECTED}." >&2; exit 2
fi
# shellcheck disable=SC1090
source "${CANN_ENV_FILE}"
if [[ "${ASCEND_STACK_MODE}" == reference ]]; then
  ops_found=0
  IFS=':' read -r -a opp_paths <<< "${ASCEND_OPP_PATH:-${CANN_ROOT_SELECTED}/opp}"
  for opp_path in "${opp_paths[@]}"; do [[ -d "${opp_path}" ]] && ops_found=1; done
  (( ops_found == 1 )) || {
    echo 'Matching 910B ops installation was not detected (ASCEND_OPP_PATH/opp missing).' >&2; exit 2; }
fi

mkdir -p "$(dirname "${RUNTIME_ENV}")"
{
  printf 'export CANN_ROOT=%q\n' "${CANN_ROOT_SELECTED}"
  printf 'export CANN_ENV_FILE=%q\n' "${CANN_ENV_FILE}"
  printf 'export CANN_VERSION=%q\n' "${CANN_VERSION_DETECTED}"
  printf 'export CANN_SELECTION_SOURCE=%q\n' "${SELECTED_SOURCE}"
  printf 'export ASCEND_STACK_MODE=%q\n' "${ASCEND_STACK_MODE}"
} > "${RUNTIME_ENV}"
chmod 600 "${RUNTIME_ENV}"
echo "Selected CANN ${CANN_VERSION_DETECTED}: ${CANN_ENV_FILE} (${SELECTED_SOURCE})"
echo "Persisted runtime selection: ${RUNTIME_ENV}"
