#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
exec bash "${ROOT}/experiments/uavflow_remote_ablation/cloud_jobs_10/run_one_8gpu.sh" D1LOG
