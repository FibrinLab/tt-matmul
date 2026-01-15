#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-${ROOT_DIR}/.tt-metal/src}"
export TT_MATMUL_KERNEL_DIR="${TT_MATMUL_KERNEL_DIR:-${ROOT_DIR}/kernels}"

exec "${ROOT_DIR}/build/tt_matmul_bench" "$@"
