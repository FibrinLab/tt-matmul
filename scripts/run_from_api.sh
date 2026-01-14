#!/usr/bin/env bash
set -euo pipefail

api_base="${API_BASE:-${1:-}}"
if [[ -z "${api_base}" ]]; then
  echo "usage: API_BASE=https://host ${0} [bench args]" >&2
  echo "   or: ${0} https://host [bench args]" >&2
  exit 1
fi

if [[ "${1:-}" == "${api_base}" ]]; then
  shift
fi

seed_path="${SEED_PATH:-/api/upow/seed}"
validate_path="${VALIDATE_PATH:-/api/upow/validate}"
work_dir="${WORK_DIR:-/tmp/tt-matmul}"

bin="${TT_MATMUL_BIN:-}"
if [[ -z "${bin}" ]]; then
  if [[ -x "./tt_matmul_bench" ]]; then
    bin="./tt_matmul_bench"
  elif [[ -x "./build/tt_matmul_bench" ]]; then
    bin="./build/tt_matmul_bench"
  else
    echo "Could not find tt_matmul_bench. Set TT_MATMUL_BIN to its path." >&2
    exit 1
  fi
fi

mkdir -p "${work_dir}"
seed_file="${work_dir}/seed.bin"
sol_file="${work_dir}/sol.bin"

curl -sS -o "${seed_file}" "${api_base%/}${seed_path}"

"${bin}" --seed-file "${seed_file}" --write-sol "${sol_file}" "$@"

curl -sS -X POST --data-binary @"${sol_file}" "${api_base%/}${validate_path}"
