#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
TT_METAL_REF="${TT_METAL_REF:-main}"
TT_METAL_ROOT="${TT_METAL_ROOT:-${ROOT_DIR}/.tt-metal}"
TT_METAL_SRC="${TT_METAL_SRC:-${TT_METAL_ROOT}/src}"
TT_METAL_BUILD_DIR="${TT_METAL_BUILD_DIR:-${TT_METAL_ROOT}/build}"
TT_METAL_INSTALL_PREFIX="${TT_METAL_INSTALL_PREFIX:-${TT_METAL_ROOT}/install}"

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
  apt-get update -y
  apt-get install -y git cmake ninja-build build-essential python3 python3-venv
fi

if [ ! -d "${TT_METAL_SRC}/.git" ]; then
  mkdir -p "${TT_METAL_SRC}"
  git clone --depth 1 --branch "${TT_METAL_REF}" https://github.com/tenstorrent/tt-metal.git "${TT_METAL_SRC}"
else
  git -C "${TT_METAL_SRC}" fetch --depth 1 origin "${TT_METAL_REF}"
  git -C "${TT_METAL_SRC}" checkout "${TT_METAL_REF}"
  git -C "${TT_METAL_SRC}" reset --hard "origin/${TT_METAL_REF}"
fi

pushd "${TT_METAL_SRC}" >/dev/null
./build_metal.sh \
  --build-dir "${TT_METAL_BUILD_DIR}" \
  --install-prefix "${TT_METAL_INSTALL_PREFIX}" \
  --release \
  --without-python-bindings
popd >/dev/null

TT_METALIUM_CONFIG="$(find "${TT_METAL_INSTALL_PREFIX}" -name TT-MetaliumConfig.cmake -print -quit 2>/dev/null || true)"
if [ -z "${TT_METALIUM_CONFIG}" ]; then
  TT_METALIUM_CONFIG="$(find "${TT_METAL_INSTALL_PREFIX}" -name tt-metalium-config.cmake -print -quit 2>/dev/null || true)"
fi
if [ -z "${TT_METALIUM_CONFIG}" ]; then
  echo "TT-MetaliumConfig.cmake not found under ${TT_METAL_INSTALL_PREFIX}" >&2
  exit 1
fi

TT_METALIUM_DIR="$(dirname "${TT_METALIUM_CONFIG}")"
export TT_METAL_HOME="${TT_METAL_SRC}"

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" -DCMAKE_BUILD_TYPE=Release -DTT-Metalium_DIR="${TT_METALIUM_DIR}"
cmake --build "${ROOT_DIR}/build" -j
