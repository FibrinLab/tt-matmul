ARG TT_PLATFORM=linux/amd64
ARG TT_BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
ARG TT_METAL_REF=main
ARG TT_METAL_SRC=/opt/tt-metal-src
ARG TT_METAL_BUILD=/opt/tt-metal-build
ARG TT_METAL_INSTALL=/opt/tt-metal

FROM --platform=$TT_PLATFORM ${TT_BASE_IMAGE}

ARG TT_METAL_REF
ARG TT_METAL_SRC
ARG TT_METAL_BUILD
ARG TT_METAL_INSTALL

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y git cmake ninja-build build-essential python3 python3-venv \
      libfmt-dev nlohmann-json3-dev libspdlog-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recurse-submodules --shallow-submodules --branch "${TT_METAL_REF}" \
    https://github.com/tenstorrent/tt-metal.git "${TT_METAL_SRC}"

WORKDIR ${TT_METAL_SRC}
RUN ./build_metal.sh \
    --build-dir "${TT_METAL_BUILD}" \
    --install-prefix "${TT_METAL_INSTALL}" \
    --release \
    --without-python-bindings

WORKDIR /app
COPY . /app

ENV TT_METAL_HOME=${TT_METAL_SRC}
RUN TT_METALIUM_CONFIG="$(find "${TT_METAL_INSTALL}" -name TT-MetaliumConfig.cmake -print -quit 2>/dev/null)" && \
    if [ -z "${TT_METALIUM_CONFIG}" ]; then \
      TT_METALIUM_CONFIG="$(find "${TT_METAL_INSTALL}" -name tt-metalium-config.cmake -print -quit 2>/dev/null)"; \
    fi && \
    if [ -z "${TT_METALIUM_CONFIG}" ]; then \
      echo "TT-MetaliumConfig.cmake not found under ${TT_METAL_INSTALL}" >&2; \
      exit 1; \
    fi && \
    TT_METALIUM_DIR="$(dirname "${TT_METALIUM_CONFIG}")" && \
    cmake -S /app -B /app/build -DCMAKE_BUILD_TYPE=Release \
      -DTT-Metalium_DIR="${TT_METALIUM_DIR}" \
      -DCMAKE_PREFIX_PATH="${TT_METAL_INSTALL}" && \
    cmake --build /app/build -j

ENV TT_MATMUL_KERNEL_DIR=/app/kernels
ENTRYPOINT ["/app/build/tt_matmul_bench"]
