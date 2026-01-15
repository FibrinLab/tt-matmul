ARG TT_PLATFORM=linux/amd64
ARG TT_BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-dev-amd64:aa43a5fc7d3f0e4ed0142d9bd1357912d9a2a5f7

FROM --platform=$TT_PLATFORM ${TT_BASE_IMAGE} AS build
WORKDIR /src
COPY . .
ARG TT_METALIUM_DIR=
ARG TT_METAL_HOME_OVERRIDE=
RUN TT_METALIUM_DIR="${TT_METALIUM_DIR}" && \
    TT_METAL_HOME_EFFECTIVE="${TT_METAL_HOME_OVERRIDE:-${TT_METAL_HOME}}" && \
    if [ -n "${TT_METAL_HOME_EFFECTIVE}" ]; then export TT_METAL_HOME="${TT_METAL_HOME_EFFECTIVE}"; fi && \
    if [ -z "${TT_METALIUM_DIR}" ]; then \
      TT_METALIUM_CONFIG="$(find / -name TT-MetaliumConfig.cmake -print -quit 2>/dev/null)" && \
      if [ -z "${TT_METALIUM_CONFIG}" ]; then \
        TT_METALIUM_CONFIG="$(find / -name tt-metalium-config.cmake -print -quit 2>/dev/null)"; \
      fi && \
      if [ -n "${TT_METALIUM_CONFIG}" ]; then \
        TT_METALIUM_DIR="$(dirname "${TT_METALIUM_CONFIG}")"; \
      fi; \
    fi && \
    echo "TT_METAL_HOME=${TT_METAL_HOME}" && \
    echo "TT_METALIUM_DIR=${TT_METALIUM_DIR}" && \
    if [ -n "${TT_METALIUM_DIR}" ]; then \
      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTT-Metalium_DIR="${TT_METALIUM_DIR}"; \
    else \
      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release; \
    fi
RUN cmake --build build -j

FROM --platform=$TT_PLATFORM ${TT_BASE_IMAGE} AS runtime
WORKDIR /app
COPY --from=build /src/build/tt_matmul_bench /app/tt_matmul_bench
COPY kernels /app/kernels
COPY scripts /app/scripts
ENV TT_MATMUL_KERNEL_DIR=/app/kernels
ENTRYPOINT ["/app/tt_matmul_bench"]
