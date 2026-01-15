ARG TT_BASE_IMAGE=docker build -t tt-matmul --build-arg TT_BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-dev-amd64:aa43a5fc7d3f0e4ed0142d9bd1357912d9a2a5f7

FROM ${TT_BASE_IMAGE} AS build
WORKDIR /src
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j

FROM ${TT_BASE_IMAGE} AS runtime
WORKDIR /app
COPY --from=build /src/build/tt_matmul_bench /app/tt_matmul_bench
COPY kernels /app/kernels
COPY scripts /app/scripts
ENV TT_MATMUL_KERNEL_DIR=/app/kernels
ENTRYPOINT ["/app/tt_matmul_bench"]
