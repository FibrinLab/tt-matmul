ARG TT_BASE_IMAGE=ghcr.io/tenstorrent/tt-metal:latest

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
