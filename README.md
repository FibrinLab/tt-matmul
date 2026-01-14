# tt-matmul

Benchmark TT-Metal matmul for Tenstorrent N550S using the uPoW seed format.

This benchmark:
- Derives A (16x50240, u8) and B (50240x16, i8) from a 240-byte seed via BLAKE3 XOF.
- Computes CPU C (16x16, i32) for validation output (sol = seed || C).
- Runs a GPU matmul on BF16 data padded to 32x50240 and 50240x32 (TT tiles are 32x32), for timing.

Note: The GPU path uses BF16 + padding, so its output is not byte-for-byte compatible with the integer C used in
validation. The validation sol is generated from the CPU integer matmul.

## Build

You need TT-Metalium installed and discoverable by CMake.

```bash
export TT_METAL_HOME=/path/to/tt-metal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/tt_matmul_bench --seed-file seed.bin --iters 20 --warmup 5
```

Common options:
- `--seed-file <path>`: 240-byte seed (or 240 + A/B bytes from `/api/upow/seed_with_matrix_a_b`).
- `--seed-hex <hex>`: 240-byte seed as hex.
- `--write-sol <path>`: write sol (seed + 16x16 i32) to file.
- `--kernel-dir <path>`: kernel source dir (default: `./kernels`).
- `--include-io`: include output readback in timing.
- `--read-output`: read output once after benchmarking.
- `--no-gpu`: skip GPU timing (CPU only).
- `--device-id <id>`: device id (default 0).

## API flow (seed + validation)

```bash
scripts/run_from_api.sh https://your-node.example --iters 10 --warmup 2
```

Use the seed-with-matrices endpoint if available:

```bash
SEED_PATH=/api/upow/seed_with_matrix_a_b scripts/run_from_api.sh https://your-node.example
```

This will:
1) fetch the seed,
2) run the benchmark and write `sol.bin`,
3) POST the solution to `/api/upow/validate`.

If the binary is not in the current directory, set `TT_MATMUL_BIN` to its path.

## Docker (Koyeb)

Pick a base image that already includes TT-Metal + build tools.

```bash
docker build -t tt-matmul --build-arg TT_BASE_IMAGE=ghcr.io/tenstorrent/tt-metal:latest .
```

Runtime:
- The image sets `TT_MATMUL_KERNEL_DIR=/app/kernels`.
- Provide a seed file or use `scripts/run_from_api.sh` inside the container.
