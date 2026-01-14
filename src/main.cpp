#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/core_coord.hpp>

#include "blake3.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kRows = 16;
constexpr uint32_t kCols = 16;
constexpr uint32_t kK = 50240;
constexpr size_t kSeedSize = 240;
constexpr size_t kABytes = static_cast<size_t>(kRows) * kK;
constexpr size_t kBBytes = static_cast<size_t>(kK) * kCols;
constexpr size_t kB2Bytes = 16 * 64;
constexpr size_t kABBytes = kABytes + kBBytes;
constexpr size_t kABWithB2Bytes = kABBytes + kB2Bytes;

struct Options {
    std::string seed_file;
    std::string seed_hex;
    std::string write_sol;
    std::string kernel_dir = "kernels";
    int device_id = 0;
    int iters = 10;
    int warmup = 2;
    bool include_io = false;
    bool read_output = false;
    bool run_gpu = true;
    bool kernel_dir_set = false;
};

struct GpuMatrices {
    std::vector<bfloat16> a_tiled;
    std::vector<bfloat16> b_tiled;
    uint32_t m_pad = 32;
    uint32_t n_pad = 32;
    uint32_t k = kK;
};

struct BenchStats {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double gops = 0.0;
};

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --seed-file <path> [options]\n"
        << "       " << argv0 << " --seed-hex <hex> [options]\n\n"
        << "Options:\n"
        << "  --seed-file <path>   240-byte seed or seed+matrix A/B payload\n"
        << "  --seed-hex <hex>     240-byte seed as hex string\n"
        << "  --write-sol <path>   write solution (seed + C) to file\n"
        << "  --kernel-dir <path>  kernel source directory (default: ./kernels)\n"
        << "  --device-id <id>     device id (default: 0)\n"
        << "  --iters <n>          benchmark iterations (default: 10)\n"
        << "  --warmup <n>         warmup iterations (default: 2)\n"
        << "  --include-io         include output readback in timing\n"
        << "  --read-output        read output once after benchmarking\n"
        << "  --no-gpu             skip GPU benchmark\n"
        << "  -h, --help           show this help\n";
}

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "Error: " << msg << "\n";
    std::exit(1);
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed-file" && i + 1 < argc) {
            opt.seed_file = argv[++i];
        } else if (arg == "--seed-hex" && i + 1 < argc) {
            opt.seed_hex = argv[++i];
        } else if (arg == "--write-sol" && i + 1 < argc) {
            opt.write_sol = argv[++i];
        } else if (arg == "--kernel-dir" && i + 1 < argc) {
            opt.kernel_dir = argv[++i];
            opt.kernel_dir_set = true;
        } else if (arg == "--device-id" && i + 1 < argc) {
            opt.device_id = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            opt.iters = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            opt.warmup = std::stoi(argv[++i]);
        } else if (arg == "--include-io") {
            opt.include_io = true;
        } else if (arg == "--read-output") {
            opt.read_output = true;
        } else if (arg == "--no-gpu") {
            opt.run_gpu = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            die("Unknown or incomplete argument: " + arg);
        }
    }

    if (!opt.kernel_dir_set) {
        if (const char* env = std::getenv("TT_MATMUL_KERNEL_DIR")) {
            opt.kernel_dir = env;
        }
    }

    if (opt.seed_file.empty() == opt.seed_hex.empty()) {
        die("Provide exactly one of --seed-file or --seed-hex");
    }

    if (opt.iters <= 0) {
        die("--iters must be > 0");
    }

    if (opt.warmup < 0) {
        die("--warmup must be >= 0");
    }

    return opt;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        die("Failed to open file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    if (size < 0) {
        die("Failed to read file size: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()), size);
    }
    return data;
}

void write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        die("Failed to open output file: " + path);
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    out.flush();
    if (!out) {
        die("Failed to write output file: " + path);
    }
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (c - 'A');
    }
    return -1;
}

std::vector<uint8_t> parse_hex(const std::string& hex) {
    std::string trimmed = hex;
    if (trimmed.rfind("0x", 0) == 0 || trimmed.rfind("0X", 0) == 0) {
        trimmed = trimmed.substr(2);
    }
    if (trimmed.size() % 2 != 0) {
        die("Hex string must have even length");
    }
    std::vector<uint8_t> out(trimmed.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        int hi = hex_value(trimmed[2 * i]);
        int lo = hex_value(trimmed[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            die("Invalid hex character");
        }
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

void derive_matrices_from_seed(
    const std::vector<uint8_t>& seed,
    std::vector<uint8_t>& a,
    std::vector<int8_t>& b) {
    if (seed.size() != kSeedSize) {
        die("Seed must be 240 bytes");
    }

    std::vector<uint8_t> out(kABWithB2Bytes);
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, seed.data(), seed.size());
    blake3_hasher_finalize_xof(&hasher, out.data(), out.size());

    a.assign(out.begin(), out.begin() + kABytes);
    b.resize(kBBytes);
    for (size_t i = 0; i < kBBytes; ++i) {
        b[i] = static_cast<int8_t>(out[kABytes + i]);
    }
}

void load_seed_and_matrices(
    const Options& opt,
    std::vector<uint8_t>& seed,
    std::vector<uint8_t>& a,
    std::vector<int8_t>& b) {
    std::vector<uint8_t> input;
    if (!opt.seed_file.empty()) {
        input = read_file(opt.seed_file);
    } else {
        input = parse_hex(opt.seed_hex);
    }

    if (input.size() == kSeedSize) {
        seed = input;
        derive_matrices_from_seed(seed, a, b);
        return;
    }

    if (input.size() == kSeedSize + kABBytes || input.size() == kSeedSize + kABWithB2Bytes) {
        seed.assign(input.begin(), input.begin() + kSeedSize);
        a.assign(input.begin() + kSeedSize, input.begin() + kSeedSize + kABytes);
        b.resize(kBBytes);
        size_t b_offset = kSeedSize + kABytes;
        for (size_t i = 0; i < kBBytes; ++i) {
            b[i] = static_cast<int8_t>(input[b_offset + i]);
        }
        return;
    }

    die("Unexpected seed payload size: " + std::to_string(input.size()));
}

std::vector<int32_t> cpu_matmul(const std::vector<uint8_t>& a, const std::vector<int8_t>& b) {
    std::vector<int32_t> c(kRows * kCols, 0);
    for (uint32_t i = 0; i < kRows; ++i) {
        size_t a_row = static_cast<size_t>(i) * kK;
        for (uint32_t j = 0; j < kCols; ++j) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < kK; ++k) {
                uint8_t a_val = a[a_row + k];
                int8_t b_val = b[static_cast<size_t>(k) * kCols + j];
                sum += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
            }
            c[i * kCols + j] = sum;
        }
    }
    return c;
}

void write_le32(uint8_t* dst, int32_t value) {
    uint32_t v = static_cast<uint32_t>(value);
    dst[0] = static_cast<uint8_t>(v & 0xFF);
    dst[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    dst[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    dst[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}

std::vector<uint8_t> build_solution(const std::vector<uint8_t>& seed, const std::vector<int32_t>& c) {
    std::vector<uint8_t> sol(kSeedSize + (kRows * kCols * sizeof(int32_t)));
    std::memcpy(sol.data(), seed.data(), kSeedSize);
    uint8_t* out = sol.data() + kSeedSize;
    for (size_t i = 0; i < c.size(); ++i) {
        write_le32(out + i * 4, c[i]);
    }
    return sol;
}

GpuMatrices prepare_gpu_matrices(const std::vector<uint8_t>& a, const std::vector<int8_t>& b) {
    GpuMatrices mats;
    mats.m_pad = 32;
    mats.n_pad = 32;
    mats.k = kK;

    std::vector<bfloat16> a_pad(static_cast<size_t>(mats.m_pad) * mats.k, bfloat16(0.0f));
    std::vector<bfloat16> b_pad(static_cast<size_t>(mats.k) * mats.n_pad, bfloat16(0.0f));

    for (uint32_t i = 0; i < kRows; ++i) {
        size_t a_row = static_cast<size_t>(i) * kK;
        for (uint32_t k = 0; k < kK; ++k) {
            a_pad[static_cast<size_t>(i) * mats.k + k] = bfloat16(static_cast<float>(a[a_row + k]));
        }
    }

    for (uint32_t k = 0; k < kK; ++k) {
        size_t b_row = static_cast<size_t>(k) * kCols;
        for (uint32_t j = 0; j < kCols; ++j) {
            b_pad[static_cast<size_t>(k) * mats.n_pad + j] = bfloat16(static_cast<float>(b[b_row + j]));
        }
    }

    mats.a_tiled = tilize_nfaces(a_pad, mats.m_pad, mats.k);
    mats.b_tiled = tilize_nfaces(b_pad, mats.k, mats.n_pad);
    return mats;
}

BenchStats run_gpu_benchmark(const GpuMatrices& mats, const Options& opt) {
    auto kernel_root = std::filesystem::path(opt.kernel_dir);
    auto reader_path = kernel_root / "dataflow" / "reader_single_core_mm.cpp";
    auto writer_path = kernel_root / "dataflow" / "writer_single_core_mm.cpp";
    auto compute_path = kernel_root / "compute" / "mm.cpp";

    if (!std::filesystem::exists(reader_path) || !std::filesystem::exists(writer_path) ||
        !std::filesystem::exists(compute_path)) {
        die("Kernel sources not found under: " + kernel_root.string());
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(opt.device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());

    Program program{};
    CoreCoord core({0, 0});

    uint32_t Mt = mats.m_pad / TILE_HEIGHT;
    uint32_t Kt = mats.k / TILE_WIDTH;
    uint32_t Nt = mats.n_pad / TILE_WIDTH;

    uint32_t tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig buffer_config_A{.size = sizeof(bfloat16) * mats.a_tiled.size()};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = sizeof(bfloat16) * mats.b_tiled.size()};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = sizeof(bfloat16) * mats.m_pad * mats.n_pad};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;

    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(num_output_tiles * tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);

    auto reader_id = tt_metal::CreateKernel(
        program,
        reader_path.string(),
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
        });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    auto writer_id = tt_metal::CreateKernel(
        program,
        writer_path.string(),
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
        });

    std::vector<uint32_t> compute_compile_time_args = {Mt, Kt, Nt};
    tt_metal::CreateKernel(
        program,
        compute_path.string(),
        core,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, Mt, Kt, Nt});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Mt, Nt});

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, mats.a_tiled, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, mats.b_tiled, false);

    workload.add_program(device_range, std::move(program));

    for (int i = 0; i < opt.warmup; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
        cq.finish();
    }

    std::vector<bfloat16> output(static_cast<size_t>(mats.m_pad) * mats.n_pad);
    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(opt.iters));

    for (int i = 0; i < opt.iters; ++i) {
        auto start = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        if (opt.include_io) {
            distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
        } else {
            cq.finish();
        }
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        samples_ms.push_back(ms);
    }

    if (opt.read_output && !opt.include_io) {
        distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
    }

    BenchStats stats;
    double sum = 0.0;
    stats.min_ms = samples_ms.front();
    stats.max_ms = samples_ms.front();
    for (double sample : samples_ms) {
        sum += sample;
        stats.min_ms = std::min(stats.min_ms, sample);
        stats.max_ms = std::max(stats.max_ms, sample);
    }
    stats.avg_ms = sum / samples_ms.size();

    double ops = 2.0 * mats.m_pad * mats.n_pad * mats.k;
    stats.gops = ops / (stats.avg_ms / 1000.0) / 1e9;

    bool pass = mesh_device->close();
    if (!pass) {
        die("Failed to close mesh device");
    }

    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    std::vector<uint8_t> seed;
    std::vector<uint8_t> a;
    std::vector<int8_t> b;

    load_seed_and_matrices(opt, seed, a, b);

    std::cout << "Seed size: " << seed.size() << " bytes\n";
    std::cout << "A bytes: " << a.size() << ", B bytes: " << b.size() << "\n";

    auto cpu_start = std::chrono::steady_clock::now();
    std::vector<int32_t> c = cpu_matmul(a, b);
    auto cpu_end = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU matmul time: " << cpu_ms << " ms\n";

    if (!opt.write_sol.empty()) {
        std::vector<uint8_t> sol = build_solution(seed, c);
        write_file(opt.write_sol, sol);
        std::cout << "Wrote sol to: " << opt.write_sol << " (" << sol.size() << " bytes)\n";
    }

    if (opt.run_gpu) {
        std::cout << "Preparing GPU BF16 matrices (padded to 32x32 tiles)...\n";
        GpuMatrices mats = prepare_gpu_matrices(a, b);
        BenchStats stats = run_gpu_benchmark(mats, opt);
        std::cout << "GPU matmul (" << mats.m_pad << "x" << mats.n_pad << "x" << mats.k << ")\n";
        std::cout << "  avg_ms: " << stats.avg_ms << "\n";
        std::cout << "  min_ms: " << stats.min_ms << "\n";
        std::cout << "  max_ms: " << stats.max_ms << "\n";
        std::cout << "  gops:   " << stats.gops << "\n";
        if (!opt.include_io) {
            std::cout << "Timing excludes output readback (--include-io to include).\n";
        }
    }

    return 0;
}
