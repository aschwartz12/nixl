/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two-process functional and latency benchmark for the GPU D2P -> CPU proxy
 * path. Select LIBFABRIC/EFA or UCX at run time and launch exactly one process
 * on each of two nodes (Slurm rank 0 initiates, rank 1 is the target):
 *
 *   nixl_efa_proxy_benchmark <shared-coordination-dir>
 *
 * Environment:
 *   NIXL_PROXY_BENCH_BACKEND          LIBFABRIC (default) or UCX
 *   NIXL_EFA_PROXY_CHANNELS/WORKERS   proxy channels / threads (default 1 / 1)
 *   NIXL_EFA_PROXY_PIPELINE_WINDOW    outstanding puts in the pipelined tests
 *   NIXL_EFA_PROXY_RING_DEPTH         proxy_ring_depth (power of two, default 256)
 *   NIXL_EFA_PROXY_RAIL_POLICY        efa_proxy_rail_policy: thread (default) or ring
 *   NIXL_EFA_PROXY_IDLE_POLL_US       efa_proxy_idle_poll_us (idle CQ polling period)
 *   NIXL_PROXY_BENCH_HOST_ROUNDS      rounds per host-transfer trial (default 200)
 *   NIXL_PROXY_BENCH_PTHR_DELAY_US    engine progress thread delay (default 1000)
 *   NIXL_PROXY_BENCH_PROGRESS_THREAD  0 disables the engine progress thread
 *   NIXL_PROXY_BENCH_ORDERING_ROUNDS  >0 runs the put -> atomicAdd ordering check
 *                                     (proxy_ordering.cuh) first, per channel
 *   NIXL_PROXY_BENCH_ORDERING_ONLY    1 stops after the ordering check
 *   NIXL_PROXY_BENCH_SKIP_{PINGPONG,SERIAL,PIPELINED}=1 skip that phase
 *   NIXL_PROXY_BENCH_HOST_XFER=1      host-path writes (postXferReq) instead of the
 *                                     device phases
 *   NIXL_PROXY_BENCH_DEVICE_PROXY=0   create the backend without the device proxy
 *                                     (host transfers only; for proxy on/off A/B)
 *   NIXL_PROXY_BENCH_PUT_ONLY, NIXL_PROXY_BENCH_COUNTER_HOST, NIXL_PROXY_BENCH_MIN_SIZE
 */
#include <cuda_runtime.h>
#include <nixl.h>
#include <gpu/nixl_device.cuh>

#include "proxy_ordering.cuh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t kDeviceId = 0;
constexpr uint64_t kSignalValue = 1;
constexpr uint32_t kDefaultPipelineWindow = 32;
constexpr uint32_t kMaxPipelineWindow = 256;
constexpr uint32_t kMaxPipelineChannels = 32;
constexpr auto kTimeout = std::chrono::seconds(90);

struct PeerMetadata {
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0;
    uint64_t counter_addr = 0;
    uint64_t buffer_size = 0;
    uint64_t ordering_addr = 0;
    std::string nixl_blob;
};

struct DeviceResult {
    nixl_status_t status = NIXL_IN_PROG;
    unsigned long long elapsed_ns = 0;
    // GPU time inside nixlPut()/nixlAtomicAdd(), including waits for ring space.
    unsigned long long enqueue_ns = 0;
    unsigned long long enqueue_ops = 0;
};

void
checkCuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void
checkNixl(nixl_status_t status, const char *what) {
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": NIXL status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void
writeFileAtomically(const std::filesystem::path &path, const std::string &payload) {
    const auto tmp = path.string() + ".tmp." + std::to_string(::getpid());
    {
        std::ofstream output(tmp, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot open " + tmp);
        }
        output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        if (!output) {
            throw std::runtime_error("Cannot write " + tmp);
        }
    }
    std::filesystem::rename(tmp, path);
}

std::string
readFileWhenReady(const std::filesystem::path &path) {
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream input(path, std::ios::binary);
        if (input) {
            return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("Timed out waiting for " + path.string());
}

template<typename T>
void
appendScalar(std::string &wire, T value) {
    wire.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

template<typename T>
T
readScalar(const std::string &wire, size_t &offset) {
    if (offset + sizeof(T) > wire.size()) {
        throw std::runtime_error("Truncated coordination metadata");
    }
    T value{};
    std::memcpy(&value, wire.data() + offset, sizeof(value));
    offset += sizeof(value);
    return value;
}

std::string
serializePeerMetadata(const PeerMetadata &metadata) {
    std::string wire;
    appendScalar(wire, metadata.src_addr);
    appendScalar(wire, metadata.dst_addr);
    appendScalar(wire, metadata.counter_addr);
    appendScalar(wire, metadata.buffer_size);
    appendScalar(wire, metadata.ordering_addr);
    appendScalar(wire, static_cast<uint64_t>(metadata.nixl_blob.size()));
    wire.append(metadata.nixl_blob);
    return wire;
}

PeerMetadata
deserializePeerMetadata(const std::string &wire) {
    size_t offset = 0;
    PeerMetadata metadata{};
    metadata.src_addr = readScalar<uint64_t>(wire, offset);
    metadata.dst_addr = readScalar<uint64_t>(wire, offset);
    metadata.counter_addr = readScalar<uint64_t>(wire, offset);
    metadata.buffer_size = readScalar<uint64_t>(wire, offset);
    metadata.ordering_addr = readScalar<uint64_t>(wire, offset);
    const uint64_t blob_size = readScalar<uint64_t>(wire, offset);
    if (offset + blob_size != wire.size()) {
        throw std::runtime_error("Invalid coordination metadata length");
    }
    metadata.nixl_blob.assign(wire.data() + offset, blob_size);
    return metadata;
}

__device__ unsigned long long
deviceTimeNs() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(value));
    return value;
}

__device__ nixl_status_t
waitForCompletion(nixlGpuXferStatusH &handle, unsigned long long deadline_ns) {
    nixl_status_t status;
    do {
        status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(handle);
    } while (status == NIXL_IN_PROG && deviceTimeNs() < deadline_ns);
    return status;
}

__global__ void
runPutBenchmark(nixlMemViewElem src,
                nixlMemViewElem dst,
                size_t size,
                uint32_t iterations,
                DeviceResult *result) {
    constexpr unsigned long long kTimeoutNs = 60ULL * 1'000'000'000ULL;
    const unsigned long long start = deviceTimeNs();
    const unsigned long long deadline = start + kTimeoutNs;
    nixl_status_t status = NIXL_SUCCESS;
    unsigned long long enqueue_ns = 0;
    unsigned long long enqueue_ops = 0;

    for (uint32_t i = 0; i < iterations && status == NIXL_SUCCESS; ++i) {
        nixlGpuXferStatusH put_status{};
        enqueue_ns -= deviceTimeNs();
        status = nixlPut<nixl_gpu_level_t::THREAD>(
            src, dst, size, /*channel_id=*/0, /*flags=*/0, &put_status);
        enqueue_ns += deviceTimeNs();
        ++enqueue_ops;
        if (status == NIXL_IN_PROG) {
            status = waitForCompletion(put_status, deadline);
        }
    }

    result->elapsed_ns = deviceTimeNs() - start;
    result->enqueue_ns = enqueue_ns;
    result->enqueue_ops = enqueue_ops;
    result->status = status;
}

__global__ void
runPipelinedPutBenchmark(nixlMemViewElem src,
                         nixlMemViewElem dst,
                         size_t size,
                         uint32_t iterations,
                         uint32_t pipeline_window,
                         DeviceResult *result) {
    constexpr unsigned long long kTimeoutNs = 60ULL * 1'000'000'000ULL;
    const unsigned long long start = deviceTimeNs();
    const unsigned long long deadline = start + kTimeoutNs;
    nixl_status_t status = NIXL_SUCCESS;
    unsigned long long enqueue_ns = 0;
    unsigned long long enqueue_ops = 0;
    const uint32_t channel_id = threadIdx.x;

    for (uint32_t base = 0; base < iterations && status == NIXL_SUCCESS; base += pipeline_window) {
        const uint32_t batch = min(pipeline_window, iterations - base);
        nixlGpuXferStatusH put_status[kMaxPipelineWindow];
        int32_t last_pending = -1;

        for (uint32_t i = 0; i < batch; ++i) {
            put_status[i] = nixlGpuXferStatusH{};
            enqueue_ns -= deviceTimeNs();
            status = nixlPut<nixl_gpu_level_t::THREAD>(
                src, dst, size, channel_id, /*flags=*/0, &put_status[i]);
            enqueue_ns += deviceTimeNs();
            ++enqueue_ops;
            if (status == NIXL_IN_PROG) {
                last_pending = static_cast<int32_t>(i);
                status = NIXL_SUCCESS;
            } else if (status != NIXL_SUCCESS) {
                break;
            }
        }

        // D2P completions form an ordered frontier. Completion of the last
        // pending PUT also completes every earlier PUT in this batch.
        if (status == NIXL_SUCCESS && last_pending >= 0) {
            status = waitForCompletion(put_status[last_pending], deadline);
        }
    }

    result[channel_id].elapsed_ns = deviceTimeNs() - start;
    result[channel_id].enqueue_ns = enqueue_ns;
    result[channel_id].enqueue_ops = enqueue_ops;
    result[channel_id].status = status;
}

__global__ void
runPipelinedPutSignalBenchmark(nixlMemViewElem src,
                               nixlMemViewElem dst,
                               nixlMemViewElem counter,
                               size_t size,
                               uint32_t iterations,
                               uint32_t pipeline_window,
                               DeviceResult *result) {
    constexpr unsigned long long kTimeoutNs = 60ULL * 1'000'000'000ULL;
    const unsigned long long start = deviceTimeNs();
    const unsigned long long deadline = start + kTimeoutNs;
    nixl_status_t status = NIXL_SUCCESS;
    unsigned long long enqueue_ns = 0;
    unsigned long long enqueue_ops = 0;
    const uint32_t channel_id = threadIdx.x;

    for (uint32_t base = 0; base < iterations && status == NIXL_SUCCESS; base += pipeline_window) {
        const uint32_t batch = min(pipeline_window, iterations - base);
        nixlGpuXferStatusH put_status[kMaxPipelineWindow];

        for (uint32_t i = 0; i < batch; ++i) {
            put_status[i] = nixlGpuXferStatusH{};
            enqueue_ns -= deviceTimeNs();
            status = nixlPut<nixl_gpu_level_t::THREAD>(
                src, dst, size, channel_id, /*flags=*/0, &put_status[i]);
            enqueue_ns += deviceTimeNs();
            ++enqueue_ops;
            if (status == NIXL_IN_PROG) {
                status = NIXL_SUCCESS;
            } else if (status != NIXL_SUCCESS) {
                break;
            }
        }
        if (status != NIXL_SUCCESS) {
            break;
        }

        // Do not wait for the PUTs. The target ACK for this signal proves
        // that every earlier PUT in the channel reached target memory.
        nixlGpuXferStatusH signal_status{};
        enqueue_ns -= deviceTimeNs();
        status = nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
            kSignalValue, counter, channel_id, /*flags=*/0, &signal_status);
        enqueue_ns += deviceTimeNs();
        ++enqueue_ops;
        if (status == NIXL_IN_PROG) {
            status = waitForCompletion(signal_status, deadline);
        }
    }

    result[channel_id].elapsed_ns = deviceTimeNs() - start;
    result[channel_id].enqueue_ns = enqueue_ns;
    result[channel_id].enqueue_ops = enqueue_ops;
    result[channel_id].status = status;
}

__device__ bool
waitForSystemCounter(uint64_t *counter, uint64_t expected, unsigned long long deadline_ns) {
    auto *system_counter = reinterpret_cast<unsigned long long *>(counter);
    while (atomicAdd_system(system_counter, 0ULL) < expected) {
        if (deviceTimeNs() >= deadline_ns) {
            return false;
        }
    }
    return true;
}

__global__ void
runOrderedSignalPingPongInitiator(nixlMemViewElem src,
                                  nixlMemViewElem dst,
                                  nixlMemViewElem remote_counter,
                                  uint64_t *local_counter,
                                  size_t size,
                                  uint32_t iterations,
                                  uint64_t initial_counter,
                                  DeviceResult *result) {
    constexpr unsigned long long kTimeoutNs = 60ULL * 1'000'000'000ULL;
    const unsigned long long start = deviceTimeNs();
    const unsigned long long deadline = start + kTimeoutNs;
    nixl_status_t status = NIXL_SUCCESS;
    unsigned long long enqueue_ns = 0;
    unsigned long long enqueue_ops = 0;

    for (uint32_t i = 0; i < iterations && status == NIXL_SUCCESS; ++i) {
        nixlGpuXferStatusH put_status{};
        enqueue_ns -= deviceTimeNs();
        status = nixlPut<nixl_gpu_level_t::THREAD>(
            src, dst, size, /*channel_id=*/0, /*flags=*/0, &put_status);
        enqueue_ns += deviceTimeNs();
        ++enqueue_ops;
        if (status == NIXL_IN_PROG) {
            status = NIXL_SUCCESS;
        }
        if (status != NIXL_SUCCESS) {
            break;
        }

        nixlGpuXferStatusH signal_status{};
        enqueue_ns -= deviceTimeNs();
        status = nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
            kSignalValue, remote_counter, /*channel_id=*/0, /*flags=*/0, &signal_status);
        enqueue_ns += deviceTimeNs();
        ++enqueue_ops;
        if (status == NIXL_IN_PROG) {
            status = waitForCompletion(signal_status, deadline);
        }
        if (status == NIXL_SUCCESS &&
            !waitForSystemCounter(local_counter, initial_counter + i + 1, deadline)) {
            status = NIXL_ERR_BACKEND;
        }
    }

    result->elapsed_ns = deviceTimeNs() - start;
    result->enqueue_ns = enqueue_ns;
    result->enqueue_ops = enqueue_ops;
    result->status = status;
}

__global__ void
runOrderedSignalPingPongResponder(nixlMemViewElem src,
                                  nixlMemViewElem dst,
                                  nixlMemViewElem remote_counter,
                                  uint64_t *local_counter,
                                  size_t size,
                                  uint32_t iterations,
                                  uint64_t initial_counter,
                                  DeviceResult *result) {
    constexpr unsigned long long kTimeoutNs = 60ULL * 1'000'000'000ULL;
    const unsigned long long start = deviceTimeNs();
    const unsigned long long deadline = start + kTimeoutNs;
    nixl_status_t status = NIXL_SUCCESS;
    unsigned long long enqueue_ns = 0;
    unsigned long long enqueue_ops = 0;

    for (uint32_t i = 0; i < iterations && status == NIXL_SUCCESS; ++i) {
        if (!waitForSystemCounter(local_counter, initial_counter + i + 1, deadline)) {
            status = NIXL_ERR_BACKEND;
            break;
        }

        nixlGpuXferStatusH put_status{};
        enqueue_ns -= deviceTimeNs();
        status = nixlPut<nixl_gpu_level_t::THREAD>(
            src, dst, size, /*channel_id=*/0, /*flags=*/0, &put_status);
        enqueue_ns += deviceTimeNs();
        ++enqueue_ops;
        if (status == NIXL_IN_PROG) {
            status = NIXL_SUCCESS;
        }
        if (status != NIXL_SUCCESS) {
            break;
        }

        nixlGpuXferStatusH signal_status{};
        enqueue_ns -= deviceTimeNs();
        status = nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
            kSignalValue, remote_counter, /*channel_id=*/0, /*flags=*/0, &signal_status);
        enqueue_ns += deviceTimeNs();
        ++enqueue_ops;
        if (status == NIXL_IN_PROG) {
            status = waitForCompletion(signal_status, deadline);
        }
    }

    result->elapsed_ns = deviceTimeNs() - start;
    result->enqueue_ns = enqueue_ns;
    result->enqueue_ops = enqueue_ops;
    result->status = status;
}

DeviceResult
copyDeviceResult(DeviceResult *device_result) {
    DeviceResult host_result{};
    checkCuda(cudaMemcpy(&host_result, device_result, sizeof(host_result), cudaMemcpyDeviceToHost),
              "copy benchmark result");
    return host_result;
}

DeviceResult
copyAggregateDeviceResult(DeviceResult *device_results, uint32_t channels) {
    std::vector<DeviceResult> host_results(channels);
    checkCuda(cudaMemcpy(host_results.data(),
                         device_results,
                         sizeof(DeviceResult) * channels,
                         cudaMemcpyDeviceToHost),
              "copy aggregate benchmark results");

    DeviceResult aggregate{};
    aggregate.status = NIXL_SUCCESS;
    for (const auto &result : host_results) {
        if (result.status != NIXL_SUCCESS) {
            aggregate.status = result.status;
            break;
        }
        aggregate.elapsed_ns = std::max(aggregate.elapsed_ns, result.elapsed_ns);
        aggregate.enqueue_ns += result.enqueue_ns;
        aggregate.enqueue_ops += result.enqueue_ops;
    }
    return aggregate;
}

uint32_t
iterationCount(size_t size) {
    if (size <= 128) {
        return 1000;
    }
    if (size <= 8192) {
        return 500;
    }
    return 200;
}

uint32_t
pipelineIterationCount(size_t size) {
    if (size == 8192) {
        return 8192;
    }
    if (size == 65536) {
        return 2048;
    }
    return 256;
}

uint64_t
pipelineSignalCount(uint32_t iterations, uint32_t pipeline_window) {
    return (iterations + pipeline_window - 1) / pipeline_window;
}

uint32_t
pipelineWindowFromEnvironment() {
    const char *value = std::getenv("NIXL_EFA_PROXY_PIPELINE_WINDOW");
    if (value == nullptr || *value == '\0') {
        return kDefaultPipelineWindow;
    }
    const unsigned long parsed = std::stoul(value);
    if (parsed == 0 || parsed > kMaxPipelineWindow) {
        throw std::runtime_error("NIXL_EFA_PROXY_PIPELINE_WINDOW must be in [1, 256]");
    }
    return static_cast<uint32_t>(parsed);
}

uint32_t
unsignedFromEnvironment(const char *name, uint32_t default_value, uint32_t max_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    const unsigned long parsed = std::stoul(value);
    if (parsed > max_value || (parsed == 0 && default_value != 0)) {
        throw std::runtime_error(std::string(name) + " must be in [" +
                                 std::to_string(default_value == 0 ? 0 : 1) + ", " +
                                 std::to_string(max_value) + "]");
    }
    return static_cast<uint32_t>(parsed);
}

void
printResult(const char *test,
            size_t size,
            uint32_t trial,
            uint32_t iterations,
            const DeviceResult &result) {
    if (result.status != NIXL_SUCCESS) {
        throw std::runtime_error(std::string(test) + " failed with NIXL status " +
                                 std::to_string(static_cast<int>(result.status)));
    }
    const double mean_us = static_cast<double>(result.elapsed_ns) / iterations / 1000.0;
    const double gbit_s = (static_cast<double>(size) * 8.0 * iterations) / result.elapsed_ns;
    std::cout << test << ',' << size << ',' << trial << ',' << iterations << ',' << mean_us << ','
              << gbit_s << '\n';
    if (result.enqueue_ops != 0) {
        std::cout << "gpu-enqueue," << test << ',' << size << ',' << trial << ','
                  << static_cast<double>(result.enqueue_ns) / result.enqueue_ops / 1000.0
                  << "us/op\n";
    }
}

void
checkDeviceResult(const char *test, const DeviceResult &result) {
    if (result.status != NIXL_SUCCESS) {
        throw std::runtime_error(std::string(test) + " failed with NIXL status " +
                                 std::to_string(static_cast<int>(result.status)));
    }
}

} // namespace

int
main(int argc, char **argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: nixl_efa_proxy_benchmark <shared-coordination-dir>");
        }
        const char *rank_env = std::getenv("SLURM_PROCID");
        if (rank_env == nullptr) {
            throw std::runtime_error("SLURM_PROCID is required");
        }
        const int rank = std::stoi(rank_env);
        if (rank < 0 || rank > 1) {
            throw std::runtime_error("Exactly two Slurm tasks are required");
        }
        const uint32_t pipeline_window = pipelineWindowFromEnvironment();
        const uint32_t pipeline_channels =
            unsignedFromEnvironment("NIXL_EFA_PROXY_CHANNELS", 1, kMaxPipelineChannels);
        const uint32_t proxy_workers =
            unsignedFromEnvironment("NIXL_EFA_PROXY_WORKERS", 1, pipeline_channels);
        const uint32_t minimum_size =
            unsignedFromEnvironment("NIXL_PROXY_BENCH_MIN_SIZE", 1, 1024 * 1024);
        const bool put_only = [] {
            const char *value = std::getenv("NIXL_PROXY_BENCH_PUT_ONLY");
            return value != nullptr && std::string(value) == "1";
        }();
        const bool counter_host = [] {
            const char *value = std::getenv("NIXL_PROXY_BENCH_COUNTER_HOST");
            return value != nullptr && std::string(value) == "1";
        }();
        // LIBFABRIC keeps the engine progress thread: a proxy's first write to a
        // peer waits for that peer's engine rail EP to answer the EFA handshake.
        const bool engine_progress_thread = [] {
            const char *value = std::getenv("NIXL_PROXY_BENCH_PROGRESS_THREAD");
            return value == nullptr || std::string(value) != "0";
        }();
        const uint32_t ordering_rounds =
            unsignedFromEnvironment("NIXL_PROXY_BENCH_ORDERING_ROUNDS", 0, 1u << 20);
        const auto env_flag = [](const char *name, bool default_value) {
            const char *value = std::getenv(name);
            return value == nullptr || *value == '\0' ? default_value : std::string(value) != "0";
        };
        const bool host_xfer = env_flag("NIXL_PROXY_BENCH_HOST_XFER", false);
        const bool device_proxy = env_flag("NIXL_PROXY_BENCH_DEVICE_PROXY", true);
        if (!device_proxy && (!host_xfer || ordering_rounds > 0)) {
            throw std::runtime_error(
                "NIXL_PROXY_BENCH_DEVICE_PROXY=0 needs NIXL_PROXY_BENCH_HOST_XFER=1 and no "
                "ordering rounds");
        }
        const char *backend_env = std::getenv("NIXL_PROXY_BENCH_BACKEND");
        const std::string backend_name = backend_env == nullptr ? "LIBFABRIC" : backend_env;
        if (backend_name != "LIBFABRIC" && backend_name != "UCX") {
            throw std::runtime_error("NIXL_PROXY_BENCH_BACKEND must be LIBFABRIC or UCX");
        }

        const std::filesystem::path coord_dir(argv[1]);
        std::filesystem::create_directories(coord_dir);
        const auto stage = [rank](const char *name) {
            std::cerr << "rank " << rank << ": " << name << std::endl;
        };
        stage("select GPU and allocate buffers");
        checkCuda(cudaSetDevice(0), "select CUDA device");

        constexpr size_t max_size = 1024 * 1024;
        void *src = nullptr;
        void *dst = nullptr;
        uint64_t *counter = nullptr;
        checkCuda(cudaMalloc(&src, max_size), "allocate source");
        checkCuda(cudaMalloc(&dst, max_size), "allocate destination");
        if (counter_host) {
            checkCuda(cudaMallocHost(&counter, sizeof(*counter)), "allocate pinned host counter");
        } else {
            checkCuda(cudaMalloc(&counter, sizeof(*counter)), "allocate GPU counter");
        }
        checkCuda(cudaMemset(src, 0x5a, max_size), "initialize source");
        checkCuda(cudaMemset(dst, 0, max_size), "initialize destination");
        if (counter_host) {
            *counter = 0;
        } else {
            checkCuda(cudaMemset(counter, 0, sizeof(*counter)), "initialize GPU counter");
        }

        const std::string agent_name = "proxy_bench_rank_" + std::to_string(rank);
        const std::string peer_name = "proxy_bench_rank_" + std::to_string(1 - rank);
        nixlAgentConfig config;
        // UCX proxy threads own the UCX workers, so its progress thread must be off.
        config.useProgThread = backend_name == "LIBFABRIC" && engine_progress_thread;
        config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        config.pthrDelay = unsignedFromEnvironment("NIXL_PROXY_BENCH_PTHR_DELAY_US", 1000, 1000000);
        stage("create agent and selected transport backend");
        nixlAgent agent(agent_name, config);

        nixlBackendH *backend = nullptr;
        nixl_b_params_t backend_params;
        if (device_proxy) {
            backend_params = {
                {"device_proxy", "true"},
                {"proxy_channel_count", std::to_string(pipeline_channels)},
                {"proxy_thread_count", std::to_string(proxy_workers)},
                {"proxy_max_peers", "1"},
            };
            if (const char *policy = std::getenv("NIXL_EFA_PROXY_RAIL_POLICY")) {
                backend_params.emplace("efa_proxy_rail_policy", policy);
            }
            if (const char *depth = std::getenv("NIXL_EFA_PROXY_RING_DEPTH")) {
                backend_params.emplace("proxy_ring_depth", depth);
            }
            if (const char *idle = std::getenv("NIXL_EFA_PROXY_IDLE_POLL_US")) {
                backend_params.emplace("efa_proxy_idle_poll_us", idle);
            }
        }
        checkNixl(agent.createBackend(backend_name, backend_params, backend),
                  "create transport backend");
        stage("backend ready; register GPU buffers");

        nixlBlobDesc src_desc(reinterpret_cast<uintptr_t>(src), max_size, kDeviceId);
        nixlBlobDesc dst_desc(reinterpret_cast<uintptr_t>(dst), max_size, kDeviceId);
        nixlBlobDesc counter_desc(
            reinterpret_cast<uintptr_t>(counter), sizeof(*counter), kDeviceId);
        nixl_reg_dlist_t registrations(VRAM_SEG);
        registrations.addDesc(src_desc);
        registrations.addDesc(dst_desc);
        nixl_reg_dlist_t counter_registrations(counter_host ? DRAM_SEG : VRAM_SEG);
        counter_registrations.addDesc(counter_desc);
        checkNixl(agent.registerMem(registrations), "register GPU buffers");
        checkNixl(agent.registerMem(counter_registrations), "register signal counter");

        // Ordering check (proxy_ordering.cuh): rank 0 sends rounds of puts, each
        // closed by atomicAdd; rank 1 verifies each round as its counter moves.
        namespace po = nixl_test::proxy_ordering;
        const po::Layout ordering_layout{pipeline_channels, ordering_rounds};
        void *ordering_buf = nullptr;
        nixl_reg_dlist_t ordering_registrations(VRAM_SEG);
        if (ordering_rounds > 0) {
            if (pipeline_channels > po::kMaxChannels) {
                throw std::runtime_error("Ordering check supports at most 32 channels");
            }
            const size_t bytes = ordering_layout.bufferBytes();
            checkCuda(cudaMalloc(&ordering_buf, bytes), "allocate ordering buffer");
            if (rank == 0) {
                po::fillKernel<<<1024, 256>>>(static_cast<uint32_t *>(ordering_buf),
                                              ordering_layout);
            } else {
                checkCuda(cudaMemset(ordering_buf, 0, bytes), "clear ordering buffer");
            }
            checkCuda(cudaDeviceSynchronize(), "initialize ordering buffer");
            ordering_registrations.addDesc(
                nixlBlobDesc(reinterpret_cast<uintptr_t>(ordering_buf), bytes, kDeviceId));
            checkNixl(agent.registerMem(ordering_registrations), "register ordering buffer");
        }
        stage("buffers registered; publish metadata");

        nixl_blob_t local_md;
        checkNixl(agent.getLocalMD(local_md), "get local metadata");
        const PeerMetadata local{reinterpret_cast<uint64_t>(src),
                                 reinterpret_cast<uint64_t>(dst),
                                 reinterpret_cast<uint64_t>(counter),
                                 max_size,
                                 reinterpret_cast<uint64_t>(ordering_buf),
                                 std::move(local_md)};
        writeFileAtomically(coord_dir / ("metadata." + std::to_string(rank)),
                            serializePeerMetadata(local));

        stage("wait for peer metadata");
        const PeerMetadata remote = deserializePeerMetadata(
            readFileWhenReady(coord_dir / ("metadata." + std::to_string(1 - rank))));
        std::string loaded_agent;
        stage("load peer metadata and connect");
        checkNixl(agent.loadRemoteMD(remote.nixl_blob, loaded_agent), "load peer metadata");
        if (loaded_agent != peer_name) {
            throw std::runtime_error("Loaded unexpected peer " + loaded_agent);
        }
        stage("peer connected");

        nixl_local_dlist_t local_src(VRAM_SEG);
        local_src.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(src), max_size, kDeviceId));
        nixl_remote_dlist_t remote_dst(VRAM_SEG);
        remote_dst.addDesc(nixlRemoteDesc(remote.dst_addr, max_size, kDeviceId, peer_name));
        nixl_remote_dlist_t remote_counter(counter_host ? DRAM_SEG : VRAM_SEG);
        remote_counter.addDesc(
            nixlRemoteDesc(remote.counter_addr, sizeof(uint64_t), kDeviceId, peer_name));

        nixlMemViewH src_mvh = nullptr;
        nixlMemViewH dst_mvh = nullptr;
        nixlMemViewH counter_mvh = nullptr;
        if (device_proxy) {
            checkNixl(agent.prepMemView(local_src, src_mvh), "prepare local source view");
            checkNixl(agent.prepMemView(remote_dst, dst_mvh), "prepare remote destination view");
            checkNixl(agent.prepMemView(remote_counter, counter_mvh),
                      "prepare remote counter view");
        }

        if (rank == 0) {
            std::cout << "backend," << backend_name << '\n';
            std::cout << "pipeline_window," << pipeline_window << '\n';
            std::cout << "pipeline_channels," << pipeline_channels << '\n';
            std::cout << "proxy_workers," << proxy_workers << '\n';
            std::cout << "engine_progress_thread," << (config.useProgThread ? 1 : 0) << '\n';
            std::cout << "ordering_rounds," << ordering_rounds << '\n';
            std::cout << "minimum_size," << minimum_size << '\n';
            std::cout << "put_only," << (put_only ? 1 : 0) << '\n';
            std::cout << "counter_memory," << (counter_host ? "host" : "gpu") << '\n';
            std::cout << "device_proxy," << (device_proxy ? 1 : 0) << '\n';
            std::cout << "host_xfer," << (host_xfer ? 1 : 0) << '\n';
            std::cout << "test,size_B,trial,iterations,mean_us,payload_Gbit_s\n";
        }

        if (ordering_rounds > 0) {
            stage("run ordering check");
            constexpr unsigned long long timeout_ns = 60ull * 1000 * 1000 * 1000;
            po::Result *result = nullptr;
            checkCuda(cudaMalloc(&result, sizeof(*result)), "allocate ordering result");
            checkCuda(cudaMemset(result, 0, sizeof(*result)), "clear ordering result");
            // Load the kernel before anything spins: a lazily loaded kernel's first
            // launch waits for running kernels.
            cudaFuncAttributes attr;
            checkCuda(cudaFuncGetAttributes(&attr, po::receiverKernel), "load receiver");
            checkCuda(cudaFuncGetAttributes(&attr, po::senderKernel), "load sender");
            if (rank == 1) {
                po::receiverKernel<<<pipeline_channels, 256>>>(
                    static_cast<uint8_t *>(ordering_buf), ordering_layout, timeout_ns, result);
                checkCuda(cudaGetLastError(), "launch ordering receiver");
                writeFileAtomically(coord_dir / "ordering-ready", "ready");
                checkCuda(cudaDeviceSynchronize(), "run ordering receiver");
            } else {
                nixl_local_dlist_t ordering_local(VRAM_SEG);
                ordering_local.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(ordering_buf),
                                                     ordering_layout.bufferBytes(),
                                                     kDeviceId));
                nixl_remote_dlist_t ordering_remote(VRAM_SEG);
                ordering_remote.addDesc(nixlRemoteDesc(
                    remote.ordering_addr, ordering_layout.bufferBytes(), kDeviceId, peer_name));
                nixlMemViewH local_mvh = nullptr;
                nixlMemViewH remote_mvh = nullptr;
                checkNixl(agent.prepMemView(ordering_local, local_mvh),
                          "prepare ordering source view");
                checkNixl(agent.prepMemView(ordering_remote, remote_mvh),
                          "prepare ordering target view");
                readFileWhenReady(coord_dir / "ordering-ready");
                cudaEvent_t start, stop;
                checkCuda(cudaEventCreate(&start), "create event");
                checkCuda(cudaEventCreate(&stop), "create event");
                checkCuda(cudaEventRecord(start), "record start");
                po::senderKernel<<<pipeline_channels, 1>>>(
                    local_mvh, remote_mvh, ordering_layout, timeout_ns, result);
                checkCuda(cudaEventRecord(stop), "record stop");
                checkCuda(cudaDeviceSynchronize(), "run ordering sender");
                float ms = 0;
                checkCuda(cudaEventElapsedTime(&ms, start, stop), "read elapsed time");
                const double rounds = double(pipeline_channels) * ordering_rounds;
                std::cout << "ordering_sender_ms," << ms << '\n'
                          << "ordering_signals_per_s," << rounds * 1e3 / ms << '\n'
                          << "ordering_payload_Gbit_s," << rounds * po::kRoundBytes * 8 / (ms * 1e6)
                          << '\n';
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                agent.releaseMemView(remote_mvh);
                agent.releaseMemView(local_mvh);
            }
            po::Result host{};
            checkCuda(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost),
                      "copy ordering result");
            checkCuda(cudaFree(result), "free ordering result");
            if (rank == 0) {
                for (uint32_t c = 0; c < pipeline_channels; ++c) {
                    if (host.sender_status[c] != NIXL_SUCCESS) {
                        throw std::runtime_error("Ordering sender failed on channel " +
                                                 std::to_string(c) + " with status " +
                                                 std::to_string(host.sender_status[c]));
                    }
                }
                double enqueue_ns = 0;
                double wait_ns = 0;
                for (uint32_t c = 0; c < pipeline_channels; ++c) {
                    enqueue_ns += host.enqueue_ns[c];
                    wait_ns += host.final_wait_ns[c];
                }
                const double ops =
                    double(pipeline_channels) * ordering_rounds * (po::kPutsPerRound + 1);
                std::cout << "ordering_gpu_enqueue_us_per_op," << enqueue_ns / ops / 1e3 << '\n'
                          << "ordering_gpu_final_wait_us," << wait_ns / pipeline_channels / 1e3
                          << '\n'
                          << "ordering,sender_ok\n";
            } else {
                std::ostringstream report;
                report << "ordering,mismatches=" << host.mismatches
                       << ",regressions=" << host.regressions << ",torn=" << host.torn
                       << ",timed_out=" << host.timed_out;
                for (uint32_t c = 0; c < pipeline_channels; ++c) {
                    report << ",counter" << c << '=' << host.final_counter[c];
                }
                if (host.mismatches != 0) {
                    report << ",first=channel" << host.first_channel << "/round" << host.first_round
                           << "/word" << (host.first_bad - 1) << "/value0x" << std::hex
                           << host.first_value << "/expected0x" << host.first_expected << std::dec;
                }
                std::cout << report.str() << std::endl;
                bool ok = host.mismatches == 0 && host.regressions == 0 && host.torn == 0 &&
                    host.timed_out == 0;
                for (uint32_t c = 0; c < pipeline_channels; ++c) {
                    ok = ok && host.final_counter[c] == ordering_rounds;
                }
                if (!ok) {
                    throw std::runtime_error("Ordering check failed: " + report.str());
                }
            }
            writeFileAtomically(coord_dir / ("ordering-done-" + std::to_string(rank)), "done");
            readFileWhenReady(coord_dir / ("ordering-done-" + std::to_string(1 - rank)));
            stage("ordering check passed");
        }
        const bool ordering_only = [] {
            const char *value = std::getenv("NIXL_PROXY_BENCH_ORDERING_ONLY");
            return value != nullptr && std::string(value) == "1";
        }();

        uint64_t pingpong_signals_sent = 0;
        // Run a subset of the phases, e.g. to profile one traffic pattern.
        const auto skip = [](const char *name) {
            const char *value = std::getenv(name);
            return value != nullptr && std::string(value) == "1";
        };
        const bool skip_pingpong = host_xfer || skip("NIXL_PROXY_BENCH_SKIP_PINGPONG");
        const bool skip_serial = host_xfer || skip("NIXL_PROXY_BENCH_SKIP_SERIAL");
        const bool skip_pipelined = host_xfer || skip("NIXL_PROXY_BENCH_SKIP_PIPELINED");

        if (!put_only && !ordering_only && !skip_pingpong) {
            if (counter_host) {
                throw std::runtime_error("Device ping-pong requires a GPU-resident local counter");
            }
            constexpr size_t pingpong_size = 8;
            constexpr uint32_t pingpong_warmups = 100;
            constexpr uint32_t pingpong_iterations = 1000;
            constexpr uint32_t pingpong_trials = 5;
            DeviceResult *pingpong_result = nullptr;
            checkCuda(cudaMalloc(&pingpong_result, sizeof(*pingpong_result)),
                      "allocate ping-pong result");
            uint64_t initial_counter = 0;

            for (uint32_t phase = 0; phase <= pingpong_trials; ++phase) {
                const uint32_t iterations = phase == 0 ? pingpong_warmups : pingpong_iterations;
                const std::string suffix = std::to_string(phase);
                if (rank == 1) {
                    runOrderedSignalPingPongResponder<<<1, 1>>>({src_mvh, 0, 0},
                                                                {dst_mvh, 0, 0},
                                                                {counter_mvh, 0, 0},
                                                                counter,
                                                                pingpong_size,
                                                                iterations,
                                                                initial_counter,
                                                                pingpong_result);
                    checkCuda(cudaGetLastError(), "launch ping-pong responder");
                    writeFileAtomically(coord_dir / ("pingpong-ready-" + suffix), "ready");
                    checkCuda(cudaDeviceSynchronize(), "run ping-pong responder");
                    checkDeviceResult("ordered PUT+signal ping-pong responder",
                                      copyDeviceResult(pingpong_result));
                    readFileWhenReady(coord_dir / ("pingpong-done-" + suffix));
                } else {
                    readFileWhenReady(coord_dir / ("pingpong-ready-" + suffix));
                    runOrderedSignalPingPongInitiator<<<1, 1>>>({src_mvh, 0, 0},
                                                                {dst_mvh, 0, 0},
                                                                {counter_mvh, 0, 0},
                                                                counter,
                                                                pingpong_size,
                                                                iterations,
                                                                initial_counter,
                                                                pingpong_result);
                    checkCuda(cudaDeviceSynchronize(), "run ping-pong initiator");
                    const DeviceResult result = copyDeviceResult(pingpong_result);
                    checkDeviceResult("ordered PUT+signal ping-pong initiator", result);
                    if (phase != 0) {
                        printResult("PUT-ordered-signal-pingpong",
                                    pingpong_size,
                                    phase,
                                    iterations,
                                    result);
                    }
                    writeFileAtomically(coord_dir / ("pingpong-done-" + suffix), "done");
                }
                initial_counter += iterations;
            }
            pingpong_signals_sent = initial_counter;
            checkCuda(cudaFree(pingpong_result), "free ping-pong result");
        }

        if (ordering_only) {
            // Nothing else to run.
        } else if (rank == 0) {
            DeviceResult *device_result = nullptr;
            checkCuda(cudaMalloc(&device_result, sizeof(*device_result) * pipeline_channels),
                      "allocate result");
            std::vector<size_t> sizes{8, 64, 128, 8192, 65536};
            sizes.erase(std::remove_if(sizes.begin(),
                                       sizes.end(),
                                       [minimum_size](size_t size) { return size < minimum_size; }),
                        sizes.end());
            if (sizes.empty()) {
                throw std::runtime_error("NIXL_PROXY_BENCH_MIN_SIZE excludes every test size");
            }
            constexpr uint32_t warmup_iterations = 50;
            constexpr uint32_t trials = 5;
            uint64_t expected_signals = pingpong_signals_sent;
            // Every PUT writes 0x5a from offset 0, so the target can only check
            // the prefix that some transfer actually covered.
            uint64_t verified_extent = 0;

            stage("run benchmark");

            // Host path: `window` requests in flight, each writing the whole buffer
            // in descriptors of `size` bytes, reposted as they complete.
            for (const size_t size :
                 host_xfer ? std::vector<size_t>{65536, 1048576} : std::vector<size_t>{}) {
                constexpr uint32_t window = 8;
                const uint32_t rounds =
                    unsignedFromEnvironment("NIXL_PROXY_BENCH_HOST_ROUNDS", 200, 1u << 20);
                nixl_xfer_dlist_t local_descs(VRAM_SEG);
                nixl_xfer_dlist_t remote_descs(VRAM_SEG);
                for (size_t offset = 0; offset + size <= max_size; offset += size) {
                    local_descs.addDesc(
                        nixlBasicDesc(reinterpret_cast<uintptr_t>(src) + offset, size, kDeviceId));
                    remote_descs.addDesc(nixlBasicDesc(remote.dst_addr + offset, size, kDeviceId));
                }
                std::vector<nixlXferReqH *> requests(window, nullptr);
                for (auto &request : requests) {
                    checkNixl(agent.createXferReq(
                                  NIXL_WRITE, local_descs, remote_descs, peer_name, request),
                              "create host transfer");
                }
                const auto run_rounds = [&](uint32_t count) {
                    for (uint32_t round = 0; round < count; ++round) {
                        for (auto *request : requests) {
                            const nixl_status_t status = agent.postXferReq(request);
                            if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                                checkNixl(status, "post host transfer");
                            }
                        }
                        for (auto *request : requests) {
                            nixl_status_t status;
                            while ((status = agent.getXferStatus(request)) == NIXL_IN_PROG) {}
                            checkNixl(status, "complete host transfer");
                        }
                    }
                };
                run_rounds(10);
                for (uint32_t trial = 1; trial <= 5; ++trial) {
                    const auto start = std::chrono::steady_clock::now();
                    run_rounds(rounds);
                    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - start)
                                        .count();
                    const double bytes = double(window) * rounds * max_size;
                    std::cout << "HOST-write," << size << ',' << trial << ',' << window * rounds
                              << ',' << ns / 1000.0 / (window * rounds) << ',' << bytes * 8.0 / ns
                              << std::endl;
                }
                for (auto *request : requests) {
                    checkNixl(agent.releaseXferReq(request), "release host transfer");
                }
                verified_extent = max_size;
            }

            for (const size_t size : skip_serial ? std::vector<size_t>{} : sizes) {
                const uint32_t iterations = iterationCount(size);
                verified_extent = std::max<uint64_t>(verified_extent, size);
                runPutBenchmark<<<1, 1>>>(
                    {src_mvh, 0, 0}, {dst_mvh, 0, 0}, size, warmup_iterations, device_result);
                checkCuda(cudaDeviceSynchronize(), "warm up PUT benchmark");
                checkDeviceResult("PUT warmup", copyDeviceResult(device_result));

                if (!put_only) {
                    // Closest NIXL equivalent to a fused put-with-signal: queue
                    // both descriptors on one ordered channel and wait only
                    // for the AtomicAdd completion. Reuse the generic pipeline
                    // kernel with a one-operation window so every backend runs
                    // exactly the same GPU code.
                    runPipelinedPutSignalBenchmark<<<1, 1>>>({src_mvh, 0, 0},
                                                             {dst_mvh, 0, 0},
                                                             {counter_mvh, 0, 0},
                                                             size,
                                                             warmup_iterations,
                                                             /*pipeline_window=*/1,
                                                             device_result);
                    checkCuda(cudaDeviceSynchronize(), "warm up ordered PUT+signal benchmark");
                    checkDeviceResult("ordered PUT+signal warmup", copyDeviceResult(device_result));
                    expected_signals += warmup_iterations;
                }

                for (uint32_t trial = 1; trial <= trials; ++trial) {
                    runPutBenchmark<<<1, 1>>>(
                        {src_mvh, 0, 0}, {dst_mvh, 0, 0}, size, iterations, device_result);
                    checkCuda(cudaDeviceSynchronize(), "run PUT benchmark");
                    printResult(
                        "PUT-target", size, trial, iterations, copyDeviceResult(device_result));

                    if (!put_only) {
                        runPipelinedPutSignalBenchmark<<<1, 1>>>({src_mvh, 0, 0},
                                                                 {dst_mvh, 0, 0},
                                                                 {counter_mvh, 0, 0},
                                                                 size,
                                                                 iterations,
                                                                 /*pipeline_window=*/1,
                                                                 device_result);
                        checkCuda(cudaDeviceSynchronize(), "run ordered PUT+signal benchmark");
                        printResult("PUT-ordered-signal",
                                    size,
                                    trial,
                                    iterations,
                                    copyDeviceResult(device_result));
                        expected_signals += iterations;
                    }
                }
            }

            // Keep pipeline_window operations outstanding. For sub-threshold
            // sizes, successive LIBFABRIC transfers rotate across selected rails.
            for (const size_t size : skip_pipelined ? std::vector<size_t>{} :
                                                      std::vector<size_t>{8192, 65536, 1048576}) {
                if (size < minimum_size) {
                    continue;
                }
                const uint32_t iterations = pipelineIterationCount(size);
                const uint32_t pipeline_warmup_iterations = pipeline_window * 2;
                verified_extent = std::max<uint64_t>(verified_extent, size);

                runPipelinedPutBenchmark<<<1, pipeline_channels>>>({src_mvh, 0, 0},
                                                                   {dst_mvh, 0, 0},
                                                                   size,
                                                                   pipeline_warmup_iterations,
                                                                   pipeline_window,
                                                                   device_result);
                checkCuda(cudaDeviceSynchronize(), "warm up pipelined PUT benchmark");
                checkDeviceResult("pipelined PUT warmup",
                                  copyAggregateDeviceResult(device_result, pipeline_channels));

                if (!put_only) {
                    runPipelinedPutSignalBenchmark<<<1, pipeline_channels>>>(
                        {src_mvh, 0, 0},
                        {dst_mvh, 0, 0},
                        {counter_mvh, 0, 0},
                        size,
                        pipeline_warmup_iterations,
                        pipeline_window,
                        device_result);
                    checkCuda(cudaDeviceSynchronize(), "warm up pipelined PUT+signal benchmark");
                    checkDeviceResult("pipelined PUT+signal warmup",
                                      copyAggregateDeviceResult(device_result, pipeline_channels));
                    expected_signals +=
                        pipelineSignalCount(pipeline_warmup_iterations, pipeline_window) *
                        pipeline_channels;
                }

                for (uint32_t trial = 1; trial <= trials; ++trial) {
                    runPipelinedPutBenchmark<<<1, pipeline_channels>>>({src_mvh, 0, 0},
                                                                       {dst_mvh, 0, 0},
                                                                       size,
                                                                       iterations,
                                                                       pipeline_window,
                                                                       device_result);
                    checkCuda(cudaDeviceSynchronize(), "run pipelined PUT benchmark");
                    printResult("PUT-pipelined",
                                size,
                                trial,
                                iterations * pipeline_channels,
                                copyAggregateDeviceResult(device_result, pipeline_channels));

                    if (!put_only) {
                        runPipelinedPutSignalBenchmark<<<1, pipeline_channels>>>(
                            {src_mvh, 0, 0},
                            {dst_mvh, 0, 0},
                            {counter_mvh, 0, 0},
                            size,
                            iterations,
                            pipeline_window,
                            device_result);
                        checkCuda(cudaDeviceSynchronize(), "run pipelined PUT+signal benchmark");
                        printResult("PUT-batch-signal",
                                    size,
                                    trial,
                                    iterations * pipeline_channels,
                                    copyAggregateDeviceResult(device_result, pipeline_channels));
                        expected_signals +=
                            pipelineSignalCount(iterations, pipeline_window) * pipeline_channels;
                    }
                }
            }

            checkCuda(cudaFree(device_result), "free result");
            writeFileAtomically(coord_dir / "done",
                                std::to_string(expected_signals) + " " +
                                    std::to_string(verified_extent));
            readFileWhenReady(coord_dir / "validated");
            stage("target validation received");
        } else {
            stage("wait for initiator completion");
            // The last signal the initiator waited for was acknowledged by this
            // rank, so every PUT it ordered ahead of that signal has landed.
            std::istringstream done(readFileWhenReady(coord_dir / "done"));
            uint64_t expected_signals = 0;
            uint64_t verified_extent = 0;
            if (!(done >> expected_signals >> verified_extent) || verified_extent > max_size) {
                throw std::runtime_error("Malformed completion record from the initiator");
            }
            uint64_t observed_signals = 0;
            if (counter_host) {
                observed_signals = *counter;
            } else {
                checkCuda(cudaMemcpy(&observed_signals,
                                     counter,
                                     sizeof(observed_signals),
                                     cudaMemcpyDeviceToHost),
                          "read target signal counter");
            }
            if (observed_signals != expected_signals) {
                throw std::runtime_error("Target counter mismatch: expected " +
                                         std::to_string(expected_signals) + ", observed " +
                                         std::to_string(observed_signals));
            }
            std::vector<unsigned char> observed(verified_extent);
            checkCuda(cudaMemcpy(observed.data(), dst, verified_extent, cudaMemcpyDeviceToHost),
                      "read target data");
            for (size_t offset = 0; offset < observed.size(); ++offset) {
                if (observed[offset] != 0x5a) {
                    throw std::runtime_error("Target payload verification failed at offset " +
                                             std::to_string(offset));
                }
            }
            writeFileAtomically(coord_dir / "validated", "ok");
            stage("target validation passed");
        }

        if (device_proxy) {
            agent.releaseMemView(counter_mvh);
            agent.releaseMemView(dst_mvh);
            agent.releaseMemView(src_mvh);
        }
        if (ordering_buf != nullptr) {
            checkNixl(agent.deregisterMem(ordering_registrations), "deregister ordering buffer");
            checkCuda(cudaFree(ordering_buf), "free ordering buffer");
        }

        checkNixl(agent.invalidateRemoteMD(peer_name), "invalidate peer metadata");
        checkNixl(agent.deregisterMem(counter_registrations), "deregister signal counter");
        checkNixl(agent.deregisterMem(registrations), "deregister GPU buffers");
        if (counter_host) {
            checkCuda(cudaFreeHost(counter), "free pinned host counter");
        } else {
            checkCuda(cudaFree(counter), "free GPU counter");
        }
        checkCuda(cudaFree(dst), "free destination");
        checkCuda(cudaFree(src), "free source");
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << "EFA proxy benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
