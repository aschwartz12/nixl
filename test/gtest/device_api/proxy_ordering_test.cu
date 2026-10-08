/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// put -> atomicAdd ordering through the device proxy, checked on the target GPU
// (see proxy_ordering.cuh), with both agents in this process.

#include "utils.cuh"
#include "common.h"
#include "proxy_ordering.cuh"
#include "libfabric_proxy_wire.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

namespace gtest::nixl::gpu::proxy_ordering {

using namespace ::nixl_test::proxy_ordering;

constexpr unsigned kChannels = 4;
constexpr Layout kLayout{kChannels, 128};
static_assert(kChannels <= kMaxChannels);
constexpr unsigned long long kTimeoutNs = 60ull * 1000 * 1000 * 1000;
constexpr uint32_t kProxyThreads = 4;

/** One thread: `count` adds of 1 to one counter on channel 0, then wait for the last. */
__global__ void
atomicLoopKernel(nixlMemViewH dst,
                 size_t counter_offset,
                 unsigned count,
                 unsigned long long timeout_ns,
                 nixl_status_t *status_out) {
    nixlGpuXferStatusH last{};
    nixl_status_t status = NIXL_SUCCESS;
    for (unsigned i = 0; i < count && status == NIXL_SUCCESS; ++i) {
        if (nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
                1, {dst, 0, counter_offset}, 0, 0, i + 1 == count ? &last : nullptr) !=
            NIXL_IN_PROG) {
            status = NIXL_ERR_BACKEND;
        }
    }
    const unsigned long long deadline = globalTimeNs() + timeout_ns;
    if (status == NIXL_SUCCESS) {
        do {
            status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(last);
        } while (status == NIXL_IN_PROG && globalTimeNs() < deadline);
    }
    *status_out = status;
}

/** One block per channel: `count` adds of 1 to one shared counter on its own ring. */
__global__ void
manyRingsAtomicKernel(nixlMemViewH dst,
                      size_t counter_offset,
                      unsigned count,
                      unsigned long long timeout_ns,
                      nixl_status_t *status_out) {
    const unsigned channel = blockIdx.x;
    nixlGpuXferStatusH last{};
    nixl_status_t status = NIXL_SUCCESS;
    for (unsigned i = 0; i < count && status == NIXL_SUCCESS; ++i) {
        if (nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
                1, {dst, 0, counter_offset}, channel, 0, i + 1 == count ? &last : nullptr) !=
            NIXL_IN_PROG) {
            status = NIXL_ERR_BACKEND;
        }
    }
    const unsigned long long deadline = globalTimeNs() + timeout_ns;
    if (status == NIXL_SUCCESS) {
        do {
            status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(last);
        } while (status == NIXL_IN_PROG && globalTimeNs() < deadline);
    }
    status_out[channel] = status;
}

/** Rounds of add(A), add(B) on one ring: B must never run ahead of A. */
__global__ void
twoCounterSenderKernel(nixlMemViewH dst,
                       size_t offset_a,
                       size_t offset_b,
                       unsigned rounds,
                       unsigned long long timeout_ns,
                       nixl_status_t *status_out) {
    nixlGpuXferStatusH last{};
    nixl_status_t status = NIXL_SUCCESS;
    for (unsigned r = 0; r < rounds && status == NIXL_SUCCESS; ++r) {
        if (nixlAtomicAdd<nixl_gpu_level_t::THREAD>(1, {dst, 0, offset_a}, 0) != NIXL_IN_PROG ||
            nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
                1, {dst, 0, offset_b}, 0, 0, r + 1 == rounds ? &last : nullptr) != NIXL_IN_PROG) {
            status = NIXL_ERR_BACKEND;
        }
    }
    const unsigned long long deadline = globalTimeNs() + timeout_ns;
    if (status == NIXL_SUCCESS) {
        do {
            status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(last);
        } while (status == NIXL_IN_PROG && globalTimeNs() < deadline);
    }
    *status_out = status;
}

__global__ void
twoCounterReceiverKernel(unsigned long long *a,
                         unsigned long long *b,
                         unsigned rounds,
                         unsigned long long timeout_ns,
                         unsigned long long *out /* violations, final b */) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> ra(*a), rb(*b);
    const unsigned long long deadline = globalTimeNs() + timeout_ns;
    unsigned long long violations = 0;
    unsigned long long bv = 0;
    do {
        bv = rb.load(cuda::memory_order_acquire);
        const unsigned long long av = ra.load(cuda::memory_order_acquire);
        violations += av < bv;
    } while (bv < rounds && globalTimeNs() < deadline);
    out[0] = violations;
    out[1] = bv;
}

/** Parameter: the backend (LIBFABRIC: the EFA proxy, which orders at the target). */
class ProxyOrderingTest : public testing::TestWithParam<std::string> {
protected:
    static constexpr size_t kSender = 0;
    static constexpr size_t kReceiver = 1;
    static constexpr uint64_t kDevId = 0;

    std::string
    backend() const {
        return GetParam();
    }

    /** The EFA proxy orders put -> atomicAdd at the target. */
    bool
    receiverOrdering() const {
        return backend() == "LIBFABRIC";
    }

    nixl_b_params_t
    backendParams() const {
        nixl_b_params_t params = {{"device_proxy", "true"},
                                  {"proxy_channel_count", std::to_string(kChannels)},
                                  {"proxy_thread_count", std::to_string(kProxyThreads)},
                                  {"proxy_max_peers", "2"}};
        // Experiments: extra EFA proxy parameters, "key=value,...".
        const char *extra = std::getenv("NIXL_GTEST_EFA_PROXY_RX_PARAMS");
        if (receiverOrdering() && extra != nullptr) {
            std::stringstream items(extra);
            std::string item;
            while (std::getline(items, item, ',')) {
                const size_t eq = item.find('=');
                if (eq != std::string::npos) {
                    params[item.substr(0, eq)] = item.substr(eq + 1);
                }
            }
        }
        return params;
    }

    void
    SetUp() override {
        if (!hasCudaGpu()) {
            GTEST_SKIP() << "No CUDA-capable GPU is available";
        }
        ASSERT_EQ(cudaSetDevice(kDevId), cudaSuccess);
        createAgents();
    }

    void
    TearDown() override {
        agents_.clear();
    }

    /** Sender and receiver agents in this process; skips if the backend is missing. */
    void
    createAgents() {
        nixlAgentConfig cfg;
        // Both agents share this thread; LIBFABRIC connection setup needs the
        // peer's rail progressed, which its proxy threads do not do.
        cfg.useProgThread = true;
        cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        // Handle handshakes promptly: one still pending at disconnect re-creates
        // the connection and logs an error at the peer (engine race).
        cfg.pthrDelay = 1000;
        for (size_t i = 0; i < 2; ++i) {
            agents_.emplace_back(std::make_unique<nixlAgent>(name(i), cfg));
            if (i == 0) {
                std::vector<nixl_backend_t> plugins;
                ASSERT_EQ(agents_.back()->getAvailPlugins(plugins), NIXL_SUCCESS);
                if (std::find(plugins.begin(), plugins.end(), backend()) == plugins.end()) {
                    GTEST_SKIP() << backend() << " plugin is unavailable";
                }
            }
            nixlBackendH *handle = nullptr;
            ASSERT_EQ(agents_.back()->createBackend(backend(), backendParams(), handle),
                      NIXL_SUCCESS);
        }
    }

    /** Replace agent @p i by a new one under the same name (a restarted process). */
    void
    recreateAgent(size_t i) {
        nixlAgentConfig cfg;
        cfg.useProgThread = true;
        cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        cfg.pthrDelay = 1000;
        agents_[i].reset();
        agents_[i] = std::make_unique<nixlAgent>(name(i), cfg);
        nixlBackendH *handle = nullptr;
        ASSERT_EQ(agents_[i]->createBackend(backend(), backendParams(), handle), NIXL_SUCCESS);
    }

    /**
     * Every channel (its own ring, on its own proxy thread) adds 1 to the same
     * counter @p count times. Each sending thread's adds go to the target thread
     * receiving on its rail, so here up to one target thread per ring writes the
     * counter: none of the adds may be lost.
     */
    void
    runManyRingsOneCounter(unsigned count) {
        const Layout layout{kChannels, 1};
        Buffers b(layout);
        setUpBuffers(layout, b);
        if (HasFatalFailure()) {
            return;
        }
        nixl_status_t *status = nullptr;
        ASSERT_EQ(cudaMalloc(&status, kChannels * sizeof(*status)), cudaSuccess);
        manyRingsAtomicKernel<<<kChannels, 1>>>(b.dst_mvh, 0, count, kTimeoutNs, status);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<nixl_status_t> host(kChannels, NIXL_IN_PROG);
        ASSERT_EQ(cudaMemcpy(host.data(), status, kChannels * sizeof(*status),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        uint64_t counter = 0;
        ASSERT_EQ(cudaMemcpy(&counter, b.dst, sizeof(counter), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (unsigned c = 0; c < kChannels; ++c) {
            EXPECT_EQ(host[c], NIXL_SUCCESS) << "channel " << c;
        }
        EXPECT_EQ(counter, uint64_t(kChannels) * count) << "adds to one counter were lost";
        cudaFree(status);
        releaseViews(b);
    }

    static std::string
    name(size_t i) {
        return "agent_" + std::to_string(i);
    }

    void
    registerBuffer(size_t agent, const MemBuffer &buf) {
        nixl_reg_dlist_t list(VRAM_SEG);
        list.addDesc(nixlBlobDesc(buf, buf.getSize(), kDevId));
        ASSERT_EQ(agents_[agent]->registerMem(list), NIXL_SUCCESS);
    }

    void
    deregisterBuffer(size_t agent, const MemBuffer &buf) {
        nixl_reg_dlist_t list(VRAM_SEG);
        list.addDesc(nixlBlobDesc(buf, buf.getSize(), kDevId));
        ASSERT_EQ(agents_[agent]->deregisterMem(list), NIXL_SUCCESS);
    }

    void
    exchangeMD() {
        for (size_t i = 0; i < agents_.size(); ++i) {
            nixl_blob_t md;
            ASSERT_EQ(agents_[i]->getLocalMD(md), NIXL_SUCCESS);
            for (size_t j = 0; j < agents_.size(); ++j) {
                if (i != j) {
                    std::string remote;
                    ASSERT_EQ(agents_[j]->loadRemoteMD(md, remote), NIXL_SUCCESS);
                }
            }
        }
    }

    /** Source filled with the round pattern, zeroed destination, and their views. */
    struct Buffers {
        MemBuffer src;
        MemBuffer dst;
        nixlMemViewH src_mvh = nullptr;
        nixlMemViewH dst_mvh = nullptr;

        explicit Buffers(const Layout &layout)
            : src(layout.bufferBytes(), VRAM_SEG),
              dst(layout.bufferBytes(), VRAM_SEG) {}
    };

    void
    setUpBuffers(const Layout &layout, Buffers &b) {
        ASSERT_NE(static_cast<void *>(b.src), nullptr);
        ASSERT_NE(static_cast<void *>(b.dst), nullptr);
        ASSERT_EQ(cudaMemset(b.dst, 0, layout.bufferBytes()), cudaSuccess);
        fillKernel<<<1024, 256>>>(static_cast<uint32_t *>(static_cast<void *>(b.src)), layout);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        registerBuffer(kSender, b.src);
        registerBuffer(kReceiver, b.dst);
        exchangeMD();

        nixl_local_dlist_t src_list(VRAM_SEG);
        src_list.addDesc(nixlBasicDesc(b.src, b.src.getSize(), kDevId));
        nixl_remote_dlist_t dst_list(VRAM_SEG);
        dst_list.addDesc(nixlRemoteDesc(b.dst, b.dst.getSize(), kDevId, name(kReceiver)));
        ASSERT_EQ(agents_[kSender]->prepMemView(src_list, b.src_mvh), NIXL_SUCCESS);
        ASSERT_EQ(agents_[kSender]->prepMemView(dst_list, b.dst_mvh), NIXL_SUCCESS);
    }

    /** Releasing a view waits for everything its rings still carry. */
    void
    releaseViews(Buffers &b) {
        agents_[kSender]->releaseMemView(b.dst_mvh);
        agents_[kSender]->releaseMemView(b.src_mvh);
        b.dst_mvh = b.src_mvh = nullptr;
    }

    /** Once the rings drained: every channel signalled @p rounds rounds, with their data. */
    void
    checkTarget(const Buffers &b, const Layout &layout, unsigned rounds) {
        std::vector<uint32_t> host(layout.bufferBytes() / sizeof(uint32_t));
        ASSERT_EQ(cudaMemcpy(host.data(), b.dst, layout.bufferBytes(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (unsigned c = 0; c < layout.channels; ++c) {
            uint64_t counter;
            std::memcpy(&counter,
                        reinterpret_cast<const char *>(host.data()) + c * kCounterStride,
                        sizeof(counter));
            EXPECT_EQ(counter, rounds * kCounterStep) << "channel " << c;
            size_t mismatches = 0;
            for (unsigned round = 0; round < rounds; ++round) {
                const size_t first = layout.roundOffset(c, round) / sizeof(uint32_t);
                for (size_t w = 0; w < kRoundBytes / sizeof(uint32_t); ++w) {
                    mismatches += host[first + w] != pattern(c, round, w);
                }
            }
            EXPECT_EQ(mismatches, 0u) << "channel " << c;
        }
    }

    /**
     * Stream every round on every channel and check it on the target GPU. With
     * @p receiver_on_legacy_stream the receiver spins on the legacy default stream,
     * which a target-side add must not wait for.
     */
    void
    runOrderingCheck(const Layout &layout, bool receiver_on_legacy_stream = false) {
        Buffers b(layout);
        setUpBuffers(layout, b);
        if (HasFatalFailure()) {
            return;
        }

        Result *result = nullptr;
        ASSERT_EQ(cudaMalloc(&result, sizeof(*result)), cudaSuccess);
        ASSERT_EQ(cudaMemset(result, 0, sizeof(*result)), cudaSuccess);

        // Separate non-blocking streams so the receiver runs alongside the sender.
        cudaStream_t rx_stream, tx_stream;
        ASSERT_EQ(cudaStreamCreateWithFlags(&rx_stream, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&tx_stream, cudaStreamNonBlocking), cudaSuccess);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Load both kernels first: with lazy module loading, the sender's first launch
        // would otherwise wait for the spinning receiver to exit.
        cudaFuncAttributes attr;
        ASSERT_EQ(cudaFuncGetAttributes(&attr, receiverKernel), cudaSuccess);
        ASSERT_EQ(cudaFuncGetAttributes(&attr, senderKernel), cudaSuccess);

        cudaStream_t receiver_stream = receiver_on_legacy_stream ? cudaStreamLegacy : rx_stream;
        receiverKernel<<<layout.channels, 256, 0, receiver_stream>>>(
            static_cast<uint8_t *>(static_cast<void *>(b.dst)), layout, kTimeoutNs, result);
        cudaEventRecord(start, tx_stream);
        senderKernel<<<layout.channels, 1, 0, tx_stream>>>(
            b.src_mvh, b.dst_mvh, layout, kTimeoutNs, result);
        cudaEventRecord(stop, tx_stream);
        ASSERT_EQ(cudaStreamSynchronize(tx_stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(receiver_stream), cudaSuccess);

        Result host{};
        ASSERT_EQ(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        const double rounds = double(layout.channels) * layout.rounds;
        Logger() << "ProxyOrdering: " << layout.channels << " channel(s) x " << layout.rounds
                 << " round(s) x " << kRoundBytes << " B in " << ms << " ms ("
                 << rounds * kRoundBytes / (ms * 1e-3) / 1e9 << " GB/s, " << rounds * 1e3 / ms
                 << " signals/s)";

        for (unsigned c = 0; c < layout.channels; ++c) {
            EXPECT_EQ(host.sender_status[c], NIXL_SUCCESS) << "channel " << c;
            EXPECT_EQ(host.final_counter[c], layout.rounds) << "channel " << c;
        }
        EXPECT_EQ(host.timed_out, 0u);
        EXPECT_EQ(host.regressions, 0u) << "a counter moved backwards";
        EXPECT_EQ(host.torn, 0u) << "a counter was read half-updated";
        EXPECT_EQ(host.mismatches, 0u)
            << "data not visible when its counter was: channel " << host.first_channel << " round "
            << host.first_round << " word " << (host.first_bad - 1) << " value 0x" << std::hex
            << host.first_value << " expected 0x" << host.first_expected;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(rx_stream);
        cudaStreamDestroy(tx_stream);
        cudaFree(result);
        releaseViews(b);
    }

    std::vector<std::unique_ptr<nixlAgent>> agents_;
};

TEST_P(ProxyOrderingTest, PutsVisibleBeforeCounter) {
    runOrderingCheck(kLayout);
}

// The sender kernel exits without waiting for completions; releasing the views
// must then wait for every queued put and signal (quiesce under load).
TEST_P(ProxyOrderingTest, ReleaseViewsWhileInFlight) {
    Buffers b(kLayout);
    setUpBuffers(kLayout, b);
    if (HasFatalFailure()) {
        return;
    }
    Result *result = nullptr;
    ASSERT_EQ(cudaMalloc(&result, sizeof(*result)), cudaSuccess);
    ASSERT_EQ(cudaMemset(result, 0, sizeof(*result)), cudaSuccess);
    senderKernel<<<kLayout.channels, 1>>>(b.src_mvh, b.dst_mvh, kLayout, 0, result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    Result host{};
    ASSERT_EQ(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(result);
    unsigned pending = 0;
    for (unsigned c = 0; c < kLayout.channels; ++c) {
        EXPECT_TRUE(host.sender_status[c] == NIXL_SUCCESS || host.sender_status[c] == NIXL_IN_PROG)
            << "channel " << c << ": " << host.sender_status[c];
        pending += host.sender_status[c] == NIXL_IN_PROG;
    }
    Logger() << "ReleaseViewsWhileInFlight: " << pending << " of " << kLayout.channels
             << " channel(s) still in flight at release";

    releaseViews(b);
    checkTarget(b, kLayout, kLayout.rounds);
}

/**
 * Fault injection in the EFA proxy (NIXL_EFA_PROXY_INJECT, read when the backend
 * is created): back-pressure must keep the order, and a failed put must reach the
 * GPU and stop every later signal on its ring.
 */
class ProxyFaultTest : public ProxyOrderingTest {
protected:
    // One channel, so put request n (from 1) is put (n - 1) % 4 of round (n - 1) / 4.
    static constexpr Layout kFaultLayout{1, 16};
    static constexpr unsigned kFailRound = 5;
    static constexpr unsigned kFailPut = 2;
    static constexpr unsigned kFailRequest = kFailRound * kPutsPerRound + kFailPut + 1;
    static constexpr unsigned long long kFaultTimeoutNs = 10ull * 1000 * 1000 * 1000;

    void
    SetUp() override {
        if (!hasCudaGpu()) {
            GTEST_SKIP() << "No CUDA-capable GPU is available";
        }
        if (backend() != "LIBFABRIC") {
            GTEST_SKIP() << "fault injection is implemented by the LIBFABRIC proxy only";
        }
#ifdef NDEBUG
        GTEST_SKIP() << "fault injection is compiled out of NDEBUG builds";
#endif
        ASSERT_EQ(cudaSetDevice(kDevId), cudaSuccess);
    }

    void
    TearDown() override {
        ProxyOrderingTest::TearDown();
        unsetenv("NIXL_EFA_PROXY_INJECT");
        ignore_.clear();
    }

    void
    createAgentsInjecting(const std::string &spec) {
        // The injected failures are expected to be logged.
        for (const char *rx : {"fault injection enabled",
                               "injected (post|completion) error",
                               "post failed on rail",
                               "GDRCopy disabled"}) {
            ignore_.push_back(std::make_unique<::gtest::LogIgnoreGuard>(std::string(rx)));
        }
        ASSERT_EQ(setenv("NIXL_EFA_PROXY_INJECT", spec.c_str(), 1), 0);
        createAgents();
    }

    std::vector<std::unique_ptr<::gtest::LogIgnoreGuard>> ignore_;

    /**
     * The failed put's round and every later round must stay unsignalled. With
     * receiver ordering and @p put_landed (an injected completion error on a put
     * that did reach the target), the target may still apply those rounds' adds,
     * having their data: then every round it signalled must have its data.
     */
    void
    runFailedPutCheck(const std::string &spec, bool put_landed = false) {
        createAgentsInjecting(spec);
        if (IsSkipped() || HasFatalFailure()) {
            return;
        }
        const Layout layout = kFaultLayout;
        Buffers b(layout);
        setUpBuffers(layout, b);
        if (HasFatalFailure()) {
            return;
        }

        Result *result = nullptr;
        ASSERT_EQ(cudaMalloc(&result, sizeof(*result)), cudaSuccess);
        ASSERT_EQ(cudaMemset(result, 0, sizeof(*result)), cudaSuccess);
        senderKernel<<<1, 1>>>(b.src_mvh, b.dst_mvh, layout, kFaultTimeoutNs, result);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        Result host{};
        ASSERT_EQ(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
        cudaFree(result);
        EXPECT_NE(host.sender_status[0], NIXL_SUCCESS) << "the failure did not reach the GPU";
        EXPECT_NE(host.sender_status[0], NIXL_IN_PROG) << "the sender timed out";

        releaseViews(b);

        if (receiverOrdering() && put_landed) {
            uint64_t counter = 0;
            ASSERT_EQ(cudaMemcpy(&counter, b.dst, sizeof(counter), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            const unsigned signalled = static_cast<unsigned>(counter / kCounterStep);
            Logger() << "receiver ordering: " << signalled << " round(s) signalled at the target, "
                     << kFailRound << " before the failed completion";
            EXPECT_GE(signalled, kFailRound);
            checkTarget(b, layout, signalled);
            return;
        }
        // Exactly the rounds before the failed put were signalled, with their data.
        checkTarget(b, layout, kFailRound);
    }
};

TEST_P(ProxyFaultTest, BackPressureKeepsOrder) {
    createAgentsInjecting("eagain_every=3");
    if (IsSkipped() || HasFatalFailure()) {
        return;
    }
    runOrderingCheck(kLayout);
}

TEST_P(ProxyFaultTest, FailedPostStopsLaterSignals) {
    runFailedPutCheck("post_error_at=" + std::to_string(kFailRequest));
}

TEST_P(ProxyFaultTest, FailedCompletionStopsLaterSignals) {
    runFailedPutCheck("cq_error_at=" + std::to_string(kFailRequest), /*put_landed=*/true);
}

// Without GDRCopy the target adds through CUDA copies; they must not wait for a
// kernel spinning on the legacy default stream (the counter's own receiver).
TEST_P(ProxyFaultTest, ManyRingsAddToOneCounterWithoutGdrCopy) {
    createAgentsInjecting("no_gdrcopy=1");
    if (IsSkipped() || HasFatalFailure()) {
        return;
    }
    runManyRingsOneCounter(500);
}

TEST_P(ProxyFaultTest, WithoutGdrCopyKeepsOrder) {
    createAgentsInjecting("no_gdrcopy=1");
    if (IsSkipped() || HasFatalFailure()) {
        return;
    }
    runOrderingCheck(kLayout, /*receiver_on_legacy_stream=*/true);
}

/** atomicAdd semantics at the target: errors reach the sender, and order holds. */
class ProxyAtomicTest : public ProxyOrderingTest {
protected:
    static constexpr Layout kSmallLayout{1, 1};
    static constexpr unsigned long long kAtomicTimeoutNs = 20ull * 1000 * 1000 * 1000;

    void
    SetUp() override {
        ProxyOrderingTest::SetUp();
        if (IsSkipped() || HasFatalFailure()) {
            return;
        }
        if (backend() != "LIBFABRIC") {
            GTEST_SKIP() << "checks the LIBFABRIC proxy's target-side adds";
        }
    }

    /** Run atomicLoopKernel to completion and return its status. */
    nixl_status_t
    runAtomics(const Buffers &b, size_t offset, unsigned count, cudaStream_t stream = nullptr) {
        nixl_status_t *status = nullptr;
        EXPECT_EQ(cudaMalloc(&status, sizeof(*status)), cudaSuccess);
        atomicLoopKernel<<<1, 1, 0, stream>>>(b.dst_mvh, offset, count, kAtomicTimeoutNs, status);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        nixl_status_t host = NIXL_IN_PROG;
        EXPECT_EQ(cudaMemcpy(&host, status, sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
        cudaFree(status);
        return host;
    }

    uint64_t
    readCounter(const Buffers &b, size_t offset) {
        uint64_t value = 0;
        EXPECT_EQ(cudaMemcpy(&value,
                             static_cast<char *>(static_cast<void *>(b.dst)) + offset,
                             sizeof(value),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return value;
    }
};

// An add the target cannot apply (here: its region is gone) fails at the sender
// instead of leaving the GPU waiting for a counter that never moves.
TEST_P(ProxyAtomicTest, AddToDeregisteredCounterFails) {
    const LogIgnoreGuard lig("atomicAdd to 0x[0-9a-f]+ failed");
    Buffers b(kSmallLayout);
    setUpBuffers(kSmallLayout, b);
    if (HasFatalFailure()) {
        return;
    }
    // Registered and known to the sender when its views were prepared, gone now.
    deregisterBuffer(kReceiver, b.dst);
    const nixl_status_t status = runAtomics(b, 0, 1);
    EXPECT_NE(status, NIXL_SUCCESS);
    EXPECT_NE(status, NIXL_IN_PROG) << "the sender timed out instead of seeing the failure";
    EXPECT_EQ(readCounter(b, 0), 0u);
    EXPECT_GE(lig.getIgnoredCount(), 1u);
    releaseViews(b);
}

// The adds of a ring are applied in order, so add(A) of a round is visible before
// add(B) on another counter.
TEST_P(ProxyAtomicTest, AddsOfOneRingStayOrdered) {
    constexpr unsigned kRounds = 2000;
    Buffers b(kSmallLayout);
    setUpBuffers(kSmallLayout, b);
    if (HasFatalFailure()) {
        return;
    }
    const size_t offset_a = 0;
    const size_t offset_b = kCounterStride;

    nixl_status_t *status = nullptr;
    unsigned long long *out = nullptr;
    ASSERT_EQ(cudaMalloc(&status, sizeof(*status)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&out, 2 * sizeof(*out)), cudaSuccess);
    cudaStream_t rx_stream, tx_stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&rx_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&tx_stream, cudaStreamNonBlocking), cudaSuccess);
    cudaFuncAttributes attr;
    ASSERT_EQ(cudaFuncGetAttributes(&attr, twoCounterReceiverKernel), cudaSuccess);
    ASSERT_EQ(cudaFuncGetAttributes(&attr, twoCounterSenderKernel), cudaSuccess);

    auto *base = static_cast<char *>(static_cast<void *>(b.dst));
    twoCounterReceiverKernel<<<1, 1, 0, rx_stream>>>(
        reinterpret_cast<unsigned long long *>(base + offset_a),
        reinterpret_cast<unsigned long long *>(base + offset_b),
        kRounds,
        kAtomicTimeoutNs,
        out);
    twoCounterSenderKernel<<<1, 1, 0, tx_stream>>>(
        b.dst_mvh, offset_a, offset_b, kRounds, kAtomicTimeoutNs, status);
    ASSERT_EQ(cudaStreamSynchronize(tx_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(rx_stream), cudaSuccess);

    nixl_status_t host_status = NIXL_IN_PROG;
    unsigned long long host_out[2] = {};
    ASSERT_EQ(cudaMemcpy(&host_status, status, sizeof(host_status), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_out, out, sizeof(host_out), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_status, NIXL_SUCCESS);
    EXPECT_EQ(host_out[1], kRounds) << "receiver saw counter B stop early";
    EXPECT_EQ(host_out[0], 0u) << "counter B ran ahead of counter A";
    EXPECT_EQ(readCounter(b, offset_a), kRounds);

    cudaStreamDestroy(rx_stream);
    cudaStreamDestroy(tx_stream);
    cudaFree(status);
    cudaFree(out);
    releaseViews(b);
}

// Threads of different rings add to one counter (GDRCopy read-modify-writes).
TEST_P(ProxyAtomicTest, ManyRingsAddToOneCounter) {
    runManyRingsOneCounter(2000);
}

// A target that restarts under the same name gets fresh rings: the sender keys its
// ring counts by the target's incarnation, so it does not continue the old
// instance's seq numbers (which the new instance would wait for in vain).
TEST_P(ProxyAtomicTest, TargetRestartStartsRingsOver) {
    constexpr unsigned kCount = 100;
    {
        Buffers b(kSmallLayout);
        setUpBuffers(kSmallLayout, b);
        if (HasFatalFailure()) {
            return;
        }
        EXPECT_EQ(runAtomics(b, 0, kCount), NIXL_SUCCESS);
        EXPECT_EQ(readCounter(b, 0), kCount);
        releaseViews(b);
        deregisterBuffer(kSender, b.src);
        deregisterBuffer(kReceiver, b.dst);
    }
    ASSERT_EQ(agents_[kSender]->invalidateRemoteMD(name(kReceiver)), NIXL_SUCCESS);
    recreateAgent(kReceiver);
    if (HasFatalFailure()) {
        return;
    }
    Buffers b(kSmallLayout);
    setUpBuffers(kSmallLayout, b);
    if (HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(runAtomics(b, 0, kCount), NIXL_SUCCESS) << "the restarted target's ring stalled";
    EXPECT_EQ(readCounter(b, 0), kCount);
    releaseViews(b);
}

// Deregistering the target region while adds stream in must neither crash the
// target's proxy threads nor leave the sender waiting.
TEST_P(ProxyAtomicTest, DeregisterDuringAtomics) {
    // Long enough (tens of microseconds per ordered add) to still run when the
    // region goes away.
    constexpr unsigned kCount = 20000;
    const LogIgnoreGuard lig("atomicAdd to 0x[0-9a-f]+ failed");
    Buffers b(kSmallLayout);
    setUpBuffers(kSmallLayout, b);
    if (HasFatalFailure()) {
        return;
    }
    nixl_status_t *status = nullptr;
    ASSERT_EQ(cudaMalloc(&status, sizeof(*status)), cudaSuccess);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    atomicLoopKernel<<<1, 1, 0, stream>>>(b.dst_mvh, 0, kCount, kAtomicTimeoutNs, status);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    deregisterBuffer(kReceiver, b.dst);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    nixl_status_t host_status = NIXL_IN_PROG;
    ASSERT_EQ(cudaMemcpy(&host_status, status, sizeof(host_status), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_NE(host_status, NIXL_IN_PROG) << "the sender timed out";
    EXPECT_NE(host_status, NIXL_SUCCESS) << "every add was applied: nothing raced";
    const uint64_t applied = readCounter(b, 0);
    EXPECT_LT(applied, kCount);
    Logger() << "DeregisterDuringAtomics: " << applied << " of " << kCount
             << " add(s) applied before the region went away, status " << host_status << ", "
             << lig.getIgnoredCount() << " reported failure(s)";
    cudaStreamDestroy(stream);
    cudaFree(status);
    releaseViews(b);
}

/**
 * A peer's handshake that arrives while this agent is loading the peer's metadata
 * must not be lost (it would make the first connection to the peer time out).
 * NIXL_LIBFABRIC_TEST_HANDSHAKE_DELAY_MS widens the window in the engine.
 */
class ProxyConnectionTest : public ProxyOrderingTest {
protected:
    void
    SetUp() override {
#ifdef NDEBUG
        GTEST_SKIP() << "the handshake delay hook is compiled out of NDEBUG builds";
#endif
        if (!hasCudaGpu()) {
            GTEST_SKIP() << "No CUDA-capable GPU is available";
        }
        ASSERT_EQ(cudaSetDevice(kDevId), cudaSuccess);
        ASSERT_EQ(setenv("NIXL_LIBFABRIC_TEST_HANDSHAKE_DELAY_MS", "500", 1), 0);
        createAgents();
    }

    void
    TearDown() override {
        ProxyOrderingTest::TearDown();
        unsetenv("NIXL_LIBFABRIC_TEST_HANDSHAKE_DELAY_MS");
    }
};

TEST_P(ProxyConnectionTest, HandshakeDuringMetadataLoad) {
    if (IsSkipped()) {
        return;
    }
    nixl_blob_t md0, md1;
    ASSERT_EQ(agents_[kSender]->getLocalMD(md0), NIXL_SUCCESS);
    ASSERT_EQ(agents_[kReceiver]->getLocalMD(md1), NIXL_SUCCESS);
    std::string remote;
    // The receiver connects first: its handshake (with its connection info) reaches
    // the sender, whose engine stalls in handleHandshake() before buffering it...
    ASSERT_EQ(agents_[kReceiver]->loadRemoteMD(md0, remote), NIXL_SUCCESS);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // ...while the sender creates the connection to the receiver.
    ASSERT_EQ(agents_[kSender]->loadRemoteMD(md1, remote), NIXL_SUCCESS);
    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(agents_[kSender]->makeConnection(name(kReceiver)), NIXL_SUCCESS);
    EXPECT_EQ(agents_[kReceiver]->makeConnection(name(kSender)), NIXL_SUCCESS);
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Logger() << "HandshakeDuringMetadataLoad: connections established in " << s << " s";
    EXPECT_LT(s, 10.0);
}

namespace {
    std::string
    testName(const testing::TestParamInfo<std::string> &info) {
        return info.param;
    }
} // namespace

INSTANTIATE_TEST_SUITE_P(libfabricProxy, ProxyOrderingTest, testing::Values("LIBFABRIC"), testName);

INSTANTIATE_TEST_SUITE_P(libfabricProxy, ProxyFaultTest, testing::Values("LIBFABRIC"), testName);

INSTANTIATE_TEST_SUITE_P(libfabricProxy,
                         ProxyConnectionTest,
                         testing::Values("LIBFABRIC"),
                         testName);

INSTANTIATE_TEST_SUITE_P(libfabricProxy, ProxyAtomicTest, testing::Values("LIBFABRIC"), testName);

} // namespace gtest::nixl::gpu::proxy_ordering
