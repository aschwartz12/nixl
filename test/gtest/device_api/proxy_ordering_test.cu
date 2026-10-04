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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

namespace gtest::nixl::gpu::proxy_ordering {

using namespace ::nixl_test::proxy_ordering;

constexpr unsigned kChannels = 4;
constexpr Layout kLayout{kChannels, 128};
static_assert(kChannels <= kMaxChannels);
constexpr unsigned long long kTimeoutNs = 60ull * 1000 * 1000 * 1000;

/** Backend, and whether EFA requests FI_DELIVERY_COMPLETE (ignored by other backends). */
using TestParams = std::tuple<std::string, bool>;

class ProxyOrderingTest : public testing::TestWithParam<TestParams> {
protected:
    static constexpr size_t kSender = 0;
    static constexpr size_t kReceiver = 1;
    static constexpr uint64_t kDevId = 0;

    std::string
    backend() const {
        return std::get<0>(GetParam());
    }

    nixl_b_params_t
    backendParams() const {
        nixl_b_params_t params = {{"device_proxy", "true"},
                                  {"proxy_channel_count", std::to_string(kChannels)},
                                  {"proxy_thread_count", "4"},
                                  {"proxy_max_peers", "2"}};
        if (backend() == "LIBFABRIC") {
            params["efa_proxy_delivery_complete"] = std::get<1>(GetParam()) ? "true" : "false";
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

    /** After the rings drained: every channel signalled exactly @p rounds rounds with their data. */
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

    /** Stream every round on every channel and check it on the target GPU. */
    void
    runOrderingCheck(const Layout &layout) {
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

        receiverKernel<<<layout.channels, 256, 0, rx_stream>>>(
            static_cast<uint8_t *>(static_cast<void *>(b.dst)), layout, kTimeoutNs, result);
        cudaEventRecord(start, tx_stream);
        senderKernel<<<layout.channels, 1, 0, tx_stream>>>(
            b.src_mvh, b.dst_mvh, layout, kTimeoutNs, result);
        cudaEventRecord(stop, tx_stream);
        ASSERT_EQ(cudaStreamSynchronize(tx_stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(rx_stream), cudaSuccess);

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
            << "data not visible when its counter was: channel " << host.first_channel
            << " round " << host.first_round << " word " << (host.first_bad - 1) << " value 0x"
            << std::hex << host.first_value << " expected 0x" << host.first_expected;

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
                               "post failed on rail"}) {
            ignore_.push_back(std::make_unique<::gtest::LogIgnoreGuard>(std::string(rx)));
        }
        ASSERT_EQ(setenv("NIXL_EFA_PROXY_INJECT", spec.c_str(), 1), 0);
        createAgents();
    }

    std::vector<std::unique_ptr<::gtest::LogIgnoreGuard>> ignore_;

    /** The failed put's round and every later round must stay unsignalled. */
    void
    runFailedPutCheck(const std::string &spec) {
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
    runFailedPutCheck("cq_error_at=" + std::to_string(kFailRequest));
}

namespace {
    std::string
    testName(const testing::TestParamInfo<TestParams> &info) {
        return std::get<1>(info.param) ? "DeliveryComplete" : "TransmitComplete";
    }
} // namespace

INSTANTIATE_TEST_SUITE_P(libfabricProxy,
                         ProxyOrderingTest,
                         testing::Values(TestParams{"LIBFABRIC", true}),
                         testName);

INSTANTIATE_TEST_SUITE_P(libfabricProxy,
                         ProxyFaultTest,
                         testing::Values(TestParams{"LIBFABRIC", true}),
                         testName);

// Without FI_DELIVERY_COMPLETE the fence rests on transmit completion only; run
// explicitly (--gtest_also_run_disabled_tests) to measure whether the flag matters.
INSTANTIATE_TEST_SUITE_P(DISABLED_libfabricProxy,
                         ProxyOrderingTest,
                         testing::Values(TestParams{"LIBFABRIC", false}),
                         testName);

} // namespace gtest::nixl::gpu::proxy_ordering
