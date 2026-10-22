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
#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_H

#ifdef NIXL_LIBFABRIC_DEVICE_PROXY

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_transport.h"

class nixlLibfabricEngine;
class nixlLibfabricRailManager;

/**
 * Device proxy transport on the engine's rails. Puts are engine writes and control messages
 * engine sends, both tagged NIXL_LIBFABRIC_MSG_PROXY and handed back by the rails' proxy
 * callback. The target keeps put -> atomicAdd order:
 *
 * - A put's immediate data names its stream (one ring's traffic to one target) and epoch slot
 *   (the index of the atomicAdd that follows it, mod 256). The target counts puts per slot.
 * - An atomicAdd is a message with its index and the count its slot reaches once its puts are
 *   in. The target applies a stream's adds in order, each once counted (GDRCopy for GPU
 *   memory, a CPU atomic for host memory), and acks; the sender completes it on the ack.
 * - A stream waits for the engine handshake with its target, which lets the target name the
 *   sender of a write. Proxy thread t posts small puts on its share of the GPU's rails and
 *   reads their CQs; the engine progress thread must be off.
 */
class nixlLibfabricProxy final : public nixl::proxyTransport {
public:
    /** @throws std::runtime_error if the GPU has no rails. */
    nixlLibfabricProxy(nixlLibfabricEngine &engine, const nixl::proxyConfig &config);
    ~nixlLibfabricProxy() override;

    /** Rails of the EFA devices next to the current CUDA device (all rails if unknown). */
    static std::vector<size_t>
    gpuRails(const nixlLibfabricRailManager &rails);

    /** atomicAdds apply only inside a registration; a GPU one is mapped with GDRCopy now. */
    void
    onRegister(uintptr_t addr, size_t len, bool is_vram);
    void
    onDeregister(uintptr_t addr, size_t len);

    [[nodiscard]] nixl_status_t
    submit(const nixl::proxyBackendSubmission &sub, nixl::proxyBackendRequest &request) override;
    [[nodiscard]] nixl_status_t
    checkCompletion(uint32_t channel,
                    uint32_t peer,
                    const nixl::proxyBackendRequest &request) override;
    void
    progress(uint32_t channel, uint32_t peer) noexcept override;
    [[nodiscard]] nixl_status_t
    quiesce(uint32_t channel, uint32_t peer) override;

private:
    struct Request;
    struct Thread;
    struct Stream;
    struct Ring;
    struct RxStream;
    struct Region;

    /** The proxy thread that owns the ring (the runtime's proxyConfig::threadShare). */
    Thread &
    threadOf(uint32_t channel, uint32_t peer);
    Ring &
    ring(uint32_t channel, uint32_t peer);
    nixl_status_t
    post(Thread &th, Request &req, const nixl::proxyBackendSubmission &sub);
    nixl_status_t
    postPut(Thread &th, Stream &stream, Request &req, const nixl::proxyBackendSubmission &sub);
    nixl_status_t
    postAtomic(Thread &th, Stream &stream, Request &req, const nixl::proxyBackendSubmission &sub);
    nixl_status_t
    sendMsg(size_t rail, uint64_t dest, const void *msg, size_t len);

    // Target side.
    void
    onTraffic(uint64_t imm, const void *data, size_t len, uint32_t agent, uint16_t rail);
    RxStream &
    rxStream(uint32_t agent, uint32_t stream_id);
    /** Apply the stream's adds whose turn and puts have come, then ack them. */
    void
    drain(RxStream &rx);
    nixl_status_t
    apply(uint64_t addr, uint64_t value);

    nixlLibfabricEngine &engine_;
    nixl::proxyConfig config_;
    uint64_t id_; // this instance, for the per-thread stream caches
    std::vector<size_t> rails_; // the GPU's rails; thread t uses rails_[i] for i % threads == t
    std::vector<std::unique_ptr<Thread>> threads_;
    std::unique_ptr<Ring[]> rings_;
    std::vector<uint32_t> ring_thread_; // the thread of ring channel * max_peers + peer

    std::mutex rx_mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<RxStream>> rx_streams_; // (agent, stream id)
    std::shared_mutex regions_mutex_; // held shared while an add is applied
    std::map<uintptr_t, Region> regions_; // registrations by base address
    void *gdr_ = nullptr; // gdr_t
    bool flush_rdma_writes_ = false; // the GPU does not order NIC writes before counter writes
};

#endif // NIXL_LIBFABRIC_DEVICE_PROXY
#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_H
