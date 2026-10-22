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
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_transport.h"

class nixlLibfabricEngine;
class nixlLibfabricRailManager;
struct nixlLibfabricConnection;

/**
 * Device proxy transport on the engine's rails: its puts are engine writes and its control
 * messages engine sends (NIXL_LIBFABRIC_MSG_PROXY), received through the rails' proxy
 * callback. put -> atomicAdd ordering is kept by the target:
 *
 * - A put fragment's immediate data names its stream (ring id) and epoch slot: the index of
 *   the atomicAdd that will follow it on the stream, mod kEpochs. A stream is one ring's
 *   traffic to one target; the target knows the sender from the write's source address.
 * - An atomicAdd is sent at once to the target's rail where the stream's small puts land,
 *   with its index (seq) and the count its epoch slot reaches once its puts are in. The
 *   target applies a stream's adds in seq order, each once that count is reached (GDRCopy
 *   for GPU memory, a CPU atomic for host memory), and acks; the sender completes the
 *   atomicAdd on the ack.
 * - A slot is reused kEpochs atomicAdds later; a ring holds at most ring_depth (<= kEpochs)
 *   requests, so the slot's previous add was applied by then.
 * - A stream waits for the engine handshake with its target, which lets the target
 *   attribute its writes.
 *
 * Proxy thread t posts small puts on its share of the GPU's rails and reads the CQs of that
 * share; thread 0 also reads the engine's control rail. The engine progress thread must be
 * off (the proxy threads replace it).
 */
class nixlLibfabricProxy final : public nixl::proxyTransport {
public:
    explicit nixlLibfabricProxy(nixlLibfabricEngine &engine);
    ~nixlLibfabricProxy() override;

    nixlLibfabricProxy(const nixlLibfabricProxy &) = delete;
    nixlLibfabricProxy &
    operator=(const nixlLibfabricProxy &) = delete;

    /** Rails of the EFA devices next to the current CUDA device (all rails if unknown). */
    static std::vector<size_t>
    gpuRails(const nixlLibfabricRailManager &rails);

    /** Local registrations: atomicAdds apply only inside one (GPU ones are mapped now). */
    void
    onRegister(uintptr_t addr, size_t len, bool is_vram);
    /** Waits for adds being applied there; later ones fail. */
    void
    onDeregister(uintptr_t addr, size_t len);

    // nixl::proxyTransport.
    [[nodiscard]] nixl_status_t
    init(const nixl::proxyConfig &config) override;
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
    [[nodiscard]] nixl_status_t
    shutdown() override;

private:
    struct Request;
    struct Thread;
    struct Stream;
    struct Ring;
    struct RxRing;
    struct Counters;

    Thread &
    threadOf(uint32_t channel) {
        return *threads_[channel % threads_.size()];
    }

    /** Whether the target of @p conn attributes our writes (handshake done). */
    bool
    ready(const nixlLibfabricConnection &conn) const;
    nixl_status_t
    post(Thread &th, Ring &ring, Request &req, const nixl::proxyBackendSubmission &sub);
    nixl_status_t
    postPut(Thread &th, Stream &stream, Request &req, const nixl::proxyBackendSubmission &sub);
    nixl_status_t
    postAtomic(Thread &th, Stream &stream, Request &req, const nixl::proxyBackendSubmission &sub);
    /** Post the operations a ring parked until its target was ready. */
    void
    flushParked(Thread &th, Ring &ring);
    void
    pollRails(Thread &th);
    /** Send a control message from @p rail to @p dest. */
    nixl_status_t
    sendMsg(size_t rail, uint64_t dest, const void *msg, size_t len);
    /** A put fragment failed: its stream's adds from its epoch on cannot be satisfied. */
    void
    onPutFailed(Request &req);
    /** Tell the targets of failed streams to fail their adds instead of waiting. */
    void
    sendAborts();

    // Target side.
    /** The rails' proxy callback. */
    void
    onTraffic(uint64_t imm, const void *data, size_t len, uint32_t agent, uint16_t rail);
    RxRing &
    rxRing(uint32_t agent, uint32_t ring_id);
    /** Apply the stream's adds whose turn and puts have come. */
    void
    drain(RxRing &rx);
    /** Whether a registration holds [lo, hi); under regions_mutex_. */
    bool
    covered(uintptr_t lo, uintptr_t hi) const;

    Request *
    allocRequest(Thread &th);
    void
    freeRequest(Thread &th, Request *req);

    nixlLibfabricEngine &engine_;
    uint64_t id_ = 0; // this instance, for the per-thread stream caches
    uint32_t max_peers_ = 0;
    std::vector<size_t> rails_; // the GPU's rails; thread t uses rails_[i] for i % threads == t
    std::vector<std::unique_ptr<Thread>> threads_;
    std::unique_ptr<Ring[]> rings_; // channel * max_peers + peer

    // Sender: streams whose puts failed, to abort at their targets.
    std::mutex aborts_mutex_;
    std::vector<Stream *> aborts_;
    std::atomic<bool> aborts_pending_{false};

    // Sender: the next ring id per target, from a random start per instance (a restarted
    // sender does not reuse the counts its target may still hold).
    uint32_t ring_id_base_ = 0;
    std::mutex ring_ids_mutex_;
    std::map<const nixlLibfabricConnection *,
             std::pair<std::weak_ptr<nixlLibfabricConnection>, uint32_t>>
        ring_ids_;

    // Target: streams by (sender agent index, ring id); never freed before the proxy.
    std::mutex rx_mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<RxRing>> rx_rings_;
    // Target: local registrations (base -> length); adds hold it shared while applied.
    std::shared_mutex regions_mutex_;
    std::multimap<uintptr_t, size_t> regions_;
    std::unique_ptr<Counters> counters_;
};

#endif // NIXL_LIBFABRIC_DEVICE_PROXY
#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_H
