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

#ifdef HAVE_NIXL_DEVICE_API

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#include "device/proxy/proxy_backend_ops.h"
#include "device/proxy/proxy_config.h"
#include "libfabric/libfabric_common.h"

class nixlLibfabricEngine;
struct nixlLibfabricConnection;

/**
 * EFA implementation of the device proxy's backend operations.
 *
 * The engine owns one instance when the device_proxy backend param is set and
 * hands its callbacks to nixlProxyRuntime. Resources follow the design:
 *
 *  - Proxy thread t owns an EP, CQ and AV on every local rail, created in the
 *    engine's per-rail fi_domain, so memory registrations and keys are reused.
 *    t = channel_id % effectiveThreadCount(), the rule ProxyWorker uses to own
 *    channels, so every CQ has exactly one polling thread and no locks are
 *    needed on the data path.
 *  - A put is one fi_writemsg per fragment (striped across rails at or above
 *    the engine's striping threshold), completed through CQ entries whose
 *    op_context identifies the exact request.
 *  - An atomicAdd is held in its ring's fence until every earlier put on that
 *    ring (channel, peer) has completed, then sent as a small message to the
 *    counter's owner thread at the target, which applies the add through
 *    GDRCopy (VRAM) or a plain CPU add (DRAM).
 */
class nixlLibfabricProxy {
public:
    /** Wire record for an atomicAdd, applied by the target's owner thread. */
    struct AtomicMsg {
        uint64_t remote_addr;
        uint64_t value;
        uint64_t reserved[2];
    };

    explicit nixlLibfabricProxy(nixlLibfabricEngine &engine);
    ~nixlLibfabricProxy();

    nixlLibfabricProxy(const nixlLibfabricProxy &) = delete;
    nixlLibfabricProxy &
    operator=(const nixlLibfabricProxy &) = delete;

    /** Build the callback table handed to nixlProxyRuntime::create(). */
    [[nodiscard]] nixlProxyBackendOps
    makeOps();

    /** Local registrations the target side may apply atomics to. */
    void
    onRegister(uintptr_t addr, size_t len, bool is_vram, int device_id);
    void
    onDeregister(uintptr_t addr);

    /** Proxy section appended to the engine's connection info. */
    [[nodiscard]] std::string
    serializeConnInfo() const;

    /** Parse a peer's proxy section into its connection; tolerates absence. */
    static nixl_status_t
    parseConnInfo(const std::string &blob, nixlLibfabricConnection &conn);

    /** Split "<engine blob><proxy blob><len><magic>" into its two parts. */
    static void
    splitConnInfo(const std::string &in, std::string &engine_part, std::string &proxy_part);

    /** Append a proxy section to the engine's connection info. */
    static std::string
    joinConnInfo(const std::string &engine_part, const std::string &proxy_part);

private:
    struct Request;
    struct RingFence;
    struct RecvBuf;
    struct PendingPost;
    struct PeerAddrs;
    struct RailRes;
    struct Thread;
    struct CounterMap;
    struct Inject;

    // nixlProxyBackendOps callbacks.
    nixl_status_t
    init(const nixlProxyConfig &config);
    nixl_status_t
    submit(const nixlBackendProxySubmission &sub, nixlBackendProxyRequest &request);
    nixl_status_t
    checkCompletion(const nixlBackendProxyRequest &request);
    nixl_status_t
    quiesce(uint32_t channel, uint32_t peer);
    nixl_status_t
    progress(uint32_t channel, uint32_t peer);
    nixl_status_t
    shutdown();

    // Data path helpers; all run on the owning proxy thread.
    nixl_status_t
    submitPut(Thread &th, const nixlBackendProxySubmission &sub, Request *req);
    nixl_status_t
    submitAtomic(Thread &th, const nixlBackendProxySubmission &sub, Request *req);
    /** Index into a buffer's rails for an unstriped put. */
    size_t
    unstripedRail(const Thread &th, RingFence &fence, size_t nrails) const;
    void
    releaseFence(Thread &th, RingFence &fence);
    void
    sendAtomic(Thread &th, Request *req);
    ssize_t
    tryPost(Thread &th, const PendingPost &pp);
    /** tryPost(), timed when profiling. */
    ssize_t
    postTimed(Thread &th, const PendingPost &pp);
    void
    post(Thread &th, PendingPost &&pp);
    void
    drainRetries(Thread &th);
    void
    pollCqs(Thread &th);
    void
    reportCqError(Thread &th, uint32_t rail, int err);
    void
    completeFragment(Thread &th, Request *req, nixl_status_t status);
    void
    handleRecv(Thread &th, RecvBuf *buf);
    nixl_status_t
    applyAtomic(const AtomicMsg &msg);
    PeerAddrs *
    peerAddrs(Thread &th, const std::shared_ptr<nixlLibfabricConnection> &conn);
    fi_addr_t
    railAddr(Thread &th,
             PeerAddrs &pa,
             const nixlLibfabricConnection &conn,
             size_t rail,
             size_t remote_ep);
    fi_addr_t
    homeAddr(Thread &th, PeerAddrs &pa, const nixlLibfabricConnection &conn, size_t owner);
    void
    dropPeer(Thread &th, PeerAddrs &pa);
    Request *
    allocRequest(Thread &th);
    void
    freeRequest(Thread &th, Request *req);
    nixl_status_t
    postRecv(Thread &th, RecvBuf *buf);
    void
    releaseThread(Thread &th);

    nixlLibfabricEngine &engine_;
    nixlProxyConfig config_{};
    uint32_t threads_ = 0;
    size_t rails_ = 0;
    bool delivery_complete_ = true;
    bool rail_per_thread_ = true; // efa_proxy_rail_policy: "thread" (default) or "ring"
    bool profile_ = false; // NIXL_EFA_PROXY_PROFILE: per-stage timers, logged at shutdown
    std::unique_ptr<Inject> inject_; // NIXL_EFA_PROXY_INJECT: test-only fault injection
    std::vector<std::unique_ptr<Thread>> thread_state_;

    // Target-side registrations, keyed by base address.
    struct Region {
        size_t len;
        bool is_vram;
        int device_id;
    };

    mutable std::mutex regions_mutex_;
    std::map<uintptr_t, Region> regions_;
    std::unique_ptr<CounterMap> counters_;
};

#endif // HAVE_NIXL_DEVICE_API
#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_H
