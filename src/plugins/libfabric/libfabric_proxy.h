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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#include "device/proxy/proxy_backend_ops.h"
#include "device/proxy/proxy_config.h"
#include "libfabric/libfabric_common.h"
#include "libfabric_proxy_wire.h"

class nixlLibfabricEngine;
struct nixlLibfabricConnection;

/**
 * EFA implementation of the device proxy's backend operations.
 *
 * The engine owns one instance when the device_proxy backend param is set and
 * hands its callbacks to nixlProxyRuntime. Resources follow the design:
 *
 *  - Proxy thread t owns an EP, CQ and AV on every local rail, created in the
 *    engine's per-rail fi_domain, so memory registrations and keys are reused,
 *    plus a control EP on its home rail for atomicAdd records and acks.
 *  - A put is one fi_writemsg per fragment with FI_DELIVERY_COMPLETE (striped
 *    across rails at or above the engine's striping threshold), completed
 *    through CQ entries whose op_context identifies the exact request.
 *  - An atomicAdd is held in its ring's fence until every earlier put and the
 *    previous atomicAdd on that ring (channel, peer) have completed, then sent
 *    (libfabric_proxy_wire.h) to the counter's owner thread at the target. The
 *    owner applies it through GDRCopy (VRAM), CUDA copies on its own
 *    non-blocking stream (VRAM without GDRCopy) or a CPU atomic (DRAM), and acks
 *    with the result: an atomicAdd completes once applied, and a failure at the
 *    target reaches the sender's GPU.
 *  - efa_proxy_ordering=receiver (opt-in prototype) orders at the target
 *    instead: every put fragment carries its ring's key as remote CQ data and goes
 *    to one data EP of the target per ring, whose thread counts it; an atomicAdd
 *    is sent at once with its ring's put count and sequence number, and its owner
 *    applies it once the ring's received puts and applied atomicAdds reach them.
 *
 * Threading: ProxyWorker owns channel c on thread c % effectiveThreadCount() and
 * calls submit/check_completion/progress/quiesce for a ring only from that
 * thread (checked), visiting ring (t, 0) on every pass; that call polls the
 * thread's CQs (every pass while it has work, every efa_proxy_idle_poll_us when
 * idle). Every CQ therefore has exactly one polling thread. The data path
 * takes no lock except, while applying an add to VRAM, its own thread's
 * (uncontended) registration lock and a brief lookup in the shared GDRCopy page
 * map.
 */
class nixlLibfabricProxy {
public:
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
    /** Waits for adds being applied to the registration, then forgets it. */
    void
    onDeregister(uintptr_t addr, size_t len);

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
    struct OpCtx;
    struct Request;
    struct RingFence;
    struct RecvBuf;
    struct AckBuf;
    struct PendingPost;
    struct PendingAck;
    struct PeerAddrs;
    struct RailRes;
    struct Thread;
    struct CounterMap;
    struct Inject;
    struct TxRing;
    struct RxRing;
    struct Deferred;

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

    /** The thread state that serves @p channel; fatal if called from another worker. */
    Thread &
    ownerOf(uint32_t channel);
    void
    checkOwner(Thread &th);

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
    /** Drain one CQ; @p rail names it in logs (the rail count for the control CQ). */
    void
    pollCq(Thread &th, struct fid_cq *cq, uint32_t rail, uint64_t &reads);
    void
    reportCqError(Thread &th, uint32_t rail, int err);
    void
    completeFragment(Thread &th, Request *req, nixl_status_t status);
    /** A fragment failed before or instead of completing; an atomicAdd then gets no ack. */
    void
    failFragment(Thread &th, Request *req, nixl_status_t status);
    /** Fail atomicAdds whose ack is overdue (their target died after receiving them). */
    void
    expireAcks(Thread &th, uint64_t now_ns);

    // Proxy-to-proxy messages on the home endpoints.
    void
    handleRecv(Thread &th, RecvBuf *buf, size_t len);
    void
    handleAtomic(Thread &th, const nixlLibfabricProxyWire::atomicAddMsg &msg);
    /** Apply a received atomicAdd now and ack it with the result. */
    void
    applyAndAck(Thread &th, const nixlLibfabricProxyWire::atomicAddMsg &msg, fi_addr_t reply);
    void
    handleAck(Thread &th, const nixlLibfabricProxyWire::atomicAckMsg &ack);
    fi_addr_t
    replyAddr(Thread &th, const nixlLibfabricProxyWire::atomicAddMsg &msg);
    void
    sendAck(Thread &th, const PendingAck &ack);
    /** False when the ack has to wait (no buffer, or -FI_EAGAIN). */
    bool
    postAck(Thread &th, const PendingAck &ack);
    nixl_status_t
    applyAtomic(Thread &th, uint64_t addr, uint64_t value);
    /** Make completed RDMA writes to the GPU of @p addr visible to it (receiver ordering). */
    nixl_status_t
    flushRdmaWrites(Thread &th, uint64_t addr);

    // Receiver-side ordering (and the test-only immediate-data puts).
    /** The sender-side counts of a ring (channel, peer) to @p conn; re-keyed after a failure. */
    TxRing *
    txRing(Thread &th,
           RingFence &fence,
           const std::shared_ptr<nixlLibfabricConnection> &conn,
           uint32_t ring);
    /** Fail the atomicAdds after a failed operation of @p req's ring, as the fence does. */
    void
    failRing(Thread &th, Request *req, nixl_status_t status);
    /** A put fragment with remote CQ data @p imm arrived: count it for its ring and epoch. */
    void
    handleImm(Thread &th, uint32_t imm);
    RxRing *
    rxRing(Thread &th, uint32_t ring_key);
    /** Apply the deferred atomicAdds whose rings caught up; expire overdue ones. */
    void
    drainDeferred(Thread &th);
    /** Apply @p d (its puts arrived and were flushed) if it is its ring's next one. */
    bool
    tryApplyDeferred(Thread &th, Deferred &d);
    fi_addr_t
    dataAddr(Thread &th,
             PeerAddrs &pa,
             const nixlLibfabricConnection &conn,
             size_t rail,
             size_t remote_thread,
             size_t remote_rail);
    /** The target thread that counts a put of ring @p key landing on @p remote_rail. */
    static size_t
    rxThread(const nixlLibfabricConnection &conn, uint32_t key, size_t remote_rail);
    void
    reportRx() const;
#ifdef HAVE_CUDA
    nixl_status_t
    addWithCuda(Thread &th, int device_id, uint64_t addr, uint64_t value);
#endif

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
    bool rail_per_thread_ = true; // efa_proxy_rail_policy: "thread" (default) or "ring"
    uint64_t idle_poll_ns_ = 0; // efa_proxy_idle_poll_us (0: poll on every pass)
    bool profile_ = false; // NIXL_EFA_PROXY_PROFILE: per-stage timers, logged at shutdown
    bool rx_trace_ = false; // NIXL_EFA_PROXY_RXTRACE: log early receiver-ordering events
    // efa_proxy_ordering: put -> atomicAdd order kept by the sender's fence (default)
    // or by the target (receiver).
    bool receiver_order_ = false;
    // Test-only (ignored in NDEBUG builds): sender ordering with puts carrying
    // remote CQ data to the target's proxy data EPs (efa_proxy_test_put_imm), and
    // atomicAdds sent without waiting for anything (efa_proxy_test_fence_off: no
    // ordering at all).
    bool test_put_imm_ = false;
    bool test_fence_off_ = false;
    size_t test_data_rx_post_ = 0; // efa_proxy_test_data_rx_post: receives per data EP
    // efa_proxy_rx_flush (receiver ordering, default on): flush GPUDirect RDMA writes
    // to the counter's GPU before applying adds whose puts arrived. A put's CQ entry
    // only says the NIC wrote it: without the flush the GPU was seen to read a put
    // still in flight after its counter moved (the counter write takes another path).
    bool rx_flush_ = true;
    size_t data_rx_size_ = 0; // efa_proxy_data_rx_size: data EP receive queue size
    // Puts carry remote CQ data to proxy data EPs; this proxy also accepts them.
    bool imm_puts_ = false;
    // efa_proxy_rx_thread: the thread that counts a put with remote CQ data is the
    // ring's (ring, default) or the one polling the destination rail (rail).
    bool rx_by_rail_ = false;
    uint64_t incarnation_ = 0; // published; peers restart their ring counts with it
    std::vector<uint32_t> rx_rails_; // data CQs every thread polls when imm_puts_
    std::unique_ptr<Inject> inject_; // NIXL_EFA_PROXY_INJECT: tests only, off under NDEBUG
    std::vector<std::unique_ptr<Thread>> thread_state_;

    // Target-side registrations by base address; duplicates and overlaps allowed.
    struct Region {
        size_t len;
        bool is_vram;
        int device_id;
    };

    /** A registration holding the 8-byte word at addr, or nullptr; under a region lock. */
    const Region *
    findRegion(uint64_t addr) const;
    /** Whether a registration overlaps [addr, addr + len); under the region locks. */
    bool
    regionOverlaps(uintptr_t addr, size_t len) const;
    /**
     * Lock the registrations for a change: every proxy thread's region lock, in
     * order, so the change waits for adds being applied (an add holds only its
     * own thread's lock and never waits for a writer for long).
     */
    std::vector<std::unique_lock<std::mutex>>
    lockRegions();

    std::mutex regions_write_mutex_; // serializes registration changes
    std::multimap<uintptr_t, Region> regions_;
    std::unique_ptr<CounterMap> counters_;

    // Target-side ring state, shared by the thread that counts a ring's puts and
    // the owners of its counters; never freed while the proxy lives.
    std::mutex rx_rings_mutex_;
    std::unordered_map<uint32_t, std::unique_ptr<RxRing>> rx_rings_;
};

#endif // HAVE_NIXL_DEVICE_API
#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_H
