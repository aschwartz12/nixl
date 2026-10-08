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
 * EFA implementation of the device proxy's backend operations, ordering put ->
 * atomicAdd at the target (libfabric_proxy_wire.h; design and flow in
 * EFA_PROXY_RECEIVER_ORDERING.md).
 *
 * The engine owns one instance when the device_proxy backend param is set and
 * hands its callbacks to nixlProxyRuntime. Per proxy thread t:
 *
 *  - a data EP (with its own CQ and AV) on every rail, in the engine's per-rail
 *    fi_domain (so registrations and keys are shared). Puts leave from them; on
 *    the receive rails (next to the GPU) t polls the data CQ of the rails it owns
 *    (wire::railThread()), counts the puts that land there and receives the
 *    atomicAdd records and ring aborts sent there. Its data EP on its home rail
 *    sends its atomicAdd records, acks and ring aborts and receives its acks.
 *    There is no separate control EP.
 *
 * Sender side, per ring (channel, peer): every put fragment is an RDMA write with
 * FI_DELIVERY_COMPLETE and remote CQ data (ring key, epoch) to the data EP of the
 * target thread that receives on the destination rail; an atomicAdd is sent at
 * once to the target's data EP on the rail where this thread's small puts land,
 * and completes on the ack of the thread polling it, i.e. once applied. A ring
 * waits for the engine handshake with its target (the ring key holds the sender's
 * index at the target) before its first operation leaves.
 *
 * Target side: the receiving thread applies an atomicAdd (GDRCopy, CUDA copy or
 * CPU atomic) once its ring's epoch slot counted its puts and the ring's previous
 * atomicAdd was applied, then acks with the result.
 *
 * Threading: ProxyWorker owns channel c on thread c % effectiveThreadCount() and
 * calls submit/check_completion/progress/quiesce for a ring only from that thread
 * (checked), visiting ring (t, 0) on every pass; that call polls the thread's CQs
 * (every pass while it has work, every efa_proxy_idle_poll_us when idle). Every CQ
 * has one polling thread. Receive-side ring state is shared between the thread
 * counting a ring's puts and the owners of its counters through atomics.
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
    struct Ring;
    struct TxRing;
    struct RxRing;
    struct RxOwned;
    struct Parked;
    struct RecvBuf;
    struct RecvPool;
    struct CtlBuf;
    struct PendingCtl;
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

    /** The thread state that serves @p channel; fatal if called from another worker. */
    Thread &
    ownerOf(uint32_t channel);
    void
    checkOwner(Thread &th);

    // Sender side; all on the owning proxy thread.
    /** Post @p sub for @p req on @p ring, or set @p parked if it must wait for a handshake. */
    nixl_status_t
    submitOp(Thread &th,
             Ring &ring,
             const nixlBackendProxySubmission &sub,
             Request *req,
             bool &parked);
    /** @p tx: the ring's counts towards the target, nullptr if it published no proxy. */
    nixl_status_t
    submitPut(Thread &th,
              Ring &ring,
              const nixlBackendProxySubmission &sub,
              Request *req,
              TxRing *tx);
    nixl_status_t
    submitAtomic(Thread &th,
                 Ring &ring,
                 const nixlBackendProxySubmission &sub,
                 Request *req,
                 TxRing *tx);
    /**
     * The sender-side counts of @p ring towards @p conn's agent, once the engine
     * handshake with it gave the sender's index there (part of the ring key):
     * nullptr until then, or (with @p status set) if they cannot be had.
     */
    TxRing *
    txRing(Thread &th,
           Ring &ring,
           const std::shared_ptr<nixlLibfabricConnection> &conn,
           nixl_status_t &status);
    /** Post the requests parked behind a pending handshake whose rings are ready. */
    void
    replayParked(Thread &th, uint64_t now_ns);
    /** Index into a buffer's rails for an unstriped put. */
    size_t
    unstripedRail(const Thread &th, Ring &ring, size_t nrails) const;
    /**
     * Position among @p conn's receive rails of the remote buffer rail @p reps[sel]
     * (or of the next of @p reps the target receives on), -1 if none.
     */
    static int
    receiveIndex(const PeerAddrs &pa, const std::vector<size_t> &reps, size_t sel, size_t &rep);
    /** Fail @p req's ring's later atomicAdds, at the sender and (abort) at the target. */
    void
    failRing(Thread &th, Request *req, nixl_status_t status);
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
    /** Drain one CQ; @p rail names it in logs. */
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

    // Control messages (atomicAdd records, acks, ring aborts) on the data EPs.
    void
    handleRecv(Thread &th, RecvBuf *buf, size_t len);
    void
    handleAtomic(Thread &th, const nixlLibfabricProxyWire::atomicAddMsg &msg);
    void
    handleAck(Thread &th, const nixlLibfabricProxyWire::atomicAckMsg &ack);
    void
    handleAbort(Thread &th, const nixlLibfabricProxyWire::ringAbortMsg &abort);
    fi_addr_t
    replyAddr(Thread &th, const nixlLibfabricProxyWire::atomicAddMsg &msg);
    /** Queue a control message (ack or abort) in order; posted as buffers allow. */
    void
    sendCtl(Thread &th, const PendingCtl &ctl);
    /** False when the message has to wait (no buffer, or -FI_EAGAIN). */
    bool
    postCtl(Thread &th, const PendingCtl &ctl);
    nixl_status_t
    applyAtomic(Thread &th, uint64_t addr, uint64_t value);
    /** Make completed RDMA writes visible to every GPU with registered memory. */
    nixl_status_t
    flushRdmaWrites(Thread &th);
#ifdef HAVE_CUDA
    nixl_status_t
    addWithCuda(Thread &th, int device_id, uint64_t addr, uint64_t value);
#endif

    // Receive side.
    /** A put fragment with remote CQ data @p imm arrived: count it for its ring and epoch. */
    void
    handleImm(Thread &th, uint32_t imm);
    RxRing *
    rxRing(Thread &th, uint32_t ring_key);
    /** Apply this thread's waiting atomicAdds whose rings caught up; expire overdue ones. */
    void
    drainDeferred(Thread &th);

    PeerAddrs *
    peerAddrs(Thread &th, const std::shared_ptr<nixlLibfabricConnection> &conn);
    /** An engine EP of a peer without a (compatible) proxy, in @p rail's AV. */
    fi_addr_t
    railAddr(Thread &th,
             PeerAddrs &pa,
             const nixlLibfabricConnection &conn,
             size_t rail,
             size_t remote_ep);
    /**
     * The target's receiving data EP on its data rail number @p data_index (puts,
     * atomicAdd records, ring aborts), in @p rail's AV.
     */
    fi_addr_t
    dataAddr(Thread &th,
             PeerAddrs &pa,
             const nixlLibfabricConnection &conn,
             size_t rail,
             size_t data_index);
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
    void
    report() const;

    nixlLibfabricEngine &engine_;
    nixlProxyConfig config_{};
    uint32_t threads_ = 0;
    size_t rails_ = 0;
    uint64_t idle_poll_ns_ = 0; // efa_proxy_idle_poll_us (0: poll on every pass)
    bool profile_ = false; // NIXL_EFA_PROXY_PROFILE: per-stage timers, reported at shutdown
    std::unique_ptr<Inject> inject_; // NIXL_EFA_PROXY_INJECT: tests only, off under NDEBUG
    // efa_proxy_rx_flush: flush GPUDirect RDMA writes to the counter's GPU before
    // applying adds. Default as NCCL (ncclTopoNeedFlush()): only before Hopper, or
    // on aarch64 hosts (C2C platforms, where the counter write and the NIC's data
    // reach the GPU on different paths).
    bool rx_flush_ = false;
    int rx_flush_param_ = -1; // efa_proxy_rx_flush as given: -1 unset, 0 or 1
    uint64_t incarnation_ = 0; // published; peers restart their ring counts with it
    std::vector<uint32_t> rx_rails_; // rails that receive puts (next to the GPU), in order
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
    std::vector<int> vram_devices_; // GPUs with registered memory; under the region locks
    std::unique_ptr<CounterMap> counters_;
    // Serialize read-modify-writes of a VRAM counter: rings of different senders (or
    // sending threads) arrive on different threads and may add to the same counter.
    std::array<std::mutex, 64> counter_locks_;

    // Receive-side ring state by key, shared by the threads counting a ring's puts
    // and applying its adds; created on first use, kept until shutdown
    // (keys are never reused: a failed ring comes back under a new key).
    std::mutex rx_rings_mutex_;
    std::unordered_map<uint32_t, std::unique_ptr<RxRing>> rx_rings_;

    // Sender side: the next ring id per target (name, incarnation), shared by the
    // threads; ids are never reused for a target incarnation.
    std::mutex ring_ids_mutex_;
    std::map<std::pair<std::string, uint64_t>, uint32_t> next_ring_id_;
};

#endif // HAVE_NIXL_DEVICE_API
#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_H
