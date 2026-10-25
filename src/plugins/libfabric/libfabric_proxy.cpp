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
#ifdef NIXL_LIBFABRIC_DEVICE_PROXY

#include "libfabric_proxy.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <random>
#include <stdexcept>
#include <string>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef HAVE_GDRCOPY
#include <gdrapi.h>
#endif

#include "common/nixl_log.h"
#include "libfabric/libfabric_rail_manager.h"
#include "libfabric_backend.h"
#include "libfabric_connection.h"

namespace {
// A put below this size stays on its thread's rail; the engine stripes larger ones.
constexpr size_t kOwnRailMax = 256 << 10;
// Thread 0 reads the rails no thread polls once per this many passes.
constexpr uint32_t kSweepPasses = 64;

// Immediate data: the engine's 4-bit message type, then the stream id and the epoch slot.
constexpr uint32_t kTypeBits = 4;
constexpr uint32_t kEpochBits = 8;
constexpr uint32_t kEpochs = 1u << kEpochBits;
constexpr uint32_t kStreamIdMask = (1u << (32 - kTypeBits - kEpochBits)) - 1;

uint64_t
proxyImm(uint32_t payload) {
    return NIXL_LIBFABRIC_MSG_PROXY | (uint64_t(payload) << kTypeBits);
}

enum : uint32_t { kAddMsg = 1, kAckMsg = 2 };

struct AddMsg {
    uint32_t type;
    uint32_t stream;
    uint64_t seq; // index of this add on its stream
    uint64_t expected; // count of slot seq % kEpochs once this add's puts are in
    uint64_t addr;
    uint64_t value;
    uint64_t token; // the sender's request, echoed in the ack
    uint64_t reply_rail; // the sender's rail it left from
};

struct AckMsg {
    uint32_t type;
    int32_t status;
    uint64_t token;
};

// Target streams of one proxy instance, cached per receiving thread.
struct RxCache {
    uint64_t owner = 0;
    std::unordered_map<uint64_t, void *> streams;
};

thread_local RxCache tls_rx;
std::atomic<uint64_t> next_proxy_id{1};
} // namespace

struct nixlLibfabricProxy::Request {
    // Held in pending while a put is posted, so that a fragment completing meanwhile on
    // another thread cannot finish it early.
    static constexpr int64_t kPosting = int64_t(1) << 40;

    std::atomic<int64_t> pending{0}; // put fragments or the ack still to come
    std::atomic<int> status{NIXL_SUCCESS}; // the first error
    Request *next_free = nullptr;

    void
    fail(nixl_status_t st) {
        int ok = NIXL_SUCCESS;
        status.compare_exchange_strong(ok, st, std::memory_order_relaxed);
    }

    void
    complete(nixl_status_t st) {
        if (st != NIXL_SUCCESS) {
            fail(st);
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
    }
};

struct nixlLibfabricProxy::Thread {
    uint32_t index = 0;
    std::vector<size_t> poll; // rails whose CQs it reads
    std::vector<size_t> post; // indexes into rails_ for its small puts
    size_t next_post = 0;
    std::vector<size_t> one_rail;
    std::vector<size_t> sweep; // rails no thread polls (thread 0), read every kSweepPasses passes
    uint32_t passes = 0;
    std::vector<std::unique_ptr<Request>> requests;
    Request *free_list = nullptr;
};

/** One ring's traffic to one target. */
struct nixlLibfabricProxy::Stream {
    std::weak_ptr<nixlLibfabricConnection> conn; // expires when the target goes away
    uint32_t id = 0;
    uint64_t atomics = 0; // atomicAdds sent: the epoch of the puts that follow
    std::array<uint64_t, kEpochs> slot_writes{}; // put fragments sent per epoch slot
    std::shared_ptr<std::atomic<bool>> failed = std::make_shared<std::atomic<bool>>(false);
    size_t rail = 0; // where its atomicAdds go
    uint64_t dest = 0;
    bool routed = false;
};

struct nixlLibfabricProxy::Ring {
    std::unordered_map<const nixlLibfabricConnection *, Stream> streams; // by target
    std::deque<std::pair<Request *, nixl::proxyBackendSubmission>> parked; // until handshake
};

struct nixlLibfabricProxy::RxStream {
    struct Add {
        AddMsg msg;
        uint32_t agent;
        uint16_t rail; // the ack leaves from where the add arrived
    };

    std::array<std::atomic<uint64_t>, kEpochs> received{}; // put fragments per epoch slot
    std::atomic<uint32_t> waiting{0}; // adds received, not applied
    std::mutex mutex;
    uint64_t applied = 0; // the next seq to apply
    std::map<uint64_t, Add> adds; // by seq
    // Where its acks go (fixed per stream), looked up on the first ack.
    std::atomic<bool> ack_routed{false};
    std::atomic<uint64_t> ack_dest{0};
};

struct nixlLibfabricProxy::PendingAck {
    RxStream *rx;
    uint32_t agent;
    uint16_t rail; // the ack leaves from where the add arrived
    uint32_t reply_rail;
    AckMsg ack;
};

/** A registration; a GPU one is pinned and mapped with GDRCopy. */
struct nixlLibfabricProxy::Region {
    size_t len = 0;
    uint32_t refs = 1; // registrations of this range
    bool vram = false;
    void *bar = nullptr; // nullptr for VRAM: not mapped, adds there fail
    uintptr_t va = 0; // the GPU address bar maps
#ifdef HAVE_GDRCOPY
    gdr_mh_t mh{};
#endif
};

namespace {
std::atomic<uint32_t> next_stream_id{std::random_device{}()}; // fresh ids for a new target
std::array<std::mutex, 64> add_stripes; // GDRCopy read-modify-writes of one counter
} // namespace

nixlLibfabricProxy::nixlLibfabricProxy(nixlLibfabricEngine &engine,
                                       const nixl::proxyConfig &config)
    : engine_(engine),
      config_(config),
      id_(next_proxy_id.fetch_add(1)),
      rails_(gpuRails(engine.rail_manager_)),
      rings_(std::make_unique<Ring[]>(config.ringCount())) {
    if (rails_.empty()) {
        throw std::runtime_error("Device proxy: no rails");
    }
    // A slot's count is reused kEpochs atomicAdds later; a ring holds at most ring_depth
    // requests, so with ring_depth <= kEpochs the slot's previous add was applied by then.
    if (config.ring_depth > kEpochs) {
        throw std::runtime_error("Device proxy: proxy_ring_depth must be at most " +
                                 std::to_string(kEpochs));
    }
    // Each ring's thread, from the runtime's own split (proxyConfig::threadShare).
    ring_thread_.assign(config.ringCount(), 0);
    for (uint32_t t = 0; t < config.thread_count; ++t) {
        const nixl::proxyThreadShare share = config.threadShare(t);
        for (uint32_t c = share.first_channel; c < share.first_channel + share.channel_count; ++c) {
            for (uint32_t p = share.peers.first; p < config.max_peers; p += share.peers.stride) {
                ring_thread_[size_t(c) * config.max_peers + p] = t;
            }
        }
    }
    for (uint32_t t = 0; t < config.thread_count; ++t) {
        auto th = std::make_unique<Thread>();
        th->index = t;
        for (size_t i = t; i < rails_.size(); i += config.thread_count) {
            th->post.push_back(i);
            th->poll.push_back(rails_[i]);
        }
        if (th->post.empty()) {
            th->post.push_back(t % rails_.size());
        }
        threads_.push_back(std::move(th));
    }
    // The engine's control rail (handshakes) has no other reader without its progress thread.
    if (std::find(rails_.begin(), rails_.end(), 0) == rails_.end()) {
        threads_[0]->poll.push_back(0);
    }
    // Puts into host memory or another GPU's memory land on other rails: read them now and then.
    for (size_t r = 0; r < engine_.rail_manager_.getNumRails(); ++r) {
        if (r != 0 && std::find(rails_.begin(), rails_.end(), r) == rails_.end()) {
            threads_[0]->sweep.push_back(r);
        }
    }
#ifdef HAVE_CUDA
    // Make the NIC's writes into GPU memory visible before a counter written after them, where
    // the platform does not order them.
    int dev = 0;
    int ordering = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, dev) ==
            cudaSuccess) {
        flush_rdma_writes_ = ordering < cudaGPUDirectRDMAWritesOrderingOwner;
    }
    NIXL_INFO << "Device proxy: GPUDirect RDMA write ordering " << ordering
              << (flush_rdma_writes_ ? ", flushing before counter writes" : "");
#endif
    for (size_t r = 0; r < engine_.rail_manager_.getNumRails(); ++r) {
        engine_.rail_manager_.getRail(r).setProxyCallback(
            [this](uint64_t imm, const void *data, size_t len, uint32_t agent, uint16_t rail) {
                onTraffic(imm, data, len, agent, rail);
            });
    }
#ifdef HAVE_GDRCOPY
    gdr_ = gdr_open();
#endif
}

nixlLibfabricProxy::~nixlLibfabricProxy() {
    // The runtime destroys its transport after joining the workers.
    for (size_t r = 0; r < engine_.rail_manager_.getNumRails(); ++r) {
        engine_.rail_manager_.getRail(r).setProxyCallback(nullptr);
    }
    while (!regions_.empty()) {
        regions_.begin()->second.refs = 1;
        onDeregister(regions_.begin()->first, regions_.begin()->second.len);
    }
#ifdef HAVE_GDRCOPY
    if (gdr_ != nullptr) {
        gdr_close(static_cast<gdr_t>(gdr_));
    }
#endif
}

std::vector<size_t>
nixlLibfabricProxy::gpuRails(const nixlLibfabricRailManager &rails) {
    std::vector<size_t> out;
#ifdef HAVE_CUDA
    int dev = 0;
    char bus_id[32] = {};
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev) == cudaSuccess) {
        out = rails.selectRailsForMemory(nullptr, VRAM_SEG, dev, bus_id);
    }
#endif
    for (size_t i = 0; out.empty() && i < rails.getNumRails(); ++i) {
        out.push_back(i);
    }
    return out;
}

nixlLibfabricProxy::Thread &
nixlLibfabricProxy::threadOf(uint32_t channel, uint32_t peer) {
    return *threads_[ring_thread_[size_t(channel) * config_.max_peers + peer]];
}

nixlLibfabricProxy::Ring &
nixlLibfabricProxy::ring(uint32_t channel, uint32_t peer) {
    return rings_[size_t(channel) * config_.max_peers + peer];
}

nixl_status_t
nixlLibfabricProxy::submit(const nixl::proxyBackendSubmission &sub,
                           nixl::proxyBackendRequest &request) {
    if (sub.opcode != nixl_proxy_opcode_t::PUT && sub.opcode != nixl_proxy_opcode_t::ATOMIC_ADD) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const auto *remote = static_cast<const nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
    if (remote == nullptr || !remote->conn_) {
        return NIXL_ERR_INVALID_PARAM;
    }
    Thread &th = threadOf(sub.channel_id, sub.peer_index);
    Request *req = th.free_list;
    if (req != nullptr) {
        th.free_list = req->next_free;
    } else {
        th.requests.push_back(std::make_unique<Request>());
        req = th.requests.back().get();
    }
    req->status.store(NIXL_SUCCESS, std::memory_order_relaxed);

    // The target names a write's sender only after the engine handshake.
    Ring &r = ring(sub.channel_id, sub.peer_index);
    const nixlLibfabricConnection &conn = *remote->conn_;
    if (!r.parked.empty() ||
        (conn.remoteAgent_ != engine_.localAgent &&
         !conn.handshake_received_.load(std::memory_order_acquire))) {
        req->pending.store(1, std::memory_order_relaxed);
        r.parked.emplace_back(req, sub);
    } else {
        const nixl_status_t status = post(th, *req, sub);
        if (status != NIXL_IN_PROG) {
            req->next_free = th.free_list;
            th.free_list = req;
            return status;
        }
    }
    request.token = reinterpret_cast<uint64_t>(req);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricProxy::post(Thread &th, Request &req, const nixl::proxyBackendSubmission &sub) {
    const auto *remote = static_cast<const nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
    Stream &stream = ring(sub.channel_id, sub.peer_index).streams[remote->conn_.get()];
    if (stream.conn.expired()) { // a new target, or a new connection under a reused address
        stream = Stream{};
        stream.conn = remote->conn_;
        stream.id = next_stream_id.fetch_add(1) & kStreamIdMask;
    }
    return sub.opcode == nixl_proxy_opcode_t::PUT ? postPut(th, stream, req, sub) :
                                                    postAtomic(th, stream, req, sub);
}

nixl_status_t
nixlLibfabricProxy::postPut(Thread &th,
                            Stream &stream,
                            Request &req,
                            const nixl::proxyBackendSubmission &sub) {
    if (sub.size == 0) {
        return NIXL_SUCCESS; // nothing to write or count
    }
    auto *local = static_cast<nixlLibfabricPrivateMetadata *>(sub.local.metadataP);
    auto *remote = static_cast<nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
    if (local == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // The rail offset picks this thread's rail at both ends (same GPU-local rail order).
    const size_t rail_offset = th.post[th.next_post++ % th.post.size()];
    const std::vector<size_t> *rails = &local->selected_rails_;
    if (sub.size < kOwnRailMax && !rails->empty()) {
        th.one_rail.assign(1, (*rails)[rail_offset % rails->size()]);
        rails = &th.one_rail;
    }
    const uint32_t slot = static_cast<uint32_t>(stream.atomics % kEpochs);
    size_t submitted = 0;
    req.pending.store(Request::kPosting, std::memory_order_relaxed);
    const nixl_status_t status = engine_.rail_manager_.prepareAndSubmitTransfer(
        nixlLibfabricReq::WRITE,
        reinterpret_cast<void *>(sub.local.addr),
        sub.size,
        sub.remote.addr,
        remote->remote_buf_addr_,
        *rails,
        local->rail_mr_list_,
        remote->rail_remote_key_list_,
        remote->remote_selected_endpoints_,
        remote->conn_->rail_remote_addr_list_,
        0,
        [r = &req, failed = stream.failed](nixl_status_t st) {
            if (st != NIXL_SUCCESS) {
                failed->store(true, std::memory_order_release); // its epoch never completes
            }
            r->complete(st);
        },
        submitted,
        0,
        rail_offset,
        false,
        static_cast<int>(sub.local.devId),
        false,
        proxyImm(stream.id << kEpochBits | slot));
    stream.slot_writes[slot] += submitted;
    if (status != NIXL_SUCCESS) {
        req.fail(status);
        stream.failed->store(true, std::memory_order_release);
    }
    if (submitted == 0) {
        req.pending.store(0, std::memory_order_relaxed);
        return status == NIXL_SUCCESS ? NIXL_ERR_BACKEND : status;
    }
    req.pending.fetch_sub(Request::kPosting - static_cast<int64_t>(submitted),
                          std::memory_order_acq_rel);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricProxy::postAtomic(Thread &th,
                               Stream &stream,
                               Request &req,
                               const nixl::proxyBackendSubmission &sub) {
    if (stream.failed->load(std::memory_order_acquire)) {
        return NIXL_ERR_BACKEND; // a put of the stream failed: its count cannot be reached
    }
    const auto *remote = static_cast<const nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
    if (!stream.routed) { // to the target rail where this thread's small puts land
        const std::vector<size_t> &eps = remote->remote_selected_endpoints_;
        const auto addrs = remote->conn_->rail_remote_addr_list_.find(rails_[th.post[0]]);
        if (eps.empty() || addrs == remote->conn_->rail_remote_addr_list_.end() ||
            eps[th.post[0] % eps.size()] >= addrs->second.size()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        stream.rail = rails_[th.post[0]];
        stream.dest = addrs->second[eps[th.post[0] % eps.size()]];
        stream.routed = true;
    }
    const AddMsg msg{kAddMsg,
                     stream.id,
                     stream.atomics,
                     stream.slot_writes[stream.atomics % kEpochs],
                     sub.remote.addr,
                     sub.value,
                     reinterpret_cast<uint64_t>(&req),
                     stream.rail};
    req.pending.store(1, std::memory_order_relaxed); // the ack
    const nixl_status_t status = sendMsg(stream.rail, stream.dest, &msg, sizeof(msg));
    if (status != NIXL_SUCCESS) {
        return status;
    }
    ++stream.atomics;
    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricProxy::sendMsg(size_t rail_id, uint64_t dest, const void *msg, size_t len) {
    nixlLibfabricRail &rail = engine_.rail_manager_.getRail(rail_id);
    nixlLibfabricReq *req = rail.allocateControlRequest(len, 0);
    if (req == nullptr) {
        return NIXL_ERR_BACKEND;
    }
    std::memcpy(req->buffer, msg, len);
    req->buffer_size = len;
    req->completion_callback = nullptr;
    const nixl_status_t status = rail.postSend(proxyImm(0), dest, req);
    if (status != NIXL_SUCCESS) {
        rail.releaseRequest(req);
    }
    return status;
}

void
nixlLibfabricProxy::onTraffic(uint64_t imm,
                              const void *data,
                              size_t len,
                              uint32_t agent,
                              uint16_t rail) {
    if (data == nullptr) { // a put fragment landed
        const uint32_t payload = static_cast<uint32_t>(imm) >> kTypeBits;
        RxStream &rx = rxStream(agent, payload >> kEpochBits);
        rx.received[payload % kEpochs].fetch_add(1, std::memory_order_seq_cst);
        if (rx.waiting.load(std::memory_order_seq_cst) != 0) {
            drain(rx);
        }
        return;
    }
    uint32_t type = 0;
    std::memcpy(&type, data, std::min(len, sizeof(type)));
    if (type == kAckMsg && len == sizeof(AckMsg)) {
        AckMsg ack;
        std::memcpy(&ack, data, sizeof(ack));
        reinterpret_cast<Request *>(ack.token)->complete(static_cast<nixl_status_t>(ack.status));
    } else if (type == kAddMsg && len == sizeof(AddMsg)) {
        AddMsg msg;
        std::memcpy(&msg, data, sizeof(msg));
        RxStream &rx = rxStream(agent, msg.stream);
        {
            std::lock_guard<std::mutex> lock(rx.mutex);
            rx.adds.emplace(msg.seq, RxStream::Add{msg, agent, rail});
        }
        rx.waiting.fetch_add(1, std::memory_order_seq_cst);
        drain(rx);
    } else {
        NIXL_ERROR << "Device proxy: unknown message (type " << type << ", " << len << " bytes)";
    }
}

nixlLibfabricProxy::RxStream &
nixlLibfabricProxy::rxStream(uint32_t agent, uint32_t stream_id) {
    const uint64_t key = uint64_t(agent) << 32 | stream_id;
    if (tls_rx.owner != id_) {
        tls_rx.owner = id_;
        tls_rx.streams.clear();
    }
    void *&cached = tls_rx.streams[key];
    if (cached == nullptr) {
        std::lock_guard<std::mutex> lock(rx_mutex_);
        auto &rx = rx_streams_[key];
        if (!rx) {
            rx = std::make_unique<RxStream>();
        }
        cached = rx.get();
    }
    return *static_cast<RxStream *>(cached);
}

void
nixlLibfabricProxy::drain(RxStream &rx) {
    // Applied in seq order under the stream's lock; acked after it.
    std::vector<std::pair<RxStream::Add, nixl_status_t>> done;
    {
        std::lock_guard<std::mutex> lock(rx.mutex);
        for (auto it = rx.adds.begin(); it != rx.adds.end() && it->first == rx.applied;
             it = rx.adds.begin()) {
            const AddMsg &msg = it->second.msg;
            if (rx.received[msg.seq % kEpochs].load(std::memory_order_seq_cst) < msg.expected) {
                break; // its puts are not all in yet
            }
            done.emplace_back(it->second, apply(msg.addr, msg.value));
            rx.adds.erase(it);
            ++rx.applied;
            rx.waiting.fetch_sub(1, std::memory_order_seq_cst);
        }
    }
    for (const auto &[add, status] : done) {
        const PendingAck ack{&rx,
                             add.agent,
                             add.rail,
                             static_cast<uint32_t>(add.msg.reply_rail),
                             AckMsg{kAckMsg, static_cast<int32_t>(status), add.msg.token}};
        if (!sendAck(ack)) {
            NIXL_WARN << "Device proxy: cannot ack an atomicAdd of agent " << add.agent
                      << " now; retrying";
            std::lock_guard<std::mutex> lock(acks_mutex_);
            acks_.push_back(ack);
            acks_pending_.store(true, std::memory_order_release);
        }
    }
}

bool
nixlLibfabricProxy::sendAck(const PendingAck &ack) {
    RxStream &rx = *ack.rx;
    if (!rx.ack_routed.load(std::memory_order_acquire)) {
        std::shared_ptr<nixlLibfabricConnection> conn;
        {
            std::lock_guard<std::mutex> lock(engine_.connection_state_mutex_);
            if (ack.agent < engine_.agent_names_.size()) {
                auto it = engine_.connections_.find(engine_.agent_names_[ack.agent]);
                if (it != engine_.connections_.end()) {
                    conn = it->second;
                }
            }
        }
        if (!conn) {
            return false;
        }
        auto it = conn->rail_remote_addr_list_.find(ack.rail);
        if (it == conn->rail_remote_addr_list_.end() || ack.reply_rail >= it->second.size()) {
            return false;
        }
        rx.ack_dest.store(it->second[ack.reply_rail], std::memory_order_relaxed);
        rx.ack_routed.store(true, std::memory_order_release);
    }
    return sendMsg(ack.rail, rx.ack_dest.load(std::memory_order_relaxed), &ack.ack,
                   sizeof(ack.ack)) == NIXL_SUCCESS;
}

void
nixlLibfabricProxy::retryAcks() {
    std::vector<PendingAck> acks;
    {
        std::lock_guard<std::mutex> lock(acks_mutex_);
        acks.swap(acks_);
        acks_pending_.store(false, std::memory_order_relaxed);
    }
    std::vector<PendingAck> failed;
    for (const PendingAck &ack : acks) {
        if (!sendAck(ack)) {
            failed.push_back(ack);
        }
    }
    if (!failed.empty()) {
        std::lock_guard<std::mutex> lock(acks_mutex_);
        acks_.insert(acks_.end(), failed.begin(), failed.end());
        acks_pending_.store(true, std::memory_order_release);
    }
}

nixl_status_t
nixlLibfabricProxy::apply(uint64_t addr, uint64_t value) {
    std::shared_lock<std::shared_mutex> lock(regions_mutex_);
    auto it = regions_.upper_bound(addr);
    if (it == regions_.begin() || addr % sizeof(uint64_t) != 0 ||
        addr + sizeof(uint64_t) > std::prev(it)->first + std::prev(it)->second.len) {
        NIXL_ERROR << "Device proxy: atomicAdd to " << reinterpret_cast<void *>(addr)
                   << " failed: outside a registration";
        return NIXL_ERR_NOT_FOUND;
    }
    const Region &r = std::prev(it)->second;
    if (!r.vram) {
        __atomic_fetch_add(reinterpret_cast<uint64_t *>(addr), value, __ATOMIC_SEQ_CST);
        return NIXL_SUCCESS;
    }
    if (r.bar == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
#ifdef HAVE_GDRCOPY
    // A read-modify-write: the GPU must not write a counter that receives remote adds.
    std::lock_guard<std::mutex> stripe(add_stripes[(addr >> 3) % add_stripes.size()]);
#ifdef HAVE_CUDA
    if (flush_rdma_writes_) {
        static_cast<void>(cudaDeviceFlushGPUDirectRDMAWrites(
            cudaFlushGPUDirectRDMAWritesTargetCurrentDevice, cudaFlushGPUDirectRDMAWritesToOwner));
    }
#endif
    // One aligned 8-byte load and store, so a polling kernel never sees a torn counter.
    auto *bar = reinterpret_cast<volatile uint64_t *>(static_cast<char *>(r.bar) + (addr - r.va));
    *bar = *bar + value;
    std::atomic_thread_fence(std::memory_order_seq_cst); // flush the write-combined store
#endif
    return NIXL_SUCCESS;
}

void
nixlLibfabricProxy::onRegister(uintptr_t addr, size_t len, bool is_vram) {
    {
        std::unique_lock<std::shared_mutex> lock(regions_mutex_);
        auto it = regions_.find(addr);
        if (it != regions_.end() && it->second.len == len) {
            ++it->second.refs; // the same range again: keep its mapping
            return;
        }
    }
    Region r;
    r.len = len;
    r.vram = is_vram;
#ifdef HAVE_GDRCOPY
    // GPU pages are 64 KiB: map the pages the registration touches.
    const uintptr_t start = addr & ~uintptr_t(0xffff);
    const size_t map_len = ((addr + len + 0xffff) & ~uintptr_t(0xffff)) - start;
    auto gdr = static_cast<gdr_t>(gdr_);
    if (is_vram && gdr != nullptr && gdr_pin_buffer(gdr, start, map_len, 0, 0, &r.mh) == 0) {
        if (gdr_map(gdr, r.mh, &r.bar, map_len) == 0) {
            gdr_info_t info{};
            gdr_get_info(gdr, r.mh, &info);
            r.va = info.va;
        } else {
            r.bar = nullptr;
            gdr_unpin_buffer(gdr, r.mh);
        }
    }
#endif
    if (is_vram && r.bar == nullptr) {
        NIXL_WARN << "Device proxy: cannot map " << reinterpret_cast<void *>(addr)
                  << " with GDRCopy; atomicAdds there fail";
    }
    std::unique_lock<std::shared_mutex> lock(regions_mutex_);
    if (!regions_.emplace(addr, r).second) {
        NIXL_WARN << "Device proxy: " << reinterpret_cast<void *>(addr)
                  << " is already registered with another length; atomicAdds use the first";
#ifdef HAVE_GDRCOPY
        if (r.bar != nullptr) {
            const uintptr_t start = addr & ~uintptr_t(0xffff);
            const size_t map_len = ((addr + len + 0xffff) & ~uintptr_t(0xffff)) - start;
            gdr_unmap(static_cast<gdr_t>(gdr_), r.mh, r.bar, map_len);
            gdr_unpin_buffer(static_cast<gdr_t>(gdr_), r.mh);
        }
#endif
    }
}

void
nixlLibfabricProxy::onDeregister(uintptr_t addr, size_t len) {
    // Waits for adds being applied; later ones fail.
    std::unique_lock<std::shared_mutex> lock(regions_mutex_);
    auto it = regions_.find(addr);
    if (it == regions_.end() || it->second.len != len) {
        return;
    }
    if (--it->second.refs > 0) {
        return;
    }
#ifdef HAVE_GDRCOPY
    if (it->second.bar != nullptr) {
        const uintptr_t start = addr & ~uintptr_t(0xffff);
        const size_t map_len = ((addr + len + 0xffff) & ~uintptr_t(0xffff)) - start;
        gdr_unmap(static_cast<gdr_t>(gdr_), it->second.mh, it->second.bar, map_len);
        gdr_unpin_buffer(static_cast<gdr_t>(gdr_), it->second.mh);
    }
#endif
    regions_.erase(it);
}

nixl_status_t
nixlLibfabricProxy::checkCompletion(uint32_t channel,
                                    uint32_t peer,
                                    const nixl::proxyBackendRequest &request) {
    Request *req = reinterpret_cast<Request *>(request.token);
    if (req->pending.load(std::memory_order_acquire) != 0) {
        return NIXL_IN_PROG;
    }
    Thread &th = threadOf(channel, peer);
    req->next_free = th.free_list;
    th.free_list = req;
    return static_cast<nixl_status_t>(req->status.load(std::memory_order_relaxed));
}

void
nixlLibfabricProxy::progress(uint32_t channel, uint32_t peer) noexcept {
    Thread &th = threadOf(channel, peer);
    const nixl::proxyThreadShare share = config_.threadShare(th.index);
    if (channel == share.first_channel && peer == share.peers.first) { // once per worker pass
        for (size_t rail : th.poll) {
            static_cast<void>(engine_.rail_manager_.getRail(rail).progressCompletionQueue());
        }
        if (!th.sweep.empty() && ++th.passes % kSweepPasses == 0) {
            for (size_t rail : th.sweep) {
                static_cast<void>(engine_.rail_manager_.getRail(rail).progressCompletionQueue());
            }
        }
        if (acks_pending_.load(std::memory_order_acquire)) {
            retryAcks();
        }
    }
    Ring &r = ring(channel, peer);
    while (!r.parked.empty()) {
        auto [req, sub] = r.parked.front();
        const auto *remote = static_cast<const nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
        const nixlLibfabricConnection &conn = *remote->conn_;
        if (conn.remoteAgent_ != engine_.localAgent &&
            !conn.handshake_received_.load(std::memory_order_acquire)) {
            return;
        }
        r.parked.pop_front();
        const nixl_status_t status = post(th, *req, sub);
        if (status != NIXL_IN_PROG) {
            req->fail(status);
            req->pending.store(0, std::memory_order_release);
        }
    }
}

nixl_status_t
nixlLibfabricProxy::quiesce(uint32_t channel, uint32_t peer) {
    // Puts complete on remote completion and atomicAdds once applied: nothing to flush.
    return ring(channel, peer).parked.empty() ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
}

#endif // NIXL_LIBFABRIC_DEVICE_PROXY
