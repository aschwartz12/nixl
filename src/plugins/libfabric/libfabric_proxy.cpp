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
#include <limits>
#include <random>
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
/**
 * A put below this size stays on one rail of its thread; the engine would stripe it from
 * its striping threshold (128 KiB) over every rail, and so over every thread's rails.
 */
constexpr size_t kOwnRailMax = 256 << 10;

// Immediate data: the engine's message type, then 28 bits of ours. A put's: its stream's ring
// id and its epoch slot.
constexpr uint32_t kTypeBits = 4;
constexpr uint32_t kEpochBits = 8;
constexpr uint32_t kEpochs = 1u << kEpochBits;
constexpr uint32_t kRingIdBits = 32 - kTypeBits - kEpochBits;
constexpr uint32_t kRingIdMask = (1u << kRingIdBits) - 1;

uint64_t
proxyImm(uint32_t payload) {
    return NIXL_LIBFABRIC_MSG_PROXY | (uint64_t(payload) << kTypeBits);
}

// Control messages. No version: every node of a job runs the same build.
enum : uint32_t { kAddMsg = 1, kAckMsg = 2, kAbortMsg = 3 };

constexpr uint64_t kNone = std::numeric_limits<uint64_t>::max();

/** Sender -> target rail of the stream's small puts. */
struct AddMsg {
    uint32_t type;
    uint32_t ring; // the stream's ring id
    uint64_t seq; // index of this add among the stream's atomicAdds
    uint64_t expected; // count of slot seq % kEpochs once this add's puts are in
    uint64_t addr;
    uint64_t value;
    uint64_t token; // the sender's request, echoed in the ack
    uint32_t reply_rail; // the sender's rail this left from
    uint32_t reserved;
};

/** Target -> the sender's reply rail, once the add was applied (or failed). */
struct AckMsg {
    uint32_t type;
    int32_t status;
    uint64_t token;
};

/** Sender -> target: a put of the stream failed, so its adds from first_seq on fail. */
struct AbortMsg {
    uint32_t type;
    uint32_t ring;
    uint64_t first_seq;
};

/** Streams by (sender agent, ring id) of one proxy, cached per thread. */
struct RxCache {
    uint64_t owner = 0;
    std::unordered_map<uint64_t, void *> rings;
};

thread_local RxCache tls_rx;
std::atomic<uint64_t> next_proxy_id{1};
} // namespace

struct nixlLibfabricProxy::Request {
    /** Held in pending while a put is being posted, so that a fragment completed meanwhile
     *  (by a thread reading the CQ) cannot finish the request early. */
    static constexpr int64_t kPosting = int64_t(1) << 40;

    std::atomic<int64_t> pending{0}; // fragments (put) or the ack (atomicAdd) to come; 0: done
    std::atomic<int> status{NIXL_SUCCESS}; // first error
    Stream *stream = nullptr; // a put's stream, and the epoch it counts for
    uint64_t epoch = 0;
    Request *next_free = nullptr;

    void
    fail(nixl_status_t st) {
        int expected = NIXL_SUCCESS;
        status.compare_exchange_strong(expected, st, std::memory_order_relaxed);
    }

    /** One fragment completed or the ack arrived; may run on any thread reading a CQ. */
    void
    complete(nixl_status_t st) {
        if (st != NIXL_SUCCESS) {
            fail(st);
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
    }

    [[nodiscard]] bool
    done() const {
        return pending.load(std::memory_order_acquire) == 0;
    }
};

struct nixlLibfabricProxy::Thread {
    uint32_t index = 0;
    std::vector<size_t> poll; // rails whose CQs this thread reads
    std::vector<size_t> post; // indexes into rails_ its small puts use
    size_t next_post = 0;
    std::vector<size_t> one_rail; // the rail of a put below kOwnRailMax
    std::vector<std::unique_ptr<Request>> requests; // every request it allocated
    Request *free_list = nullptr;
};

/** One ring's traffic to one target; used by the ring's proxy thread only. */
struct nixlLibfabricProxy::Stream {
    std::weak_ptr<nixlLibfabricConnection> conn; // expires when the target goes away
    uint32_t ring_id = 0;
    uint64_t atomics = 0; // atomicAdds sent: the epoch of the puts that follow
    std::array<uint64_t, kEpochs> slot_writes{}; // write fragments sent per epoch slot
    nixl_status_t error = NIXL_SUCCESS; // a failed post: the target cannot count the epoch
    // Where its atomicAdds (and an abort) go, once the first one went.
    size_t rail = 0;
    uint64_t dest = 0;
    std::atomic<bool> routed{false};
    // The first epoch with a failed put fragment (set by the thread reading its completion).
    std::atomic<uint64_t> failed_from{kNone};
    std::atomic<bool> abort_queued{false};
};

/** Sender side of one (channel, peer) ring; used by its proxy thread only. */
struct nixlLibfabricProxy::Ring {
    // By target: a ring's operations may go to several agents over time.
    std::unordered_map<const nixlLibfabricConnection *, std::unique_ptr<Stream>> streams;
    std::deque<std::pair<Request *, nixl::proxyBackendSubmission>> parked; // target not ready
};

/** Target side of one sender stream, shared by the threads receiving its traffic. */
struct nixlLibfabricProxy::RxRing {
    struct Add {
        AddMsg msg;
        uint32_t agent;
        uint16_t rail; // where it arrived: the ack leaves from there
    };

    std::array<std::atomic<uint64_t>, kEpochs> received{}; // put fragments per epoch slot
    std::atomic<uint32_t> waiting{0}; // adds received, not applied
    std::mutex mutex; // below
    uint64_t applied = 0; // the next seq to apply
    uint64_t aborted_from = kNone; // the sender failed its adds from this seq on
    std::map<uint64_t, Add> adds; // by seq
};

/** Applies adds: GDRCopy read-modify-writes for GPU memory, CPU atomics for host memory. */
struct nixlLibfabricProxy::Counters {
    static constexpr uintptr_t kPage = 64 << 10; // GPU page

    struct Page {
        bool usable = false;
        bool device = false;
#ifdef HAVE_GDRCOPY
        gdr_mh_t mh{};
        void *bar = nullptr;
        uintptr_t va = 0; // the GPU address bar maps
#endif
    };

    /** A registration's whole GPU pages, pinned and mapped at registration. */
    struct Region {
        uintptr_t start; // first and end GPU page inside the registration
        uintptr_t end;
        Page page;
    };

    // BAR1 a process maps at registration; beyond it, pages are mapped on a first add.
    static constexpr size_t kPremapLimit = 4ull << 30;

    std::mutex mutex; // below, and GDRCopy calls other than copies
    std::unordered_map<uintptr_t, Page> pages; // mapped on a first add
    std::map<uintptr_t, Region> regions; // by start: mapped at registration
    size_t region_bytes = 0;
    // Adds to one GPU counter may come from several streams on several threads.
    std::array<std::mutex, 64> stripes;
#ifdef HAVE_GDRCOPY
    gdr_t gdr = gdr_open();
#endif

    ~Counters() {
#ifdef HAVE_GDRCOPY
        for (auto &[page, p] : pages) {
            if (p.device && p.usable) {
                gdr_unmap(gdr, p.mh, p.bar, kPage);
                gdr_unpin_buffer(gdr, p.mh);
            }
        }
        for (auto &[start, r] : regions) {
            gdr_unmap(gdr, r.page.mh, r.page.bar, r.end - r.start);
            gdr_unpin_buffer(gdr, r.page.mh);
        }
        if (gdr) {
            gdr_close(gdr);
        }
#endif
    }

    Page
    map(uintptr_t page) {
        Page p;
#ifdef HAVE_CUDA
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, reinterpret_cast<void *>(page)) != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            attr.type = cudaMemoryTypeUnregistered;
        }
        p.device = attr.type == cudaMemoryTypeDevice;
#endif
        if (!p.device) {
            p.usable = true;
            return p;
        }
#ifdef HAVE_GDRCOPY
        if (gdr != nullptr && gdr_pin_buffer(gdr, page, kPage, 0, 0, &p.mh) == 0) {
            if (gdr_map(gdr, p.mh, &p.bar, kPage) == 0) {
                gdr_info_t info{};
                gdr_get_info(gdr, p.mh, &info);
                p.va = info.va;
                p.usable = true;
            } else {
                gdr_unpin_buffer(gdr, p.mh);
            }
        }
#endif
        if (!p.usable) {
            NIXL_ERROR << "Device proxy: cannot map GPU page " << reinterpret_cast<void *>(page)
                       << " with GDRCopy; atomicAdds there fail";
        }
        return p;
    }

    /**
     * Pin and map a GPU registration's whole pages now, so that its first add does not pin
     * on the proxy thread while kernels run. Best effort: otherwise pages map on a first add.
     */
    void
    premap(uintptr_t base, size_t len) {
#ifdef HAVE_GDRCOPY
        const uintptr_t start = (base + kPage - 1) & ~(kPage - 1);
        const uintptr_t end = (base + len) & ~(kPage - 1);
        std::lock_guard<std::mutex> lock(mutex);
        if (gdr == nullptr || end <= start || region_bytes + (end - start) > kPremapLimit ||
            regions.count(start) != 0) {
            return;
        }
        Region r{start, end, {}};
        if (gdr_pin_buffer(gdr, start, end - start, 0, 0, &r.page.mh) != 0) {
            return;
        }
        if (gdr_map(gdr, r.page.mh, &r.page.bar, end - start) != 0) {
            gdr_unpin_buffer(gdr, r.page.mh);
            return;
        }
        gdr_info_t info{};
        gdr_get_info(gdr, r.page.mh, &info);
        r.page.va = info.va;
        r.page.device = r.page.usable = true;
        regions.emplace(start, r);
        region_bytes += end - start;
#else
        static_cast<void>(base);
        static_cast<void>(len);
#endif
    }

    /** Forget the pages of [lo, hi) that @p covered no longer covers. */
    template<typename Covered>
    void
    unmap(uintptr_t lo, uintptr_t hi, Covered covered) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = pages.begin(); it != pages.end();) {
            const uintptr_t page = it->first;
            if (page + kPage <= lo || page >= hi || covered(page, page + kPage)) {
                ++it;
                continue;
            }
#ifdef HAVE_GDRCOPY
            if (it->second.device && it->second.usable) {
                gdr_unmap(gdr, it->second.mh, it->second.bar, kPage);
                gdr_unpin_buffer(gdr, it->second.mh);
            }
#endif
            it = pages.erase(it);
        }
        for (auto it = regions.begin(); it != regions.end();) {
            const Region &r = it->second;
            if (r.end <= lo || r.start >= hi || covered(r.start, r.end)) {
                ++it;
                continue;
            }
#ifdef HAVE_GDRCOPY
            gdr_unmap(gdr, r.page.mh, r.page.bar, r.end - r.start);
            gdr_unpin_buffer(gdr, r.page.mh);
#endif
            region_bytes -= r.end - r.start;
            it = regions.erase(it);
        }
    }

    nixl_status_t
    add(uint64_t addr, uint64_t value) {
        if (addr % sizeof(uint64_t) != 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const uintptr_t page = addr & ~(kPage - 1);
        Page p;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto r = regions.upper_bound(addr);
            if (r != regions.begin() && addr + sizeof(uint64_t) <= std::prev(r)->second.end) {
                p = std::prev(r)->second.page;
            } else {
                auto it = pages.find(page);
                if (it == pages.end()) {
                    it = pages.emplace(page, map(page)).first;
                }
                p = it->second;
            }
        }
        if (!p.usable) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
        if (!p.device) {
            __atomic_fetch_add(reinterpret_cast<uint64_t *>(addr), value, __ATOMIC_SEQ_CST);
            return NIXL_SUCCESS;
        }
#ifdef HAVE_GDRCOPY
        // A read-modify-write, not an atomic: the GPU must not write a counter that receives
        // remote adds.
        std::lock_guard<std::mutex> lock(stripes[(addr >> 3) % stripes.size()]);
        char *bar = static_cast<char *>(p.bar) + (addr - p.va);
        uint64_t v = 0;
        gdr_copy_from_mapping(p.mh, &v, bar, sizeof(v));
        v += value;
        gdr_copy_to_mapping(p.mh, bar, &v, sizeof(v));
        return NIXL_SUCCESS;
#else
        return NIXL_ERR_NOT_SUPPORTED;
#endif
    }
};

nixlLibfabricProxy::nixlLibfabricProxy(nixlLibfabricEngine &engine)
    : engine_(engine),
      id_(next_proxy_id.fetch_add(1)),
      counters_(std::make_unique<Counters>()) {}

nixlLibfabricProxy::~nixlLibfabricProxy() = default;

std::vector<size_t>
nixlLibfabricProxy::gpuRails(const nixlLibfabricRailManager &rails) {
    std::vector<size_t> out;
#ifdef HAVE_CUDA
    int dev = 0;
    char bus_id[32] = {};
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev) == cudaSuccess) {
        out = rails.railsForAccelerator(bus_id);
    }
#endif
    if (out.empty()) {
        for (size_t i = 0; i < rails.getNumRails(); ++i) {
            out.push_back(i);
        }
    }
    return out;
}

nixl_status_t
nixlLibfabricProxy::init(const nixl::proxyConfig &config) {
    rails_ = gpuRails(engine_.rail_manager_);
    if (rails_.empty()) {
        NIXL_ERROR << "Device proxy: no rails";
        return NIXL_ERR_BACKEND;
    }
    max_peers_ = config.max_peers;
    rings_ = std::make_unique<Ring[]>(config.ringCount());
    ring_id_base_ = std::random_device{}() & kRingIdMask;

    // The runtime runs channel c on worker c % threads.
    const uint32_t nthreads = config.effectiveThreadCount();
    for (uint32_t t = 0; t < nthreads; ++t) {
        auto th = std::make_unique<Thread>();
        th->index = t;
        for (size_t i = t; i < rails_.size(); i += nthreads) {
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
    for (size_t r = 0; r < engine_.rail_manager_.getNumRails(); ++r) {
        engine_.rail_manager_.getRail(r).setProxyCallback(
            [this](uint64_t imm, const void *data, size_t len, uint32_t agent, uint16_t rail) {
                onTraffic(imm, data, len, agent, rail);
            });
    }

    std::string rails;
    for (size_t r : rails_) {
        rails += (rails.empty() ? "" : ",") + std::to_string(r);
    }
    NIXL_INFO << "Device proxy on engine rails " << rails << ": " << nthreads << " thread(s)";
    return NIXL_SUCCESS;
}

bool
nixlLibfabricProxy::ready(const nixlLibfabricConnection &conn) const {
    return conn.remoteAgent_ == engine_.localAgent ||
        conn.handshake_received_.load(std::memory_order_acquire);
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
    Thread &th = threadOf(sub.channel_id);
    Ring &ring = rings_[size_t(sub.channel_id) * max_peers_ + sub.peer_index];
    Request *req = allocRequest(th);

    if (!ring.parked.empty() || !ready(*remote->conn_)) {
        req->pending.store(1, std::memory_order_relaxed);
        ring.parked.emplace_back(req, sub);
    } else {
        const nixl_status_t status = post(th, ring, *req, sub);
        if (status != NIXL_IN_PROG) {
            freeRequest(th, req);
            return status;
        }
    }
    request.token = reinterpret_cast<uint64_t>(req);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricProxy::post(Thread &th,
                         Ring &ring,
                         Request &req,
                         const nixl::proxyBackendSubmission &sub) {
    const auto *remote = static_cast<const nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
    std::unique_ptr<Stream> &slot = ring.streams[remote->conn_.get()];
    if (!slot) {
        slot = std::make_unique<Stream>();
    }
    Stream &stream = *slot;
    if (stream.conn.expired()) {
        // A new target, or a restarted one under a reused address: a fresh ring id, so that
        // its counts start over at the target.
        stream.conn = remote->conn_;
        stream.atomics = 0;
        stream.slot_writes.fill(0);
        stream.error = NIXL_SUCCESS;
        stream.routed.store(false, std::memory_order_relaxed);
        stream.failed_from.store(kNone, std::memory_order_relaxed);
        stream.abort_queued.store(false, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(ring_ids_mutex_);
        auto &[target, next] = ring_ids_[remote->conn_.get()];
        if (target.expired()) {
            target = remote->conn_;
            next = 0;
        }
        stream.ring_id = (ring_id_base_ + next++) & kRingIdMask;
    }
    return sub.opcode == nixl_proxy_opcode_t::PUT ? postPut(th, stream, req, sub) :
                                                    postAtomic(th, stream, req, sub);
}

nixl_status_t
nixlLibfabricProxy::postPut(Thread &th,
                            Stream &stream,
                            Request &req,
                            const nixl::proxyBackendSubmission &sub) {
    auto *local = static_cast<nixlLibfabricPrivateMetadata *>(sub.local.metadataP);
    auto *remote = static_cast<nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
    if (local == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // The rail offset selects this thread's rail for a put below the striping threshold, at
    // both ends (local and remote rails are taken in the same GPU-local order).
    const size_t rail_offset = th.post[th.next_post++ % th.post.size()];
    const std::vector<size_t> *rails = &local->selected_rails_;
    if (sub.size < kOwnRailMax && !rails->empty()) {
        th.one_rail.assign(1, (*rails)[rail_offset % rails->size()]);
        rails = &th.one_rail;
    }
    const uint32_t slot = static_cast<uint32_t>(stream.atomics % kEpochs);
    size_t submitted = 0;
    req.stream = &stream;
    req.epoch = stream.atomics;
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
        [this, r = &req](nixl_status_t st) {
            if (st != NIXL_SUCCESS) {
                onPutFailed(*r);
            }
            r->complete(st);
        },
        submitted,
        0,
        rail_offset,
        false,
        static_cast<int>(sub.local.devId),
        false,
        proxyImm(stream.ring_id << kEpochBits | slot));
    stream.slot_writes[slot] += submitted;
    if (status != NIXL_SUCCESS) {
        req.fail(status);
        stream.error = status; // the epoch's count cannot be reached: fail its atomicAdd
    }
    if (submitted == 0) {
        req.pending.store(0, std::memory_order_relaxed);
        return status == NIXL_SUCCESS ? NIXL_ERR_BACKEND : status;
    }
    // A failure after some fragments were posted completes once those complete.
    req.pending.fetch_sub(Request::kPosting - static_cast<int64_t>(submitted),
                          std::memory_order_acq_rel);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricProxy::postAtomic(Thread &th,
                               Stream &stream,
                               Request &req,
                               const nixl::proxyBackendSubmission &sub) {
    if (stream.error != NIXL_SUCCESS) {
        return stream.error;
    }
    if (stream.failed_from.load(std::memory_order_acquire) != kNone) {
        return NIXL_ERR_BACKEND; // a put of the stream failed: its target aborts it
    }
    const auto *remote = static_cast<const nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
    if (!stream.routed.load(std::memory_order_relaxed)) {
        // To the target rail where this thread's small puts land.
        const std::vector<size_t> &eps = remote->remote_selected_endpoints_;
        const size_t rail = rails_[th.post[0]];
        const auto addrs = remote->conn_->rail_remote_addr_list_.find(rail);
        if (eps.empty() || addrs == remote->conn_->rail_remote_addr_list_.end() ||
            eps[th.post[0] % eps.size()] >= addrs->second.size()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        stream.rail = rail;
        stream.dest = addrs->second[eps[th.post[0] % eps.size()]];
        stream.routed.store(true, std::memory_order_release);
    }

    AddMsg msg{};
    msg.type = kAddMsg;
    msg.ring = stream.ring_id;
    msg.seq = stream.atomics;
    msg.expected = stream.slot_writes[stream.atomics % kEpochs];
    msg.addr = sub.remote.addr;
    msg.value = sub.value;
    msg.token = reinterpret_cast<uint64_t>(&req);
    msg.reply_rail = static_cast<uint32_t>(stream.rail);
    req.pending.store(1, std::memory_order_relaxed); // the ack
    const nixl_status_t status = sendMsg(stream.rail, stream.dest, &msg, sizeof(msg));
    if (status != NIXL_SUCCESS) {
        req.pending.store(0, std::memory_order_relaxed);
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
nixlLibfabricProxy::onPutFailed(Request &req) {
    Stream *stream = req.stream;
    if (stream == nullptr) {
        return;
    }
    uint64_t from = stream->failed_from.load(std::memory_order_acquire);
    while (req.epoch < from &&
           !stream->failed_from.compare_exchange_weak(from, req.epoch, std::memory_order_acq_rel)) {
    }
    if (!stream->abort_queued.exchange(true)) {
        std::lock_guard<std::mutex> lock(aborts_mutex_);
        aborts_.push_back(stream);
        aborts_pending_.store(true, std::memory_order_release);
    }
}

void
nixlLibfabricProxy::sendAborts() {
    std::vector<Stream *> streams;
    {
        std::lock_guard<std::mutex> lock(aborts_mutex_);
        streams.swap(aborts_);
        aborts_pending_.store(false, std::memory_order_relaxed);
    }
    for (Stream *stream : streams) {
        if (!stream->routed.load(std::memory_order_acquire)) {
            continue; // no atomicAdd went to the target: none waits there
        }
        const AbortMsg msg{kAbortMsg, stream->ring_id, stream->failed_from.load()};
        NIXL_ERROR << "Device proxy: a put failed; aborting stream " << stream->ring_id
                   << " from atomicAdd " << msg.first_seq;
        if (sendMsg(stream->rail, stream->dest, &msg, sizeof(msg)) != NIXL_SUCCESS) {
            NIXL_ERROR << "Device proxy: failed to send a stream abort";
        }
    }
}

void
nixlLibfabricProxy::flushParked(Thread &th, Ring &ring) {
    while (!ring.parked.empty()) {
        Request *req = ring.parked.front().first;
        const nixl::proxyBackendSubmission sub = ring.parked.front().second;
        const auto *remote = static_cast<const nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
        if (!ready(*remote->conn_)) {
            return;
        }
        ring.parked.pop_front();
        const nixl_status_t status = post(th, ring, *req, sub);
        if (status != NIXL_IN_PROG) {
            req->fail(status);
            req->pending.store(0, std::memory_order_release);
        }
    }
}

void
nixlLibfabricProxy::pollRails(Thread &th) {
    // Errors are logged by the rail and reach the request through its callback.
    for (size_t rail : th.poll) {
        static_cast<void>(engine_.rail_manager_.getRail(rail).progressCompletionQueue());
    }
}

void
nixlLibfabricProxy::onTraffic(uint64_t imm,
                              const void *data,
                              size_t len,
                              uint32_t agent,
                              uint16_t rail) {
    if (data == nullptr) { // a put fragment landed
        const uint32_t payload = static_cast<uint32_t>(imm) >> kTypeBits;
        RxRing &rx = rxRing(agent, payload >> kEpochBits);
        rx.received[payload % kEpochs].fetch_add(1, std::memory_order_seq_cst);
        if (rx.waiting.load(std::memory_order_seq_cst) != 0) {
            drain(rx);
        }
        return;
    }
    uint32_t type = 0;
    if (len >= sizeof(type)) {
        std::memcpy(&type, data, sizeof(type));
    }
    if (type == kAckMsg && len == sizeof(AckMsg)) {
        AckMsg ack;
        std::memcpy(&ack, data, sizeof(ack));
        reinterpret_cast<Request *>(ack.token)->complete(static_cast<nixl_status_t>(ack.status));
    } else if (type == kAddMsg && len == sizeof(AddMsg)) {
        AddMsg msg;
        std::memcpy(&msg, data, sizeof(msg));
        RxRing &rx = rxRing(agent, msg.ring);
        {
            std::lock_guard<std::mutex> lock(rx.mutex);
            rx.adds.emplace(msg.seq, RxRing::Add{msg, agent, rail});
        }
        rx.waiting.fetch_add(1, std::memory_order_seq_cst);
        drain(rx);
    } else if (type == kAbortMsg && len == sizeof(AbortMsg)) {
        AbortMsg msg;
        std::memcpy(&msg, data, sizeof(msg));
        RxRing &rx = rxRing(agent, msg.ring);
        {
            std::lock_guard<std::mutex> lock(rx.mutex);
            rx.aborted_from = std::min(rx.aborted_from, msg.first_seq);
        }
        drain(rx);
    } else {
        NIXL_ERROR << "Device proxy: unknown message (type " << type << ", " << len << " bytes)";
    }
}

nixlLibfabricProxy::RxRing &
nixlLibfabricProxy::rxRing(uint32_t agent, uint32_t ring_id) {
    const uint64_t key = uint64_t(agent) << 32 | ring_id;
    if (tls_rx.owner != id_) {
        tls_rx.owner = id_;
        tls_rx.rings.clear();
    }
    auto cached = tls_rx.rings.find(key);
    if (cached != tls_rx.rings.end()) {
        return *static_cast<RxRing *>(cached->second);
    }
    RxRing *rx;
    {
        std::lock_guard<std::mutex> lock(rx_mutex_);
        auto &slot = rx_rings_[key];
        if (!slot) {
            slot = std::make_unique<RxRing>();
        }
        rx = slot.get();
    }
    tls_rx.rings.emplace(key, rx);
    return *rx;
}

void
nixlLibfabricProxy::drain(RxRing &rx) {
    // Applied under the stream's lock, in seq order; acked after it.
    std::vector<std::pair<RxRing::Add, nixl_status_t>> done;
    {
        std::lock_guard<std::mutex> lock(rx.mutex);
        for (auto it = rx.adds.begin(); it != rx.adds.end() && it->first == rx.applied;
             it = rx.adds.begin()) {
            const AddMsg &msg = it->second.msg;
            nixl_status_t status = NIXL_ERR_BACKEND; // aborted: a put of it failed
            if (msg.seq < rx.aborted_from) {
                if (rx.received[msg.seq % kEpochs].load(std::memory_order_seq_cst) <
                    msg.expected) {
                    break; // its puts are not all in yet
                }
                std::shared_lock<std::shared_mutex> regions(regions_mutex_);
                status = covered(msg.addr, msg.addr + sizeof(uint64_t)) ?
                    counters_->add(msg.addr, msg.value) :
                    NIXL_ERR_NOT_FOUND;
            }
            done.emplace_back(it->second, status);
            rx.adds.erase(it);
            ++rx.applied;
            rx.waiting.fetch_sub(1, std::memory_order_seq_cst);
        }
    }
    for (const auto &[add, status] : done) {
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Device proxy: atomicAdd to " << reinterpret_cast<void *>(add.msg.addr)
                       << " failed: " << status;
        }
        std::shared_ptr<nixlLibfabricConnection> conn;
        {
            std::lock_guard<std::mutex> lock(engine_.connection_state_mutex_);
            if (add.agent < engine_.agent_names_.size()) {
                auto it = engine_.connections_.find(engine_.agent_names_[add.agent]);
                if (it != engine_.connections_.end()) {
                    conn = it->second;
                }
            }
        }
        uint64_t dest = 0;
        bool have_dest = false;
        if (conn) {
            const auto addrs = conn->rail_remote_addr_list_.find(add.rail);
            if (addrs != conn->rail_remote_addr_list_.end() &&
                add.msg.reply_rail < addrs->second.size()) {
                dest = addrs->second[add.msg.reply_rail];
                have_dest = true;
            }
        }
        if (!have_dest) {
            NIXL_ERROR << "Device proxy: no address to ack an atomicAdd of agent " << add.agent;
            continue;
        }
        const AckMsg ack{kAckMsg, static_cast<int32_t>(status), add.msg.token};
        if (sendMsg(add.rail, dest, &ack, sizeof(ack)) != NIXL_SUCCESS) {
            NIXL_ERROR << "Device proxy: failed to ack an atomicAdd of agent " << add.agent;
        }
    }
}

bool
nixlLibfabricProxy::covered(uintptr_t lo, uintptr_t hi) const {
    for (auto it = regions_.begin(); it != regions_.end() && it->first < hi; ++it) {
        if (lo >= it->first && hi <= it->first + it->second) {
            return true;
        }
    }
    return false;
}

void
nixlLibfabricProxy::onRegister(uintptr_t addr, size_t len, bool is_vram) {
    if (is_vram) {
        counters_->premap(addr, len);
    }
    std::unique_lock<std::shared_mutex> lock(regions_mutex_);
    regions_.emplace(addr, len);
}

void
nixlLibfabricProxy::onDeregister(uintptr_t addr, size_t len) {
    std::unique_lock<std::shared_mutex> lock(regions_mutex_);
    for (auto it = regions_.lower_bound(addr); it != regions_.end() && it->first == addr; ++it) {
        if (it->second == len) {
            regions_.erase(it);
            break;
        }
    }
    // Unpin the GDRCopy pages no registration covers any more.
    counters_->unmap(addr, addr + len, [this](uintptr_t lo, uintptr_t hi) {
        for (const auto &[base, length] : regions_) {
            if (base < hi && lo < base + length) {
                return true;
            }
        }
        return false;
    });
}

nixl_status_t
nixlLibfabricProxy::checkCompletion(uint32_t channel,
                                    uint32_t,
                                    const nixl::proxyBackendRequest &request) {
    Request *req = reinterpret_cast<Request *>(request.token);
    if (!req->done()) {
        return NIXL_IN_PROG;
    }
    const auto status = static_cast<nixl_status_t>(req->status.load(std::memory_order_relaxed));
    freeRequest(threadOf(channel), req);
    return status;
}

void
nixlLibfabricProxy::progress(uint32_t channel, uint32_t peer) noexcept {
    Thread &th = threadOf(channel);
    if (peer == 0 && channel == th.index) { // once per worker pass, on its first ring
        pollRails(th);
        if (aborts_pending_.load(std::memory_order_acquire)) {
            sendAborts();
        }
    }
    Ring &ring = rings_[size_t(channel) * max_peers_ + peer];
    if (!ring.parked.empty()) {
        flushParked(th, ring);
    }
}

nixl_status_t
nixlLibfabricProxy::quiesce(uint32_t channel, uint32_t peer) {
    // Puts complete on remote completion and atomicAdds once applied, so a ring with nothing
    // in flight has nothing left to flush.
    if (!rings_[size_t(channel) * max_peers_ + peer].parked.empty()) {
        NIXL_ERROR << "Device proxy: quiesce with parked operations";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricProxy::shutdown() {
    for (size_t r = 0; r < engine_.rail_manager_.getNumRails(); ++r) {
        engine_.rail_manager_.getRail(r).setProxyCallback(nullptr);
    }
    return NIXL_SUCCESS;
}

nixlLibfabricProxy::Request *
nixlLibfabricProxy::allocRequest(Thread &th) {
    Request *req = th.free_list;
    if (req != nullptr) {
        th.free_list = req->next_free;
    } else {
        th.requests.push_back(std::make_unique<Request>());
        req = th.requests.back().get();
    }
    req->status.store(NIXL_SUCCESS, std::memory_order_relaxed);
    req->stream = nullptr;
    return req;
}

void
nixlLibfabricProxy::freeRequest(Thread &th, Request *req) {
    req->next_free = th.free_list;
    th.free_list = req;
}

#endif // NIXL_LIBFABRIC_DEVICE_PROXY
