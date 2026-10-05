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
#include "libfabric_proxy.h"

#ifdef HAVE_NIXL_DEVICE_API

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_set>

#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include "common/nixl_log.h"
#include "libfabric_backend.h"
#include "libfabric_connection.h"
#include "libfabric_proxy_conninfo.h"
#include "libfabric_proxy_fence.h"
#include "libfabric_proxy_profile.h"
#include "serdes/serdes.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef HAVE_GDRCOPY
#include <gdrapi.h>
#endif

namespace {

/** Upper bound on rails a single put is striped over. */
constexpr size_t kMaxStripes = 8;
/** Posted receive buffers per proxy thread for incoming atomicAdd records and acks. */
constexpr size_t kRecvPoolSize = 1024;
/** Ack send buffers per proxy thread; acks beyond them wait in a queue. */
constexpr size_t kAckPoolSize = 1024;
/** An atomicAdd whose ack has not arrived by then failed (its target died). */
constexpr uint64_t kAckTimeoutNs = 10ull * 1000 * 1000 * 1000;
/** Passes between scans for overdue acks while any atomicAdd awaits one. */
constexpr uint32_t kAckScanInterval = 4096;
/** Default for efa_proxy_idle_poll_us: how often an idle thread polls its CQs. */
constexpr uint64_t kDefaultIdlePollUs = 2;
/** quiesce() gives up on a ring that has not drained by then (fatal in the runtime). */
constexpr std::chrono::seconds kQuiesceTimeout{30};
/** Requests beyond the ring capacity, for safety margins. */
constexpr size_t kRequestSlack = 64;
constexpr size_t kCqBatch = 32;
/** Proxy-thread passes between engine rail progress calls (engine progress thread off). */
constexpr uint32_t kEngineProgressInterval = 64;
/** Profiling: log any single post or CQ read slower than this. */
constexpr uint64_t kSlowCallNs = 1000000;
/** After a thread's first CQ error, report further ones at most this often. */
constexpr uint64_t kCqErrorReportNs = 1000000000;
constexpr size_t kCqSize = 16384;
/**
 * Proxy EP queue sizes. The provider allocates, faults in and registers each EP's
 * packet pools in one chunk sized by these the first time the EP is used (the
 * defaults, 4096 tx / 8192 rx, are ~36 MB and ~72 MB and took 5-20 ms under the
 * domain lock). Writes need no receive buffers, so only a thread's control EP
 * (atomicAdd records and acks) gets a real receive queue; posts beyond the
 * transmit queues wait in the retry queues.
 */
constexpr size_t kTxSize = 1024;
constexpr size_t kCtlTxSize = 1024;
constexpr size_t kCtlRxSize = 1024;
constexpr size_t kIdleRxSize = 64;
/** RecvBuf::rail of the control EP's receive buffers. */
constexpr uint32_t kCtlRail = std::numeric_limits<uint32_t>::max();

namespace wire = nixlLibfabricProxyWire;
static_assert(LF_EP_NAME_MAX_LEN <= wire::kMaxEpName, "endpoint names must fit in a message");

uint64_t
nowNs() {
    return nixlLibfabricProxyProfile::now();
}

/** Transparent hash, so lookups by string_view allocate nothing. */
struct NameHash {
    using is_transparent = void;

    size_t
    operator()(std::string_view name) const noexcept {
        return std::hash<std::string_view>{}(name);
    }
};

} // namespace

/* ---------------------------------------------------------------------------
 * Internal state
 * ------------------------------------------------------------------------- */

/**
 * Every posted operation's libfabric context starts with this, so a CQ entry's
 * op_context says what completed.
 */
struct nixlLibfabricProxy::OpCtx {
    enum class Kind : uint8_t { FRAG, RECV, ACK };

    struct fi_context2 ctx; // must stay first: op_context points here
    Kind kind;
};

struct nixlLibfabricProxy::Request {
    /** One context per posted fragment (or the atomicAdd record). */
    struct FragCtx {
        OpCtx op; // must stay first
        Request *owner;
    };

    FragCtx frag[kMaxStripes];
    uint32_t thread = 0;
    uint32_t index = 0; // in the thread's pool; with the generation, names the request in acks
    uint32_t generation = 0; // bumped per allocation, so late acks of a reused slot are ignored
    uint16_t frags_left = 0; // an atomicAdd counts its send and its ack
    bool in_use = false;
    bool awaiting_ack = false; // atomicAdd record posted, the owner's ack not received yet
    nixl_status_t status = NIXL_IN_PROG;
    RingFence *fence = nullptr;
    nixlLibfabricRingFence<Request>::Epoch *epoch = nullptr;
    fi_addr_t dest = FI_ADDR_UNSPEC; // atomic target (owner thread's home EP)
    uint64_t ack_deadline = 0; // nowNs() by which the ack must arrive
    wire::atomicAddMsg msg{}; // atomic send buffer; the request pool is registered memory
    bool inject_cq_error = false; // NIXL_EFA_PROXY_INJECT: fail its first completion
    // Receiver ordering and fence-off modes: counted in fence->outstanding instead
    // of the fence's epochs.
    bool counted = false;
    bool is_atomic = false;
    TxRing *tx = nullptr; // receiver ordering: the ring's sender-side counts
    uint64_t rx_seq = 0; // a put: its epoch; an atomicAdd: its index on the ring

    uint64_t
    token() const noexcept {
        return static_cast<uint64_t>(generation) << 32 | index;
    }

    // Stage timestamps, set only with NIXL_EFA_PROXY_PROFILE.
    uint64_t t_submit = 0;
    uint64_t t_post = 0; // last successful post
    uint64_t t_done = 0; // last completion
    uint8_t prof_op = 0;
    uint8_t prof_class = 0;
};

/**
 * Sender-side counts of one (target incarnation, channel) for receiver ordering.
 * They outlive the ring's fence (quiesce), since the target's counts do; after a
 * failure the target may never see some counted put, so the next fence re-keys
 * the ring (fresh counts on both sides).
 */
struct nixlLibfabricProxy::TxRing {
    uint32_t key = 0; // random, 24 bits (wire::ringImm())
    uint64_t atomics = 0; // atomicAdds sent to the target; the epoch of the next puts
    std::array<uint64_t, wire::kRingEpochs> slot_puts{}; // fragments sent per epoch slot
    uint32_t inflight = 0; // requests not complete yet
    nixl_status_t error = NIXL_SUCCESS; // sticky until re-keyed
};

/** Target-side state of one sender ring: what its next atomicAdd waits for. */
struct nixlLibfabricProxy::RxRing {
    // Put fragments arrived per epoch slot (release by the counting thread).
    std::array<std::atomic<uint64_t>, wire::kRingEpochs> received{};
    std::atomic<uint64_t> applied{0}; // atomicAdds applied or failed, in seq order
    std::atomic<int32_t> error{NIXL_SUCCESS}; // first failed atomicAdd: later ones fail
    std::mutex sender_mutex;
    std::string sender; // reply EP of the first atomicAdd (detects key collisions)
};

/** A receiver-ordered atomicAdd waiting for its ring. */
struct nixlLibfabricProxy::Deferred {
    RxRing *ring;
    wire::atomicAddMsg msg;
    fi_addr_t reply;
    uint64_t arrived;
    bool data_ready = false; // its ring's puts arrived (seen before the flush below)
    bool flushed = false; // GPUDirect RDMA writes flushed since data_ready
};

struct nixlLibfabricProxy::RingFence {
    nixlLibfabricRingFence<Request> order;
    uint32_t next_rail = 0; // unstriped puts rotate over the buffer's rails
    // Receiver ordering and fence-off: requests in flight, and the first failure
    // (later atomicAdds fail, as with the fence).
    uint32_t outstanding = 0;
    nixl_status_t error = NIXL_SUCCESS;
    TxRing *tx = nullptr;
    const nixlLibfabricConnection *tx_conn = nullptr;

    [[nodiscard]] bool
    idle() const {
        return order.idle() && outstanding == 0;
    }
};

struct nixlLibfabricProxy::RecvBuf {
    OpCtx op; // must stay first
    uint32_t rail = kCtlRail; // whose EP it is posted on: a data rail, or the control EP
    wire::anyMsg msg;
};

struct nixlLibfabricProxy::AckBuf {
    OpCtx op; // must stay first
    wire::atomicAckMsg msg;
};

/** An ack waiting for a send buffer or transmit queue room. */
struct nixlLibfabricProxy::PendingAck {
    fi_addr_t dest;
    uint64_t token;
    nixl_status_t status;
};

struct nixlLibfabricProxy::PendingPost {
    enum class Kind : uint8_t { WRITE, SEND };
    Kind kind;
    uint32_t rail;
    Request::FragCtx *fctx;
    void *local;
    size_t len;
    void *desc;
    fi_addr_t dest;
    uint64_t raddr;
    uint64_t rkey;
    uint64_t queued_at = 0; // entry into the retry queue (profiling only)
    bool inject_error = false; // NIXL_EFA_PROXY_INJECT: fail this post
    bool has_imm = false; // a write with remote CQ data
    uint32_t imm = 0;
    uint64_t rx_index = 0; // the fragment's epoch (tracing)
};

struct nixlLibfabricProxy::PeerAddrs {
    // Expired once the connection is gone; its address may then be reused.
    std::weak_ptr<const nixlLibfabricConnection> conn;
    // Inserted into this thread's AVs on first use; FI_ADDR_UNSPEC until then.
    // rail_ep[local_rail][remote_ep] -> remote engine EP in that rail's AV
    std::vector<std::vector<fi_addr_t>> rail_ep;
    // home[remote_thread] -> remote proxy home EP in this thread's home-rail AV
    std::vector<fi_addr_t> home;
    // data_ep[local_rail][remote_thread * remote_rails + remote_rail] -> remote proxy
    // data EP in that rail's AV (puts with remote CQ data); sized on first use
    std::vector<std::vector<fi_addr_t>> data_ep;
    bool warned_no_data_eps = false;
};

struct nixlLibfabricProxy::RailRes {
    struct fid_domain *domain = nullptr;
    struct fi_info *info = nullptr;
    struct fid_ep *ep = nullptr;
    struct fid_cq *cq = nullptr;
    struct fid_av *av = nullptr;
    bool virt_addr = true;
    uint64_t posts = 0; // profiling only
    uint64_t cq_reads = 0; // profiling only
    std::array<char, LF_EP_NAME_MAX_LEN> name{}; // published when imm_puts_
    // efa_proxy_test_data_rx_post: receives posted on this data EP
    std::vector<nixlLibfabricProxy::RecvBuf> rx;
    struct fid_mr *rx_mr = nullptr;
    void *rx_desc = nullptr;
};

struct nixlLibfabricProxy::Thread {
    uint32_t id = 0;
    uint32_t home = 0;
    std::thread::id owner{}; // the worker that drives this thread's channels
    // Held while this thread applies an add; registration changes take all of them.
    std::mutex regions_lock;
    std::vector<RailRes> rails;
    // Data CQs worth polling: every rail this thread has written on (control
    // traffic has its own CQ). The rail domains are FI_THREAD_SAFE, so skipping
    // idle CQs also avoids taking their domain locks.
    std::vector<uint32_t> poll_rails;
    std::vector<bool> polled;
    uint32_t passes = 0;
    uint64_t next_idle_poll = 0; // nowNs() before which an idle thread skips its CQs
    uint64_t cq_errors = 0; // since the last report
    uint64_t cq_error_report = 0; // time of the last report, 0 before the first
    std::unique_ptr<nixlLibfabricProxyProfile> prof; // NIXL_EFA_PROXY_PROFILE only
    uint64_t last_pass = 0;
    // Control EP on the home rail, with its own CQ and AV (EFA does not share
    // them between EPs): atomicAdd records and acks have their own send queue,
    // so they never wait behind bulk writes. Every control address (counter
    // owners, ack destinations) lives in ctl_av.
    struct fid_ep *ctl_ep = nullptr;
    struct fid_cq *ctl_cq = nullptr;
    struct fid_av *ctl_av = nullptr;
    struct fi_info *ctl_info = nullptr;
    uint64_t ctl_cq_reads = 0; // profiling only
    std::array<char, LF_EP_NAME_MAX_LEN> home_name{}; // the control EP's name
    size_t home_name_len = 0;

    std::vector<Request> reqs;
    std::vector<Request *> free_reqs;
    struct fid_mr *req_mr = nullptr;
    void *req_desc = nullptr;

    std::vector<RecvBuf> recvs;
    struct fid_mr *recv_mr = nullptr;
    void *recv_desc = nullptr;

    std::vector<AckBuf> acks;
    std::vector<AckBuf *> free_acks;
    struct fid_mr *ack_mr = nullptr;
    void *ack_desc = nullptr;
    std::deque<PendingAck> pending_acks; // in arrival order
    // Senders' home endpoints, by name, in the home rail's AV (where acks go).
    std::unordered_map<std::string, fi_addr_t, NameHash, std::equal_to<>> reply_addrs;
    // atomicAdds whose ack is outstanding (entries go stale once acked; pruned on scans).
    std::vector<Request *> awaiting;
    uint32_t ack_scan = 0;

    // Back-pressured posts per rail, plus one for the control EP (last): a full
    // transmit queue does not hold back the others (the fence, not posting
    // order, orders a ring).
    std::vector<std::deque<PendingPost>> retry;
    size_t retry_count = 0;
    std::vector<RecvBuf *> recv_retry; // receive buffers the EP could not take yet
    std::unordered_map<uint64_t, RingFence> fences;
    std::unordered_map<const nixlLibfabricConnection *, PeerAddrs> peers;
    int cuda_dev = -1;

    // Receiver ordering, sender side: counts per (peer, incarnation, ring).
    std::map<std::tuple<std::string, uint64_t, uint32_t>, TxRing> tx_rings;
    std::mt19937 rng{std::random_device{}()};
    // Target side: rings by key (shared RxRing objects, cached), the atomicAdds
    // this thread owns that wait for their rings, and whether the last CQ sweep
    // found remote CQ data (the thread then keeps polling on every pass).
    std::unordered_map<uint32_t, RxRing *> rx_cache;
    uint32_t rx_last_key = 0;
    RxRing *rx_last = nullptr;
    std::vector<Deferred> deferred;
    bool rx_active = false;

    // Profiling of the remote CQ data path (NIXL_EFA_PROXY_PROFILE only).
    struct RxStats {
        uint64_t entries = 0; // CQ entries with remote CQ data
        uint64_t reads = 0; // fi_cq_read calls that returned any
        uint64_t read_ns = 0; // in those calls
        uint64_t decode_ns = 0; // decoding and counting their entries
        uint64_t sweeps = 0; // pollCqs() sweeps that found any
        uint64_t sweep_ns = 0; // in those sweeps
        uint64_t first = 0; // first and last entry (nowNs)
        uint64_t last = 0;
        uint64_t consumed_rx = 0; // entries that consumed a posted data-EP receive
        uint64_t deferred = 0; // atomicAdds that had to wait at the target
    } rx_stats;

    uint64_t sweep_imm = 0; // entries found in the current sweep
    // Profiling, any ordering mode: time in progress() passes that did work (read a
    // CQ entry or applied a deferred add), and the span from the first to the last.
    uint64_t events = 0;
    uint64_t busy_ns = 0;
    uint64_t busy_first = 0;
    uint64_t busy_last = 0;
#ifdef HAVE_CUDA
    // Per device, for adds without GDRCopy: never the legacy default stream, which
    // would wait behind a kernel spinning on the very counter being added to.
    std::unordered_map<int, cudaStream_t> streams;
#endif
};

/**
 * Host access to VRAM counters for the target-side add. GDRCopy maps each 64 KB
 * GPU page on first use. add() runs under its thread's region lock and
 * dropRange() under all of them, so a page is never unmapped while an add
 * writes to it; the map's own mutex is never held across a pin, a map or a copy.
 */
struct nixlLibfabricProxy::CounterMap {
#ifdef HAVE_GDRCOPY
    static constexpr uintptr_t kPage = GPU_PAGE_SIZE;

    struct Mapping {
        gdr_mh_t mh{};
        void *bar = nullptr;
        size_t off = 0;
    };

    gdr_t gdr = nullptr;
    // GDRCopy's handle is not thread-safe: pin, map, info, unmap and unpin take
    // gdr_mutex (the copies through a mapping do not need it).
    std::mutex gdr_mutex;
    std::mutex mutex; // pages and unmappable; never held across a GDRCopy call
    std::unordered_map<uintptr_t, Mapping> pages;
    std::unordered_set<uintptr_t> unmappable; // pages GDRCopy refused; logged once

    explicit CounterMap(bool use_gdrcopy) {
        if (!use_gdrcopy) {
            NIXL_WARN << "EFA proxy: GDRCopy disabled; VRAM atomicAdd uses CUDA copies";
            return;
        }
        gdr = gdr_open();
        if (!gdr) {
            NIXL_WARN << "EFA proxy: gdr_open failed; VRAM atomicAdd uses CUDA copies (slow)";
        }
    }

    ~CounterMap() {
        for (auto &kv : pages) {
            unmap(kv.second);
        }
        pages.clear();
        if (gdr) {
            gdr_close(gdr);
        }
    }

    [[nodiscard]] bool
    usable() const {
        return gdr != nullptr;
    }

    /** NIXL_ERR_NOT_SUPPORTED if GDRCopy cannot map the page (the caller falls back). */
    nixl_status_t
    add(uintptr_t addr, uint64_t value) {
        const uintptr_t page = addr & ~(kPage - 1);
        Mapping m;
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = pages.find(page);
            if (it != pages.end()) {
                m = it->second;
                found = true;
            }
        }
        if (!found) {
            bool known_bad = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                known_bad = unmappable.count(page) != 0;
            }
            if (known_bad || !pinAndMap(page, m)) {
                if (!known_bad) {
                    std::lock_guard<std::mutex> lock(mutex);
                    unmappable.insert(page);
                }
                return NIXL_ERR_NOT_SUPPORTED;
            }
            Mapping duplicate;
            bool lost = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto [it, inserted] = pages.emplace(page, m);
                if (!inserted) { // another thread mapped the page first
                    duplicate = m;
                    m = it->second;
                    lost = true;
                }
            }
            if (lost) {
                unmap(duplicate);
            }
        }
        // 8-byte aligned (checked by the caller), so the word never crosses the page.
        auto *word =
            reinterpret_cast<uint64_t *>(static_cast<char *>(m.bar) + m.off + (addr - page));
        uint64_t v = 0;
        gdr_copy_from_mapping(m.mh, &v, word, sizeof(v));
        v += value;
        gdr_copy_to_mapping(m.mh, word, &v, sizeof(v));
        return NIXL_SUCCESS;
    }

    /** Unmap the pages of [base, base + len) for which @p keep(page, kPage) is false. */
    template<typename Keep>
    void
    dropRange(uintptr_t base, size_t len, Keep &&keep) {
        std::vector<Mapping> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto it = pages.begin(); it != pages.end();) {
                if (it->first + kPage > base && it->first < base + len && !keep(it->first, kPage)) {
                    dropped.push_back(it->second);
                    it = pages.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto it = unmappable.begin(); it != unmappable.end();) {
                it = *it + kPage > base && *it < base + len ? unmappable.erase(it) : std::next(it);
            }
        }
        for (Mapping &m : dropped) {
            unmap(m);
        }
    }

private:
    bool
    pinAndMap(uintptr_t page, Mapping &m) {
        std::lock_guard<std::mutex> lock(gdr_mutex);
        if (gdr_pin_buffer(gdr, page, kPage, 0, 0, &m.mh) != 0) {
            NIXL_ERROR << "EFA proxy: gdr_pin_buffer failed for page " << std::hex << page;
            return false;
        }
        if (gdr_map(gdr, m.mh, &m.bar, kPage) != 0) {
            gdr_unpin_buffer(gdr, m.mh);
            NIXL_ERROR << "EFA proxy: gdr_map failed for page " << std::hex << page;
            return false;
        }
        gdr_info_t info{};
        gdr_get_info(gdr, m.mh, &info);
        m.off = static_cast<size_t>(info.va - page);
        return true;
    }

    void
    unmap(Mapping &m) {
        std::lock_guard<std::mutex> lock(gdr_mutex);
        gdr_unmap(gdr, m.mh, m.bar, kPage);
        gdr_unpin_buffer(gdr, m.mh);
    }
#else
    explicit CounterMap(bool) {}

    [[nodiscard]] bool
    usable() const {
        return false;
    }

    nixl_status_t
    add(uintptr_t, uint64_t) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    template<typename Keep>
    void
    dropRange(uintptr_t, size_t, Keep &&) {}
#endif
};

/**
 * Test-only fault injection, enabled by NIXL_EFA_PROXY_INJECT (comma-separated)
 * in builds without NDEBUG:
 *  - eagain_every=K:  every K-th post attempt returns -FI_EAGAIN without reaching
 *                     libfabric (back-pressure and the retry queues);
 *  - post_error_at=N: the N-th put request fails to post;
 *  - cq_error_at=N:   the N-th put request completes with an error;
 *  - no_gdrcopy=1:    apply VRAM atomicAdds with CUDA copies instead of GDRCopy.
 * Put requests are counted from 1 over all proxy threads of this backend.
 */
struct nixlLibfabricProxy::Inject {
    uint64_t eagain_every = 0;
    uint64_t post_error_at = 0;
    uint64_t cq_error_at = 0;
    bool no_gdrcopy = false;
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> puts{0};

    static std::unique_ptr<Inject>
    fromEnv() {
        const char *env = std::getenv("NIXL_EFA_PROXY_INJECT");
        if (env == nullptr || *env == '\0') {
            return nullptr;
        }
#ifdef NDEBUG
        NIXL_WARN << "EFA proxy: NIXL_EFA_PROXY_INJECT is ignored in NDEBUG builds";
        return nullptr;
#else
        auto inject = std::make_unique<Inject>();
        std::stringstream spec(env);
        std::string item;
        while (std::getline(spec, item, ',')) {
            const size_t eq = item.find('=');
            const std::string key = item.substr(0, eq);
            uint64_t value = 0;
            try {
                value = eq == std::string::npos ? 0 : std::stoull(item.substr(eq + 1));
            }
            catch (const std::exception &) {
                value = 0;
            }
            if (key == "eagain_every") {
                inject->eagain_every = value;
            } else if (key == "post_error_at") {
                inject->post_error_at = value;
            } else if (key == "cq_error_at") {
                inject->cq_error_at = value;
            } else if (key == "no_gdrcopy") {
                inject->no_gdrcopy = value != 0;
            } else {
                NIXL_WARN << "EFA proxy: unknown NIXL_EFA_PROXY_INJECT item '" << item << "'";
            }
        }
        NIXL_WARN << "EFA proxy: fault injection enabled: eagain_every=" << inject->eagain_every
                  << " post_error_at=" << inject->post_error_at
                  << " cq_error_at=" << inject->cq_error_at << " no_gdrcopy=" << inject->no_gdrcopy;
        return inject;
#endif
    }
};

/* ---------------------------------------------------------------------------
 * Construction and callback table
 * ------------------------------------------------------------------------- */

nixlLibfabricProxy::nixlLibfabricProxy(nixlLibfabricEngine &engine) : engine_(engine) {
    const nixl_b_params_t &params = engine_.getCustomParams();
    if (params.count("efa_proxy_delivery_complete") != 0) {
        NIXL_WARN << "EFA proxy: efa_proxy_delivery_complete is no longer supported; puts always "
                  << "use FI_DELIVERY_COMPLETE, which the put -> atomicAdd ordering relies on";
    }
    idle_poll_ns_ = kDefaultIdlePollUs * 1000;
    if (auto idle = params.find("efa_proxy_idle_poll_us"); idle != params.end()) {
        try {
            idle_poll_ns_ = std::stoull(idle->second) * 1000;
        }
        catch (const std::exception &) {
            NIXL_WARN << "EFA proxy: invalid efa_proxy_idle_poll_us '" << idle->second
                      << "'; using " << kDefaultIdlePollUs;
        }
    }
    auto it = params.find("efa_proxy_rail_policy");
    if (it != params.end()) {
        if (it->second == "ring") {
            rail_per_thread_ = false;
        } else if (it->second != "thread") {
            NIXL_WARN << "EFA proxy: unknown efa_proxy_rail_policy '" << it->second
                      << "'; using 'thread'";
        }
    }
    const char *profile = std::getenv("NIXL_EFA_PROXY_PROFILE");
    profile_ = profile != nullptr && *profile != '\0' && std::string(profile) != "0";
    const char *rx_trace = std::getenv("NIXL_EFA_PROXY_RXTRACE");
    rx_trace_ = rx_trace != nullptr && *rx_trace != '\0' && std::string(rx_trace) != "0";
    inject_ = Inject::fromEnv();
    counters_ = std::make_unique<CounterMap>(!(inject_ && inject_->no_gdrcopy));

    if (auto rx = params.find("efa_proxy_rx_thread"); rx != params.end()) {
        if (rx->second == "rail") {
            rx_by_rail_ = true;
        } else if (rx->second != "ring") {
            NIXL_WARN << "EFA proxy: unknown efa_proxy_rx_thread '" << rx->second
                      << "'; using 'ring'";
        }
    }
    if (auto order = params.find("efa_proxy_ordering"); order != params.end()) {
        if (order->second == "receiver") {
            receiver_order_ = true;
        } else if (order->second != "sender") {
            NIXL_WARN << "EFA proxy: unknown efa_proxy_ordering '" << order->second
                      << "'; using 'sender'";
        }
    }
    const auto count_param = [&params](const char *name, size_t &value) {
        auto it = params.find(name);
        if (it == params.end()) {
            return false;
        }
        try {
            value = std::stoull(it->second);
        }
        catch (const std::exception &) {
            NIXL_WARN << "EFA proxy: invalid " << name << " '" << it->second << "'; ignored";
            return false;
        }
        return true;
    };
    data_rx_size_ = kIdleRxSize;
    count_param("efa_proxy_data_rx_size", data_rx_size_);
    size_t rx_flush = 1;
    count_param("efa_proxy_rx_flush", rx_flush);
    rx_flush_ = rx_flush != 0;
    size_t put_imm = 0, fence_off = 0;
    const bool test_params = count_param("efa_proxy_test_put_imm", put_imm) |
        count_param("efa_proxy_test_fence_off", fence_off) |
        count_param("efa_proxy_test_data_rx_post", test_data_rx_post_);
#ifdef NDEBUG
    if (test_params) {
        NIXL_WARN << "EFA proxy: efa_proxy_test_* parameters are ignored in NDEBUG builds";
    }
    put_imm = fence_off = test_data_rx_post_ = 0;
#else
    static_cast<void>(test_params);
#endif
    test_put_imm_ = put_imm != 0;
    test_fence_off_ = fence_off != 0;
    if (receiver_order_ && !rx_flush_) {
        NIXL_WARN << "EFA proxy: efa_proxy_rx_flush=0: a receiver-ordered atomicAdd may become "
                  << "visible to the GPU before the puts it orders (measurement only)";
    }
    if (test_fence_off_ && receiver_order_) {
        NIXL_WARN << "EFA proxy: efa_proxy_test_fence_off applies to sender ordering; ignored";
        test_fence_off_ = false;
    }
    if (test_fence_off_) {
        NIXL_WARN << "EFA proxy: TEST ONLY: atomicAdds do not wait for earlier operations; "
                  << "put -> atomicAdd ordering is NOT guaranteed";
    }
    if (test_put_imm_) {
        NIXL_WARN << "EFA proxy: TEST ONLY: puts carry remote CQ data to the target's proxy "
                  << "threads (sender-side ordering unchanged)";
    }
    imm_puts_ = receiver_order_ || test_put_imm_;
    incarnation_ = std::random_device{}() | static_cast<uint64_t>(std::random_device{}()) << 32;
}

nixlLibfabricProxy::~nixlLibfabricProxy() {
    static_cast<void>(shutdown());
}

nixlProxyBackendOps
nixlLibfabricProxy::makeOps() {
    nixlProxyBackendOps ops;
    ops.init = [this](const nixlProxyConfig &c) { return init(c); };
    ops.submit = [this](const nixlBackendProxySubmission &s, nixlBackendProxyRequest &r) {
        return submit(s, r);
    };
    ops.check_completion = [this](const nixlBackendProxyRequest &r) { return checkCompletion(r); };
    ops.quiesce = [this](uint32_t channel, uint32_t peer) { return quiesce(channel, peer); };
    ops.progress = [this](uint32_t channel, uint32_t peer) { return progress(channel, peer); };
    ops.shutdown = [this]() { return shutdown(); };
    return ops;
}

/* ---------------------------------------------------------------------------
 * init / shutdown
 * ------------------------------------------------------------------------- */

nixl_status_t
nixlLibfabricProxy::init(const nixlProxyConfig &config) {
    config_ = config;
    threads_ = config.effectiveThreadCount();
    rails_ = engine_.rail_manager_.getNumRails();
    if (threads_ == 0 || rails_ == 0) {
        NIXL_ERROR << "EFA proxy: needs at least one thread and one rail";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Home rails (atomicAdd records) on the EFA devices next to this process's
    // GPU, so signals stay off the other GPUs' NICs; all rails without a GPU.
    std::vector<size_t> home_rails;
#ifdef HAVE_CUDA
    int dev = 0;
    char bus_id[32] = {};
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev) == cudaSuccess) {
        home_rails = engine_.rail_manager_.railsForAccelerator(bus_id);
    }
#endif
    if (home_rails.empty()) {
        for (size_t r = 0; r < rails_; ++r) {
            home_rails.push_back(r);
        }
    }
    rx_rails_.clear();
    for (size_t r : home_rails) {
        rx_rails_.push_back(static_cast<uint32_t>(r));
    }
    if (receiver_order_ && config.ring_depth > wire::kRingEpochs) {
        // Epoch slots are reused every kRingEpochs atomicAdds of a ring.
        NIXL_ERROR << "EFA proxy: receiver ordering supports a ring depth of at most "
                   << wire::kRingEpochs << ", not " << config.ring_depth;
        return NIXL_ERR_INVALID_PARAM;
    }

    for (uint32_t t = 0; t < threads_; ++t) {
        auto th = std::make_unique<Thread>();
        th->id = t;
        if (profile_) {
            th->prof = std::make_unique<nixlLibfabricProxyProfile>();
        }
        th->home = static_cast<uint32_t>(home_rails[t % home_rails.size()]);
        th->rails.resize(rails_);
        th->retry.resize(rails_ + 1);
        th->polled.assign(rails_, false); // data CQs join poll_rails once posted on

        for (size_t r = 0; r < rails_; ++r) {
            RailRes &rr = th->rails[r];
            const nixlLibfabricRail &rail = engine_.rail_manager_.getRail(r);
            rr.domain = rail.getDomain();
            rr.info = fi_dupinfo(rail.getRailInfo());
            if (!rr.info) {
                NIXL_ERROR << "EFA proxy: fi_dupinfo failed for thread " << t << " rail " << r;
                releaseThread(*th);
                return NIXL_ERR_BACKEND;
            }
            rr.info->tx_attr->size = std::min(rr.info->tx_attr->size, kTxSize);
            rr.info->rx_attr->size = std::min(rr.info->rx_attr->size, data_rx_size_);
            rr.virt_addr = (rr.info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;

            struct fi_cq_attr cq_attr = {};
            cq_attr.format = FI_CQ_FORMAT_DATA;
            cq_attr.wait_obj = FI_WAIT_NONE;
            cq_attr.size = kCqSize;
            struct fi_av_attr av_attr = {};
            int ret = fi_cq_open(rr.domain, &cq_attr, &rr.cq, nullptr);
            if (!ret) {
                ret = fi_av_open(rr.domain, &av_attr, &rr.av, nullptr);
            }
            if (!ret) {
                ret = fi_endpoint(rr.domain, rr.info, &rr.ep, nullptr);
            }
            if (!ret) {
                ret = fi_ep_bind(rr.ep, &rr.cq->fid, FI_TRANSMIT | FI_RECV);
            }
            if (!ret) {
                ret = fi_ep_bind(rr.ep, &rr.av->fid, 0);
            }
            if (!ret && rail.configureProxyEndpoint(rr.ep) != NIXL_SUCCESS) {
                ret = -FI_EINVAL;
            }
            if (!ret) {
                ret = fi_enable(rr.ep);
            }
            if (!ret) {
                size_t len = rr.name.size();
                ret = fi_getname(&rr.ep->fid, rr.name.data(), &len);
            }
            if (ret) {
                NIXL_ERROR << "EFA proxy: endpoint setup failed for thread " << t << " rail " << r
                           << ": " << fi_strerror(-ret);
                releaseThread(*th);
                return NIXL_ERR_BACKEND;
            }
        }

        // The control EP: published as this thread's home EP.
        {
            RailRes &home = th->rails[th->home];
            const nixlLibfabricRail &rail = engine_.rail_manager_.getRail(th->home);
            int ret = -FI_ENOMEM;
            th->ctl_info = fi_dupinfo(rail.getRailInfo());
            if (th->ctl_info) {
                th->ctl_info->tx_attr->size = std::min(th->ctl_info->tx_attr->size, kCtlTxSize);
                th->ctl_info->rx_attr->size = std::min(th->ctl_info->rx_attr->size, kCtlRxSize);
                struct fi_cq_attr cq_attr = {};
                cq_attr.format = FI_CQ_FORMAT_DATA;
                cq_attr.wait_obj = FI_WAIT_NONE;
                cq_attr.size = kCqSize;
                struct fi_av_attr av_attr = {};
                ret = fi_cq_open(home.domain, &cq_attr, &th->ctl_cq, nullptr);
                if (!ret) {
                    ret = fi_av_open(home.domain, &av_attr, &th->ctl_av, nullptr);
                }
                if (!ret) {
                    ret = fi_endpoint(home.domain, th->ctl_info, &th->ctl_ep, nullptr);
                }
            }
            if (!ret) {
                ret = fi_ep_bind(th->ctl_ep, &th->ctl_cq->fid, FI_TRANSMIT | FI_RECV);
            }
            if (!ret) {
                ret = fi_ep_bind(th->ctl_ep, &th->ctl_av->fid, 0);
            }
            if (!ret && rail.configureProxyEndpoint(th->ctl_ep) != NIXL_SUCCESS) {
                ret = -FI_EINVAL;
            }
            if (!ret) {
                ret = fi_enable(th->ctl_ep);
            }
            if (!ret) {
                size_t len = th->home_name.size();
                ret = fi_getname(&th->ctl_ep->fid, th->home_name.data(), &len);
                th->home_name_len = len;
            }
            if (ret) {
                NIXL_ERROR << "EFA proxy: control endpoint setup failed for thread " << t << ": "
                           << fi_strerror(-ret);
                releaseThread(*th);
                return NIXL_ERR_BACKEND;
            }
        }

        // One request per ring slot this thread can have in flight.
        uint32_t owned_channels = 0;
        for (uint32_t c = t; c < config.channel_count; c += threads_) {
            ++owned_channels;
        }
        const size_t ring_slots =
            static_cast<size_t>(owned_channels) * config.max_peers * config.ring_depth;
        const size_t nreq = ring_slots + kRequestSlack;
        th->reqs.resize(nreq);
        th->free_reqs.reserve(nreq);
        for (size_t i = 0; i < th->reqs.size(); ++i) {
            Request &req = th->reqs[i];
            req.thread = t;
            req.index = static_cast<uint32_t>(i);
            for (auto &f : req.frag) {
                f.op.kind = OpCtx::Kind::FRAG;
                f.owner = &req;
            }
            th->free_reqs.push_back(&req);
        }

        const size_t rx_size = th->ctl_info->rx_attr ? th->ctl_info->rx_attr->size : kRecvPoolSize;
        th->recvs.resize(std::max<size_t>(1, std::min(kRecvPoolSize, rx_size)));
        for (auto &buf : th->recvs) {
            buf.op.kind = OpCtx::Kind::RECV;
        }
        th->acks.resize(kAckPoolSize);
        th->free_acks.reserve(kAckPoolSize);
        for (auto &buf : th->acks) {
            buf.op.kind = OpCtx::Kind::ACK;
            th->free_acks.push_back(&buf);
        }
        struct fid_domain *home_domain = th->rails[th->home].domain;
        int ret = fi_mr_reg(home_domain,
                            th->reqs.data(),
                            th->reqs.size() * sizeof(Request),
                            FI_SEND,
                            0,
                            0,
                            0,
                            &th->req_mr,
                            nullptr);
        if (!ret) {
            ret = fi_mr_reg(home_domain,
                            th->recvs.data(),
                            th->recvs.size() * sizeof(RecvBuf),
                            FI_RECV,
                            0,
                            0,
                            0,
                            &th->recv_mr,
                            nullptr);
        }
        if (!ret) {
            ret = fi_mr_reg(home_domain,
                            th->acks.data(),
                            th->acks.size() * sizeof(AckBuf),
                            FI_SEND,
                            0,
                            0,
                            0,
                            &th->ack_mr,
                            nullptr);
        }
        if (ret) {
            NIXL_ERROR << "EFA proxy: fi_mr_reg failed for thread " << t << ": "
                       << fi_strerror(-ret);
            releaseThread(*th);
            return NIXL_ERR_BACKEND;
        }
        th->req_desc = fi_mr_desc(th->req_mr);
        th->recv_desc = fi_mr_desc(th->recv_mr);
        th->ack_desc = fi_mr_desc(th->ack_mr);

        for (auto &buf : th->recvs) {
            if (postRecv(*th, &buf) == NIXL_ERR_BACKEND) {
                releaseThread(*th);
                return NIXL_ERR_BACKEND;
            }
        }

        // Puts with remote CQ data arrive on the data EPs of the rails next to
        // the GPU (where GPU buffers are registered): their CQs are polled too, by
        // every thread, or with efa_proxy_rx_thread=rail only by the rail's own.
        if (imm_puts_) {
            for (size_t i = 0; i < rx_rails_.size(); ++i) {
                const uint32_t r = rx_rails_[i];
                if (rx_by_rail_ && i % threads_ != t) {
                    continue;
                }
                if (!th->polled[r]) {
                    th->polled[r] = true;
                    th->poll_rails.push_back(r);
                }
            }
        }
        for (size_t r = 0; test_data_rx_post_ != 0 && r < rails_; ++r) {
            if (!th->polled[r]) {
                continue;
            }
            RailRes &rr = th->rails[r];
            rr.rx.resize(test_data_rx_post_);
            int rc = fi_mr_reg(rr.domain,
                               rr.rx.data(),
                               rr.rx.size() * sizeof(RecvBuf),
                               FI_RECV,
                               0,
                               0,
                               0,
                               &rr.rx_mr,
                               nullptr);
            if (rc) {
                NIXL_ERROR << "EFA proxy: fi_mr_reg failed for data receives: " << fi_strerror(-rc);
                releaseThread(*th);
                return NIXL_ERR_BACKEND;
            }
            rr.rx_desc = fi_mr_desc(rr.rx_mr);
            for (auto &buf : rr.rx) {
                buf.op.kind = OpCtx::Kind::RECV;
                buf.rail = static_cast<uint32_t>(r);
                if (postRecv(*th, &buf) == NIXL_ERR_BACKEND) {
                    releaseThread(*th);
                    return NIXL_ERR_BACKEND;
                }
            }
        }
        thread_state_.push_back(std::move(th));
    }

    const fi_info *ctl = thread_state_[0]->ctl_info;
    NIXL_INFO << "EFA proxy: EP queues: transmit " << kTxSize << " and receive " << kIdleRxSize
              << " (at most) on data EPs; transmit " << ctl->tx_attr->size << " and receive "
              << ctl->rx_attr->size << " on control EPs";
    std::string homes;
    for (const auto &th : thread_state_) {
        homes += (homes.empty() ? "" : ",") + std::to_string(th->home);
    }
    NIXL_INFO << "EFA proxy: " << threads_ << " thread(s) x " << rails_
              << " rail(s) = " << threads_ * rails_ << " endpoint(s); home rails " << homes
              << "; small puts per " << (rail_per_thread_ ? "thread" : "ring")
              << " rail; idle poll every " << idle_poll_ns_ / 1000 << " us; ordering "
              << (receiver_order_     ? "receiver" :
                      test_fence_off_ ? "none (test)" :
                                        "sender")
              << (test_put_imm_ ? " with remote CQ data puts (test)" : "")
              << (imm_puts_ ? (rx_by_rail_ ? ", counted by rail" : ", counted by ring") : "")
              << "; fabric "
              << (ctl->fabric_attr && ctl->fabric_attr->name ? ctl->fabric_attr->name : "?")
              << ", data EP receive queue " << data_rx_size_ << ", " << test_data_rx_post_
              << " receive(s) posted per data EP"
              << (receiver_order_ ? (rx_flush_ ? "; GPUDirect RDMA writes flushed before adds" :
                                                 "; GPUDirect RDMA writes NOT flushed") :
                                    "");
    return NIXL_SUCCESS;
}

void
nixlLibfabricProxy::releaseThread(Thread &th) {
    // Endpoints first, so no operation references the CQs, AVs or MRs below.
    const auto close = [](struct fid *f) {
        if (f) {
            fi_close(f);
        }
    };
    close(th.ctl_ep ? &th.ctl_ep->fid : nullptr);
    th.ctl_ep = nullptr;
    for (auto &rr : th.rails) {
        close(rr.ep ? &rr.ep->fid : nullptr);
        rr.ep = nullptr;
    }
    close(th.ctl_cq ? &th.ctl_cq->fid : nullptr);
    close(th.ctl_av ? &th.ctl_av->fid : nullptr);
    th.ctl_cq = nullptr;
    th.ctl_av = nullptr;
    for (auto &rr : th.rails) {
        close(rr.cq ? &rr.cq->fid : nullptr);
        close(rr.av ? &rr.av->fid : nullptr);
        rr.cq = nullptr;
        rr.av = nullptr;
    }
    for (auto &rr : th.rails) {
        if (rr.info) {
            fi_freeinfo(rr.info);
            rr.info = nullptr;
        }
    }
    if (th.ctl_info) {
        fi_freeinfo(th.ctl_info);
        th.ctl_info = nullptr;
    }
    close(th.req_mr ? &th.req_mr->fid : nullptr);
    close(th.recv_mr ? &th.recv_mr->fid : nullptr);
    close(th.ack_mr ? &th.ack_mr->fid : nullptr);
    for (auto &rr : th.rails) {
        close(rr.rx_mr ? &rr.rx_mr->fid : nullptr);
        rr.rx_mr = nullptr;
    }
    th.req_mr = nullptr;
    th.recv_mr = nullptr;
    th.ack_mr = nullptr;
    th.retry.clear();
    th.retry_count = 0;
    th.recv_retry.clear();
    th.pending_acks.clear();
    th.reply_addrs.clear();
    th.awaiting.clear();
    th.poll_rails.clear();
    th.fences.clear();
    th.peers.clear();
    th.tx_rings.clear();
    th.rx_cache.clear();
    th.rx_last = nullptr;
    th.deferred.clear();
#ifdef HAVE_CUDA
    for (auto &[device, stream] : th.streams) {
        if (stream != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream));
        }
    }
    th.streams.clear();
#endif
}

nixl_status_t
nixlLibfabricProxy::shutdown() {
    // Runs after the runtime has joined every proxy thread.
    if (profile_ && !thread_state_.empty()) {
        nixlLibfabricProxyProfile total;
        for (const auto &th : thread_state_) {
            total.merge(*th->prof);
        }
        for (const std::string &line : total.report()) {
            NIXL_INFO << "EFA proxy profile: " << line;
        }
        reportRx();
    }
    for (auto &th : thread_state_) {
        if (!th->deferred.empty()) {
            NIXL_WARN << "EFA proxy: thread " << th->id << ": " << th->deferred.size()
                      << " receiver-ordered atomicAdd(s) still waiting at shutdown";
        }
        if (th->cq_errors != 0) {
            NIXL_ERROR << "EFA proxy: thread " << th->id << ": " << th->cq_errors
                       << " CQ error(s) since the last report";
        }
        if (!th->pending_acks.empty()) {
            NIXL_WARN << "EFA proxy: thread " << th->id << ": " << th->pending_acks.size()
                      << " atomicAdd ack(s) not sent at shutdown";
        }
        releaseThread(*th);
    }
    thread_state_.clear();
    return NIXL_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Requests
 * ------------------------------------------------------------------------- */

nixlLibfabricProxy::Request *
nixlLibfabricProxy::allocRequest(Thread &th) {
    if (th.free_reqs.empty()) {
        return nullptr;
    }
    Request *req = th.free_reqs.back();
    th.free_reqs.pop_back();
    req->in_use = true;
    ++req->generation;
    req->frags_left = 0;
    req->awaiting_ack = false;
    req->status = NIXL_IN_PROG;
    req->fence = nullptr;
    req->epoch = nullptr;
    req->dest = FI_ADDR_UNSPEC;
    req->inject_cq_error = false;
    req->counted = false;
    req->is_atomic = false;
    req->tx = nullptr;
    req->t_post = 0;
    req->t_done = 0;
    return req;
}

void
nixlLibfabricProxy::freeRequest(Thread &th, Request *req) {
    req->in_use = false;
    th.free_reqs.push_back(req);
}

/* ---------------------------------------------------------------------------
 * submit / check_completion
 * ------------------------------------------------------------------------- */

nixlLibfabricProxy::Thread &
nixlLibfabricProxy::ownerOf(uint32_t channel) {
    Thread &th = *thread_state_[channel % threads_];
    checkOwner(th);
    return th;
}

void
nixlLibfabricProxy::checkOwner(Thread &th) {
    // The design rests on ProxyWorker's striping: channel c only ever on the
    // worker c % threads, so each thread state (and CQ) has one caller.
    const std::thread::id self = std::this_thread::get_id();
    if (th.owner == self) {
        return;
    }
    if (th.owner != std::thread::id{}) {
        NIXL_FATAL << "EFA proxy: thread state " << th.id << " used by a second worker thread; "
                   << "the runtime no longer stripes channels as channel % threads";
    }
    th.owner = self;
}

nixl_status_t
nixlLibfabricProxy::submit(const nixlBackendProxySubmission &sub,
                           nixlBackendProxyRequest &request) {
    Thread &th = ownerOf(sub.channel_id);
    const uint32_t t = th.id;
    const uint64_t ring_key =
        static_cast<uint64_t>(sub.channel_id) * config_.max_peers + sub.peer_index;
    RingFence &fence = th.fences[ring_key];
    // A put that fails before joining the fence must still fail the atomicAdd
    // that closes its epoch, or the target would get a signal without the data.
    const auto fail_put = [&](nixl_status_t status) {
        if (sub.opcode == nixl_proxy_opcode_t::PUT) {
            if (receiver_order_) {
                // Never counted, so no atomicAdd sent so far waits for it: only
                // later ones must fail.
                if (fence.error == NIXL_SUCCESS) {
                    fence.error = status;
                }
            } else if (!test_fence_off_) {
                fence.order.complete(fence.order.addPut(), status);
                releaseFence(th, fence);
            }
        }
        return status;
    };

    Request *req = allocRequest(th);
    if (!req) {
        NIXL_ERROR << "EFA proxy: request pool exhausted on thread " << t;
        return fail_put(NIXL_ERR_BACKEND);
    }
    req->fence = &fence;
    if (th.prof) {
        req->t_submit = nixlLibfabricProxyProfile::now();
        req->prof_op = sub.opcode == nixl_proxy_opcode_t::PUT ? nixlLibfabricProxyProfile::PUT :
                                                                nixlLibfabricProxyProfile::ATOMIC;
        req->prof_class = static_cast<uint8_t>(nixlLibfabricProxyProfile::sizeClass(sub.size));
    }

#ifdef HAVE_CUDA
    if (sub.local.mem_type == VRAM_SEG && th.cuda_dev != static_cast<int>(sub.local.desc.devId)) {
        if (cudaSetDevice(static_cast<int>(sub.local.desc.devId)) == cudaSuccess) {
            th.cuda_dev = static_cast<int>(sub.local.desc.devId);
        }
    }
#endif

    nixl_status_t status;
    switch (sub.opcode) {
    case nixl_proxy_opcode_t::PUT:
        status = submitPut(th, sub, req);
        break;
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        status = submitAtomic(th, sub, req);
        break;
    default:
        status = NIXL_ERR_NOT_SUPPORTED;
        break;
    }
    if (status != NIXL_SUCCESS) {
        freeRequest(th, req);
        return fail_put(status);
    }

    request.token = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req));
    request.context = t;
    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricProxy::checkCompletion(const nixlBackendProxyRequest &request) {
    auto *req = reinterpret_cast<Request *>(static_cast<uintptr_t>(request.token));
    if (!req || !req->in_use) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (req->frags_left != 0 || req->status == NIXL_IN_PROG) {
        return NIXL_IN_PROG;
    }
    // Terminal: the runtime never asks again, so the request is returned here.
    const nixl_status_t status = req->status;
    Thread &th = *thread_state_[req->thread];
    checkOwner(th);
    if (th.prof && status == NIXL_SUCCESS && req->t_post != 0) {
        using P = nixlLibfabricProxyProfile;
        const uint64_t now = P::now();
        const auto op = static_cast<P::Op>(req->prof_op);
        th.prof->add(op, req->prof_class, P::SUBMIT_TO_POST, req->t_post - req->t_submit);
        th.prof->add(op, req->prof_class, P::POST_TO_DONE, req->t_done - req->t_post);
        th.prof->add(op, req->prof_class, P::DONE_TO_COLLECT, now - req->t_done);
        th.prof->add(op, req->prof_class, P::RESIDENCE, now - req->t_submit);
    }
    freeRequest(th, req);
    return status;
}

nixl_status_t
nixlLibfabricProxy::submitPut(Thread &th, const nixlBackendProxySubmission &sub, Request *req) {
    auto *local = static_cast<nixlLibfabricPrivateMetadata *>(sub.local.desc.metadataP);
    auto *remote = static_cast<nixlLibfabricPublicMetadata *>(sub.remote.desc.metadataP);
    if (!local || !remote || !remote->conn_) {
        NIXL_ERROR << "EFA proxy: put without libfabric metadata";
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::vector<size_t> &lrails = local->selected_rails_;
    const std::vector<size_t> &reps = remote->remote_selected_endpoints_;
    if (lrails.empty() || reps.empty()) {
        NIXL_ERROR << "EFA proxy: no usable rail for put";
        return NIXL_ERR_INVALID_PARAM;
    }
    PeerAddrs *peer = peerAddrs(th, remote->conn_);
    if (!peer) {
        return NIXL_ERR_BACKEND;
    }

    const size_t size = sub.size;
    if (size == 0) {
        req->status = NIXL_SUCCESS;
        return NIXL_SUCCESS;
    }

    // Puts with remote CQ data go to the data EPs of the target threads that
    // count them (rxThread()).
    TxRing *tx = nullptr;
    if (imm_puts_) {
        const auto &data = remote->conn_->remote_proxy_data_ep_names_;
        if (data.empty()) {
            if (!peer->warned_no_data_eps) {
                peer->warned_no_data_eps = true;
                NIXL_ERROR << "EFA proxy: peer " << remote->conn_->remoteAgent_
                           << " does not accept puts with remote CQ data (receiver ordering)";
            }
            return NIXL_ERR_NOT_SUPPORTED;
        }
        tx = txRing(
            th, *req->fence, remote->conn_, sub.channel_id * config_.max_peers + sub.peer_index);
    }

    // Same rule as the host path: stripe at or above the threshold over >1 rail.
    // Smaller puts go to one rail each (unstripedRail()).
    const bool stripe = size >= engine_.striping_threshold_ && lrails.size() > 1;
    const size_t nfrag = stripe ? std::min(lrails.size(), kMaxStripes) : 1;
    const size_t chunk = size / nfrag;
    const size_t first = stripe ? 0 : unstripedRail(th, *req->fence, lrails.size());

    // Resolve every destination before the request joins the fence.
    std::array<PendingPost, kMaxStripes> posts{};
    for (size_t i = 0; i < nfrag; ++i) {
        const size_t sel = first + i;
        const size_t rail = lrails[sel];
        const size_t rep = reps[sel % reps.size()];
        const size_t off = stripe ? i * chunk : 0;
        const size_t len = stripe ? (i + 1 == nfrag ? size - off : chunk) : size;
        const uint64_t target = sub.remote.desc.addr + off;

        PendingPost &pp = posts[i];
        pp.kind = PendingPost::Kind::WRITE;
        pp.rail = static_cast<uint32_t>(rail);
        pp.fctx = &req->frag[i];
        pp.local = reinterpret_cast<void *>(sub.local.desc.addr + off);
        pp.len = len;
        pp.desc = fi_mr_desc(local->rail_mr_list_[rail]);
        pp.dest = tx ?
            dataAddr(th, *peer, *remote->conn_, rail, rxThread(*remote->conn_, tx->key, rep), rep) :
            railAddr(th, *peer, *remote->conn_, rail, rep);
        pp.raddr = th.rails[rail].virt_addr ? target : target - remote->remote_buf_addr_;
        pp.rkey = remote->rail_remote_key_list_[rep];
        pp.has_imm = tx != nullptr;
        pp.imm = tx ? wire::ringImm(tx->key, tx->atomics) : 0;
        pp.rx_index = tx ? tx->atomics : 0;
        if (pp.dest == FI_ADDR_UNSPEC) {
            return NIXL_ERR_BACKEND;
        }
    }

    if (inject_ && (inject_->post_error_at != 0 || inject_->cq_error_at != 0)) {
        const uint64_t n = ++inject_->puts;
        posts[0].inject_error = n == inject_->post_error_at;
        req->inject_cq_error = n == inject_->cq_error_at;
    }

    req->frags_left = static_cast<uint16_t>(nfrag);
    if (tx) {
        // Counted when sent: the next atomicAdd waits for nfrag more in its slot.
        req->tx = tx;
        req->rx_seq = tx->atomics;
        tx->slot_puts[tx->atomics % wire::kRingEpochs] += nfrag;
        ++tx->inflight;
    }
    if (receiver_order_ || test_fence_off_) {
        req->counted = true;
        ++req->fence->outstanding;
    } else {
        req->epoch = req->fence->order.addPut();
    }
    for (size_t i = 0; i < nfrag; ++i) {
        post(th, std::move(posts[i]));
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricProxy::submitAtomic(Thread &th, const nixlBackendProxySubmission &sub, Request *req) {
    auto *remote = static_cast<nixlLibfabricPublicMetadata *>(sub.remote.desc.metadataP);
    if (!remote || !remote->conn_) {
        NIXL_ERROR << "EFA proxy: atomicAdd without libfabric metadata";
        return NIXL_ERR_INVALID_PARAM;
    }
    PeerAddrs *peer = peerAddrs(th, remote->conn_);
    if (!peer) {
        return NIXL_ERR_BACKEND;
    }
    if (peer->home.empty()) {
        NIXL_ERROR << "EFA proxy: peer " << remote->conn_->remoteAgent_
                   << " did not publish proxy endpoints; atomicAdd unavailable";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // All senders must agree on one owner thread per counter at the target.
    const uint32_t owner =
        wire::counterOwner(sub.remote.desc.addr, static_cast<uint32_t>(peer->home.size()));
    req->dest = homeAddr(th, *peer, *remote->conn_, owner);
    if (req->dest == FI_ADDR_UNSPEC) {
        return NIXL_ERR_BACKEND;
    }
    NIXL_DEBUG << "EFA proxy: atomicAdd " << sub.value << " to " << std::hex << sub.remote.desc.addr
               << std::dec << " -> " << remote->conn_->remoteAgent_ << " proxy thread " << owner
               << " (fi_addr " << req->dest << ")";
    req->msg = wire::atomicAddMsg{};
    req->msg.hdr = wire::msgHeader{wire::kVersion, wire::msgType::ATOMIC_ADD, 0};
    req->msg.remote_addr = sub.remote.desc.addr;
    req->msg.value = sub.value;
    req->msg.token = req->token();
    req->msg.reply_name_len = static_cast<uint32_t>(th.home_name_len);
    std::memcpy(req->msg.reply_name, th.home_name.data(), th.home_name_len);
    req->msg.order = wire::orderMode::NONE;
    req->is_atomic = true;
    // Completes on its send and on the owner's ack (the add applied, or why not).
    req->frags_left = 2;

    if (receiver_order_) {
        // Sent at once; the owner applies it after the ring's earlier puts and
        // atomicAdds reached the target.
        RingFence &fence = *req->fence;
        TxRing *tx =
            txRing(th, fence, remote->conn_, sub.channel_id * config_.max_peers + sub.peer_index);
        const nixl_status_t error = fence.error != NIXL_SUCCESS ? fence.error : tx->error;
        if (error != NIXL_SUCCESS) {
            req->status = error; // never signal after a failure on the ring
            req->frags_left = 0;
            return NIXL_SUCCESS;
        }
        req->msg.order = wire::orderMode::RING;
        req->msg.ring = tx->key;
        req->msg.seq = tx->atomics++;
        req->msg.expected_puts = tx->slot_puts[req->msg.seq % wire::kRingEpochs];
        req->tx = tx;
        req->rx_seq = req->msg.seq;
        if (rx_trace_ && req->rx_seq < 8) {
            NIXL_INFO << "RXTRACE t=" << nowNs() << " atomic key=" << std::hex << tx->key
                      << std::dec << " seq=" << req->rx_seq
                      << " expected=" << req->msg.expected_puts << " addr=" << std::hex
                      << req->msg.remote_addr << std::dec;
        }
        req->counted = true;
        ++fence.outstanding;
        ++tx->inflight;
        sendAtomic(th, req);
        return NIXL_SUCCESS;
    }
    if (test_fence_off_) {
        req->counted = true;
        ++req->fence->outstanding;
        sendAtomic(th, req);
        return NIXL_SUCCESS;
    }

    // Close the ring's open epoch with this atomic; later puts join a new one.
    req->fence->order.addAtomic(req);
    releaseFence(th, *req->fence);
    return NIXL_SUCCESS;
}

size_t
nixlLibfabricProxy::unstripedRail(const Thread &th, RingFence &fence, size_t nrails) const {
    if (!rail_per_thread_) {
        return fence.next_rail++ % nrails; // every ring rotates over all rails
    }
    // The buffer's rails are split among the proxy threads, and each thread
    // rotates over its share: a rail's domain (FI_THREAD_SAFE, one lock) and CQ
    // are then shared by ceil(threads / rails) threads instead of all of them.
    if (threads_ >= nrails) {
        return th.id % nrails;
    }
    const size_t share =
        (nrails - th.id + threads_ - 1) / threads_; // i < nrails, i % threads_ == id
    return th.id + (fence.next_rail++ % share) * threads_;
}

/* ---------------------------------------------------------------------------
 * Fence
 * ------------------------------------------------------------------------- */

void
nixlLibfabricProxy::releaseFence(Thread &th, RingFence &fence) {
    fence.order.release(
        [&](Request *atomic, nixlLibfabricRingFence<Request>::Epoch *epoch) {
            // The next atomic on this ring waits for this one too (strict order).
            atomic->epoch = epoch;
            sendAtomic(th, atomic);
        },
        [](Request *atomic, nixl_status_t error) {
            // An earlier op on this ring failed: never signal over bad data.
            atomic->status = error;
            atomic->frags_left = 0;
        });
}

void
nixlLibfabricProxy::sendAtomic(Thread &th, Request *req) {
    req->awaiting_ack = true;
    req->ack_deadline = nowNs() + kAckTimeoutNs;
    th.awaiting.push_back(req);

    PendingPost pp{};
    pp.kind = PendingPost::Kind::SEND;
    pp.rail = th.home;
    pp.fctx = &req->frag[0];
    pp.local = &req->msg;
    pp.len = sizeof(wire::atomicAddMsg);
    pp.desc = th.req_desc;
    pp.dest = req->dest;
    post(th, std::move(pp));
}

/* ---------------------------------------------------------------------------
 * Posting and completions
 * ------------------------------------------------------------------------- */

ssize_t
nixlLibfabricProxy::tryPost(Thread &th, const PendingPost &pp) {
    if (inject_) {
        if (inject_->eagain_every != 0 && ++inject_->attempts % inject_->eagain_every == 0) {
            return -FI_EAGAIN;
        }
        if (pp.inject_error) {
            NIXL_WARN << "EFA proxy: injected post error";
            return -FI_EIO;
        }
    }
    struct iovec iov = {pp.local, pp.len};
    void *desc = pp.desc;
    if (pp.kind == PendingPost::Kind::WRITE) {
        // The fence releases a ring's atomicAdd once its puts complete, which is
        // only safe if completion means the data is placed at the target.
        const uint64_t flags =
            FI_COMPLETION | FI_DELIVERY_COMPLETE | (pp.has_imm ? FI_REMOTE_CQ_DATA : 0);
        struct fi_rma_iov rma = {pp.raddr, pp.len, pp.rkey};
        struct fi_msg_rma msg = {};
        msg.msg_iov = &iov;
        msg.desc = &desc;
        msg.iov_count = 1;
        msg.addr = pp.dest;
        msg.rma_iov = &rma;
        msg.rma_iov_count = 1;
        msg.context = &pp.fctx->op.ctx;
        msg.data = pp.imm;
        return fi_writemsg(th.rails[pp.rail].ep, &msg, flags);
    }
    // An atomicAdd record, on the control EP: the owner's ack, not the send
    // completion, finishes it.
    struct fi_msg msg = {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = pp.dest;
    msg.context = &pp.fctx->op.ctx;
    return fi_sendmsg(th.ctl_ep, &msg, FI_COMPLETION);
}

ssize_t
nixlLibfabricProxy::postTimed(Thread &th, const PendingPost &pp) {
    if (rx_trace_ && pp.has_imm && pp.rx_index < 8) {
        const ssize_t rc = tryPost(th, pp);
        NIXL_INFO << "RXTRACE t=" << nowNs() << " post key=" << std::hex << pp.imm << std::dec
                  << " frag=" << pp.rx_index << " rail=" << pp.rail << " len=" << pp.len
                  << " rc=" << rc << " thread=" << th.id;
        return rc;
    }
    if (!th.prof) {
        return tryPost(th, pp);
    }
    const uint64_t start = nixlLibfabricProxyProfile::now();
    const ssize_t rc = tryPost(th, pp);
    const uint64_t end = nixlLibfabricProxyProfile::now();
    th.prof->add(nixlLibfabricProxyProfile::POST_CALL, end - start);
    RailRes &rr = th.rails[pp.rail];
    if (end - start > kSlowCallNs) {
        NIXL_INFO << "EFA proxy profile: slow "
                  << (pp.kind == PendingPost::Kind::WRITE ? "write" : "send") << " post: thread "
                  << th.id << " rail " << pp.rail << " len " << pp.len << " post #" << rr.posts
                  << " took " << (end - start) / 1000 << " us";
    }
    ++rr.posts;
    if (rc == 0) {
        pp.fctx->owner->t_post = end;
    } else if (rc == -FI_EAGAIN) {
        th.prof->add(nixlLibfabricProxyProfile::POST_EAGAIN, end - start);
    }
    return rc;
}

void
nixlLibfabricProxy::post(Thread &th, PendingPost &&pp) {
    if (pp.kind == PendingPost::Kind::WRITE && !th.polled[pp.rail]) {
        th.polled[pp.rail] = true;
        th.poll_rails.push_back(pp.rail);
    }
    std::deque<PendingPost> &queue =
        th.retry[pp.kind == PendingPost::Kind::SEND ? th.retry.size() - 1 : pp.rail];
    ssize_t rc = -FI_EAGAIN;
    if (queue.empty()) { // otherwise stay behind this rail's back-pressured posts
        rc = postTimed(th, pp);
    }
    if (rc == -FI_EAGAIN) {
        if (th.prof) {
            pp.queued_at = nixlLibfabricProxyProfile::now();
        }
        queue.push_back(pp);
        ++th.retry_count;
    } else if (rc) {
        NIXL_ERROR << "EFA proxy: post failed on rail " << pp.rail << ": " << fi_strerror(-rc);
        failFragment(th, pp.fctx->owner, NIXL_ERR_BACKEND);
    }
}

void
nixlLibfabricProxy::drainRetries(Thread &th) {
    for (size_t r = 0; th.retry_count != 0 && r < th.retry.size(); ++r) {
        std::deque<PendingPost> &queue = th.retry[r];
        while (!queue.empty()) {
            const ssize_t rc = postTimed(th, queue.front());
            if (rc == -FI_EAGAIN) {
                break; // this rail is still back-pressured; the others go on
            }
            const PendingPost pp = queue.front();
            queue.pop_front();
            --th.retry_count;
            if (th.prof && rc == 0) {
                th.prof->add(nixlLibfabricProxyProfile::RETRY_WAIT,
                             nixlLibfabricProxyProfile::now() - pp.queued_at);
            }
            if (rc) {
                NIXL_ERROR << "EFA proxy: post failed on rail " << pp.rail << ": "
                           << fi_strerror(-rc);
                failFragment(th, pp.fctx->owner, NIXL_ERR_BACKEND);
            }
        }
    }
    while (!th.pending_acks.empty() && postAck(th, th.pending_acks.front())) {
        th.pending_acks.pop_front();
    }
    while (!th.recv_retry.empty()) {
        RecvBuf *buf = th.recv_retry.back();
        th.recv_retry.pop_back();
        if (postRecv(th, buf) != NIXL_SUCCESS) {
            break; // postRecv re-queued it
        }
    }
}

void
nixlLibfabricProxy::completeFragment(Thread &th, Request *req, nixl_status_t status) {
    if (status != NIXL_SUCCESS && req->status == NIXL_IN_PROG) {
        req->status = status;
    }
    if (req->frags_left == 0 || --req->frags_left != 0) {
        return;
    }
    if (req->status == NIXL_IN_PROG) {
        req->status = NIXL_SUCCESS;
    }
    if (th.prof) {
        req->t_done = nixlLibfabricProxyProfile::now();
    }
    if (req->counted) {
        req->counted = false;
        --req->fence->outstanding;
        if (req->tx) {
            --req->tx->inflight;
        }
        if (req->status != NIXL_SUCCESS && receiver_order_) {
            failRing(th, req, req->status);
        }
        return;
    }
    if (req->tx) {
        --req->tx->inflight; // test-only puts with remote CQ data under the fence
    }
    if (RingFence *fence = req->fence) {
        fence->order.complete(req->epoch, req->status);
        req->epoch = nullptr;
        releaseFence(th, *fence);
    }
}

void
nixlLibfabricProxy::failFragment(Thread &th, Request *req, nixl_status_t status) {
    if (req->awaiting_ack) {
        // The record was not delivered (or its target is gone): no ack will come.
        req->awaiting_ack = false;
        completeFragment(th, req, status);
    }
    completeFragment(th, req, status);
}

void
nixlLibfabricProxy::expireAcks(Thread &th, uint64_t now_ns) {
    for (size_t i = 0; i < th.awaiting.size();) {
        Request *req = th.awaiting[i];
        if (!req->awaiting_ack) {
            th.awaiting[i] = th.awaiting.back(); // acked or failed since: drop the entry
            th.awaiting.pop_back();
            continue;
        }
        if (now_ns >= req->ack_deadline) {
            NIXL_ERROR << "EFA proxy: no ack for an atomicAdd to " << std::hex
                       << req->msg.remote_addr << std::dec << " after "
                       << kAckTimeoutNs / 1000000000 << " s; failing it";
            req->awaiting_ack = false;
            completeFragment(th, req, NIXL_ERR_REMOTE_DISCONNECT);
            continue; // dropped on the next look, now that it is no longer awaiting
        }
        ++i;
    }
}

void
nixlLibfabricProxy::pollCqs(Thread &th) {
    const uint64_t start = th.prof && imm_puts_ ? nowNs() : 0;
    th.sweep_imm = 0;
    // By index: completions release atomics, whose posts may extend the list.
    for (size_t i = 0; i < th.poll_rails.size(); ++i) {
        const uint32_t r = th.poll_rails[i];
        pollCq(th, th.rails[r].cq, r, th.rails[r].cq_reads);
    }
    pollCq(th, th.ctl_cq, static_cast<uint32_t>(rails_), th.ctl_cq_reads);
    th.rx_active = th.sweep_imm != 0;
    if (start != 0 && th.sweep_imm != 0) {
        ++th.rx_stats.sweeps;
        th.rx_stats.sweep_ns += nowNs() - start;
    }
}

void
nixlLibfabricProxy::pollCq(Thread &th, struct fid_cq *cq, uint32_t rail, uint64_t &reads) {
    struct fi_cq_data_entry entries[kCqBatch];
    {
        const uint32_t r = rail; // rails_ names the control CQ
        for (;;) {
            const uint64_t read_start = th.prof ? nixlLibfabricProxyProfile::now() : 0;
            const ssize_t n = fi_cq_read(cq, entries, kCqBatch);
            if (th.prof) {
                const uint64_t took = nixlLibfabricProxyProfile::now() - read_start;
                if (took > kSlowCallNs) {
                    NIXL_INFO << "EFA proxy profile: slow cq read: thread " << th.id << " rail "
                              << r << " read #" << reads << " took " << took / 1000 << " us";
                }
                ++reads;
            }
            if (n > 0) {
                const uint64_t decode_start = th.prof && imm_puts_ ? nowNs() : 0;
                uint64_t imm = 0;
                th.events += static_cast<uint64_t>(n);
                for (ssize_t i = 0; i < n; ++i) {
                    if (entries[i].flags & FI_REMOTE_CQ_DATA) {
                        // A put with remote CQ data landed: count it for its ring.
                        ++imm;
                        handleImm(th, static_cast<uint32_t>(entries[i].data));
                        if (entries[i].op_context != nullptr) {
                            // It consumed a posted receive (efa_proxy_test_data_rx_post).
                            ++th.rx_stats.consumed_rx;
                            if (test_data_rx_post_ != 0) {
                                postRecv(th, static_cast<RecvBuf *>(entries[i].op_context));
                            }
                        }
                        continue;
                    }
                    // Each context type starts with its OpCtx.
                    auto *op = static_cast<OpCtx *>(entries[i].op_context);
                    switch (op->kind) {
                    case OpCtx::Kind::RECV:
                        handleRecv(th, reinterpret_cast<RecvBuf *>(op), entries[i].len);
                        break;
                    case OpCtx::Kind::ACK:
                        th.free_acks.push_back(reinterpret_cast<AckBuf *>(op));
                        break;
                    case OpCtx::Kind::FRAG: {
                        auto *f = reinterpret_cast<Request::FragCtx *>(op);
                        nixl_status_t status = NIXL_SUCCESS;
                        if (rx_trace_ && f->owner->tx && !f->owner->is_atomic &&
                            f->owner->rx_seq < 8) {
                            NIXL_INFO << "RXTRACE t=" << nowNs() << " done key=" << std::hex
                                      << f->owner->tx->key << std::dec
                                      << " epoch=" << f->owner->rx_seq
                                      << " frag=" << (f - f->owner->frag) << " rail=" << rail;
                        }
                        if (f->owner->inject_cq_error) {
                            f->owner->inject_cq_error = false;
                            NIXL_WARN << "EFA proxy: injected completion error";
                            status = NIXL_ERR_BACKEND;
                        }
                        if (status == NIXL_SUCCESS) {
                            completeFragment(th, f->owner, status);
                        } else {
                            failFragment(th, f->owner, status);
                        }
                        break;
                    }
                    }
                }
                if (imm != 0) {
                    th.sweep_imm += imm;
                    if (decode_start != 0) {
                        auto &st = th.rx_stats;
                        const uint64_t end = nowNs();
                        st.entries += imm;
                        ++st.reads;
                        st.read_ns += decode_start - read_start;
                        st.decode_ns += end - decode_start;
                        st.first = st.first != 0 ? st.first : decode_start;
                        st.last = end;
                    }
                }
                if (static_cast<size_t>(n) < kCqBatch) {
                    break;
                }
                continue;
            }
            if (n == -FI_EAVAIL) {
                struct fi_cq_err_entry err = {};
                if (fi_cq_readerr(cq, &err, 0) > 0) {
                    reportCqError(th, r, err.err);
                    auto *op = static_cast<OpCtx *>(err.op_context);
                    if (op == nullptr) {
                        continue;
                    }
                    switch (op->kind) {
                    case OpCtx::Kind::RECV:
                        postRecv(th, reinterpret_cast<RecvBuf *>(op));
                        break;
                    case OpCtx::Kind::ACK:
                        // This ack is lost: the sender's atomicAdd fails by timeout.
                        th.free_acks.push_back(reinterpret_cast<AckBuf *>(op));
                        break;
                    case OpCtx::Kind::FRAG:
                        failFragment(
                            th, reinterpret_cast<Request::FragCtx *>(op)->owner, NIXL_ERR_BACKEND);
                        break;
                    }
                }
                continue;
            }
            break; // -FI_EAGAIN or another error: nothing more this pass
        }
    }
}

void
nixlLibfabricProxy::reportCqError(Thread &th, uint32_t rail, int err) {
    // A dead peer fails every operation in flight to it: report the first
    // error, then at most one summary per interval.
    ++th.cq_errors;
    const uint64_t now = nixlLibfabricProxyProfile::now();
    if (th.cq_error_report != 0 && now - th.cq_error_report < kCqErrorReportNs) {
        return;
    }
    NIXL_ERROR << "EFA proxy: thread " << th.id << " rail " << rail << ": " << th.cq_errors
               << " CQ error(s)" << (th.cq_error_report != 0 ? " since the last report" : "")
               << ", last: " << fi_strerror(err);
    th.cq_errors = 0;
    th.cq_error_report = now;
}

/* ---------------------------------------------------------------------------
 * progress / quiesce
 * ------------------------------------------------------------------------- */

nixl_status_t
nixlLibfabricProxy::progress(uint32_t channel, uint32_t peer) {
    // The runtime calls this once per owned ring and pass; ring (t, 0) polls
    // thread t's CQs, at most once per pass.
    const uint32_t t = channel % threads_;
    if (channel != t || peer != 0) {
        return NIXL_SUCCESS;
    }
    Thread &th = ownerOf(channel);
    const uint64_t pass_start = th.prof ? nowNs() : 0;
    const uint64_t events = th.events;
    // Every CQ read takes its rail domain's lock (FI_THREAD_SAFE) and progresses
    // the whole domain, so a thread spinning on idle CQs starves the engine's host
    // transfers on the same rails. With nothing of its own outstanding, a thread
    // polls at most every idle_poll_ns_: only an incoming atomicAdd can be waiting.
    const bool busy = th.free_reqs.size() != th.reqs.size() || th.retry_count != 0 ||
        !th.pending_acks.empty() || !th.recv_retry.empty() || !th.deferred.empty() || th.rx_active;
    bool poll = true;
    if (!busy && idle_poll_ns_ != 0) {
        const uint64_t now = nowNs();
        poll = now >= th.next_idle_poll;
        if (poll) {
            th.next_idle_poll = now + idle_poll_ns_;
        }
    }
    if (poll) {
        uint64_t poll_start = 0;
        if (th.prof) {
            poll_start = nixlLibfabricProxyProfile::now();
            if (th.last_pass != 0) {
                th.prof->add(nixlLibfabricProxyProfile::PASS_PERIOD, poll_start - th.last_pass);
            }
            th.last_pass = poll_start;
        }
        drainRetries(th);
        pollCqs(th);
        if (th.prof) {
            th.prof->add(nixlLibfabricProxyProfile::CQ_POLL,
                         nixlLibfabricProxyProfile::now() - poll_start);
        }
        if (!th.awaiting.empty() && ++th.ack_scan % kAckScanInterval == 0) {
            expireAcks(th, nowNs());
        }
    }
    if (!th.deferred.empty()) {
        drainDeferred(th);
    }
    if (th.prof && th.events != events) {
        const uint64_t end = nowNs();
        th.busy_ns += end - pass_start;
        th.busy_first = th.busy_first != 0 ? th.busy_first : pass_start;
        th.busy_last = end;
    }

    // Without the engine progress thread, nothing may progress the engine's
    // rails in a device-only application: peers could not finish connecting to
    // us, and pending control sends would block endpoint close at teardown.
    if (t == 0 && !engine_.progress_thread_enabled_ && ++th.passes % kEngineProgressInterval == 0) {
        static_cast<void>(engine_.rail_manager_.progressActiveRails());
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricProxy::quiesce(uint32_t channel, uint32_t peer) {
    // Called after every owned ring has drained and published, so the ring's
    // fence must already be idle; wait briefly for any trailing completion.
    Thread &th = ownerOf(channel);
    const uint64_t key = static_cast<uint64_t>(channel) * config_.max_peers + peer;
    auto it = th.fences.find(key);
    if (it == th.fences.end()) {
        return NIXL_SUCCESS;
    }
    const auto deadline = std::chrono::steady_clock::now() + kQuiesceTimeout;
    while (!it->second.idle() && std::chrono::steady_clock::now() < deadline) {
        drainRetries(th);
        pollCqs(th);
        expireAcks(th, nowNs());
        if (!th.deferred.empty()) {
            drainDeferred(th);
        }
    }
    if (!it->second.idle()) {
        NIXL_ERROR << "EFA proxy: ring (" << channel << ", " << peer << ") did not quiesce in "
                   << kQuiesceTimeout.count() << " s";
        return NIXL_ERR_BACKEND;
    }
    th.fences.erase(it);
    return NIXL_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Home endpoints: atomicAdd records and their acks (libfabric_proxy_wire.h)
 * ------------------------------------------------------------------------- */

nixl_status_t
nixlLibfabricProxy::postRecv(Thread &th, RecvBuf *buf) {
    const bool ctl = buf->rail == kCtlRail;
    struct iovec iov = {&buf->msg, sizeof(wire::anyMsg)};
    void *desc = ctl ? th.recv_desc : th.rails[buf->rail].rx_desc;
    struct fi_msg msg = {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = FI_ADDR_UNSPEC;
    msg.context = &buf->op.ctx;
    const ssize_t rc = fi_recvmsg(ctl ? th.ctl_ep : th.rails[buf->rail].ep, &msg, 0);
    if (rc == 0) {
        return NIXL_SUCCESS;
    }
    th.recv_retry.push_back(buf); // retried from progress()
    if (rc != -FI_EAGAIN) {
        NIXL_ERROR << "EFA proxy: fi_recvmsg failed: " << fi_strerror(-rc);
    }
    return rc == -FI_EAGAIN ? NIXL_IN_PROG : NIXL_ERR_BACKEND;
}

void
nixlLibfabricProxy::handleRecv(Thread &th, RecvBuf *buf, size_t len) {
    const wire::anyMsg &msg = buf->msg;
    if (len < sizeof(wire::msgHeader) || msg.hdr.version != wire::kVersion) {
        NIXL_ERROR << "EFA proxy: dropped a " << len << "-byte message of protocol version "
                   << (len >= sizeof(wire::msgHeader) ? msg.hdr.version : 0) << " (expected "
                   << wire::kVersion << ")";
    } else if (msg.hdr.type == wire::msgType::ATOMIC_ADD && len >= sizeof(wire::atomicAddMsg)) {
        handleAtomic(th, msg.add);
    } else if (msg.hdr.type == wire::msgType::ATOMIC_ACK && len >= sizeof(wire::atomicAckMsg)) {
        handleAck(th, msg.ack);
    } else {
        NIXL_ERROR << "EFA proxy: dropped a malformed " << len << "-byte message of type "
                   << static_cast<int>(msg.hdr.type);
    }
    postRecv(th, buf);
}

void
nixlLibfabricProxy::handleAtomic(Thread &th, const wire::atomicAddMsg &msg) {
    const fi_addr_t reply = replyAddr(th, msg);
    if (reply == FI_ADDR_UNSPEC) {
        return; // logged; the sender fails the atomicAdd when its ack times out
    }
    if (msg.order == wire::orderMode::NONE) {
        applyAndAck(th, msg, reply);
        return;
    }
    if (msg.order != wire::orderMode::RING || !imm_puts_) {
        NIXL_ERROR << "EFA proxy: atomicAdd with ordering " << static_cast<int>(msg.order)
                   << ", which this proxy does not accept (efa_proxy_ordering=receiver)";
        sendAck(th, PendingAck{reply, msg.token, NIXL_ERR_NOT_SUPPORTED});
        return;
    }
    RxRing *ring = rxRing(th, msg.ring);
    {
        // Ring keys are random per sender ring: tell a collision apart from a
        // ring's later atomicAdds by the sender's reply EP.
        const std::string_view name(reinterpret_cast<const char *>(msg.reply_name),
                                    std::min<size_t>(msg.reply_name_len, wire::kMaxEpName));
        std::lock_guard<std::mutex> lock(ring->sender_mutex);
        if (ring->sender.empty()) {
            ring->sender.assign(name);
        } else if (ring->sender != name) {
            NIXL_ERROR << "EFA proxy: two senders use ring key " << std::hex << msg.ring
                       << "; failing the atomicAdd";
            sendAck(th, PendingAck{reply, msg.token, NIXL_ERR_MISMATCH});
            return;
        }
    }
    th.deferred.push_back(Deferred{ring, msg, reply, nowNs()});
    drainDeferred(th);
    for (const Deferred &d : th.deferred) {
        if (d.ring == ring && d.msg.token == msg.token) {
            ++th.rx_stats.deferred; // still waiting for its ring
            break;
        }
    }
}

void
nixlLibfabricProxy::applyAndAck(Thread &th, const wire::atomicAddMsg &msg, fi_addr_t reply) {
    const uint64_t start = th.prof ? nixlLibfabricProxyProfile::now() : 0;
    const nixl_status_t status = applyAtomic(th, msg.remote_addr, msg.value);
    if (th.prof) {
        th.prof->add(nixlLibfabricProxyProfile::ATOMIC_APPLY,
                     nixlLibfabricProxyProfile::now() - start);
    }
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "EFA proxy: atomicAdd to " << std::hex << msg.remote_addr << std::dec
                   << " failed with status " << status << "; reporting it to the sender";
    }
    sendAck(th, PendingAck{reply, msg.token, status});
}

void
nixlLibfabricProxy::handleAck(Thread &th, const wire::atomicAckMsg &ack) {
    const auto index = static_cast<uint32_t>(ack.token);
    const auto generation = static_cast<uint32_t>(ack.token >> 32);
    if (index >= th.reqs.size()) {
        NIXL_ERROR << "EFA proxy: dropped an ack for unknown request " << ack.token;
        return;
    }
    Request &req = th.reqs[index];
    if (!req.in_use || req.generation != generation || !req.awaiting_ack) {
        NIXL_DEBUG << "EFA proxy: ignoring a late ack for request " << ack.token;
        return;
    }
    req.awaiting_ack = false;
    completeFragment(th, &req, static_cast<nixl_status_t>(ack.status));
}

fi_addr_t
nixlLibfabricProxy::replyAddr(Thread &th, const wire::atomicAddMsg &msg) {
    if (msg.reply_name_len == 0 || msg.reply_name_len > wire::kMaxEpName) {
        NIXL_ERROR << "EFA proxy: atomicAdd without a valid reply address";
        return FI_ADDR_UNSPEC;
    }
    const std::string_view name(reinterpret_cast<const char *>(msg.reply_name), msg.reply_name_len);
    if (auto it = th.reply_addrs.find(name); it != th.reply_addrs.end()) {
        return it->second;
    }
    fi_addr_t addr = FI_ADDR_UNSPEC;
    if (fi_av_insert(th.ctl_av, msg.reply_name, 1, &addr, 0, nullptr) != 1) {
        NIXL_ERROR << "EFA proxy: fi_av_insert failed for an atomicAdd sender";
        return FI_ADDR_UNSPEC;
    }
    th.reply_addrs.emplace(std::string(name), addr);
    return addr;
}

void
nixlLibfabricProxy::sendAck(Thread &th, const PendingAck &ack) {
    if (!th.pending_acks.empty() || !postAck(th, ack)) {
        th.pending_acks.push_back(ack); // keep acks in order; drained from progress()
    }
}

bool
nixlLibfabricProxy::postAck(Thread &th, const PendingAck &ack) {
    if (th.free_acks.empty()) {
        return false;
    }
    AckBuf *buf = th.free_acks.back();
    buf->msg = wire::atomicAckMsg{};
    buf->msg.hdr = wire::msgHeader{wire::kVersion, wire::msgType::ATOMIC_ACK, 0};
    buf->msg.token = ack.token;
    buf->msg.status = static_cast<int32_t>(ack.status);

    struct iovec iov = {&buf->msg, sizeof(buf->msg)};
    void *desc = th.ack_desc;
    struct fi_msg msg = {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = ack.dest;
    msg.context = &buf->op.ctx;
    const ssize_t rc = fi_sendmsg(th.ctl_ep, &msg, FI_COMPLETION);
    if (rc == -FI_EAGAIN) {
        return false;
    }
    if (rc == 0) {
        th.free_acks.pop_back();
    } else {
        NIXL_ERROR << "EFA proxy: ack send failed: " << fi_strerror(-rc)
                   << "; the sender's atomicAdd fails when its ack times out";
    }
    return true;
}

const nixlLibfabricProxy::Region *
nixlLibfabricProxy::findRegion(uint64_t addr) const {
    // Registrations may overlap, so an earlier, larger one can still hold addr.
    for (auto it = regions_.upper_bound(addr); it != regions_.begin();) {
        --it;
        if (addr + sizeof(uint64_t) <= it->first + it->second.len) {
            return &it->second;
        }
    }
    return nullptr;
}

nixl_status_t
nixlLibfabricProxy::flushRdmaWrites(Thread &th, uint64_t addr) {
#ifdef HAVE_CUDA
    int device = th.cuda_dev >= 0 ? th.cuda_dev : 0;
    {
        std::lock_guard<std::mutex> lock(th.regions_lock);
        const Region *region = findRegion(addr - addr % sizeof(uint64_t));
        if (region == nullptr || !region->is_vram) {
            return NIXL_SUCCESS; // nothing to order on a GPU (or the add fails anyway)
        }
        device = region->device_id;
    }
    const uint64_t start = th.prof ? nowNs() : 0;
    if (th.cuda_dev != device && cudaSetDevice(device) == cudaSuccess) {
        th.cuda_dev = device;
    }
    if (cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                           cudaFlushGPUDirectRDMAWritesToOwner) != cudaSuccess) {
        NIXL_ERROR << "EFA proxy: cudaDeviceFlushGPUDirectRDMAWrites failed on device " << device;
        return NIXL_ERR_BACKEND;
    }
    if (th.prof) {
        th.prof->add(nixlLibfabricProxyProfile::RX_FLUSH, nowNs() - start);
    }
#else
    static_cast<void>(th);
    static_cast<void>(addr);
#endif
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricProxy::applyAtomic(Thread &th, uint64_t addr, uint64_t value) {
    if (addr % sizeof(uint64_t) != 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // Held across the add: a deregistration waits for it before unmapping.
    std::lock_guard<std::mutex> lock(th.regions_lock);
    const Region *region = findRegion(addr);
    if (region == nullptr) {
        return NIXL_ERR_NOT_FOUND;
    }

    NIXL_DEBUG << "EFA proxy: applying atomicAdd " << value << " at " << std::hex << addr
               << std::dec << (region->is_vram ? " (VRAM)" : " (DRAM)");
    if (!region->is_vram) {
        // Every add to this counter arrives on this thread: a plain RMW is safe
        // against other proxy threads; the atomic also covers concurrent CPU users.
        __atomic_fetch_add(reinterpret_cast<uint64_t *>(addr), value, __ATOMIC_SEQ_CST);
        return NIXL_SUCCESS;
    }
    // A read-modify-write through the BAR or a copy, not an atomic: the GPU must
    // not write the counter while remote adds to it can arrive.
    if (counters_->usable()) {
        const nixl_status_t status = counters_->add(addr, value);
        if (status != NIXL_ERR_NOT_SUPPORTED) {
            return status;
        }
    }
#ifdef HAVE_CUDA
    return addWithCuda(th, region->device_id, addr, value);
#else
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

#ifdef HAVE_CUDA
nixl_status_t
nixlLibfabricProxy::addWithCuda(Thread &th, int device_id, uint64_t addr, uint64_t value) {
    if (cudaSetDevice(device_id) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    th.cuda_dev = device_id;
    cudaStream_t &stream = th.streams[device_id];
    if (stream == nullptr &&
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        stream = nullptr;
        return NIXL_ERR_BACKEND;
    }
    uint64_t v = 0;
    void *dev = reinterpret_cast<void *>(addr);
    if (cudaMemcpyAsync(&v, dev, sizeof(v), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    v += value;
    if (cudaMemcpyAsync(dev, &v, sizeof(v), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}
#endif

std::vector<std::unique_lock<std::mutex>>
nixlLibfabricProxy::lockRegions() {
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(thread_state_.size() + 1);
    locks.emplace_back(regions_write_mutex_);
    for (auto &th : thread_state_) {
        locks.emplace_back(th->regions_lock);
    }
    return locks;
}

bool
nixlLibfabricProxy::regionOverlaps(uintptr_t addr, size_t len) const {
    for (const auto &[base, region] : regions_) {
        if (base < addr + len && addr < base + region.len) {
            return true;
        }
    }
    return false;
}

void
nixlLibfabricProxy::onRegister(uintptr_t addr, size_t len, bool is_vram, int device_id) {
    const auto locks = lockRegions();
    regions_.emplace(addr, Region{len, is_vram, device_id});
}

void
nixlLibfabricProxy::onDeregister(uintptr_t addr, size_t len) {
    // Waits for adds being applied; later adds miss the region.
    const auto locks = lockRegions();
    auto [first, last] = regions_.equal_range(addr);
    if (first == last) {
        return;
    }
    auto victim = first;
    while (victim != last && victim->second.len != len) {
        ++victim;
    }
    if (victim == last) {
        NIXL_WARN << "EFA proxy: deregistering " << std::hex << addr << std::dec << " with length "
                  << len << ", registered with " << first->second.len;
        victim = first;
        len = std::max(len, first->second.len);
    }
    regions_.erase(victim);
    // Keep the counter pages another registration still covers.
    counters_->dropRange(
        addr, len, [this](uintptr_t page, size_t size) { return regionOverlaps(page, size); });
}

/* ---------------------------------------------------------------------------
 * Receiver-side ordering (efa_proxy_ordering=receiver)
 * ------------------------------------------------------------------------- */

nixlLibfabricProxy::TxRing *
nixlLibfabricProxy::txRing(Thread &th,
                           RingFence &fence,
                           const std::shared_ptr<nixlLibfabricConnection> &conn,
                           uint32_t ring) {
    if (fence.tx != nullptr && fence.tx_conn == conn.get()) {
        return fence.tx;
    }
    // Counts follow the target's (its incarnation), not this fence: a quiesced
    // ring that comes back continues them. One per ring, so that at most
    // ring_depth of its requests (<= kRingEpochs) are in flight.
    TxRing &tx =
        th.tx_rings[std::make_tuple(conn->remoteAgent_, conn->remote_proxy_incarnation_, ring)];
    if (tx.key == 0 || (tx.error != NIXL_SUCCESS && tx.inflight == 0)) {
        // New, or failed: a put counted here may never reach the target, so start
        // over under a new key, which the target counts from zero.
        std::uniform_int_distribution<uint32_t> id(1, wire::kMaxRingKey);
        uint32_t key;
        do {
            key = id(th.rng);
        } while (key == tx.key);
        tx = TxRing{};
        tx.key = key;
    }
    fence.tx = &tx;
    fence.tx_conn = conn.get();
    return &tx;
}

void
nixlLibfabricProxy::failRing(Thread &th, Request *req, nixl_status_t status) {
    // The fence's rule: the failed operation's ring sends no atomicAdd after it.
    // Those already sent may wait at the target for a put that will never come
    // (or a failed atomicAdd before them): fail them here instead of by timeout.
    if (req->fence != nullptr && req->fence->error == NIXL_SUCCESS) {
        req->fence->error = status;
    }
    TxRing *tx = req->tx;
    if (tx == nullptr) {
        return;
    }
    if (tx->error == NIXL_SUCCESS) {
        tx->error = status;
    }
    // A failed put of epoch e: atomicAdds e and later; a failed atomicAdd s: after s.
    const uint64_t first = req->is_atomic ? req->rx_seq + 1 : req->rx_seq;
    for (size_t i = 0; i < th.awaiting.size(); ++i) {
        Request *other = th.awaiting[i];
        if (!other->awaiting_ack || other->tx != tx) {
            continue;
        }
        if (other->rx_seq >= first) {
            other->awaiting_ack = false;
            completeFragment(th, other, status);
        }
    }
}

nixlLibfabricProxy::RxRing *
nixlLibfabricProxy::rxRing(Thread &th, uint32_t ring_key) {
    if (th.rx_last != nullptr && th.rx_last_key == ring_key) {
        return th.rx_last;
    }
    RxRing *&cached = th.rx_cache[ring_key];
    if (cached == nullptr) {
        std::lock_guard<std::mutex> lock(rx_rings_mutex_);
        std::unique_ptr<RxRing> &ring = rx_rings_[ring_key];
        if (!ring) {
            ring = std::make_unique<RxRing>();
        }
        cached = ring.get();
    }
    th.rx_last_key = ring_key;
    th.rx_last = cached;
    return cached;
}

void
nixlLibfabricProxy::handleImm(Thread &th, uint32_t imm) {
    // The data is in place once its CQ entry is: release it with the count.
    const uint32_t slot = wire::immSlot(imm);
    const uint64_t n =
        rxRing(th, wire::immRing(imm))->received[slot].fetch_add(1, std::memory_order_release);
    if (rx_trace_ && n < 32 && slot < 8) {
        NIXL_INFO << "RXTRACE t=" << nowNs() << " imm key=" << std::hex << wire::immRing(imm)
                  << std::dec << " slot=" << slot << " received=" << n + 1 << " thread=" << th.id;
    }
}

bool
nixlLibfabricProxy::tryApplyDeferred(Thread &th, Deferred &d) {
    RxRing &ring = *d.ring;
    if (!d.flushed || ring.applied.load(std::memory_order_acquire) != d.msg.seq) {
        return false;
    }
    if (th.prof) {
        th.prof->add(nixlLibfabricProxyProfile::RX_ORDER_WAIT, nowNs() - d.arrived);
    }
    auto status = static_cast<nixl_status_t>(ring.error.load(std::memory_order_relaxed));
    if (rx_trace_ && d.msg.seq < 8) {
        NIXL_INFO << "RXTRACE t=" << nowNs() << " apply key=" << std::hex << d.msg.ring << std::dec
                  << " seq=" << d.msg.seq << " expected=" << d.msg.expected_puts
                  << " received=" << ring.received[d.msg.seq % wire::kRingEpochs].load()
                  << " thread=" << th.id;
    }
    if (status == NIXL_SUCCESS) {
        const uint64_t start = th.prof ? nowNs() : 0;
        status = applyAtomic(th, d.msg.remote_addr, d.msg.value);
        if (th.prof) {
            th.prof->add(nixlLibfabricProxyProfile::ATOMIC_APPLY, nowNs() - start);
        }
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "EFA proxy: atomicAdd to " << std::hex << d.msg.remote_addr << std::dec
                       << " failed with status " << status
                       << "; reporting it to the sender, later ones on its ring fail too";
            ring.error.store(status, std::memory_order_relaxed);
        }
    }
    ++th.events;
    // The ring's next atomicAdd may be another thread's: publish the add first.
    ring.applied.store(d.msg.seq + 1, std::memory_order_release);
    sendAck(th, PendingAck{d.reply, d.msg.token, status});
    return true;
}

void
nixlLibfabricProxy::drainDeferred(Thread &th) {
    // Which adds have all their puts in, then one flush for all of them: only puts
    // whose CQ entries were counted before the flush are made visible by it.
    bool flush = false;
    const Deferred *first = nullptr;
    for (Deferred &d : th.deferred) {
        if (!d.data_ready &&
            d.ring->received[d.msg.seq % wire::kRingEpochs].load(std::memory_order_acquire) >=
                d.msg.expected_puts) {
            d.data_ready = true;
        }
        if (d.data_ready && !d.flushed) {
            flush = true;
            first = first ? first : &d;
        }
    }
    if (flush) {
        nixl_status_t status = NIXL_SUCCESS;
        if (rx_flush_) {
            status = flushRdmaWrites(th, first->msg.remote_addr);
        }
        for (Deferred &d : th.deferred) {
            if (d.data_ready && !d.flushed) {
                d.flushed = true;
                if (status != NIXL_SUCCESS &&
                    d.ring->error.load(std::memory_order_relaxed) == NIXL_SUCCESS) {
                    d.ring->error.store(status, std::memory_order_relaxed);
                }
            }
        }
    }

    // Arrival order is not ring order (no message ordering on EFA RDM), so retry
    // all of them until none moves.
    const uint64_t now = nowNs();
    bool progressed = true;
    while (progressed && !th.deferred.empty()) {
        progressed = false;
        for (size_t i = 0; i < th.deferred.size();) {
            Deferred &d = th.deferred[i];
            bool done = tryApplyDeferred(th, d);
            if (!done && now - d.arrived >= kAckTimeoutNs) {
                // The sender has given up on it too.
                NIXL_ERROR << "EFA proxy: receiver-ordered atomicAdd to " << std::hex
                           << d.msg.remote_addr << std::dec << " still waited for its ring after "
                           << kAckTimeoutNs / 1000000000 << " s (received "
                           << d.ring->received[d.msg.seq % wire::kRingEpochs].load() << " of "
                           << d.msg.expected_puts << " puts, applied " << d.ring->applied.load()
                           << " of " << d.msg.seq << " atomicAdds); dropping it";
                sendAck(th, PendingAck{d.reply, d.msg.token, NIXL_ERR_REMOTE_DISCONNECT});
                done = true;
            }
            if (done) {
                th.deferred[i] = th.deferred.back();
                th.deferred.pop_back();
                progressed = true;
            } else {
                ++i;
            }
        }
    }
}

fi_addr_t
nixlLibfabricProxy::dataAddr(Thread &th,
                             PeerAddrs &pa,
                             const nixlLibfabricConnection &conn,
                             size_t rail,
                             size_t remote_thread,
                             size_t remote_rail) {
    const auto &names = conn.remote_proxy_data_ep_names_;
    const size_t remote_rails = names[0].size();
    static const std::array<char, LF_EP_NAME_MAX_LEN> kNoName{};
    if (remote_thread >= names.size() || remote_rail >= remote_rails ||
        names[remote_thread][remote_rail] == kNoName) {
        NIXL_ERROR << "EFA proxy: " << conn.remoteAgent_ << " accepts no remote CQ data on rail "
                   << remote_rail << " (thread " << remote_thread
                   << "); receiver ordering needs buffers on the target GPU's rails";
        return FI_ADDR_UNSPEC;
    }
    if (pa.data_ep.empty()) {
        pa.data_ep.resize(rails_);
    }
    std::vector<fi_addr_t> &slots = pa.data_ep[rail];
    if (slots.empty()) {
        slots.assign(names.size() * remote_rails, FI_ADDR_UNSPEC);
    }
    fi_addr_t &addr = slots[remote_thread * remote_rails + remote_rail];
    if (addr == FI_ADDR_UNSPEC &&
        fi_av_insert(
            th.rails[rail].av, names[remote_thread][remote_rail].data(), 1, &addr, 0, nullptr) !=
            1) {
        addr = FI_ADDR_UNSPEC;
        NIXL_ERROR << "EFA proxy: fi_av_insert failed for " << conn.remoteAgent_ << " proxy thread "
                   << remote_thread << " data EP on rail " << remote_rail;
    }
    return addr;
}

size_t
nixlLibfabricProxy::rxThread(const nixlLibfabricConnection &conn,
                             uint32_t key,
                             size_t remote_rail) {
    const size_t threads = conn.remote_proxy_data_ep_names_.size();
    if (!conn.remote_proxy_rx_by_rail_) {
        return wire::ringThread(key, static_cast<uint32_t>(threads));
    }
    const auto &rails = conn.remote_proxy_data_rails_;
    const auto it = std::find(rails.begin(), rails.end(), remote_rail);
    // A rail it does not accept on: dataAddr() reports it.
    return it == rails.end() ? 0 : static_cast<size_t>(it - rails.begin()) % threads;
}

void
nixlLibfabricProxy::reportRx() const {
    for (const auto &th : thread_state_) {
        if (th->busy_ns == 0) {
            continue;
        }
        const double window = double(th->busy_last - th->busy_first);
        std::ostringstream out;
        out << std::fixed << std::setprecision(1) << "EFA proxy cpu: thread " << th->id
            << ": events=" << th->events << " busy_ms=" << th->busy_ns / 1e6
            << " window_ms=" << window / 1e6
            << " busy_pct=" << (window > 0 ? 100.0 * th->busy_ns / window : 0.0);
        NIXL_INFO << out.str();
    }
    if (!imm_puts_) {
        return;
    }
    for (const auto &th : thread_state_) {
        const auto &st = th->rx_stats;
        if (st.entries == 0 && st.deferred == 0) {
            continue;
        }
        const double window_ns = st.last > st.first ? double(st.last - st.first) : 0.0;
        const double entries = st.entries != 0 ? double(st.entries) : 1.0;
        std::ostringstream out;
        out << std::fixed << std::setprecision(1) << "EFA proxy rx: thread " << th->id
            << ": entries=" << st.entries << " reads=" << st.reads << " sweeps=" << st.sweeps
            << " window_ms=" << window_ns / 1e6
            << " sweep_ns_per_entry=" << double(st.sweep_ns) / entries
            << " read_ns_per_entry=" << double(st.read_ns) / entries
            << " decode_ns_per_entry=" << double(st.decode_ns) / entries
            << " entries_per_read=" << double(st.entries) / std::max<uint64_t>(st.reads, 1)
            << " busy_pct=" << (window_ns > 0 ? 100.0 * double(st.sweep_ns) / window_ns : 0.0)
            << " entries_per_s=" << (window_ns > 0 ? double(st.entries) * 1e9 / window_ns : 0.0)
            << " consumed_rx=" << st.consumed_rx << " deferred_atomics=" << st.deferred;
        NIXL_INFO << out.str();
    }
}

/* ---------------------------------------------------------------------------
 * Peer addresses
 * ------------------------------------------------------------------------- */

nixlLibfabricProxy::PeerAddrs *
nixlLibfabricProxy::peerAddrs(Thread &th, const std::shared_ptr<nixlLibfabricConnection> &conn) {
    auto it = th.peers.find(conn.get());
    if (it != th.peers.end() && !it->second.conn.expired()) {
        return &it->second;
    }

    // A connection this thread has not used yet, possibly at the address of a
    // dead one: release the AV entries of every connection that is gone.
    for (auto p = th.peers.begin(); p != th.peers.end();) {
        if (p->second.conn.expired()) {
            dropPeer(th, p->second);
            p = th.peers.erase(p);
        } else {
            ++p;
        }
    }

    PeerAddrs pa;
    pa.conn = conn;
    pa.rail_ep.assign(rails_,
                      std::vector<fi_addr_t>(conn->remote_rail_ep_names_.size(), FI_ADDR_UNSPEC));
    pa.home.assign(conn->remote_proxy_ep_names_.size(), FI_ADDR_UNSPEC);
    return &th.peers.emplace(conn.get(), std::move(pa)).first->second;
}

fi_addr_t
nixlLibfabricProxy::railAddr(Thread &th,
                             PeerAddrs &pa,
                             const nixlLibfabricConnection &conn,
                             size_t rail,
                             size_t remote_ep) {
    fi_addr_t &addr = pa.rail_ep[rail][remote_ep];
    if (addr == FI_ADDR_UNSPEC &&
        fi_av_insert(th.rails[rail].av,
                     conn.remote_rail_ep_names_[remote_ep].data(),
                     1,
                     &addr,
                     0,
                     nullptr) != 1) {
        addr = FI_ADDR_UNSPEC;
        NIXL_ERROR << "EFA proxy: fi_av_insert failed for " << conn.remoteAgent_ << " EP "
                   << remote_ep << " on rail " << rail;
    }
    return addr;
}

fi_addr_t
nixlLibfabricProxy::homeAddr(Thread &th,
                             PeerAddrs &pa,
                             const nixlLibfabricConnection &conn,
                             size_t owner) {
    fi_addr_t &addr = pa.home[owner];
    if (addr == FI_ADDR_UNSPEC &&
        fi_av_insert(th.ctl_av, conn.remote_proxy_ep_names_[owner].data(), 1, &addr, 0, nullptr) !=
            1) {
        addr = FI_ADDR_UNSPEC;
        NIXL_ERROR << "EFA proxy: fi_av_insert failed for " << conn.remoteAgent_ << " proxy thread "
                   << owner;
    }
    return addr;
}

void
nixlLibfabricProxy::dropPeer(Thread &th, PeerAddrs &pa) {
    // Only for dead connections: their views were quiesced, so nothing is in flight.
    for (size_t r = 0; r < pa.rail_ep.size(); ++r) {
        for (fi_addr_t &addr : pa.rail_ep[r]) {
            if (addr != FI_ADDR_UNSPEC) {
                fi_av_remove(th.rails[r].av, &addr, 1, 0);
            }
        }
    }
    for (size_t r = 0; r < pa.data_ep.size(); ++r) {
        for (fi_addr_t &addr : pa.data_ep[r]) {
            if (addr != FI_ADDR_UNSPEC) {
                fi_av_remove(th.rails[r].av, &addr, 1, 0);
            }
        }
    }
    for (fi_addr_t &addr : pa.home) {
        if (addr == FI_ADDR_UNSPEC) {
            continue;
        }
        // The provider returns one entry per address, so an ack route may share
        // it, with acks queued or in flight: keep those entries.
        bool acks_use_it = false;
        for (const auto &route : th.reply_addrs) {
            acks_use_it = acks_use_it || route.second == addr;
        }
        if (!acks_use_it) {
            fi_av_remove(th.ctl_av, &addr, 1, 0);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Connection info (libfabric_proxy_conninfo.h)
 * ------------------------------------------------------------------------- */

std::string
nixlLibfabricProxy::serializeConnInfo() const {
    nixlLibfabricProxyConnInfo::ProxyEps eps;
    eps.home.reserve(thread_state_.size());
    if (imm_puts_) {
        // Only the rails whose data CQs are polled accept puts with remote CQ data
        // (and the handshake carrying this has a size limit).
        eps.data_rails = rx_rails_;
        eps.rx_by_rail = rx_by_rail_;
    }
    for (const auto &th : thread_state_) {
        eps.home.push_back(th->home_name);
        if (imm_puts_) {
            eps.data.emplace_back();
            for (uint32_t r : rx_rails_) {
                eps.data.back().push_back(th->rails[r].name);
            }
        }
    }
    eps.incarnation = incarnation_;
    return nixlLibfabricProxyConnInfo::serialize(eps);
}

std::string
nixlLibfabricProxy::joinConnInfo(const std::string &engine_part, const std::string &proxy_part) {
    return nixlLibfabricProxyConnInfo::join(engine_part, proxy_part);
}

void
nixlLibfabricProxy::splitConnInfo(const std::string &in,
                                  std::string &engine_part,
                                  std::string &proxy_part) {
    nixlLibfabricProxyConnInfo::split(in, engine_part, proxy_part);
}

nixl_status_t
nixlLibfabricProxy::parseConnInfo(const std::string &blob, nixlLibfabricConnection &conn) {
    nixlLibfabricProxyConnInfo::ProxyEps eps;
    const nixl_status_t status = nixlLibfabricProxyConnInfo::parse(blob, eps);
    conn.remote_proxy_ep_names_ = std::move(eps.home);
    conn.remote_proxy_data_ep_names_.clear();
    conn.remote_proxy_data_rails_ = eps.data_rails;
    conn.remote_proxy_rx_by_rail_ = eps.rx_by_rail;
    if (!eps.data.empty()) {
        // Dense [thread][rail]; rails that accept no remote CQ data stay all-zero.
        const uint32_t rails = *std::max_element(eps.data_rails.begin(), eps.data_rails.end()) + 1;
        conn.remote_proxy_data_ep_names_.assign(
            eps.data.size(), std::vector<std::array<char, LF_EP_NAME_MAX_LEN>>(rails));
        for (size_t t = 0; t < eps.data.size(); ++t) {
            for (size_t i = 0; i < eps.data_rails.size(); ++i) {
                conn.remote_proxy_data_ep_names_[t][eps.data_rails[i]] = eps.data[t][i];
            }
        }
    }
    conn.remote_proxy_incarnation_ = eps.incarnation;
    return status;
}

#endif // HAVE_NIXL_DEVICE_API
