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
#include <cstdio>
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
/** Posted receive buffers per proxy thread for control messages. */
constexpr size_t kRecvPoolSize = 1024;
/** Control send buffers (acks, ring aborts) per proxy thread; more wait in a queue. */
constexpr size_t kCtlPoolSize = 1024;
/** An atomicAdd whose ack has not arrived by then failed (its target died). */
constexpr uint64_t kAckTimeoutNs = 10ull * 1000 * 1000 * 1000;
/** A target-side atomicAdd still waiting for its ring by then is dropped. */
constexpr uint64_t kDeferredTimeoutNs = kAckTimeoutNs;
/** Requests wait this long for the engine handshake with their target. */
constexpr uint64_t kHandshakeTimeoutNs = 60ull * 1000 * 1000 * 1000;
/** Passes between scans for overdue acks while any atomicAdd awaits one. */
constexpr uint32_t kAckScanInterval = 4096;
/** Period of the scan for overdue target-side atomicAdds. */
constexpr uint64_t kDeferredScanNs = 1000ull * 1000 * 1000;
/**
 * Longest a thread applies atomicAdds before it polls its CQs again (the rest wait
 * for the next pass): its receive CQs fill meanwhile, with no back-pressure on the
 * senders (kRxCqSize), and applies can be slow (a GDRCopy pin, CUDA copies).
 */
constexpr uint64_t kDrainBudgetNs = 1000ull * 1000;
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
 * CQs of the data EPs on receive rails. Puts with remote CQ data use the EFA
 * device's unsolicited write receive (where supported, as on p5): they consume no
 * receive buffer, so nothing slows a sender down while the receiving thread is
 * busy (a GDRCopy pin, a CUDA copy). The CQ must then hold what arrives meanwhile:
 * the device maximum on p5 (max_cqe), ~10 ms of 4 KiB writes at a rail's 100
 * Gbit/s. Halved until the device accepts it.
 */
constexpr size_t kRxCqSize = 32768;
/**
 * Proxy EP queue sizes. The provider allocates, faults in and registers each EP's
 * packet pools in one chunk sized by these the first time the EP is used (the
 * defaults, 4096 tx / 8192 rx, are ~36 MB and ~72 MB and took 5-20 ms under the
 * domain lock). Writes with remote CQ data need no posted receives (see
 * kRxCqSize; without unsolicited write receive they use the provider's receive
 * buffers instead, and back-pressure the sender); posts beyond the transmit
 * queues wait in the retry queues.
 */
constexpr size_t kTxSize = 1024;
constexpr size_t kCtlTxSize = 1024;
constexpr size_t kCtlRxSize = 1024;
constexpr size_t kDataRxSize = 64;

namespace wire = nixlLibfabricProxyWire;
namespace ci = nixlLibfabricProxyConnInfo;
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
    enum class Kind : uint8_t { FRAG, RECV, CTL };

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
    bool counted = false; // in ring->outstanding until it completes
    bool is_atomic = false;
    bool awaiting_ack = false; // atomicAdd record posted, the owner's ack not received yet
    nixl_status_t status = NIXL_IN_PROG;
    Ring *ring = nullptr;
    TxRing *tx = nullptr; // the ring's counts towards the request's target
    uint64_t seq = 0; // a put: its epoch (the next atomicAdd's index); an atomicAdd: its index
    fi_addr_t dest = FI_ADDR_UNSPEC; // atomicAdd: the counter owner's control EP
    uint64_t ack_deadline = 0; // nowNs() by which the ack must arrive
    wire::atomicAddMsg msg{}; // atomicAdd send buffer; the request pool is registered memory
    bool inject_cq_error = false; // NIXL_EFA_PROXY_INJECT: fail its first completion

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
 * Sender-side counts of one runtime ring towards one target incarnation. They
 * outlive the ring (quiesce), since the target's counts do. After a failure the
 * target may never see some counted put, so the ring's next use takes a new key
 * (fresh counts on both sides).
 */
struct nixlLibfabricProxy::TxRing {
    uint32_t key = 0; // wire::ringKey(): the sender's index at the target, a ring id
    uint64_t atomics = 0; // atomicAdds sent; the epoch of the puts that follow
    std::array<uint64_t, wire::kEpochs> slot_puts{}; // fragments sent per epoch slot
    uint32_t inflight = 0; // requests not complete yet
    // Completions of its requests, and when the ack scan last saw that count move:
    // an atomicAdd times out only after kAckTimeoutNs without any progress (its
    // puts may legitimately take long to land).
    uint64_t completions = 0;
    uint64_t scanned_completions = 0;
    uint64_t progress_ns = 0;
    nixl_status_t error = NIXL_SUCCESS; // sticky until the next use re-keys the ring
    bool ready = false; // key assigned (the engine handshake gave the sender's index)
    uint64_t aborted_from = std::numeric_limits<uint64_t>::max(); // abort sent from this seq
    std::weak_ptr<nixlLibfabricConnection> conn; // the target (for ring aborts)
};

/** Target-side state of one sender ring, shared between proxy threads. */
struct nixlLibfabricProxy::RxRing {
    // Put fragments arrived per epoch slot (release by the counting thread).
    std::array<std::atomic<uint64_t>, wire::kEpochs> received{};
    std::atomic<uint64_t> applied{0}; // atomicAdds applied or failed, in seq order
    std::atomic<int32_t> error{NIXL_SUCCESS}; // first failed atomicAdd: later ones fail
    // The sender aborted the ring from this seq on (UINT64_MAX: not aborted).
    std::atomic<uint64_t> aborted_from{std::numeric_limits<uint64_t>::max()};
};

/** An owner thread's atomicAdds of one ring that wait for it, by seq. */
struct nixlLibfabricProxy::RxOwned {
    struct Add {
        wire::atomicAddMsg msg;
        fi_addr_t reply;
        uint64_t arrived;
    };

    RxRing *ring = nullptr;
    std::map<uint64_t, Add> adds;
    // The ring's put arrivals (all slots; they only grow) and applied count when
    // the expiry scan last looked, and when either last moved: adds are dropped
    // only after kDeferredTimeoutNs without progress (puts may take long to land).
    uint64_t seen_received = 0;
    uint64_t seen_applied = 0;
    uint64_t progress_ns = 0;
};

/** A request waiting for the engine handshake with its target. */
struct nixlLibfabricProxy::Parked {
    nixlBackendProxySubmission sub;
    Request *req;
    uint64_t since;
};

/** One runtime ring (channel, peer slot) at the sender. */
struct nixlLibfabricProxy::Ring {
    uint32_t channel = 0;
    uint64_t index = 0; // channel * max_peers + peer slot
    uint32_t next_rail = 0; // unstriped puts rotate over the buffer's rails
    uint32_t outstanding = 0; // requests in flight, parked ones included
    nixl_status_t error = NIXL_SUCCESS; // first failure: later atomicAdds fail
    // Last target and its counts (most rings have one target).
    const nixlLibfabricConnection *tx_conn = nullptr;
    TxRing *tx = nullptr;
    bool multi_target_warned = false;
    std::deque<Parked> parked; // in ring order, behind a pending handshake

    [[nodiscard]] bool
    idle() const {
        return outstanding == 0;
    }
};

struct nixlLibfabricProxy::RecvBuf {
    OpCtx op; // must stay first
    wire::anyMsg msg;
};

struct nixlLibfabricProxy::CtlBuf {
    OpCtx op; // must stay first
    wire::anyMsg msg;
    fi_addr_t dest = FI_ADDR_UNSPEC; // while in flight (dropPeer() keeps its AV entry)
};

/** A control message (ack or ring abort) waiting for a send buffer or queue room. */
struct nixlLibfabricProxy::PendingCtl {
    fi_addr_t dest;
    wire::msgType type;
    uint64_t token; // ack
    nixl_status_t status; // ack
    uint32_t ring; // abort
    uint64_t first_seq; // abort
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
    bool has_imm = false; // a write with remote CQ data (to a proxy data EP)
    uint32_t imm = 0;
    uint64_t queued_at = 0; // entry into the retry queue (profiling only)
    bool inject_error = false; // NIXL_EFA_PROXY_INJECT: fail this post
};

struct nixlLibfabricProxy::PeerAddrs {
    // Expired once the connection is gone; its address may then be reused.
    std::weak_ptr<const nixlLibfabricConnection> conn;
    // Inserted into this thread's AVs on first use; FI_ADDR_UNSPEC until then.
    // rail_ep[local_rail][remote_ep] -> remote engine EP (peers without a proxy)
    std::vector<std::vector<fi_addr_t>> rail_ep;
    // home[remote_thread] -> its control EP, in this thread's control AV
    std::vector<fi_addr_t> home;
    // data_ep[local_rail][data_index] -> the remote data EP receiving on the remote's
    // data rail number data_index, in that local rail's AV
    std::vector<std::vector<fi_addr_t>> data_ep;
    // data_index[remote_rail]: its position among the remote's data rails, or -1
    std::vector<int> data_index;
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
    std::string name; // the data EP's name (published for the receive rails)
};

struct nixlLibfabricProxy::Thread {
    uint32_t id = 0;
    uint32_t home = 0;
    std::thread::id owner{}; // the worker that drives this thread's channels
    // Held while this thread applies an add; registration changes take all of them.
    std::mutex regions_lock;
    std::vector<RailRes> rails;
    // efa_proxy_tx_domain=private: per rail, a send EP in a domain of its own (domain
    // unset elsewhere), the rails it wrote on, and its registrations of source
    // buffers there, by (rail, region start).
    std::vector<RailRes> tx;
    std::vector<uint32_t> tx_poll_rails;
    std::map<std::pair<uint32_t, uintptr_t>, struct fid_mr *> tx_mrs;
    // Data CQs this thread polls: the receive rails it owns, plus every rail it has
    // written on (its write completions). The rail domains are FI_THREAD_SAFE, so
    // skipping other CQs also avoids taking their domain locks.
    std::vector<uint32_t> poll_rails;
    std::vector<bool> polled;
    uint32_t passes = 0;
    uint64_t next_idle_poll = 0; // nowNs() before which an idle thread skips its CQs
    uint64_t cq_errors = 0; // since the last report
    uint64_t cq_error_report = 0; // time of the last report, 0 before the first
    std::unique_ptr<nixlLibfabricProxyProfile> prof; // NIXL_EFA_PROXY_PROFILE only
    uint64_t last_pass = 0;
    // Control EP on the home rail, with its own CQ and AV (EFA does not share them
    // between EPs): atomicAdd records, acks and ring aborts never wait behind bulk
    // writes. Every control address (counter owners, ack destinations) is in ctl_av.
    struct fid_ep *ctl_ep = nullptr;
    struct fid_cq *ctl_cq = nullptr;
    struct fid_av *ctl_av = nullptr;
    struct fi_info *ctl_info = nullptr;
    uint64_t ctl_cq_reads = 0; // profiling only
    std::string home_name; // the control EP's name

    std::vector<Request> reqs;
    std::vector<Request *> free_reqs;
    struct fid_mr *req_mr = nullptr;
    void *req_desc = nullptr;

    std::vector<RecvBuf> recvs;
    struct fid_mr *recv_mr = nullptr;
    void *recv_desc = nullptr;

    std::vector<CtlBuf> ctls;
    std::vector<CtlBuf *> free_ctls;
    struct fid_mr *ctl_mr = nullptr;
    void *ctl_desc = nullptr;
    std::deque<PendingCtl> pending_ctls; // in order
    // Senders' control endpoints, by name, in ctl_av (where acks go).
    std::unordered_map<std::string, fi_addr_t, NameHash, std::equal_to<>> reply_addrs;
    // atomicAdds whose ack is outstanding (entries go stale once acked; pruned on scans).
    std::vector<Request *> awaiting;
    uint32_t ack_scan = 0;

    // Back-pressured posts per rail, plus one for the control EP (last): a full
    // transmit queue does not hold back the others (the target, not posting order,
    // orders a ring).
    std::vector<std::deque<PendingPost>> retry;
    size_t retry_count = 0;
    std::vector<RecvBuf *> recv_retry; // receive buffers the EP could not take yet
    std::unordered_map<uint64_t, Ring> rings; // by channel * max_peers + peer
    std::vector<uint64_t> parked_rings; // rings with parked requests
    std::unordered_map<const nixlLibfabricConnection *, PeerAddrs> peers;
    int cuda_dev = -1;

    // Sender side: counts per (target, incarnation, runtime ring).
    std::map<std::tuple<std::string, uint64_t, uint64_t>, TxRing> tx_rings;
    // Target side: rings by key (shared RxRing objects, cached), the atomicAdds this
    // thread owns that wait for their rings, and whether the last CQ sweep found
    // remote CQ data (the thread then keeps polling on every pass).
    std::unordered_map<uint32_t, RxRing *> rx_cache;
    uint32_t rx_last_key = 0;
    RxRing *rx_last = nullptr;
    std::unordered_map<RxRing *, RxOwned> owned;
    uint64_t next_deferred_scan = 0;
    bool rx_active = false;

    // Profiling (NIXL_EFA_PROXY_PROFILE only).
    struct RxStats {
        uint64_t entries = 0; // CQ entries with remote CQ data
        uint64_t reads = 0; // fi_cq_read calls that returned any
        uint64_t read_ns = 0; // in those calls
        uint64_t decode_ns = 0; // decoding and counting their entries
        uint64_t sweeps = 0; // pollCqs() sweeps that found any
        uint64_t sweep_ns = 0; // in those sweeps
        uint64_t first = 0; // first and last entry (nowNs)
        uint64_t last = 0;
        uint64_t flushes = 0; // GPUDirect RDMA write flushes before atomicAdds
    } rx_stats;
    uint64_t sweep_imm = 0; // entries found in the current sweep
    // Time in progress() passes that did work (read a CQ entry or applied an add),
    // and the span from the first to the last.
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
    for (const char *gone : {"efa_proxy_delivery_complete",
                             "efa_proxy_rx_thread",
                             "efa_proxy_data_rx_size",
                             "efa_proxy_test_put_imm",
                             "efa_proxy_test_fence_off",
                             "efa_proxy_test_data_rx_post"}) {
        if (params.count(gone) != 0) {
            NIXL_WARN << "EFA proxy: " << gone << " is no longer supported; ignored";
        }
    }
    if (auto order = params.find("efa_proxy_ordering");
        order != params.end() && order->second != "receiver") {
        NIXL_WARN << "EFA proxy: efa_proxy_ordering=" << order->second
                  << " is no longer supported: put -> atomicAdd order is kept at the target";
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
    if (auto it = params.find("efa_proxy_rail_policy"); it != params.end()) {
        if (it->second == "ring") {
            rail_per_thread_ = false;
        } else if (it->second != "thread") {
            NIXL_WARN << "EFA proxy: unknown efa_proxy_rail_policy '" << it->second
                      << "'; using 'thread'";
        }
    }
    if (auto it = params.find("efa_proxy_tx_domain"); it != params.end()) {
        if (it->second == "private") {
            tx_private_ = true;
        } else if (it->second != "shared") {
            NIXL_WARN << "EFA proxy: unknown efa_proxy_tx_domain '" << it->second
                      << "'; using 'shared'";
        }
    }
    if (auto it = params.find("efa_proxy_rx_flush"); it != params.end()) {
        if (it->second == "1" || it->second == "0") {
            rx_flush_param_ = it->second == "1" ? 1 : 0;
        } else {
            NIXL_WARN << "EFA proxy: invalid efa_proxy_rx_flush '" << it->second
                      << "'; using the platform default";
        }
    }
    const char *profile = std::getenv("NIXL_EFA_PROXY_PROFILE");
    profile_ = profile != nullptr && *profile != '\0' && std::string(profile) != "0";
    inject_ = Inject::fromEnv();
    counters_ = std::make_unique<CounterMap>(!(inject_ && inject_->no_gdrcopy));
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
    if (config.ring_depth > wire::kEpochs) {
        // A ring's epoch slots are reused every kEpochs atomicAdds.
        NIXL_ERROR << "EFA proxy: supports a ring depth of at most " << wire::kEpochs << ", not "
                   << config.ring_depth;
        return NIXL_ERR_INVALID_PARAM;
    }

    // Home rails (control EPs) and receive rails on the EFA devices next to this
    // process's GPU, where its GPU buffers are registered; all rails without a GPU.
    std::vector<size_t> home_rails;
    int cc_major = 0;
#ifdef HAVE_CUDA
    int dev = 0;
    char bus_id[32] = {};
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev) == cudaSuccess) {
        home_rails = engine_.rail_manager_.railsForAccelerator(bus_id);
    }
    if (cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        cc_major = 0;
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
    // As NCCL (ncclTopoNeedFlush): GPUDirect RDMA writes need a flush before a
    // dependent signal only before Hopper, and on C2C (aarch64) platforms.
#if defined(__aarch64__)
    const bool flush_default = true;
#else
    const bool flush_default = cc_major < 9;
#endif
    rx_flush_ = rx_flush_param_ < 0 ? flush_default : rx_flush_param_ == 1;

    for (uint32_t t = 0; t < threads_; ++t) {
        auto th = std::make_unique<Thread>();
        th->id = t;
        if (profile_) {
            th->prof = std::make_unique<nixlLibfabricProxyProfile>();
        }
        th->home = static_cast<uint32_t>(home_rails[t % home_rails.size()]);
        th->rails.resize(rails_);
        th->tx.resize(rails_);
        th->retry.resize(rails_ + 1);
        th->polled.assign(rails_, false);

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
            rr.info->rx_attr->size = std::min(rr.info->rx_attr->size, kDataRxSize);
            rr.virt_addr = (rr.info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;

            const bool receives = std::find(rx_rails_.begin(), rx_rails_.end(), r) != rx_rails_.end();
            struct fi_cq_attr cq_attr = {};
            cq_attr.format = FI_CQ_FORMAT_DATA;
            cq_attr.wait_obj = FI_WAIT_NONE;
            cq_attr.size = receives ? kRxCqSize : kCqSize;
            struct fi_av_attr av_attr = {};
            int ret = fi_cq_open(rr.domain, &cq_attr, &rr.cq, nullptr);
            while (ret == -FI_EINVAL && cq_attr.size > kCqSize) {
                cq_attr.size /= 2; // above the device's CQ size limit
                ret = fi_cq_open(rr.domain, &cq_attr, &rr.cq, nullptr);
            }
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
                std::array<char, LF_EP_NAME_MAX_LEN> name{};
                size_t len = name.size();
                ret = fi_getname(&rr.ep->fid, name.data(), &len);
                rr.name.assign(name.data(), std::min(len, name.size()));
            }
            if (ret) {
                NIXL_ERROR << "EFA proxy: endpoint setup failed for thread " << t << " rail " << r
                           << ": " << fi_strerror(-ret);
                releaseThread(*th);
                return NIXL_ERR_BACKEND;
            }
        }

        if (tx_private_) {
            for (const uint32_t r : rx_rails_) {
                if (const int ret = openTxRail(*th, r)) {
                    NIXL_ERROR << "EFA proxy: private send domain failed for thread " << t
                               << " rail " << r << ": " << fi_strerror(-ret);
                    releaseThread(*th);
                    return NIXL_ERR_BACKEND;
                }
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
                std::array<char, LF_EP_NAME_MAX_LEN> name{};
                size_t len = name.size();
                ret = fi_getname(&th->ctl_ep->fid, name.data(), &len);
                th->home_name.assign(name.data(), std::min(len, name.size()));
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
        th->ctls.resize(kCtlPoolSize);
        th->free_ctls.reserve(kCtlPoolSize);
        for (auto &buf : th->ctls) {
            buf.op.kind = OpCtx::Kind::CTL;
            th->free_ctls.push_back(&buf);
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
                            th->ctls.data(),
                            th->ctls.size() * sizeof(CtlBuf),
                            FI_SEND,
                            0,
                            0,
                            0,
                            &th->ctl_mr,
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
        th->ctl_desc = fi_mr_desc(th->ctl_mr);

        for (auto &buf : th->recvs) {
            if (postRecv(*th, &buf) == NIXL_ERR_BACKEND) {
                releaseThread(*th);
                return NIXL_ERR_BACKEND;
            }
        }

        // The receive rails this thread owns: it polls their data CQs from the start
        // (puts arrive there before this thread posts anything on them).
        for (size_t i = 0; i < rx_rails_.size(); ++i) {
            const uint32_t r = rx_rails_[i];
            if (wire::railThread(static_cast<uint32_t>(i), threads_) == t && !th->polled[r]) {
                th->polled[r] = true;
                th->poll_rails.push_back(r);
            }
        }
        thread_state_.push_back(std::move(th));
    }

    std::string homes, receivers;
    for (const auto &th : thread_state_) {
        homes += (homes.empty() ? "" : ",") + std::to_string(th->home);
    }
    for (size_t i = 0; i < rx_rails_.size(); ++i) {
        receivers += (receivers.empty() ? "" : ",") + std::to_string(rx_rails_[i]) + ":" +
            std::to_string(wire::railThread(static_cast<uint32_t>(i), threads_));
    }
    NIXL_INFO << "EFA proxy: " << threads_ << " thread(s) x " << rails_
              << " rail(s); control EPs on rails " << homes << "; puts received on rail:thread "
              << receivers << "; small puts per " << (rail_per_thread_ ? "thread" : "ring")
              << " rail; idle poll every " << idle_poll_ns_ / 1000 << " us; GPUDirect RDMA "
              << (rx_flush_ ? "flushed" : "not flushed") << " before atomicAdds (compute "
              << cc_major << ".x)";
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
    for (auto &rr : th.tx) {
        close(rr.ep ? &rr.ep->fid : nullptr);
        rr.ep = nullptr;
    }
    for (auto &[key, mr] : th.tx_mrs) {
        close(&mr->fid);
    }
    th.tx_mrs.clear();
    for (auto &rr : th.tx) {
        close(rr.cq ? &rr.cq->fid : nullptr);
        close(rr.av ? &rr.av->fid : nullptr);
        close(rr.domain ? &rr.domain->fid : nullptr);
        if (rr.info) {
            fi_freeinfo(rr.info);
        }
        rr = RailRes{};
    }
    th.tx_poll_rails.clear();
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
    close(th.ctl_mr ? &th.ctl_mr->fid : nullptr);
    th.req_mr = nullptr;
    th.recv_mr = nullptr;
    th.ctl_mr = nullptr;
    th.retry.clear();
    th.retry_count = 0;
    th.recv_retry.clear();
    th.pending_ctls.clear();
    th.reply_addrs.clear();
    th.awaiting.clear();
    th.poll_rails.clear();
    th.rings.clear();
    th.parked_rings.clear();
    th.peers.clear();
    th.tx_rings.clear();
    th.rx_cache.clear();
    th.rx_last = nullptr;
    th.owned.clear();
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
        report();
    }
    for (auto &th : thread_state_) {
        if (th->cq_errors != 0) {
            NIXL_ERROR << "EFA proxy: thread " << th->id << ": " << th->cq_errors
                       << " CQ error(s) since the last report";
        }
        size_t waiting = 0;
        for (const auto &[ring, owned] : th->owned) {
            waiting += owned.adds.size();
        }
        if (waiting != 0) {
            NIXL_WARN << "EFA proxy: thread " << th->id << ": " << waiting
                      << " atomicAdd(s) still waiting for their rings at shutdown";
        }
        if (!th->pending_ctls.empty()) {
            NIXL_WARN << "EFA proxy: thread " << th->id << ": " << th->pending_ctls.size()
                      << " control message(s) not sent at shutdown";
        }
        releaseThread(*th);
    }
    thread_state_.clear();
    return NIXL_SUCCESS;
}

void
nixlLibfabricProxy::report() const {
    // Straight to stderr: the summary is wanted without turning on INFO logging.
    nixlLibfabricProxyProfile total;
    for (const auto &th : thread_state_) {
        total.merge(*th->prof);
    }
    for (const std::string &line : total.report()) {
        std::fprintf(stderr, "EFA proxy profile: %s\n", line.c_str());
    }
    for (const auto &th : thread_state_) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(1);
        if (th->busy_ns != 0) {
            const double window = double(th->busy_last - th->busy_first);
            out << "EFA proxy cpu: thread " << th->id << ": events=" << th->events
                << " busy_ms=" << th->busy_ns / 1e6 << " window_ms=" << window / 1e6
                << " busy_pct=" << (window > 0 ? 100.0 * th->busy_ns / window : 0.0) << "\n";
        }
        const auto &st = th->rx_stats;
        if (st.entries != 0 || st.flushes != 0) {
            const double window = st.last > st.first ? double(st.last - st.first) : 0.0;
            const double entries = st.entries != 0 ? double(st.entries) : 1.0;
            out << "EFA proxy rx: thread " << th->id << ": entries=" << st.entries
                << " reads=" << st.reads << " sweeps=" << st.sweeps << " window_ms=" << window / 1e6
                << " sweep_ns_per_entry=" << double(st.sweep_ns) / entries
                << " read_ns_per_entry=" << double(st.read_ns) / entries
                << " decode_ns_per_entry=" << double(st.decode_ns) / entries
                << " entries_per_read=" << double(st.entries) / std::max<uint64_t>(st.reads, 1)
                << " busy_pct=" << (window > 0 ? 100.0 * double(st.sweep_ns) / window : 0.0)
                << " flushes=" << st.flushes << "\n";
        }
        std::fputs(out.str().c_str(), stderr);
    }
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
    req->counted = false;
    req->is_atomic = false;
    req->awaiting_ack = false;
    req->status = NIXL_IN_PROG;
    req->ring = nullptr;
    req->tx = nullptr;
    req->seq = 0;
    req->dest = FI_ADDR_UNSPEC;
    req->inject_cq_error = false;
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
    const uint64_t ring_index =
        static_cast<uint64_t>(sub.channel_id) * config_.max_peers + sub.peer_index;
    auto [it, created] = th.rings.try_emplace(ring_index);
    Ring &ring = it->second;
    if (created) {
        ring.channel = sub.channel_id;
        ring.index = ring_index;
    }
    const bool is_put = sub.opcode == nixl_proxy_opcode_t::PUT;

    Request *req = allocRequest(th);
    if (!req) {
        NIXL_ERROR << "EFA proxy: request pool exhausted on thread " << th.id;
        if (is_put && ring.error == NIXL_SUCCESS) {
            ring.error = NIXL_ERR_BACKEND; // never signal after a put that did not leave
        }
        return NIXL_ERR_BACKEND;
    }
    req->ring = &ring;
    req->is_atomic = !is_put;
    if (th.prof) {
        req->t_submit = nixlLibfabricProxyProfile::now();
        req->prof_op = is_put ? nixlLibfabricProxyProfile::PUT : nixlLibfabricProxyProfile::ATOMIC;
        req->prof_class = static_cast<uint8_t>(nixlLibfabricProxyProfile::sizeClass(sub.size));
    }

#ifdef HAVE_CUDA
    // Puts only: an atomicAdd's submission has no local buffer (its desc is unset).
    if (is_put && sub.local.mem_type == VRAM_SEG &&
        th.cuda_dev != static_cast<int>(sub.local.desc.devId)) {
        if (cudaSetDevice(static_cast<int>(sub.local.desc.devId)) == cudaSuccess) {
            th.cuda_dev = static_cast<int>(sub.local.desc.devId);
        }
    }
#endif

    // Behind requests parked for a handshake, so the ring keeps its order.
    bool park = !ring.parked.empty();
    nixl_status_t status = NIXL_SUCCESS;
    if (!park) {
        status = submitOp(th, ring, sub, req, park);
    }
    if (park) {
        if (ring.parked.empty()) {
            th.parked_rings.push_back(ring_index);
        }
        ring.parked.push_back(Parked{sub, req, nowNs()});
        req->counted = true;
        ++ring.outstanding;
    } else if (status != NIXL_SUCCESS) {
        freeRequest(th, req);
        if (is_put && ring.error == NIXL_SUCCESS) {
            ring.error = status; // never signal after a put that did not leave
        }
        return status;
    }

    request.token = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req));
    request.context = th.id;
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
nixlLibfabricProxy::submitOp(Thread &th,
                             Ring &ring,
                             const nixlBackendProxySubmission &sub,
                             Request *req,
                             bool &parked) {
    parked = false;
    auto *remote = static_cast<nixlLibfabricPublicMetadata *>(sub.remote.desc.metadataP);
    if (!remote || !remote->conn_) {
        NIXL_ERROR << "EFA proxy: operation without libfabric metadata";
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto &conn = remote->conn_;
    TxRing *tx = nullptr;
    if (!conn->remote_proxy_ep_names_.empty()) {
        nixl_status_t status = NIXL_SUCCESS;
        tx = txRing(th, ring, conn, status);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        if (tx == nullptr) {
            parked = true; // the handshake with the target has not arrived yet
            return NIXL_SUCCESS;
        }
    }
    switch (sub.opcode) {
    case nixl_proxy_opcode_t::PUT:
        return submitPut(th, ring, sub, req, tx);
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        if (tx == nullptr) {
            NIXL_ERROR << "EFA proxy: peer " << conn->remoteAgent_
                       << " published no (compatible) device proxy; atomicAdd unavailable";
            return NIXL_ERR_NOT_SUPPORTED;
        }
        return submitAtomic(th, ring, sub, req, tx);
    default:
        return NIXL_ERR_NOT_SUPPORTED;
    }
}

nixlLibfabricProxy::TxRing *
nixlLibfabricProxy::txRing(Thread &th,
                           Ring &ring,
                           const std::shared_ptr<nixlLibfabricConnection> &conn,
                           nixl_status_t &status) {
    TxRing *tx = ring.tx;
    if (tx == nullptr || ring.tx_conn != conn.get()) {
        if (ring.tx_conn != nullptr && ring.tx_conn != conn.get() && !ring.multi_target_warned) {
            ring.multi_target_warned = true;
            NIXL_WARN << "EFA proxy: ring (channel " << ring.channel << ", slot "
                      << ring.index % config_.max_peers << ") carries operations to more than "
                      << "one agent; an atomicAdd is ordered only after the puts to its own "
                      << "target";
        }
        tx = &th.tx_rings[std::make_tuple(
            conn->remoteAgent_, conn->remote_proxy_incarnation_, ring.index)];
        if (tx->error != NIXL_SUCCESS && tx->inflight == 0) {
            // Failed before: some counted put may never have reached the target, so
            // the counts are no longer shared. Start over under a new key.
            *tx = TxRing{};
        }
        ring.tx = tx;
        ring.tx_conn = conn.get();
        tx->conn = conn;
    }
    if (tx->ready) {
        return tx;
    }
    // The key holds the sender's index at the target, which the engine handshake
    // carries; requests wait for it (they are parked) rather than guess.
    uint32_t sender_index = 0;
    if (conn->remoteAgent_ != engine_.localAgent) {
        if (!conn->handshake_received_.load(std::memory_order_acquire)) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(conn->handshake_mutex_); // written under it
        sender_index = conn->local_agent_idx_at_remote_;
    }
    if (sender_index > wire::kMaxSenderIndex || conn->remote_proxy_data_rails_.empty()) {
        NIXL_ERROR << "EFA proxy: cannot order operations to " << conn->remoteAgent_ << " (index "
                   << sender_index << ", " << conn->remote_proxy_data_rails_.size()
                   << " receive rail(s))";
        status = NIXL_ERR_NOT_SUPPORTED;
        return nullptr;
    }
    uint32_t ring_id = 0;
    {
        std::lock_guard<std::mutex> lock(ring_ids_mutex_);
        uint32_t &next =
            next_ring_id_[std::make_pair(conn->remoteAgent_, conn->remote_proxy_incarnation_)];
        if (next >= wire::kRingIds) {
            NIXL_ERROR << "EFA proxy: ring ids towards " << conn->remoteAgent_ << " exhausted";
            status = NIXL_ERR_BACKEND;
            return nullptr;
        }
        ring_id = next++;
    }
    tx->key = wire::ringKey(sender_index, ring_id);
    tx->ready = true;
    return tx;
}

void
nixlLibfabricProxy::replayParked(Thread &th, uint64_t now_ns) {
    for (size_t i = 0; i < th.parked_rings.size();) {
        auto it = th.rings.find(th.parked_rings[i]);
        if (it == th.rings.end()) {
            th.parked_rings[i] = th.parked_rings.back();
            th.parked_rings.pop_back();
            continue;
        }
        Ring &ring = it->second;
        while (!ring.parked.empty()) {
            Parked &p = ring.parked.front();
            // Uncounted while it is tried again; submitOp counts it once it leaves.
            p.req->counted = false;
            --ring.outstanding;
            bool park = false;
            nixl_status_t status = submitOp(th, ring, p.sub, p.req, park);
            if (park && now_ns - p.since < kHandshakeTimeoutNs) {
                p.req->counted = true;
                ++ring.outstanding;
                break;
            }
            if (park) {
                NIXL_ERROR << "EFA proxy: no engine handshake with the target of ring (channel "
                           << ring.channel << ") after " << kHandshakeTimeoutNs / 1000000000
                           << " s; failing its requests";
                status = NIXL_ERR_REMOTE_DISCONNECT;
            }
            if (status != NIXL_SUCCESS) {
                p.req->status = status;
                p.req->frags_left = 0;
                if (!p.req->is_atomic && ring.error == NIXL_SUCCESS) {
                    ring.error = status;
                }
            }
            ring.parked.pop_front();
        }
        if (ring.parked.empty()) {
            th.parked_rings[i] = th.parked_rings.back();
            th.parked_rings.pop_back();
        } else {
            ++i;
        }
    }
}

nixl_status_t
nixlLibfabricProxy::submitPut(Thread &th,
                              Ring &ring,
                              const nixlBackendProxySubmission &sub,
                              Request *req,
                              TxRing *tx) {
    auto *local = static_cast<nixlLibfabricPrivateMetadata *>(sub.local.desc.metadataP);
    auto *remote = static_cast<nixlLibfabricPublicMetadata *>(sub.remote.desc.metadataP);
    if (!local) {
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
    const nixlLibfabricConnection &conn = *remote->conn_;

    const size_t size = sub.size;
    if (size == 0) {
        req->status = NIXL_SUCCESS;
        return NIXL_SUCCESS;
    }

    // Same rule as the host path: stripe at or above the threshold over >1 rail.
    // Smaller puts go to one rail each (unstripedRail()).
    const bool stripe = size >= engine_.striping_threshold_ && lrails.size() > 1;
    const size_t nfrag = stripe ? std::min(lrails.size(), kMaxStripes) : 1;
    const size_t chunk = size / nfrag;
    const size_t first = stripe ? 0 : unstripedRail(th, ring, lrails.size());

    // Resolve every destination before anything is counted or posted.
    std::array<PendingPost, kMaxStripes> posts{};
    for (size_t i = 0; i < nfrag; ++i) {
        const size_t sel = first + i;
        const size_t rail = lrails[sel];
        size_t rep = reps[sel % reps.size()];
        const size_t off = stripe ? i * chunk : 0;
        const size_t len = stripe ? (i + 1 == nfrag ? size - off : chunk) : size;
        const uint64_t target = sub.remote.desc.addr + off;

        PendingPost &pp = posts[i];
        pp.kind = PendingPost::Kind::WRITE;
        pp.rail = static_cast<uint32_t>(rail);
        pp.fctx = &req->frag[i];
        pp.local = reinterpret_cast<void *>(sub.local.desc.addr + off);
        pp.len = len;
        pp.desc = txDesc(th, rail, sub.local.desc.addr, fi_mr_desc(local->rail_mr_list_[rail]));
        if (pp.desc == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        if (tx != nullptr) {
            // To the data EP receiving on the destination rail; a buffer rail the
            // target does not receive on (e.g. host memory on other rails) is
            // replaced by one of the buffer's rails it does receive on.
            int index = rep < peer->data_index.size() ? peer->data_index[rep] : -1;
            for (size_t k = 1; index < 0 && k < reps.size(); ++k) {
                const size_t alt = reps[(sel + k) % reps.size()];
                if (alt < peer->data_index.size() && peer->data_index[alt] >= 0) {
                    rep = alt;
                    index = peer->data_index[alt];
                }
            }
            if (index < 0) {
                NIXL_ERROR << "EFA proxy: no rail of the destination buffer at "
                           << conn.remoteAgent_ << " receives puts (receive rails are those "
                           << "next to its GPU)";
                return NIXL_ERR_NOT_SUPPORTED;
            }
            pp.dest = dataAddr(th, *peer, conn, rail, static_cast<size_t>(index));
            pp.has_imm = true;
            pp.imm = wire::putImm(tx->key, tx->atomics);
        } else {
            pp.dest = railAddr(th, *peer, conn, rail, rep);
        }
        const uint64_t base = remote->remote_buf_addr_;
        pp.raddr = th.rails[rail].virt_addr ? target : target - base;
        pp.rkey = remote->rail_remote_key_list_[rep];
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
    req->counted = true;
    ++ring.outstanding;
    if (tx != nullptr) {
        // The next atomicAdd's slot waits for nfrag more.
        req->tx = tx;
        req->seq = tx->atomics;
        tx->slot_puts[tx->atomics % wire::kEpochs] += nfrag;
        ++tx->inflight;
    }
    for (size_t i = 0; i < nfrag; ++i) {
        post(th, std::move(posts[i]));
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricProxy::submitAtomic(Thread &th,
                                 Ring &ring,
                                 const nixlBackendProxySubmission &sub,
                                 Request *req,
                                 TxRing *tx) {
    auto *remote = static_cast<nixlLibfabricPublicMetadata *>(sub.remote.desc.metadataP);
    PeerAddrs *peer = peerAddrs(th, remote->conn_);
    if (!peer) {
        return NIXL_ERR_BACKEND;
    }
    const nixlLibfabricConnection &conn = *remote->conn_;

    // Never signal after a failure on the ring (as with the fence it replaces).
    const nixl_status_t error = ring.error != NIXL_SUCCESS ? ring.error : tx->error;
    if (error != NIXL_SUCCESS) {
        req->status = error;
        req->frags_left = 0;
        return NIXL_SUCCESS;
    }

    // All senders must agree on one owner thread per counter at the target.
    const uint32_t owner = wire::counterOwner(
        sub.remote.desc.addr, static_cast<uint32_t>(conn.remote_proxy_ep_names_.size()));
    req->dest = homeAddr(th, *peer, conn, owner);
    if (req->dest == FI_ADDR_UNSPEC) {
        return NIXL_ERR_BACKEND;
    }
    req->msg = wire::atomicAddMsg{};
    req->msg.hdr = wire::msgHeader{wire::kVersion, wire::msgType::ATOMIC_ADD, 0};
    req->msg.remote_addr = sub.remote.desc.addr;
    req->msg.value = sub.value;
    req->msg.token = req->token();
    req->msg.ring = tx->key;
    req->msg.seq = tx->atomics++;
    req->msg.expected_puts = tx->slot_puts[req->msg.seq % wire::kEpochs];
    req->msg.reply_name_len = static_cast<uint32_t>(th.home_name.size());
    std::memcpy(req->msg.reply_name, th.home_name.data(), th.home_name.size());
    NIXL_DEBUG << "EFA proxy: atomicAdd " << sub.value << " to " << std::hex << sub.remote.desc.addr
               << " ring " << tx->key << std::dec << " seq " << req->msg.seq << " expecting "
               << req->msg.expected_puts << " -> " << conn.remoteAgent_ << " thread " << owner;

    req->tx = tx;
    req->seq = req->msg.seq;
    // Completes on its send and on the owner's ack (the add applied, or why not).
    req->frags_left = 2;
    req->counted = true;
    ++ring.outstanding;
    ++tx->inflight;
    sendAtomic(th, req);
    return NIXL_SUCCESS;
}

size_t
nixlLibfabricProxy::unstripedRail(const Thread &th, Ring &ring, size_t nrails) const {
    if (!rail_per_thread_) {
        return ring.next_rail++ % nrails; // every ring rotates over all rails
    }
    // The buffer's rails are split among the proxy threads, and each thread
    // rotates over its share: a rail's domain (FI_THREAD_SAFE, one lock) and CQ
    // are then shared by ceil(threads / rails) threads instead of all of them.
    if (threads_ >= nrails) {
        return th.id % nrails;
    }
    const size_t share =
        (nrails - th.id + threads_ - 1) / threads_; // i < nrails, i % threads_ == id
    return th.id + (ring.next_rail++ % share) * threads_;
}

void
nixlLibfabricProxy::failRing(Thread &th, Request *req, nixl_status_t status) {
    // As the fence did: nothing on the ring signals after a failed operation.
    if (req->ring != nullptr && req->ring->error == NIXL_SUCCESS) {
        req->ring->error = status;
    }
    TxRing *tx = req->tx;
    if (tx == nullptr) {
        return;
    }
    if (tx->error == NIXL_SUCCESS) {
        tx->error = status;
    }
    // A failed put of epoch e: atomicAdds e and later wait for it in vain. A failed
    // atomicAdd s (lost, or failed at the target): later ones wait for it in vain.
    // Claimed first: failing those calls back here, with no more to do.
    const uint64_t first = req->seq;
    if (first >= tx->aborted_from) {
        return;
    }
    tx->aborted_from = first;
    for (size_t i = 0; i < th.awaiting.size(); ++i) {
        Request *other = th.awaiting[i];
        if (other != req && other->awaiting_ack && other->tx == tx && other->seq >= first) {
            other->awaiting_ack = false;
            completeFragment(th, other, status);
        }
    }
    // Tell the target's threads to drop them too (if any was sent).
    if (first >= tx->atomics) {
        return;
    }
    auto conn = tx->conn.lock();
    if (!conn) {
        return;
    }
    PeerAddrs *peer = peerAddrs(th, conn);
    for (size_t t = 0; peer != nullptr && t < conn->remote_proxy_ep_names_.size(); ++t) {
        const fi_addr_t dest = homeAddr(th, *peer, *conn, t);
        if (dest != FI_ADDR_UNSPEC) {
            sendCtl(th,
                    PendingCtl{dest, wire::msgType::RING_ABORT, 0, NIXL_SUCCESS, tx->key, first});
        }
    }
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
        // A put completes (to the GPU) only once its data is placed at the target.
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
        return fi_writemsg(sendRes(th, pp.rail).ep, &msg, flags);
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
    if (!th.prof) {
        return tryPost(th, pp);
    }
    const uint64_t start = nixlLibfabricProxyProfile::now();
    const ssize_t rc = tryPost(th, pp);
    const uint64_t end = nixlLibfabricProxyProfile::now();
    th.prof->add(nixlLibfabricProxyProfile::POST_CALL, end - start);
    RailRes &rr = sendRes(th, pp.rail);
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
    if (pp.kind == PendingPost::Kind::WRITE) {
        if (th.tx[pp.rail].ep != nullptr) {
            if (std::find(th.tx_poll_rails.begin(), th.tx_poll_rails.end(), pp.rail) ==
                th.tx_poll_rails.end()) {
                th.tx_poll_rails.push_back(pp.rail); // its write completions arrive there
            }
        } else if (!th.polled[pp.rail]) {
            th.polled[pp.rail] = true;
            th.poll_rails.push_back(pp.rail);
        }
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
    while (!th.pending_ctls.empty() && postCtl(th, th.pending_ctls.front())) {
        th.pending_ctls.pop_front();
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
        --req->ring->outstanding;
    }
    if (req->tx != nullptr) {
        --req->tx->inflight;
        ++req->tx->completions;
    }
    if (req->status != NIXL_SUCCESS) {
        failRing(th, req, req->status);
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
        if (TxRing *tx = req->tx; tx != nullptr && tx->completions != tx->scanned_completions) {
            tx->scanned_completions = tx->completions; // its ring still moves
            tx->progress_ns = now_ns;
        }
        const uint64_t progress = req->tx != nullptr ? req->tx->progress_ns : 0;
        if (now_ns >= req->ack_deadline && now_ns >= progress + kAckTimeoutNs) {
            NIXL_ERROR << "EFA proxy: no ack for an atomicAdd to " << std::hex
                       << req->msg.remote_addr << std::dec << " and no progress on its ring for "
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
    const uint64_t start = th.prof ? nowNs() : 0;
    th.sweep_imm = 0;
    // By index: completions may post, which may extend the list.
    for (size_t i = 0; i < th.poll_rails.size(); ++i) {
        const uint32_t r = th.poll_rails[i];
        pollCq(th, th.rails[r].cq, r, th.rails[r].cq_reads);
    }
    for (size_t i = 0; i < th.tx_poll_rails.size(); ++i) {
        const uint32_t r = th.tx_poll_rails[i];
        pollCq(th, th.tx[r].cq, r, th.tx[r].cq_reads);
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
    for (;;) {
        const uint64_t read_start = th.prof ? nixlLibfabricProxyProfile::now() : 0;
        const ssize_t n = fi_cq_read(cq, entries, kCqBatch);
        if (th.prof) {
            const uint64_t took = nixlLibfabricProxyProfile::now() - read_start;
            if (took > kSlowCallNs) {
                NIXL_INFO << "EFA proxy profile: slow cq read: thread " << th.id << " rail " << rail
                          << " read #" << reads << " took " << took / 1000 << " us";
            }
            ++reads;
        }
        if (n > 0) {
            th.events += static_cast<uint64_t>(n);
            const uint64_t decode_start = th.prof ? nowNs() : 0;
            uint64_t imm = 0;
            for (ssize_t i = 0; i < n; ++i) {
                if (entries[i].flags & FI_REMOTE_CQ_DATA) {
                    // A put landed on a rail this thread receives on: count it.
                    ++imm;
                    handleImm(th, static_cast<uint32_t>(entries[i].data));
                    continue;
                }
                // Each context type starts with its OpCtx.
                auto *op = static_cast<OpCtx *>(entries[i].op_context);
                switch (op->kind) {
                case OpCtx::Kind::RECV:
                    handleRecv(th, reinterpret_cast<RecvBuf *>(op), entries[i].len);
                    break;
                case OpCtx::Kind::CTL:
                    reinterpret_cast<CtlBuf *>(op)->dest = FI_ADDR_UNSPEC;
                    th.free_ctls.push_back(reinterpret_cast<CtlBuf *>(op));
                    break;
                case OpCtx::Kind::FRAG: {
                    auto *f = reinterpret_cast<Request::FragCtx *>(op);
                    if (f->owner->inject_cq_error) {
                        f->owner->inject_cq_error = false;
                        NIXL_WARN << "EFA proxy: injected completion error";
                        failFragment(th, f->owner, NIXL_ERR_BACKEND);
                    } else {
                        completeFragment(th, f->owner, NIXL_SUCCESS);
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
                reportCqError(th, rail, err.err);
                auto *op = static_cast<OpCtx *>(err.op_context);
                if (op == nullptr) {
                    continue;
                }
                switch (op->kind) {
                case OpCtx::Kind::RECV:
                    postRecv(th, reinterpret_cast<RecvBuf *>(op));
                    break;
                case OpCtx::Kind::CTL:
                    // This message is lost: an ack's atomicAdd fails by timeout at the
                    // sender; an abort's adds expire at the target.
                    reinterpret_cast<CtlBuf *>(op)->dest = FI_ADDR_UNSPEC;
                    th.free_ctls.push_back(reinterpret_cast<CtlBuf *>(op));
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
    // transfers on the same rails. With nothing of its own outstanding and no puts
    // landing, a thread polls at most every idle_poll_ns_.
    const bool busy = th.free_reqs.size() != th.reqs.size() || th.retry_count != 0 ||
        !th.pending_ctls.empty() || !th.recv_retry.empty() || !th.owned.empty() || th.rx_active ||
        !th.parked_rings.empty();
    bool poll = true;
    uint64_t now = 0;
    if (!busy && idle_poll_ns_ != 0) {
        now = nowNs();
        poll = now >= th.next_idle_poll;
        if (poll) {
            th.next_idle_poll = now + idle_poll_ns_;
        }
    }
    if (poll) {
        if (th.prof) {
            const uint64_t start = nixlLibfabricProxyProfile::now();
            if (th.last_pass != 0) {
                th.prof->add(nixlLibfabricProxyProfile::PASS_PERIOD, start - th.last_pass);
            }
            th.last_pass = start;
        }
        drainRetries(th);
        pollCqs(th);
        if (th.prof) {
            th.prof->add(nixlLibfabricProxyProfile::CQ_POLL,
                         nixlLibfabricProxyProfile::now() - th.last_pass);
        }
        if (!th.awaiting.empty() && ++th.ack_scan % kAckScanInterval == 0) {
            expireAcks(th, nowNs());
        }
    }
    if (!th.owned.empty()) {
        drainDeferred(th);
    }
    if (!th.parked_rings.empty()) {
        replayParked(th, now != 0 ? now : nowNs());
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
    // Called after every owned ring has drained and published; wait briefly for
    // any trailing completion.
    Thread &th = ownerOf(channel);
    const uint64_t key = static_cast<uint64_t>(channel) * config_.max_peers + peer;
    auto it = th.rings.find(key);
    if (it == th.rings.end()) {
        return NIXL_SUCCESS;
    }
    const auto deadline = std::chrono::steady_clock::now() + kQuiesceTimeout;
    while (!it->second.idle() && std::chrono::steady_clock::now() < deadline) {
        drainRetries(th);
        pollCqs(th);
        expireAcks(th, nowNs());
        if (!th.owned.empty()) {
            drainDeferred(th);
        }
        if (!th.parked_rings.empty()) {
            replayParked(th, nowNs());
        }
    }
    if (!it->second.idle()) {
        NIXL_ERROR << "EFA proxy: ring (" << channel << ", " << peer << ") did not quiesce in "
                   << kQuiesceTimeout.count() << " s";
        return NIXL_ERR_BACKEND;
    }
    th.rings.erase(it);
    return NIXL_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Control messages: atomicAdd records, acks, ring aborts (libfabric_proxy_wire.h)
 * ------------------------------------------------------------------------- */

nixl_status_t
nixlLibfabricProxy::postRecv(Thread &th, RecvBuf *buf) {
    struct iovec iov = {&buf->msg, sizeof(wire::anyMsg)};
    void *desc = th.recv_desc;
    struct fi_msg msg = {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = FI_ADDR_UNSPEC;
    msg.context = &buf->op.ctx;
    const ssize_t rc = fi_recvmsg(th.ctl_ep, &msg, 0);
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
    } else if (msg.hdr.type == wire::msgType::RING_ABORT && len >= sizeof(wire::ringAbortMsg)) {
        handleAbort(th, msg.abort);
    } else {
        NIXL_ERROR << "EFA proxy: dropped a malformed " << len << "-byte message of type "
                   << static_cast<int>(msg.hdr.type);
    }
    postRecv(th, buf);
}

void
nixlLibfabricProxy::handleAtomic(Thread &th, const wire::atomicAddMsg &msg) {
    RxRing *ring = rxRing(th, msg.ring);
    if (msg.seq >= ring->aborted_from.load(std::memory_order_acquire)) {
        return; // the sender aborted the ring and already failed this atomicAdd
    }
    const fi_addr_t reply = replyAddr(th, msg);
    if (reply == FI_ADDR_UNSPEC) {
        return; // logged; the sender fails the atomicAdd when its ack times out
    }
    RxOwned &owned = th.owned[ring];
    owned.ring = ring;
    if (msg.seq < ring->applied.load(std::memory_order_acquire) ||
        !owned.adds.emplace(msg.seq, RxOwned::Add{msg, reply, nowNs()}).second) {
        // A seq already applied or already waiting (a broken sender; the transport
        // delivers once): fail it rather than let it block the ring's later adds.
        NIXL_ERROR << "EFA proxy: duplicate atomicAdd record (ring " << std::hex << msg.ring
                   << std::dec << ", seq " << msg.seq << "); failing it";
        if (owned.adds.empty()) {
            th.owned.erase(ring);
        }
        sendCtl(th,
                PendingCtl{reply, wire::msgType::ATOMIC_ACK, msg.token, NIXL_ERR_INVALID_PARAM, 0, 0});
        return;
    }
    // Applied by drainDeferred() in this same pass, batched with the others.
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

void
nixlLibfabricProxy::handleAbort(Thread &th, const wire::ringAbortMsg &abort) {
    RxRing *ring = rxRing(th, abort.ring);
    uint64_t current = ring->aborted_from.load(std::memory_order_relaxed);
    while (abort.first_seq < current &&
           !ring->aborted_from.compare_exchange_weak(
               current, abort.first_seq, std::memory_order_acq_rel)) {}
    auto it = th.owned.find(ring);
    if (it != th.owned.end()) {
        auto &adds = it->second.adds;
        adds.erase(adds.lower_bound(abort.first_seq), adds.end());
        if (adds.empty()) {
            th.owned.erase(it);
        }
    }
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
    const ci::EpName padded = ci::paddedName(std::string(name));
    fi_addr_t addr = FI_ADDR_UNSPEC;
    if (fi_av_insert(th.ctl_av, padded.data(), 1, &addr, 0, nullptr) != 1) {
        NIXL_ERROR << "EFA proxy: fi_av_insert failed for an atomicAdd sender";
        return FI_ADDR_UNSPEC;
    }
    th.reply_addrs.emplace(std::string(name), addr);
    return addr;
}

void
nixlLibfabricProxy::sendCtl(Thread &th, const PendingCtl &ctl) {
    if (!th.pending_ctls.empty() || !postCtl(th, ctl)) {
        th.pending_ctls.push_back(ctl); // keep them in order; drained from progress()
    }
}

bool
nixlLibfabricProxy::postCtl(Thread &th, const PendingCtl &ctl) {
    if (th.free_ctls.empty()) {
        return false;
    }
    CtlBuf *buf = th.free_ctls.back();
    size_t len = 0;
    if (ctl.type == wire::msgType::ATOMIC_ACK) {
        buf->msg.ack = wire::atomicAckMsg{};
        buf->msg.ack.hdr = wire::msgHeader{wire::kVersion, wire::msgType::ATOMIC_ACK, 0};
        buf->msg.ack.token = ctl.token;
        buf->msg.ack.status = static_cast<int32_t>(ctl.status);
        len = sizeof(wire::atomicAckMsg);
    } else {
        buf->msg.abort = wire::ringAbortMsg{};
        buf->msg.abort.hdr = wire::msgHeader{wire::kVersion, wire::msgType::RING_ABORT, 0};
        buf->msg.abort.ring = ctl.ring;
        buf->msg.abort.first_seq = ctl.first_seq;
        len = sizeof(wire::ringAbortMsg);
    }
    struct iovec iov = {&buf->msg, len};
    void *desc = th.ctl_desc;
    struct fi_msg msg = {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = ctl.dest;
    msg.context = &buf->op.ctx;
    const ssize_t rc = fi_sendmsg(th.ctl_ep, &msg, FI_COMPLETION);
    if (rc == -FI_EAGAIN) {
        return false;
    }
    if (rc == 0) {
        buf->dest = ctl.dest;
        th.free_ctls.pop_back();
    } else {
        NIXL_ERROR << "EFA proxy: control send failed: " << fi_strerror(-rc)
                   << (ctl.type == wire::msgType::ATOMIC_ACK ?
                           "; the sender's atomicAdd fails when its ack times out" :
                           "; the target's waiting atomicAdds expire");
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * Receive side: counting puts, applying atomicAdds in ring order
 * ------------------------------------------------------------------------- */

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
    rxRing(th, wire::immRing(imm))
        ->received[wire::immSlot(imm)]
        .fetch_add(1, std::memory_order_release);
}

void
nixlLibfabricProxy::drainDeferred(Thread &th) {
    using Adds = std::map<uint64_t, RxOwned::Add>;
    // Rounds: for every ring, take the run of consecutive atomicAdds this thread
    // owns from the ring's next one (seq == applied) whose epoch slots counted their
    // puts; flush once for all of them where needed (a flush covers only the puts
    // counted before it); apply them in ring order. An add of the ring owned by
    // another thread ends the run; once that thread applied it, a later round (or
    // pass) continues.
    std::vector<std::pair<RxOwned *, Adds::iterator>> ready;
    const uint64_t deadline = nowNs() + kDrainBudgetNs;
    bool over_budget = false;
    while (!over_budget) {
        ready.clear();
        for (auto it = th.owned.begin(); it != th.owned.end();) {
            RxOwned &owned = it->second;
            RxRing &ring = *owned.ring;
            const uint64_t aborted = ring.aborted_from.load(std::memory_order_acquire);
            owned.adds.erase(owned.adds.lower_bound(aborted), owned.adds.end());
            if (owned.adds.empty()) {
                it = th.owned.erase(it);
                continue;
            }
            uint64_t next = ring.applied.load(std::memory_order_acquire);
            for (auto add = owned.adds.begin(); add != owned.adds.end() && add->first == next;
                 ++add, ++next) {
                if (ring.received[next % wire::kEpochs].load(std::memory_order_acquire) <
                    add->second.msg.expected_puts) {
                    break;
                }
                ready.emplace_back(&owned, add);
            }
            ++it;
        }
        if (ready.empty()) {
            break;
        }
        if (rx_flush_) {
            const nixl_status_t status = flushRdmaWrites(th);
            ++th.rx_stats.flushes;
            for (auto &[owned, add] : ready) {
                if (status != NIXL_SUCCESS) {
                    owned->ring->error.store(status, std::memory_order_relaxed);
                }
            }
        }
        const uint64_t now = nowNs();
        for (auto &[owned, add] : ready) {
            RxRing &ring = *owned->ring;
            const wire::atomicAddMsg &msg = add->second.msg;
            if (th.prof) {
                th.prof->add(nixlLibfabricProxyProfile::RX_ORDER_WAIT, now - add->second.arrived);
            }
            auto status = static_cast<nixl_status_t>(ring.error.load(std::memory_order_relaxed));
            if (status == NIXL_SUCCESS) {
                const uint64_t start = th.prof ? nowNs() : 0;
                status = applyAtomic(th, msg.remote_addr, msg.value);
                if (th.prof) {
                    th.prof->add(nixlLibfabricProxyProfile::ATOMIC_APPLY, nowNs() - start);
                }
                if (status != NIXL_SUCCESS) {
                    NIXL_ERROR << "EFA proxy: atomicAdd to " << std::hex << msg.remote_addr
                               << std::dec << " failed with status " << status
                               << "; reporting it to the sender, later ones on its ring fail too";
                    ring.error.store(status, std::memory_order_relaxed);
                }
            }
            ++th.events;
            // The ring's next atomicAdd may be another thread's: publish the add first.
            ring.applied.store(msg.seq + 1, std::memory_order_release);
            sendCtl(th,
                    PendingCtl{add->second.reply, wire::msgType::ATOMIC_ACK, msg.token, status, 0, 0});
            owned->adds.erase(add);
            if (nowNs() >= deadline) {
                over_budget = true; // the batch's rest stays waiting, still in order
                break;
            }
        }
    }

    // Atomics whose ring made no progress for too long: their sender has failed them.
    const uint64_t now = nowNs();
    if (now < th.next_deferred_scan) {
        return;
    }
    th.next_deferred_scan = now + kDeferredScanNs;
    for (auto it = th.owned.begin(); it != th.owned.end();) {
        RxOwned &owned = it->second;
        const RxRing &ring = *owned.ring;
        auto &adds = owned.adds;
        const uint64_t applied = ring.applied.load(std::memory_order_acquire);
        uint64_t received = 0;
        for (const auto &slot : ring.received) {
            received += slot.load(std::memory_order_relaxed);
        }
        if (owned.progress_ns == 0 || applied != owned.seen_applied ||
            received != owned.seen_received) {
            owned.seen_applied = applied;
            owned.seen_received = received;
            owned.progress_ns = now;
        }
        if (now - owned.progress_ns < kDeferredTimeoutNs) {
            ++it;
            continue;
        }
        const auto &head = *adds.begin();
        NIXL_ERROR << "EFA proxy: " << adds.size() << " atomicAdd(s) of ring " << std::hex
                   << head.second.msg.ring << std::dec << " waited " << kDeferredTimeoutNs / 1000000000
                   << " s without progress (next seq " << head.first << ": slot count "
                   << ring.received[head.first % wire::kEpochs].load(std::memory_order_relaxed)
                   << " of " << head.second.msg.expected_puts << ", applied " << applied
                   << "); dropping them";
        for (const auto &[seq, add] : adds) {
            sendCtl(th,
                    PendingCtl{add.reply,
                               wire::msgType::ATOMIC_ACK,
                               add.msg.token,
                               NIXL_ERR_REMOTE_DISCONNECT,
                               0,
                               0});
        }
        it = th.owned.erase(it);
    }
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
nixlLibfabricProxy::flushRdmaWrites(Thread &th) {
#ifdef HAVE_CUDA
    // The puts may target any GPU with registered memory, whatever the counter's
    // memory: flush them all (NCCL flushes where the data lands, too).
    std::vector<int> devices;
    {
        std::lock_guard<std::mutex> lock(th.regions_lock);
        devices = vram_devices_;
    }
    const uint64_t start = th.prof ? nowNs() : 0;
    for (const int device : devices) {
        // Always set: the runtime also switches this thread's device (th.cuda_dev
        // may be stale), and a flush must target the right GPU.
        if (cudaSetDevice(device) != cudaSuccess) {
            NIXL_ERROR << "EFA proxy: cudaSetDevice(" << device << ") failed before a flush";
            return NIXL_ERR_BACKEND;
        }
        th.cuda_dev = device;
        if (cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                               cudaFlushGPUDirectRDMAWritesToOwner) !=
            cudaSuccess) {
            NIXL_ERROR << "EFA proxy: cudaDeviceFlushGPUDirectRDMAWrites failed on device "
                       << device;
            return NIXL_ERR_BACKEND;
        }
    }
    if (th.prof) {
        th.prof->add(nixlLibfabricProxyProfile::RX_FLUSH, nowNs() - start);
    }
#else
    static_cast<void>(th);
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
    if (is_vram &&
        std::find(vram_devices_.begin(), vram_devices_.end(), device_id) == vram_devices_.end()) {
        vram_devices_.push_back(device_id);
    }
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
    const uintptr_t start = victim->first;
    regions_.erase(victim);
    for (auto &th : thread_state_) { // their owners hold no region lock: none is in use
        for (auto it = th->tx_mrs.begin(); it != th->tx_mrs.end();) {
            if (it->first.second == start) {
                fi_close(&it->second->fid);
                it = th->tx_mrs.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Keep the counter pages another registration still covers.
    counters_->dropRange(
        addr, len, [this](uintptr_t page, size_t size) { return regionOverlaps(page, size); });
}


/* ---------------------------------------------------------------------------
 * Private send domains (efa_proxy_tx_domain=private)
 * ------------------------------------------------------------------------- */

nixlLibfabricProxy::RailRes &
nixlLibfabricProxy::sendRes(Thread &th, size_t rail) const {
    return th.tx[rail].ep != nullptr ? th.tx[rail] : th.rails[rail];
}

int
nixlLibfabricProxy::openTxRail(Thread &th, size_t r) {
    RailRes &rr = th.tx[r];
    const nixlLibfabricRail &rail = engine_.rail_manager_.getRail(r);
    rr.info = fi_dupinfo(rail.getRailInfo());
    if (!rr.info) {
        return -FI_ENOMEM;
    }
    rr.info->tx_attr->size = std::min(rr.info->tx_attr->size, kTxSize);
    rr.info->rx_attr->size = std::min(rr.info->rx_attr->size, kDataRxSize);
    rr.virt_addr = (rr.info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;
    int ret = fi_domain(rail.getFabric(), rr.info, &rr.domain, nullptr);
    struct fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_DATA;
    cq_attr.wait_obj = FI_WAIT_NONE;
    cq_attr.size = kCqSize;
    struct fi_av_attr av_attr = {};
    if (!ret) {
        ret = fi_cq_open(rr.domain, &cq_attr, &rr.cq, nullptr);
    }
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
    if (ret && rr.ep) {
        fi_close(&rr.ep->fid);
        rr.ep = nullptr; // sendRes() falls back to the shared domain
    }
    return ret;
}

void *
nixlLibfabricProxy::txDesc(Thread &th, size_t rail, uint64_t addr, void *shared_desc) {
    RailRes &rr = th.tx[rail];
    if (rr.ep == nullptr) {
        return shared_desc;
    }
    std::lock_guard<std::mutex> lock(th.regions_lock);
    // The registration holding addr (the put's start; NIXL registered the whole put).
    auto it = regions_.upper_bound(addr);
    while (it != regions_.begin()) {
        --it;
        if (addr < it->first + it->second.len) {
            break;
        }
    }
    if (regions_.empty() || addr < it->first || addr >= it->first + it->second.len) {
        NIXL_ERROR << "EFA proxy: no registration holds the put source " << std::hex << addr;
        return nullptr;
    }
    const auto key = std::make_pair(static_cast<uint32_t>(rail), it->first);
    auto found = th.tx_mrs.find(key);
    if (found == th.tx_mrs.end()) {
        struct iovec iov = {reinterpret_cast<void *>(it->first), it->second.len};
        struct fi_mr_attr attr = {};
        attr.mr_iov = &iov;
        attr.iov_count = 1;
        attr.access = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        attr.iface = it->second.is_vram ? FI_HMEM_CUDA : FI_HMEM_SYSTEM;
        if (it->second.is_vram) {
            attr.device.cuda = it->second.device_id;
        }
        struct fid_mr *mr = nullptr;
        const int ret = fi_mr_regattr(rr.domain, &attr, 0, &mr);
        if (ret) {
            NIXL_ERROR << "EFA proxy: registration in thread " << th.id << "'s domain on rail "
                       << rail << " failed: " << fi_strerror(-ret);
            return nullptr;
        }
        found = th.tx_mrs.emplace(key, mr).first;
    }
    return fi_mr_desc(found->second);
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
    const auto &data_rails = conn->remote_proxy_data_rails_;
    pa.data_ep.assign(rails_, std::vector<fi_addr_t>(data_rails.size(), FI_ADDR_UNSPEC));
    size_t max_rail = 0;
    for (uint32_t r : data_rails) {
        max_rail = std::max<size_t>(max_rail, r); // bounded by ci::kMaxRails (parse())
    }
    pa.data_index.assign(data_rails.empty() ? 0 : max_rail + 1, -1);
    for (size_t i = 0; i < data_rails.size(); ++i) {
        pa.data_index[data_rails[i]] = static_cast<int>(i);
    }
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
        fi_av_insert(sendRes(th, rail).av,
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
                             size_t thread) {
    fi_addr_t &addr = pa.home[thread];
    if (addr == FI_ADDR_UNSPEC &&
        fi_av_insert(th.ctl_av, conn.remote_proxy_ep_names_[thread].data(), 1, &addr, 0, nullptr) !=
            1) {
        addr = FI_ADDR_UNSPEC;
        NIXL_ERROR << "EFA proxy: fi_av_insert failed for " << conn.remoteAgent_ << " proxy thread "
                   << thread;
    }
    return addr;
}

fi_addr_t
nixlLibfabricProxy::dataAddr(Thread &th,
                             PeerAddrs &pa,
                             const nixlLibfabricConnection &conn,
                             size_t rail,
                             size_t data_index) {
    fi_addr_t &addr = pa.data_ep[rail][data_index];
    if (addr == FI_ADDR_UNSPEC &&
        fi_av_insert(sendRes(th, rail).av,
                     conn.remote_proxy_data_ep_names_[data_index].data(),
                     1,
                     &addr,
                     0,
                     nullptr) != 1) {
        addr = FI_ADDR_UNSPEC;
        NIXL_ERROR << "EFA proxy: fi_av_insert failed for " << conn.remoteAgent_
                   << " data EP on its rail " << conn.remote_proxy_data_rails_[data_index];
    }
    return addr;
}

void
nixlLibfabricProxy::dropPeer(Thread &th, PeerAddrs &pa) {
    // Only for dead connections: their views were quiesced, so nothing is in flight.
    // The provider returns one entry per address, so a live connection to the same
    // agent (a reconnect) may share an entry: keep those.
    const auto shared = [&](size_t rail, fi_addr_t addr) {
        for (const auto &[conn, other] : th.peers) {
            if (&other == &pa || other.conn.expired()) {
                continue;
            }
            const auto has = [addr](const std::vector<fi_addr_t> &v) {
                return std::find(v.begin(), v.end(), addr) != v.end();
            };
            if ((rail < other.rail_ep.size() && has(other.rail_ep[rail])) ||
                (rail < other.data_ep.size() && has(other.data_ep[rail]))) {
                return true;
            }
        }
        return false;
    };
    for (size_t r = 0; r < pa.rail_ep.size(); ++r) {
        for (fi_addr_t &addr : pa.rail_ep[r]) {
            if (addr != FI_ADDR_UNSPEC && !shared(r, addr)) {
                fi_av_remove(sendRes(th, r).av, &addr, 1, 0);
            }
        }
    }
    for (size_t r = 0; r < pa.data_ep.size(); ++r) {
        for (fi_addr_t &addr : pa.data_ep[r]) {
            if (addr != FI_ADDR_UNSPEC && !shared(r, addr)) {
                fi_av_remove(sendRes(th, r).av, &addr, 1, 0);
            }
        }
    }
    for (fi_addr_t &addr : pa.home) {
        if (addr == FI_ADDR_UNSPEC) {
            continue;
        }
        // An ack route (acks queued or in flight), a live peer, or a control
        // message still queued or in flight to it (a ring abort) may use it: removed,
        // the entry could be reused for another agent while that message is sent.
        bool used = false;
        for (const auto &route : th.reply_addrs) {
            used = used || route.second == addr;
        }
        for (const auto &ctl : th.pending_ctls) {
            used = used || ctl.dest == addr;
        }
        for (const auto &buf : th.ctls) {
            used = used || buf.dest == addr;
        }
        for (const auto &[conn, other] : th.peers) {
            if (&other != &pa && !other.conn.expired()) {
                used = used || std::find(other.home.begin(), other.home.end(), addr) !=
                        other.home.end();
            }
        }
        if (!used) {
            fi_av_remove(th.ctl_av, &addr, 1, 0);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Connection info (libfabric_proxy_conninfo.h)
 * ------------------------------------------------------------------------- */

std::string
nixlLibfabricProxy::serializeConnInfo() const {
    ci::ProxyEps eps;
    for (const auto &th : thread_state_) {
        eps.home.push_back(th->home_name);
    }
    // Each receive rail with the data EP of the thread that polls it.
    for (size_t i = 0; i < rx_rails_.size() && !thread_state_.empty(); ++i) {
        const uint32_t owner = wire::railThread(static_cast<uint32_t>(i), threads_);
        eps.data_rails.push_back(rx_rails_[i]);
        eps.data.push_back(thread_state_[owner]->rails[rx_rails_[i]].name);
    }
    eps.incarnation = incarnation_;
    return ci::serialize(eps);
}

std::string
nixlLibfabricProxy::joinConnInfo(const std::string &engine_part, const std::string &proxy_part) {
    return ci::join(engine_part, proxy_part);
}

void
nixlLibfabricProxy::splitConnInfo(const std::string &in,
                                  std::string &engine_part,
                                  std::string &proxy_part) {
    ci::split(in, engine_part, proxy_part);
}

nixl_status_t
nixlLibfabricProxy::parseConnInfo(const std::string &blob, nixlLibfabricConnection &conn) {
    ci::ProxyEps eps;
    const nixl_status_t status = ci::parse(blob, eps);
    conn.remote_proxy_ep_names_.clear();
    conn.remote_proxy_data_ep_names_.clear();
    for (const std::string &name : eps.home) {
        conn.remote_proxy_ep_names_.push_back(ci::paddedName(name));
    }
    for (const std::string &name : eps.data) {
        conn.remote_proxy_data_ep_names_.push_back(ci::paddedName(name));
    }
    conn.remote_proxy_data_rails_ = eps.data_rails;
    conn.remote_proxy_incarnation_ = eps.incarnation;
    return status;
}

#endif // HAVE_NIXL_DEVICE_API
