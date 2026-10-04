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

struct nixlLibfabricProxy::RingFence {
    nixlLibfabricRingFence<Request> order;
    uint32_t next_rail = 0; // unstriped puts rotate over the buffer's rails
};

struct nixlLibfabricProxy::RecvBuf {
    OpCtx op; // must stay first
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
};

struct nixlLibfabricProxy::PeerAddrs {
    // Expired once the connection is gone; its address may then be reused.
    std::weak_ptr<const nixlLibfabricConnection> conn;
    // Inserted into this thread's AVs on first use; FI_ADDR_UNSPEC until then.
    // rail_ep[local_rail][remote_ep] -> remote engine EP in that rail's AV
    std::vector<std::vector<fi_addr_t>> rail_ep;
    // home[remote_thread] -> remote proxy home EP in this thread's home-rail AV
    std::vector<fi_addr_t> home;
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
    inject_ = Inject::fromEnv();
    counters_ = std::make_unique<CounterMap>(!(inject_ && inject_->no_gdrcopy));
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
            rr.info->rx_attr->size = std::min(rr.info->rx_attr->size, kIdleRxSize);
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
              << " rail; idle poll every " << idle_poll_ns_ / 1000 << " us";
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
    }
    for (auto &th : thread_state_) {
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
            fence.order.complete(fence.order.addPut(), status);
            releaseFence(th, fence);
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
        pp.dest = railAddr(th, *peer, *remote->conn_, rail, rep);
        pp.raddr = th.rails[rail].virt_addr ? target : target - remote->remote_buf_addr_;
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
    req->epoch = req->fence->order.addPut();
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
    // Completes on its send and on the owner's ack (the add applied, or why not).
    req->frags_left = 2;

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
        const uint64_t flags = FI_COMPLETION | FI_DELIVERY_COMPLETE;
        struct fi_rma_iov rma = {pp.raddr, pp.len, pp.rkey};
        struct fi_msg_rma msg = {};
        msg.msg_iov = &iov;
        msg.desc = &desc;
        msg.iov_count = 1;
        msg.addr = pp.dest;
        msg.rma_iov = &rma;
        msg.rma_iov_count = 1;
        msg.context = &pp.fctx->op.ctx;
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
    // By index: completions release atomics, whose posts may extend the list.
    for (size_t i = 0; i < th.poll_rails.size(); ++i) {
        const uint32_t r = th.poll_rails[i];
        pollCq(th, th.rails[r].cq, r, th.rails[r].cq_reads);
    }
    pollCq(th, th.ctl_cq, static_cast<uint32_t>(rails_), th.ctl_cq_reads);
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
                for (ssize_t i = 0; i < n; ++i) {
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
    // Every CQ read takes its rail domain's lock (FI_THREAD_SAFE) and progresses
    // the whole domain, so a thread spinning on idle CQs starves the engine's host
    // transfers on the same rails. With nothing of its own outstanding, a thread
    // polls at most every idle_poll_ns_: only an incoming atomicAdd can be waiting.
    const bool busy = th.free_reqs.size() != th.reqs.size() || th.retry_count != 0 ||
        !th.pending_acks.empty() || !th.recv_retry.empty();
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
    while (!it->second.order.idle() && std::chrono::steady_clock::now() < deadline) {
        drainRetries(th);
        pollCqs(th);
        expireAcks(th, nowNs());
    }
    if (!it->second.order.idle()) {
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
    } else {
        NIXL_ERROR << "EFA proxy: dropped a malformed " << len << "-byte message of type "
                   << static_cast<int>(msg.hdr.type);
    }
    postRecv(th, buf);
}

void
nixlLibfabricProxy::handleAtomic(Thread &th, const wire::atomicAddMsg &msg) {
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
    const fi_addr_t dest = replyAddr(th, msg);
    if (dest == FI_ADDR_UNSPEC) {
        return; // logged; the sender fails the atomicAdd when its ack times out
    }
    sendAck(th, PendingAck{dest, msg.token, status});
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
    std::vector<nixlLibfabricProxyConnInfo::EpName> home_eps;
    home_eps.reserve(thread_state_.size());
    for (const auto &th : thread_state_) {
        home_eps.push_back(th->home_name);
    }
    return nixlLibfabricProxyConnInfo::serialize(home_eps);
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
    return nixlLibfabricProxyConnInfo::parse(blob, conn.remote_proxy_ep_names_);
}

#endif // HAVE_NIXL_DEVICE_API
