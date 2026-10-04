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
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

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
/** Posted receive buffers per proxy thread for incoming atomicAdd records. */
constexpr size_t kRecvPoolSize = 1024;
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
 * domain lock). Writes need no receive buffers, so only a thread's home EP
 * (atomicAdd records) gets a real receive queue; posts beyond the transmit queue
 * wait in the retry queue.
 */
constexpr size_t kTxSize = 1024;
constexpr size_t kHomeRxSize = 1024;
constexpr size_t kIdleRxSize = 64;

/** Deterministic mixer: every sender must pick the same owner for a counter. */
uint64_t
mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

} // namespace

/* ---------------------------------------------------------------------------
 * Internal state
 * ------------------------------------------------------------------------- */

struct nixlLibfabricProxy::Request {
    /** One libfabric context per posted operation; op_context points here. */
    struct FragCtx {
        struct fi_context2 ctx; // must stay first
        Request *owner;
    };

    FragCtx frag[kMaxStripes];
    uint32_t thread = 0;
    uint16_t frags_left = 0;
    bool in_use = false;
    nixl_status_t status = NIXL_IN_PROG;
    RingFence *fence = nullptr;
    nixlLibfabricRingFence<Request>::Epoch *epoch = nullptr;
    fi_addr_t dest = FI_ADDR_UNSPEC; // atomic target (owner thread's home EP)
    AtomicMsg msg{}; // atomic send buffer; the request pool is registered memory
    bool inject_cq_error = false; // NIXL_EFA_PROXY_INJECT: fail its first completion

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
    struct fi_context2 ctx; // must stay first
    AtomicMsg msg;
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
    std::vector<RailRes> rails;
    // CQs worth polling: the home rail (incoming atomics) and every rail this
    // thread has posted on. The rail domains are FI_THREAD_SAFE, so skipping
    // idle CQs also avoids taking their domain locks.
    std::vector<uint32_t> poll_rails;
    std::vector<bool> polled;
    uint32_t passes = 0;
    uint64_t cq_errors = 0; // since the last report
    uint64_t cq_error_report = 0; // time of the last report, 0 before the first
    std::unique_ptr<nixlLibfabricProxyProfile> prof; // NIXL_EFA_PROXY_PROFILE only
    uint64_t last_pass = 0;
    std::array<char, LF_EP_NAME_MAX_LEN> home_name{};

    std::vector<Request> reqs;
    std::vector<Request *> free_reqs;
    struct fid_mr *req_mr = nullptr;
    void *req_desc = nullptr;

    std::vector<RecvBuf> recvs;
    struct fid_mr *recv_mr = nullptr;
    void *recv_desc = nullptr;

    std::deque<PendingPost> retry;
    std::vector<RecvBuf *> recv_retry; // receive buffers the EP could not take yet
    std::unordered_map<uint64_t, RingFence> fences;
    std::unordered_map<const nixlLibfabricConnection *, PeerAddrs> peers;
    int cuda_dev = -1;
};

/**
 * Host access to VRAM counters for the target-side add. GDRCopy maps each 64 KB
 * GPU page on first use; without GDRCopy the add falls back to cudaMemcpy.
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
    std::mutex mutex;
    std::unordered_map<uintptr_t, Mapping> pages;

    CounterMap() {
        gdr = gdr_open();
        if (!gdr) {
            NIXL_WARN << "EFA proxy: gdr_open failed; VRAM atomicAdd falls back to cudaMemcpy";
        }
    }

    ~CounterMap() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto &kv : pages) {
            gdr_unmap(gdr, kv.second.mh, kv.second.bar, kPage);
            gdr_unpin_buffer(gdr, kv.second.mh);
        }
        pages.clear();
        if (gdr) {
            gdr_close(gdr);
        }
    }

    /** Host pointer for one aligned 8-byte counter, or nullptr. */
    uint64_t *
    lookup(uintptr_t addr, gdr_mh_t &mh_out) {
        if (!gdr) {
            return nullptr;
        }
        const uintptr_t page = addr & ~(kPage - 1);
        std::lock_guard<std::mutex> lock(mutex);
        auto it = pages.find(page);
        if (it == pages.end()) {
            Mapping m;
            if (gdr_pin_buffer(gdr, page, kPage, 0, 0, &m.mh) != 0) {
                NIXL_ERROR << "EFA proxy: gdr_pin_buffer failed for page 0x" << std::hex << page;
                return nullptr;
            }
            if (gdr_map(gdr, m.mh, &m.bar, kPage) != 0) {
                gdr_unpin_buffer(gdr, m.mh);
                NIXL_ERROR << "EFA proxy: gdr_map failed for page 0x" << std::hex << page;
                return nullptr;
            }
            gdr_info_t info{};
            gdr_get_info(gdr, m.mh, &info);
            m.off = static_cast<size_t>(info.va - page);
            it = pages.emplace(page, m).first;
        }
        mh_out = it->second.mh;
        char *base = static_cast<char *>(it->second.bar) + it->second.off;
        return reinterpret_cast<uint64_t *>(base + (addr - page));
    }

    void
    dropRange(uintptr_t base, size_t len) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = pages.begin(); it != pages.end();) {
            if (it->first + kPage > base && it->first < base + len) {
                gdr_unmap(gdr, it->second.mh, it->second.bar, kPage);
                gdr_unpin_buffer(gdr, it->second.mh);
                it = pages.erase(it);
            } else {
                ++it;
            }
        }
    }
#else
    void
    dropRange(uintptr_t, size_t) {}
#endif
};

/**
 * Test-only fault injection, enabled by NIXL_EFA_PROXY_INJECT (comma-separated):
 *  - eagain_every=K:  every K-th post attempt returns -FI_EAGAIN without reaching
 *                     libfabric (back-pressure and the retry queue);
 *  - post_error_at=N: the N-th put request fails to post;
 *  - cq_error_at=N:   the N-th put request completes with an error.
 * Put requests are counted from 1 over all proxy threads of this backend.
 */
struct nixlLibfabricProxy::Inject {
    uint64_t eagain_every = 0;
    uint64_t post_error_at = 0;
    uint64_t cq_error_at = 0;
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> puts{0};

    static std::unique_ptr<Inject>
    fromEnv() {
        const char *env = std::getenv("NIXL_EFA_PROXY_INJECT");
        if (env == nullptr || *env == '\0') {
            return nullptr;
        }
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
            } else {
                NIXL_WARN << "EFA proxy: unknown NIXL_EFA_PROXY_INJECT item '" << item << "'";
            }
        }
        NIXL_WARN << "EFA proxy: fault injection enabled: eagain_every=" << inject->eagain_every
                  << " post_error_at=" << inject->post_error_at
                  << " cq_error_at=" << inject->cq_error_at;
        return inject;
    }
};

/* ---------------------------------------------------------------------------
 * Construction and callback table
 * ------------------------------------------------------------------------- */

nixlLibfabricProxy::nixlLibfabricProxy(nixlLibfabricEngine &engine)
    : engine_(engine),
      counters_(std::make_unique<CounterMap>()) {
    const nixl_b_params_t &params = engine_.getCustomParams();
    auto it = params.find("efa_proxy_delivery_complete");
    if (it != params.end()) {
        delivery_complete_ = !(it->second == "0" || it->second == "false");
    }
    it = params.find("efa_proxy_rail_policy");
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
        th->polled.assign(rails_, false);
        th->polled[th->home] = true;
        th->poll_rails.push_back(th->home);

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
            rr.info->rx_attr->size =
                std::min(rr.info->rx_attr->size, r == th->home ? kHomeRxSize : kIdleRxSize);
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
            if (!ret && r == th->home) {
                size_t len = th->home_name.size();
                ret = fi_getname(&rr.ep->fid, th->home_name.data(), &len);
            }
            if (ret) {
                NIXL_ERROR << "EFA proxy: endpoint setup failed for thread " << t << " rail " << r
                           << ": " << fi_strerror(-ret);
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
        for (auto &req : th->reqs) {
            req.thread = t;
            for (auto &f : req.frag) {
                f.owner = &req;
            }
            th->free_reqs.push_back(&req);
        }

        const size_t rx_size = th->rails[th->home].info->rx_attr ?
            th->rails[th->home].info->rx_attr->size :
            kRecvPoolSize;
        th->recvs.resize(std::max<size_t>(1, std::min(kRecvPoolSize, rx_size)));
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
        if (ret) {
            NIXL_ERROR << "EFA proxy: fi_mr_reg failed for thread " << t << ": "
                       << fi_strerror(-ret);
            releaseThread(*th);
            return NIXL_ERR_BACKEND;
        }
        th->req_desc = fi_mr_desc(th->req_mr);
        th->recv_desc = fi_mr_desc(th->recv_mr);

        for (auto &buf : th->recvs) {
            if (postRecv(*th, &buf) == NIXL_ERR_BACKEND) {
                releaseThread(*th);
                return NIXL_ERR_BACKEND;
            }
        }
        thread_state_.push_back(std::move(th));
    }

    const fi_info *home = thread_state_[0]->rails[thread_state_[0]->home].info;
    NIXL_INFO << "EFA proxy: EP queues: transmit " << home->tx_attr->size << ", receive "
              << home->rx_attr->size << " on home EPs and at most " << kIdleRxSize
              << " on the others";
    std::string homes;
    for (const auto &th : thread_state_) {
        homes += (homes.empty() ? "" : ",") + std::to_string(th->home);
    }
    NIXL_INFO << "EFA proxy: " << threads_ << " thread(s) x " << rails_
              << " rail(s) = " << threads_ * rails_
              << " endpoint(s); delivery_complete=" << delivery_complete_ << "; home rails "
              << homes << "; small puts per " << (rail_per_thread_ ? "thread" : "ring") << " rail";
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
    for (auto &rr : th.rails) {
        close(rr.ep ? &rr.ep->fid : nullptr);
        rr.ep = nullptr;
    }
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
    close(th.req_mr ? &th.req_mr->fid : nullptr);
    close(th.recv_mr ? &th.recv_mr->fid : nullptr);
    th.req_mr = nullptr;
    th.recv_mr = nullptr;
    th.retry.clear();
    th.recv_retry.clear();
    th.poll_rails.clear();
    th.fences.clear();
    th.peers.clear();
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
    req->frags_left = 0;
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

nixl_status_t
nixlLibfabricProxy::submit(const nixlBackendProxySubmission &sub,
                           nixlBackendProxyRequest &request) {
    const uint32_t t = sub.channel_id % threads_;
    Thread &th = *thread_state_[t];

    Request *req = allocRequest(th);
    if (!req) {
        NIXL_ERROR << "EFA proxy: request pool exhausted on thread " << t;
        return NIXL_ERR_BACKEND;
    }
    const uint64_t ring_key =
        static_cast<uint64_t>(sub.channel_id) * config_.max_peers + sub.peer_index;
    req->fence = &th.fences[ring_key];
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
        return status;
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
    const size_t owner = mix64(sub.remote.desc.addr >> 3) % peer->home.size();
    req->dest = homeAddr(th, *peer, *remote->conn_, owner);
    if (req->dest == FI_ADDR_UNSPEC) {
        return NIXL_ERR_BACKEND;
    }
    NIXL_DEBUG << "EFA proxy: atomicAdd " << sub.value << " to 0x" << std::hex
               << sub.remote.desc.addr << std::dec << " -> " << remote->conn_->remoteAgent_
               << " proxy thread " << owner << " (fi_addr " << req->dest << ")";
    req->msg = AtomicMsg{sub.remote.desc.addr, sub.value, {0, 0}};
    req->frags_left = 1;

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
    PendingPost pp{};
    pp.kind = PendingPost::Kind::SEND;
    pp.rail = th.home;
    pp.fctx = &req->frag[0];
    pp.local = &req->msg;
    pp.len = sizeof(AtomicMsg);
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
    const uint64_t flags = FI_COMPLETION | (delivery_complete_ ? FI_DELIVERY_COMPLETE : 0);
    if (pp.kind == PendingPost::Kind::WRITE) {
        struct fi_rma_iov rma = {pp.raddr, pp.len, pp.rkey};
        struct fi_msg_rma msg = {};
        msg.msg_iov = &iov;
        msg.desc = &desc;
        msg.iov_count = 1;
        msg.addr = pp.dest;
        msg.rma_iov = &rma;
        msg.rma_iov_count = 1;
        msg.context = &pp.fctx->ctx;
        return fi_writemsg(th.rails[pp.rail].ep, &msg, flags);
    }
    struct fi_msg msg = {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = pp.dest;
    msg.context = &pp.fctx->ctx;
    return fi_sendmsg(th.rails[pp.rail].ep, &msg, flags);
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
    if (!th.polled[pp.rail]) {
        th.polled[pp.rail] = true;
        th.poll_rails.push_back(pp.rail);
    }
    if (!th.retry.empty()) {
        if (th.prof) {
            pp.queued_at = nixlLibfabricProxyProfile::now();
        }
        th.retry.push_back(pp); // keep order behind earlier back-pressured posts
        return;
    }
    const ssize_t rc = postTimed(th, pp);
    if (rc == -FI_EAGAIN) {
        if (th.prof) {
            pp.queued_at = nixlLibfabricProxyProfile::now();
        }
        th.retry.push_back(pp);
    } else if (rc) {
        NIXL_ERROR << "EFA proxy: post failed on rail " << pp.rail << ": " << fi_strerror(-rc);
        completeFragment(th, pp.fctx->owner, NIXL_ERR_BACKEND);
    }
}

void
nixlLibfabricProxy::drainRetries(Thread &th) {
    while (!th.retry.empty()) {
        const ssize_t rc = postTimed(th, th.retry.front());
        if (rc == -FI_EAGAIN) {
            break; // still back-pressured; keep order and retry next pass
        }
        const PendingPost pp = th.retry.front();
        th.retry.pop_front();
        if (th.prof && rc == 0) {
            th.prof->add(nixlLibfabricProxyProfile::RETRY_WAIT,
                         nixlLibfabricProxyProfile::now() - pp.queued_at);
        }
        if (rc) {
            NIXL_ERROR << "EFA proxy: post failed on rail " << pp.rail << ": " << fi_strerror(-rc);
            completeFragment(th, pp.fctx->owner, NIXL_ERR_BACKEND);
        }
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
nixlLibfabricProxy::pollCqs(Thread &th) {
    struct fi_cq_data_entry entries[kCqBatch];
    for (const uint32_t r : th.poll_rails) {
        RailRes &rr = th.rails[r];
        for (;;) {
            const uint64_t read_start = th.prof ? nixlLibfabricProxyProfile::now() : 0;
            const ssize_t n = fi_cq_read(rr.cq, entries, kCqBatch);
            if (th.prof) {
                const uint64_t took = nixlLibfabricProxyProfile::now() - read_start;
                if (took > kSlowCallNs) {
                    NIXL_INFO << "EFA proxy profile: slow cq read: thread " << th.id << " rail "
                              << r << " read #" << rr.cq_reads << " took " << took / 1000 << " us";
                }
                ++rr.cq_reads;
            }
            if (n > 0) {
                for (ssize_t i = 0; i < n; ++i) {
                    if (entries[i].flags & FI_RECV) {
                        handleRecv(th, static_cast<RecvBuf *>(entries[i].op_context));
                    } else {
                        auto *f = static_cast<Request::FragCtx *>(entries[i].op_context);
                        nixl_status_t status = NIXL_SUCCESS;
                        if (f->owner->inject_cq_error) {
                            f->owner->inject_cq_error = false;
                            NIXL_WARN << "EFA proxy: injected completion error";
                            status = NIXL_ERR_BACKEND;
                        }
                        completeFragment(th, f->owner, status);
                    }
                }
                if (static_cast<size_t>(n) < kCqBatch) {
                    break;
                }
                continue;
            }
            if (n == -FI_EAVAIL) {
                struct fi_cq_err_entry err = {};
                if (fi_cq_readerr(rr.cq, &err, 0) > 0) {
                    reportCqError(th, r, err.err);
                    if (err.flags & FI_RECV) {
                        if (err.op_context) {
                            postRecv(th, static_cast<RecvBuf *>(err.op_context));
                        }
                    } else if (err.op_context) {
                        auto *f = static_cast<Request::FragCtx *>(err.op_context);
                        completeFragment(th, f->owner, NIXL_ERR_BACKEND);
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
    // The runtime calls this once per owned ring. ProxyWorker visits channel t,
    // peer 0 first, so poll the thread's CQs exactly once per pass.
    const uint32_t t = channel % threads_;
    if (channel != t || peer != 0) {
        return NIXL_SUCCESS;
    }
    Thread &th = *thread_state_[t];
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
    const uint32_t t = channel % threads_;
    Thread &th = *thread_state_[t];
    const uint64_t key = static_cast<uint64_t>(channel) * config_.max_peers + peer;
    auto it = th.fences.find(key);
    if (it == th.fences.end()) {
        return NIXL_SUCCESS;
    }
    for (size_t spin = 0; spin < 1000000 && !it->second.order.idle(); ++spin) {
        drainRetries(th);
        pollCqs(th);
    }
    if (!it->second.order.idle()) {
        NIXL_ERROR << "EFA proxy: ring (" << channel << ", " << peer << ") did not quiesce";
        return NIXL_ERR_BACKEND;
    }
    th.fences.erase(it);
    return NIXL_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Target side: incoming atomicAdd records
 * ------------------------------------------------------------------------- */

nixl_status_t
nixlLibfabricProxy::postRecv(Thread &th, RecvBuf *buf) {
    struct iovec iov = {&buf->msg, sizeof(AtomicMsg)};
    void *desc = th.recv_desc;
    struct fi_msg msg = {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = FI_ADDR_UNSPEC;
    msg.context = &buf->ctx;
    const ssize_t rc = fi_recvmsg(th.rails[th.home].ep, &msg, 0);
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
nixlLibfabricProxy::handleRecv(Thread &th, RecvBuf *buf) {
    const uint64_t start = th.prof ? nixlLibfabricProxyProfile::now() : 0;
    if (applyAtomic(buf->msg) != NIXL_SUCCESS) {
        NIXL_ERROR << "EFA proxy: dropped atomicAdd for 0x" << std::hex << buf->msg.remote_addr;
    }
    if (th.prof) {
        th.prof->add(nixlLibfabricProxyProfile::ATOMIC_APPLY,
                     nixlLibfabricProxyProfile::now() - start);
    }
    postRecv(th, buf);
}

nixl_status_t
nixlLibfabricProxy::applyAtomic(const AtomicMsg &msg) {
    Region region{};
    {
        std::lock_guard<std::mutex> lock(regions_mutex_);
        auto it = regions_.upper_bound(msg.remote_addr);
        if (it == regions_.begin()) {
            return NIXL_ERR_NOT_FOUND;
        }
        --it;
        if (msg.remote_addr + sizeof(uint64_t) > it->first + it->second.len) {
            return NIXL_ERR_NOT_FOUND;
        }
        region = it->second;
    }

    NIXL_DEBUG << "EFA proxy: applying atomicAdd " << msg.value << " at 0x" << std::hex
               << msg.remote_addr << std::dec << (region.is_vram ? " (VRAM)" : " (DRAM)");
    if (!region.is_vram) {
        // Every add to this counter arrives on this thread: a plain RMW is safe
        // against other proxy threads; the atomic also covers concurrent CPU users.
        __atomic_fetch_add(
            reinterpret_cast<uint64_t *>(msg.remote_addr), msg.value, __ATOMIC_SEQ_CST);
        return NIXL_SUCCESS;
    }

#ifdef HAVE_GDRCOPY
    gdr_mh_t mh{};
    if (uint64_t *ptr = counters_->lookup(msg.remote_addr, mh)) {
        uint64_t v = 0;
        gdr_copy_from_mapping(mh, &v, ptr, sizeof(v));
        v += msg.value;
        gdr_copy_to_mapping(mh, ptr, &v, sizeof(v));
        return NIXL_SUCCESS;
    }
#endif
#ifdef HAVE_CUDA
    // Fallback without GDRCopy: correct but slow (two synchronous copies).
    if (cudaSetDevice(region.device_id) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    uint64_t v = 0;
    void *dev = reinterpret_cast<void *>(msg.remote_addr);
    if (cudaMemcpy(&v, dev, sizeof(v), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    v += msg.value;
    if (cudaMemcpy(dev, &v, sizeof(v), cudaMemcpyHostToDevice) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
#else
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

void
nixlLibfabricProxy::onRegister(uintptr_t addr, size_t len, bool is_vram, int device_id) {
    std::lock_guard<std::mutex> lock(regions_mutex_);
    regions_[addr] = Region{len, is_vram, device_id};
}

void
nixlLibfabricProxy::onDeregister(uintptr_t addr) {
    size_t len = 0;
    {
        std::lock_guard<std::mutex> lock(regions_mutex_);
        auto it = regions_.find(addr);
        if (it == regions_.end()) {
            return;
        }
        len = it->second.len;
        regions_.erase(it);
    }
    counters_->dropRange(addr, len);
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
        fi_av_insert(th.rails[th.home].av,
                     conn.remote_proxy_ep_names_[owner].data(),
                     1,
                     &addr,
                     0,
                     nullptr) != 1) {
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
        if (addr != FI_ADDR_UNSPEC) {
            fi_av_remove(th.rails[th.home].av, &addr, 1, 0);
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
