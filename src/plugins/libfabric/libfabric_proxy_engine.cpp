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

#include "libfabric_proxy_engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <gdrapi.h>

#include "common/nixl_log.h"
#include "device/device_memview.h"
#include "device/proxy/proxy_runtime.h"

namespace {
constexpr size_t kOwnRailMax = 256 << 10;
constexpr uint32_t kSweepPasses = 64;

constexpr uint32_t kEpochBits = 8;
constexpr uint32_t kEpochs = 1u << kEpochBits;
constexpr uint32_t kMaxChannels = 1u << (32 - NIXL_MSG_TYPE_BITS - kEpochBits);

uint64_t
proxyImm(uint32_t payload) {
    return NIXL_LIBFABRIC_MSG_PROXY | (uint64_t(payload) << NIXL_MSG_TYPE_BITS);
}

struct AddMsg {
    uint64_t channel;
    uint64_t seq;
    uint64_t expected;
    uint64_t addr;
    uint64_t value;
    uint64_t token;
};

struct AckMsg {
    int64_t status;
    uint64_t token;
};

static_assert(sizeof(AddMsg) != sizeof(AckMsg));

struct Request {
    static constexpr int64_t kPosting = int64_t(1) << 40;

    std::atomic<int64_t> pending{0};
    std::atomic<int> status{NIXL_SUCCESS};
    bool add = false;

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

struct Thread {
    std::vector<size_t> poll;
    std::vector<size_t> post;
    size_t next_post = 0;
    std::vector<size_t> one_rail;
    std::vector<size_t> sweep;
    uint32_t passes = 0;
};

struct Fence {
    uint64_t atomics = 0;
    std::array<uint64_t, kEpochs> slot_writes{};
    size_t rail = 0;
    uint64_t dest = 0;
};

struct Ring {
    Thread *thread = nullptr;
    bool lead = false;
    Request *requests = nullptr;
    uint64_t next = 0;
    std::atomic<bool> failed{false};
    std::unordered_map<const nixlLibfabricConnection *, Fence> fences;
};

struct RxFence {
    struct Add {
        AddMsg msg;
        size_t rail;
        fi_addr_t src;
    };

    std::array<std::atomic<uint64_t>, kEpochs> received{};
    std::atomic<uint32_t> waiting{0};
    std::mutex mutex;
    uint64_t applied = 0;
    std::map<uint64_t, Add> adds;
};

struct PendingAck {
    RxFence::Add add;
    nixl_status_t status;
};

struct Region {
    size_t len = 0;
    uint32_t refs = 1;
    bool vram = false;
    void *bar = nullptr;
    uintptr_t va = 0;
    size_t map_len = 0;
    gdr_mh_t mh{};
};

struct RxCache {
    uint64_t owner = 0;
    std::unordered_map<uint64_t, RxFence *> fences;
};

thread_local RxCache tls_rx;
std::atomic<uint64_t> next_proxy_id{1};
std::array<std::mutex, 64> add_stripes;

nixl::deviceOps &
requireDeviceOps() {
    nixl::deviceOps *ops = nixl::getDeviceOps();
    if (ops == nullptr) {
        throw std::runtime_error("device proxy: no device operations loaded");
    }
    return *ops;
}
} // namespace

namespace nixl::libfabric {
class proxyEngine::transport final : public nixl::proxyTransport {
public:
    transport(proxyEngine &engine, const nixl::proxyConfig &config)
        : engine_(engine),
          config_(config),
          id_(next_proxy_id.fetch_add(1)),
          rails_(gpuRails()),
          rings_(config.ringCount()),
          requests_(std::make_unique<Request[]>(rings_.size() * config.ring_depth)) {
        if (config.ring_depth > kEpochs) {
            throw std::runtime_error("Device proxy: proxy_ring_depth must be at most " +
                                     std::to_string(kEpochs));
        }
        if (config.channel_count > kMaxChannels) {
            throw std::runtime_error("Device proxy: proxy_channel_count must be at most " +
                                     std::to_string(kMaxChannels));
        }
        gdr_ = gdr_open();
        if (gdr_ == nullptr) {
            throw std::runtime_error("Device proxy: cannot open GDRCopy");
        }
        for (size_t i = 0; i < rings_.size(); ++i) {
            rings_[i].requests = &requests_[i * config.ring_depth];
        }
        for (uint32_t t = 0; t < config.thread_count; ++t) {
            Thread &th = threads_.emplace_back();
            for (size_t i = t; i < rails_.size(); i += config.thread_count) {
                th.post.push_back(i);
                th.poll.push_back(rails_[i]);
            }
            if (th.post.empty()) {
                th.post.push_back(t % rails_.size());
            }
            const nixl::proxyThreadShare share = config.threadShare(t);
            for (uint32_t c = share.first_channel; c < share.first_channel + share.channel_count;
                 ++c) {
                for (uint32_t p = share.peers.first; p < config.max_peers;
                     p += share.peers.stride) {
                    ring(c, p).thread = &th;
                }
            }
            ring(share.first_channel, share.peers.first).lead = true;
        }
        nixlLibfabricRailManager &rm = engine_.rail_manager_;
        for (size_t r = 0; r < rm.getNumRails(); ++r) {
            if (std::find(rails_.begin(), rails_.end(), r) == rails_.end()) {
                (r == 0 ? threads_[0].poll : threads_[0].sweep).push_back(r);
            }
            rm.getRail(r).setProxyCallback(
                [this,
                 r](uint64_t imm, const void *data, size_t len, uint32_t agent, fi_addr_t src) {
                    onTraffic(imm, data, len, agent, r, src);
                });
        }
        int ordering = 0;
        if (cudaGetDevice(&dev_) == cudaSuccess &&
            cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, dev_) ==
                cudaSuccess) {
            flush_rdma_writes_ = ordering < cudaGPUDirectRDMAWritesOrderingOwner;
        }
    }

    ~transport() override {
        for (size_t r = 0; r < engine_.rail_manager_.getNumRails(); ++r) {
            engine_.rail_manager_.getRail(r).setProxyCallback(nullptr);
        }
        for (auto &[addr, region] : regions_) {
            unmap(region);
        }
        gdr_close(gdr_);
    }

    void
    onRegister(uintptr_t addr, size_t len, bool is_vram) {
        std::unique_lock<std::shared_mutex> lock(regions_mutex_);
        auto [it, inserted] = regions_.try_emplace(addr);
        Region &r = it->second;
        if (!inserted) {
            ++r.refs;
            return;
        }
        r.len = len;
        r.vram = is_vram;
        if (!is_vram) {
            return;
        }
        r.va = addr & ~uintptr_t(0xffff);
        r.map_len = ((addr + len + 0xffff) & ~uintptr_t(0xffff)) - r.va;
        const bool pinned = gdr_pin_buffer(gdr_, r.va, r.map_len, 0, 0, &r.mh) == 0;
        if (!pinned || gdr_map(gdr_, r.mh, &r.bar, r.map_len) != 0) {
            if (pinned) {
                gdr_unpin_buffer(gdr_, r.mh);
            }
            r.bar = nullptr;
            NIXL_WARN << "Device proxy: cannot map " << reinterpret_cast<void *>(addr)
                      << " with GDRCopy; atomicAdds there fail";
        }
    }

    void
    onDeregister(uintptr_t addr) {
        std::unique_lock<std::shared_mutex> lock(regions_mutex_);
        auto it = regions_.find(addr);
        if (it == regions_.end() || --it->second.refs > 0) {
            return;
        }
        unmap(it->second);
        regions_.erase(it);
    }

    nixl_status_t
    submit(const nixl::proxyBackendSubmission &sub, nixl::proxyBackendRequest &request) override {
        Ring &r = ring(sub.channel_id, sub.peer_index);
        if (r.failed.load(std::memory_order_acquire)) {
            return NIXL_ERR_BACKEND;
        }
        const auto &remote =
            *static_cast<const nixlLibfabricPublicMetadata *>(sub.remote.metadataP);
        Thread &th = *r.thread;
        auto [it, inserted] = r.fences.try_emplace(remote.conn_.get());
        Fence &fence = it->second;
        if (inserted) {
            const std::vector<size_t> &eps = remote.remote_selected_endpoints_;
            fence.rail = rails_[th.post[0]];
            fence.dest =
                remote.conn_->rail_remote_addr_list_.at(fence.rail)[eps[th.post[0] % eps.size()]];
        }
        Request &req = r.requests[r.next % config_.ring_depth];
        req.status.store(NIXL_SUCCESS, std::memory_order_relaxed);
        req.add = sub.opcode != nixl_proxy_opcode_t::PUT;
        if (req.add) {
            const AddMsg msg{sub.channel_id,
                             fence.atomics,
                             fence.slot_writes[fence.atomics % kEpochs],
                             sub.remote.addr,
                             sub.value,
                             static_cast<uint64_t>(&req - requests_.get())};
            req.pending.store(1, std::memory_order_relaxed);
            const nixl_status_t status = sendMsg(fence.rail, fence.dest, &msg, sizeof(msg));
            if (status != NIXL_SUCCESS) {
                return status;
            }
            ++fence.atomics;
        } else {
            if (sub.size == 0) {
                return NIXL_SUCCESS;
            }
            const auto &local =
                *static_cast<const nixlLibfabricPrivateMetadata *>(sub.local.metadataP);
            const size_t rail_offset = th.post[th.next_post++ % th.post.size()];
            const std::vector<size_t> *rails = &local.selected_rails_;
            if (sub.size < kOwnRailMax) {
                th.one_rail.assign(1, (*rails)[rail_offset % rails->size()]);
                rails = &th.one_rail;
            }
            const uint32_t slot = static_cast<uint32_t>(fence.atomics % kEpochs);
            size_t submitted = 0;
            req.pending.store(Request::kPosting, std::memory_order_relaxed);
            const nixl_status_t status = engine_.rail_manager_.prepareAndSubmitTransfer(
                nixlLibfabricReq::WRITE,
                reinterpret_cast<void *>(sub.local.addr),
                sub.size,
                sub.remote.addr,
                remote.remote_buf_addr_,
                *rails,
                local.rail_mr_list_,
                remote.rail_remote_key_list_,
                remote.remote_selected_endpoints_,
                remote.conn_->rail_remote_addr_list_,
                0,
                [req = &req, failed = &r.failed](nixl_status_t st) {
                    if (st != NIXL_SUCCESS) {
                        failed->store(true, std::memory_order_release);
                    }
                    req->complete(st);
                },
                submitted,
                0,
                rail_offset,
                false,
                static_cast<int>(sub.local.devId),
                false,
                proxyImm(sub.channel_id << kEpochBits | slot));
            fence.slot_writes[slot] += submitted;
            if (status != NIXL_SUCCESS) {
                req.fail(status);
                r.failed.store(true, std::memory_order_release);
            }
            if (submitted == 0) {
                return status == NIXL_SUCCESS ? NIXL_ERR_BACKEND : status;
            }
            req.pending.fetch_sub(Request::kPosting - static_cast<int64_t>(submitted),
                                  std::memory_order_acq_rel);
        }
        ++r.next;
        request.token = reinterpret_cast<uint64_t>(&req);
        return NIXL_IN_PROG;
    }

    nixl_status_t
    checkCompletion(uint32_t channel,
                    uint32_t peer,
                    const nixl::proxyBackendRequest &request) override {
        const auto *req = reinterpret_cast<const Request *>(request.token);
        if (req->pending.load(std::memory_order_acquire) != 0) {
            if (req->add && ring(channel, peer).failed.load(std::memory_order_acquire)) {
                return NIXL_ERR_BACKEND;
            }
            return NIXL_IN_PROG;
        }
        return static_cast<nixl_status_t>(req->status.load(std::memory_order_relaxed));
    }

    void
    progress(uint32_t channel, uint32_t peer) noexcept override {
        const Ring &r = ring(channel, peer);
        if (!r.lead) {
            return;
        }
        Thread &th = *r.thread;
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

    nixl_status_t
    quiesce(uint32_t, uint32_t) override {
        return NIXL_SUCCESS;
    }

private:
    std::vector<size_t>
    gpuRails() const {
        const nixlLibfabricRailManager &rm = engine_.rail_manager_;
        std::vector<size_t> out;
        int dev = 0;
        char bus_id[32] = {};
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev) == cudaSuccess) {
            out = rm.selectRailsForMemory(nullptr, VRAM_SEG, dev, bus_id);
        }
        for (size_t i = 0; out.empty() && i < rm.getNumRails(); ++i) {
            out.push_back(i);
        }
        return out;
    }

    Ring &
    ring(uint32_t channel, uint32_t peer) {
        return rings_[size_t(channel) * config_.max_peers + peer];
    }

    nixl_status_t
    sendMsg(size_t rail_id, uint64_t dest, const void *msg, size_t len) {
        nixlLibfabricRail &rail = engine_.rail_manager_.getRail(rail_id);
        nixlLibfabricReq *req = rail.allocateControlRequest(len, 0);
        if (req == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        std::memcpy(req->buffer, msg, len);
        const nixl_status_t status = rail.postSend(proxyImm(0), dest, req);
        if (status != NIXL_SUCCESS) {
            rail.releaseRequest(req);
        }
        return status;
    }

    void
    onTraffic(uint64_t imm,
              const void *data,
              size_t len,
              uint32_t agent,
              size_t rail,
              fi_addr_t src) {
        if (data == nullptr) {
            const uint32_t payload = static_cast<uint32_t>(imm) >> NIXL_MSG_TYPE_BITS;
            RxFence &rx = rxFence(agent, payload >> kEpochBits);
            rx.received[payload % kEpochs].fetch_add(1, std::memory_order_seq_cst);
            if (rx.waiting.load(std::memory_order_seq_cst) != 0) {
                drain(rx);
            }
        } else if (len == sizeof(AckMsg)) {
            AckMsg ack;
            std::memcpy(&ack, data, sizeof(ack));
            if (ack.token >= rings_.size() * config_.ring_depth) {
                NIXL_ERROR << "Device proxy: dropping an ack for unknown request " << ack.token;
                return;
            }
            requests_[ack.token].complete(static_cast<nixl_status_t>(ack.status));
        } else if (len == sizeof(AddMsg)) {
            AddMsg msg;
            std::memcpy(&msg, data, sizeof(msg));
            RxFence &rx = rxFence(agent, static_cast<uint32_t>(msg.channel));
            {
                std::lock_guard<std::mutex> lock(rx.mutex);
                rx.adds.emplace(msg.seq, RxFence::Add{msg, rail, src});
            }
            rx.waiting.fetch_add(1, std::memory_order_seq_cst);
            drain(rx);
        } else {
            NIXL_ERROR << "Device proxy: unknown message of " << len << " bytes";
        }
    }

    RxFence &
    rxFence(uint32_t agent, uint32_t channel) {
        const uint64_t key = uint64_t(agent) << 32 | channel;
        if (tls_rx.owner != id_) {
            tls_rx.owner = id_;
            tls_rx.fences.clear();
        }
        RxFence *&cached = tls_rx.fences[key];
        if (cached == nullptr) {
            std::lock_guard<std::mutex> lock(rx_mutex_);
            auto &rx = rx_fences_[key];
            if (!rx) {
                rx = std::make_unique<RxFence>();
            }
            cached = rx.get();
        }
        return *cached;
    }

    void
    drain(RxFence &rx) {
        std::vector<PendingAck> done;
        {
            std::lock_guard<std::mutex> lock(rx.mutex);
            for (auto it = rx.adds.begin(); it != rx.adds.end() && it->first == rx.applied;
                 it = rx.adds.begin()) {
                const AddMsg &msg = it->second.msg;
                if (rx.received[msg.seq % kEpochs].load(std::memory_order_seq_cst) < msg.expected) {
                    break;
                }
                done.push_back({it->second, apply(msg.addr, msg.value)});
                rx.adds.erase(it);
                ++rx.applied;
                rx.waiting.fetch_sub(1, std::memory_order_seq_cst);
            }
        }
        for (const PendingAck &ack : done) {
            if (!sendAck(ack)) {
                NIXL_WARN << "Device proxy: cannot ack an atomicAdd now; retrying";
                std::lock_guard<std::mutex> lock(acks_mutex_);
                acks_.push_back(ack);
                acks_pending_.store(true, std::memory_order_release);
            }
        }
    }

    bool
    sendAck(const PendingAck &ack) {
        const AckMsg msg{ack.status, ack.add.msg.token};
        return sendMsg(ack.add.rail, ack.add.src, &msg, sizeof(msg)) == NIXL_SUCCESS;
    }

    void
    retryAcks() {
        std::vector<PendingAck> acks;
        {
            std::lock_guard<std::mutex> lock(acks_mutex_);
            acks.swap(acks_);
            acks_pending_.store(false, std::memory_order_relaxed);
        }
        std::erase_if(acks, [this](const PendingAck &ack) { return sendAck(ack); });
        if (!acks.empty()) {
            std::lock_guard<std::mutex> lock(acks_mutex_);
            acks_.insert(acks_.end(), acks.begin(), acks.end());
            acks_pending_.store(true, std::memory_order_release);
        }
    }

    nixl_status_t
    apply(uint64_t addr, uint64_t value) {
        std::shared_lock<std::shared_mutex> lock(regions_mutex_);
        auto it = regions_.upper_bound(addr);
        if (it == regions_.begin() ||
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
        std::lock_guard<std::mutex> stripe(add_stripes[(addr >> 3) % add_stripes.size()]);
        if (flush_rdma_writes_) {
            int prev = dev_;
            static_cast<void>(cudaGetDevice(&prev));
            static_cast<void>(cudaSetDevice(dev_));
            static_cast<void>(
                cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                   cudaFlushGPUDirectRDMAWritesToOwner));
            static_cast<void>(cudaSetDevice(prev));
        }
        auto *bar =
            reinterpret_cast<volatile uint64_t *>(static_cast<char *>(r.bar) + (addr - r.va));
        *bar = *bar + value;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return NIXL_SUCCESS;
    }

    void
    unmap(Region &r) {
        if (r.bar != nullptr) {
            gdr_unmap(gdr_, r.mh, r.bar, r.map_len);
            gdr_unpin_buffer(gdr_, r.mh);
        }
    }

    proxyEngine &engine_;
    const nixl::proxyConfig config_;
    const uint64_t id_;
    const std::vector<size_t> rails_;
    std::vector<Ring> rings_;
    const std::unique_ptr<Request[]> requests_;
    std::deque<Thread> threads_;
    gdr_t gdr_ = nullptr;
    int dev_ = 0;
    bool flush_rdma_writes_ = false;

    std::mutex rx_mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<RxFence>> rx_fences_;
    std::mutex acks_mutex_;
    std::vector<PendingAck> acks_;
    std::atomic<bool> acks_pending_{false};
    std::shared_mutex regions_mutex_;
    std::map<uintptr_t, Region> regions_;
};

proxyEngine::proxyEngine(const nixlBackendInitParams &init_params, const nixl::proxyConfig &config)
    : nixlLibfabricEngine(&init_params),
      ops_(requireDeviceOps()) {
    auto t = std::make_unique<transport>(*this, config);
    transport_ = t.get();
    runtime_ = nixl::proxyRuntime::create(std::move(t), config, ops_);
}

proxyEngine::~proxyEngine() = default;

nixl_status_t
proxyEngine::registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) {
    const nixl_status_t status = nixlLibfabricEngine::registerMem(mem, nixl_mem, out);
    if (status == NIXL_SUCCESS) {
        transport_->onRegister(mem.addr, mem.len, nixl_mem == VRAM_SEG);
    }
    return status;
}

nixl_status_t
proxyEngine::deregisterMem(nixlBackendMD *meta) {
    const auto addr =
        reinterpret_cast<uintptr_t>(static_cast<nixlLibfabricPrivateMetadata *>(meta)->buffer_);
    const nixl_status_t status = nixlLibfabricEngine::deregisterMem(meta);
    if (status == NIXL_SUCCESS) {
        transport_->onDeregister(addr);
    }
    return status;
}

template<typename DlistT>
nixl_status_t
proxyEngine::prepProxyMemView(const DlistT &dlist, nixlMemViewH &mvh) const {
    nixlMemViewH proxy_mvh = nullptr;
    nixl_status_t status = runtime_->prepMemView(dlist, &proxy_mvh);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    status = nixl::deviceMemViewAllocate(ops_, nixl_device_exec_mode_t::PROXY, proxy_mvh, mvh);
    if (status != NIXL_SUCCESS) {
        runtime_->releaseMemView(proxy_mvh);
    }
    return status;
}

nixl_status_t
proxyEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                         nixlMemViewH &mvh,
                         const nixl_opt_b_args_t *) const {
    for (const auto &desc : dlist) {
        if (desc.metadataP != nullptr) {
            const nixl_status_t status =
                establishConnection(static_cast<const nixlLibfabricPublicMetadata *>(desc.metadataP)
                                        ->conn_->remoteAgent_);
            if (status != NIXL_SUCCESS) {
                return status;
            }
        }
    }
    return prepProxyMemView(dlist, mvh);
}

nixl_status_t
proxyEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                         nixlMemViewH &mvh,
                         const nixl_opt_b_args_t *) const {
    return prepProxyMemView(dlist, mvh);
}

void
proxyEngine::releaseMemView(nixlMemViewH mem_view) const {
    if (mem_view == nullptr) {
        return;
    }
    nixlMemViewH proxy_mvh = nullptr;
    if (nixl::deviceMemViewGetBackend(ops_, mem_view, proxy_mvh) != NIXL_SUCCESS) {
        NIXL_FATAL << "Failed to read proxy memory view before retirement";
    }
    runtime_->releaseMemView(proxy_mvh);
    nixl::deviceMemViewFree(ops_, mem_view);
}
} // namespace nixl::libfabric
