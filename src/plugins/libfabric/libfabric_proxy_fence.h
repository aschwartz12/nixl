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
#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_FENCE_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_FENCE_H

#include <cstdint>
#include <deque>

#include "nixl_types.h"

/**
 * Ordering state of one proxy ring (channel, peer): an atomicAdd is sent only
 * after every earlier put on the ring and the ring's previous atomicAdd have
 * completed. Pure bookkeeping, so it can be tested without a fabric.
 *
 * Operations are grouped into epochs, each closed by one atomic. Puts join the
 * open (last) epoch. A closed epoch releases its atomic once its count drops to
 * zero; the released atomic then counts against the next epoch, so the next
 * atomic waits for it too. A failure belongs to the epoch of the failed
 * operation: that epoch's atomic and every later one complete with the error
 * instead of being sent (never signal over bad data), while atomics of earlier
 * epochs are still sent once their own operations succeed.
 *
 * Epoch pointers stay valid until the epoch is released: std::deque keeps
 * references stable when elements are added or removed at either end.
 */
template<typename Op> class nixlLibfabricRingFence {
public:
    struct Epoch {
        uint32_t outstanding = 0; // puts (and the previous atomic) not yet complete
        Op *held = nullptr; // atomic that closes this epoch; null for the open one
        nixl_status_t error = NIXL_SUCCESS; // first failure among its operations
    };

    nixlLibfabricRingFence() {
        epochs_.emplace_back();
    }

    /** A put joins the open epoch; pass the result to complete(). */
    Epoch *
    addPut() {
        Epoch *epoch = &epochs_.back();
        ++epoch->outstanding;
        return epoch;
    }

    /** An atomic closes the open epoch; later puts join a new one. Call release() next. */
    void
    addAtomic(Op *atomic) {
        epochs_.back().held = atomic;
        epochs_.emplace_back();
    }

    /** An operation counted against @p epoch finished. Call release() next. */
    void
    complete(Epoch *epoch, nixl_status_t status) {
        if (epoch == nullptr) {
            return;
        }
        if (status != NIXL_SUCCESS && epoch->error == NIXL_SUCCESS) {
            epoch->error = status;
        }
        --epoch->outstanding;
    }

    /**
     * Release the atomics whose epochs have drained, oldest first:
     * send(atomic, epoch) must post it and later complete(epoch, ...);
     * fail(atomic, error) completes it without sending after a failure.
     */
    template<typename Send, typename Fail>
    void
    release(Send &&send, Fail &&fail) {
        while (epochs_.size() > 1 && epochs_.front().outstanding == 0) {
            Op *atomic = epochs_.front().held;
            if (error_ == NIXL_SUCCESS) {
                error_ = epochs_.front().error;
            }
            epochs_.pop_front();
            if (error_ != NIXL_SUCCESS) {
                fail(atomic, error_);
                continue;
            }
            Epoch *next = &epochs_.front();
            ++next->outstanding;
            send(atomic, next);
        }
    }

    /** No operation of this ring is outstanding or held. */
    [[nodiscard]] bool
    idle() const {
        return epochs_.size() == 1 && epochs_.front().outstanding == 0;
    }

    /** First failure that has blocked an atomic on the ring, or NIXL_SUCCESS. */
    [[nodiscard]] nixl_status_t
    error() const {
        return error_;
    }

    /** Atomics held behind unfinished operations. */
    [[nodiscard]] size_t
    heldAtomics() const {
        return epochs_.size() - 1;
    }

private:
    std::deque<Epoch> epochs_;
    nixl_status_t error_ = NIXL_SUCCESS;
};

#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_FENCE_H
