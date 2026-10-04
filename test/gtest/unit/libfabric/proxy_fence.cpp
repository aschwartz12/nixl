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

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <random>
#include <vector>

#include "libfabric_proxy_fence.h"

namespace {

struct Op;
using Fence = nixlLibfabricRingFence<Op>;

struct Op {
    int id = 0;
    Fence::Epoch *epoch = nullptr; // set once sent
    bool sent = false;
    bool failed = false;
    nixl_status_t status = NIXL_IN_PROG;
};

/** Drives a fence the way the EFA proxy does, with completions supplied by the test. */
class Driver {
public:
    Fence fence;
    std::vector<int> sent;
    std::vector<int> failed;

    Fence::Epoch *
    put() {
        return fence.addPut();
    }

    void
    atomic(Op &op) {
        fence.addAtomic(&op);
        release();
    }

    void
    completePut(Fence::Epoch *epoch, nixl_status_t status = NIXL_SUCCESS) {
        fence.complete(epoch, status);
        release();
    }

    void
    completeAtomic(Op &op, nixl_status_t status = NIXL_SUCCESS) {
        ASSERT_TRUE(op.sent) << "atomic " << op.id << " completed before it was sent";
        fence.complete(op.epoch, status);
        op.epoch = nullptr;
        op.status = status;
        release();
    }

private:
    void
    release() {
        fence.release(
            [this](Op *op, Fence::Epoch *epoch) {
                op->epoch = epoch;
                op->sent = true;
                sent.push_back(op->id);
            },
            [this](Op *op, nixl_status_t error) {
                op->failed = true;
                op->status = error;
                failed.push_back(op->id);
            });
    }
};

TEST(LibfabricProxyFenceTest, AtomicWithoutPutsIsSentAtOnce) {
    Driver d;
    EXPECT_TRUE(d.fence.idle());
    Op a{1};
    d.atomic(a);
    EXPECT_EQ(d.sent, std::vector<int>{1});
    EXPECT_FALSE(d.fence.idle()); // the atomic itself is in flight
    d.completeAtomic(a);
    EXPECT_TRUE(d.fence.idle());
}

TEST(LibfabricProxyFenceTest, AtomicWaitsForEveryEarlierPut) {
    Driver d;
    Fence::Epoch *p1 = d.put();
    Fence::Epoch *p2 = d.put();
    Op a{1};
    d.atomic(a);
    EXPECT_TRUE(d.sent.empty());
    EXPECT_EQ(d.fence.heldAtomics(), 1u);
    d.completePut(p2);
    EXPECT_TRUE(d.sent.empty());
    d.completePut(p1);
    EXPECT_EQ(d.sent, std::vector<int>{1});
    d.completeAtomic(a);
    EXPECT_TRUE(d.fence.idle());
}

TEST(LibfabricProxyFenceTest, LaterPutsDoNotDelayAnAtomic) {
    Driver d;
    Fence::Epoch *before = d.put();
    Op a{1};
    d.atomic(a);
    Fence::Epoch *after = d.put();
    d.completePut(before);
    EXPECT_EQ(d.sent, std::vector<int>{1});
    d.completeAtomic(a);
    EXPECT_FALSE(d.fence.idle());
    d.completePut(after);
    EXPECT_TRUE(d.fence.idle());
}

TEST(LibfabricProxyFenceTest, AtomicsAreStrictlyOrdered) {
    Driver d;
    Op a1{1};
    d.atomic(a1);
    Fence::Epoch *p = d.put();
    Op a2{2};
    d.atomic(a2);
    d.completePut(p);
    // Its puts are done, but the previous atomic is still in flight.
    EXPECT_EQ(d.sent, std::vector<int>{1});
    d.completeAtomic(a1);
    EXPECT_EQ(d.sent, (std::vector<int>{1, 2}));
    d.completeAtomic(a2);
    EXPECT_TRUE(d.fence.idle());
}

TEST(LibfabricProxyFenceTest, PutsOfALaterEpochMayCompleteFirst) {
    Driver d;
    Fence::Epoch *p0 = d.put();
    Op a1{1};
    d.atomic(a1);
    Fence::Epoch *p1 = d.put();
    Op a2{2};
    d.atomic(a2);
    d.completePut(p1);
    EXPECT_TRUE(d.sent.empty());
    d.completePut(p0);
    EXPECT_EQ(d.sent, std::vector<int>{1});
    d.completeAtomic(a1);
    EXPECT_EQ(d.sent, (std::vector<int>{1, 2}));
    d.completeAtomic(a2);
    EXPECT_TRUE(d.fence.idle());
}

TEST(LibfabricProxyFenceTest, FailedPutBlocksItsAtomicAndEveryLaterOne) {
    Driver d;
    Fence::Epoch *p0 = d.put();
    Op a1{1};
    d.atomic(a1);
    Fence::Epoch *p1 = d.put();
    Op a2{2};
    d.atomic(a2);

    d.completePut(p0, NIXL_ERR_BACKEND);
    EXPECT_TRUE(d.sent.empty());
    EXPECT_EQ(d.failed, std::vector<int>{1});
    EXPECT_EQ(a1.status, NIXL_ERR_BACKEND);
    EXPECT_EQ(d.fence.error(), NIXL_ERR_BACKEND);

    // Later puts still run; the ring's later atomics fail instead of signalling.
    d.completePut(p1);
    EXPECT_TRUE(d.sent.empty());
    EXPECT_EQ(d.failed, (std::vector<int>{1, 2}));
    EXPECT_EQ(a2.status, NIXL_ERR_BACKEND);
    EXPECT_TRUE(d.fence.idle());
}

// The proxy posts puts as soon as they arrive, so a later round's put can fail
// while earlier rounds' atomics still wait their turn; those must still be sent.
TEST(LibfabricProxyFenceTest, FailureDoesNotBlockEarlierAtomics) {
    Driver d;
    Op a0{0};
    d.atomic(a0);
    Fence::Epoch *p1 = d.put();
    Op a1{1};
    d.atomic(a1);
    Fence::Epoch *p2 = d.put();
    Op a2{2};
    d.atomic(a2);

    d.completePut(p2, NIXL_ERR_BACKEND);
    d.completePut(p1);
    EXPECT_EQ(d.sent, std::vector<int>{0}); // a1 waits for a0
    EXPECT_EQ(d.fence.error(), NIXL_SUCCESS);
    d.completeAtomic(a0);
    EXPECT_EQ(d.sent, (std::vector<int>{0, 1}));
    EXPECT_TRUE(d.failed.empty());
    d.completeAtomic(a1);
    EXPECT_EQ(d.failed, std::vector<int>{2});
    EXPECT_EQ(a2.status, NIXL_ERR_BACKEND);
    EXPECT_EQ(d.fence.error(), NIXL_ERR_BACKEND);
    EXPECT_TRUE(d.fence.idle());
}

TEST(LibfabricProxyFenceTest, FailedAtomicBlocksTheNextAtomic) {
    Driver d;
    Op a1{1};
    d.atomic(a1);
    Op a2{2};
    d.atomic(a2);
    EXPECT_EQ(d.sent, std::vector<int>{1});
    d.completeAtomic(a1, NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(d.sent, std::vector<int>{1});
    EXPECT_EQ(d.failed, std::vector<int>{2});
    EXPECT_EQ(a2.status, NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_TRUE(d.fence.idle());
}

TEST(LibfabricProxyFenceTest, FirstFailureIsReported) {
    Driver d;
    Fence::Epoch *p0 = d.put();
    Fence::Epoch *p1 = d.put();
    Op a{1};
    d.atomic(a);
    d.completePut(p1, NIXL_ERR_REMOTE_DISCONNECT);
    d.completePut(p0, NIXL_ERR_BACKEND);
    EXPECT_EQ(d.fence.error(), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(a.status, NIXL_ERR_REMOTE_DISCONNECT);
}

/**
 * Many rounds of puts closed by an atomic, with every completion delivered in a
 * random order. Each atomic must be sent only after all puts of earlier rounds
 * and the previous atomic have completed, and atomics must go out in order.
 */
TEST(LibfabricProxyFenceTest, RandomCompletionOrderKeepsRoundsOrdered) {
    constexpr int kRounds = 2000;
    std::mt19937 rng(12345);
    Driver d;
    std::deque<Op> atomics; // stable addresses

    struct Pending {
        int round;
        Fence::Epoch *epoch; // put: its epoch; atomic: nullptr
        Op *atomic;
    };

    std::vector<Pending> in_flight;
    std::vector<int> puts_left(kRounds, 0);
    size_t checked = 0;

    const auto check_new_sends = [&]() {
        for (; checked < d.sent.size(); ++checked) {
            const int round = d.sent[checked];
            ASSERT_EQ(round, static_cast<int>(checked)) << "atomics sent out of order";
            for (int r = 0; r <= round; ++r) {
                ASSERT_EQ(puts_left[r], 0)
                    << "atomic " << round << " overtook a put of round " << r;
            }
            if (round > 0) {
                ASSERT_EQ(atomics[round - 1].status, NIXL_SUCCESS)
                    << "atomic " << round << " overtook atomic " << round - 1;
            }
            in_flight.push_back({round, nullptr, &atomics[round]});
        }
    };

    const auto complete_random = [&]() {
        std::uniform_int_distribution<size_t> pick(0, in_flight.size() - 1);
        const size_t i = pick(rng);
        const Pending p = in_flight[i];
        in_flight[i] = in_flight.back();
        in_flight.pop_back();
        if (p.atomic != nullptr) {
            d.completeAtomic(*p.atomic);
        } else {
            --puts_left[p.round];
            d.completePut(p.epoch);
        }
        check_new_sends();
    };

    std::uniform_int_distribution<int> nputs(0, 4);
    std::uniform_int_distribution<int> coin(0, 2);
    for (int round = 0; round < kRounds; ++round) {
        const int n = nputs(rng);
        for (int i = 0; i < n; ++i) {
            ++puts_left[round];
            in_flight.push_back({round, d.put(), nullptr});
        }
        atomics.push_back(Op{round});
        d.atomic(atomics.back());
        check_new_sends();
        // Deliver some completions while later rounds keep arriving.
        while (!in_flight.empty() && coin(rng) != 0) {
            complete_random();
        }
    }
    while (!in_flight.empty()) {
        complete_random();
    }

    EXPECT_EQ(d.sent.size(), static_cast<size_t>(kRounds));
    EXPECT_TRUE(d.failed.empty());
    EXPECT_TRUE(d.fence.idle());
}

} // namespace
