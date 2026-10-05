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

// put -> atomicAdd ordering check, shared by the gtest and the two-node benchmark.
//
// Per channel, the sender streams rounds of puts followed by an atomicAdd on one
// (channel, peer) ring. A receiver kernel spins on each counter and, every time it
// moves, verifies every byte of the rounds it now covers. A counter that runs ahead
// of its data, moves backwards, or is seen half-updated, is a failure.

#ifndef NIXL_TEST_GTEST_DEVICE_API_PROXY_ORDERING_CUH
#define NIXL_TEST_GTEST_DEVICE_API_PROXY_ORDERING_CUH

#include <gpu/nixl_device.cuh>
#include <cuda/atomic>

#include <cstddef>
#include <cstdint>

namespace nixl_test::proxy_ordering {

constexpr unsigned kMaxChannels = 32;
constexpr unsigned kPutsPerRound = 4;

/** Put sizes of a round; the last one is above the striping threshold (128 KiB). */
__host__ __device__ constexpr size_t
putSize(unsigned put) {
    return put == 0 ? 64 : put == 1 ? 4096 : put == 2 ? 64 * 1024 : 256 * 1024;
}

constexpr size_t kRoundBytes = putSize(0) + putSize(1) + putSize(2) + putSize(3);
// Counters live in the same registered buffer as the data: a ring is keyed by
// (channel, destination descriptor index), and the fence only orders one ring.
constexpr size_t kCounterStride = 64;
constexpr size_t kDataOffset = 4096;

static_assert(kMaxChannels * kCounterStride <= kDataOffset);

// Each round adds one to both 32-bit halves of its counter. The target applies
// adds with a CPU read-modify-write while the GPU polls, so a reader that sees the
// halves differ caught a torn update.
constexpr unsigned long long kCounterStep = (1ull << 32) | 1ull;

__host__ __device__ constexpr bool
counterTorn(unsigned long long counter) {
    return (counter >> 32) != (counter & 0xffffffffull);
}

__host__ __device__ constexpr unsigned long long
counterRounds(unsigned long long counter) {
    return counter & 0xffffffffull;
}

static_assert(kRoundBytes % sizeof(uint32_t) == 0);

/** Buffer layout: counters, then kRoundBytes per (channel, round). */
struct Layout {
    unsigned channels;
    unsigned rounds;

    __host__ __device__ size_t
    roundOffset(unsigned channel, unsigned round) const {
        return kDataOffset + (size_t(channel) * rounds + round) * kRoundBytes;
    }

    __host__ __device__ size_t
    bufferBytes() const {
        return roundOffset(channels, 0);
    }
};

/** Value of 32-bit word `word` of a round's data; distinct per channel and round. */
__host__ __device__ constexpr uint32_t
pattern(unsigned channel, unsigned round, size_t word) {
    return ((round + 1) * 2654435761u) ^ (channel << 28) ^ static_cast<uint32_t>(word);
}

struct Result {
    unsigned long long mismatches;
    unsigned long long regressions;
    unsigned long long torn; // counter reads with differing halves
    unsigned long long timed_out;
    unsigned long long first_bad; // 1 + index of the first mismatching word, 0 if none
    unsigned first_channel;
    unsigned first_round;
    uint32_t first_value;
    uint32_t first_expected;
    unsigned long long final_counter[kMaxChannels]; // rounds signalled
    nixl_status_t sender_status[kMaxChannels];
    // Sender GPU time inside nixlPut()/nixlAtomicAdd() (including waits for ring
    // space), and in the final completion wait.
    unsigned long long enqueue_ns[kMaxChannels];
    unsigned long long final_wait_ns[kMaxChannels];
};

__device__ inline unsigned long long
globalTimeNs() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

__global__ void
fillKernel(uint32_t *buf, Layout layout) {
    const size_t words_per_round = kRoundBytes / sizeof(uint32_t);
    const size_t total = size_t(layout.channels) * layout.rounds * words_per_round;
    uint32_t *data = buf + kDataOffset / sizeof(uint32_t);
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total;
         i += size_t(gridDim.x) * blockDim.x) {
        const size_t round_idx = i / words_per_round;
        data[i] = pattern(static_cast<unsigned>(round_idx / layout.rounds),
                          static_cast<unsigned>(round_idx % layout.rounds),
                          i % words_per_round);
    }
}

/** One block (one thread) per channel: rounds of puts, each closed by atomicAdd. */
__global__ void
senderKernel(nixlMemViewH src,
             nixlMemViewH dst,
             Layout layout,
             unsigned long long timeout_ns,
             Result *result) {
    const unsigned channel = blockIdx.x;
    nixlGpuXferStatusH last{};
    nixl_status_t status = NIXL_SUCCESS;
    unsigned long long enqueue_ns = 0;

    for (unsigned round = 0; round < layout.rounds && status == NIXL_SUCCESS; ++round) {
        size_t offset = layout.roundOffset(channel, round);
        enqueue_ns -= globalTimeNs();
        for (unsigned put = 0; put < kPutsPerRound; ++put) {
            if (nixlPut<nixl_gpu_level_t::THREAD>(
                    {src, 0, offset}, {dst, 0, offset}, putSize(put), channel) != NIXL_IN_PROG) {
                status = NIXL_ERR_BACKEND;
                break;
            }
            offset += putSize(put);
        }
        if (status == NIXL_SUCCESS &&
            nixlAtomicAdd<nixl_gpu_level_t::THREAD>(kCounterStep,
                                                    {dst, 0, channel * kCounterStride},
                                                    channel,
                                                    0,
                                                    round + 1 == layout.rounds ? &last : nullptr) !=
                NIXL_IN_PROG) {
            status = NIXL_ERR_BACKEND;
        }
        enqueue_ns += globalTimeNs();
    }

    // Completions publish in ring order: the last atomic covers everything before it.
    const unsigned long long wait_start = globalTimeNs();
    const unsigned long long deadline = wait_start + timeout_ns;
    if (status == NIXL_SUCCESS) {
        do {
            status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(last);
        } while (status == NIXL_IN_PROG && globalTimeNs() < deadline);
    }
    result->sender_status[channel] = status;
    result->enqueue_ns[channel] = enqueue_ns;
    result->final_wait_ns[channel] = globalTimeNs() - wait_start;
}

/**
 * Like senderKernel, but each round's signal is an 8-byte put of the counter's new
 * value ("PUT_VALUE") instead of an atomicAdd: nothing orders it after the round's
 * puts. The value is staged in the source buffer's counter area (8 slots per
 * channel, each reused once its previous put completed).
 */
__global__ void
valueSenderKernel(nixlMemViewH src,
                  nixlMemViewH dst,
                  uint8_t *src_buf,
                  Layout layout,
                  unsigned long long timeout_ns,
                  Result *result) {
    constexpr unsigned kSlots = kCounterStride / sizeof(unsigned long long);
    const unsigned channel = blockIdx.x;
    nixlGpuXferStatusH slot_status[kSlots];
    bool slot_busy[kSlots] = {};
    nixl_status_t status = NIXL_SUCCESS;
    unsigned long long enqueue_ns = 0;
    const unsigned long long deadline = globalTimeNs() + timeout_ns;

    for (unsigned round = 0; round < layout.rounds && status == NIXL_SUCCESS; ++round) {
        size_t offset = layout.roundOffset(channel, round);
        enqueue_ns -= globalTimeNs();
        for (unsigned put = 0; put < kPutsPerRound; ++put) {
            if (nixlPut<nixl_gpu_level_t::THREAD>(
                    {src, 0, offset}, {dst, 0, offset}, putSize(put), channel) != NIXL_IN_PROG) {
                status = NIXL_ERR_BACKEND;
                break;
            }
            offset += putSize(put);
        }
        const unsigned slot = round % kSlots;
        if (status == NIXL_SUCCESS && slot_busy[slot]) {
            do {
                status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(slot_status[slot]);
            } while (status == NIXL_IN_PROG && globalTimeNs() < deadline);
        }
        const size_t value_offset = channel * kCounterStride + slot * sizeof(unsigned long long);
        if (status == NIXL_SUCCESS) {
            *reinterpret_cast<volatile unsigned long long *>(src_buf + value_offset) =
                (round + 1) * kCounterStep;
            __threadfence_system();
            slot_status[slot] = nixlGpuXferStatusH{};
            if (nixlPut<nixl_gpu_level_t::THREAD>({src, 0, value_offset},
                                                  {dst, 0, channel * kCounterStride},
                                                  sizeof(unsigned long long),
                                                  channel,
                                                  0,
                                                  &slot_status[slot]) != NIXL_IN_PROG) {
                status = NIXL_ERR_BACKEND;
            }
            slot_busy[slot] = true;
        }
        enqueue_ns += globalTimeNs();
    }
    const unsigned long long wait_start = globalTimeNs();
    for (unsigned slot = 0; slot < kSlots && status == NIXL_SUCCESS; ++slot) {
        while (slot_busy[slot] &&
               (status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(slot_status[slot])) ==
                   NIXL_IN_PROG &&
               globalTimeNs() < deadline) {}
    }
    result->sender_status[channel] = status;
    result->enqueue_ns[channel] = enqueue_ns;
    result->final_wait_ns[channel] = globalTimeNs() - wait_start;
}

/** One block per channel: follow the counter and check the rounds it covers. */
__global__ void
receiverKernel(uint8_t *buf, Layout layout, unsigned long long timeout_ns, Result *result) {
    const unsigned channel = blockIdx.x;
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> counter(
        *reinterpret_cast<unsigned long long *>(buf + channel * kCounterStride));
    __shared__ unsigned long long s_now;
    __shared__ bool s_stop;

    const unsigned long long deadline = globalTimeNs() + timeout_ns;
    unsigned long long seen = 0; // rounds
    for (;;) {
        if (threadIdx.x == 0) {
            unsigned long long now = seen;
            bool timed_out = false;
            for (;;) {
                const unsigned long long raw = counter.load(cuda::memory_order_acquire);
                if (counterTorn(raw)) {
                    atomicAdd(&result->torn, 1ull);
                } else if ((now = counterRounds(raw)) != seen) {
                    break;
                }
                if (globalTimeNs() > deadline) {
                    timed_out = true;
                    now = seen;
                    break;
                }
            }
            if (now < seen) {
                atomicAdd(&result->regressions, 1ull);
                now = seen;
            }
            if (timed_out) {
                atomicAdd(&result->timed_out, 1ull);
            }
            s_now = now;
            s_stop = timed_out || now >= layout.rounds;
        }
        __syncthreads();
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);

        const unsigned long long now = s_now;
        const size_t words = kRoundBytes / sizeof(uint32_t);
        for (unsigned long long round = seen; round < now && round < layout.rounds; ++round) {
            const volatile uint32_t *data = reinterpret_cast<const volatile uint32_t *>(
                buf + layout.roundOffset(channel, static_cast<unsigned>(round)));
            for (size_t w = threadIdx.x; w < words; w += blockDim.x) {
                const uint32_t expected = pattern(channel, static_cast<unsigned>(round), w);
                const uint32_t value = data[w];
                if (value != expected) {
                    atomicAdd(&result->mismatches, 1ull);
                    if (atomicCAS(&result->first_bad, 0ull, w + 1) == 0ull) {
                        result->first_channel = channel;
                        result->first_round = static_cast<unsigned>(round);
                        result->first_value = value;
                        result->first_expected = expected;
                    }
                }
            }
        }
        seen = now;
        const bool stop = s_stop;
        __syncthreads();
        if (stop) {
            break;
        }
    }
    if (threadIdx.x == 0) {
        const unsigned long long raw = counter.load(cuda::memory_order_acquire);
        if (counterTorn(raw)) {
            atomicAdd(&result->torn, 1ull);
        }
        result->final_counter[channel] = counterRounds(raw);
    }
}

} // namespace nixl_test::proxy_ordering

#endif // NIXL_TEST_GTEST_DEVICE_API_PROXY_ORDERING_CUH
