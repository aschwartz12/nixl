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
#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_WIRE_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_WIRE_H

#include <cstdint>

/**
 * What EFA proxy threads put on the wire. Dependency-free so tests can include it.
 *
 * put -> atomicAdd ordering is kept by the target (receiver-side ordering):
 *
 *  - Every put fragment is an RDMA write carrying 32 bits of remote CQ data,
 *    putImm(): its ring's key and its epoch, the index of the atomicAdd that will
 *    follow it on the ring (mod kEpochs). The target thread that polls the rail it
 *    lands on counts it per (ring, epoch slot).
 *  - An atomicAdd leaves at once as an atomicAddMsg to the counter's owner thread
 *    (counterOwner()), with its index on the ring (seq) and the fragment count its
 *    epoch slot reaches once its own puts are in (expected_puts). The owner applies
 *    it when that count is reached and the ring's previous atomicAdd was applied,
 *    then acks; the sender completes the atomicAdd on the ack.
 *  - A ring key names one sender ring at one target, deterministically:
 *    ringKey(the sender's index at the target, a ring id). The sender allocates
 *    ring ids per target, one per (runtime ring, target), and a fresh one when a
 *    failed ring is used again, so its counts start over at the target.
 *  - Slot s of a ring is reused kEpochs atomicAdds later; a ring with at most
 *    kEpochs requests in flight cannot get there before atomicAdd s was applied.
 *
 * Control messages (atomicAdd, ack, ring abort) go between the threads' control
 * EPs; puts go to the data EP of the thread that receives on the destination rail.
 */
namespace nixlLibfabricProxyWire {

/** Bumped on any layout or protocol change; also published in the connection info. */
inline constexpr uint16_t kVersion = 4;

/** Room for a libfabric endpoint name (LF_EP_NAME_MAX_LEN, checked by the proxy). */
inline constexpr uint32_t kMaxEpName = 56;

/** Epoch slots per ring: the put immediate's low bits. */
inline constexpr uint32_t kEpochBits = 8;
inline constexpr uint32_t kEpochs = 1u << kEpochBits;
/** Ring key fields (24 bits): the sender's index at the target, then a ring id. */
inline constexpr uint32_t kMaxSenderIndex = 0xff; // the engine's NIXL_AGENT_INDEX_MASK
inline constexpr uint32_t kRingIds = 0x10000;

enum class msgType : uint16_t { ATOMIC_ADD = 1, ATOMIC_ACK = 2, RING_ABORT = 3 };

struct msgHeader {
    uint16_t version;
    msgType type;
    uint32_t reserved;
};

/** Sender -> counter owner. */
struct atomicAddMsg {
    msgHeader hdr;
    uint64_t remote_addr;
    uint64_t value;
    uint64_t token; // sender's request, echoed in the ack
    uint32_t ring; // ringKey() of the sender's ring
    uint32_t reply_name_len;
    uint64_t seq; // index of this add among the ring's atomicAdds
    uint64_t expected_puts; // count of slot seq % kEpochs once this add's puts are in
    uint8_t reply_name[kMaxEpName]; // sender's control endpoint, where the ack goes
};

/** Counter owner -> sender, after the add was applied or failed. */
struct atomicAckMsg {
    msgHeader hdr;
    uint64_t token;
    int32_t status; // nixl_status_t
    uint32_t reserved;
};

/**
 * Sender -> every proxy thread of the target: the ring failed at the sender, so its
 * atomicAdds from first_seq on will never be satisfiable (or are not wanted). The
 * owners drop them without applying (no ack: the sender already failed them) and
 * forget the ring once nothing of it waits.
 */
struct ringAbortMsg {
    msgHeader hdr;
    uint32_t ring;
    uint32_t reserved;
    uint64_t first_seq;
};

/** One receive buffer holds any message. */
union anyMsg {
    msgHeader hdr;
    atomicAddMsg add;
    atomicAckMsg ack;
    ringAbortMsg abort;
};

constexpr uint32_t
ringKey(uint32_t sender_index, uint32_t ring_id) {
    return (sender_index & kMaxSenderIndex) << 16 | (ring_id & (kRingIds - 1));
}

constexpr uint32_t
putImm(uint32_t ring_key, uint64_t epoch) {
    return ring_key << kEpochBits | static_cast<uint32_t>(epoch & (kEpochs - 1));
}

constexpr uint32_t
immRing(uint32_t imm) {
    return imm >> kEpochBits;
}

constexpr uint32_t
immSlot(uint32_t imm) {
    return imm & (kEpochs - 1);
}

/** Deterministic mixer: every sender must pick the same owner for a counter. */
constexpr uint64_t
mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/** Proxy thread of a target with @p threads proxy threads that applies adds to @p addr. */
constexpr uint32_t
counterOwner(uint64_t addr, uint32_t threads) {
    return static_cast<uint32_t>(mix64(addr >> 3) % threads);
}

/**
 * Proxy thread of a target with @p threads proxy threads that receives (polls and
 * counts) the puts landing on its receive rail number @p rail_index.
 */
constexpr uint32_t
railThread(uint32_t rail_index, uint32_t threads) {
    return rail_index % threads;
}

} // namespace nixlLibfabricProxyWire

#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_WIRE_H
