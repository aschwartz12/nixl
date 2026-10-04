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
 * Messages between EFA proxy threads, sent over their home endpoints. A sender's
 * atomicAdd goes to the counter's owner thread at the target, which applies it and
 * answers with an ack carrying the result; the sender completes the atomicAdd only
 * then. Dependency-free so tests can include it.
 */
namespace nixlLibfabricProxyWire {

/** Bumped on any layout or protocol change; also published in the connection info. */
inline constexpr uint16_t kVersion = 2;

/** Room for a libfabric endpoint name (LF_EP_NAME_MAX_LEN, checked by the proxy). */
inline constexpr uint32_t kMaxEpName = 56;

enum class msgType : uint16_t { ATOMIC_ADD = 1, ATOMIC_ACK = 2 };

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
    uint32_t reply_name_len;
    uint32_t reserved;
    uint8_t reply_name[kMaxEpName]; // sender's home endpoint, where the ack goes
};

/** Counter owner -> sender, after the add was applied or failed. */
struct atomicAckMsg {
    msgHeader hdr;
    uint64_t token;
    int32_t status; // nixl_status_t
    uint32_t reserved;
};

/** One receive buffer holds either message. */
union anyMsg {
    msgHeader hdr;
    atomicAddMsg add;
    atomicAckMsg ack;
};

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

} // namespace nixlLibfabricProxyWire

#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_WIRE_H
