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
#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_CONNINFO_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_CONNINFO_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "libfabric/libfabric_common.h"
#include "libfabric_proxy_wire.h"
#include "serdes/serdes.h"

/**
 * The EFA device proxy's section of the engine's connection info:
 *
 *   "<engine blob><proxy blob><8-byte proxy blob length>EFAPRXY1"
 *
 * The proxy blob carries the proxy protocol version, every proxy thread's control
 * EP (atomicAdd records, acks and ring aborts go there), the rails that receive
 * puts (those next to the GPU) with the data EP of the thread receiving on each
 * (wire::railThread()), and a random incarnation that changes whenever the proxy is
 * re-created (senders restart their ring counts with it). Names are published at
 * their real length: the whole connection info travels in the engine's handshake,
 * which is limited to 8 KiB.
 *
 * A peer without the proxy sends only the engine blob, which split() returns
 * unchanged; a peer with another protocol version is rejected by parse(), so no
 * device operation is sent to it.
 */
namespace nixlLibfabricProxyConnInfo {

using EpName = std::array<char, LF_EP_NAME_MAX_LEN>;

inline constexpr char kMagic[] = "EFAPRXY1";
inline constexpr size_t kMagicLen = sizeof(kMagic) - 1;
inline constexpr char kVersionTag[] = "efa_proxy_version";
inline constexpr char kThreadsTag[] = "efa_proxy_threads";
inline constexpr char kEpTagPrefix[] = "efa_proxy_ep_";
inline constexpr char kRailsTag[] = "efa_proxy_data_rails";
inline constexpr char kRailTagPrefix[] = "efa_proxy_data_rail_";
inline constexpr char kDataEpTagPrefix[] = "efa_proxy_data_ep_";
inline constexpr char kIncarnationTag[] = "efa_proxy_incarnation";
/** Rail numbers and counts above this are malformed (EFA nodes have tens of rails). */
inline constexpr uint64_t kMaxRails = 1024;

/** A proxy's endpoints, as published. */
struct ProxyEps {
    std::vector<std::string> home; // per proxy thread: its control EP name
    std::vector<uint32_t> data_rails; // rails that receive puts, in order
    std::vector<std::string> data; // per data rail: the receiving thread's data EP
    uint64_t incarnation = 0;
};

/** Proxy blob for @p eps. */
inline std::string
serialize(const ProxyEps &eps) {
    nixlSerDes sd;
    sd.addStr(kVersionTag, std::to_string(nixlLibfabricProxyWire::kVersion));
    sd.addStr(kThreadsTag, std::to_string(eps.home.size()));
    for (size_t t = 0; t < eps.home.size(); ++t) {
        sd.addBuf(kEpTagPrefix + std::to_string(t), eps.home[t].data(), eps.home[t].size());
    }
    sd.addStr(kRailsTag, std::to_string(eps.data_rails.size()));
    for (size_t i = 0; i < eps.data_rails.size(); ++i) {
        sd.addStr(kRailTagPrefix + std::to_string(i), std::to_string(eps.data_rails[i]));
        sd.addBuf(kDataEpTagPrefix + std::to_string(i), eps.data[i].data(), eps.data[i].size());
    }
    sd.addStr(kIncarnationTag, std::to_string(eps.incarnation));
    return sd.exportStr();
}

namespace detail {
    inline bool
    getName(nixlSerDes &sd, const std::string &tag, std::string &name) {
        const ssize_t len = sd.getBufLen(tag);
        if (len <= 0 || static_cast<size_t>(len) > LF_EP_NAME_MAX_LEN) {
            return false;
        }
        name.assign(static_cast<size_t>(len), '\0');
        return sd.getBuf(tag, name.data(), len) == NIXL_SUCCESS;
    }

    inline bool
    getCount(nixlSerDes &sd, const std::string &tag, uint64_t &value) {
        try {
            size_t used = 0;
            const std::string text = sd.getStr(tag);
            value = std::stoull(text, &used);
            return used == text.size();
        }
        catch (const std::exception &) {
            return false;
        }
    }
} // namespace detail

/**
 * Parse a proxy blob; an empty blob means no proxy. On error @p eps is empty:
 * NIXL_ERR_MISMATCH for a malformed blob or another protocol version.
 */
inline nixl_status_t
parse(const std::string &blob, ProxyEps &eps) {
    eps = ProxyEps{};
    if (blob.empty()) {
        return NIXL_SUCCESS;
    }
    nixlSerDes sd;
    if (sd.importStr(blob) != NIXL_SUCCESS) {
        return NIXL_ERR_MISMATCH;
    }
    if (sd.getStr(kVersionTag) != std::to_string(nixlLibfabricProxyWire::kVersion)) {
        return NIXL_ERR_MISMATCH;
    }
    uint64_t threads = 0;
    if (!detail::getCount(sd, kThreadsTag, threads) || threads == 0 || threads > kMaxRails) {
        return NIXL_ERR_MISMATCH;
    }
    ProxyEps out;
    for (uint64_t t = 0; t < threads; ++t) {
        std::string name;
        if (!detail::getName(sd, kEpTagPrefix + std::to_string(t), name)) {
            return NIXL_ERR_MISMATCH;
        }
        out.home.push_back(std::move(name));
    }
    uint64_t rails = 0;
    if (!detail::getCount(sd, kRailsTag, rails) || rails == 0 || rails > kMaxRails) {
        return NIXL_ERR_MISMATCH; // every proxy receives puts on some rail
    }
    for (uint64_t i = 0; i < rails; ++i) {
        uint64_t rail = 0;
        std::string name;
        if (!detail::getCount(sd, kRailTagPrefix + std::to_string(i), rail) || rail >= kMaxRails ||
            !detail::getName(sd, kDataEpTagPrefix + std::to_string(i), name)) {
            return NIXL_ERR_MISMATCH;
        }
        out.data_rails.push_back(static_cast<uint32_t>(rail));
        out.data.push_back(std::move(name));
    }
    if (!detail::getCount(sd, kIncarnationTag, out.incarnation)) {
        return NIXL_ERR_MISMATCH;
    }
    eps = std::move(out);
    return NIXL_SUCCESS;
}

/** A published name, zero-padded to the fixed size the AV insert reads from. */
inline EpName
paddedName(const std::string &name) {
    EpName out{};
    std::memcpy(out.data(), name.data(), std::min(name.size(), out.size()));
    return out;
}

/** Append a proxy blob to the engine's connection info. */
inline std::string
join(const std::string &engine_part, const std::string &proxy_part) {
    std::string out = engine_part;
    out += proxy_part;
    const uint64_t len = proxy_part.size();
    out.append(reinterpret_cast<const char *>(&len), sizeof(len));
    out.append(kMagic, kMagicLen);
    return out;
}

/** Split connection info into its engine and proxy parts; no trailer: all engine. */
inline void
split(const std::string &in, std::string &engine_part, std::string &proxy_part) {
    engine_part = in;
    proxy_part.clear();
    const size_t trailer = sizeof(uint64_t) + kMagicLen;
    if (in.size() < trailer || in.compare(in.size() - kMagicLen, kMagicLen, kMagic) != 0) {
        return;
    }
    uint64_t len = 0;
    std::memcpy(&len, in.data() + in.size() - trailer, sizeof(len));
    if (len > in.size() - trailer) {
        return;
    }
    const size_t start = in.size() - trailer - len;
    engine_part = in.substr(0, start);
    proxy_part = in.substr(start, len);
}

} // namespace nixlLibfabricProxyConnInfo

#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_CONNINFO_H
