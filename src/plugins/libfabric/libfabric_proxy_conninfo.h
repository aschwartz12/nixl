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
 * The proxy blob carries the proxy protocol version and the home EP name of
 * every proxy thread (targets of atomicAdd records). A peer without the proxy
 * sends only the engine blob, which split() returns unchanged; a peer with
 * another protocol version is rejected by parse(), so no atomicAdd is sent to it.
 */
namespace nixlLibfabricProxyConnInfo {

using EpName = std::array<char, LF_EP_NAME_MAX_LEN>;

inline constexpr char kMagic[] = "EFAPRXY1";
inline constexpr size_t kMagicLen = sizeof(kMagic) - 1;
inline constexpr char kVersionTag[] = "efa_proxy_version";
inline constexpr char kThreadsTag[] = "efa_proxy_threads";
inline constexpr char kEpTagPrefix[] = "efa_proxy_ep_";

/** Proxy blob for the given home EP names, one per proxy thread. */
inline std::string
serialize(const std::vector<EpName> &home_eps) {
    nixlSerDes sd;
    sd.addStr(kVersionTag, std::to_string(nixlLibfabricProxyWire::kVersion));
    sd.addStr(kThreadsTag, std::to_string(home_eps.size()));
    for (size_t t = 0; t < home_eps.size(); ++t) {
        sd.addBuf(kEpTagPrefix + std::to_string(t), home_eps[t].data(), home_eps[t].size());
    }
    return sd.exportStr();
}

/**
 * Parse a proxy blob; an empty blob means no proxy. On error @p home_eps is empty:
 * NIXL_ERR_MISMATCH for a malformed blob or another protocol version.
 */
inline nixl_status_t
parse(const std::string &blob, std::vector<EpName> &home_eps) {
    home_eps.clear();
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
    size_t count = 0;
    try {
        count = std::stoul(sd.getStr(kThreadsTag));
    }
    catch (const std::exception &) {
        return NIXL_ERR_MISMATCH;
    }
    for (size_t t = 0; t < count; ++t) {
        const std::string tag = kEpTagPrefix + std::to_string(t);
        EpName name{};
        const ssize_t len = sd.getBufLen(tag);
        if (len <= 0 || static_cast<size_t>(len) > name.size() ||
            sd.getBuf(tag, name.data(), len) != NIXL_SUCCESS) {
            home_eps.clear();
            return NIXL_ERR_MISMATCH;
        }
        home_eps.push_back(name);
    }
    return NIXL_SUCCESS;
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
