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

#include <set>
#include <string>
#include <vector>

#include "libfabric_proxy_conninfo.h"

namespace {

namespace ci = nixlLibfabricProxyConnInfo;
namespace wire = nixlLibfabricProxyWire;

/** An endpoint name of @p len bytes (EFA names are shorter than LF_EP_NAME_MAX_LEN). */
std::string
epName(unsigned seed, size_t len = 32) {
    std::string name(len, '\0');
    for (size_t i = 0; i < len; ++i) {
        name[i] = static_cast<char>((seed * 131 + i * 7) & 0xff); // includes NUL bytes
    }
    return name;
}

ci::ProxyEps
proxyEps(size_t threads = 4, size_t rails = 4) {
    ci::ProxyEps eps;
    for (size_t t = 0; t < threads; ++t) {
        eps.home.push_back(epName(static_cast<unsigned>(t)));
    }
    for (size_t i = 0; i < rails; ++i) {
        eps.data_rails.push_back(static_cast<uint32_t>(16 + i));
        eps.data.push_back(epName(static_cast<unsigned>(100 + i)));
    }
    eps.incarnation = 0xfedcba9876543210ull;
    return eps;
}

/** An engine blob as the rail manager produces it (serdes, binary EP names). */
std::string
engineBlob() {
    nixlSerDes sd;
    sd.addStr("dest_num_rails", "2");
    const std::string a = epName(200, LF_EP_NAME_MAX_LEN), b = epName(201, LF_EP_NAME_MAX_LEN);
    sd.addBuf("dest_ep_0", a.data(), a.size());
    sd.addBuf("dest_ep_1", b.data(), b.size());
    return sd.exportStr();
}

void
expectEqual(const ci::ProxyEps &a, const ci::ProxyEps &b) {
    EXPECT_EQ(a.home, b.home);
    EXPECT_EQ(a.data_rails, b.data_rails);
    EXPECT_EQ(a.data, b.data);
    EXPECT_EQ(a.incarnation, b.incarnation);
}

TEST(LibfabricProxyConnInfoTest, RoundTrip) {
    const ci::ProxyEps eps = proxyEps();
    const std::string engine = engineBlob();
    const std::string blob = ci::join(engine, ci::serialize(eps));

    std::string engine_part, proxy_part;
    ci::split(blob, engine_part, proxy_part);
    EXPECT_EQ(engine_part, engine);

    ci::ProxyEps parsed;
    ASSERT_EQ(ci::parse(proxy_part, parsed), NIXL_SUCCESS);
    expectEqual(parsed, eps);
}

TEST(LibfabricProxyConnInfoTest, PeerWithoutProxy) {
    const std::string engine = engineBlob();
    std::string engine_part, proxy_part = "stale";
    ci::split(engine, engine_part, proxy_part);
    EXPECT_EQ(engine_part, engine);
    EXPECT_TRUE(proxy_part.empty());

    ci::ProxyEps parsed = proxyEps();
    EXPECT_EQ(ci::parse(proxy_part, parsed), NIXL_SUCCESS);
    EXPECT_TRUE(parsed.home.empty());
    EXPECT_TRUE(parsed.data.empty());
}

TEST(LibfabricProxyConnInfoTest, ShortAndEmptyInputs) {
    for (const std::string in : {std::string(), std::string("x"), std::string(ci::kMagic)}) {
        std::string engine_part, proxy_part;
        ci::split(in, engine_part, proxy_part);
        EXPECT_EQ(engine_part, in);
        EXPECT_TRUE(proxy_part.empty());
    }
}

TEST(LibfabricProxyConnInfoTest, ImpossibleLengthMeansNoTrailer) {
    // Ends with the magic, but the length field claims more bytes than exist.
    std::string blob = "abc";
    const uint64_t len = 1000;
    blob.append(reinterpret_cast<const char *>(&len), sizeof(len));
    blob.append(ci::kMagic, ci::kMagicLen);
    std::string engine_part, proxy_part;
    ci::split(blob, engine_part, proxy_part);
    EXPECT_EQ(engine_part, blob);
    EXPECT_TRUE(proxy_part.empty());
}

// The handshake carries the connection info and is limited to 8 KiB: names are
// published at their length, one data EP per receive rail.
TEST(LibfabricProxyConnInfoTest, FitsTheHandshakeWithManyThreads) {
    const std::string blob = ci::serialize(proxyEps(/*threads=*/32, /*rails=*/4));
    EXPECT_LT(blob.size(), 4096u) << "proxy blob of " << blob.size() << " bytes";
}

TEST(LibfabricProxyConnInfoTest, MalformedProxySectionsAreRejected) {
    const ci::ProxyEps eps = proxyEps(2, 2);
    const std::string good = ci::serialize(eps);
    const auto base = [](const std::string &threads) {
        nixlSerDes sd;
        sd.addStr(ci::kVersionTag, std::to_string(wire::kVersion));
        sd.addStr(ci::kThreadsTag, threads);
        return sd;
    };

    std::vector<std::string> bad;
    bad.push_back("not a serdes blob");
    bad.push_back(good.substr(0, good.size() / 2)); // truncated
    {
        nixlSerDes sd = base("3"); // claims three threads, carries two
        sd.addBuf(std::string(ci::kEpTagPrefix) + "0", eps.home[0].data(), eps.home[0].size());
        sd.addBuf(std::string(ci::kEpTagPrefix) + "1", eps.home[1].data(), eps.home[1].size());
        bad.push_back(sd.exportStr());
    }
    {
        nixlSerDes sd = base("many"); // thread count is not a number
        bad.push_back(sd.exportStr());
    }
    {
        nixlSerDes sd = base("1"); // EP name longer than any libfabric name
        const std::string long_name(LF_EP_NAME_MAX_LEN + 1, 'x');
        sd.addBuf(std::string(ci::kEpTagPrefix) + "0", long_name.data(), long_name.size());
        bad.push_back(sd.exportStr());
    }
    {
        nixlSerDes sd = base("1"); // no receive rails: nowhere to put
        sd.addBuf(std::string(ci::kEpTagPrefix) + "0", eps.home[0].data(), eps.home[0].size());
        sd.addStr(ci::kRailsTag, "0");
        sd.addStr(ci::kIncarnationTag, "7");
        bad.push_back(sd.exportStr());
    }
    {
        nixlSerDes sd = base("1"); // claims two receive rails, carries one
        sd.addBuf(std::string(ci::kEpTagPrefix) + "0", eps.home[0].data(), eps.home[0].size());
        sd.addStr(ci::kRailsTag, "2");
        sd.addStr(std::string(ci::kRailTagPrefix) + "0", "16");
        sd.addBuf(std::string(ci::kDataEpTagPrefix) + "0", eps.data[0].data(), eps.data[0].size());
        sd.addStr(ci::kIncarnationTag, "7");
        bad.push_back(sd.exportStr());
    }

    for (size_t i = 0; i < bad.size(); ++i) {
        ci::ProxyEps parsed = eps;
        EXPECT_EQ(ci::parse(bad[i], parsed), NIXL_ERR_MISMATCH) << "case " << i;
        EXPECT_TRUE(parsed.home.empty()) << "case " << i;
        EXPECT_TRUE(parsed.data.empty()) << "case " << i;
    }
}

TEST(LibfabricProxyConnInfoTest, NamesArePaddedForTheAv) {
    const std::string name("\x01\0\x02", 3);
    const ci::EpName padded = ci::paddedName(name);
    EXPECT_EQ(std::string(padded.data(), 3), name);
    for (size_t i = 3; i < padded.size(); ++i) {
        EXPECT_EQ(padded[i], '\0');
    }
}

// A peer speaking another proxy protocol gets no device operations (parse fails,
// so the backend treats it as having no proxy).
TEST(LibfabricProxyConnInfoTest, OtherProtocolVersionIsRejected) {
    for (const uint16_t version : {uint16_t(wire::kVersion - 1), uint16_t(wire::kVersion + 1)}) {
        nixlSerDes sd;
        sd.addStr(ci::kVersionTag, std::to_string(version));
        sd.addStr(ci::kThreadsTag, "1");
        ci::ProxyEps parsed = proxyEps();
        EXPECT_EQ(ci::parse(sd.exportStr(), parsed), NIXL_ERR_MISMATCH) << "version " << version;
        EXPECT_TRUE(parsed.home.empty());
    }
}

TEST(LibfabricProxyWireTest, PutImmediateCarriesRingAndEpoch) {
    const uint32_t key = wire::ringKey(wire::kMaxSenderIndex, wire::kRingIds - 1);
    EXPECT_EQ(key, 0xffffffu);
    const uint32_t imm = wire::putImm(key, 5);
    EXPECT_EQ(wire::immRing(imm), key);
    EXPECT_EQ(wire::immSlot(imm), 5u);
    // Epochs wrap: epoch e + kEpochs uses e's slot.
    EXPECT_EQ(wire::immSlot(wire::putImm(7, 3 + wire::kEpochs)), 3u);
    EXPECT_EQ(wire::immRing(wire::putImm(7, 3 + wire::kEpochs)), 7u);
}

TEST(LibfabricProxyWireTest, RingKeysAreUniquePerSenderAndRing) {
    std::set<uint32_t> keys;
    for (uint32_t sender = 0; sender <= wire::kMaxSenderIndex; sender += 17) {
        for (uint32_t ring = 0; ring < wire::kRingIds; ring += 4099) {
            EXPECT_TRUE(keys.insert(wire::ringKey(sender, ring)).second)
                << "sender " << sender << " ring " << ring;
        }
    }
}

TEST(LibfabricProxyWireTest, RailsMapToThreads) {
    // Four receive rails over four threads: one each; over two: two each.
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(wire::railThread(i, 4), i);
        EXPECT_EQ(wire::railThread(i, 2), i % 2);
        EXPECT_EQ(wire::railThread(i, 1), 0u);
    }
}

TEST(LibfabricProxyWireTest, EveryCounterHasOneOwner) {
    // Deterministic, in range, and spread over the threads.
    std::vector<size_t> hits(4, 0);
    for (uint64_t addr = 0x7f0000000000ull; addr < 0x7f0000000000ull + 4096 * 8; addr += 8) {
        const uint32_t owner = wire::counterOwner(addr, 4);
        ASSERT_LT(owner, 4u);
        EXPECT_EQ(owner, wire::counterOwner(addr, 4));
        ++hits[owner];
    }
    for (size_t t = 0; t < hits.size(); ++t) {
        EXPECT_GT(hits[t], 512u) << "thread " << t;
    }
    static_assert(sizeof(wire::anyMsg) == sizeof(wire::atomicAddMsg));
    static_assert(sizeof(wire::atomicAddMsg) <= 128, "keep atomicAdd records small");
}

} // namespace
