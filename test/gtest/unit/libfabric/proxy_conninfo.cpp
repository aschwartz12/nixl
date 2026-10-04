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

#include <string>
#include <vector>

#include "libfabric_proxy_conninfo.h"

namespace {

namespace ci = nixlLibfabricProxyConnInfo;

ci::EpName
epName(unsigned seed) {
    ci::EpName name{};
    for (size_t i = 0; i < name.size(); ++i) {
        name[i] = static_cast<char>((seed * 131 + i * 7) & 0xff); // includes NUL bytes
    }
    return name;
}

/** An engine blob as the rail manager produces it (serdes, binary EP names). */
std::string
engineBlob() {
    nixlSerDes sd;
    sd.addStr("dest_num_rails", "2");
    const ci::EpName a = epName(100), b = epName(101);
    sd.addBuf("dest_ep_0", a.data(), a.size());
    sd.addBuf("dest_ep_1", b.data(), b.size());
    return sd.exportStr();
}

TEST(LibfabricProxyConnInfoTest, RoundTrip) {
    const std::vector<ci::EpName> homes = {epName(0), epName(1), epName(2), epName(3)};
    const std::string engine = engineBlob();
    const std::string wire = ci::join(engine, ci::serialize(homes));

    std::string engine_part, proxy_part;
    ci::split(wire, engine_part, proxy_part);
    EXPECT_EQ(engine_part, engine);

    std::vector<ci::EpName> parsed;
    ASSERT_EQ(ci::parse(proxy_part, parsed), NIXL_SUCCESS);
    EXPECT_EQ(parsed, homes);
}

TEST(LibfabricProxyConnInfoTest, PeerWithoutProxy) {
    const std::string engine = engineBlob();
    std::string engine_part, proxy_part = "stale";
    ci::split(engine, engine_part, proxy_part);
    EXPECT_EQ(engine_part, engine);
    EXPECT_TRUE(proxy_part.empty());

    std::vector<ci::EpName> parsed = {epName(9)};
    EXPECT_EQ(ci::parse(proxy_part, parsed), NIXL_SUCCESS);
    EXPECT_TRUE(parsed.empty());
}

TEST(LibfabricProxyConnInfoTest, ShortAndEmptyInputs) {
    for (const std::string in : {std::string(), std::string("x"), std::string(ci::kMagic)}) {
        std::string engine_part, proxy_part;
        ci::split(in, engine_part, proxy_part);
        EXPECT_EQ(engine_part, in);
        EXPECT_TRUE(proxy_part.empty());
    }
}

TEST(LibfabricProxyConnInfoTest, EmptyProxySection) {
    const std::string engine = engineBlob();
    const std::string wire = ci::join(engine, std::string());
    std::string engine_part, proxy_part;
    ci::split(wire, engine_part, proxy_part);
    EXPECT_EQ(engine_part, engine);
    EXPECT_TRUE(proxy_part.empty());

    std::vector<ci::EpName> parsed;
    EXPECT_EQ(ci::parse(ci::serialize({}), parsed), NIXL_SUCCESS);
    EXPECT_TRUE(parsed.empty());
}

TEST(LibfabricProxyConnInfoTest, ImpossibleLengthMeansNoTrailer) {
    // Ends with the magic, but the length field claims more bytes than exist.
    std::string wire = "abc";
    const uint64_t len = 1000;
    wire.append(reinterpret_cast<const char *>(&len), sizeof(len));
    wire.append(ci::kMagic, ci::kMagicLen);
    std::string engine_part, proxy_part;
    ci::split(wire, engine_part, proxy_part);
    EXPECT_EQ(engine_part, wire);
    EXPECT_TRUE(proxy_part.empty());
}

TEST(LibfabricProxyConnInfoTest, MalformedProxySectionsAreRejected) {
    const std::vector<ci::EpName> homes = {epName(0), epName(1)};
    const std::string good = ci::serialize(homes);

    std::vector<std::string> bad;
    bad.push_back("not a serdes blob");
    bad.push_back(good.substr(0, good.size() / 2)); // truncated
    {
        nixlSerDes sd; // claims three threads, carries two
        sd.addStr(ci::kThreadsTag, "3");
        sd.addBuf(std::string(ci::kEpTagPrefix) + "0", homes[0].data(), homes[0].size());
        sd.addBuf(std::string(ci::kEpTagPrefix) + "1", homes[1].data(), homes[1].size());
        bad.push_back(sd.exportStr());
    }
    {
        nixlSerDes sd; // thread count is not a number
        sd.addStr(ci::kThreadsTag, "many");
        bad.push_back(sd.exportStr());
    }
    {
        nixlSerDes sd; // EP name longer than any libfabric name
        const std::string long_name(LF_EP_NAME_MAX_LEN + 1, 'x');
        sd.addStr(ci::kThreadsTag, "1");
        sd.addBuf(std::string(ci::kEpTagPrefix) + "0", long_name.data(), long_name.size());
        bad.push_back(sd.exportStr());
    }

    for (size_t i = 0; i < bad.size(); ++i) {
        std::vector<ci::EpName> parsed = {epName(7)};
        EXPECT_EQ(ci::parse(bad[i], parsed), NIXL_ERR_MISMATCH) << "case " << i;
        EXPECT_TRUE(parsed.empty()) << "case " << i;
    }
}

TEST(LibfabricProxyConnInfoTest, ShortNamesArePadded) {
    // Providers may report names shorter than LF_EP_NAME_MAX_LEN.
    nixlSerDes sd;
    const char short_name[] = {'\x01', '\0', '\x02'};
    sd.addStr(ci::kThreadsTag, "1");
    sd.addBuf(std::string(ci::kEpTagPrefix) + "0", short_name, sizeof(short_name));
    std::vector<ci::EpName> parsed;
    ASSERT_EQ(ci::parse(sd.exportStr(), parsed), NIXL_SUCCESS);
    ASSERT_EQ(parsed.size(), 1u);
    ci::EpName expected{};
    std::copy(short_name, short_name + sizeof(short_name), expected.begin());
    EXPECT_EQ(parsed[0], expected);
}

} // namespace
