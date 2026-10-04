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
#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_PROFILE_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_PROFILE_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/**
 * Opt-in stage timers for the EFA device proxy (NIXL_EFA_PROXY_PROFILE=1).
 *
 * Each proxy thread owns one instance and is its only writer. Durations go into
 * histograms with 50 ns buckets up to 100 us and power-of-two buckets above, so
 * recording is a few instructions and percentiles are exact to 50 ns where
 * latency matters.
 */
class nixlLibfabricProxyProfile {
public:
    enum Op : uint8_t { PUT, ATOMIC, OP_COUNT };

    /** Life of one request, from the runtime's submit() to its terminal check_completion(). */
    enum Stage : uint8_t {
        SUBMIT_TO_POST, // proxy work (and, for an atomic, the fence wait) before the last post
        POST_TO_DONE, // libfabric, NIC and network until the last completion is read
        DONE_TO_COLLECT, // until the runtime asks for the status and publishes it
        RESIDENCE, // submit to collect
        STAGE_COUNT
    };

    /** Per-thread loop costs. */
    enum Loop : uint8_t {
        PASS_PERIOD, // between the thread's successive progress passes
        CQ_POLL, // one pollCqs() sweep
        POST_CALL, // one fi_writemsg()/fi_sendmsg() call, including -FI_EAGAIN
        ATOMIC_APPLY, // target side: applying one received atomicAdd
        RETRY_WAIT, // a back-pressured post's time in the thread's retry queue
        POST_EAGAIN, // post calls that returned -FI_EAGAIN (n = how many)
        LOOP_COUNT
    };

    static constexpr size_t kSizeClasses = 5;

    static uint64_t
    now() {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }

    /** <= 64 B, <= 4 KiB, <= 64 KiB, <= 1 MiB, larger. */
    static size_t
    sizeClass(size_t bytes) {
        return bytes <= 64 ? 0 : bytes <= 4096 ? 1 : bytes <= 65536 ? 2 : bytes <= 1048576 ? 3 : 4;
    }

    void
    add(Op op, size_t size_class, Stage stage, uint64_t ns) {
        stages_[op][size_class][stage].add(ns);
    }

    void
    add(Loop loop, uint64_t ns) {
        loops_[loop].add(ns);
    }

    void
    merge(const nixlLibfabricProxyProfile &other) {
        for (size_t o = 0; o < OP_COUNT; ++o) {
            for (size_t c = 0; c < kSizeClasses; ++c) {
                for (size_t s = 0; s < STAGE_COUNT; ++s) {
                    stages_[o][c][s].merge(other.stages_[o][c][s]);
                }
            }
        }
        for (size_t l = 0; l < LOOP_COUNT; ++l) {
            loops_[l].merge(other.loops_[l]);
        }
    }

    /** One line per non-empty series. */
    std::vector<std::string>
    report() const {
        static const char *ops[] = {"put", "atomic"};
        static const char *classes[] = {"<=64B", "<=4KiB", "<=64KiB", "<=1MiB", ">1MiB"};
        static const char *stages[] = {"submit->post", "post->done", "done->collect", "residence"};
        static const char *loops[] = {
            "pass period", "cq poll", "post call", "atomic apply", "retry wait", "post -FI_EAGAIN"};
        std::vector<std::string> lines;
        for (size_t o = 0; o < OP_COUNT; ++o) {
            for (size_t c = 0; c < kSizeClasses; ++c) {
                for (size_t s = 0; s < STAGE_COUNT; ++s) {
                    const Series &series = stages_[o][c][s];
                    if (series.count != 0) {
                        lines.push_back(std::string(ops[o]) + " " + classes[c] + " " + stages[s] +
                                        ": " + series.summary());
                    }
                }
            }
        }
        for (size_t l = 0; l < LOOP_COUNT; ++l) {
            if (loops_[l].count != 0) {
                lines.push_back(std::string(loops[l]) + ": " + loops_[l].summary());
            }
        }
        return lines;
    }

private:
    class Series {
    public:
        uint64_t count = 0;

        void
        add(uint64_t ns) {
            if (buckets_.empty()) {
                buckets_.assign(kBuckets, 0);
            }
            ++buckets_[bucket(ns)];
            ++count;
            sum_ += ns;
            max_ = std::max(max_, ns);
        }

        void
        merge(const Series &other) {
            if (other.count == 0) {
                return;
            }
            if (buckets_.empty()) {
                buckets_.assign(kBuckets, 0);
            }
            for (size_t i = 0; i < kBuckets; ++i) {
                buckets_[i] += other.buckets_[i];
            }
            count += other.count;
            sum_ += other.sum_;
            max_ = std::max(max_, other.max_);
        }

        std::string
        summary() const {
            std::ostringstream out;
            out << std::fixed << std::setprecision(2) << "n=" << count
                << " mean=" << double(sum_) / count / 1e3 << "us p50=" << percentile(0.50) / 1e3
                << "us p90=" << percentile(0.90) / 1e3 << "us p99=" << percentile(0.99) / 1e3
                << "us max=" << double(max_) / 1e3 << "us";
            return out.str();
        }

    private:
        static constexpr uint64_t kLinearStepNs = 50;
        static constexpr size_t kLinearBuckets = 2000; // up to 100 us
        static constexpr size_t kBuckets = kLinearBuckets + 40;

        static size_t
        bucket(uint64_t ns) {
            if (ns < kLinearStepNs * kLinearBuckets) {
                return ns / kLinearStepNs;
            }
            size_t log = 0;
            for (uint64_t v = ns / (kLinearStepNs * kLinearBuckets); v > 1; v >>= 1) {
                ++log;
            }
            return std::min(kLinearBuckets + log, kBuckets - 1);
        }

        /** Upper edge of the bucket holding the q-quantile, in ns. */
        double
        percentile(double q) const {
            const uint64_t rank = static_cast<uint64_t>(q * (count - 1)) + 1;
            uint64_t seen = 0;
            for (size_t i = 0; i < kBuckets; ++i) {
                seen += buckets_[i];
                if (seen >= rank) {
                    if (i < kLinearBuckets) {
                        return double((i + 1) * kLinearStepNs);
                    }
                    return double(kLinearStepNs * kLinearBuckets) *
                        double(2ull << (i - kLinearBuckets));
                }
            }
            return double(max_);
        }

        std::vector<uint64_t> buckets_;
        uint64_t sum_ = 0;
        uint64_t max_ = 0;
    };

    std::array<std::array<std::array<Series, STAGE_COUNT>, kSizeClasses>, OP_COUNT> stages_;
    std::array<Series, LOOP_COUNT> loops_;
};

#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_PROFILE_H
