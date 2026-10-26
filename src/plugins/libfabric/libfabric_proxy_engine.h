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

#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_ENGINE_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_ENGINE_H

#include <memory>

#include "libfabric_backend.h"

namespace nixl {
class deviceOps;
class proxyRuntime;
struct proxyConfig;
} // namespace nixl

namespace nixl::libfabric {
class proxyEngine final : public nixlLibfabricEngine {
public:
    proxyEngine(const nixlBackendInitParams &init_params, const nixl::proxyConfig &config);

    ~proxyEngine() override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist,
                nixlMemViewH &mvh,
                const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist,
                nixlMemViewH &mvh,
                const nixl_opt_b_args_t *opt_args = nullptr) const override;

    void
    releaseMemView(nixlMemViewH mem_view) const override;

private:
    class transport;

    template<typename DlistT>
    nixl_status_t
    prepProxyMemView(const DlistT &dlist, nixlMemViewH &mvh) const;

    nixl::deviceOps &ops_;
    transport *transport_ = nullptr;
    std::unique_ptr<nixl::proxyRuntime> runtime_;
};
} // namespace nixl::libfabric

#endif
