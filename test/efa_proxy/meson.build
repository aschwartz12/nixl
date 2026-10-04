# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Two-rank device-proxy benchmark and ordering check (see the file header).
efa_proxy_benchmark = executable(
    'nixl_efa_proxy_benchmark',
    sources: ['nixl_efa_proxy_benchmark.cu'],
    include_directories: [nixl_inc_dirs, nixl_device_api_inc_dirs,
                          include_directories('../gtest/device_api')],
    dependencies: [nixl_dep, cuda_dep],
    link_with: [nixl_build_lib],
    install: false,
)
