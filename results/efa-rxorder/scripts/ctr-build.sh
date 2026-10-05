#!/usr/bin/env bash
# Configure, build and install the efa-cpu-proxy worktree with nixl_ep inside
# an NGC PyTorch container.
# Usage (inside the container): [RECONFIGURE=1] [PLUGINS=LIBFABRIC] ctr-build.sh [ninja targets...]
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/ctr-env.sh"

echo "host=$(hostname) commit=$(git -C "${source_dir}" rev-parse --short HEAD 2>/dev/null) dirty=$(git -C "${source_dir}" status --porcelain 2>/dev/null | wc -l)"
plugins=${PLUGINS:-LIBFABRIC}
ucx_opt=-Ducx_path=
if [[ ${plugins} == *UCX* ]]; then
    ucx_opt=-Ducx_path=${ucx_dir}
    export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${ucx_dir}/lib/pkgconfig
fi
echo "nvcc: $(nvcc --version | tail -1); python: $(python3 --version); torch: $(python3 -c 'import torch; print(torch.__version__)')"
if [[ ! -f ${build_dir}/build.ninja || -n ${RECONFIGURE:-} ]]; then
    reconfigure=
    [[ -f ${build_dir}/build.ninja ]] && reconfigure=--reconfigure
    "${meson_bin}" setup "${build_dir}" "${source_dir}" ${reconfigure} \
        -Dbuildtype=debugoptimized \
        -Dprefix="${install_dir}" \
        -Dpkg_config_path="${PKG_CONFIG_PATH}" \
        -Denable_plugins=${plugins} \
        -Dlibfabric_path=${efa_dir} \
        ${ucx_opt} \
        -Dnixl_cuda_arch_list=90 \
        -Dbuild_tests=true -Dbuild_examples=false -Dbuild_nixl_ep=true \
        -Dwerror=false
fi

echo "--- proxy compile flags"
"${ninja_bin}" -C "${build_dir}" -t commands src/plugins/libfabric/libplugin_LIBFABRIC.so |
    grep 'libfabric_proxy.cpp' | grep -o -- '-DHAVE_[A-Z_]*' | sort -u | tr '\n' ' '
echo

if [[ $# -gt 0 ]]; then
    "${ninja_bin}" -C "${build_dir}" -k 0 "$@"
else
    "${ninja_bin}" -C "${build_dir}" -k 0 install
fi
