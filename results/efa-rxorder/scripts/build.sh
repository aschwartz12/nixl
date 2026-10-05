#!/usr/bin/env bash
# Configure and build the efa-cpu-proxy worktree.
# Usage: [RECONFIGURE=1 [WIPE=1]] [PLUGINS=LIBFABRIC] build.sh [ninja targets...]
# No targets = everything. UCX needs >= 1.19 (HPC-X 1.17 on the host is too old),
# so the default build is LIBFABRIC-only.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/env.sh"

plugins=${PLUGINS:-LIBFABRIC}
ucx_opt=-Ducx_path=
if [[ ${plugins} == *UCX* ]]; then
    ucx_opt=-Ducx_path=${UCX_DIR:-${ucx_dir}}
fi

echo "host=$(hostname) commit=$(git -C "${source_dir}" rev-parse --short HEAD) dirty=$(git -C "${source_dir}" status --porcelain | wc -l)"
if [[ ! -f ${build_dir}/build.ninja || -n ${RECONFIGURE:-} ]]; then
    reconfigure=
    [[ -f ${build_dir}/build.ninja ]] && reconfigure=--reconfigure
    # WIPE=1 re-detects the compilers too (a --reconfigure keeps the cached ones).
    [[ -f ${build_dir}/build.ninja && -n ${WIPE:-} ]] && reconfigure=--wipe
    "${meson_bin}" setup "${build_dir}" "${source_dir}" ${reconfigure} \
        -Dbuildtype=debugoptimized \
        -Dpkg_config_path=${PKG_CONFIG_PATH} \
        -Denable_plugins=${plugins} \
        -Dlibfabric_path=${efa_dir} \
        ${ucx_opt} \
        -Dcudapath_inc=${cuda_dir}/targets/x86_64-linux/include \
        -Dcudapath_lib=${cuda_dir}/targets/x86_64-linux/lib \
        -Dcudapath_stub=${cuda_dir}/targets/x86_64-linux/lib/stubs \
        -Dnixl_cuda_arch_list=90 \
        -Dbuild_tests=true -Dbuild_examples=false \
        -Dwerror=false
fi

echo "--- proxy compile flags"
"${ninja_bin}" -C "${build_dir}" -t commands src/plugins/libfabric/libplugin_LIBFABRIC.so |
    grep 'libfabric_proxy.cpp' | grep -o -- '-DHAVE_[A-Z_]*' | sort -u | tr '\n' ' '
echo

"${ninja_bin}" -C "${build_dir}" -k 0 "$@"
