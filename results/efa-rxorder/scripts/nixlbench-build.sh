#!/usr/bin/env bash
# Configure and build nixlbench inside the NGC PyTorch container against a NIXL
# install made by ctr-build.sh (same SOURCE_DIR / INSTALL_DIR conventions).
# Usage (inside the container): SOURCE_DIR=... INSTALL_DIR=... NIXLBENCH_BUILD_DIR=... nixlbench-build.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/ctr-env.sh"

nb_build_dir=${NIXLBENCH_BUILD_DIR:-${install_dir}-nixlbench}
echo "host=$(hostname) commit=$(git -C "${source_dir}" rev-parse --short HEAD 2>/dev/null) nixl=${install_dir}"
if [[ ! -f ${nb_build_dir}/build.ninja || -n ${RECONFIGURE:-} ]]; then
    reconfigure=
    [[ -f ${nb_build_dir}/build.ninja ]] && reconfigure=--reconfigure
    "${meson_bin}" setup "${nb_build_dir}" "${source_dir}/benchmark/nixlbench" ${reconfigure} \
        -Dbuildtype=release \
        -Dnixl_path="${install_dir}" \
        -Dpkg_config_path="${PKG_CONFIG_PATH}" \
        -Ducx_device_cuda_arch_list=90 \
        -Dbuild_tests=false
fi
"${ninja_bin}" -C "${nb_build_dir}"
ls -l "${nb_build_dir}/nixlbench"
