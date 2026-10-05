#!/usr/bin/env bash
# Run a binary from the efa-cpu-proxy build with the runtime environment.
# Usage: [NIXL_LOG_LEVEL=...] run.sh <build-relative-or-absolute binary> [args...]
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/env.sh"

export NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR:-${build_dir}/src/plugins/libfabric}
export NIXL_LOG_LEVEL=${NIXL_LOG_LEVEL:-INFO}
export FI_LOG_LEVEL=${FI_LOG_LEVEL:-warn}
export FI_PROVIDER=${FI_PROVIDER:-efa}

bin=$1
shift
[[ ${bin} == /* ]] || bin=${build_dir}/${bin}
echo "host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-} commit=$(git -C "${source_dir}" rev-parse --short HEAD) dirty=$(git -C "${source_dir}" status --porcelain | wc -l)"
exec "${bin}" "$@"
