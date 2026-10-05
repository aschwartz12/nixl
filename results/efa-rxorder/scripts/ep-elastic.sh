#!/usr/bin/env bash
# Run nixl_ep's elastic test over LIBFABRIC (EFA device proxy) inside the
# PyTorch 25.11 container; one task per node. Node 0 hosts the TCP store and
# rank server; other nodes join it.
# Env: EP_PLAN (tests/elastic/*.json), EP_PROCS (processes per node), EP_MASTER
# (node 0's host name), EP_ARGS (extra elastic.py arguments), NIXL_EP_BACKEND.
set -uo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/ctr-env.sh"

export NIXL_EP_BACKEND=${NIXL_EP_BACKEND:-LIBFABRIC}
export FI_PROVIDER=efa
export NIXL_LOG_LEVEL=${NIXL_LOG_LEVEL:-WARN}
export FI_LOG_LEVEL=${FI_LOG_LEVEL:-warn}
py_dir=$(find "${install_dir}" -maxdepth 6 -type d -name nixl_ep_cu13 | head -1)
export PYTHONPATH=$(dirname "${py_dir}"):${source_dir}/src/bindings/python/nixl-meta

cd "${source_dir}/examples/device/ep" || exit 1
plan=${EP_PLAN:-no_expansion.json}
[[ ${plan} == /* ]] || plan=tests/elastic/${plan}
args=(tests/elastic/elastic.py --plan "${plan}" --num-processes "${EP_PROCS:-4}")
if [[ ${SLURM_NODEID:-0} != 0 ]]; then
    args+=(--tcp-server "${EP_MASTER}")
    sleep "${EP_JOIN_DELAY:-30}" # let node 0 start its servers and first phase
fi
# shellcheck disable=SC2206
args+=(${EP_ARGS:-})
echo "node=$(hostname) nodeid=${SLURM_NODEID:-0} master=${EP_MASTER:-self} backend=${NIXL_EP_BACKEND}" \
     "plan=${plan} procs=${EP_PROCS:-4} gpus=$(nvidia-smi -L | wc -l)" \
     "channels=${NIXL_EP_PROXY_CHANNELS:-default} workers=${NIXL_EP_PROXY_WORKER_COUNT:-default}" \
     "params=${NIXL_EP_BACKEND_PARAMS:-}"
python3 "${args[@]}"
rc=$?
echo "node=$(hostname) elastic.py exit=${rc}"
exit ${rc}
