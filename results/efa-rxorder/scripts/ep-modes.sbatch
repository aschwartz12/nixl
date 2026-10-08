#!/bin/bash
# nixl_ep (static 8-rank plan) per ordering mode on the same two nodes, modes
# interleaved within each repetition. Submit with --nodes=2 --gpus-per-node=4.
# Env: REPS (default 3), EP_MODES (default: base sender receiver rxrail fenceoff),
# BASE_INSTALL / STUDY_INSTALL, EP_PLAN, EP_STEP_MINUTES (per-run limit, default 5).
#SBATCH --account=network_research_advdev
#SBATCH --partition=pool0
#SBATCH --job-name=ep-modes
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
set -uo pipefail
script_dir=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/milestones/06-implement-optimize-nixl-efa/scripts/efa-cpu-proxy
build=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build
image=/lustre/fsw/portfolios/general/projects/general_cs_infra/reframe_container_cache/dl+dgx+pytorch+25.11-py3-base-x86_64.sqsh
base_install=${BASE_INSTALL:-${build}/nixl-efa-cpu-proxy-ctr-install}
study_install=${STUDY_INSTALL:-${build}/nixl-rxorder-ctr-install}
gpus=${SLURM_GPUS_PER_NODE:-4}
gpus=${gpus##*:}
export EP_PROCS=${EP_PROCS:-${gpus}}
export EP_MASTER=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
export EP_PLAN=${EP_PLAN:-${script_dir}/ep-plans/static_8.json}
export NIXL_EP_PROXY_CHANNELS=4 NIXL_EP_PROXY_WORKER_COUNT=4
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} gpus/node=${gpus} plan=${EP_PLAN}"
echo "base_install=${base_install} ($(cat "${base_install}/build-id.txt" 2>/dev/null))"
echo "study_install=${study_install} ($(cat "${study_install}/build-id.txt" 2>/dev/null))"

params_of() {
    case $1 in
    base | new) echo "" ;;
    basering | newring) echo "efa_proxy_rail_policy=ring" ;;
    newctl) echo "" ;; # + NIXL_EFA_PROXY_INJECT=ctl_on_data=1 (below)
    newrx64) echo "" ;; # + NIXL_EFA_PROXY_INJECT=msg_rx=64 (below)
    newpriv) echo "efa_proxy_tx_domain=private" ;;
    newprivring) echo "efa_proxy_tx_domain=private,efa_proxy_rail_policy=ring" ;;
    newflush) echo "efa_proxy_rx_flush=1" ;;
    sender) echo "efa_proxy_ordering=sender" ;;
    receiver) echo "efa_proxy_ordering=receiver" ;;
    rxrail) echo "efa_proxy_ordering=receiver,efa_proxy_rx_thread=rail" ;;
    rxnoflush) echo "efa_proxy_ordering=receiver,efa_proxy_rx_thread=rail,efa_proxy_rx_flush=0" ;;
    fenceoff) echo "efa_proxy_ordering=sender,efa_proxy_test_fence_off=1" ;;
    *) echo "unknown mode $1" >&2; exit 1 ;;
    esac
}

rc=0
for rep in $(seq 1 "${REPS:-3}"); do
    for mode in ${EP_MODES:-base sender receiver rxrail fenceoff}; do
        if [[ ${mode} == base* ]]; then
            export INSTALL_DIR=${base_install}
        else
            export INSTALL_DIR=${study_install}
        fi
        export NIXL_EP_BACKEND_PARAMS=$(params_of "${mode}") EP_RUN_LABEL=${mode}-${rep}
        if [[ ${mode} == newctl ]]; then
            export NIXL_EFA_PROXY_INJECT=ctl_on_data=1 # control messages behind the puts
        elif [[ ${mode} == newrx64 ]]; then
            export NIXL_EFA_PROXY_INJECT=msg_rx=64 # small receive queues on the data EPs
        else
            unset NIXL_EFA_PROXY_INJECT
        fi
        echo "=== mode=${mode} rep=${rep} install=${INSTALL_DIR} params=${NIXL_EP_BACKEND_PARAMS}"
        # A hung run (e.g. a lost handshake) ends at the step limit, not the job's.
        srun --time="${EP_STEP_MINUTES:-5}" --ntasks-per-node=1 --cpus-per-task=$((24 * gpus)) --container-image="${image}" \
            --container-mounts=/lustre:/lustre,/dev/infiniband:/dev/infiniband \
            "${script_dir}/ep-elastic.sh" || rc=1
    done
done
exit ${rc}
