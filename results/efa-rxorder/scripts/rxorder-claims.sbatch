#!/usr/bin/env bash
#SBATCH --account=network_research_advdev
#SBATCH --partition=pool0
#SBATCH --job-name=rx-claims
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=milestones/06-implement-optimize-nixl-efa/logs/efa-cpu-proxy/rx-claims-submission-%j.out
#SBATCH --error=milestones/06-implement-optimize-nixl-efa/logs/efa-cpu-proxy/rx-claims-submission-%j.err

# Checks of the "adopt receiver-side ordering" recommendation, one allocation:
#  gin     NCCL GIN proxy put+signal ping-pong (NCCL 2.28.8, aws-ofi-nccl GIN, libfabric
#          2.5.1, container) interleaved with this proxy's ping-pong (sender, rxrail)
#  inject  ordering check under injected back-pressure (eagain_every=3) per mode:
#          does it catch the fence-off race, and do sender/receiver pass it?
#  value   ordering check with 8-byte value puts as signals ("PUT_VALUE")
#  valueinject  the same under injected back-pressure
# No --exclusive/--mem/job-level CPUs; steps use the 24 CPUs of their GPU.
# Env: PARTS (default "gin inject value"), REPS (default 3).
set -uo pipefail

workspace=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa
scripts=${workspace}/milestones/06-implement-optimize-nixl-efa/scripts/efa-cpu-proxy
run_dir=${workspace}/milestones/06-implement-optimize-nixl-efa/logs/$(date +%F)-rxorder-claims-job-${SLURM_JOB_ID}
study_build=${STUDY_BUILD:-${workspace}/build/nixl-rxorder}
bench=test/efa_proxy/nixl_efa_proxy_benchmark
parts=${PARTS:-gin inject value}
reps=${REPS:-3}
mkdir -p "${run_dir}"
{
    echo "parts=${parts} reps=${reps}"
    echo "study_build=${study_build} $(cat "${study_build}/build-id.txt" 2>/dev/null)"
} > "${run_dir}/config.txt"
srun --ntasks-per-node=1 bash -c 'echo "== $(hostname)"; nvidia-smi -L; /opt/amazon/efa/bin/fi_info --version | head -1' \
    > "${run_dir}/environment.txt" 2>&1

# This proxy's benchmark: run <name> <env...>
run() {
    local name=$1
    shift
    local coord=${run_dir}/coord-${name}
    rm -rf "${coord}"
    mkdir -p "${coord}"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --cpu-bind=cores \
         --gpus-per-task=1 --gpu-bind=single:1 \
         --output="${run_dir}/${name}-rank-%t.out" --error="${run_dir}/${name}-rank-%t.err" \
         env BUILD_DIR="${study_build}" NIXL_LOG_LEVEL=WARN FI_LOG_LEVEL=warn "$@" \
         timeout -s KILL 300 "${scripts}/run.sh" "${bench}" "${coord}"
    echo "${name} exit=$?" | tee -a "${run_dir}/summary.txt"
}

params() {
    case $1 in
    new) echo "" ;;
    sender) echo "efa_proxy_ordering=sender" ;;
    rxrail) echo "efa_proxy_ordering=receiver,efa_proxy_rx_thread=rail" ;;
    fenceoff) echo "efa_proxy_ordering=sender,efa_proxy_test_fence_off=1" ;;
    esac
}

# NCCL GIN proxy ping-pong (milestone 3 build), one GPU per node.
nccl_out=$(ls -d ${workspace}/build/nccl-internal-v2.28.8-device-gin-job-* | tail -1)
ofi_install=$(readlink -f "${workspace}/build/aws-ofi-nccl-iad-latest")/install
lf_install=$(readlink -f "${workspace}/build/libfabric-iad-latest")/install
image=/lustre/fsw/portfolios/general/projects/general_cs_infra/reframe_container_cache/dl+dgx+pytorch+25.11-py3-base-x86_64.sqsh
gin() {
    local name=$1
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --cpu-bind=cores --mpi=pmix \
        --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
        --container-image="${image}" --container-mount-home \
        --container-mounts="${workspace}:${workspace},/dev/infiniband:/dev/infiniband" \
        --no-container-remap-root \
        --output="${run_dir}/${name}.out" --error="${run_dir}/${name}.err" \
        env LD_LIBRARY_PATH=${nccl_out}/lib:${ofi_install}/lib:${lf_install}/lib:/usr/local/cuda/compat/lib:/opt/amazon/efa/lib:/usr/local/mpi/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
        NCCL_NET_PLUGIN=ofi NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=2 OFI_NCCL_GIN_TYPE=PROXY \
        FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 \
        FI_HMEM_CUDA_USE_DMABUF=0 NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_MNNVL_ENABLE=0 \
        NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN \
        timeout -s KILL 300 "${nccl_out}/test/unit/put_signal_ping_pong_gin" -b 8 -e 8 -w 500 -i 5000
    echo "${name} exit=$?" | tee -a "${run_dir}/summary.txt"
}

ord=(NIXL_EFA_PROXY_CHANNELS=4 NIXL_EFA_PROXY_WORKERS=4 NIXL_PROXY_BENCH_ORDERING_ROUNDS=2048
     NIXL_PROXY_BENCH_ORDERING_ONLY=1)
latency=(NIXL_EFA_PROXY_CHANNELS=1 NIXL_EFA_PROXY_WORKERS=1 NIXL_PROXY_BENCH_SKIP_PIPELINED=1
         NIXL_PROXY_BENCH_MIN_SIZE=8 NIXL_PROXY_BENCH_PUT_ONLY=0)

for rep in $(seq 1 "${reps}"); do
    if [[ " ${parts} " == *" gin "* ]]; then
        gin "gin-pingpong-${rep}"
        for mode in sender rxrail; do
            run "${mode}-latency-${rep}" NIXL_PROXY_BENCH_BACKEND_PARAMS="$(params ${mode})" "${latency[@]}"
        done
    fi
    if [[ " ${parts} " == *" inject "* ]]; then
        for mode in ${INJECT_MODES:-fenceoff sender rxrail}; do
            run "${mode}-inject-${rep}" NIXL_PROXY_BENCH_BACKEND_PARAMS="$(params ${mode})" "${ord[@]}" \
                NIXL_EFA_PROXY_INJECT=eagain_every=3
        done
    fi
    if [[ " ${parts} " == *" valueinject "* ]]; then
        for mode in sender rxrail; do
            run "${mode}-valueinject-${rep}" NIXL_PROXY_BENCH_BACKEND_PARAMS="$(params ${mode})" "${ord[@]}" \
                NIXL_PROXY_BENCH_ORDERING_SIGNAL=put NIXL_EFA_PROXY_INJECT=eagain_every=3
        done
    fi
    if [[ " ${parts} " == *" value "* ]]; then
        for mode in sender rxrail; do
            run "${mode}-value-${rep}" NIXL_PROXY_BENCH_BACKEND_PARAMS="$(params ${mode})" "${ord[@]}" \
                NIXL_PROXY_BENCH_ORDERING_SIGNAL=put
        done
    fi
done
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList > "${run_dir}/slurm-accounting.txt"
