#!/bin/bash
# nixlbench between two p5 nodes, one GPU and one process each (pairwise, ASIO
# runtime: the first node binds the address and becomes rank 0). VRAM -> VRAM
# WRITE over LIBFABRIC, modes interleaved within each repetition.
# Submit with --nodes=2 --gpus-per-node=1.
# Env: INSTALL_DIR (NIXL install from ctr-build.sh), NIXLBENCH (binary; default
#      ${INSTALL_DIR}-nixlbench/nixlbench), MODES (default: host1 host4 dev1 dev4),
#      REPS (default 3), NB_ARGS (extra nixlbench arguments).
#   hostN: CPU-initiated transfers, N worker threads
#   devN:  device API over the CPU proxy, N GPU groups on N proxy channels/threads
#SBATCH --account=network_research_advdev
#SBATCH --partition=pool0
#SBATCH --job-name=nixlbench
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
set -uo pipefail
script_dir=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/milestones/06-implement-optimize-nixl-efa/scripts/efa-cpu-proxy
image=/lustre/fsw/portfolios/general/projects/general_cs_infra/reframe_container_cache/dl+dgx+pytorch+25.11-py3-base-x86_64.sqsh
: "${INSTALL_DIR:?set INSTALL_DIR to a ctr-build.sh install}"
export INSTALL_DIR
export NIXLBENCH=${NIXLBENCH:-${INSTALL_DIR}-nixlbench/nixlbench}
master=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
export NB_ADDR=$(srun --nodes=1 --ntasks=1 -w "${master}" hostname -i | awk '{print $1}')
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} master=${master} addr=${NB_ADDR}"
echo "install=${INSTALL_DIR} ($(cat "${INSTALL_DIR}/build-id.txt" 2>/dev/null))"

args_of() {
    case $1 in
    host*) echo "--num_threads=${1#host}" ;;
    dev*) echo "--use_device_api=1 --device_mode=proxy --num_threads=${1#dev} --device_channel_num=${1#dev}" ;;
    *) echo "unknown mode $1" >&2; exit 1 ;;
    esac
}

port=$((20000 + SLURM_JOB_ID % 20000))
rc=0
for rep in $(seq 1 "${REPS:-3}"); do
    for mode in ${MODES:-host1 host4 dev1 dev4}; do
        port=$((port + 1))
        export NB_MODE_ARGS="$(args_of "${mode}") ${NB_ARGS:-}" NB_PORT=${port}
        echo "=== mode=${mode} rep=${rep} args=${NB_MODE_ARGS}"
        srun --time=10 --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 \
            --container-image="${image}" \
            --container-mounts=/lustre:/lustre,/dev/infiniband:/dev/infiniband \
            bash -c 'source '"${script_dir}"'/ctr-env.sh
                [[ ${SLURM_PROCID} == 1 ]] && sleep 5 # rank 0 binds first
                exec "${NIXLBENCH}" --backend=LIBFABRIC --runtime_type=ASIO \
                    --asio_address="${NB_ADDR}" --asio_port="${NB_PORT}" \
                    --initiator_seg_type=VRAM --target_seg_type=VRAM --op_type=WRITE \
                    --scheme=pairwise ${NB_MODE_ARGS}' || rc=1
    done
done
exit ${rc}
