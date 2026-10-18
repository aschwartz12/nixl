#!/bin/bash
# nixlbench between two p5 nodes, one GPU and one process each (pairwise, ASIO
# runtime: the first node binds the address and becomes rank 0). VRAM -> VRAM
# WRITE over EFA, modes interleaved within each repetition.
# Submit with --nodes=2 --gpus-per-node=1.
# Env: INSTALL_DIR (NIXL install from ctr-build.sh), NIXLBENCH (binary; default
#      ${INSTALL_DIR}-nixlbench/nixlbench), MODES (default: host1 host4 dev1 dev4),
#      REPS (default 3), NB_ARGS (extra nixlbench arguments).
#   hostN / devN:   LIBFABRIC: CPU-initiated transfers on N worker threads / device
#                   API over the CPU proxy, N GPU groups on N proxy channels/threads
#   uhostN / udevN: the same over UCX (EFA SRD transport; needs a build with the UCX
#                   plugin). UCX_MAX_RMA_RAILS / UCX_MAX_RNDV_RAILS (default here 4) and UCX_NET_DEVICES (default:
#                   the 4 EFA devices on the GPU's PCIe switch). SRD has no remote atomics:
#                   UCX applies them on the target CPU, so udevN keeps the completion
#                   counter in host memory (--device_counter_host=1).
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
    host*) echo "--backend=LIBFABRIC --num_threads=${1#host}" ;;
    dev*) echo "--backend=LIBFABRIC --use_device_api=1 --device_mode=proxy --num_threads=${1#dev} --device_channel_num=${1#dev}" ;;
    uhost*) echo "--backend=UCX --num_threads=${1#uhost}" ;;
    udev*) echo "--backend=UCX --use_device_api=1 --device_mode=proxy --num_threads=${1#udev} --device_channel_num=${1#udev} --device_counter_host=1" ;;
    *) echo "unknown mode $1" >&2; exit 1 ;;
    esac
}
export UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-4} UCX_MAX_RMA_RAILS=${UCX_MAX_RMA_RAILS:-4}

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
                export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/hpcx/ucx/lib
                if [[ ${NB_MODE_ARGS} == *--backend=UCX* && -z ${UCX_NET_DEVICES:-} ]]; then
                    gbus=$((16#$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | cut -d: -f2)))
                    devs=
                    for d in /sys/class/infiniband/*; do
                        b=$((16#$(basename "$(readlink -f "$d/device")" | cut -d: -f2)))
                        (( b < gbus && b >= gbus - 4 )) && devs+="${devs:+,}$(basename "$d"):1"
                    done
                    export UCX_NET_DEVICES=${devs}
                    echo "rank ${SLURM_PROCID}: UCX_NET_DEVICES=${UCX_NET_DEVICES}"
                fi
                exec "${NIXLBENCH}" --runtime_type=ASIO \
                    --asio_address="${NB_ADDR}" --asio_port="${NB_PORT}" \
                    --initiator_seg_type=VRAM --target_seg_type=VRAM --op_type=WRITE \
                    --scheme=pairwise ${NB_MODE_ARGS}' || rc=1
    done
done
exit ${rc}
