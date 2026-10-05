#!/bin/bash
# nixl_ep elastic test over the EFA device proxy, PROCS GPUs per node.
# Submit: sbatch --nodes=N --gpus-per-node=PROCS ep-elastic.sbatch
# Env: EP_PLAN (tests/elastic name or absolute path), EP_PROCS (default: GPUs per
# node), EP_ARGS, EP_SWEEP (see below), NIXL_EP_* knobs.
#SBATCH --account=network_research_advdev
#SBATCH --partition=pool0
#SBATCH --job-name=ep-elastic
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
set -uo pipefail
script_dir=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/milestones/06-implement-optimize-nixl-efa/scripts/efa-cpu-proxy
image=/lustre/fsw/portfolios/general/projects/general_cs_infra/reframe_container_cache/dl+dgx+pytorch+25.11-py3-base-x86_64.sqsh
gpus=${SLURM_GPUS_PER_NODE:-4}
gpus=${gpus##*:}
export EP_PROCS=${EP_PROCS:-${gpus}}
export EP_MASTER=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} gpus/node=${gpus} plan=${EP_PLAN:-no_expansion.json} master=${EP_MASTER}"
run() {
    srun --ntasks-per-node=1 --cpus-per-task=$((24 * gpus)) --container-image="${image}" \
        --container-mounts=/lustre:/lustre,/dev/infiniband:/dev/infiniband \
        "${script_dir}/ep-elastic.sh"
}
# EP_SWEEP="channels:workers[:backend params] ..." repeats the run on the same
# nodes per setting, e.g. "8:8 8:8:efa_proxy_rail_policy=ring". EP_INSTALLS="dir ..."
# repeats the sweep per install directory (A/B of builds on the same nodes).
if [[ -z ${EP_SWEEP:-} && -z ${EP_INSTALLS:-} ]]; then
    run
    exit $?
fi
rc=0
for install in ${EP_INSTALLS:-default}; do
    [[ ${install} == default ]] || export INSTALL_DIR=${install}
    for setting in ${EP_SWEEP:-::}; do
        IFS=: read -r channels workers params <<<"${setting}"
        if [[ -n ${channels} ]]; then
            export NIXL_EP_PROXY_CHANNELS=${channels} NIXL_EP_PROXY_WORKER_COUNT=${workers}
        fi
        export NIXL_EP_BACKEND_PARAMS=${params}
        echo "=== install=${install} channels=${channels} workers=${workers} params=${params}"
        run || rc=1
    done
done
exit ${rc}
