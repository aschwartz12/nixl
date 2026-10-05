#!/usr/bin/env bash
#SBATCH --account=network_research_advdev
#SBATCH --partition=pool0
#SBATCH --job-name=efa-rxorder
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=milestones/06-implement-optimize-nixl-efa/logs/efa-cpu-proxy/rxorder-submission-%j.out
#SBATCH --error=milestones/06-implement-optimize-nixl-efa/logs/efa-cpu-proxy/rxorder-submission-%j.err

# Sender- vs receiver-side ordering study (branch efa-cpu-proxy-rxorder) on two
# p5 nodes, one GPU and one benchmark process per node. No --exclusive, --mem or
# job-level --cpus-per-task (the cluster sizes CPU/memory by GPU count); steps
# ask for the 24 CPUs of their GPU explicitly.
#
# Env:
#   MODES   ordering modes, run in this order within each repetition (default: base)
#             base      baseline build (BASE_BUILD, efa-cpu-proxy 5a762a8)
#             sender    study build, efa_proxy_ordering=sender (the default)
#             imm       study build, sender ordering + test-only immediate-data puts
#             imm4k     imm with 4096-entry data EP receive queues (efa_proxy_data_rx_size)
#             immpost   imm with 32 receives posted per data EP (do the writes consume them?)
#             receiver  study build, efa_proxy_ordering=receiver (ring-hashed counting thread)
#             rxrail    receiver, each put counted by the target thread of its rail
#             rxnoflush rxrail without the GPUDirect RDMA flush (unsafe; bounds its cost)
#             immrail   imm, each put counted by the target thread of its rail
#             fenceoff  study build, test-only fence off (ordering NOT guaranteed)
#   SUITES  suites per mode (default: "c1 c4 ord host")
#             c1    1 channel/thread: ping-pong, serial put and put+signal, pipelined
#             c4    4 channels/threads, puts only: serial and pipelined
#             ord   ordering check, 4 channels/threads, 2048 rounds
#             ordp  ord with NIXL_EFA_PROXY_PROFILE (proxy thread busy time per side)
#             host  host-path writes with the device proxy on (4 threads) and idle
#             conc  host-path writes while 4 channels pipeline 16 KiB device puts
#             sweep pipelined device puts of SWEEP_SIZES on SWEEP_THREADS, profiled
#             sweepc  sweep with host-path 1 MiB writes running concurrently
#   REPS    repetitions (default 3); LABEL names the log directory
#   BASE_BUILD / STUDY_BUILD  build directories
# Submit from the efa workspace root.

set -uo pipefail

workspace=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa
scripts=${workspace}/milestones/06-implement-optimize-nixl-efa/scripts/efa-cpu-proxy
label=${LABEL:-rxorder}
run_dir=${workspace}/milestones/06-implement-optimize-nixl-efa/logs/$(date +%F)-${label}-job-${SLURM_JOB_ID}
source_dir=${workspace}/nixl-efa-cpu-proxy
base_build=${BASE_BUILD:-${workspace}/build/nixl-efa-cpu-proxy}
study_build=${STUDY_BUILD:-${workspace}/build/nixl-rxorder}
bench=test/efa_proxy/nixl_efa_proxy_benchmark
modes=${MODES:-base}
suites=${SUITES:-c1 c4 ord host}
reps=${REPS:-3}
sweep_sizes=${SWEEP_SIZES:-4096 7168 16384 65536}
sweep_threads=${SWEEP_THREADS:-1 4}

mkdir -p "${run_dir}"
{
    echo "modes=${modes}"
    echo "suites=${suites}"
    echo "reps=${reps} sweep_sizes=${sweep_sizes} sweep_threads=${sweep_threads}"
    echo "base_build=${base_build}"
    cat "${base_build}/build-id.txt" 2>/dev/null
    echo "study_build=${study_build}"
    cat "${study_build}/build-id.txt" 2>/dev/null
    echo "worktree=$(git -C "${source_dir}" rev-parse HEAD) dirty=$(git -C "${source_dir}" status --porcelain | wc -l)"
} > "${run_dir}/config.txt"
git -C "${source_dir}" diff > "${run_dir}/worktree-diff.patch"
scontrol show job "${SLURM_JOB_ID}" > "${run_dir}/slurm-job.txt"
cp "$0" "${run_dir}/rxorder.sbatch" 2>/dev/null || cp "${scripts}/rxorder.sbatch" "${run_dir}/"
srun --ntasks-per-node=1 bash -c 'echo "== $(hostname)"; date --iso-8601=seconds; nvidia-smi -L;
  /opt/amazon/efa/bin/fi_info --version; cat /opt/amazon/efa_installed_packages | head -2;
  lsmod | grep gdrdrv' > "${run_dir}/environment.txt" 2>&1

# run <name> <build dir> <env assignments...>
run() {
    local name=$1 build=$2
    shift 2
    local coord=${run_dir}/coord-${name}
    rm -rf "${coord}"
    mkdir -p "${coord}"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --cpu-bind=cores \
         --gpus-per-task=1 --gpu-bind=single:1 \
         --output="${run_dir}/${name}-rank-%t.out" --error="${run_dir}/${name}-rank-%t.err" \
         env BUILD_DIR="${build}" NIXL_LOG_LEVEL=WARN FI_LOG_LEVEL=warn "$@" \
         timeout -s KILL 300 "${scripts}/run.sh" "${bench}" "${coord}"
    echo "${name} exit=$?" | tee -a "${run_dir}/summary.txt"
}

mode_build() {
    [[ $1 == base ]] && echo "${base_build}" || echo "${study_build}"
}

mode_params() {
    case $1 in
    base) echo "" ;;
    sender) echo "efa_proxy_ordering=sender" ;;
    imm) echo "efa_proxy_ordering=sender,efa_proxy_test_put_imm=1" ;;
    imm4k) echo "efa_proxy_ordering=sender,efa_proxy_test_put_imm=1,efa_proxy_data_rx_size=4096" ;;
    immpost) echo "efa_proxy_ordering=sender,efa_proxy_test_put_imm=1,efa_proxy_test_data_rx_post=32" ;;
    receiver) echo "efa_proxy_ordering=receiver" ;;
    rxrail) echo "efa_proxy_ordering=receiver,efa_proxy_rx_thread=rail" ;;
    rxnoflush) echo "efa_proxy_ordering=receiver,efa_proxy_rx_thread=rail,efa_proxy_rx_flush=0" ;;
    immrail) echo "efa_proxy_ordering=sender,efa_proxy_test_put_imm=1,efa_proxy_rx_thread=rail" ;;
    fenceoff) echo "efa_proxy_ordering=sender,efa_proxy_test_fence_off=1" ;;
    *) echo "unknown mode $1" >&2; exit 1 ;;
    esac
}

c4=(NIXL_EFA_PROXY_CHANNELS=4 NIXL_EFA_PROXY_WORKERS=4)
host=(NIXL_PROXY_BENCH_HOST_XFER=1 NIXL_PROXY_BENCH_PTHR_DELAY_US=10 NIXL_PROXY_BENCH_HOST_ROUNDS=50)
profile=(NIXL_EFA_PROXY_PROFILE=1 NIXL_LOG_LEVEL=INFO)

for rep in $(seq 1 "${reps}"); do
    for mode in ${modes}; do
        build=$(mode_build "${mode}")
        params=(NIXL_PROXY_BENCH_BACKEND_PARAMS="$(mode_params "${mode}")")
        for suite in ${suites}; do
            case ${suite} in
            c1)
                run "${mode}-c1-${rep}" "${build}" "${params[@]}" \
                    NIXL_EFA_PROXY_CHANNELS=1 NIXL_EFA_PROXY_WORKERS=1
                ;;
            c4)
                run "${mode}-c4-${rep}" "${build}" "${params[@]}" "${c4[@]}" \
                    NIXL_PROXY_BENCH_PUT_ONLY=1
                ;;
            ord)
                run "${mode}-ord-${rep}" "${build}" "${params[@]}" "${c4[@]}" \
                    NIXL_PROXY_BENCH_ORDERING_ROUNDS=2048 NIXL_PROXY_BENCH_ORDERING_ONLY=1
                ;;
            ordp)
                run "${mode}-ordp-${rep}" "${build}" "${params[@]}" "${c4[@]}" \
                    NIXL_PROXY_BENCH_ORDERING_ROUNDS=2048 NIXL_PROXY_BENCH_ORDERING_ONLY=1 \
                    "${profile[@]}"
                ;;
            host)
                run "${mode}-host-${rep}" "${build}" "${params[@]}" "${c4[@]}" "${host[@]}" \
                    NIXL_PROXY_BENCH_PUT_ONLY=1
                ;;
            conc)
                run "${mode}-conc-${rep}" "${build}" "${params[@]}" "${c4[@]}" \
                    NIXL_PROXY_BENCH_PUT_ONLY=1 NIXL_PROXY_BENCH_SKIP_SERIAL=1 \
                    NIXL_PROXY_BENCH_PIPE_SIZES=16384 NIXL_PROXY_BENCH_PIPE_BYTES=4294967296 \
                    NIXL_PROXY_BENCH_CONCURRENT_HOST=1 NIXL_PROXY_BENCH_PTHR_DELAY_US=10 \
                    "${profile[@]}"
                ;;
            sweep | sweepc)
                conc=()
                if [[ ${suite} == sweepc ]]; then
                    conc=(NIXL_PROXY_BENCH_CONCURRENT_HOST=1 NIXL_PROXY_BENCH_PTHR_DELAY_US=10)
                fi
                for threads in ${sweep_threads}; do
                    for size in ${sweep_sizes}; do
                        run "${mode}-${suite}-t${threads}-${size}-${rep}" "${build}" "${params[@]}" \
                            NIXL_EFA_PROXY_CHANNELS="${threads}" NIXL_EFA_PROXY_WORKERS="${threads}" \
                            NIXL_PROXY_BENCH_PUT_ONLY=1 NIXL_PROXY_BENCH_SKIP_SERIAL=1 \
                            NIXL_PROXY_BENCH_PIPE_SIZES="${size}" \
                            NIXL_PROXY_BENCH_PIPE_BYTES=${SWEEP_BYTES:-4294967296} \
                            NIXL_PROXY_BENCH_PIPE_TRIALS=3 "${profile[@]}" "${conc[@]}" ${SWEEP_ENV:-}
                    done
                done
                ;;
            *)
                echo "unknown suite ${suite}" >&2
                ;;
            esac
        done
    done
done

sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList \
    > "${run_dir}/slurm-accounting.txt"
