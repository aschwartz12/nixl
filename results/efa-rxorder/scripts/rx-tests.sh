#!/usr/bin/env bash
# Unit and single-node device gtests of a build (default: the rxorder study build).
# Run inside an allocation step with one GPU: srun --cpus-per-task=24 rx-tests.sh
# Env: BUILD_DIR, FILTER (device gtest filter, default *libfabric*), REPEAT (default 3).
set -uo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa
export BUILD_DIR=${BUILD_DIR:-${workspace}/build/nixl-rxorder}
echo "build=${BUILD_DIR} host=$(hostname)"
log_dir=${LOG_DIR:-${workspace}/milestones/06-implement-optimize-nixl-efa/logs/efa-cpu-proxy/rx-tests-${SLURM_JOB_ID:-local}}
mkdir -p "${log_dir}"
echo "full logs: ${log_dir}"
cat "${BUILD_DIR}/build-id.txt" 2>/dev/null

NIXL_PLUGIN_DIR=${BUILD_DIR}/test/gtest/mocks NIXL_LOG_LEVEL=WARN \
    "${script_dir}/run.sh" test/gtest/unit/unit 2>&1 | grep -E "FAILED|PASSED|tests from .* ran"
echo "unit-exit=${PIPESTATUS[0]}"

NIXL_LOG_LEVEL=WARN "${script_dir}/run.sh" test/gtest/gtest --gtest_filter="${FILTER:-*libfabric*}" 2>&1 |
    tee "${log_dir}/gtest.log" |
    grep -v close_shm_resources | grep -E "^\[|INFO|Failure|Expected|Which is|ATTENTION|receiver ordering|ProxyOrdering"
echo "gtest-exit=${PIPESTATUS[0]}"

NIXL_LOG_LEVEL=WARN "${script_dir}/run.sh" test/gtest/gtest --gtest_filter="*ProxyAtomic*:*ProxyFault*" \
    --gtest_repeat="${REPEAT:-3}" 2>&1 | tee "${log_dir}/repeat.log" |
    grep -E "PASSED|FAILED|ATTENTION|Failure|DeregisterDuring|receiver ordering"
echo "repeat-exit=${PIPESTATUS[0]}"
