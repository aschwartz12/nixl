# Shared environment for the aschwartz/efa-cpu-proxy branch (engine-owned EFA
# device proxy on top of tomerdav codex/cpu-proxy-v2-rebase-6-ep-enable).
# Source on a pool0 compute node (GDRCopy headers/libs exist only there).

workspace=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa
source_dir=${SOURCE_DIR:-${workspace}/nixl-efa-cpu-proxy}
build_dir=${BUILD_DIR:-${workspace}/build/nixl-efa-cpu-proxy}
cuda_dir=/cm/shared/apps/cuda12.4/toolkit/12.4.1
efa_dir=/opt/amazon/efa
ucx_dir=/cm/shared/apps/hpcx/mlnx-ofed-cuda12/2.19/ucx

# gcc 13 by path: batch shells may not define `module`, and the system gcc 9
# lacks C++20 headers (<span>).
gcc_dir=/cm/local/apps/gcc/13.1.0

export CUDA_HOME=${cuda_dir}
export PATH=${workspace}/build/deps/python-build-tools/bin:${gcc_dir}/bin:${cuda_dir}/bin:${PATH}
export PYTHONPATH=${workspace}/build/deps/meson-1.9.1:${workspace}/build/deps/python-build-tools
gtest_dir=${workspace}/build/deps/googletest-install
export PKG_CONFIG_PATH=${workspace}/build/bench-pkgconfig:${gtest_dir}/lib/pkgconfig:${efa_dir}/lib/pkgconfig
export LD_LIBRARY_PATH=${build_dir}/src/infra:${build_dir}/src/core:${build_dir}/src/utils/common:${build_dir}/src/utils/serdes:${build_dir}/src/utils/stream:${build_dir}/src/utils/device:${build_dir}/src/utils/device/proxy:${build_dir}/src/utils/libfabric:${workspace}/build/deps/hwloc-host-2.10.0/lib:${workspace}/build/deps/numa-host/lib:${efa_dir}/lib:${gtest_dir}/lib:/cm/local/apps/gcc/13.1.0/lib64:${cuda_dir}/targets/x86_64-linux/lib

meson_bin=${workspace}/build/deps/meson-1.9.1/bin/meson
ninja_bin=${workspace}/build/deps/python-build-tools/bin/ninja
