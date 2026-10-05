# Environment for building and running the efa-cpu-proxy worktree (with
# nixl_ep) inside an NGC PyTorch container. Source inside the container.
#   srun ... --container-image=${ctr_image} --container-mounts=/lustre:/lustre bash -c '...'
# The container's CUDA 13.0 runs on the host's driver 575 through forward
# compatibility; it brings pybind11, gdrapi.h and libfabric 2.1.0amzn5.0. Its
# UCX (1.20) lacks UCP_ERR_HANDLING_MODE_FAILOVER, so the UCX plugin is off by default.

workspace=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa
source_dir=${SOURCE_DIR:-${workspace}/nixl-efa-cpu-proxy}
build_dir=${BUILD_DIR:-${workspace}/build/nixl-efa-cpu-proxy-ctr}
install_dir=${INSTALL_DIR:-${build_dir}-install}
efa_dir=/opt/amazon/efa
ucx_dir=/opt/hpcx/ucx
gtest_dir=${workspace}/build/deps/googletest-install

export PATH=${workspace}/build/deps/ctr-bin:/usr/local/cuda/bin:${PATH}
export PYTHONPATH=${workspace}/build/deps/meson-1.9.1${PYTHONPATH:+:${PYTHONPATH}}
export PKG_CONFIG_PATH=${workspace}/build/ctr-pkgconfig:${gtest_dir}/lib/pkgconfig:${efa_dir}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}
lib_dir=${install_dir}/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=${lib_dir}:${efa_dir}/lib:${ucx_dir}/lib:${gtest_dir}/lib:${workspace}/build/deps/hwloc-host-2.10.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:${workspace}/build/deps/numa-dev/root/usr/lib/x86_64-linux-gnu
export NIXL_PLUGIN_DIR=${lib_dir}/plugins

meson_bin=${workspace}/build/deps/meson-1.9.1/bin/meson-any
ninja_bin=${workspace}/build/deps/python-build-tools/bin/ninja
