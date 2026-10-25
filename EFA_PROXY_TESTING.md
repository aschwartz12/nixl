# EFA device proxy: build, run and test

Branch `efa-proxy-test` of https://github.com/aschwartz12/nixl is the reviewed branch
`efa-proxy-dev` (PR to `hbadihi/nixl` `dev`) plus three testing-only commits:

| Commit | Content | In the PR? |
|---|---|---|
| nixlbench: build and configure the device proxy independently of UCX | nixlbench device API without UCX's GPU-side API | yes |
| libfabric: device proxy on the engine's rails (put, atomicAdd) | the proxy | yes |
| nixlbench: device API over the LIBFABRIC CPU proxy | `--backend=LIBFABRIC` in proxy mode | yes |
| nixl ep: let EP use the LIBFABRIC backend | `NIXL_EP_BACKEND=LIBFABRIC` | yes |
| test: LIBFABRIC device proxy gtests | ordering, atomicAdd, single write; hwloc dep for a unit test | no (testing only) |
| nixl ep: build on the NGC 25.11 container's PyTorch 2.10 pre-release | stream-shim fallback, a no-op on PyTorch >= 2.10 final | no (testing only) |
| this file | | no |

## Requirements

- AWS EFA nodes (tested on p5.48xlarge: 8x H100, 4 EFA NICs per GPU), `/dev/infiniband` visible
  (mount it into containers).
- CUDA >= 12.8 with nvcc, GPU arch 90 for H100.
- libfabric with the `efa` provider (`/opt/amazon/efa` from the EFA installer, or a source build;
  results below used libfabric 2.7).
- GDRCopy (`gdrapi.h` + `libgdrapi`): the target applies atomicAdds to GPU memory with it. Without
  it the build works but atomicAdds to GPU memory fail.
- hwloc, numa, meson >= 1.x, ninja, googletest (for `-Dbuild_tests=true`).
- nixl_ep: PyTorch >= 2.10 (the 2.10 pre-release in NGC 25.11 works through the fallback commit),
  pybind11.
- One process per GPU (see "Limits").

## Build

```bash
git clone https://github.com/aschwartz12/nixl.git && cd nixl && git checkout efa-proxy-test

meson setup build . -Dbuildtype=debugoptimized -Dprefix=$PWD/install \
    -Denable_plugins=LIBFABRIC -Dlibfabric_path=/opt/amazon/efa \
    -Dnixl_cuda_arch_list=90 -Dbuild_tests=true -Dbuild_nixl_ep=true
ninja -C build install

meson setup build-nb benchmark/nixlbench -Dbuildtype=release -Dnixl_path=$PWD/install \
    -Ducx_device_cuda_arch_list=90 -Dbuild_tests=false
ninja -C build-nb
```

Check in the configure output:
- NIXL: `-DHAVE_GDRCOPY` in the LIBFABRIC plugin's compile flags
  (`ninja -C build -t commands src/plugins/libfabric/libplugin_LIBFABRIC.so | grep -o -- -DHAVE_GDRCOPY`).
- nixlbench summary: `NIXL Device API: YES` (`UCX GPU Device API` may be NO: not needed).

Runtime environment for all runs:

```bash
export LD_LIBRARY_PATH=$PWD/install/lib/x86_64-linux-gnu:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=$PWD/install/lib/x86_64-linux-gnu/plugins
export FI_PROVIDER=efa NIXL_LOG_LEVEL=WARN
```

## Test 1: gtests (one node, one GPU)

```bash
./build/test/gtest/gtest --gtest_filter='*libfabric*'
```

We run each test in its own process (`--gtest_list_tests`, then one `--gtest_filter=<test>` per
run, 180 s timeout each). Expected: all pass except
`libfabricProxy/ProxyAtomicTest.AddsOfOneRingStayOrdered/LIBFABRIC` (fails on our nodes for the
earlier PR build too; under investigation).

## Test 2: nixlbench, 8 KiB WRITE (two nodes, one GPU each)

Same command on both nodes; start node 0 first, node 1 within a few seconds (ASIO receive timeout
is 15 s). Pin each process to CPU cores of its GPU's NUMA node and give it >= 8 cores.

```bash
./build-nb/nixlbench --runtime_type=ASIO --asio_address=<node0 ip> --asio_port=<free port> \
  --backend=LIBFABRIC --use_device_api=1 --device_mode=proxy \
  --num_threads=1024 --device_channel_num=8 --device_proxy_threads=4 \
  --initiator_seg_type=VRAM --target_seg_type=VRAM --op_type=WRITE --scheme=pairwise \
  --start_block_size=8192 --max_block_size=8192 --start_batch_size=128 --max_batch_size=1024 \
  --num_iter=10000 --warmup_iter=1000
```

`--num_threads=1024` = 32 GPU warps posting, `--device_channel_num=8` proxy channels,
`--device_proxy_threads=4` = one proxy thread per GPU-local EFA NIC. It prints
`Device API mode: configuring 8 CPU proxy channels for 32 execution groups`.

## Test 3: nixl_ep, 2 ranks (two nodes, one GPU each)

Plan file `static_2.json`: `[[0, 1]]`.

```bash
export NIXL_EP_BACKEND=LIBFABRIC NIXL_EP_DEVICE_MODE=proxy NIXL_EP_NUM_CHANNELS=8 NIXL_EP_PROXY_THREADS=4
export PYTHONPATH=$(dirname $(find $PWD/install -maxdepth 6 -type d -name 'nixl_ep_cu*' | head -1)):$PWD/src/bindings/python/nixl-meta
cd examples/device/ep
# node 0
python3 tests/elastic/elastic.py --plan /path/static_2.json --num-processes 1 --num-experts-per-rank 8 --kineto
# node 1 (start ~30 s after node 0)
python3 tests/elastic/elastic.py --plan /path/static_2.json --num-processes 1 --num-experts-per-rank 8 --kineto --tcp-server <node0 hostname>
```

Each rank prints `Dispatch + combine bandwidth: X GB/s` and `Dispatch bandwidth ..., avg_t=... us |
Combine bandwidth ..., avg_t=... us`. Shape: 128 tokens, hidden 7168, top-8.

## Our results (p5.48xlarge H100, 2 nodes, libfabric 2.7, same nodes, 3 repetitions)

| Test | This branch | `efa-baseline` (hbadihi) |
|---|---|---|
| nixlbench 8 KiB, batch 128 / 256 / 512 / 1024 | 27–29 / 29–30 / 30 / 30–31 GB/s | ~28 / 29 / 30 / 30 GB/s |
| nixl_ep 2 ranks, GB/s/rank | 36.5–37.6 (dispatch ~240 us, combine ~340 us) | 29.5 (~340 / ~397 us) |
| nixl_ep 16 ranks (8 GPUs x 2 nodes, 256 experts), GB/s/rank | 5.0–5.5 | 1.78–1.84 (needs `FI_EFA_ENABLE_SHM_TRANSFER=0`; fails at connection setup without) |

- gtests: all LIBFABRIC proxy tests pass except `AddsOfOneRingStayOrdered` (see above).
- 8 KiB line rate (48 GB/s per GPU on p5) is not reached: the GPU-side command enqueue caps
  nixlbench at ~30 GB/s for any proxy (also the UCX proxy's runtime); not specific to this branch.

## Limits to know when testing

- A process's proxy reads only the EFA NICs of its own GPU (plus NIC 0 for handshakes). Puts into
  host memory or into another GPU's memory of the same target process can land on NICs it does not
  read: the data is written but not counted, so the following atomicAdd hangs. Use one GPU per
  process and GPU-memory targets (as nixl_ep and nixlbench do).
- An atomicAdd already sent when a put of its epoch fails is not aborted.
- More than 3 rails (endpoints) per NIC exceeds GPU registration resources (not used by default).
