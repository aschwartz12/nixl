# NIXL Libfabric Plugin

This plugin provides a high-performance RDMA backend for NIXL using the OpenFabrics Interfaces (OFI) Libfabric library.

## Overview

The Libfabric plugin provides a high-performance RDMA communication backend with the following key capabilities:

- **Multi-Rail RDMA**: Automatic discovery and utilization of multiple network devices for increased bandwidth
- **GPU Direct Support**: Zero-copy transfers between GPU memory (VRAM) and remote systems with CUDA integration. GDR (GPU Direct RDMA) support is currently required.
- **Scalable Connection Management**: Efficient multi-agent connectivity with robust state tracking and automatic reconnection
- **Asynchronous Processing**: Non-blocking RDMA operations with pre-allocated request pools and completion processing
- **Thread-Safe Concurrency**: Background progress threads with lock-free data structures and configurable threading patterns
- **Topology-Aware Optimization**: Hardware-aware GPU-to-EFA and NUMA-to-EFA mapping using hwloc for optimal performance (EFA-specific)

## Dependencies

### Required Dependencies

- **Libfabric**
  - Many systems will have libfabric already installed. If not, custom libfabric installation is available via https://ofiwg.github.io/libfabric/ - Minimum required version: `v1.21.0`
  - For EFA enabled AWS instances, it is recommended to install through AWS EFA installer: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html - Recommend to use the latest version

- **hwloc**
  - hwloc is used to understand the underlying architecture to optimize application performance. Suggested version: 2.10.0 or newer

- **numa**
  - numa (libnuma-dev on Debian/Ubuntu or libnuma-devel on RPM-based systems) is required for supporting DRAM_SEG memory type NUMA-aware rail selection (for imposing NUMA-aware bandwidth limitation). Suggested version: 2.0.18 or newer.

### Network Hardware Requirements

Validated compatibility with:

- **AWS EFA** (Elastic Fabric Adapter)

Any other Libfabric providers should also work but have not been validated in production environments. Community validation and feedback are highly appreciated!

## Build Instructions

```bash
# Basic build setup with default options
$ meson setup <name_of_build_dir>

# Setup with custom options (example)
$ meson setup <name_of_build_dir> \
    -Dlibfabric_path=/path/to/libfabric

# Build and install
$ cd <name_of_build_dir>
$ ninja && ninja install
```

## Runtime Configuration

Following are the environment variables that control the runtime behavior of the plugin.

### NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG

Normally, DRAM_SEG memory type buffers should not use more bandwidth than the PCIe switches can
sustain, as buffers travel from host (main memory) to EFA device via PCIe topology.

For this reason, the plugin computes the maximum bandwidth limit that would cause the PCIe switches
on each NUMA node **not** to be saturated. This way when DRAM_SEG memory type is used, only a
limited number of rails is selected, such that PCIe congestion is avoided. The rail selection is
made only from the NUMA node of the origin memory buffer. This is because NUMA nodes interconnect
bandwidth is much smaller than the PCIe link, and it is counterproductive to stress the interconnect
for only reduced additional network bandwidth.

In case it is desired though to set a different bandwidth limit (e.g. when computed bandwidth limit
is not suitable on some PCIe topology), the user can override this computed value through the
environment variable NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG.

To summarize:

- NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG is used to configure NUMA-aware rail selection policy for
DRAM_SEG memory type registration
- It controls the bandwidth limit on DRAM_SEG memory type buffers
- It should be specified as decimal Gbps (Gigabits per second), e.g. 100, 200, 400, etc.
- If not specified, then it is computed as the maximum possible bandwidth that would not saturate
the topmost PCIe bridge/switch devices of the NUMA node of the origin buffer
- It can also be passed as a custom parameter during plugin/backend creation (see
nixlAgent::createBackend()), with key "max_bw_per_dram_seg"
- Environment variable override takes precedence over custom parameter configuration

Notes:

- The bandwidth limit is converted to a rail count limit. During memory registration phase of
DRAM_SEG memory type, a subset of rails is selected, such that the bandwidth limit is enforced
- The subset of rails being selected is made sure not to saturate any topmost PCIe switch of the NUMA node
- The subset of rails being selected is limited to the NUMA node of the origin buffer
- The subset of rails being selected each time uses different rails to ensure optimal resource utilization
- Rail selection is thread-safe
- If user override exceeds total topmost PCIe switch capacity, then additional rails are chosen from
the same NUMA node (while causing saturation of one or more topmost PCIe switches)
- If user override exceeds total capacity of EFA devices connected to the NUMA node, then additional
rails are selected from adjacent NUMA nodes, according to NUMA distance (i.e. rails from closer
nodes are selected first), while keeping the same effort to avoid saturating topmost PCIe bridges
- If user override exceeds total capacity of all EFA devices on the machine, then all rails will be
used for DRAM_SEG memory type

### Summary

The following table summarizes briefly the plugin's runtime configuration:

| Name | Effect | Configuration Source | Values | Examples | Notes |
|--|--|--|--|--|--|
| max_bw_per_dram_seg | Controls the bandwidth limit on DRAM_SEG memory type buffers per NUMA node | Backend init param or `NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG` environment variable | integer | 100, 200 | Units are Gbps (Gigabits per second), auto-computed by PCIe topology, normally does not require user override |
| num_threads | Enables a thread pool for parallel descriptor posting in postXfer | Backend init param | integer | 4, 8 | Default 0 keeps the serial posting path |
| split_batch_size | Minimum descriptor count before postXfer uses the posting thread pool | Backend init param | integer | 1024, 4096 | Default 1024; only applies when num_threads is greater than 0 |

## Device API (GPU-initiated transfers through the CPU proxy)

With the `device_proxy=true` backend parameter, the engine owns a device proxy
runtime, so GPU kernels can call the NIXL device API (`put()`, `atomicAdd()`) on
memory views prepared by this backend. Requires a CUDA-enabled build. The design,
wire encoding and thread/endpoint mapping are described in
[EFA_PROXY_RECEIVER_ORDERING.md](EFA_PROXY_RECEIVER_ORDERING.md).

- **Parameters** (shared with the UCX proxy): `device_proxy`, `proxy_channel_count`,
  `proxy_thread_count`, `proxy_max_peers`, `proxy_ring_depth` (at most 256 here),
  `proxy_pthr_delay_us`. EFA-specific: `efa_proxy_rail_policy` = `thread` (default:
  a buffer's rails are split among the proxy threads for unstriped puts) or `ring`;
  `efa_proxy_idle_poll_us` (default 2: how often a proxy thread with nothing to do
  polls its CQs; `0` polls on every pass); `efa_proxy_rx_flush` = `1`/`0` forces the
  GPUDirect RDMA flush before target-side adds on/off (default as NCCL: only before
  Hopper and on aarch64 hosts). Use one proxy thread per EFA device of the GPU.
- **Endpoints:** each proxy thread has a data EP (own CQ and AV) on every rail, in the
  rail's domain (registrations and keys are shared with the host path), and a control
  EP on one of the GPU's EFA devices. The rail domains are `FI_THREAD_SAFE` while the
  proxy is on. Each of the GPU's EFA devices is polled for incoming puts by exactly one
  proxy thread (rail `i` of the GPU by thread `i % threads`).
- **put():** one RDMA write per fragment (striped across rails at or above the striping
  threshold) with `FI_DELIVERY_COMPLETE` and 32 bits of remote CQ data (ring key and
  epoch) to the data EP of the target thread that receives on the destination rail. It
  completes once placed at the target.
- **atomicAdd():** sent at once to the counter's owner thread at the target, which
  applies it (GDRCopy, CUDA copies without GDRCopy, or a CPU atomic for host memory)
  only after the target counted every earlier put of the ring and applied the ring's
  previous atomicAdd, then acks; the atomicAdd completes on the ack. Ordering is per
  ring and target. The add is a read-modify-write: the GPU must not write a counter
  while remote adds to it can arrive; counters must be 8-byte aligned and registered.
- **Failures:** a failed put or atomicAdd fails the later atomicAdds of its ring (at
  the sender, and the target drops them through a ring abort) until the ring's memory
  views are released. An atomicAdd waiting for its ack fails once its ring completed
  nothing for 10 s. Known gap: a command the runtime cannot resolve (for example a
  stale memory view) never reaches the backend, so it does not fail later atomicAdds
  on its ring.
- **Connection info:** a proxy section (version 4: control EPs, receive rails and
  their data EPs, incarnation) is appended after the rail endpoints. Peers without it,
  or with another version, still get puts and host transfers, but no atomicAdd. A
  ring's first operation towards a peer waits for the engine handshake with it.
- **Diagnostics:** `NIXL_EFA_PROXY_PROFILE=1` prints per-stage latency histograms, per
  thread busy time and the cost of receiving puts to stderr at shutdown.
  `NIXL_EFA_PROXY_INJECT` (tests only; ignored in `NDEBUG` builds) injects
  back-pressure, failed posts or completions, or disables GDRCopy.

## API Reference

### Core Classes

- **`nixlLibfabricEngine`** - Main backend engine providing multi-rail RDMA operations with GPU Direct support
- **`nixlLibfabricRailManager`** - Manages multiple network rails with topology-aware selection and striping strategies
- **`nixlLibfabricRail`** - Individual network rail handling libfabric resources and completion processing
- **`nixlLibfabricTopology`** - Hardware topology discovery for optimal GPU-to-EFA and NUMA-to-EFA mapping
- **`nixlLibfabricBackendH`** - Request handle for tracking multi-request transfer completion with atomic counters
- **`nixlLibfabricConnection`** - Multi-rail connection metadata for remote agents with state management

## Troubleshooting

### Debug Information

Enable debug logging by setting environment variables:

```bash
# Libfabric debug logging
export FI_LOG_LEVEL=debug
export FI_LOG_PROV=efa  # or verbs, tcp, etc.

# NIXL debug logging
export NIXL_LOG_LEVEL=debug
```

### Common Issues

**No network devices detected:**

```bash
# Check available fabric interfaces
fi_info -l

# For checking specific devices (e.g. EFA as an example)
fi_info -p efa
```

For additional support, check the NIXL documentation and Libfabric provider-specific guides.
