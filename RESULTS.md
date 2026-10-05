# EFA device proxy: sender-side fence vs receiver-side ordering

Question: for put -> atomicAdd ordering on EFA, is it cheaper to hold each
atomicAdd at the sender until its ring's puts complete (today: one network round
trip, ~30-40 us), or to let the target CPU order it, which costs per-write
decoding of remote CQ data at the target?

All numbers: two AWS p5 nodes (H100), one benchmark process with one GPU per node
unless noted; median of 3 runs (range in parentheses) unless noted. Branch
`efa-cpu-proxy-rxorder` (from `efa-cpu-proxy` 5a762a8). Summaries and trimmed raw
logs: `results/efa-rxorder/data/`; scripts: `results/efa-rxorder/scripts/`.

## Recommendation

> Superseded: receiver-side ordering is now the only mode (sender fence removed); see
> "Receiver-only implementation" at the end.

1. The per-write cost at the target is ~0.03 us to decode and count an entry, plus
   ~0.22-0.26 us of `fi_cq_read` per entry (0.34-0.37 us per entry for the whole CQ
   sweep with 4 threads): well below the ~1 us bar, in the 0.2-0.4 us range.
2. Receiver-side ordering wins on signal latency and rate, at no throughput cost, but
   only with the rail-affine layout (each target thread counts the puts that land on
   its own rail): ping-pong 58.7 vs 114.4 us, put+signal 49.3 vs 76.1 us, +20% signals/s,
   +0.2 target cores (busy time), host path and pipelined bandwidth unchanged.
3. The layout as first specified (one counting thread per ring, hashed) makes every
   thread poll every GPU rail: domain-lock contention costs 31% at 8 KiB x 4 channels,
   2 busy target cores, and starved host transfers (2 of 3 runs made no progress).
4. Receiver ordering also needs per-epoch counts (a single cumulative count let later
   puts satisfy an earlier signal: real ordering violations in the gtest), and it does
   not speed up nixl_ep (8.47 vs 8.43 GB/s): nixl_ep is not bound by the fence.
5. So: keep sender ordering as the default; receiver ordering (rail-affine, per-epoch,
   opt-in) is worth productizing only for latency-bound signalling (small puts,
   ping-pong-like exchanges), where it saves one round trip (~27 us) per signal.

## Step 4: comparison (one allocation, modes interleaved per repetition)

Two-node runs: job 7683596 (pool0-0321/0736); nixl_ep: job 7683462 (pool0-0624/0964,
4 GPUs per node, 8 ranks, 4 channels/4 proxy threads per rank, static 8-rank plan).
"receiver (rail)" is `efa_proxy_ordering=receiver,efa_proxy_rx_thread=rail`;
"receiver (ring)" is `efa_proxy_ordering=receiver` (the counting thread per ring is a
hash of the ring, as in the task); "fence off" is the test-only
`efa_proxy_test_fence_off=1` (atomicAdds sent without waiting: NO ordering).

| Metric | sender (today) | receiver (rail) | receiver (ring) | fence off |
|---|---|---|---|---|
| ping-pong, 8 B put+signal each way (us) | 114.4 (113.9-114.8) | 58.7 (58.6-59.0) | 58.7 (58.2-58.8) | 58.4 (58.0-58.6) |
| 8 B put, completion at the sender (us) | 38.4 (37.9-38.8) | 39.0 (38.6-39.0) | 38.8 (38.5-38.9) | 38.4 (38.1-38.4) |
| 8 B put+signal, completion at the sender (us) | 76.1 (75.0-76.5) | 49.3 (49.0-49.3) | 49.1 (49.1-49.4) | 48.6 (48.4-48.8) |
| 64 KiB put+signal at the sender (us) | 87.9 (87.8-88.5) | 60.6 (60.4-61.0) | 60.9 (60.1-61.4) | 52.8 (51.8-52.8) |
| ordering check, 4 ch x 2048 rounds (signals/s) | 48346 (47926-49014) | 58019 (56910-59386) | 65571 (64890-65593) | 56949 (53708-57428) |
| pipelined 1 MiB, 1 channel (Gbit/s) | 356.8 (356.6-356.9) | 354.8 (354.8-355.3) | 355.0 (353.2-355.1) | 356.6 (356.4-357.1) |
| pipelined 1 MiB, 4 channels (Gbit/s) | 389.7 (389.7-389.7) | 389.7 (389.7-389.7) | 389.7 (294.1-389.7) | 389.7 (389.7-389.8) |
| pipelined 64 KiB, 4 channels (Gbit/s) | 291.4 (291.3-291.5) | 287.8 (286.6-287.8) | 261.3 (207.6-273.0) | 291.2 (291.1-291.5) |
| pipelined 8 KiB, 4 channels (Gbit/s) | 58.6 (58.6-58.7) | 58.2 (57.6-58.3) | 40.5 (37.1-40.6) | 58.6 (58.2-58.6) |
| nixl_ep, 8 ranks, dispatch+combine (GB/s) | 8.43 (8.23-9.32) | 8.47 (7.91-8.54) | 5.78 (5.74-5.85) | 13.13 (12.57, 13.69; 3rd run failed) |
| target CPU: busy proxy-thread cores, ordering check | 0.49 (0.47-0.50) | 0.70 (0.66-0.71) | 2.05 (1.90-2.48) | 0.60 (0.58-0.61) |
| target CPU: busy cores, 16 KiB puts at line rate | 0 (puts need no target CPU) | 0.21 (0.21-0.22) | 0.63 (0.55-0.65) | 0 |
| host path, 1 MiB writes, proxy on and idle (Gbit/s) | 200.9 (199.1-202.7) | 202.7 (199.0-203.6) | 177.0 (1 run; 2 of 3 made no progress in 300 s) | 201.5 (200.6-204.8) |
| host path, 1 MiB writes while 4 channels pipeline 16 KiB puts (Gbit/s) | 195.2 (185.8-197.6) | 198.4 (183.9-199.1) | 187.7 (185.2-189.0) | 198.8 (195.0-201.2) |
| ... the device puts meanwhile (Gbit/s) | 58.4 (58.3-62.7) | 57.9 (56.6-62.4) | 48.8 (48.4-49.8) | 58.2 (57.7-58.6) |

Notes:

- Target CPU "cores used": proxy threads busy-poll, so each one occupies a core in
  every mode (4 per process here). The rows give the useful part: per thread, time in
  progress passes that read a CQ entry or applied an add, over its active window,
  summed over the target's threads (`EFA proxy cpu:` lines, `NIXL_EFA_PROXY_PROFILE`).
- In the same allocation the 5a762a8 build ("base") measured ping-pong 114.4, put+signal
  75.9 us, 47828 signals/s, pipelined 389.6 / 58.6 Gbit/s, host 200.5 Gbit/s: sender
  mode of this branch is unchanged. nixl_ep base vs sender: see "Default unchanged".
- Fence off is a bound, not a design: nixl_ep failed 1 of 3 runs with "Remote key (RKEY)
  not registered" errors (signals ahead of data let it move on), and its signals may
  overtake data. The two-node ordering check passed in every mode, fence off included:
  it does not expose missing ordering on this setup (the back-pressure gtest does).
- receiver (rail) without the GPUDirect RDMA flush (`efa_proxy_rx_flush=0`): ping-pong
  57.7, put+signal 48.5 us, 56949 signals/s, nixl_ep 8.38 GB/s. The flush itself costs
  0.33 us per call (mean; one call per batch of ready adds). Applying an add through
  GDRCopy costs 3.2 us in every mode (a PCIe read of the counter); the receiver-ordered
  add waited at the target 12.8 us median (p90 23.9 us) for its ring.
- nixl_ep absolute numbers are about half of the previous day's (16.5-17.5 GB/s): every
  build, including the pre-review one, measured 8-10 GB/s on 2026-10-05. Only compare
  within one allocation.

## Step 1: baseline (5a762a8, no changes)

Job 7682249 (pool0-0101/0963); nixl_ep job 7682251 (pool0-0624/0977).

| Metric | Median (range) |
|---|---|
| ping-pong 8 B (us) | 117.0 (116.6-117.3) |
| 8 B put (us) | 38.9 (38.8-39.1) |
| 8 B put+signal at the sender (us) | 77.7 (77.4-77.7) |
| pipelined 1 MiB, 1 channel (Gbit/s) | 356.6 (356.4-356.6) |
| pipelined 1 MiB, 4 channels (Gbit/s) | 389.7 (389.7-389.8) |
| ordering check, 4 channels (signals/s) | 46913 (46623-47146) |
| nixl_ep, 8 ranks (GB/s) | 8.34 (7.83-8.89) |
| host 1 MiB / 64 KiB writes, proxy idle (Gbit/s) | 199.7 / 182.2 |

Full table: `results/efa-rxorder/data/step1-baseline.md`.

## Step 2: per-write cost at the target

Test-only mode `efa_proxy_test_put_imm=1`: puts carry 32 bits of remote CQ data and go
to a proxy data EP of the target, whose thread decodes the entry and increments a
per-(ring, epoch) counter, nothing else (atomicAdds keep the sender fence). Pipelined
puts at line rate, 2 GiB per trial, 3 trials per run, 3 runs; job 7683597
(pool0-0321/0736). "sweep" = `fi_cq_read` sweep over all of a thread's CQs, timed only
when it returned remote CQ data, divided by those entries; "read" = the `fi_cq_read`
calls that returned them; "decode" = decoding and counting. imm = counted by a hashed
thread per ring (every thread polls the GPU's 4 rails), imm (rail) = counted by the
thread that owns the destination rail.

| layout | threads | size | writes/s (M) sender / imm | Gbit/s sender / imm | sweep ns/entry | read ns/entry | decode ns/entry | entries per read | target busy cores |
|---|---|---|---|---|---|---|---|---|---|
| imm (rail) | 1 | 4 KiB | 0.242 / 0.242 | 7.9 / 7.9 | 586 | 250 | 33 | 1.0 | 0.14 |
| imm (rail) | 1 | 7 KiB | 0.238 / 0.239 | 13.7 / 13.7 | 590 | 249 | 33 | 1.0 | 0.14 |
| imm (rail) | 1 | 16 KiB | 0.236 / 0.233 | 30.9 / 30.5 | 568 | 250 | 33 | 1.0 | 0.13 |
| imm (rail) | 1 | 64 KiB | 0.220 / 0.217 | 115 / 114 | 622 | 259 | 33 | 1.0 | 0.13 |
| imm (rail) | 4 | 4 KiB | 0.902 / 0.905 | 29.6 / 29.6 | 357 | 220 | 27 | 1.2 | 0.32 |
| imm (rail) | 4 | 7 KiB | 0.896 / 0.899 | 51.4 / 51.5 | 361 | 221 | 28 | 1.2 | 0.32 |
| imm (rail) | 4 | 16 KiB | 0.878 / 0.865 | 115 / 113 | 343 | 215 | 25 | 1.3 | 0.30 |
| imm (rail) | 4 | 64 KiB | 0.554 / 0.543 | 291 / 285 | 373 | 233 | 28 | 1.2 | 0.20 |
| imm (ring) | 4 | 4 KiB | 0.902 / 0.611 | 29.6 / 20.0 | 1680 | 558 | 28 | 1.3 | 0.96 |
| imm (ring) | 4 | 7 KiB | 0.896 / 0.620 | 51.4 / 35.6 | 1550 | 531 | 28 | 1.3 | 0.92 |
| imm (ring) | 4 | 16 KiB | 0.878 / 0.599 | 115 / 78.6 | 1460 | 532 | 28 | 1.3 | 0.81 |
| imm (ring) | 4 | 64 KiB | 0.554 / 0.518 | 291 / 272 | 1620 | 544 | 28 | 1.3 | 0.78 |

With 1 thread the ring and rail layouts are the same (that thread polls all 4 rails),
and throughput is bound by GPU enqueue (~3 us per op) in every mode.

With host-path 1 MiB writes running concurrently in the same process (sweepc):

| layout | threads | size | host Gbit/s sender / imm | device writes/s (M) sender / imm |
|---|---|---|---|---|
| imm (rail) | 4 | 4 KiB | 205 / 201 | 0.480 / 0.484 |
| imm (rail) | 4 | 16 KiB | 195 / 193 | 0.446 / 0.444 |
| imm (rail) | 4 | 64 KiB | 163 / 164 | 0.352 / 0.351 |
| imm (ring) | 4 | 4 KiB | 205 / 190 | 0.480 / 0.391 |
| imm (ring) | 4 | 16 KiB | 195 / 185 | 0.446 / 0.372 |
| imm (ring) | 4 | 64 KiB | 163 / 160 | 0.352 / 0.310 |

Where the ring layout's cost comes from (4 threads, 4 KiB, same job, mean per call, all
3 runs): the sender's CQ poll takes 0.21 us with sender ordering, 0.25 us with imm
(rail) and 2.05 us with imm (ring); a post takes 0.58, 0.58 and 1.24 us; the target's
CQ poll 0.13, 0.30 and 1.5 us. Each CQ read takes its rail's `FI_THREAD_SAFE` domain
lock and progresses the whole domain; with every thread polling every rail, the
threads serialize on 4 locks (and the engine's host transfers wait on them too). Full tables:
`results/efa-rxorder/data/step2-sweep.md`.

### Receive buffers

- Writes with remote CQ data do not consume posted receive buffers on the `efa`
  fabric: with 32 receives posted on every data EP (`efa_proxy_test_data_rx_post=32`),
  none of 17.8 million entries consumed one (`consumed_rx=0`). `fi_info -p efa` lists only `FI_CONTEXT2` (no
  `FI_RX_CQ_DATA`) for the `efa` fabric, and `cq_data_size: 4`.
- The provider sends such a write as one device RDMA-write-with-immediate (single IOV;
  it asserts the write is not segmented), so the target's CQ entry follows the data.
  Its own internal receive buffers are sized by the EP's receive queue: 64 (today's
  data-EP size) and 4096 (`efa_proxy_data_rx_size=4096`) gave the same throughput at
  every size (`results/efa-rxorder/data/step2-rx-buffers.md`; job 7682516, whose first
  7 minutes overlapped another job on the same nodes, so only its within-job
  comparison is used), so no resizing or reposting is needed. Data EPs post no
  receives by default.
- Note: data EPs "post only 64 receives" in the task's background is the receive-queue
  size (`rx_attr->size = 64`); no receive is actually posted on them today.

## Step 3: what receiver-side ordering needed

Implemented as `efa_proxy_ordering=receiver` (default `sender`, unchanged):

- Puts carry `ringImm(key, epoch)`: a random 24-bit ring key and the index (mod 256) of
  the atomicAdd that follows the put; they go to proxy data EPs (never the engine's).
- The atomicAdd leaves immediately with its index and the cumulative fragment count of
  its epoch slot. The counting thread increments `received[slot]` (release); the
  counter's owner (unchanged, `mix64(addr >> 3) % threads`) applies it once its slot
  count is reached and the ring's previous atomicAdd was applied (acquire), flushes
  GPUDirect RDMA writes once per batch, applies with GDRCopy and acks as today.
- Ordering bug found and fixed: with one cumulative count per ring (as first
  specified), a put issued after an atomicAdd can arrive before a put ahead of it
  (other rails, back-pressure retry queues) and satisfy the count: the gtest under
  injected back-pressure saw a whole 64 KiB stripe missing when its counter moved
  (10 failures in 60 receiver back-pressure runs, flush on or off; `RXTRACE` timeline in
  `data/raw/rx-trace-run.txt`). Per-epoch counts fix it: 0 failures in the same
  15 x 2 repetitions (630 test runs, 60 of them receiver back-pressure). Slots are reused every 256 adds of a ring, safe
  because a ring has at most `proxy_ring_depth` <= 256 requests in flight (checked).
- Also needed: atomicAdd-to-atomicAdd order (adds of one ring to counters of different
  owner threads must apply in order); a sticky per-ring error at the sender (a failed
  put or add fails the ring's later adds, already-sent ones included); re-keying a ring
  after a failure (its counts may have diverged); the conn-info size: the handshake
  carries it and is limited to 8 KiB, so data EPs are published only for the 4 rails
  next to the GPU.
- Semantics difference (opt-in mode only): if a put's completion fails at the sender
  after the data did land, the sender fails the later atomicAdds but the target may
  still apply them (they have their data). The fault gtest checks exactly that.
- GPUDirect RDMA flush (`cudaDeviceFlushGPUDirectRDMAWrites`) before applying: on by
  default in receiver mode, as the CUDA-documented way to order NIC writes before the
  CPU's counter write. No failure was observed without it (315 runs after the epoch
  fix); it costs 0.33 us per call.
- Tests: the ordering, fault-injection and atomicAdd gtests run in three variants
  (sender, receiver ring, receiver rail): 36/36 pass, and 15 x 21 fault/atomic runs per
  variant set; unit tests 171/171; the two-node ordering check passes in every mode on
  both nodes (job 7683596: 0 mismatches, regressions, torn reads or timeouts in 36 runs).

## Default unchanged

Sender mode differs from 5a762a8 only by shared plumbing: wire/conn-info version 3
(peers on version 2 get no atomicAdds, as for any version change), a flag test per CQ
entry, and profiling counters. Micro-benchmarks in the same allocation are identical
(table notes above). nixl_ep, 5 more interleaved runs of each on one allocation (job
7683844, pool0-0438/0624): 5a762a8 14.17 GB/s (14.08-14.31), this branch's sender mode
14.21 GB/s (14.11-14.29). (The step 4 job's 9.43 vs 8.43 was within its run-to-run
spread; this allocation was also faster overall.)

## Commands and environment

- Nodes: p5 (8x H100 80GB HBM3, 32 EFA devices `0xefa1`, AMD EPYC 7R13), Linux
  5.15.0-1055-aws, driver 575.57.08, EFA installer 1.39.0, libfabric 2.1.0amzn1.0 (host
  build) / 2.1.0amzn5.0 (NGC PyTorch 25.11 container, nixl_ep), CUDA 12.4.1 (host build)
  / 13.0 (container), GDRCopy (gdrdrv loaded). Slurm: one GPU per node for the two-node
  benchmark, 4 GPUs per node for nixl_ep; no `--exclusive`; `srun --cpus-per-task=24`.
- Builds (`scripts/build.sh`, `scripts/ctr-build.sh`, meson `debugoptimized`):
  baseline `BUILD_DIR=build/nixl-efa-cpu-proxy` at 5a762a8, study `BUILD_DIR=build/nixl-rxorder`
  and `build/nixl-rxorder-ctr` at 8e9e25ee (code of this branch minus RESULTS/results).
- Step 1: `MODES=base SUITES="c1 c4 ord host" REPS=3 sbatch scripts/rxorder.sbatch`;
  `EP_PLAN=.../static_8.json EP_SWEEP="4:4 4:4 4:4" sbatch --nodes=2 --gpus-per-node=4 scripts/ep-elastic.sbatch`.
- Step 2: `MODES="sender imm immrail" SUITES="sweep sweepc" SWEEP_BYTES=2147483648 REPS=3 sbatch scripts/rxorder.sbatch`;
  receive buffers: `MODES="sender imm imm4k immpost" SUITES=sweep ...` (job 7682516).
- Step 3: `scripts/rx-tests.sh` (unit + device gtests, 3x repeat); race experiment:
  `NIXL_GTEST_EFA_PROXY_RX_PARAMS=efa_proxy_rx_flush=0 run.sh test/gtest/gtest --gtest_filter='*ProxyAtomic*:*ProxyFault*' --gtest_repeat=15`.
- Step 4: `MODES="base sender receiver rxrail rxnoflush fenceoff" SUITES="c1 c4 ord ordp host conc" REPS=3 sbatch scripts/rxorder.sbatch`;
  `REPS=3 EP_MODES="base sender receiver rxrail rxnoflush fenceoff" sbatch --nodes=2 --gpus-per-node=4 scripts/ep-modes.sbatch`.
- Summaries: `python3 scripts/rxorder_summarize.py <run dirs> [--ep <logs>]` and `--sweep`.
- Benchmark knobs (`test/efa_proxy/nixl_efa_proxy_benchmark.cu`):
  `NIXL_PROXY_BENCH_BACKEND_PARAMS`, `NIXL_PROXY_BENCH_PIPE_{SIZES,BYTES,TRIALS}`,
  `NIXL_PROXY_BENCH_CONCURRENT_HOST`; profiles: `NIXL_EFA_PROXY_PROFILE=1`.
- Pitfall: two of these two-node jobs submitted at once can land on the same nodes and
  slow each other to timeouts; their results were discarded and the runs repeated
  back-to-back (all reported jobs ran alone).

## Follow-up: checks of "adopt receiver-side ordering with rail-affine receiving"

Run after the study to test that recommendation's claims (scripts:
`results/efa-rxorder/scripts/rxorder-claims.sbatch`, `ep-modes.sbatch`; data:
`results/efa-rxorder/data/raw/*claims*`).

**NCCL GIN on the same nodes** (job 7684248, pool0-0516/0762; GIN = NCCL 2.28.8
`put_signal_ping_pong_gin`, aws-ofi-nccl GIN proxy, libfabric 2.5.1 `efa-direct`; ours =
libfabric 2.1 `efa`; median of 3):

| 8 B put+signal ping-pong, full round trip | us |
|---|---|
| NCCL GIN proxy | 59.9 (59.8-61.5) |
| this proxy, sender fence | 101.3 (100.9-101.9) |
| this proxy, receiver (rail) | 53.8 (53.7-53.9) |

Put+signal completion at the sender: 67.3 (sender) vs 44.3 us (receiver); 8 B put 33.8
us in both. GIN completes `iputSignal` at the source on local completion only (32.4 us
measured in milestone 3), not when the target applied the signal as here.

**GDAKI on EFA is not available on these nodes**: `efadv_query_device` reports
`comp_cntr=0` and `cq_ext_mem_dmabuf=0` on all 32 EFA devices (p5.48xlarge, efa driver
2.15.0g); EFA GIN GDAKI needs both, and offers weak (per-put) signals only.

**A two-node ordering test that catches violations** (ordering check, 4 ch x 2048 rounds,
`NIXL_EFA_PROXY_INJECT=eagain_every=3`, 3 runs per mode):

| Signal | Mode | Runs with data missing at signal time | Mismatched words |
|---|---|---|---|
| atomicAdd | fence off | 3 of 3 | 16384-32768 |
| atomicAdd | sender | 0 of 3 | 0 |
| atomicAdd | receiver (rail) | 0 of 3 | 0 |
| 8-byte value put ("PUT_VALUE") | sender | 3 of 3 | 16384-32768 |
| 8-byte value put ("PUT_VALUE") | receiver (rail) | 2 of 3 | 16384-34896 |

Without injected back-pressure, value puts showed no violation in 6 runs (and fence off
none in 3): the race needs a retried put. A value put is placed by the NIC directly, so
no target-side counting can hold it back.

**nixl_ep at 16 ranks** (4 nodes x 4 GPUs, static 16-rank plan, modes interleaved, two
allocations):

| | job 7684249 | job 7684444 |
|---|---|---|
| sender | 11.89 (11.64-11.97) | 11.89 (11.34-11.97) |
| receiver (rail) | 11.12 (10.43-11.38), -6.5% | 10.96 (10.66-11.29), -7.8% |
| receiver (rail), no flush | - | 11.29 (2 runs), -5% |

The third no-flush run hung in connection setup ("Handshake from peer '15' not received
after 60s"): 1 hang in 9 receiver runs at 16 ranks, 0 in 6 sender runs.

**Fault plan (rank killed mid-dispatch)**: inconclusive. On 2026-10-05 it failed before
its first phase for every build and mode, including 5a762a8, which passed it the day
before (jobs 7684250, 7684381, 7684447).

## Receiver-only implementation (after the study)

Decision: remove the sender fence; receiver-side ordering (rail-affine, per-epoch
counts) is the only mode. Commits: 6bcd3f8b (implementation), fd7a0e55 and 7cd04926
(two code-review passes), 0c253377 (connection-setup deadlock). Design, wire encoding
and thread/endpoint mapping: `src/plugins/libfabric/EFA_PROXY_RECEIVER_ORDERING.md`.
Baseline below ("base") is `efa-cpu-proxy` 5a762a8 (sender fence). Data:
`results/efa-rxorder/data/raw/rx{3,4,5}-*`, `hsdiag16b-*`.

**Tests** (each of fd7a0e55, 7cd04926, 0c253377): unit 162/162; device gtests 19/19
(ordering under injected back-pressure, failed puts/atomics and ring aborts, handshake
during metadata load); atomic and fault gtests 3 x 7. Two-node ordering check with
injected back-pressure (`eagain_every=3`), 5 runs each on fd7a0e55 and 7cd04926: no
data missing at signal time.

**Micro-benchmarks** (two nodes, one GPU each, one allocation, 3 runs; job 7686044,
fd7a0e55):

| | base | receiver-only |
|---|---|---|
| ping-pong, 8 B put+signal each way (us) | 114.8 | 58.4 |
| 8 B put+signal, completion at the sender (us) | 75.8 | 48.3 |
| 64 KiB put+signal at the sender (us) | 88.4 | 60.9 |
| ordering check, 4 ch (signals/s) | 50192 | 58948 (+17%) |
| pipelined 64 KiB / 1 MiB, 4 ch (Gbit/s) | 290.8 / 389.8 | 287.3 / 389.7 |
| target cores busy, ordering check / 16 KiB puts + host writes | - | 0.65 / 0.23 |

(In that allocation the base build's host-write benchmark made no progress after its
first trial in 3 of 3 runs; the receiver-only build's completed: 212.9 Gbit/s at 1 MiB.
Not investigated. Job 7686343, 7cd04926: ping-pong 119.5 -> 60.5 us, signals/s
47232 -> 57854.)

### The connection-setup hang

About one 16-rank nixl_ep run in four hung ("Handshake from peer 'N' not received
after 60s"), in every build. Two engine bugs:

1. A handshake that arrived between `handleHandshake()`'s peer lookup and its buffering
   was lost (separate critical sections). Fixed in 6bcd3f8b; deterministic gtest
   (`ProxyConnectionTest.HandshakeDuringMetadataLoad`: 60 s failure before, < 1 ms
   after). The 16-rank hang persisted (2 of 8 runs, job 7686801).
2. Agent-tagged handshake logs (`hsdiag.patch`) showed the peer everyone waited for
   frozen: its application thread stopped inside `insertAllAddresses` for one peer
   (`av-insert-begin` without `av-insert-end`), its proxy thread 0 at the same moment,
   and peers sending to it spun on `-FI_EAGAIN`. Stacks of all its threads, printed
   on a signal (`hang-dump-watch.sh`; ptrace is not allowed on these nodes):

   ```
   main:     nixlAgent::loadRemoteMD -> ... -> insertAllAddresses -> insertAddress ->
             fi_av_insert -> libfabric.so.1 -> pthread_mutex_lock
   proxy 0:  ProxyWorker::runOnce -> nixlLibfabricProxy::progress -> progressActiveRails ->
             progressCompletionQueue -> fi_cq_read -> libfabric.so.1 -> pthread_mutex_lock
   ```

   A lock-order deadlock in the efa provider of libfabric 2.1 (EFA installer and NGC
   container): the CQ read inserts a peer it does not know yet (the handshake it just
   received) under the domain's SRX lock, and `fi_av_insert` takes the AV locks first.
   libfabric 2.5's `efa_av.c` takes the SRX lock first in `fi_av_insert` "to prevent
   deadlocks". Fixed in 0c253377: the engine's AV inserts and removals run under the
   rail's endpoint lock, which every engine CQ read holds.

| 16 ranks, hung runs | before 0c253377 | 0c253377 |
|---|---|---|
| base (5a762a8) | 1 of 1, 1 of 4 (jobs 7685206, 7686046) | 1 of 5 (job 7687222), 3 of 6 (7687420) |
| receiver-only | 3 of 8 (7686046, flush on and off), 1 of 6 (7686595), 2 of 8 (7686801) | 0 of 5 (7687222), 0 of 6 (7687420) |

(Job 7687420 includes runs with `efa_proxy_rail_policy=ring`. At the earlier rate of
about one in four, 11 runs without a hang happen by chance with probability ~4%.)

### The 16-rank throughput loss

| nixl_ep, dispatch+combine (GB/s) | base | receiver-only | |
|---|---|---|---|
| 8 ranks, job 7686047 (fd7a0e55) | 14.27 (14.16-14.29) | 13.82 (13.57-13.86) | -3.2% |
| 8 ranks, profiled, job 7686683 | 13.59 (13.58-13.60) | 13.10 (13.10-13.11) | -3.6% |
| 8 ranks, job 7687223 (0c253377) | 8.70 (8.28-9.84) | 7.57 (6.79-9.07) | slow, noisy allocation |
| 16 ranks, job 7687222 (0c253377) | 11.77 (11.43-11.96) | 11.22 (10.84-11.34) | -4.7% |
| 16 ranks, job 7687420 (0c253377, slow allocation) | 6.01 (5.70-6.33) | 5.83 (5.61-6.42) | -3.0% |

- The GPUDirect RDMA flush is no longer issued on Hopper/x86 (NCCL's rule,
  `efa_proxy_rx_flush` overrides): about 3 of the 6.5-7.8 points measured before.
- Where the rest goes (8 ranks, same allocation, mean per operation, job 7686854): a
  signal (atomicAdd) completes 31 us sooner (232.7 -> 201.7 us submit to completion:
  the saved round trip), but puts take 3-4% longer (post -> done 164.1 -> 169.4 us,
  submit -> completion 181.3 -> 188.1 us). nixl_ep is bandwidth-bound, so it is the
  puts that count.
- Not the proxy CPU: the two proxy threads that carry nixl_ep's traffic are ~10% busy,
  7 points of it counting puts (~300 ns each, ~113k per thread per 0.48 s); the other
  two are idle (nixl_ep uses 2 channels: channel = local expert, 2 experts per rank).
- Not the network: EFA hw counters per 16-rank run are identical in both builds
  (3.90 M RDMA writes, 43.6 GB, 6.28 M packets each way) with no drops or retransmits
  (job 7687222).
- It is the write-with-immediate: each put now makes the target NIC write a completion,
  and the step 2 sweep already measured that cost in a plain stream (sender fence,
  4 threads): 0.878 -> 0.865 M writes/s at 16 KiB (-1.5%), 291 -> 285 Gbit/s at 64 KiB
  (-2%). With EFA's unordered delivery, receiver-side ordering needs one per put.
- Spreading a ring's puts over all of the GPU's rails (`efa_proxy_rail_policy=ring`)
  would use the rails nixl_ep leaves idle, but costs more than it gains: 16 ranks, job
  7687420, base 6.01 -> 4.73 GB/s, receiver-only 5.83 -> 4.57 GB/s (-22%; more threads
  per rail domain, which serializes on its lock). The default stays `thread`.
- So the remaining 3-5% at 8-16 ranks is the price of per-write immediates, against
  signals that complete about a round trip sooner (2x lower put+signal latency in the
  micro-benchmarks). Recovering it would take a hybrid (sender fence for bandwidth-bound
  epochs), which this change deliberately removed.

**Commands** (scripts in `results/efa-rxorder/scripts/`; same nodes and environment as
above):

- Builds and tests: `BUILD_DIR=build/nixl-rxorder scripts/build.sh`,
  `BUILD_DIR=build/nixl-rxorder-ctr INSTALL_DIR=build/nixl-rxorder-ctr-install scripts/ctr-build.sh`
  (in the container), `srun --cpus-per-task=24 scripts/rx-tests.sh`.
- Micro-benchmarks: `MODES="base new" SUITES="c1 c4 ord ordp host conc" REPS=3 sbatch scripts/rxorder.sbatch`;
  ordering under back-pressure: `PARTS=inject INJECT_MODES=new REPS=5 sbatch scripts/rxorder-claims.sbatch`.
- nixl_ep: `EP_HWC_DIR=<dir> EP_PLAN=scripts/static_16.json REPS=5 EP_MODES="base new" sbatch --nodes=4 --gpus-per-node=4 scripts/ep-modes.sbatch`
  (8 ranks: `--nodes=2`, `static_8.json`; each run is limited to `EP_STEP_MINUTES`=5 so a
  hang costs one run, not the job); summaries `rxorder_summarize.py --ep <log>` and
  `hwc_summarize.py <dir>` (EFA hw counter deltas per run).
- Profiles: `NIXL_EFA_PROXY_PROFILE=1` (the base build prints its report at
  `NIXL_LOG_LEVEL=INFO`).
- Hang diagnosis: build with `scripts/hsdiag.patch` (agent-tagged handshake logs and a
  stack dump on signal 44), run `EP_MODES=new` at 16 ranks, and
  `scripts/hang-dump-watch.sh <job id>` alongside.
