# EFA device proxy: receiver-side put -> atomicAdd ordering

How the LIBFABRIC (AWS EFA) device proxy executes GPU-initiated `put()` and
`atomicAdd()` and keeps a signal behind the data it covers. Code:
`libfabric_proxy.{h,cpp}` (logic), `libfabric_proxy_wire.h` (wire format),
`libfabric_proxy_conninfo.h` (connection info).

## 1. Guarantees

| Operation | Completes at the sender (GPU sees success) when | Failure reaches the GPU when |
|---|---|---|
| `put()` | every fragment is placed at the target (`FI_DELIVERY_COMPLETE`) | a fragment fails to post or completes with an error |
| `atomicAdd()` | the target applied the add (the thread that received it acked it) | the add fails at the target, the ring failed earlier, or no ack while the ring made no progress for 10 s |

- **Ordering:** an `atomicAdd()` is applied at the target only after every earlier
  `put()` on the same ring **to the same target** has landed there, and after the
  ring's previous `atomicAdd()` to that target was applied. A ring is a runtime
  (channel, peer slot) pair; the peer slot is the destination memory-view descriptor
  index. A ring that carries operations to several agents orders each target's
  operations separately (warned once): no target can see another target's puts.
- **No sender waiting:** the sender never holds an `atomicAdd()` behind earlier puts;
  the target does the waiting.
- **Failures:** after a put or `atomicAdd()` of a ring fails, that ring sends no
  further `atomicAdd()` (they complete with the error) until its views are
  released, and the target drops the ring's waiting adds at once (ring abort). A put
  that failed at the sender after its data did land may still be covered by an
  `atomicAdd()` that was already sent; that add is applied (its data is there) while
  the sender reports the failure.

## 2. Threads and endpoints

Per process (one GPU), with `T` proxy threads and the rails of the node:

```
proxy thread t (t = 0..T-1)
 ├─ data EP + CQ + AV on EVERY rail r          (fi_domain of rail r, FI_THREAD_SAFE)
 │    - sends: puts (RDMA writes) leave from here; from the home rail's EP also
 │      atomicAdd records, acks and ring aborts (fi_sendmsg)
 │    - receives: on the receive rails t owns, puts (remote CQ data), atomicAdd
 │      records and ring aborts; on its home rail, acks (posted receive buffers)
 └─ request pool, retry queues, rings, waiting adds
```

There is no separate control EP: control messages share the data EPs, and queue
behind the puts there as on an RC queue pair (measured: no cost, see `RESULTS.md`).

- **Receive rails:** the rails next to the process's GPU (`railsForAccelerator`), in
  order `rx_rails[0..R-1]` (R = 4 on p5). Receive rail number `i` is owned by thread
  `railThread(i, T) = i % T`: only that thread polls its data CQ there, and the
  connection info publishes that thread's data EP for the rail.
- **Home rail** of thread t: `rx_rails[t % R]`; its control messages leave from its
  data EP there, and acks come back to it.
- **Who applies an add:** the target thread that receives it, i.e. the one polling
  the receive rail where the sending thread's small puts land (no owner formula).
  That thread also counted those puts. Rings of different senders or sending threads
  may add to the same counter from different threads: a VRAM counter's
  read-modify-write holds one of 64 lock stripes (by counter address); host-memory
  counters use a CPU atomic.
- **Channel owner:** channel `c` runs on thread `c % T` (the runtime's striping,
  checked: a thread state used by a second worker is fatal).
- **Receive CQs:** the data CQs of receive rails hold 32768 entries, the device
  maximum on p5 (others 16384). Writes with remote CQ data use the device's
  unsolicited write receive (p5 has it): they need no posted receive buffer, so
  nothing slows a sender while the receiving thread is busy (a GDRCopy pin, a CUDA
  copy), and the CQ absorbs what arrives meanwhile (~10 ms of 4 KiB writes at a
  rail's 100 Gbit/s). Without that capability the provider uses its receive buffers
  and back-pressures the sender instead.

### Example: p5, one GPU per process, 4 channels, 4 threads

| thread | channels | home rail | receives puts and adds on | sends small puts and adds on |
|---|---|---|---|---|
| 0 | 0 | rx_rails[0] | rx_rails[0] | buffer rail 0 |
| 1 | 1 | rx_rails[1] | rx_rails[1] | buffer rail 1 |
| 2 | 2 | rx_rails[2] | rx_rails[2] | buffer rail 2 |
| 3 | 3 | rx_rails[3] | rx_rails[3] | buffer rail 3 |

A GPU buffer's rails are its GPU's four rails, so a ring's small puts and its adds
leave on one rail and land on one rail of the target, where one target thread counts
the puts and applies the adds. Puts at or above the striping threshold (128 KiB) are
split over the buffer's rails and counted by several target threads (they share the
ring's counters, see 4.2).

### Sender-side mapping of one put fragment

```
local rail   = the thread's share of the buffer's rails (one rail when there are
               at least as many threads as rails), or stripe i
remote rail  = the remote buffer's rail paired with it; if the target does not
               receive on that rail, the next buffer rail it does receive on
destination  = data EP of the target thread railThread(index of the remote rail in
               the target's receive rails, its thread count), from the target's
               connection info, inserted in the local rail's AV
remote key   = the remote buffer's key on the remote rail
```

### End to end: who sends and who receives what

```
 sender process (T threads)                           target process (T' threads)
 ──────────────────────────                           ───────────────────────────
 channel c ─> thread s = c % T
   put, < 128 KiB (one fragment)
     local rail  lrails[k], k = s's share (policy thread)
     from: s's data EP on lrails[k] ──fi_writemsg + imm──> remote buffer rail reps[k] = rx_rails'[i]
                                                           to: data EP of thread i % T' there
                                                           (it polls that CQ, counts the imm)
   put, >= 128 KiB (fragment f = 0..n-1, n <= 4)
     from: s's data EP on lrails[f] ──fi_writemsg + imm──> reps[f] = rx_rails'[i_f]
                                                           -> thread i_f % T' (one per fragment)
   atomicAdd(counter A) on ring key K
     from: s's data EP, home rail ──fi_sendmsg──────────> counter buffer rail reps[s % n] = rx_rails'[i]
                                                           to: data EP of thread o = i % T' there
                                                           o waits, applies, then acks:
     s's home data EP <──────────── atomicAckMsg ──────── from o's home data EP (s's EP: from
                                                           the sender's connection info)
```

With equal thread counts and the GPU's buffers on the GPU's rails (the usual case),
a small put of channel `c` is sent by thread `c % T` on rail index `c % T` and counted
by target thread `c % T`, which also receives and applies the ring's signals.

### Receiver side: which thread does what

| Work | Thread |
|---|---|
| poll the data CQ of receive rail `i`, count put immediates | `railThread(i, T) = i % T` |
| receive atomicAdd records and ring aborts sent to receive rail `i` | the same thread |
| wait for a ring's slot count, flush, apply its adds in order, ack | the thread that received the add |
| receive acks | the sending thread, on its home rail |
| drop a ring's waiting adds on a ring abort | every thread (the abort mark is shared) |
| progress the engine's rails when the engine has no progress thread | thread 0 |

## 3. Encoding

### 3.1 Put immediate (32-bit remote CQ data on every put fragment)

```
 31            24 23                         8 7              0
+----------------+-----------------------------+----------------+
| sender index   |          ring id            |  epoch slot    |
| at the target  |   (allocated by sender)     | (epoch mod 256)|
+----------------+-----------------------------+----------------+
 \____________ ring key (24 bits) _____________/
```

- **sender index at the target:** the index the target's engine assigned to this
  agent (`connections_` order, max 255), learned from the engine handshake
  (`local_agent_idx_at_remote_`). Unique at the target, so keys never collide.
- **ring id:** allocated by the sending proxy per target incarnation, one per (runtime
  ring, target) TxRing; a ring used again after a failure gets a new one.
- **epoch:** the index of the `atomicAdd()` that will follow the put on its ring
  (`TxRing::atomics` when the put is submitted).

### 3.2 Control messages (data EPs, `fi_sendmsg`)

| Message | Direction | Fields |
|---|---|---|
| `atomicAddMsg` (56 B) | sender -> target receive rail | header, remote_addr, value, token, ring key, sender_thread, **seq**, **expected_puts** |
| `atomicAckMsg` (24 B) | applying thread -> sender's home EP | header, token, status |
| `ringAbortMsg` (24 B) | sender -> target receive rail | header, ring key, first_seq |

`header = {type, reserved}`: no version, since every node of a job runs the same
build. `seq` is the add's index on its ring; `expected_puts` is the total count of
slot `seq % 256` once this add's puts are in (the slot's counts accumulate over its
uses `seq - 256, seq - 512, ...`). The ack goes to the home data EP of `sender_thread`
in the sender's connection info, found by the sender's index in the ring key
(indexes are never reused: a reconnected agent gets a new one, and acks to an index
whose connection was replaced are dropped). `token` names the sender's request: its
slot in the sending thread's request pool and that slot's generation (bumped on
reuse), so the ack finds the request directly and a late ack for a reused slot is
ignored.

### 3.3 Connection info (proxy section, appended to the engine's)

```
efa_proxy_threads      = T
efa_proxy_ep_<t>       = data EP of thread t on its home rail  (t < T; acks)
efa_proxy_data_rails   = R
efa_proxy_data_rail_<i> = rail number of receive rail i         (i < R)
efa_proxy_data_ep_<i>  = data EP of thread i % T on that rail
efa_proxy_incarnation  = random 64-bit, new per proxy instance
```

Names are published at their real length (32 B on EFA); with 4 threads and 4 rails
the section is under 1 KB. The whole connection info travels in the engine's handshake,
which is limited to 8 KiB. Peers without the section (or with a malformed one) still
get puts (to their engine EPs, without remote CQ data); `atomicAdd()` to them fails
with `NIXL_ERR_NOT_SUPPORTED`. There is no version: a job's nodes are updated
together.

## 4. How it runs

### 4.1 One proxy thread pass (ProxyWorker::runOnce)

```
submitOwnedChannels   for each owned ring: new GPU records -> backend submit()
driveBackendProgress  backend progress(); thread t's work happens on ring (t, 0):
                        drain retry queues and pending control messages
                        poll CQs: data CQs of its home and receive rails and of rails
                                  it wrote on
                          - remote-CQ-data entry -> count the put
                          - write/send completion -> complete a put fragment / record
                          - message receive      -> atomicAdd record / ack / abort
                        apply waiting adds whose rings caught up (drainDeferred)
                        post requests parked for a handshake, once it arrived
                        thread 0, no engine progress thread: progress engine rails
publishOwnedChannels  check_completion() in ring order; publish to the GPU
```

An idle thread (nothing outstanding, no puts landing, nothing waiting) polls its CQs
at most every `efa_proxy_idle_poll_us` (default 2 us): every CQ read takes the rail
domain's lock and progresses the domain, which host transfers share.

### 4.2 put + atomicAdd on one ring

```mermaid
sequenceDiagram
    participant G as Sender GPU
    participant S as Sender proxy thread (owns the channel)
    participant N as EFA
    participant R as Target thread (owns the receive rail)
    participant TG as Target GPU

    G->>S: put(dst, n bytes) on ring (c, p)
    S->>S: TxRing towards the target: epoch e = atomics; slot_puts[e % 256] += fragments
    S->>N: fi_writemsg(FI_DELIVERY_COMPLETE | FI_REMOTE_CQ_DATA, imm = key|e)
    G->>S: atomicAdd(counter, v) on ring (c, p)
    S->>S: seq = atomics++ (= e); expected = slot_puts[seq % 256]
    S->>N: atomicAddMsg(key, seq, expected) to the data EP of the same receive rail
    N-->>R: data placed, then CQ entry with imm
    R->>R: received[key][e % 256] += 1 (release)
    N-->>R: atomicAddMsg (same CQ)
    R->>R: wait: applied[key] == seq and received[key][seq % 256] >= expected (acquire)
    R->>TG: GDRCopy read-modify-write of the counter under its lock stripe (after a flush where needed)
    R->>R: applied[key] = seq + 1 (release)
    R->>N: atomicAckMsg(token, status) to the sender thread's home EP
    N-->>S: write completion (put done) and ack (atomicAdd done)
    S->>G: completions published in ring order
```

Messages and writes travel independently (EFA SRD has no ordering, even on one EP
pair): the record may arrive before, between or after the put fragments; fragments of
one put may land on different rails and be counted by different target threads.

### 4.3 Why a signal cannot overtake its data

1. The add with `seq = s` is applied only when slot `s % 256` counted
   `expected_puts`: every fragment the sender put into that slot up to this add.
   Fragments of later epochs go to other slots, so they cannot satisfy it (a single
   cumulative count per ring could: a later put on another rail, or out of a retry
   queue, landed first, and the gtest under injected back-pressure caught signals
   ahead of their data).
2. A slot is reused 256 adds later. The runtime publishes completions in ring order
   and a ring holds at most `proxy_ring_depth` (<= 256, checked at init) requests, so
   the puts of epoch `s + 256` cannot be issued before add `s` completed, i.e. was
   applied: they never mix with epoch `s`'s count.
3. Adds of a ring are applied in `seq` order (`applied` counter, release/acquire
   between the counting and applying threads). A sending thread's adds all go to one
   receive rail, so normally one target thread applies a ring's adds; if they arrive
   on several (a ring moved between rails), each waits for `applied` to reach its seq.
4. A fragment is counted only after its remote-CQ-data completion, which EFA
   generates after the write is placed. Making those writes visible to the GPU before
   a later CPU write of the counter: on Hopper and later GPUs on x86, as NCCL relies
   on (`ncclTopoNeedFlush()`), no flush is needed; before Hopper and on aarch64 (C2C)
   hosts, the applying thread flushes every GPU with registered memory
   (`cudaDeviceFlushGPUDirectRDMAWrites`; the puts may land on any of them, whatever
   the counter's memory) once per batch of ready adds, after their slots were counted
   and before applying them (`efa_proxy_rx_flush=0|1` overrides).

### 4.4 Ring keys and the engine handshake

The first operation of a (ring, target) pair needs the sender's index at the target,
which the engine handshake delivers. Until it arrives the ring's requests are
**parked** in order and re-tried every pass (failed after 60 s). Self-connections
use index 0 (the agent's own slot).

Connection setup at scale (16 ranks) used to hang ("Handshake from peer 'N' not
received after 60s", about one nixl_ep run in four). Two engine bugs, both fixed:

- **Handshake lost between lookup and buffering.** `handleHandshake()` looked the peer
  up and buffered an early handshake in separate critical sections; a connection
  created in between never saw it. Both now happen under `connection_state_mutex_`
  (lock order `connection_state_mutex_` -> `pending_handshake_mutex_` /
  `handshake_mutex_`, as in `createAgentConnection()`).
- **Deadlock inside the efa provider (libfabric 2.1) between `fi_av_insert` and a CQ
  read on the same rail.** The application thread inserted a new peer's addresses
  (`loadRemoteMD` -> `insertAllAddresses` -> `fi_av_insert`) while proxy thread 0,
  progressing the engine's rails (there is no engine progress thread with the device
  proxy), read the rail's CQ and received a handshake from a peer not in the AV yet,
  which the provider inserts implicitly under its domain (SRX) lock. Both threads
  stopped in `pthread_mutex_lock` inside libfabric. libfabric 2.5's `efa_av.c` takes
  the SRX lock before the AV locks in `fi_av_insert`, "to prevent deadlocks" with the
  CQ read path; 2.1 (the EFA installer's and the NGC container's) does not. The stuck
  process stops progressing; its peers retry `-FI_EAGAIN` forever or time out. Found
  with all-thread stacks dumped on a signal (ptrace is not allowed on these nodes);
  fixed by doing the engine's AV inserts and removals under the rail's endpoint lock,
  which every engine CQ read holds.

### 4.5 Failure path

```mermaid
sequenceDiagram
    participant S as Sender thread
    participant T as Target threads
    S->>S: a put of epoch e fails (post or completion error)
    S->>S: ring.error, tx.error = status (sticky); later atomicAdds complete with it
    S->>S: atomicAdds already sent with seq >= e complete with the error now
    S->>T: ringAbortMsg(key, first_seq = e) to where its last add went (once per ring and seq)
    T->>T: aborted_from = min(aborted_from, e); drop waiting adds with seq >= e (no ack)
    T->>T: later records of the ring with seq >= aborted_from are dropped on arrival
    Note over S: next use of the ring after its views are released: new ring id
```

A failed `atomicAdd()` (lost record, error ack, ack timeout) does the same from its
own `seq`. The target applies nothing it cannot prove complete. Timeouts measure
stalls, not age: the sender fails an unacked add only after its ring completed nothing
for 10 s, and a target thread drops waiting adds (with an error ack) only after their ring
counted no put and applied no add for 10 s (sender dead), since a large put may take
that long to land.

### 4.6 Target-side data structures

- `RxRing` per ring key, shared by the threads (`rx_rings_`, created on first use,
  kept until shutdown since keys are never reused): `received[256]` (atomic counts per
  slot), `applied`, `error`, `aborted_from`.
- Each thread caches key -> RxRing (one-entry cache plus a map), so counting a put
  costs one atomic increment.
- `owned`: per thread, the waiting adds it received, by ring, ordered by `seq`. A drain
  round takes, per ring, the run of consecutive adds from the ring's next one
  (`seq == applied`) whose slots are complete, flushes once for the whole batch (if
  needed) and applies it in order; the next round (or pass) picks up what the batch
  unblocked.
- Counter mappings: a VRAM registration is pinned and mapped through GDRCopy when it
  is registered, all at once (~1 ms for 256 MiB; BAR1 holds all of an H100's memory),
  so the first add to a counter does not pin and map its page on the proxy thread
  (~60-420 us each). This takes BAR1 space equal to the registered VRAM, up to 4 GiB
  per process; beyond that, or if it fails, pages are mapped on their first add. A drain stops after 1 ms of
  applies (the rest wait, in order, for the next pass), so the thread polls its receive
  CQs at least that often. A record for a seq that was already applied or is already
  waiting (only a broken sender sends one) is failed with an error ack.

## 5. Parameters and diagnostics

| Name | Default | Meaning |
|---|---|---|
| `proxy_ring_depth` | 256 | must be <= 256 (epoch slots) |
| `efa_proxy_idle_poll_us` | 2 | how often an idle thread polls its CQs |
| `efa_proxy_rx_flush` | platform | 1/0 forces the GPUDirect RDMA flush before adds on/off |
| `NIXL_EFA_PROXY_PROFILE=1` | off | per-stage histograms, per-thread busy time and put-receive cost, printed to stderr at shutdown |
| `NIXL_EFA_PROXY_INJECT` | off | tests only (not in NDEBUG builds): back-pressure, failed posts/completions, no GDRCopy |

Small puts always use the thread's share of the buffer's rails. Spreading a ring's
puts over all rails (the former `efa_proxy_rail_policy=ring`, now ignored with a
warning) cost nixl_ep 22% on the shared rail domains; with a private send domain
per thread and rail it gained 1% at 16 ranks but lost 25% of the signal rate, since
the adds then wait for counts from several target threads (`RESULTS.md`).

## 6. Measured (two to four p5 nodes; details in `RESULTS.md`)

| | sender fence (before) | receiver ordering |
|---|---|---|
| ping-pong, 8 B put+signal each way | 114.8 us | 58.4 us |
| put+signal, completion at the sender (8 B / 64 KiB) | 75.8 / 88.4 us | 48.3 / 60.9 us |
| signals/s, 4 channels | 50192 | 58948 |
| pipelined 64 KiB / 1 MiB puts, 4 channels | 290.8 / 389.8 Gbit/s | 287.3 / 389.7 Gbit/s |
| nixl_ep, 8 ranks (fast allocation) | 14.27 GB/s | 13.82 GB/s (-3.2%) |
| nixl_ep, 16 ranks | 11.77 GB/s | 11.22 GB/s (-4.7%) |
| target proxy CPU | - | ~300 ns per put counted; 0.65 busy cores at 59k signals/s |

Signals complete about one round trip sooner (nixl_ep, 8 ranks: 232.7 -> 201.7 us);
bulk puts lose 1.5-2% (plain stream, 16-64 KiB) to 3-5% (nixl_ep) of bandwidth to the
per-write immediate, which the target NIC completes into a CQ. Neither the proxy CPU
nor the network (no drops or retransmissions; identical write and packet counts) is
the limit.

## 7. Limits

- At most 255 peers (the engine's index width) and ring depth <= 256.
- 65536 ring ids per (target, target incarnation): one per (runtime ring, target)
  used, plus one per re-key after a failure. Ids are not reused (the target keeps an
  `RxRing`, ~2 KB, per key until shutdown); once exhausted, that target's new rings
  fail with `NIXL_ERR_BACKEND` until it restarts (a new incarnation).
- Puts are received on the GPU's rails; a destination buffer registered only on other
  rails cannot be targeted (the error names the rails).
- Ordering is per (ring, target), see section 1.
- The target-side add is a read-modify-write: the GPU must not write a counter while
  remote adds to it can arrive.
- Releasing memory views drains the rings on the proxy threads, which synchronize the
  device (runtime code shared with UCX). Until a running kernel ends, those threads
  count no puts and apply no adds for remote senders: release views only when no
  kernel waits for a remote signal.
- A put the runtime cannot resolve (stale view, bad offset) never reaches the backend:
  the GPU sees the error in the channel status, but the ring is not failed, so a later
  `atomicAdd()` on it is still sent and applied.
- Every proxy thread has a data EP on every rail (lazy creation would be cheaper).
- Each put carries a 32-bit immediate, which costs EFA 1.5-5% of put bandwidth (see 6).
- A ring's small puts use one rail: an application that drives few channels (nixl_ep:
  one per local expert) leaves the GPU's other rails and proxy threads idle.
- Every EP that receives control messages (each thread's home and receive rails)
  holds 1024 posted receive buffers.
