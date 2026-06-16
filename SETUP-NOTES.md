# DGX Spark dual-node inference — setup notes & gotchas

Hard-won lessons from standing up 2× DGX Spark (GB10, 128 GB unified, sm_121, arm64) for
DeepSeek-V4-Flash. Read this before the next bring-up — most of these cost hours.

## Nodes
- **e1f0** (head): user `<head-user>`, Tailscale (head)
- **378f** (worker): user `<worker-user>`, `ssh <worker-user>@<worker>`
- Both: driver 580.159.03, CUDA 13.0, Ubuntu 24.04.

## 1. nvidia-smi "failed" on a fresh node = pending reboot, NOT broken
378f's GPU looked dead (`NVIDIA-SMI has failed…`). Cause: it was still on the **old kernel**
(`6.11`) while the 580 driver modules were built for the **new** (`6.17`), with a
`/var/run/reboot-required` flag set. **Fix: reboot.** It comes up matching the other node. No reinstall.

## 2. HF Xet downloads HANG silently — use the watchdog
The `hf` Xet client deadlocks on long downloads: process alive, **0 bytes/s, no error**, frozen for
hours. Burned ~12 h on this. Also: unauthenticated HF throttles hard after an initial burst.
- **Always `hf auth login` first** (lifts throttle).
- **Always download via `scripts/resilient-hf.sh`** — it watches *network RX* (not `du`, which lies for
  Xet's sparse-preallocated shards) and kills+restarts `hf` if it stalls. Resumes from cache.
- Large files are **Xet-only** (classic HTTP is refused) — you can't avoid Xet, just wrap it.
- `du` reports the model "done" while orphan `*.incomplete` files linger — verify via the snapshot's
  `model.safetensors.index.json`, then `find … -name '*.incomplete' -delete`.

## 3. The QSFP link (the big one)
- The Spark's CX-7 ports are `enp1s0f1np1` (rocep1s0f1) and **`enP2p1s0f1np1` (roceP2p1s0f1)**.
  **The rightmost physical port = `enP2p1s0f1np1` on BOTH machines** — that's the one we cabled.
  Point NCCL at the matching port name on each node (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`).
- **`ip addr` you set by hand gets WIPED by NetworkManager.** Pin it: `nmcli dev set <iface> managed no`
  before `ip addr add` (baked into `scripts/setup-qsfp.sh`).
- **ufw is active on 378f and blocks the link.** `ufw allow from 192.168.100.0/24` (also in the script).
- Verify with `ping` over the link + `ip neigh` shows `REACHABLE`. ARP `FAILED`/`INCOMPLETE` = wrong port
  pair or no IP on the peer. RoCE state via `rdma link show <dev>/1` (want `ACTIVE / LINK_UP`).
- IPs: head `192.168.100.1/24`, worker `192.168.100.2/24`. `MASTER_ADDR=192.168.100.1`, master-port 25000.
- **This must be redone after every reboot** (manual IPs aren't persistent): run `setup-qsfp.sh` on each.

## 4. Docker without re-login
`sudo usermod -aG docker $USER` needs a fresh login to take effect. A **new `ssh` session picks it up**
(so worker `docker compose` over ssh just works); but the long-lived local shell doesn't — use
`sg docker -c "docker …"` on the head until you re-login.

## 5. vLLM dual-node specifics
- Each node loads its TP shard but **reads all 46 safetensors** — keep them cached on **both** nodes.
- Cold start ~6–9 min (149 GB load + CUDA-graph capture + FlashInfer autotune). Be patient.
- It's a **reasoning model** (`reasoning_effort=high`) — multi-second TTFT is normal (it thinks first).
- Bench decode via **vLLM's own log metric** ("Avg generation throughput") — client-side streaming
  token-timing gets fooled by buffering + reasoning_content and reports garbage.

## 6. ik_llama.cpp does NOT build on aarch64
`iqk_cpu_ops.cpp` uses x86/AVX-only helpers (`v_expf`, `v_silu`) + missing `<cstdint>`. The cstdint
include is fixable (`-DCMAKE_CXX_FLAGS="-include cstdint"`); the SIMD helpers are not (without a port).
Use mainline llama.cpp with standard quants instead. (mainline also needed `-DLLAMA_BUILD_UI=OFF`-ish:
a UI asset downloaded empty → build it with prebuilt-UI disabled.)

## 7. Kernel crash / spontaneous reboot (2026-06-13, e1f0 only)
Symptom: e1f0 hard-hung and rebooted under heavy dual-Spark vLLM load; 378f (worker) was fine.
Root cause (from `journalctl -k -b -1`):
- **Trigger:** NVIDIA DGX telemetry (`nvidia-dgx-telemetry.service`) periodically runs **`mstflint`** to
  poll the ConnectX-7 firmware. One poll **NULL-deref'd the kernel** in `pci_bus_read_config_dword`
  (reading the NIC's PCI config) **with IRQs disabled** — a kernel/MST-driver bug, likely tickled while
  the CX-7 was hammered by TP=2 RoCE traffic.
- **Wedge:** that Oops tainted the kernel; then `kcompactd0` (memory compaction) **soft-locked a CPU for
  48 s** under heavy memory pressure (149 GB model + ~470 GB of model files in page cache) → RCU stall → hang.
- **Red herring:** `mlx5_core … insufficient power on the PCIe slot (27W)` logs on **every boot, both
  nodes, all 4 NIC functions** — it's a normal trait of the integrated CX-7 on a PCIe x4 link, NOT the cause.
Fixes:
- **ROOT memory-pressure source FIXED 2026-06-16:** it was a **vLLM prefix-cache memory leak**
  ([PR #44237](https://github.com/vllm-project/vllm/pull/44237), *"linear host RSS growth under sustained load
  with prefix caching"*) feeding the climb that wedged `kcompactd`. Upgrade to
  `aidendle94/sparkrun-vllm-ds4-gb10:production-v2` — re-validated: 15-min sustained run, **container memory
  flat (+4 MB)**, stable throughput, clean concurrent + long-context. The mstflint NULL-deref trigger is still
  a latent NVIDIA bug — keep the ticket + mitigations below as defense-in-depth.
- It's largely an NVIDIA bug — **file a DGX support ticket** (BIOS 5.36_0ACUM018, driver 580.159.03,
  kernel 6.17.0-1021-nvidia; mstflint NULL-deref + kcompactd soft-lockup) and check for firmware updates.
- Reduce the memory-pressure wedge so a future Oops is survivable: `vm.compaction_proactiveness=0`
  (stops the background daemon that locked up — on-demand compaction still works, no perf loss),
  `drop_caches` after big downloads, and optionally lower vLLM `gpu_memory_utilization`.
- High-context × high-concurrency profiles (e.g. 500K) raise memory pressure — apply the above + watch `free`.

## 8. NCCL "unhandled system error" on dual-node bringup = RoCE GID index mismatch
After a re-link, vLLM TP=2 failed: `ncclCommInitRank` → `ibv_modify_qp failed ... local GID index 3,
local GID :: (empty)`. **Not** memory, **not** MTU. The recipe hardcodes `NCCL_IB_GID_INDEX=3` (the RoCEv2
IPv4 GID slot), but the worker's GID table had a **gap** (IPv4 RoCEv2 landed at index 4, index 3 empty) —
leftover cruft from re-assigning the link IP across ports during cabling debug, on a node that hadn't
rebooted since. The head (freshly rebooted) had it at index 3.
Diagnose: `for i in 0..5; do cat /sys/class/infiniband/<rocedev>/ports/1/gids/$i; .../gid_attrs/types/$i; done`
— find the index whose GID = `::ffff:<your-ipv4>` AND type = `RoCE v2`.
Fix: made `NCCL_IB_GID_INDEX` env-driven per node (head=3, worker=4). A clean reboot rebuilds the GID table
without the gap (then both = 3). Set per-node in each node's `.env`.

## 9. Context vs concurrency: KV is ONE shared pool (and how it scales with nodes)
`--max-model-len` (1M / 256K) is a per-request **ceiling**, NOT a reservation; `--max-num-seqs` (6 / 36) is the
max batch width. The KV cache is a single fixed pool sized at boot from `gpu_memory_utilization` — measured
`GPU KV cache size: 1,105,096 tokens` (2 Sparks, fp8 KV, util 0.82). It is **shared**:
Σ(live tokens across all running requests) ≤ ~1.1M.
- `base` 1M/6   → ~**1** full-1M request, or 6 concurrent averaging ≤184K each (vLLM: 1.1M ÷ 1.0M = 1.05×).
- `batch` 256K/36 → ~**4** full-256K, or 36 averaging ≤30K (vLLM logs "Maximum concurrency for 262,144 … 4.22×").

Contexts are **independent** (each request owns its KV blocks) — so this is a *memory* limit, not a design one;
over-subscription **preempts/recomputes** (time-slices), it does NOT OOM on KV. Scale the pool by adding DGX
nodes (the 149 GB of weights amortizes over more nodes → more KV room on every node; ~94 GiB/node for
weights+KV, ~37 KB/token):

> **pool(N) ≈ 28,300 tok/GiB × (94·N − 149) GiB**

| nodes | 2 | 3 | 4 | 5 | 6 | 8 |
|--|--|--|--|--|--|--|
| KV pool | 1.1M | 3.8M | 6.4M | 9.1M | 11.8M | 17.1M |
| 6-conc each | 184K | 628K | 1.07M | 1.52M | 1.96M | 2.85M |
| 32-conc each | 35K | 118K | 201K | 284K | 367K | 534K |

**4 Sparks → 6×~1M or 32×~200K. To get 256K on EACH of 32 → 5 Sparks. To get 1M on EACH of 6 → 4 Sparks.**
First-order MEMORY estimate only (one sharded instance, weights counted once). Only TP=2 / 2-node is validated
here — going wider needs pipeline-parallel + more RoCE links and hits compute/bandwidth (and 1M-prefill latency)
walls before the memory wall. Full write-up: repo `results/dual-spark-vllm.md`.
