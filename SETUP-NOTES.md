# DGX Spark dual-node inference — setup notes & gotchas

Hard-won lessons from standing up 2× DGX Spark (GB10, 128 GB unified, sm_121, arm64) for
DeepSeek-V4-Flash. Read this before the next bring-up — most of these cost hours.

## Nodes
- **e1f0** (head): user `justblaze1`, Tailscale (head)
- **378f** (worker): user `justblaze0`, `ssh justblaze0@spark-378f`
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
- It's largely an NVIDIA bug — **file a DGX support ticket** (BIOS 5.36_0ACUM018, driver 580.159.03,
  kernel 6.17.0-1021-nvidia; mstflint NULL-deref + kcompactd soft-lockup) and check for firmware updates.
- Reduce the memory-pressure wedge so a future Oops is survivable: `vm.compaction_proactiveness=0`
  (stops the background daemon that locked up — on-demand compaction still works, no perf loss),
  `drop_caches` after big downloads, and optionally lower vLLM `gpu_memory_utilization`.
- High-context × high-concurrency profiles (e.g. 500K) raise memory pressure — apply the above + watch `free`.
