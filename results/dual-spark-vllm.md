# DeepSeek-V4-Flash — dual DGX Spark (vLLM TP=2)

Official FP8 (284B/13B-active MoE) across 2× GB10 over a 200G QSFP RoCE link.
vLLM 0.21.1 (Aiden image), TP=2, MTP draft=2, fp8 KV, expert-parallel.

## Single-stream
- vLLM server-side **Avg generation throughput: ~38.5 tok/s** (steady state)
- End-to-end (incl. prefill): **42–44 tok/s** (229 tok/5.3s, 500 tok/11.8s)
- MTP: mean acceptance length **2.04–2.29**, draft acceptance **51–64%** (higher on code)
- Cold start ~7–9 min; TTFT a few s (reasoning model, reasoning_effort=high)

## Concurrency — two profiles (clean: ignore_eos, fixed-length gen)

> ### ⚠️ Context is a *shared* KV budget — not per-request
> The KV cache is **one fixed pool**, measured at boot: `GPU KV cache size: 1,105,096 tokens`
> (vLLM startup log, `production-v2`, fp8 KV, `gpu_memory_utilization=0.82`). That **~1.1M-token pool is
> shared across all concurrent requests.**
> - `--max-model-len` (1M / 256K / 32K) is the **per-request ceiling** — **not** a per-request reservation.
> - `--max-num-seqs` (6 / 36) is the **max batch width** — only reachable when the requests' *combined*
>   length fits the one pool.
>
> The real limit is **Σ(live tokens across all running requests) ≤ ~1.1M**:
>
> | Profile | full-context requests at once | …or N concurrent if each ≤ |
> |---|--|--|
> | `base` 1M / 6 | **~1** — `1,105,096 ÷ 1,048,576 = 1.05×` | ~184K tokens (1.1M ÷ 6) |
> | `batch` 256K / 36 | **~4** — vLLM logs *"Maximum concurrency for 262,144 tokens per request: **4.22×**"* | ~30K tokens (1.1M ÷ 36) |
>
> So ~~6 concurrent requests each holding 1M context~~ → that would need **6M tokens of KV, which we do not
> have.** The 6 (or 36) slots all fill **only** if the requests' lengths sum to ≤ ~1.1M. Over-subscribe and
> vLLM **preempts and recomputes** (time-slices the pool) — it does **not** OOM on KV; the pool is locked at
> boot. Bottom line: **high per-request context and high concurrency do not coexist** — you trade one for the
> other along the ~1.1M-token line.
>
> **Are the contexts independent? Yes.** Each request gets its *own* KV blocks (PagedAttention) — request A's
> tokens never share or borrow from request B's. The ~1.1M ceiling is **purely a memory-capacity limit, not a
> design limit**: with enough KV memory you absolutely *could* run 6 requests each at a full 1M context — you'd
> just need ~6× the KV. (The one intentional sharing: **prefix caching** dedups *identical* prefixes across
> requests — distinct contexts stay fully separate.) So the way to buy more simultaneous long-context headroom
> is **more memory = more DGX nodes** (next section).

### Scaling the KV pool with more DGX Spark nodes
KV capacity grows **faster than linearly** as you add nodes, because the 149 GB of weights is sharded over more
nodes (TP/PP) — freeing per-node memory for KV on *every* node. From the measured anchors (`production-v2`,
fp8 KV, `gpu_memory_utilization=0.82`): **~37 KB/token** (≈28,300 tokens/GiB), and each node contributes
**~94 GiB** to *weights + KV*, of which weights eat `149 GB ÷ N`. So:

> **pool(N) ≈ 28,300 tok/GiB × (94·N − 149) GiB**

| DGX nodes | weights/node | KV/node | **shared KV pool** | 6 concurrent → each | 32 concurrent → each |
|--:|--:|--:|--:|--:|--:|
| **2** (today) | 74.5 GB | 19.5 GiB | **~1.1M tok** | ~184K | ~35K |
| 3 | 49.7 GB | 44.4 GiB | ~3.8M tok | ~628K | ~118K |
| **4** | 37.3 GB | 56.8 GiB | **~6.4M tok** | **~1.07M** | ~201K |
| 5 | 29.8 GB | 64.2 GiB | ~9.1M tok | ~1.52M | ~284K |
| 6 | 24.8 GB | 69.2 GiB | ~11.8M tok | ~1.96M | ~367K |
| 8 | 18.6 GB | 75.4 GiB | ~17.1M tok | ~2.85M | ~534K |

**Answers to the obvious questions:**
- **4 DGX Sparks:** ~6.4M-token pool → **6 concurrent at ~1.07M each**, or **32 concurrent at ~200K each**.
- **256K context on *each* of 32 concurrent** (needs 32 × 256K = 8.4M tok) → **5 Sparks** (N=5 gives 9.1M; N=4's 6.4M falls short).
- **1M context on *each* of 6 concurrent** (needs 6 × 1M = 6.3M tok) → **4 Sparks** (N=4's 6.4M just covers it).
- (bonus: 256K × 6 → **3 Sparks**; 1M × 32 → ~**15 Sparks**.)

> ⚠️ **These are first-order *memory* estimates — planning upper bounds, not validated runs.** They assume one
> sharded instance (weights counted once, TP/PP — not DP replicas), ~94 GiB/node for weights+KV, and fp8 KV at
> the measured ~37 KB/token. They ignore the **compute & interconnect** walls you'd hit first at large N: this
> recipe is only validated at **TP=2 / 2 nodes**; going wider needs pipeline-parallel + more RoCE links, prefill
> compute for 32×256K concurrently is enormous (1M prefill alone is already minutes), and NCCL/activation
> overhead grows. Memory is necessary, not sufficient. Also: 1M is the model's trained context ceiling — more
> memory won't exceed it.

**"base" profile (1M ctx, max_num_seqs=6)** — interactive / long-context:
| c | 1 | 2 | 4 | 6 | 8 | 12 |
|---|--:|--:|--:|--:|--:|--:|
| agg tok/s | 32 | 59 | 66 | **103** | 96 | 110 |
| **per-request t/s** | **~32** | ~28 | ~17 | ~17 | ~12 | ~9 |

The pair is **compute-bound (~350 t/s total)**, so per-request ≈ total ÷ active streams. **40+ tok/s per
request is single-stream only** — you can't have both high per-request speed and high concurrency. `base`
gives ~32–41/req for 1–2 users (with 1M context); use `batch` (256K/36) when you want raw aggregate.
Note: the ~15 GB system-RAM headroom is fixed by `gpu_memory_utilization=0.82`, the same for every profile.

**High-throughput, 32K ctx, max_num_seqs=36:**
| c | 1 | 8 | 16 | 24 | 32 | 36 |
|---|--:|--:|--:|--:|--:|--:|
| agg tok/s | 36 | 130 | 180 | 233 | 270 | 193 |

**Sweet spot — 256K ctx, max_num_seqs=36** (single-stream decode 41 t/s, prefill ~1785 t/s):
| c | 1 | 8 | 16 | 24 | 32 |
|---|--:|--:|--:|--:|--:|
| agg tok/s | 39 | 143 | 195 | 289 | **351** |

- **Peak ~350 tok/s aggregate** at 256K ctx, c=32 (vLLM metric ~345 sustained) — *higher* than the 32K
  profile and with 8× the context. ~8.5× single-stream. The `vm.compaction_proactiveness=0` mitigation
  (added after the kernel crash, see SETUP-NOTES §7) appears to have lifted throughput too.
- **256K/36 is the memory edge**, though: ~15 GB system RAM free, 1 NVRM-OOM event logged (recovered).
  ~~KV cache itself is only ~6% used (short requests), but the KV *pool* pre-allocation (`ctx×slots`) is what
  eats memory.~~ **Correction:** the KV *token* pool is **fixed at ~1.1M** by `gpu_memory_utilization` (see the
  shared-budget box above) — it is **not** `ctx×slots`, and short requests really do leave it ~6% used. What
  pushes 36 slots to the edge is the **non-KV overhead** — per-sequence bookkeeping + CUDA-graph / activation
  buffers that scale with `slots × max-model-len` — compounded, *before* `production-v2`, by the prefix-cache
  **host-RSS leak** (§7 / SETUP-NOTES §9, now fixed). **500K at 36 slots still OOMs** — for >256K, drop
  `MAX_NUM_SEQS` (trade concurrency for context).
- Profiles switch via `.env`: `MAX_MODEL_LEN` + `MAX_NUM_SEQS`. Recommended default: **256K / 36**.

Measure it yourself: `mytoolz dsv4-bench --n 1 --n 8 --n 24 --n 32`
