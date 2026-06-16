# Long-context — prefill & decode vs depth (dual DGX Spark, FP8 TP=2)

Prior numbers were all fresh/short-prompt. This measures prefill + decode at real context depths.
Method: cache-free (unique fresh prompts, time-randomized salt). `bench-longctx.py`. Profile: `base` (1M/6).

## Single-stream vs context depth
| context (actual tok) | TTFT / prefill time | prefill t/s | decode t/s |
|--:|--:|--:|--:|
| 7.8k | 3.9 s | 1974 | 39.8 |
| 31k | 15.5 s | 2016 | 38.6 |
| 100k | 52.5 s | 1901 | 39.5 |
| 200k | 115 s | 1747 | 37.6 |

## Findings
1. **Prefill is ~linear (~1900 t/s, roughly flat 8k→200k)** — DeepSeek-V4's compressed/sparse attention
   avoids the usual O(n²) prefill blowup. **Cost = TTFT:** a 200k prompt is ~2 min to first token.
2. **Decode barely degrades with depth** (~40 → ~38 at 200k). Cross-checked with a cached-prefix-reuse
   method (~31–39 at depth); a streaming method gave ~16 but is believed to be an artifact. Net: V4
   largely preserves decode speed at long context — its core design goal.

## Stability note (important)
NVRM `NV_ERR_NO_MEMORY` events rose **1 → 3** this boot during the long-context runs (system RAM ~11 GB
free; `gpu_memory_utilization=0.82`). Long-context load stresses memory — the same pressure that fed the
earlier kernel crash (SETUP-NOTES §7). ~~**Sustained / concurrent big-context runs carry crash risk**~~
**Update (2026-06-16):** the workload trigger — a vLLM prefix-cache host-RSS leak — is **fixed in
`production-v2`** (PR #44237); a 15-min sustained re-run held container memory flat (+4 MB). Keep the
mitigations (compaction off, headroom) + monitoring as defense-in-depth for the latent `mstflint` NVIDIA bug.

## TODO (campaign in progress)
- Concurrent long-context at the edge (`batch` 256K/36: N × ~200–250k).
- 500k single + low concurrency (`custom 524288 / ~10`).
- Sustained/continuous (30–60 min) — throughput decay? memory creep? stability?
- Cross-box: RTX PRO 6000 (≤131k), Mac M2 Ultra (≤1M).
- → ideal settings per use case.

## Long-context across machines (single-stream)
Same depth ramp on each box. ds4.c boxes (Mac, RTX PRO 6000) measured at their own depths.
| Machine | engine | TTFT @100k | prefill t/s (8k→100k) | decode t/s (8k→100k) |
|---|---|--:|--:|--:|
| **Dual DGX Spark** | vLLM **FP8** | **52 s** | ~1974 → ~1901 | ~40 → ~40 |
| RTX PRO 6000 | ds4.c | 349 s | ~317 → ~286 | ~30 → ~26 |
| Mac M2 Ultra | ds4.c | 315 s | ~395 → ~317 | ~24 → ~20 |

**The dual Spark's edge GROWS with context:** ~6× faster TTFT at 100k (52 s vs 5–6 min on the ds4.c boxes),
and decode stays flat (~40) where ds4.c decode sags ~20–30%. vLLM FP8 + V4 sparse-attention kernels ≫ ds4.c
at depth. (TTFT is still real on every box — a 100k prompt is never instant.)

## Stress & stability (monitored; watchdog auto-aborts if free RAM < 4 GB)
| test | result | mem low-water | NVRM-OOM Δ | crash? |
|---|---|--:|--:|---|
| concurrent 4 × 150k (`base`) | stable, completed | 12.0 GB | 0 | no |
| 500k single-stream (`base`) | TTFT 351 s, prefill 1379, **decode 31.9** | 11.5 GB | 0 | no |

**Key memory insight:** `gpu_memory_utilization=0.82` reserves the GPU budget at startup (~105 GB/node:
weights + a **fixed ~1.1M-token KV pool** + overhead), so GPU memory is bounded up front and workload
(concurrency / context depth) **fills that fixed pool but does NOT spike system RAM** — free RAM stays
~11–15 GB. ~~So crash risk ≈ a constant baseline + the `mstflint`/NIC kernel bug, not the workload.~~
**Correction (2026-06-16):** the crash *did* have a workload component — a vLLM prefix-cache **host-RSS leak**
(PR #44237) that climbed under sustained load until `kcompactd` wedged; fixed in `production-v2` (SETUP-NOTES §7).
~~500k at 36 slots would OOM — pool = ctx×slots = 18M tokens; 500k single on `base`'s 6M pool fits fine.~~
**Correction:** the KV pool is **~1.1M tokens, fixed** (not `ctx×slots`). A 500k *single* request fits because
500k < 1.1M; **500k × 36 OOMs** on non-KV per-seq overhead (+ pre-v2 the host leak), not on a "18M pool."
See [`dual-spark-vllm.md`](dual-spark-vllm.md#scaling-the-kv-pool-with-more-dgx-spark-nodes).

## Sustained / continuous (15 min, base 1M/6, gen=256, sequential)
140 requests over 15 min: **decode stayed 39.6 → 40.2 t/s (no decay)**, **free RAM 11589 → 11576 MB
(no creep)**, **0 new NVRM-OOM**, no crash. Continuous running does not degrade throughput or leak memory.

## Concurrent long-context at the edge (batch 256K/36)
8 concurrent × ~200k context (1.55M tokens prefilled): wall **876 s**, aggregate prefill **~1765 t/s**,
**stable** (13 GB low-water, 0 NVRM-OOM, no crash, KV pool ~13% used). **Concurrent big-context is
prefill-bound** — the prompts dominate wall time; decode is fast once prefilled. (Pool didn't fill — 36×250k
would, but stability is already established; the pre-allocated pool means system RAM is untouched regardless.)

## Ideal settings per use case (dual DGX Spark, DeepSeek-V4-Flash FP8)
| Use case | Profile (`mytoolz dsv4-up …`) | Why |
|---|---|---|
| **Interactive chat / agents, 1–2 users** | **`base` (1M / 6)** | ~40 tok/s per request, up to 1M context, stable. The everyday default. |
| **Long doc / whole-codebase, single big context** | **`base` (1M / 6)** | decode holds ~32–40 even at 200k–500k; the cost is TTFT (200k≈2 min, 500k≈6 min to first token). |
| **Batch / many concurrent agents / evals (short–mid ctx)** | **`batch` (256K / 36)** | ~350 tok/s aggregate; ~12 tok/s per request. |
| **Max raw throughput, short context** | **`fast` (32K / 36)** | ~270–350 aggregate. |
| **Avoid** | `ctx × slots ≳ 9M tokens` (e.g. 500K/36) | KV pool OOM on cold start. For >256k, drop slots. |

**Rules of thumb:**
- **Prefill ~linear (~1900 t/s), but TTFT grows** — keep interactive prompts under ~100k for <1-min TTFT.
- **Decode barely drops with depth** (~40 → ~32 at 500k) — long context slows *time-to-first-token*, not generation.
- **Per-request 40+ tok/s ⇒ single-stream only** (compute-bound ~350 t/s total; per-req ≈ total ÷ active).
- **Everything tested was stable** — 15-min sustained (no decay/creep), concurrent, and 500k single all held,
  thanks to the **fixed ~1.1M-token KV pool** (workload fills the GPU pool, not system RAM) and, since
  2026-06-16, the **`production-v2`** prefix-cache-leak fix (PR #44237). Residual crash risk is now just the
  latent NVIDIA `mstflint`/NIC kernel bug (SETUP-NOTES §7) — mitigations stay as defense-in-depth.
