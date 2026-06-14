# DeepSeek-V4-Flash — dual DGX Spark (vLLM TP=2)

Official FP8 (284B/13B-active MoE) across 2× GB10 over a 200G QSFP RoCE link.
vLLM 0.21.1 (Aiden image), TP=2, MTP draft=2, fp8 KV, expert-parallel.

## Single-stream
- vLLM server-side **Avg generation throughput: ~38.5 tok/s** (steady state)
- End-to-end (incl. prefill): **42–44 tok/s** (229 tok/5.3s, 500 tok/11.8s)
- MTP: mean acceptance length **2.04–2.29**, draft acceptance **51–64%** (higher on code)
- Cold start ~7–9 min; TTFT a few s (reasoning model, reasoning_effort=high)

## Concurrency — two profiles (clean: ignore_eos, fixed-length gen)

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
  KV cache itself is only ~6% used (short requests), but the KV *pool* pre-allocation (`ctx×slots`) is what
  eats memory. **500K at 36 slots OOMs** — for >256K, drop `MAX_NUM_SEQS` (trade concurrency for context).
- Profiles switch via `.env`: `MAX_MODEL_LEN` + `MAX_NUM_SEQS`. Recommended default: **256K / 36**.

Measure it yourself: `mytoolz dsv4-bench --n 1 --n 8 --n 24 --n 32`
