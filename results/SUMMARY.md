# DGX Spark (GB10) GGUF benchmark — analysis

Hardware: **NVIDIA GB10**, 128 GB LPDDR5X **unified** memory (~273 GB/s), sm_121, arm64, CUDA 13.0.
Engine: llama.cpp (CUDA, `-DCMAKE_CUDA_ARCHITECTURES=121`). Single-stream, fully GPU-resident
(`-ngl 999 -fa 1 -p 512 -n 128 -r 3`). No offload — 128 GB fits all of these.

## Results, ranked by decode speed
| Model | Type (active) | decode t/s | prefill t/s |
|---|---|--:|--:|
| Qwen3.6-35B-A3B Q4_K_M | **MoE** (3B) | **67.4** | 2305 |
| Gemma4-26B-A4B IQ4_XS | **MoE** (4B) | **60.3** | 2917 |
| Qwen3.5-9B Q5_K_XL | dense 9B | 33.6 | 2337 |
| MiniMax-172B Q3_K_S | **MoE** (10B) | **31.2** | 525 |
| Gemma4-12B Q4_K_XL | dense 12B | 27.8 | 1951 |
| MiniMax-139B Q3_K_M | **MoE** (10B) | **26.5** | 558 |
| Gemma3-27B QAT Q4_0 | dense 27B | 13.8 | 1031 |
| Qwen3.6-27B IQ4_XS | dense 27B | 13.7 | 899 |
| Gemma4-31B Q3_K_S | dense 31B | 12.9 | 719 |

(ds4 / DeepSeek-V4-Flash IQ2_XXS, 284B/13B-active, measured via ds4-server: **~14 t/s** single-stream w/ MTP.)

## The three takeaways

### 1. Decode is memory-bandwidth bound — small models are ~2.5–3.4× slower than a 16 GB GDDR7 card
Qwen3.5-9B: **33.6 t/s here** vs **113 t/s** on the reference RTX 5070 Ti (672 GB/s GDDR7).
Ratio ≈ bandwidth ratio. The dense 27–31B models all cluster ~13 t/s — they must read ~13–16 GB
of weights per token, and 273 GB/s ÷ ~15 GB ≈ 18 t/s ceiling (measured 13). **The Spark is not a
latency machine for models that already fit a consumer card.**

### 2. MoE is the Spark's superpower — "prefer MoE" from the source repos, but stronger here
Because decode only reads *active* params, MoE models leap to the top:
- **Qwen3.6-35B-A3B (3B active): 67 t/s** — a 35B model that runs **5× faster than a 27B dense** (13 t/s).
- **MiniMax-172B (10B active): 31 t/s** — a **172-billion-parameter** model, faster than any 27B dense.

On a bandwidth-bound box the active/total ratio *is* the speed multiplier. Pick MoE every time.

### 3. Capacity is the whole point — these run *only because* of 128 GB unified memory
MiniMax-172B Q3 (74 GB) and 139B (67 GB), and DeepSeek-V4-Flash (80 GB), **cannot load on a 16 GB or
even 96 GB card.** The Spark trades peak bandwidth for the ability to run frontier-size MoEs locally
at usable speeds (26–31 t/s). That's the niche: **big models, single box, no cluster.**

## Notes / limitations
- **MTP** speedups aren't in this table — `llama-bench` can't measure speculative decoding (server feature).
  ds4's ~14 t/s above *is* with MTP draft=2. A server-based MTP A/B is a TODO (`results/mtp-gb10.md`).
- **ik_llama.cpp** doesn't build on aarch64 (x86-tuned IQK kernels); the IQ_K-quant rows use mainline-loadable
  quants instead. ik ARM port is a TODO.
- **Phase 2** (NVFP4 via TRT-LLM, W4A16 via vLLM+LMCache) needs arm64 builds — not yet run.
