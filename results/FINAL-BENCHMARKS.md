# DGX Spark — final benchmarks (all engines)

Hardware: NVIDIA **GB10**, 128 GB LPDDR5X unified (~273 GB/s), sm_121, arm64, CUDA 13.0.
Single-Spark numbers are llama.cpp `llama-bench` (`-ngl 999 -fa 1`, fully resident).
Dual-Spark is vLLM TP=2 over a 200 G QSFP RoCE link.

## A) Single Spark — llama.cpp GGUF (decode, single-stream)
Ranked by decode tok/s. **MoE dominates** a bandwidth-bound box.

| Model | Type (active) | decode t/s | prefill t/s |
|---|---|--:|--:|
| Qwen3.6-35B-A3B Q4_K_M | MoE (3B) | **67.4** | 2305 |
| Gemma4-26B-A4B IQ4_XS | MoE (4B) | **60.3** | 2917 |
| Qwen3.5-9B Q5_K_XL | dense 9B | 33.6 | 2337 |
| MiniMax-172B Q3_K_S | MoE (10B) | **31.2** | 525 |
| Gemma4-12B Q4_K_XL | dense 12B | 27.8 | 1951 |
| MiniMax-139B Q3_K_M | MoE (10B) | 26.5 | 558 |
| Gemma3-27B QAT Q4_0 | dense 27B | 13.8 | 1031 |
| Qwen3.6-27B IQ4_XS | dense 27B | 13.7 | 899 |
| Gemma4-31B Q3_K_S | dense 31B | 12.9 | 719 |

## B) Single Spark — DeepSeek-V4-Flash (antirez ds4, GGUF IQ2_XXS)
284B/13B-active MoE, ~80 GB, MTP draft=2, via `ds4-server`: **~14 tok/s** single-stream.

## C) Dual Spark — DeepSeek-V4-Flash official FP8 (vLLM TP=2)
284B/13B-active, ~149 GB across both nodes, RoCE over QSFP, MTP draft=2, fp8 KV, 1M ctx.

**Single-stream:** ~38.5 tok/s (vLLM metric) · 42–44 tok/s end-to-end · MTP accept-len 2.0–2.3
**Aggregate throughput (concurrency):**

| Profile | peak agg tok/s | bound by |
|---|--:|---|
| 1M ctx, 6 slots | ~103 (c≈6) | slot count |
| **32K ctx, 36 slots** | **~270 (c≈32)** | **GPU compute (95% util both nodes)** |

~270 tok/s aggregate = ~7× single-stream; compute-bound (not memory — KV only ~7% used). See
`dual-spark-vllm.md` for the full sweeps. Profile switchable via `.env` (`MAX_MODEL_LEN`/`MAX_NUM_SEQS`).

## D) DeepSeek-V4-Flash across machines (single-stream, same model)
| Machine | Hardware | Engine/quant | decode t/s | concurrency |
|---|---|---|--:|---|
| RTX PRO 6000 | 96 GB GDDR7 (~1.8 TB/s) | ds4.c | **46.9** | single only |
| **Dual DGX Spark** | 2×128 GB, TP=2 | vLLM **FP8** | ~38–44 | **~270 agg** |
| Mac Studio M2 Ultra | 192 GB (~800 GB/s) | ds4.c | 29.7 | single only |
| Single DGX Spark | 128 GB (~273 GB/s) | ds4.c IQ2 | ~14 | single |

Single-stream tracks memory bandwidth (RTX 6000 wins). But only the **dual Spark runs the full FP8**
(best quality) and does real **multi-stream throughput (~270 agg)** — the ds4.c boxes are single-stream.
Full breakdown in `cross-machine.md`.

## Takeaways
1. **Decode is memory-bandwidth bound** — small dense models run ~2.5–3.4× slower than a 16 GB GDDR7
   card (9B: 33 vs 113 t/s). The Spark is a *capacity* machine, not a latency one.
2. **Prefer MoE** — active/total ratio is the speed multiplier. A 35B MoE (67) beats a 27B dense (13)
   by 5×; the 172B MoE runs at 31 t/s.
3. **The Spark's niche: frontier-size models on one (or two) boxes.** MiniMax-172B, DeepSeek-V4-Flash
   don't fit a 16 GB or 96 GB card — here they run at usable speed, and **two Sparks deliver the full
   FP8 V4-Flash at ~40 tok/s** — the headline result this kit set out to prove.

Run any of it: `mytoolz` (see the menu).
