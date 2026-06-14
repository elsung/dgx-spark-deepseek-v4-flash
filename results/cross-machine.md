# DeepSeek-V4-Flash — cross-machine comparison (single-stream)

Same model, different hardware/engines. Single-stream, fair method (non-streaming,
median of iters; decode = completion_tokens/wall, prefill = prompt_tokens/wall on a unique
~2.4k-token prompt to defeat KV caching; `bench-remote-ds4.py`).

| Machine | Hardware (mem BW) | Engine / quant | decode t/s | prefill t/s | concurrency |
|---|---|---|--:|--:|---|
| **RTX PRO 6000** | 96 GB GDDR7 (~1.8 TB/s) | ds4.c | **46.9** | 344 | single-stream only¹ |
| **Dual DGX Spark** | 2×128 GB unified, TP=2 (QSFP RoCE) | vLLM **FP8** | **~41** | **~1785** | **~350 t/s agg @ c=32** |
| **Mac Studio M2 Ultra** | 192 GB unified (~800 GB/s) | ds4.c | **29.7** | 389 | single-stream only¹ |
| **Single DGX Spark** | 128 GB unified (~273 GB/s) | ds4.c IQ2_XXS | ~14 | — | single-stream |

¹ antirez ds4.c is single-stream (concurrent requests queue).

## Reading it
- **Single-stream ≈ memory bandwidth.** RTX PRO 6000 (fat GDDR7) leads at 47; M2 Ultra at 30; a single Spark's
  273 GB/s puts it last on raw latency. The dual Spark lands at ~40 — TP=2 over the 200G link roughly
  **3× a single Spark** and past the Mac.
- **Two axes the table doesn't fully show:**
  - **Quality:** only the dual Spark runs the **official FP8** (~149 GB) — everyone else is on smaller
    GGUF quants (IQ2/q-class). So the Sparks deliver the *best-quality* V4-Flash, at competitive speed.
  - **Serving throughput:** the ds4.c boxes are single-stream; the dual-Spark vLLM does **~270 tok/s
    aggregate** at concurrency. For multi-user / agentic serving the Spark pair is in a different class.

**Takeaway:** for one fast chat, the RTX PRO 6000 wins. For **highest-quality V4-Flash and real multi-stream
throughput on hardware you can buy two of**, the dual DGX Spark is the pick.
