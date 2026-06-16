# DeepSeek-V4-Flash on DGX Spark — dual-node TP=2, single-node & a cross-machine shootout

Getting **DeepSeek-V4-Flash** running on **NVIDIA DGX Spark (GB10)** — the headline being the **official FP8
model across *two* Sparks via vLLM tensor-parallel (TP=2) over a QSFP56 RoCE link** — plus a single-Spark
path (antirez `ds4`), a 10-model llama.cpp GGUF suite, a cross-machine comparison (Mac M2 Ultra, RTX PRO
6000), long-context + stability benchmarks, and **every gotcha that cost us hours**.

Builds on my [blackwell-16gb-llm-starter](https://github.com/elsung/blackwell-16gb-llm-starter) and
[blackwell-llm-toolkit](https://github.com/elsung/blackwell-llm-toolkit), ported to GB10 (sm_121, arm64).

> **Disclaimer:** Documentation written by AI, benchmark runs executed by AI. Numbers are real though.
> Please treat it all with a great grain of salt as YMMV — but hopefully helpful for anyone viewing / using this.

**Hardware:** 2× NVIDIA DGX Spark (GB10, 128 GB LPDDR5X unified, sm_121, arm64, CUDA 13), single **200 G
QSFP56** direct cable (RoCE / NCCL).

---

## 🏁 Headline results
| Setup | Model / quant | single-stream | aggregate |
|---|---|--:|--:|
| **2× DGX Spark, TP=2 (vLLM)** | DeepSeek-V4-Flash **official FP8** | **~41 tok/s** | **~350 tok/s @ c=32** |
| 1× DGX Spark (antirez `ds4`) | DeepSeek-V4-Flash IQ2_XXS | ~14 tok/s | — |
| 1× DGX Spark (llama.cpp GGUF suite) | 9B → **172B** MoE | up to **67** (35B-A3B) | — |

## 📈 Same model, four machines — single-stream
| Machine | engine / quant | decode t/s | prefill t/s | concurrency |
|---|---|--:|--:|---|
| RTX PRO 6000 (96 GB GDDR7) | ds4.c | **46.9** | 344 | single-stream only |
| **2× DGX Spark** | vLLM **FP8** | ~41 | **~1785** | **~350 agg @ c=32** |
| Mac Studio M2 Ultra (192 GB) | ds4.c | 29.7 | 389 | single-stream only |
| 1× DGX Spark | ds4.c IQ2_XXS | ~14 | 410 | single-stream |

Only the Sparks run the **full FP8** quality, have **~5× the prefill**, *and* do real multi-stream throughput
— the ds4.c boxes are single-stream. → [`cross-machine.md`](results/cross-machine.md)

## ⚙️ Pick a config — what to expect & how to get it
Dual-Spark FP8; switch profiles with one command:
| Launch | Context | Single-stream | Peak aggregate | Best for |
|---|--:|--:|--:|---|
| `mytoolz dsv4-up base` | **1M** | ~41 t/s | ~103 @ c6 | interactive chat · long docs / whole codebases |
| `mytoolz dsv4-up batch` | 256K | ~41 t/s | **~350 @ c32** | many agents · batch · evals |
| `mytoolz dsv4-up fast` | 32K | ~41 t/s | ~270 @ c32 | max throughput, short context |

> **⚠️ "Context" is the per-request *max*, not what every stream gets at peak concurrency.** The KV cache is a
> **shared ~1.1M-token pool** (`GPU KV cache size: 1,105,096`), split across all live requests — ~~6 concurrent
> requests each at full 1M~~ would need 6M tokens of KV. Reality: `base` holds **~1** full-1M request (or 6
> averaging ≤184K each); `batch` holds **~4** full-256K (vLLM: *"max concurrency … 4.22×"*) or 32 averaging
> ≤30K. Long context and high concurrency trade off along that 1.1M line; over-subscription time-slices
> (preempts), it doesn't OOM.
>
> **Contexts are independent, though** — each request has its own KV blocks, so this is a *memory* limit, not a
> design one. Add DGX nodes and the pool grows **faster than linearly** (weights amortize over more nodes):
> **4 Sparks → 6 × ~1M each** (or 32 × ~200K); **5 Sparks → 32 × 256K each**. Full table + math →
> [dual-spark-vllm.md](results/dual-spark-vllm.md#scaling-the-kv-pool-with-more-dgx-spark-nodes).

→ [profile sweeps](results/dual-spark-vllm.md) · [ideal settings per use case](results/long-context.md)

## 📏 Long context — what to expect
**2× DGX Spark · DeepSeek-V4-Flash official FP8 · vLLM TP=2** (single-stream; holds across profiles).
| Context | TTFT (to first token) | prefill t/s | decode t/s |
|--:|--:|--:|--:|
| 8k | 4 s | ~1970 | 40 |
| 100k | 52 s | ~1900 | 40 |
| 200k | ~2 min | ~1750 | 38 |
| 500k | ~6 min | ~1380 | **32** |

Prefill is **~linear** (V4's sparse attention dodges the O(n²) wall) and **decode barely drops even at 500k**
— the only cost is **time-to-first-token**. Sustained / continuous use: **no throughput decay, no memory
creep**. *40+ t/s per request ⇒ single-stream only* (compute-bound ~350 t/s total). → [`long-context.md`](results/long-context.md)

## 💡 Key takeaways
- **Decode is memory-bandwidth bound** — the Spark is a **capacity** machine, not a latency one (small models
  run ~2.5–3.4× slower than a fat-GDDR card).
- **Prefer MoE** — active/total params is the speed multiplier: a 35B MoE (67 t/s) beats a 27B dense (13) by
  5×; a **172B MoE runs at 31 t/s**.
- **The niche: frontier-size models on one or two boxes** — MiniMax-172B and full-FP8 V4-Flash don't fit a
  16 GB (or even 96 GB) card; **two Sparks deliver V4-Flash FP8 at ~40 tok/s single + ~350 aggregate**.

## ✅ Stability (root cause fixed 2026-06-16)
Higher-concurrency stress once triggered a **kernel crash + reboot**. Root cause: a **vLLM prefix-cache memory
leak** ([PR #44237](https://github.com/vllm-project/vllm/pull/44237) — *"linear host RSS growth under sustained
load with prefix caching"*). Host RAM climbed under sustained load until `kcompactd` (memory compaction)
soft-locked a CPU, colliding with a latent NVIDIA `dgx-telemetry`/`mstflint` kernel NULL-deref under RoCE load.
(The `mlx5 "insufficient power 27W"` log is a normal **red herring**.) **Fix: upgrade to the
`aidendle94/sparkrun-vllm-ds4-gb10:production-v2` image** (PR #44237 baked in) — re-validated with a 15-min
sustained run showing **container memory flat (+4 MB)** + stable throughput, plus clean concurrent and
long-context passes. The `mstflint` trigger is still a latent NVIDIA bug — keep `vm.compaction_proactiveness=0`
+ memory headroom as defense-in-depth. Full forensics → [`SETUP-NOTES.md §7`](SETUP-NOTES.md).
*Seen it on your Spark? Let's compare notes.*

## 📂 What's in the repo
| Path | What |
|---|---|
| [`results/FINAL-BENCHMARKS.md`](results/FINAL-BENCHMARKS.md) | one-page summary of every number |
| [`results/long-context.md`](results/long-context.md) | long-context curves, stress/stability, **ideal settings per use case** |
| [`results/cross-machine.md`](results/cross-machine.md) | DeepSeek-V4-Flash across the 4 machines |
| [`results/dual-spark-vllm.md`](results/dual-spark-vllm.md) | dual-Spark profiles + concurrency sweeps |
| [`results/llamacpp-gb10.md`](results/llamacpp-gb10.md) · [`SUMMARY.md`](results/SUMMARY.md) | 10-model GGUF suite + analysis |
| [`SETUP-NOTES.md`](SETUP-NOTES.md) | **the gotchas** — HF Xet hangs, QSFP/NCCL/GID, NetworkManager, the kernel crash |
| [`scripts/`](scripts/) · `run-spark-bench.py` · `models.tsv` | serve presets, benches, the GGUF-suite runner |
| [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) | credits |

## 🚀 Quickstart
```bash
python3 run-spark-bench.py                  # download + bench the 10-model GGUF suite
./scripts/serve-gb10 gemma4-26b-mtp         # serve a GGUF model, GB10-tuned (:8080)
./scripts/conc-probe.py --n 1 --n 2 --n 4   # concurrency probe against a running server
```
Dual-Spark vLLM recipe + launch: [`SETUP-NOTES.md`](SETUP-NOTES.md) + the
[MiaAI-Lab compose](https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context).

## 🙏 Acknowledgments
Stands entirely on others' work — **antirez** (`ds4` engine + GGUF weights), **Aiden** (`aidendle94`, the
dual-Spark vLLM recipe + GB10 image), the **NVIDIA DGX Spark forum community** (the TP=2 recipe threads),
**llama.cpp**, **vLLM**, **DeepSeek**, and the GGUF quant authors. Full credits + links →
[`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md). Hoping to **contribute back**. 🙏

## Roadmap
- ✅ Dual-Spark FP8 TP=2 · single-Spark `ds4` · 10-model GGUF suite · cross-machine · long-context + stability
- ⏳ Pool-saturating (36 × 250k) stress + an NVIDIA firmware fix for the crash
- ⏳ Phase 2: TRT-LLM NVFP4 + vLLM W4A16/LMCache on arm64 · ik_llama aarch64 port · full MTP A/B

*Personal 2× DGX Spark lab. PRs and other-machine numbers welcome.*
