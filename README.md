# DeepSeek-V4-Flash on DGX Spark — dual-node TP=2, single-node, and a cross-machine shootout

Getting **DeepSeek-V4-Flash** running on the **NVIDIA DGX Spark (GB10)** — including the headline:
the **official FP8 model across *two* Sparks via vLLM tensor-parallel (TP=2) over a QSFP56 RoCE link** —
plus a single-Spark path (antirez `ds4`), a 10-model llama.cpp GGUF benchmark suite, and a cross-machine
comparison (Mac Studio M2 Ultra, RTX PRO 6000). Written down with **all the gotchas that cost us hours**.

Builds on my earlier [blackwell-16gb-llm-starter](https://github.com/elsung/blackwell-16gb-llm-starter)
and [blackwell-llm-toolkit](https://github.com/elsung/blackwell-llm-toolkit), ported to GB10 (sm_121, arm64).

> **How this was made:** the runs, benchmarks, configs, gotchas, and write-ups in this repo are produced by
> **AI agents (Claude Code)** that I ([@elsung](https://github.com/elsung)) direct — I drive the hardware,
> set the goals, and push the models to do the benchmarking, tune the configs, and document the findings.
> Numbers are real (measured on the hardware below); treat the prose as AI-authored and verify before relying on it.

## Hardware
2× **NVIDIA DGX Spark** (GB10, 128 GB LPDDR5X unified, sm_121, arm64, CUDA 13) joined by a single
**200 G QSFP56** direct cable (RoCE / NCCL).

## 🏁 Headline results
| Setup | Model / quant | single-stream | aggregate |
|---|---|--:|--:|
| **2× DGX Spark, TP=2 (vLLM)** | DeepSeek-V4-Flash **official FP8** | **~41 tok/s** | **~350 tok/s @ c=32 (256K ctx)** |
| 1× DGX Spark (antirez ds4) | DeepSeek-V4-Flash IQ2_XXS | ~14 tok/s | — |
| 1× DGX Spark (llama.cpp GGUF suite) | 9B → **172B** MoE | up to 67 (35B-A3B) | — |

**Cross-machine, same model (single-stream decode / prefill tok/s):** RTX PRO 6000 **46.9 / 344** ·
Mac M2 Ultra **29.7 / 389** · dual Spark FP8 **~41 / ~1785** · single Spark IQ2 ~14. Only the Sparks run the
**full FP8** quality, have **~5× the prefill**, *and* do real **multi-stream throughput (~350 agg)** —
the ds4.c boxes are single-stream.

Full numbers: [`results/FINAL-BENCHMARKS.md`](results/FINAL-BENCHMARKS.md) ·
[`cross-machine.md`](results/cross-machine.md) · [`dual-spark-vllm.md`](results/dual-spark-vllm.md) ·
[`llamacpp-gb10.md`](results/llamacpp-gb10.md) · [`SUMMARY.md`](results/SUMMARY.md).

### What the numbers say
1. **Decode is memory-bandwidth bound** — small models run ~2.5–3.4× slower than a fat-GDDR card; the
   Spark is a **capacity** machine, not a latency one.
2. **Prefer MoE** — active/total params is the speed multiplier: a 35B MoE (67 t/s) beats a 27B dense
   (13) by 5×; a **172B MoE runs at 31 t/s**.
3. **The Spark's niche: frontier-size models on one or two boxes** — MiniMax-172B and DeepSeek-V4-Flash
   don't fit a 16 GB (or even 96 GB) card; **two Sparks deliver the full FP8 V4-Flash at ~40 tok/s**.

## ⚠️ Stability — open issue (help wanted, contributions welcome)
While stress-testing **higher concurrency**, the head node hit a **kernel crash + spontaneous reboot.**
Root cause (full forensics in [`SETUP-NOTES.md §7`](SETUP-NOTES.md)):
- **Trigger:** `nvidia-dgx-telemetry` periodically runs **`mstflint`** to poll the ConnectX-7 firmware; one
  poll **NULL-deref'd the kernel** in `pci_bus_read_config_dword` (with IRQs disabled) under heavy RoCE load
  — a kernel/MST-driver bug.
- **Wedge:** that Oops tainted the kernel, then **`kcompactd` (memory compaction) soft-locked a CPU for 48 s**
  under memory pressure → RCU stall → hang.
- **Red herring:** `mlx5 "insufficient power on the PCIe slot (27W)"` logs on *every* boot, both nodes — it's
  a normal trait of the integrated CX-7 on a PCIe x4 link, **not** the cause.

**Current mitigations** (don't lose capability): `vm.compaction_proactiveness=0` (persistent), drop page
cache before big loads, keep memory headroom. **Sustained high-concurrency stability testing is still TODO** —
and this looks like an NVIDIA firmware/driver bug worth a DGX support ticket. *If you've seen this on a Spark,
let's compare notes.*

## What's in here
- [`results/`](results/) — every number (final summary, cross-machine, dual-Spark vLLM sweeps, GGUF suite)
- [`SETUP-NOTES.md`](SETUP-NOTES.md) — **the gotchas**: HF Xet download hangs + watchdog; QSFP rightmost-port /
  firewall / NetworkManager-wipes-IP; docker-group-without-relogin; ik_llama-on-arm64; **the kernel crash §7**
- [`scripts/`](scripts/) — GB10-tuned serve presets, the concurrency bench, the download watchdog
- `run-spark-bench.py` / `models.tsv` — the 10-model GGUF suite (filenames HF-verified)

## Run it
```bash
python3 run-spark-bench.py                      # download + bench the 10-model GGUF suite
./scripts/serve-gb10 gemma4-26b-mtp             # serve a model, GB10-tuned (:8080)
./scripts/conc-probe.py --n 1 --n 2 --n 4       # concurrency probe
```
Dual-Spark vLLM recipe + launch: see [`SETUP-NOTES.md`](SETUP-NOTES.md) and the
[MiaAI-Lab compose](https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context).

## 🙏 Acknowledgments
This stands entirely on others' work — see [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md). In short:
**antirez** (the `ds4`/DwarfStar engine + GGUF weights), **Aiden** (`aidendle94`, the dual-Spark vLLM recipe
& GB10 image), the **NVIDIA DGX Spark forum community** (the TP=2 recipe threads), **llama.cpp**, **vLLM**,
**DeepSeek**, and the GGUF quant authors. More improvements are landing in these threads fast — we're
watching and hope to **contribute back**.

## Status / roadmap
- ✅ Dual-Spark FP8 TP=2 working; single-Spark ds4; 10-model GGUF suite; cross-machine comparison
- ⏳ Higher-context (256K/500K) + sustained-concurrency **stability** validation
- ⏳ Phase 2: TRT-LLM (NVFP4) + vLLM W4A16/LMCache on arm64; ik_llama aarch64 port; full MTP A/B

*Hardware/results from a personal 2× DGX Spark lab. PRs and other-machine numbers welcome.*
