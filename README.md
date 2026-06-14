# dgx-spark-llm-bench

LLM inference benchmarks on the **NVIDIA DGX Spark (GB10, sm_121, 128 GB LPDDR5X unified memory, arm64)**.
Ports [@elsung](https://github.com/elsung)'s [blackwell-16gb-llm-starter](https://github.com/elsung/blackwell-16gb-llm-starter)
and [blackwell-llm-toolkit](https://github.com/elsung/blackwell-llm-toolkit) (RTX 50-series 16 GB / RTX PRO 6000 96 GB, sm_120, x86)
to the Spark. Intended to be pushed as a standalone DGX-Spark benchmark repo.

## What's different on GB10 vs the reference cards
- **128 GB unified memory** → every model is **fully GPU-resident (`-ngl 999`, no `-ncmoe` offload)**.
  The big models that needed a 96 GB card (MiniMax-172B/139B, DeepSeek-V4-Flash) **fit here**, and the
  16 GB-card MoE-offload penalty disappears — these numbers should beat the reference where offload was forced.
- **sm_121** (not sm_120): all GGUF engines built from source with `-DCMAKE_CUDA_ARCHITECTURES=121`.
- **arm64**: the toolkit's x86 NVFP4/W4A16 wheels (TRT-LLM / vLLM / LMCache) don't apply — those are
  **Phase 2** (need arm64 builds). This repo is the **GGUF suite** (llama.cpp + ik_llama.cpp) for now.

## Engines (built in ~/AI)
- mainline `llama.cpp` (`~/AI/llama.cpp/build/bin`) — Q/UD/IQ4_XS quants, vision, mainline MTP
- `ik_llama.cpp` (`~/AI/ik_llama.cpp/build/bin`) — IQ_K quants + Gemma-4 MTP assistants

## Model suite (GGUF) — `models.tsv`
9B → 172B, spanning dense + MoE, the exact quants from the starter/toolkit benches plus the big MoEs the
Spark uniquely fits. Weights land in `~/LLMs/models/gguf/` (flat symlinks like `Qwen3.6-27B-IQ4KS.gguf`).

## Run
```bash
# unattended: waits for FP8 dl, pulls ds4, then downloads+benches every model
nohup bash run-everything.sh > ~/LLMs/.spark-pipeline.log 2>&1 &

# or just the benchmark suite (weights auto-download)
python3 run-spark-bench.py

# serve any model with GB10-tuned settings (OpenAI API :8080)
./scripts/serve-gb10 gemma4-26b-mtp
# concurrency probe (other terminal)
./scripts/conc-probe.py --n 1 --n 2 --n 4 --max-tokens 300
```

## Results
`results/llamacpp-gb10.md` (+ `.jsonl`) — pp512 (prefill) + tg128 (decode) single-stream, MTP A/B.
Method mirrors the source repos: `llama-bench -ngl 999 -fa 1 -p 512 -n 128 -r 3`.

## Known limitations on GB10/arm64
- **ik_llama.cpp does not build cleanly on aarch64** (`iqk_cpu_ops.cpp`: `v_expf`/`v_silu`/`ggml_half`
  undeclared — the IQK kernels are x86/AVX-tuned). The ik-only `IQ4_KS` Qwen3.6-27B is substituted with the
  mainline `IQ4_XS` of the same model; Gemma-4 26B/31B use their standard quants on mainline. ik ARM port = TODO.
- **MTP/speculative speedup is a llama-server feature** — `llama-bench` can't measure it, so the suite reports
  base pp512/tg128 only. MTP A/B is measured separately via `llama-server` + timed decode (see ds4: ~14 tok/s
  with MTP draft=2 on one Spark). `results/mtp-gb10.md` TODO.

## Phase 2 (TODO, needs arm64 builds)
- vLLM (AWQ/W4A16) + LMCache, TRT-LLM (NVFP4 Nemotron) on GB10/arm64
- `rapid_bench.py` 41-prompt quality eval + concurrency harness
- DeepSeek-V4-Flash dual-Spark TP=2 numbers (see `~/AI/spark-vllm-ds4`)
