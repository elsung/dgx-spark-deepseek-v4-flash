# Acknowledgments

None of this would exist without the people who did the hard parts in the open. Huge thanks to:

## The engines & models
- **[antirez](https://github.com/antirez) (Salvatore Sanfilippo)** — **[`ds4` / DwarfStar](https://github.com/antirez/ds4)**,
  a from-scratch CUDA/Metal/ROCm DeepSeek-V4-Flash inference engine, plus the quantized
  [GGUF weights](https://huggingface.co/antirez/deepseek-v4-gguf). This is what makes DeepSeek-V4-Flash run
  on a *single* Spark (and on the Mac / RTX boxes in our comparison). Beautiful, focused work.
- **[Aiden](https://hub.docker.com/r/aidendle94/sparkrun-vllm-ds4-gb10) (`aidendle94`)** — the dual-Spark
  **vLLM recipe and GB10 Docker image** (`aidendle94/sparkrun-vllm-ds4-gb10`) that powers our TP=2 run.
  The whole headline result rides on this.
- **[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)** — for DeepSeek-V4-Flash itself.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org / Georgi Gerganov & contributors) — the
  foundation under the GGUF benchmark suite and so much of local inference.
- **[vLLM](https://github.com/vllm-project/vllm)** — the serving engine for the dual-Spark FP8 path
  (incl. the GB10/`jasl` fork lineage).

## The DGX Spark forum community
The TP=2-across-two-Sparks recipe was reverse-engineered and shared in the open on the
[NVIDIA DGX Spark / GB10 developer forums](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark/):
- The **["official FP8 across 2× DGX Spark — TP=2, MTP, 200K ctx"](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309)** thread.
- The **[Aiden recipe / 1M-token-session thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-aiden-recipe-from-reddit-1m-token-session-operational-cuda-12-1-tailored-for-dgx-spark-gb10/372268)**.
- The **[antirez ds4 on 1× Spark](https://forums.developer.nvidia.com/t/fully-custom-cuda-native-deepseek-4-flash-optimized-for-1x-spark-antirez-ds4/369791)** thread.
- **[MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context)** — the packaged
  dual-Spark `docker-compose` + scripts we built on.
- The `jasl` vLLM fork, `eugr/spark-vllm-docker` (PR #219 lineage), `tonyd2wild`, and the many forum members
  posting benchmarks, patches, and debugging notes. This community moves fast and shares generously.

## The quant authors (GGUF benchmark suite)
**unsloth**, **cHunter789**, **ji-farthing**, **exdysa**, **dervig**, **teamblobfish**, and
**lmstudio-community** — for the imatrix/UD/IQ_K quants and MoE GGUFs we benchmarked.

## And
Prior art this builds on: my own [blackwell-16gb-llm-starter](https://github.com/elsung/blackwell-16gb-llm-starter)
and [blackwell-llm-toolkit](https://github.com/elsung/blackwell-llm-toolkit).

There are more improvements still landing in these threads as of mid-2026 — we're watching closely and hope
to **contribute back** (numbers, the stability findings, and fixes) rather than just take. 🙏
