# DGX Spark (GB10, sm_121, 128GB unified) — llama.cpp/ik single-stream

`-ngl 999 -fa 1 -p 512 -n 128 -r 3` (fully GPU-resident, no offload).

| Model | Engine | pp512 (prefill t/s) | tg128 (decode t/s) | +MTP decode |
|---|---|--:|--:|--:|
| Qwen3.5-9B-Q5KXL | llama | 2337.0 | 33.6 | — |
| Qwen3.5-9B-MTP | llama | 2473.9 | 33.5 | — |
| Gemma4-12B-Q4KXL | llama | 1951.0 | 27.8 | — |
| Gemma4-26B-A4B-IQ4XS | llama | 2916.8 | 60.3 | — |
| Qwen3.6-27B-IQ4XS | llama | 898.9 | 13.7 | — |
| Gemma4-31B-Q3KS | llama | 718.8 | 12.9 | — |
| Qwen3.6-35B-A3B-Q4KM | llama | 2305.0 | 67.4 | — |
| Gemma3-27B-QAT-Q4_0 | llama | 1030.6 | 13.8 | — |
| MiniMax-172B-Q3KS | llama | 525.1 | 31.2 | — |
| MiniMax-139B-Q3KM | llama | 558.2 | 26.5 | — |
