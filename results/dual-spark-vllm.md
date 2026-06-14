# DeepSeek-V4-Flash — dual DGX Spark (vLLM TP=2)

Official FP8 (284B/13B-active MoE) across 2× GB10 over a 200G QSFP RoCE link.
vLLM 0.21.1 (Aiden image), TP=2, MTP draft=2, fp8 KV, 1M context, expert-parallel.

## Single-stream
- vLLM server-side **Avg generation throughput: ~38.5 tok/s** (steady state)
- End-to-end (incl. prefill): **42–44 tok/s** (229 tok/5.3s, 500 tok/11.8s)
- MTP: mean acceptance length **2.04–2.29**, draft acceptance **51–64%** (higher on code)
- Cold start ~8.7 min; TTFT a few s (reasoning model, reasoning_effort=high)

## Concurrency sweep (max_tokens=256, max_num_seqs=6)
| N | aggregate tok/s | per-stream tok/s | wall s |
|--:|--:|--:|--:|
| 1 | 32.2 | 32.2 | 7.9 |
| 2 | 49.0 | 24.5 | 10.0 |
| 4 | 44.7 | 11.2 | 19.4 |
| 6 | 66.0 | 11.0 | 19.6 |

Aggregate scales ~1.5× at c=2 and ~2× at c=6 vs single-stream. (Numbers vary run-to-run because the
reasoning model emits variable-length think chains before max_tokens.)
