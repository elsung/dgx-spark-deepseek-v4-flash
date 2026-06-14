# DeepSeek-V4-Flash — dual DGX Spark (vLLM TP=2)

Official FP8 (284B/13B-active MoE) across 2× GB10 over a 200G QSFP RoCE link.
vLLM 0.21.1 (Aiden image), TP=2, MTP draft=2, fp8 KV, expert-parallel.

## Single-stream
- vLLM server-side **Avg generation throughput: ~38.5 tok/s** (steady state)
- End-to-end (incl. prefill): **42–44 tok/s** (229 tok/5.3s, 500 tok/11.8s)
- MTP: mean acceptance length **2.04–2.29**, draft acceptance **51–64%** (higher on code)
- Cold start ~7–9 min; TTFT a few s (reasoning model, reasoning_effort=high)

## Concurrency — two profiles (clean: ignore_eos, fixed-length gen)

**Long-context profile (1M ctx, max_num_seqs=6)** — slot-limited:
| c | 1 | 2 | 4 | 6 | 8 | 12 |
|---|--:|--:|--:|--:|--:|--:|
| agg tok/s | 32 | 59 | 66 | **103** | 96 | 110 |

**High-throughput profile (32K ctx, max_num_seqs=36)** — compute-limited:
| c | 1 | 8 | 16 | 24 | 32 | 36 |
|---|--:|--:|--:|--:|--:|--:|
| agg tok/s | 36 | 130 | 180 | 233 | **270** | 193 |

- **Peak ~270 tok/s aggregate** (vLLM server metric confirms 265–275 t/s at 32 running reqs),
  ~7× single-stream. **Compute-bound: both GB10s at 95–96% GPU util** at c=32 — the ceiling,
  not a memory limit (**KV cache only ~7% used** at 36 seqs/32K ctx — lots of context headroom).
- Profiles switch via `.env`: `MAX_MODEL_LEN` + `MAX_NUM_SEQS` (compose defaults 32768/36).
  KV is so cheap (V4 compressed attention) you can run ~256K ctx at 36 slots and still fit.

Measure it yourself: `mytoolz dsv4-bench --n 1 --n 8 --n 24 --n 32`
