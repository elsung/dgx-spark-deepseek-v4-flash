# Long-context — prefill & decode vs depth (dual DGX Spark, FP8 TP=2)

Prior numbers were all fresh/short-prompt. This measures prefill + decode at real context depths.
Method: cache-free (unique fresh prompts, time-randomized salt). `bench-longctx.py`. Profile: `base` (1M/6).

## Single-stream vs context depth
| context (actual tok) | TTFT / prefill time | prefill t/s | decode t/s |
|--:|--:|--:|--:|
| 7.8k | 3.9 s | 1974 | 39.8 |
| 31k | 15.5 s | 2016 | 38.6 |
| 100k | 52.5 s | 1901 | 39.5 |
| 200k | 115 s | 1747 | 37.6 |

## Findings
1. **Prefill is ~linear (~1900 t/s, roughly flat 8k→200k)** — DeepSeek-V4's compressed/sparse attention
   avoids the usual O(n²) prefill blowup. **Cost = TTFT:** a 200k prompt is ~2 min to first token.
2. **Decode barely degrades with depth** (~40 → ~38 at 200k). Cross-checked with a cached-prefix-reuse
   method (~31–39 at depth); a streaming method gave ~16 but is believed to be an artifact. Net: V4
   largely preserves decode speed at long context — its core design goal.

## Stability note (important)
NVRM `NV_ERR_NO_MEMORY` events rose **1 → 3** this boot during the long-context runs (system RAM ~11 GB
free; `gpu_memory_utilization=0.82`). Long-context load stresses memory — the same pressure that fed the
earlier kernel crash (SETUP-NOTES §7). **Sustained / concurrent big-context runs carry crash risk** and
need the mitigations (compaction off, headroom) + monitoring.

## TODO (campaign in progress)
- Concurrent long-context at the edge (`batch` 256K/36: N × ~200–250k).
- 500k single + low concurrency (`custom 524288 / ~10`).
- Sustained/continuous (30–60 min) — throughput decay? memory creep? stability?
- Cross-box: RTX PRO 6000 (≤131k), Mac M2 Ultra (≤1M).
- → ideal settings per use case.
