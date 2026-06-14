#!/usr/bin/env bash
# Master unattended pipeline for the DGX Spark bench repo.
# Order (serial, to avoid HF Xet connection contention):
#   1) wait for the FP8 weights download to finish
#   2) download antirez ds4 GGUFs (user priority: ds4 today)
#   3) download + benchmark the GGUF model suite (run-spark-bench.py)
# Log: ~/LLMs/.spark-pipeline.log
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_XET_HIGH_PERFORMANCE=1
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
RHF="$HOME/LLMs/scripts/resilient-hf.sh"   # self-healing hf (auto-restarts on Xet hang)

# (FP8 is downloaded separately with the watchdog before this script is launched.)

# --- ds4 (antirez), resilient ---
GG="$HOME/LLMs/models/ds4-gguf"
Q2="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
MTP="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
log "downloading ds4 MTP draft..."; "$RHF" file antirez/deepseek-v4-gguf "$GG" "$MTP" && log "ds4 MTP done"
log "downloading ds4 Q2 (~81GB)...";  "$RHF" file antirez/deepseek-v4-gguf "$GG" "$Q2"  && log "ds4 Q2 done"
ln -sf "$GG/$Q2" "$HOME/AI/ds4/ds4flash.gguf"
log "ds4 weights ready."

# --- benchmark suite (downloads + benches each model) ---
log "starting GGUF benchmark suite..."
python3 "$HOME/AI/dgx-spark-llm-bench/run-spark-bench.py"
log "PIPELINE COMPLETE."
