#!/usr/bin/env python3
"""
DGX Spark (GB10, sm_121, 128 GB unified) GGUF benchmark pipeline.

Adapts elsung/blackwell-16gb-llm-starter + blackwell-llm-toolkit to the Spark:
 - 128 GB unified memory -> every model fully GPU-resident (-ngl 999), NO -ncmoe offload
 - measures pp512 (prefill) + tg128 (decode) single-stream via llama-bench (-o json)
 - mainline llama.cpp for Q/UD/IQ4_XS quants; ik_llama.cpp for IQ_K quants
 - MTP A/B where an assistant exists

Runs unattended: waits for the FP8 download to free HF connection slots, then
downloads each model serially (reliable) and benches it right after.
Results -> results/llamacpp-gb10.md (+ .jsonl). Re-running skips done models.
"""
import json, os, subprocess, sys, time, shutil
from pathlib import Path

HOME = Path.home()
GG = HOME / "LLMs/models/gguf"
RESULTS = Path(__file__).parent / "results"
LLAMA = HOME / "AI/llama.cpp/build/bin"
IK = HOME / "AI/ik_llama.cpp/build/bin"
HF = str(HOME / ".local/bin/hf")
ENV = {**os.environ, "PATH": f"{HOME}/.local/bin:" + os.environ.get("PATH", ""),
       "HF_XET_HIGH_PERFORMANCE": "1"}

# name, engine(llama|ik), repo, srcfile, mmproj(repo:file|None), mtp(repo:file|None)
J = "ji-farthing/gemma-4-qat-q4_0-MTP-assistants-ik-llama-GGUF"
MODELS = [
 ("Qwen3.5-9B-Q5KXL","llama","unsloth/Qwen3.5-9B-GGUF","Qwen3.5-9B-UD-Q5_K_XL.gguf",
    "unsloth/Qwen3.5-9B-GGUF:mmproj-F16.gguf", None),
 ("Qwen3.5-9B-MTP","llama","unsloth/Qwen3.5-9B-MTP-GGUF","Qwen3.5-9B-UD-Q5_K_XL.gguf", None, "self"),
 ("Gemma4-12B-Q4KXL","llama","unsloth/gemma-4-12b-it-GGUF","gemma-4-12b-it-UD-Q4_K_XL.gguf",
    "unsloth/gemma-4-12b-it-GGUF:mmproj-F16.gguf", None),
 ("Gemma4-26B-A4B-IQ4XS","llama","unsloth/gemma-4-26B-A4B-it-GGUF","gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
    "unsloth/gemma-4-26B-A4B-it-GGUF:mmproj-F16.gguf", None),
 ("Qwen3.6-27B-IQ4XS","llama","unsloth/Qwen3.6-27B-GGUF","Qwen3.6-27B-IQ4_XS.gguf",
    "unsloth/Qwen3.6-27B-GGUF:mmproj-F16.gguf", None),
 ("Gemma4-31B-Q3KS","llama","unsloth/gemma-4-31B-it-GGUF","gemma-4-31B-it-Q3_K_S.gguf",
    None, None),
 ("Qwen3.6-35B-A3B-Q4KM","llama","unsloth/Qwen3.6-35B-A3B-GGUF","Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "unsloth/Qwen3.6-35B-A3B-GGUF:mmproj-F16.gguf", None),
 ("Gemma3-27B-QAT-Q4_0","llama","lmstudio-community/gemma-3-27B-it-qat-GGUF","gemma-3-27B-it-QAT-Q4_0.gguf",
    "lmstudio-community/gemma-3-27B-it-qat-GGUF:mmproj-model-f16.gguf", None),
 ("MiniMax-172B-Q3KS","llama","exdysa/MiniMax-M2.7-REAP-172B-A10B-GGUF","MiniMax-M2.7-REAP-172B-A10B-Q3_K_S.gguf", None, None),
 ("MiniMax-139B-Q3KM","llama","dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-GGUF","MiniMax-M2.7-REAP-139B-Q3_K_M.gguf", None, None),
]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def dl(repo, srcfile, outname):
    out = GG / outname
    if out.exists() and out.stat().st_size > 1_000_000:
        log(f"  have {outname}"); return out
    log(f"  downloading {repo} :: {srcfile} -> {outname}")
    # hf downloads into a repo subdir under GG/.dl, then we symlink/rename to a flat name
    dest = GG / ".dl" / repo.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    # self-healing download (auto-restarts if Xet client hangs)
    rhf = str(HOME / "LLMs/scripts/resilient-hf.sh")
    r = subprocess.run(["bash", rhf, "file", repo, str(dest), srcfile], env=ENV)
    if r.returncode != 0:
        log(f"  !! download failed {repo} {srcfile}"); return None
    src = dest / srcfile
    if not src.exists():
        log(f"  !! missing after download: {src}"); return None
    if out.exists() or out.is_symlink(): out.unlink()
    out.symlink_to(src)
    return out

def wait_for_fp8():
    log("waiting for FP8 download to finish (frees HF Xet slots)...")
    while subprocess.run(["pgrep","-f","hf download deepseek"],capture_output=True).returncode==0:
        time.sleep(30)
    log("FP8 done — starting model pipeline.")

def wait_for_ik():
    while not (IK/"llama-bench").exists():
        log("  (ik_llama not built yet — waiting 30s)"); time.sleep(30)

def bench(bindir, model, extra=None, label=""):
    """Run llama-bench, return {pp512, tg128} tok/s."""
    cmd = [str(bindir/"llama-bench"), "-m", str(model), "-ngl", "999", "-fa", "1",
           "-p", "512", "-n", "128", "-r", "3", "-o", "json"]
    if extra: cmd += extra
    log(f"  bench {label or model.name}: {' '.join(cmd[-12:])}")
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        log(f"  !! bench failed: {r.stderr[-400:]}"); return None
    try:
        data = json.loads(r.stdout)
        out = {}
        for row in data:
            n = row.get("n_prompt",0); g = row.get("n_gen",0); ts = row.get("avg_ts")
            if n and not g: out["pp512"] = round(ts,1)
            if g and not n: out["tg128"] = round(ts,1)
        return out
    except Exception as e:
        log(f"  !! parse failed: {e}; raw: {r.stdout[:200]}"); return None

def main():
    GG.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    wait_for_fp8()
    jsonl = RESULTS/"llamacpp-gb10.jsonl"
    md = RESULTS/"llamacpp-gb10.md"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try: done.add(json.loads(line)["name"])
            except: pass
    for name, engine, repo, srcfile, mmproj, mtp in MODELS:
        if name in done: log(f"== {name}: already benched, skip"); continue
        log(f"== {name} ({engine}) ==")
        model = dl(repo, srcfile, f"{name}.gguf")
        if mmproj: r,f = mmproj.split(":",1); dl(r,f,f"{name}.mmproj.gguf")
        mtp_path = None
        if mtp == "self": mtp_path = model
        elif mtp: r,f = mtp.split(":",1); mtp_path = dl(r,f,f"{name}.mtp.gguf")
        if not model: continue
        binu = IK if engine=="ik" else LLAMA
        if engine=="ik" and not (IK/"llama-bench").exists():
            log(f"  SKIP {name}: ik_llama not built on aarch64 (ik-only quant)"); continue
        base = bench(binu, model, label=f"{name} base")
        rec = {"name":name,"engine":engine,"file":srcfile,"base":base}
        # NOTE: MTP/speculative speedup is a llama-SERVER feature; llama-bench can't measure it.
        # Measured separately via llama-server + timed decode (results/mtp-gb10.md, TODO).
        with open(jsonl,"a") as fh: fh.write(json.dumps(rec)+"\n")
        # rewrite md table
        rows=[json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        with open(md,"w") as fh:
            fh.write("# DGX Spark (GB10, sm_121, 128GB unified) — llama.cpp/ik single-stream\n\n")
            fh.write("`-ngl 999 -fa 1 -p 512 -n 128 -r 3` (fully GPU-resident, no offload).\n\n")
            fh.write("| Model | Engine | pp512 (prefill t/s) | tg128 (decode t/s) | +MTP decode |\n|---|---|--:|--:|--:|\n")
            for x in rows:
                b=x.get("base") or {}; m=x.get("mtp") or {}
                fh.write(f"| {x['name']} | {x['engine']} | {b.get('pp512','—')} | {b.get('tg128','—')} | {m.get('tg128','—')} |\n")
        log(f"== {name} done: {base}")
    log("ALL DONE.")

if __name__=="__main__":
    main()
