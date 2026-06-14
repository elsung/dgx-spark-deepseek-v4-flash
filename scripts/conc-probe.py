#!/usr/bin/env python3
"""Concurrency probe for an OpenAI-compatible llama.cpp / ik_llama.cpp server.

Fires N requests at once and reports aggregate tok/s, per-stream tok/s, and the
slowest request's latency. Works for text or (with --image) a vision request.

Usage:
  conc-probe.py --host 127.0.0.1:8080 --n 4 --max-tokens 256            # text
  conc-probe.py --n 2 --max-tokens 256 --image picture.png             # vision
  conc-probe.py --n 1 --n 2 --n 4 --max-tokens 256                     # sweep

Notes:
  * --ignore-eos forces every request to emit exactly --max-tokens, giving a
    clean throughput number (default on). Turn off with --no-ignore-eos.
  * Text uses /v1/completions; vision uses /v1/chat/completions with a base64
    data URL, so it needs a server started with --mmproj.
"""
import argparse, base64, concurrent.futures, json, sys, time, urllib.request


def build_text_body(max_tokens, prompt_tokens, ignore_eos):
    prompt = "Write a detailed technical essay about distributed systems. " * max(1, prompt_tokens // 9)
    return json.dumps({"model": "x", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.7, "ignore_eos": ignore_eos, "stream": False}).encode()


def build_vision_body(max_tokens, image_path):
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Describe this image and read all text exactly."}]}]
    return json.dumps({"model": "x", "messages": msg, "max_tokens": max_tokens,
                       "temperature": 0.2, "stream": False}).encode()


def fire(host, path, body, timeout):
    req = urllib.request.Request(f"http://{host}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d.get("usage", {}).get("completion_tokens", 0), time.time() - t0


def run(host, n, body, path, timeout):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda _: fire(host, path, body, timeout), range(n)))
    wall = time.time() - t0
    gen = sum(r[0] for r in res)
    lat = max(r[1] for r in res)
    agg = gen / wall if wall else 0
    print(f"N={n:<2} gen={gen:<5} wall={wall:5.1f}s | AGG={agg:6.1f} tok/s | "
          f"per-stream~{agg/n:5.1f} | max-latency={lat:.1f}s")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:8080")
    ap.add_argument("--n", type=int, action="append", help="concurrency level (repeatable for a sweep)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--image", default=None, help="path to image → runs a vision request")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    args = ap.parse_args()

    levels = args.n or [1, 2, 4]
    if args.image:
        body, path = build_vision_body(args.max_tokens, args.image), "/v1/chat/completions"
        print(f"# vision sweep ({args.image}), max_tokens={args.max_tokens}")
    else:
        body, path = build_text_body(args.max_tokens, args.prompt_tokens, args.ignore_eos), "/v1/completions"
        print(f"# text sweep, max_tokens={args.max_tokens}, prompt~{args.prompt_tokens} tok")
    for n in levels:
        run(args.host, n, body, path, args.timeout)


if __name__ == "__main__":
    main()
