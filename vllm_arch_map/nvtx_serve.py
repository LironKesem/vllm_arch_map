#!/usr/bin/env python3
"""Wrap vllm serve with NVTX markers for per-request profiling.

Adds NVTX range_push/range_pop around each request so nsys can
segment the kernel timeline into individual requests.

How it works:

  vllm bench serve (client)                    nvtx_serve.py (server, GPU process)
  ─────────────────────────                    ────────────────────────────────────
  sends HTTP request  ──────────────────────►  receives request
                                               │
                                               ▼
                                               engine.generate()  ◄── patched
                                               │  nvtx.range_push("request_xxx")
                                               │  run model forward (GPU kernels)
                                               │  nvtx.range_pop()
                                               │
  returns response   ◄──────────────────────   sends response back

Every request — curl, benchmark, random dataset — goes through generate().
The patch wraps it with NVTX markers. nsys records them in NVTX_EVENTS.

Usage:
  nsys profile -t cuda,nvtx -o trace \
      python nvtx_serve.py Qwen/Qwen3-8B --max-model-len 8192

  # Then in another terminal:
  vllm bench serve --backend vllm --model Qwen/Qwen3-8B \
      --endpoint /v1/completions --dataset-name random \
      --num-prompts 15 --max-concurrency 1

  # Export and view:
  nsys export --type=sqlite trace.nsys-rep

  # NVTX_EVENTS will have rows like "request_cmpl-abc123" marking each request.

See README.md → "Quickstart: NSYS Per-Request Timeline" for full steps.
"""
import importlib
import sys

import torch.cuda.nvtx as nvtx

# Engine class candidates across vLLM versions
_ENGINE_CANDIDATES = [
    ("vllm.v1.engine.async_llm", "AsyncLLM"),               # v0.29+ (V2 default)
    ("vllm.engine.async_llm_engine", "AsyncLLMEngine"),      # v0.4–v0.28
    ("vllm.engine.async_llm", "AsyncLLMEngine"),             # some intermediate versions
]

# CLI entry point candidates across vLLM versions
_CLI_CANDIDATES = [
    "vllm.entrypoints.cli.main",                             # v0.28+
    "vllm.entrypoints.openai.api_server",                    # older
]


def patch_engine():
    """Find and patch vLLM's engine generate() with NVTX markers."""
    for mod_path, cls_name in _ENGINE_CANDIDATES:
        try:
            mod = importlib.import_module(mod_path)
            engine_cls = getattr(mod, cls_name)
            break
        except (ImportError, AttributeError):
            continue
    else:
        print("[nvtx_serve] WARNING: could not find vLLM engine class — NVTX markers disabled")
        print(f"[nvtx_serve] Searched: {[f'{m}.{c}' for m, c in _ENGINE_CANDIDATES]}")
        return

    _orig = engine_cls.generate

    async def _nvtx_generate(self, *args, **kwargs):
        req_id = "unknown"
        if len(args) >= 3 and isinstance(args[2], str):
            req_id = args[2]
        elif "request_id" in kwargs:
            req_id = kwargs["request_id"]
        elif args and hasattr(args[0], "request_id"):
            req_id = getattr(args[0], "request_id", "unknown")

        nvtx.range_push(f"request_{req_id}")
        try:
            async for output in _orig(self, *args, **kwargs):
                yield output
        finally:
            nvtx.range_pop()

    engine_cls.generate = _nvtx_generate
    print(f"[nvtx_serve] Patched {mod_path}.{cls_name}.generate with NVTX markers")


def main():
    patch_engine()

    # Insert 'serve' if not already there (python nvtx_serve.py MODEL ... → vllm serve MODEL ...)
    if len(sys.argv) > 1 and sys.argv[1] != "serve":
        sys.argv.insert(1, "serve")

    for cli_mod in _CLI_CANDIDATES:
        try:
            mod = importlib.import_module(cli_mod)
            cli_main = getattr(mod, "main")
            cli_main()
            return
        except (ImportError, AttributeError):
            continue

    print("[nvtx_serve] ERROR: could not find vLLM CLI entry point")
    print(f"[nvtx_serve] Searched: {_CLI_CANDIDATES}")
    sys.exit(1)


if __name__ == "__main__":
    main()
