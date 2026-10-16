#!/usr/bin/env python3
"""Per-request profiling using vLLM offline mode.

Accepts the same dataset flags as `vllm bench serve` so profiled prompts
match your benchmark exactly.

Usage:
  python profile_serve.py Qwen/Qwen3-8B --num-prompts 3 --input-len 1024
  python profiler_to_diagram.py --per-request-dir traces/per_request --out_dir samples/
"""
import argparse
import os
import shutil
import uuid
from pathlib import Path

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

from walk_model import install_recorder, collect_recorder


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument("--profile-dir", default="./traces/per_request")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override model context length (default: from model config)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--max-tokens", type=int, default=4,
                        help="Max output tokens per request (keep small for profiling)")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tokenizer-mode", type=str, default="auto")

    # Dataset args — mirrors vllm bench serve
    from vllm.benchmarks.datasets import add_dataset_parser
    add_dataset_parser(parser)
    # Override vllm's default of 1000 — profiling wants few prompts
    parser.set_defaults(num_prompts=3)

    # General convenience aliases (same as vllm bench serve)
    parser.add_argument("--input-len", type=int, default=None,
                        help="Maps to --random-input-len, --sonnet-input-len, etc.")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Maps to --random-output-len, --sonnet-output-len, etc.")

    args = parser.parse_args()

    # Map general --input-len/--output-len to dataset-specific args
    if args.input_len is not None:
        for attr in ('random_input_len', 'sonnet_input_len'):
            if hasattr(args, attr):
                setattr(args, attr, args.input_len)
    if args.output_len is not None:
        for attr in ('random_output_len', 'sonnet_output_len', 'sharegpt_output_len',
                      'custom_output_len', 'hf_output_len'):
            if hasattr(args, attr):
                setattr(args, attr, args.output_len)

    from vllm import LLM, SamplingParams

    profile_dir = Path(args.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    trace_dir = profile_dir / f"profiles-{run_id}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=getattr(args, 'trust_remote_code', False),
        tokenizer=args.tokenizer or args.model,
        tokenizer_mode=args.tokenizer_mode,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(trace_dir),
            "torch_profiler_record_shapes": True,
        },
    )

    # Build prompts using vllm's dataset machinery — same prompts as bench serve
    from vllm.benchmarks.datasets import get_samples
    if not hasattr(args, 'backend'):
        args.backend = "openai-completions"
    tokenizer = llm.get_tokenizer()
    samples = get_samples(args, tokenizer)
    prompts = [s.prompt if isinstance(s.prompt, str) else str(s.prompt)
               for s in samples[:args.num_prompts]]

    for i, p in enumerate(prompts, 1):
        toks = len(tokenizer.encode(p))
        print(f"[profile_serve] Prompt {i}: {toks} tokens")

    params = SamplingParams(max_tokens=args.max_tokens, temperature=0)

    print(f"[profile_serve] Warming up...")
    llm.generate(["Warmup prompt."], params)

    llm.apply_model(install_recorder)
    print(f"[profile_serve] Installed module recorder hooks")

    for i, prompt in enumerate(prompts, 1):
        print(f"[profile_serve] Profiling request {i}/{len(prompts)}...")
        llm.start_profile()
        llm.generate([prompt], params)
        llm.stop_profile()

    llm.apply_model(collect_recorder)

    traces = sorted(trace_dir.glob("*.json")) + sorted(trace_dir.glob("*.json.gz"))
    print(f"[profile_serve] Found {len(traces)} trace files")

    for i, trace_path in enumerate(traces, 1):
        suffix = ".json.gz" if trace_path.name.endswith(".json.gz") else ".json"
        dest = profile_dir / f"request_{i:03d}{suffix}"
        shutil.move(str(trace_path), str(dest))
        sz = dest.stat().st_size / 1e6
        print(f"  {dest.name} ({sz:.1f} MB)")

    shutil.rmtree(trace_dir, ignore_errors=True)

    print(f"\n[profile_serve] Done — {len(traces)} traces in {profile_dir}")
    print(f"[profile_serve] Next: python profiler_to_diagram.py "
          f"--per-request-dir {args.profile_dir} --out_dir samples/")


if __name__ == "__main__":
    main()
