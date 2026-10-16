#!/usr/bin/env python3
"""Merge benchmark JSON + Prometheus dump + nsys sqlite into workload_data.json.

Usage:
  python parse_workload.py \
    --bench benchmark_outputs/qwen3_8b_random_2req.json \
    --prometheus benchmark_outputs/prometheus_2req.txt \
    --nsys samples/traces/vllm_2req_nvtx.nsys-rep.sqlite \
    --dataset-name random \
    -o samples/workload_data.json

All inputs except --bench are optional.
"""
import argparse
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path


# ── Benchmark JSON ──────────────────────────────────────────────────────────

def load_bench(path):
    with open(path) as f:
        return json.load(f)


def build_requests_from_bench(bench):
    n = bench.get("num_prompts", 0)
    start_times = bench.get("start_times", [])
    t0 = start_times[0] if start_times else 0
    requests = []
    for i in range(n):
        ttft_s = bench["ttfts"][i] if i < len(bench.get("ttfts", [])) else 0
        latency_s = bench["latencies"][i] if i < len(bench.get("latencies", [])) else 0
        queue_s = bench["queue_times"][i] if i < len(bench.get("queue_times", [])) else 0
        start_ms = round((start_times[i] - t0) * 1000, 3) if i < len(start_times) else 0
        ttft_ms = ttft_s * 1000
        latency_ms = latency_s * 1000
        queue_ms = queue_s * 1000
        decode_ms = max(0, latency_ms - ttft_ms)

        itl_list = bench.get("itls", [])[i] if i < len(bench.get("itls", [])) else []
        itl_ms_list = [v * 1000 for v in itl_list]
        itl_median_ms = round(statistics.median(itl_ms_list), 3) if itl_ms_list else 0
        itl_p99_ms = round(_percentile(itl_ms_list, 99), 3) if itl_ms_list else 0

        requests.append({
            "id": None,
            "label": f"Req-{i + 1:03d}",
            "start_ms": start_ms,
            "input_tokens": bench["input_lens"][i] if i < len(bench.get("input_lens", [])) else 0,
            "output_tokens": bench["output_lens"][i] if i < len(bench.get("output_lens", [])) else 0,
            "ttft_ms": round(ttft_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "queue_ms": round(queue_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "itl_ms": [round(v, 3) for v in itl_ms_list],
            "itl_median_ms": itl_median_ms,
            "itl_p99_ms": itl_p99_ms,
        })
    return requests


def build_aggregate(bench):
    keys = [
        "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "std_itl_ms", "p99_itl_ms",
        "output_throughput", "total_token_throughput",
        "request_throughput", "duration",
    ]
    agg = {}
    for k in keys:
        if k in bench:
            agg[k] = round(bench[k], 4) if isinstance(bench[k], float) else bench[k]
    return agg


def build_gaps_from_bench(bench):
    starts = bench.get("start_times", [])
    latencies = bench.get("latencies", [])
    gaps = []
    for i in range(len(starts) - 1):
        end_i = starts[i] + latencies[i] if i < len(latencies) else starts[i]
        gap_s = starts[i + 1] - end_i
        gaps.append({
            "after": f"Req-{i + 1:03d}",
            "before": f"Req-{i + 2:03d}",
            "gap_ms": round(gap_s * 1000, 3),
            "source": "bench_client",
        })
    return gaps


# ── Prometheus ──────────────────────────────────────────────────────────────

_HISTOGRAM_METRICS = {
    "vllm:request_prefill_time_seconds": "prefill_time",
    "vllm:request_decode_time_seconds": "decode_time",
    "vllm:request_queue_time_seconds": "queue_time",
    "vllm:time_to_first_token_seconds": "ttft",
    "vllm:inter_token_latency_seconds": "itl",
    "vllm:e2e_request_latency_seconds": "e2e_latency",
    "vllm:request_inference_time_seconds": "inference_time",
}

_GAUGE_METRICS = {
    "vllm:kv_cache_usage_perc": "kv_cache_usage_pct",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    "vllm:num_preemptions_total": "num_preemptions",
}


def parse_prometheus(path):
    text = Path(path).read_text()
    result = {}

    for base_name, key in _HISTOGRAM_METRICS.items():
        buckets = []
        count = None
        sum_val = None
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(f"{base_name}_bucket{{"):
                m = re.search(r'le="([^"]+)"', line)
                val = line.rsplit(" ", 1)[-1]
                if m:
                    buckets.append({"le": m.group(1), "count": float(val)})
            elif line.startswith(f"{base_name}_count{{"):
                count = float(line.rsplit(" ", 1)[-1])
            elif line.startswith(f"{base_name}_sum{{"):
                sum_val = float(line.rsplit(" ", 1)[-1])
        if count is not None:
            result[key] = {
                "count": int(count),
                "sum_s": round(sum_val, 6) if sum_val is not None else 0,
                "buckets": buckets,
            }

    for metric_name, key in _GAUGE_METRICS.items():
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(f"{metric_name}{{") or line.startswith(f"{metric_name} "):
                val = float(line.rsplit(" ", 1)[-1])
                result[key] = val
                break

    return result


# ── nsys sqlite ─────────────────────────────────────────────────────────────

def load_nsys(path, num_layers=0):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    gpu_info = _load_gpu_info(conn)
    nvtx_ranges = _load_nvtx_requests(conn)
    for rng in nvtx_ranges:
        rng["kernels"] = _load_kernels_for_range(conn, rng["start"], rng["end"], gpu_info, num_layers)
    conn.close()
    return gpu_info, nvtx_ranges


def _load_gpu_info(conn):
    defaults = {
        "name": "Unknown GPU",
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "max_regs_per_sm": 65536,
        "max_shared_per_sm": 232448,
        "max_blocks_per_sm": 32,
        "max_warps_per_sm": 64,
        "warp_size": 32,
    }
    try:
        row = conn.execute("SELECT * FROM TARGET_INFO_GPU LIMIT 1").fetchone()
        if row:
            sm = row["smCount"] or defaults["sm_count"]
            warp_size = row["threadsPerWarp"] or defaults["warp_size"]
            max_threads = (row["maxWarpsPerSm"] or defaults["max_warps_per_sm"]) * warp_size
            return {
                "name": row["name"] or defaults["name"],
                "sm_count": sm,
                "max_threads_per_sm": max_threads,
                "max_regs_per_sm": row["maxRegistersPerSm"] or defaults["max_regs_per_sm"],
                "max_shared_per_sm": row["maxShmemPerSm"] or defaults["max_shared_per_sm"],
                "max_blocks_per_sm": row["maxBlocksPerSm"] or defaults["max_blocks_per_sm"],
                "max_warps_per_sm": row["maxWarpsPerSm"] or defaults["max_warps_per_sm"],
                "warp_size": warp_size,
            }
    except Exception:
        pass
    return defaults


def _load_nvtx_requests(conn):
    rows = conn.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE text LIKE 'request_%' AND end IS NOT NULL "
        "ORDER BY start"
    ).fetchall()
    ranges = []
    for i, row in enumerate(rows):
        req_text = row["text"]
        req_id = req_text.replace("request_", "", 1) if req_text.startswith("request_") else req_text
        ranges.append({
            "id": req_id,
            "label": f"Req-{i + 1:03d}",
            "start": row["start"],
            "end": row["end"],
            "dur_ms": round((row["end"] - row["start"]) / 1e6, 3),
        })
    return ranges


def _load_kernels_for_range(conn, range_start, range_end, gpu_info, num_layers=0):
    try:
        rows = conn.execute(
            """
            SELECT
                k.start, k.end,
                COALESCE(s.value, 'unknown') AS kernelName,
                k.registersPerThread,
                k.gridX, k.gridY, k.gridZ,
                k.blockX, k.blockY, k.blockZ,
                k.staticSharedMemory, k.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.demangledName = s.id
            WHERE k.start >= ? AND k.end <= ?
            ORDER BY k.start
            """,
            (range_start, range_end),
        ).fetchall()
    except Exception:
        return {"kernel_count": 0, "kernel_time_ms": 0, "top_kernels": []}

    by_name = {}
    total_dur_ns = 0
    kernel_durs = []
    for row in rows:
        name = row["kernelName"]
        dur = row["end"] - row["start"]
        total_dur_ns += dur
        kernel_durs.append(dur)
        if name not in by_name:
            by_name[name] = {
                "name": name,
                "count": 0,
                "total_dur_ns": 0,
                "grid": [row["gridX"], row["gridY"], row["gridZ"]],
                "block": [row["blockX"], row["blockY"], row["blockZ"]],
                "regs": row["registersPerThread"] or 0,
                "shared_mem": (row["staticSharedMemory"] or 0) + (row["dynamicSharedMemory"] or 0),
            }
        by_name[name]["count"] += 1
        by_name[name]["total_dur_ns"] += dur

    top = sorted(by_name.values(), key=lambda k: k["total_dur_ns"], reverse=True)[:20]
    for k in top:
        k["avg_dur_ns"] = round(k["total_dur_ns"] / k["count"])
        k["pct"] = round(100 * k["total_dur_ns"] / total_dur_ns, 1) if total_dur_ns else 0
        k["occupancy"] = _calc_occupancy(
            k["block"], k["regs"], k["shared_mem"], gpu_info
        )

    result = {
        "kernel_count": len(rows),
        "kernel_time_ms": round(total_dur_ns / 1e6, 3),
        "top_kernels": top,
    }

    if num_layers > 0 and kernel_durs:
        chunk = len(kernel_durs) // num_layers
        if chunk > 0:
            per_layer = []
            for i in range(num_layers):
                start = i * chunk
                end = start + chunk if i < num_layers - 1 else len(kernel_durs)
                layer_ns = sum(kernel_durs[start:end])
                per_layer.append(round(layer_ns))
            result["per_layer_gpu_ns"] = per_layer

    return result


def _calc_occupancy(block, regs, smem, gpu):
    threads = block[0] * block[1] * block[2]
    if not threads:
        return 0
    warp_size = gpu["warp_size"]
    warps = -(-threads // warp_size)  # ceil division
    max_warps = gpu["max_warps_per_sm"]
    max_blocks = gpu["max_blocks_per_sm"]

    blk_by_threads = gpu["max_threads_per_sm"] // threads if threads else 0
    blk_by_regs = (
        gpu["max_regs_per_sm"] // (regs * threads)
        if regs else max_blocks
    )
    blk_by_smem = (
        gpu["max_shared_per_sm"] // smem
        if smem else max_blocks
    )

    active_blocks = min(blk_by_threads, blk_by_regs, blk_by_smem, max_blocks)
    active_blocks = max(active_blocks, 0)
    active_warps = active_blocks * warps
    return round(100 * active_warps / max_warps, 1) if max_warps else 0


# ── Auto-detect layers ─────────────────────────────────────────────────────

def detect_num_layers(model_id):
    if not model_id or model_id == "unknown":
        return 0
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        n = getattr(config, "num_hidden_layers", 0)
        if n:
            print(f"Auto-detected {n} layers from {model_id}")
        return n
    except Exception:
        return 0


# ── Merge ───────────────────────────────────────────────────────────────────

def merge(bench_path, prom_path, nsys_path, dataset_name, output_path, num_layers=0):
    bench = load_bench(bench_path) if bench_path else {}
    requests = build_requests_from_bench(bench) if bench else []

    if num_layers == 0 and bench:
        num_layers = detect_num_layers(bench.get("model_id", ""))

    meta = {
        "model_id": bench.get("model_id", "unknown"),
        "dataset": dataset_name or bench.get("dataset_name", "unknown"),
        "num_prompts": bench.get("num_prompts", len(requests)),
        "request_rate": bench.get("request_rate"),
        "date": bench.get("date"),
    }

    summary = build_aggregate(bench) if bench else {}
    gaps = build_gaps_from_bench(bench) if bench else []

    gpu = {"name": "Unknown GPU", "sm_count": 0}
    try:
        import torch
        if torch.cuda.is_available():
            gpu["name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if nsys_path:
        gpu_info, nvtx_ranges = load_nsys(nsys_path, num_layers)
        gpu = {"name": gpu_info["name"], "sm_count": gpu_info["sm_count"]}

        for i, req in enumerate(requests):
            if i < len(nvtx_ranges):
                rng = nvtx_ranges[i]
                req["id"] = rng["id"]
                req["nvtx_start_ns"] = rng["start"]
                req["nvtx_end_ns"] = rng["end"]
                req["nvtx_dur_ms"] = rng["dur_ms"]
                req["kernel_count"] = rng["kernels"]["kernel_count"]
                req["kernel_time_ms"] = rng["kernels"]["kernel_time_ms"]
                req["top_kernels"] = rng["kernels"]["top_kernels"]
                if "per_layer_gpu_ns" in rng["kernels"]:
                    req["per_layer_gpu_ns"] = rng["kernels"]["per_layer_gpu_ns"]

        # NVTX-based gaps (more precise than client-side)
        if len(nvtx_ranges) > 1:
            gaps = []
            for i in range(len(nvtx_ranges) - 1):
                gap_ns = nvtx_ranges[i + 1]["start"] - nvtx_ranges[i]["end"]
                gaps.append({
                    "after": nvtx_ranges[i]["label"],
                    "before": nvtx_ranges[i + 1]["label"],
                    "gap_ms": round(gap_ns / 1e6, 3),
                    "source": "nvtx",
                })

    prom = {}
    if prom_path:
        prom = parse_prometheus(prom_path)

    prom_flat = {}
    if prom:
        if "prefill_time" in prom:
            prom_flat["prefill_sum_s"] = prom["prefill_time"]["sum_s"]
        if "decode_time" in prom:
            prom_flat["decode_sum_s"] = prom["decode_time"]["sum_s"]
        if "queue_time" in prom:
            prom_flat["queue_sum_s"] = prom["queue_time"]["sum_s"]
        if "kv_cache_usage_pct" in prom:
            prom_flat["kv_cache_usage_pct"] = prom["kv_cache_usage_pct"]
        if "prefix_cache_hits" in prom:
            prom_flat["prefix_cache_hits"] = prom["prefix_cache_hits"]
        if "num_preemptions" in prom:
            prom_flat["num_preemptions"] = prom["num_preemptions"]
        prom_flat["_raw"] = prom

    output = {
        "meta": meta,
        "gpu": gpu,
        "requests": requests,
        "gaps": gaps,
        "prometheus": prom_flat if prom_flat else None,
        "summary": summary,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _percentile(data, pct):
    if not data:
        return 0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Merge benchmark + Prometheus + nsys into workload_data.json")
    p.add_argument("--bench", required=True, help="Benchmark JSON from vllm bench serve --save-detailed")
    p.add_argument("--prometheus", default=None, help="Prometheus metrics dump (text)")
    p.add_argument("--nsys", default=None, help="nsys sqlite export")
    p.add_argument("--dataset-name", default=None, help="Dataset name (random, sharegpt, etc.)")
    p.add_argument("--num-layers", type=int, default=0, help="Number of decoder layers (for per-layer GPU timing split)")
    p.add_argument("-o", "--output", default="samples/workload_data.json", help="Output path")
    args = p.parse_args()
    merge(args.bench, args.prometheus, args.nsys, args.dataset_name, args.output, args.num_layers)


if __name__ == "__main__":
    main()
