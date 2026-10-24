#!/usr/bin/env python3
"""
GRASP stage 4 — runtime profiler for GRASP itself.

Measures the execution time of:
1. Teacher-forced forward pass with eager attention hooks (GRASP feature extraction).
2. Post-processing and claim scoring (GRASP CPU scoring).

Generation is not timed here, so this script reports no overhead percentage.
For overhead against real generation on the same GPU, use scripts/11_overhead.py.

Usage:
    python scripts/07_timing.py --much-root /content/much --config <cfg> --n-samples 50 --out results/timing_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp.features import FeatureConfig, extract_features  # noqa: E402
from grasp.score import CorpusNormaliser, claim_scores, token_signal  # noqa: E402


def load_much(much_root: Path):
    sys.path.insert(0, str(much_root))
    from src.generation import (  # type: ignore
        get_all_generations,
        get_model,
        get_tokenizer,
        template_question,
    )

    return get_all_generations, get_model, get_tokenizer, template_question


def benchmark_runtime(
    model,
    tokenizer,
    template_question_fn,
    gens: list,
    n_samples: int = 50,
    sink_size: int = 1,
    selected_heads: dict[int, int] | dict[int, list[int]] | None = None,
) -> dict:
    """Measures precise time for forward pass extraction vs CPU scoring."""
    sample_gens = gens[: min(n_samples, len(gens))]
    print(f"Benchmarking on {len(sample_gens)} sample generations...")

    cfg = FeatureConfig(
        sink_size=sink_size,
        keep_heads=selected_heads,
        align="current",
        saliency_norm="per_later_token",
    )

    # Warmup
    print("Warming up GPU...")
    for g in sample_gens[:3]:
        p = template_question_fn(g.prompt, tokenizer, brief=g.generation_cfg.brief)
        _ = extract_features(model, tokenizer, p, g.output_tokens, cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure Feature Extraction (GPU)
    print("Measuring GPU feature extraction...")
    extract_times = []
    extracted_feats = []
    total_tokens = 0

    t_start = time.perf_counter()
    for g in sample_gens:
        p = template_question_fn(g.prompt, tokenizer, brief=g.generation_cfg.brief)
        t0 = time.perf_counter()
        f = extract_features(model, tokenizer, p, g.output_tokens, cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        extract_times.append(dt)
        extracted_feats.append(f)
        total_tokens += len(g.output_tokens)

    total_extract_time = time.perf_counter() - t_start

    # Fit quick normaliser
    norm = CorpusNormaliser().fit(extracted_feats)

    # Measure Scoring (CPU)
    print("Measuring CPU claim scoring...")
    scoring_times = []
    t_start_cpu = time.perf_counter()
    for f, g in zip(extracted_feats, sample_gens):
        t0 = time.perf_counter()
        ts = token_signal(f, norm, norm_mode="corpus")
        chunks = [list(range(0, min(8, f.shape[2])))]
        _ = claim_scores(ts, f, chunks, tau=8.0)
        scoring_times.append(time.perf_counter() - t0)
    total_score_time = time.perf_counter() - t_start_cpu

    # Synthesis
    n = len(sample_gens)
    avg_extract_per_gen = total_extract_time / n
    avg_score_per_gen = total_score_time / n
    avg_total_grasp = avg_extract_per_gen + avg_score_per_gen
    avg_tokens_per_gen = total_tokens / n

    return {
        "n_samples": n,
        "total_tokens": total_tokens,
        "avg_tokens_per_gen": avg_tokens_per_gen,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_extract_seconds_total": total_extract_time,
        "gpu_extract_seconds_per_gen": avg_extract_per_gen,
        "cpu_score_seconds_total": total_score_time,
        "cpu_score_seconds_per_gen": avg_score_per_gen,
        "total_grasp_latency_per_gen": avg_total_grasp,
        "grasp_seconds_per_token": avg_total_grasp / max(avg_tokens_per_gen, 1e-6),
        "throughput_gens_per_sec": 1.0 / max(avg_total_grasp, 1e-6),
        "throughput_tokens_per_sec": total_tokens / max(total_extract_time, 1e-6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--sink-size", type=int, default=1)
    ap.add_argument("--heads-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/timing_profile.json"))
    args = ap.parse_args()

    get_all_generations, get_model, get_tokenizer, template_question = load_much(
        args.much_root
    )
    gens = list(get_all_generations().values())
    if args.config:
        gens = [g for g in gens if g.generation_config == args.config]
    if not gens:
        print("No generations found.")
        return 1

    model_name = gens[0].generation_cfg.model_name
    print(f"Loading model {model_name} for timing benchmark...")

    model = get_model(model_name, eager_attn=True)
    model.eval()
    tokenizer = get_tokenizer(model_name)

    selected_heads = None
    if args.heads_file and args.heads_file.is_file():
        data = json.loads(args.heads_file.read_text(encoding="utf-8"))
        # 1-head files map layer -> head; top-k files map layer -> [heads]
        selected_heads = {
            int(k): [int(h) for h in v] if isinstance(v, list) else int(v)
            for k, v in data.items()
        }

    profile = benchmark_runtime(
        model,
        tokenizer,
        template_question,
        gens,
        n_samples=args.n_samples,
        sink_size=args.sink_size,
        selected_heads=selected_heads,
    )

    print("\n" + "=" * 60)
    print("TIMING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Device: {profile['device']}")
    print(f"Extraction (GPU) per gen:  {profile['gpu_extract_seconds_per_gen']*1000:.2f} ms")
    print(f"Scoring (CPU) per gen:     {profile['cpu_score_seconds_per_gen']*1000:.2f} ms")
    print(f"Total GRASP per gen:       {profile['total_grasp_latency_per_gen']*1000:.2f} ms")
    print(f"Throughput:                {profile['throughput_tokens_per_sec']:.1f} tokens/s")
    print("Overhead vs generation:    not measured here; run scripts/11_overhead.py")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"\nWrote profile to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
