#!/usr/bin/env python3
"""
GRASP stage 7 — measured runtime overhead against real generation, same GPU.

For one generation config:
  1. load the model the way MUCH generates (default attention), time MUCH's own
     `generate_one_question` on N questions — the denominator;
  2. reload with eager attention, time GRASP's teacher-forced extraction (k heads
     per layer, as in the cache) on exactly those generated outputs, plus the CPU
     claim-feature/probe cost — the numerator.

Model loading is excluded from both sides. Writes overhead = numerator / denominator.

    python scripts/11_overhead.py --much-root ../much --config <cfg> \
        --heads-file <cache>/selected_heads.json --n 20 --out overhead_<cfg>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp.features import FeatureConfig, extract_features  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--heads-file", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.much_root))
    from src.generation import get_all_generations, get_tokenizer, template_question  # type: ignore
    from src.generation.utils import GenerationConfig, generate_one_question, get_model  # type: ignore
    from src.questions.utils import get_all_questions  # type: ignore

    gens = [g for g in get_all_generations().values() if g.generation_config == args.config]
    cfg = gens[0].generation_cfg
    gen_cfg = GenerationConfig(seed=cfg.seed, model_name=cfg.model_name,
                               temperature=cfg.temperature, top_p=cfg.top_p,
                               top_k=cfg.top_k, max_new_tokens=cfg.max_new_tokens,
                               brief=cfg.brief)
    questions = get_all_questions()
    rng = np.random.RandomState(42)
    picked = [questions[i] for i in rng.choice(len(questions), args.n + 1, replace=False)]
    device = args.device_map or None
    tok = get_tokenizer(cfg.model_name)

    # ---- denominator: real generation, MUCH's code path, default attention ----
    model = get_model(cfg.model_name, eager_attn=False, **({"device": device} if device else {}))
    model.eval()
    generate_one_question(picked[0], model=model, tokenizer=tok, generation_cfg=gen_cfg, save_disk=False)
    outputs, gen_s = [], []
    for q in picked[1:]:
        sync(); t0 = time.perf_counter()
        g = generate_one_question(q, model=model, tokenizer=tok, generation_cfg=gen_cfg, save_disk=False)
        sync(); gen_s.append(time.perf_counter() - t0)
        outputs.append(g)
    del model
    get_model.cache_clear(); gc.collect(); torch.cuda.empty_cache()

    # ---- numerator: GRASP extraction on the same outputs, eager attention ----
    heads = {int(k): v for k, v in json.loads(args.heads_file.read_text()).items()}
    fcfg = FeatureConfig(keep_heads=heads, align="current", saliency_norm="per_later_token")
    model = get_model(cfg.model_name, eager_attn=True, **({"device": device} if device else {}))
    model.eval()
    p0 = template_question(outputs[0].prompt, tok, brief=cfg.brief)
    extract_features(model, tok, p0, outputs[0].output_tokens, fcfg)  # warm-up
    ext_s, n_tok = [], []
    for g in outputs:
        prompt = template_question(g.prompt, tok, brief=cfg.brief)
        sync(); t0 = time.perf_counter()
        feats = extract_features(model, tok, prompt, g.output_tokens, fcfg)
        sync(); ext_s.append(time.perf_counter() - t0)
        n_tok.append(len(g.output_tokens))
    # CPU side: claim features + a 300-tree probe is ~1 ms/claim; bound it generously
    t0 = time.perf_counter()
    for _ in range(len(outputs)):
        z = (feats[..., :3] - feats[..., :3].mean()) / (feats[..., :3].std() + 1e-9)
        _ = [z[:, :, : max(1, z.shape[2] // 2)].mean(axis=2) for _ in range(8)]
    cpu_s = (time.perf_counter() - t0) / len(outputs)

    res = {
        "config": args.config,
        "model": cfg.model_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_gpus": torch.cuda.device_count(),
        "n": len(outputs),
        "mean_generated_tokens": float(np.mean(n_tok)),
        "generation_s_per_gen": float(np.mean(gen_s)),
        "grasp_extract_s_per_gen": float(np.mean(ext_s)),
        "grasp_cpu_s_per_gen": float(cpu_s),
        "overhead_pct": float(100 * (np.mean(ext_s) + cpu_s) / np.mean(gen_s)),
        "note": "CCP's overhead on MUCH is ~124% (MUCH paper, A100); not re-measured here.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("OVERHEAD", json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
