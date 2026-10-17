#!/usr/bin/env python3
"""
GRASP stage 1 — the only step that needs a GPU.

For every MUCH generation, replay the stored token ids through the model that
produced them and cache the four per-token attention statistics.

Two passes:

  calibration   a small unlabelled subset, ALL heads kept, used to choose one
                head per layer by grounding variance and to fit the corpus
                normaliser
  full          every generation, only the selected head per layer kept

Nothing here reads a hallucination label, which is what keeps the method
training-free.

Usage
-----
    python scripts/02_extract_attention.py \
        --much-root /content/much \
        --config <generation_config_id> \
        --out output/grasp_cache \
        --calib 200

Run once per generation config (model x temperature).
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
from grasp.heads import select_heads, select_heads_variance  # noqa: E402
from grasp.score import CorpusNormaliser  # noqa: E402


def load_much(much_root: Path):
    """Import the MUCH package so we reuse its exact prompt templating."""
    sys.path.insert(0, str(much_root))
    from src.generation import (  # type: ignore
        get_all_generations,
        get_model,
        get_tokenizer,
        template_question,
    )

    return get_all_generations, get_model, get_tokenizer, template_question


def sample_calib_gens(gens: list, n_calib: int, balanced: bool = False, seed: int = 42) -> list:
    """Sample calibration generations, optionally balancing across languages."""
    if len(gens) <= n_calib:
        return gens
    if not balanced:
        return gens[:n_calib]

    # Stratified sampling across languages
    by_lang: dict[str, list] = {}
    for g in gens:
        by_lang.setdefault(getattr(g, "lang", "default"), []).append(g)

    rng = np.random.RandomState(seed)
    sampled = []
    per_lang = max(1, n_calib // len(by_lang))
    for lang, items in by_lang.items():
        perm = rng.permutation(len(items))
        take = [items[i] for i in perm[:per_lang]]
        sampled.extend(take)

    if len(sampled) < n_calib:
        remaining = [g for g in gens if g not in sampled]
        perm = rng.permutation(len(remaining))
        sampled.extend([remaining[i] for i in perm[: (n_calib - len(sampled))]])

    return sampled[:n_calib]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--config", type=str, default=None,
                    help="generation_config id; default = every config found")
    ap.add_argument("--out", type=Path, default=Path("output/grasp_cache"))
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--calib-balanced", action="store_true",
                    help="Stratify calibration samples evenly across languages")
    ap.add_argument("--head-select", choices=["variance", "variance_topk", "random", "fixed", "all_heads"],
                    default="variance", help="Attention head selection strategy")
    ap.add_argument("--top-k", type=int, default=4,
                    help="heads kept per layer with --head-select variance_topk")
    ap.add_argument("--align", choices=["legacy", "current"], default="legacy",
                    help="Attention query alignment ('current' fixes off-by-one)")
    ap.add_argument("--saliency-norm", choices=["none", "per_later_token"], default="none",
                    help="Saliency normalisation mode")
    ap.add_argument("--profile", choices=["legacy", "corrected"], default="legacy",
                    help="Configuration preset ('corrected' turns on align=current, saliency-norm=per_later_token, calib-balanced)")
    ap.add_argument("--sink-size", type=int, default=1)
    ap.add_argument("--device-map", default=None,
                    help="e.g. 'auto' to shard a model over all GPUs (8B on 2x T4)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (debug aid)")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    if args.profile == "corrected":
        args.align = "current"
        args.saliency_norm = "per_later_token"
        args.calib_balanced = True

    get_all_generations, get_model, get_tokenizer, template_question = load_much(
        args.much_root
    )

    gens = list(get_all_generations().values())
    if args.config:
        gens = [g for g in gens if g.generation_config == args.config]
    if not gens:
        print("No generations found. Did you run `python -m src import-much`?")
        return 1
    if args.limit:
        gens = gens[: args.limit]

    model_name = gens[0].generation_cfg.model_name
    print(f"{len(gens)} generations | model {model_name} | profile={args.profile} (align={args.align}, head-select={args.head_select})")

    # eager attention is mandatory: fused kernels never materialise the weights
    try:
        if args.device_map:
            model = get_model(model_name, eager_attn=True, device=args.device_map)
        else:
            model = get_model(model_name, eager_attn=True)
        model.eval()
        tokenizer = get_tokenizer(model_name)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "gated" in msg.lower() or "403" in msg:
            print(
                f"\n{model_name} is GATED and your account is not authorised.\n"
                f"  1. open https://huggingface.co/{model_name} and accept the licence\n"
                f"  2. wait for approval (usually minutes; Gemma can take longer)\n"
                f"  3. huggingface_hub.login() with a token that has read access\n"
                f"Pick a different --config in the meantime; the CCP checkpoint\n"
                f"needs no model at all."
            )
            return 2
        raise

    feat_dir = args.out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Record how features are extracted, so scoring never re-applies a
    # normalisation or alignment fix the cache already carries. A resumed run
    # must match, or one cache would mix alignments.
    meta = {
        "model": model_name,
        "profile": args.profile,
        "align": args.align,
        "saliency_norm": args.saliency_norm,
        "sink_size": args.sink_size,
        "head_select": args.head_select,
    }
    meta_path = args.out / "cache_meta.json"
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        clash = {
            k: (old.get(k), v)
            for k, v in meta.items()
            if k in ("align", "saliency_norm", "sink_size") and old.get(k) != v
        }
        if clash:
            print(f"cache {args.out} was extracted with different settings: {clash}")
            print("use a fresh --out directory instead of resuming into this one.")
            return 1
    elif any(feat_dir.glob("*.npy")):
        print(
            f"warning: {feat_dir} has features but no cache_meta.json (pre-metadata "
            "cache, align=legacy, saliency_norm=none). Resuming would mix settings "
            "unless this run uses the same ones."
        )
        if args.align != "legacy" or args.saliency_norm != "none":
            print("refusing to resume; use a fresh --out directory.")
            return 1
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def features_for(g, cfg):
        prompt = template_question(g.prompt, tokenizer, brief=g.generation_cfg.brief)
        return extract_features(model, tokenizer, prompt, g.output_tokens, cfg)

    # ---------------- pass 1: calibration, all heads ----------------
    calib_gens = sample_calib_gens(gens, args.calib, balanced=args.calib_balanced)
    print(f"calibration pass over {len(calib_gens)} generations (all heads)")
    all_head_cfg = FeatureConfig(
        sink_size=args.sink_size,
        keep_heads=None,
        align=args.align,
        saliency_norm=args.saliency_norm,
    )

    calib_feats = []
    t0 = time.time()
    for i, g in enumerate(calib_gens):
        try:
            calib_feats.append(features_for(g, all_head_cfg))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {g.id}: {exc}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(calib_gens)}  ({time.time() - t0:.0f}s)")

    if not calib_feats:
        print("Calibration produced nothing — check attn_implementation='eager'.")
        return 1

    if args.head_select == "all_heads":
        selected = None
        print("keeping ALL heads (no reduction)")
    else:
        selected = select_heads(
            args.head_select,
            calib_features=calib_feats,
            k=args.top_k,
        )
        print(f"selected heads ({args.head_select}): {selected}")

    (args.out).mkdir(parents=True, exist_ok=True)
    if selected is not None:
        (args.out / "selected_heads.json").write_text(
            json.dumps({str(k): v for k, v in selected.items()}, indent=2)
        )

    # ---------------- pass 2: full corpus, selected head per layer ----------------
    one_head_cfg = FeatureConfig(
        sink_size=args.sink_size,
        keep_heads=selected,
        align=args.align,
        saliency_norm=args.saliency_norm,
    )
    print(f"full pass over {len(gens)} generations ({'all heads' if selected is None else '1 head/layer'})")

    kept = []
    t0 = time.time()
    for i, g in enumerate(gens):
        path = feat_dir / f"{g.id}.npy"
        if path.is_file():
            if len(kept) < args.calib:
                kept.append(np.load(path))
            continue
        try:
            f = features_for(g, one_head_cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {g.id}: {exc}")
            continue
        np.save(path, f)
        if len(kept) < args.calib:
            kept.append(f)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(gens)}  ({el:.0f}s, {el / (i + 1):.3f}s/gen)")

    total = time.time() - t0
    print(f"extraction wall-clock: {total:.1f}s for {len(gens)} generations")
    (args.out / "timing.json").write_text(
        json.dumps(
            {
                "n_generations": len(gens),
                "seconds": total,
                "seconds_per_generation": total / max(len(gens), 1),
                "model": model_name,
                "profile": args.profile,
                "align": args.align,
                "saliency_norm": args.saliency_norm,
                "head_select": args.head_select,
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu",
            },
            indent=2,
        )
    )

    # ---------------- corpus normaliser (unlabelled) ----------------
    norm = CorpusNormaliser().fit(kept)
    np.savez(args.out / "normaliser.npz", mean=norm.mean_, std=norm.std_)
    print(f"wrote cache to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
