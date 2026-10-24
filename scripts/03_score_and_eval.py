#!/usr/bin/env python3
"""
GRASP stage 2 — CPU only, seconds per run.

Reads the cached attention statistics, scores every claim, and evaluates under
MUCH's exact protocol (see `grasp.much_eval`: label polarity, dropped EOS chunk,
stopword masking, annotator consensus).

Two scoring routes are reported, deliberately:

  grasp/much-agg   GRASP's per-token signal fed through MUCH's own aggregators
                   (mean / max / geometric / product). This is the apples-to-
                   apples row: identical protocol to every published baseline.
  grasp/native     GRASP's saliency-weighted soft-min claim aggregator, the
                   method's own contribution.

Reporting only the second would invite the objection that the gain came from a
different aggregation rule rather than from attention. Reporting both settles it.

All hyperparameters are swept on the TRAINING split. Score the 200 human-annotated
test samples once, with the frozen configuration, and say so in the paper.

Usage
-----
    python scripts/03_score_and_eval.py --much-root /content/much \
        --cache results/cache_<cfg> --lang en --sweep
    python scripts/03_score_and_eval.py ... --ablate --fuse-with ccp_24_8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp.much_eval import (  # noqa: E402
    AGGREGATORS,
    HALLUCINATION_LABEL,
    aggregate,
    evaluate,
    has_consensus,
    mask_stopwords,
    metrics,
)
from grasp.score import (  # noqa: E402
    CorpusNormaliser,
    apply_boundary_reset,
    claim_scores,
    resolve_scoring_options,
    token_signal,
)


def load_much(much_root: Path):
    sys.path.insert(0, str(much_root))
    from src.annotation import get_all_annotations  # type: ignore
    from src.generation import get_tokenizer  # type: ignore

    return get_all_annotations, get_tokenizer


def load_normaliser(cache: Path) -> CorpusNormaliser:
    d = np.load(cache / "normaliser.npz")
    n = CorpusNormaliser()
    n.mean_, n.std_ = d["mean"], d["std"]
    return n


def load_cache_meta(cache: Path) -> dict:
    """Extraction settings for a cache. cache_meta.json is written by
    02_extract_attention.py; older caches only have timing.json, and caches older
    than that were always align='legacy', saliency_norm='none' (empty dict)."""
    for name in ("cache_meta.json", "timing.json"):
        path = cache / name
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
            if "saliency_norm" in meta or "align" in meta:
                return meta
    print("  (no extraction metadata in cache; assuming align=legacy, saliency_norm=none)")
    return {}


def rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


AUTO_ANNOTATOR_PREFIX = "gpt-"


def get_split(annot) -> str:
    """'test' for MUCH's 200 human-annotated samples, 'train' for the rest.

    MUCH's Annotation has no split field. GPT labels are stored under
    `gpt-<model>` keys and human labels under the annotator's name (e.g. `an0`,
    via `python -m src add-human-annotations`), so the test set is exactly the
    annotations carrying at least one non-`gpt-` label key. Test samples also
    carry GPT labels, and are scored on them like every other sample, matching
    MUCH's own evaluator.
    """
    labels = getattr(annot, "labels", None) or {}
    if any(not str(k).startswith(AUTO_ANNOTATOR_PREFIX) for k in labels):
        return "test"
    return "train"


def grasp_token_signal(
    feats,
    gen,
    norm,
    *,
    use_signals=(True, True, True),
    reset_mode="reset",
    alpha=0.6,
    chunks=None,
    tokenizer_for=None,
    norm_mode="per_gen_rank",
    align_shift=0,
):
    """GRASP's per-token hallucination signal, stopword-masked like every baseline."""
    ts = token_signal(
        feats,
        norm,
        use_signals=use_signals,
        norm_mode=norm_mode,
        align_shift=align_shift,
    )
    n_tok = len(gen.output_tokens)
    if ts.shape[0] < n_tok:
        ts = np.pad(ts, (0, n_tok - ts.shape[0]), constant_values=0.5)
    ts = ts[:n_tok]
    if reset_mode != "none":
        ts = apply_boundary_reset(ts, chunks or [], alpha=alpha, reset_mode=reset_mode)
    if tokenizer_for is not None:
        ts = mask_stopwords(
            ts, gen.output_tokens, tokenizer_for(gen.generation_cfg.model_name)
        )
    return ts


def score_corpus(
    annotations,
    cache: Path,
    norm,
    tokenizer_for,
    *,
    lang=None,
    model=None,
    config=None,
    split="all",
    tau=8.0,
    alpha=0.6,
    reset_mode="reset",
    use_saliency=True,
    use_signals=(True, True, True),
    norm_mode="per_gen_rank",
    align_shift=0,
    saliency_norm="none",
    claim_agg="softmin",
):
    """Returns (labels, native_scores, {agg: scores}, counts, claim_records)."""
    feat_dir = cache / "features"
    ys: list[int] = []
    native: list[float] = []
    by_agg: dict[str, list[float]] = {a: [] for a in AGGREGATORS}
    records: list[dict] = []
    n_annot = n_missing = 0

    for annot in annotations:
        gen = annot.generation
        if lang and gen.lang != lang:
            continue
        gen_model = getattr(getattr(gen, "generation_cfg", None), "model_name", None)
        if model and gen_model != model:
            continue
        gen_cfg = getattr(gen, "generation_config", None)
        if config and gen_cfg != config:
            continue

        item_split = get_split(annot)
        if split != "all" and item_split != split:
            continue

        if not has_consensus(annot.labels):
            continue
        path = feat_dir / f"{gen.id}.npy"
        if not path.is_file():
            n_missing += 1
            continue

        feats = np.load(path)
        chunks = [list(c) for c in annot.token_chunks]
        ts = grasp_token_signal(
            feats,
            gen,
            norm,
            use_signals=use_signals,
            reset_mode=reset_mode,
            alpha=alpha,
            chunks=chunks,
            tokenizer_for=tokenizer_for,
            norm_mode=norm_mode,
            align_shift=align_shift,
        )
        cs = claim_scores(
            ts,
            feats,
            chunks,
            tau=tau,
            use_saliency=use_saliency,
            saliency_norm=saliency_norm,
            claim_agg=claim_agg,
        )
        n_annot += 1

        # MUCH protocol: drop the trailing EOS chunk, -1 is the positive class
        labels = annot.labels["gpt-4o"]
        for i, (chunk, label) in enumerate(zip(chunks[:-1], labels[:-1])):
            start, stop = chunk[0], chunk[-1] + 1
            piece = ts[start:stop]
            if piece.size == 0:
                continue
            y_val = 1 if label == HALLUCINATION_LABEL else 0
            n_val = float(cs[i]) if i < len(cs) else 0.5
            ys.append(y_val)
            native.append(n_val)

            rec = {
                "claim_id": f"{gen.id}_chunk_{i}",
                "gen_id": gen.id,
                "lang": gen.lang,
                "model": gen_model,
                "split": item_split,
                "chunk_index": i,
                "chunk_len": len(chunk),
                "label": y_val,
                "native_score": n_val,
            }
            for agg in AGGREGATORS:
                val = aggregate(piece, agg)
                by_agg[agg].append(val)
                rec[f"much_{agg}"] = val
            records.append(rec)

    if n_missing:
        print(f"  ({n_missing} generations had no cached features)")
    return (
        np.array(ys),
        np.array(native),
        {a: np.array(v) for a, v in by_agg.items()},
        {"annotations": n_annot, "chunks": len(ys), "missing": n_missing},
        records,
    )


def summarise(y, native, by_agg) -> dict:
    out = {"native": metrics(y, native)}
    for agg in AGGREGATORS:
        out[f"much_agg/{agg}"] = metrics(y, by_agg[agg])
    scored = {k: v for k, v in out.items() if "roc_auc" in v}
    if scored:
        out["best"] = max(scored, key=lambda k: scored[k]["roc_auc"])
    return out


def line(tag: str, m: dict) -> str:
    if "roc_auc" not in m:
        return f"  {tag:<22} (single class)"
    return (
        f"  {tag:<22} ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  "
        f"TPR@10={m['tpr_at_10fpr']:.3f}"
    )


def export_csv(records: list[dict], path: Path):
    import csv

    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"exported {len(records)} claim records to {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("output/grasp_cache"))
    ap.add_argument("--lang", default=None)
    ap.add_argument("--model", default=None, help="Filter by model name")
    ap.add_argument("--config", default=None, help="Filter by generation_config")
    ap.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Evaluation split (train/test grouped by generation)",
    )
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument(
        "--norm-mode",
        choices=["per_gen_rank", "corpus"],
        default="per_gen_rank",
        help="Token risk normalisation mode",
    )
    ap.add_argument(
        "--reset-mode",
        choices=["reset", "none", "cross_claim_ema"],
        default="reset",
        help="Boundary recurrence mode",
    )
    ap.add_argument(
        "--align-shift",
        type=int,
        default=0,
        help="Legacy caches only: shift features forward (token i <- index i-1) to fix the off-by-one",
    )
    ap.add_argument(
        "--saliency-norm",
        choices=["none", "per_later_token"],
        default="none",
        help="Saliency length normalisation",
    )
    ap.add_argument(
        "--profile",
        choices=["legacy", "corrected"],
        default="legacy",
        help="Profile preset ('corrected' enables corpus norm, reset, and frozen fusion)",
    )
    ap.add_argument("--sweep", action="store_true", help="Sweep tau/alpha on TRAIN split only")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--fuse-with", default=None)
    ap.add_argument(
        "--fuse-agg",
        choices=["mean", "max", "geometric", "product"],
        default=None,
        help="Frozen baseline aggregator for fusion (if None: legacy argmax or 'max')",
    )
    ap.add_argument(
        "--no-stopword-mask",
        action="store_true",
        help="skip MUCH's stopword masking. Not comparable with the published table — debugging only.",
    )
    ap.add_argument("--export-csv", type=Path, default=None, help="Export per-claim CSV table")
    ap.add_argument("--out", type=Path, default=Path("results/grasp_eval.json"))
    args = ap.parse_args()

    if args.profile == "corrected":
        args.norm_mode = "corpus"
        args.reset_mode = "reset"
        args.saliency_norm = "per_later_token"
        if args.fuse_agg is None:
            args.fuse_agg = "max"

    get_all_annotations, get_tokenizer = load_much(args.much_root)
    annotations = list(get_all_annotations().values())
    print(f"{len(annotations)} annotations loaded")
    norm = load_normaliser(args.cache)
    cache_meta = load_cache_meta(args.cache)
    try:
        args.saliency_norm, args.align_shift, notes = resolve_scoring_options(
            cache_meta, args.saliency_norm, args.align_shift
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    for note in notes:
        print(f"  note: {note}")

    n_test = sum(get_split(a) == "test" for a in annotations)
    print(f"  split: {n_test} human-annotated test / {len(annotations) - n_test} train")
    if args.split == "test" and n_test == 0:
        print("error: no human-annotated samples found; --split test would be empty")
        return 2

    tokenizer_for = None
    if not args.no_stopword_mask:
        cache: dict[str, object] = {}

        def tokenizer_for(model_name: str):  # noqa: F811
            if model_name not in cache:
                cache[model_name] = get_tokenizer(model_name)
            return cache[model_name]

    results: dict = {
        "profile": args.profile,
        "norm_mode": args.norm_mode,
        "reset_mode": args.reset_mode,
        "split": args.split,
        "cache_meta": cache_meta,
        "scoring_saliency_norm": args.saliency_norm,
        "align_shift": args.align_shift,
        "lang": args.lang,
        "model": args.model,
    }

    y, native, by_agg, counts, records = score_corpus(
        annotations,
        args.cache,
        norm,
        tokenizer_for,
        lang=args.lang,
        model=args.model,
        config=args.config,
        split=args.split,
        tau=args.tau,
        alpha=args.alpha,
        reset_mode=args.reset_mode,
        norm_mode=args.norm_mode,
        align_shift=args.align_shift,
        saliency_norm=args.saliency_norm,
    )
    results["grasp"] = summarise(y, native, by_agg)
    results["counts"] = counts
    print(f"\n{counts['annotations']} annotations, {counts['chunks']} chunks (split={args.split})")
    for k in ["native"] + [f"much_agg/{a}" for a in AGGREGATORS]:
        print(line(k, results["grasp"][k]))
    print(f"  best -> {results['grasp'].get('best')}")

    if args.sweep:
        print("\n-- tau / alpha sweep (TRAINING SPLIT ONLY) --")
        best, grid = None, {}
        for tau in (1.0, 2.0, 4.0, 8.0, 16.0):
            for alpha in (0.3, 0.6, 0.9, 1.0):
                yy, nn, aa, _, _ = score_corpus(
                    annotations,
                    args.cache,
                    norm,
                    tokenizer_for,
                    lang=args.lang,
                    model=args.model,
                    config=args.config,
                    split="train",
                    tau=tau,
                    alpha=alpha,
                    reset_mode=args.reset_mode,
                    norm_mode=args.norm_mode,
                    align_shift=args.align_shift,
                    saliency_norm=args.saliency_norm,
                )
                s = summarise(yy, nn, aa)
                grid[f"tau{tau}_alpha{alpha}"] = s
                r = s["native"].get("roc_auc", float("nan"))
                print(f"  tau={tau:<5} alpha={alpha:<4} train native ROC={r:.4f}")
                if best is None or (r == r and r > best[0]):
                    best = (r, tau, alpha)
        results["sweep_train"] = grid
        results["best_sweep"] = {
            "train_roc_auc": best[0] if best else None,
            "tau": best[1] if best else args.tau,
            "alpha": best[2] if best else args.alpha,
        }
        if best:
            print(f"  winner on train split: tau={best[1]} alpha={best[2]} (train ROC={best[0]:.4f})")

    if args.ablate:
        print("\n-- ablations (native aggregator) --")
        variants = {
            "full": {},
            "no_grounding": {"use_signals": (False, True, True)},
            "no_sink": {"use_signals": (True, False, True)},
            "no_dispersion": {"use_signals": (True, True, False)},
            "grounding_only": {"use_signals": (True, False, False)},
            "sink_only": {"use_signals": (False, True, False)},
            "dispersion_only": {"use_signals": (False, False, True)},
            "no_saliency": {"use_saliency": False},
            "no_reset": {"reset_mode": "none"},
            "cross_claim_ema": {"reset_mode": "cross_claim_ema"},
            "mean_agg": {"tau": 0.01},
            "max_agg": {"tau": 200.0},
        }
        abl = {}
        for name, kw in variants.items():
            kw_full = {
                "tau": args.tau,
                "alpha": args.alpha,
                "reset_mode": args.reset_mode,
                "norm_mode": args.norm_mode,
                "align_shift": args.align_shift,
                "saliency_norm": args.saliency_norm,
            }
            kw_full.update(kw)
            yy, nn, aa, _, _ = score_corpus(
                annotations,
                args.cache,
                norm,
                tokenizer_for,
                lang=args.lang,
                model=args.model,
                config=args.config,
                split=args.split,
                **kw_full,
            )
            abl[name] = summarise(yy, nn, aa)
            print(line(name, abl[name]["native"]))
        results["ablation"] = abl

    if args.fuse_with:
        print(f"\n-- GRASP-fuse with {args.fuse_with} --")
        sig_path = args.much_root / "output" / "signals" / f"{args.fuse_with}.jsonl"
        if not sig_path.is_file():
            print(f"  not found at {sig_path}; skipping")
        else:
            other = {}
            for ln in sig_path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    other[d["generation_id"]] = np.asarray(d["values"], dtype=float)

            base = evaluate(
                annotations,
                lambda g: other.get(g),
                tokenizer_for=tokenizer_for,
                lang=args.lang,
            )
            if args.fuse_agg:
                base_agg = args.fuse_agg
                print(f"  using frozen baseline aggregator: {base_agg}")
            else:
                base_agg = max(
                    AGGREGATORS,
                    key=lambda a: base["per_aggregator"][a].get("roc_auc", -1),
                )
                print(f"  selected baseline aggregator from sweep: {base_agg}")

            results["baseline_alone"] = base["per_aggregator"][base_agg]
            print(line(f"{args.fuse_with}/{base_agg}", results["baseline_alone"]))

            # re-score both on the identical claim set so the fusion is aligned
            yb, sg, sb, cid = [], [], [], []
            for annot in annotations:
                gen = annot.generation
                if args.lang and gen.lang != args.lang:
                    continue
                gen_model = getattr(getattr(gen, "generation_cfg", None), "model_name", None)
                if args.model and gen_model != args.model:
                    continue
                if args.split != "all" and get_split(annot) != args.split:
                    continue
                if not has_consensus(annot.labels):
                    continue
                fp = args.cache / "features" / f"{gen.id}.npy"
                if not fp.is_file() or gen.id not in other:
                    continue
                feats = np.load(fp)
                chunks = [list(c) for c in annot.token_chunks]
                ts = grasp_token_signal(
                    feats,
                    gen,
                    norm,
                    use_signals=(True, True, True),
                    reset_mode=args.reset_mode,
                    alpha=args.alpha,
                    chunks=chunks,
                    tokenizer_for=tokenizer_for,
                    norm_mode=args.norm_mode,
                    align_shift=args.align_shift,
                )
                cs = claim_scores(
                    ts,
                    feats,
                    chunks,
                    tau=args.tau,
                    saliency_norm=args.saliency_norm,
                )
                ov = other[gen.id]
                if tokenizer_for is not None:
                    ov = mask_stopwords(
                        ov,
                        gen.output_tokens,
                        tokenizer_for(gen.generation_cfg.model_name),
                    )
                labels = annot.labels["gpt-4o"]
                for i, (chunk, label) in enumerate(zip(chunks[:-1], labels[:-1])):
                    start, stop = chunk[0], chunk[-1] + 1
                    if stop > len(ov) or i >= len(cs):
                        continue
                    yb.append(1 if label == HALLUCINATION_LABEL else 0)
                    sg.append(float(cs[i]))
                    sb.append(aggregate(ov[start:stop], base_agg))
                    cid.append(f"{gen.id}_chunk_{i}")
            if yb:
                fused = 0.5 * (rank(np.array(sg)) + rank(np.array(sb)))
                results["grasp_alone_aligned"] = metrics(yb, sg)
                results["baseline_aligned"] = metrics(yb, sb)
                results["grasp_fuse"] = metrics(yb, fused)
                print(line("grasp (aligned)", results["grasp_alone_aligned"]))
                print(line("baseline (aligned)", results["baseline_aligned"]))
                print(line("FUSED", results["grasp_fuse"]))

                # The fusion loop filters claims differently from score_corpus
                # (baseline signal present, chunk inside it), so match by claim id;
                # claims without a fused score get NaN rather than a neighbour's.
                fused_by_id = dict(zip(cid, fused))
                for r in records:
                    r["fused_score"] = float(fused_by_id.get(r["claim_id"], float("nan")))

    if args.export_csv:
        export_csv(records, args.export_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
