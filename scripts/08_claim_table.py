#!/usr/bin/env python3
"""
GRASP stage 5a — build a per-claim feature table from one or more feature caches.

One row per scored claim (MUCH protocol: consensus labels, trailing EOS chunk
dropped). Columns:

  ids        gen_id, config, model, lang, split, chunk index
  label      1 = hallucinated (gpt-4o label == -1)
  ccp        `ccp_24_8` claim score: stopword-masked, `product` aggregator (frozen)
  grasp0     the training-free GRASP-0 claim score at a given tau/alpha
  X          three feature groups, by name prefix:
             b{bin}_h{rank}_*  attention, model-agnostic: for each of K
                 relative-depth bins x kept-head rank x {grounding, sink,
                 dispersion}: mean / max / min over the claim's content tokens
                 of the z-scored value, and the saliency-weighted mean
             tp_*  token probabilities MUCH ships (top-24 per token): chosen-
                 token surprisal, entropy, top-1 confidence, top-1/top-2 gap.
                 No extra model is run for these.
             len_tokens, len_content, rel_start, gen_len  length / position

Works with 1-head caches (one variance-selected head per layer) and k-head caches
(`--head-select variance_topk`); head rank 0 is the top-variance head in both.
The table carries every split; selection between train and test happens in the
experiment scripts, never here.

    python scripts/08_claim_table.py --much-root ../much \
        --cache results/.../cache_<cfg> [--cache ...] --out results/dev/claims.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp.much_eval import (  # noqa: E402
    HALLUCINATION_LABEL,
    aggregate,
    extended_stopwords_set,
    has_consensus,
    mask_stopwords,
)
from grasp.score import (  # noqa: E402
    CorpusNormaliser,
    apply_boundary_reset,
    claim_scores,
    token_signal,
)

N_BINS = 8
STATS = ("mean", "max", "min", "salw")
SIGNALS = ("grounding", "sink", "dispersion")
TP_NAMES = ["tp_nll_mean", "tp_nll_max", "tp_ent_mean", "tp_ent_max",
            "tp_top1_min", "tp_gap_mean", "tp_nll_mean_all"]
LEN_NAMES = ["len_tokens", "len_content", "rel_start", "gen_len"]
AUTO_ANNOTATOR_PREFIX = "gpt-"


def feature_names(n_heads: int = 1, n_bins: int = N_BINS) -> list[str]:
    names = [
        f"b{b}_h{h}_{sig}_{st}"
        for b in range(n_bins) for h in range(n_heads) for sig in SIGNALS for st in STATS
    ]
    return names + TP_NAMES + LEN_NAMES


def signal_of(name: str) -> str:
    """'b3_h0_grounding_min' -> 'grounding'."""
    return name.split("_")[2]


def get_split(annot) -> str:
    labels = getattr(annot, "labels", None) or {}
    if any(not str(k).startswith(AUTO_ANNOTATOR_PREFIX) for k in labels):
        return "test"
    return "train"


def load_signal(much_root: Path, name: str) -> dict[str, np.ndarray]:
    path = much_root / "output" / "signals" / f"{name}.jsonl"
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            out[d["generation_id"]] = np.asarray(d["values"], dtype=float)
    return out


def depth_bins(n_layers: int, n_bins: int = N_BINS) -> list[list[int]]:
    edges = np.linspace(0, n_layers, n_bins + 1)
    return [
        list(range(int(round(edges[b])), max(int(round(edges[b + 1])), int(round(edges[b])) + 1)))
        for b in range(n_bins)
    ]


def content_mask(output_tokens, tokenizer, stop) -> np.ndarray:
    keep = np.ones(len(output_tokens), dtype=bool)
    for i, tok in enumerate(output_tokens):
        if tokenizer.decode([tok]).strip().lower() in stop:
            keep[i] = False
    return keep


def token_prob_stats(output_tokens, topn) -> np.ndarray:
    """Per-token (nll, entropy, top1, gap) from MUCH's shipped top-N probabilities.
    Mass outside the top-N is kept as one extra bucket for the entropy."""
    out = np.zeros((len(output_tokens), 4))
    for i, (tok, d) in enumerate(zip(output_tokens, topn)):
        probs = np.sort(np.fromiter(d.values(), dtype=float))[::-1]
        floor = probs[-1] * 0.5 if probs.size else 1e-6
        p_tok = d.get(tok, d.get(str(tok), floor))
        rest = max(1.0 - probs.sum(), 0.0)
        q = np.append(probs, rest) if rest > 0 else probs
        q = q[q > 0]
        out[i] = [
            -np.log(max(p_tok, 1e-12)),
            -(q * np.log(q)).sum(),
            probs[0] if probs.size else 0.0,
            (probs[0] - probs[1]) if probs.size > 1 else 0.0,
        ]
    return out


def fit_head_normaliser(feat_files) -> tuple[np.ndarray, np.ndarray]:
    """Per-(layer, head) mean/std of G,S,D over all generations of one config.
    Uses no labels, so it stays inside the training-free part of the method."""
    s1 = s2 = None
    n = 0
    for fp in feat_files:
        f = np.load(fp).astype(np.float64)[..., :3]  # (L,H,T,3)
        if f.shape[2] == 0:
            continue
        a, b = f.sum(axis=2), (f ** 2).sum(axis=2)
        s1 = a if s1 is None else s1 + a
        s2 = b if s2 is None else s2 + b
        n += f.shape[2]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0)) + 1e-9
    return mean, std  # (L,H,3)


def claim_features(z, W, tp, chunk, keep, bins, T) -> np.ndarray | None:
    """z: (L,H,T,3) z-scored G,S,D; W: (T,) saliency; tp: (n_out,4) token-prob stats."""
    idx = np.asarray([i for i in chunk if i < T], dtype=int)
    if idx.size == 0:
        return None
    content = idx[keep[idx]] if keep[idx].any() else idx
    w = np.clip(W[content], 0, None)
    w = w / w.sum() if w.sum() > 1e-12 else np.full(content.size, 1.0 / content.size)
    row = []
    for layers in bins:
        zb = z[layers][:, :, content, :].mean(axis=0)  # (H, n_content, 3)
        for h in range(zb.shape[0]):
            for c in range(3):
                v = zb[h, :, c]
                row += [v.mean(), v.max(), v.min(), float(np.sum(w * v))]
    tc = tp[content[content < len(tp)]]
    ta = tp[idx[idx < len(tp)]]
    if tc.size == 0:
        tc = ta if ta.size else np.zeros((1, 4))
    if ta.size == 0:
        ta = tc
    row += [tc[:, 0].mean(), tc[:, 0].max(), tc[:, 1].mean(), tc[:, 1].max(),
            tc[:, 2].min(), tc[:, 3].mean(), ta[:, 0].mean()]
    row += [idx.size, content.size, idx[0] / max(T, 1), T]
    return np.asarray(row, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--cache", type=Path, action="append", required=True,
                    help="feature cache dir (repeatable, one per generation config)")
    ap.add_argument("--baseline", default="ccp_24_8")
    ap.add_argument("--baseline-agg", default="product")
    ap.add_argument("--tau", type=float, default=16.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.much_root))
    from src.annotation import get_all_annotations  # type: ignore
    from src.generation import get_tokenizer  # type: ignore

    annotations = list(get_all_annotations().values())
    base = load_signal(args.much_root, args.baseline)
    stop = extended_stopwords_set()
    toks: dict[str, object] = {}

    rows, meta = [], {k: [] for k in ("gen_id", "config", "model", "lang", "split", "chunk")}
    labels, ccp, grasp0 = [], [], []
    n_heads = None
    for cache in args.cache:
        d = np.load(cache / "normaliser.npz")
        norm = CorpusNormaliser()
        norm.mean_, norm.std_ = d["mean"], d["std"]
        feat_dir = cache / "features"
        hmean, hstd = fit_head_normaliser(sorted(feat_dir.glob("*.npy")))
        n_cache = 0
        for annot in annotations:
            gen = annot.generation
            fp = feat_dir / f"{gen.id}.npy"
            if not fp.is_file() or not has_consensus(annot.labels) or gen.id not in base:
                continue
            model = gen.generation_cfg.model_name
            if model not in toks:
                toks[model] = get_tokenizer(model)
            tok = toks[model]
            feats = np.load(fp).astype(np.float64)
            L, H, T = feats.shape[0], feats.shape[1], feats.shape[2]
            if n_heads is None:
                n_heads = H
            assert H == n_heads, f"mixed head counts across caches: {H} vs {n_heads}"
            n_out = len(gen.output_tokens)
            chunks = [list(c) for c in annot.token_chunks]

            # GRASP-0 claim scores, exactly as 03_score_and_eval.py computes them
            # (head 0 is the variance-selected head in both 1-head and k-head caches)
            f0 = feats[:, :1]
            ts = token_signal(f0, norm, norm_mode="corpus")
            ts = np.pad(ts, (0, max(0, n_out - ts.shape[0])), constant_values=0.5)[:n_out]
            ts = apply_boundary_reset(ts, chunks, alpha=args.alpha)
            ts = mask_stopwords(ts, gen.output_tokens, tok)
            cs = claim_scores(ts, f0, chunks, tau=args.tau)

            ov = mask_stopwords(base[gen.id], gen.output_tokens, tok)
            z = (feats[..., :3] - hmean[:, :, None, :]) / hstd[:, :, None, :]  # (L,H,T,3)
            W = feats[:, 0, :, 3].mean(axis=0)
            tp = token_prob_stats(gen.output_tokens, gen.topn_prob_dicts)
            keep = content_mask(gen.output_tokens[:T], tok, stop)
            if keep.size < T:
                keep = np.pad(keep, (0, T - keep.size), constant_values=True)
            bins = depth_bins(L)
            split = get_split(annot)
            lab = annot.labels["gpt-4o"]
            for i, (chunk, label) in enumerate(zip(chunks[:-1], lab[:-1])):
                start, stop_ = chunk[0], chunk[-1] + 1
                if stop_ > len(ov) or i >= len(cs):
                    continue
                f = claim_features(z, W, tp, chunk, keep, bins, T)
                if f is None:
                    continue
                rows.append(f)
                labels.append(1 if label == HALLUCINATION_LABEL else 0)
                ccp.append(aggregate(ov[start:stop_], args.baseline_agg))
                grasp0.append(float(cs[i]))
                for k, v in (("gen_id", gen.id), ("config", gen.generation_config),
                             ("model", model), ("lang", gen.lang), ("split", split), ("chunk", i)):
                    meta[k].append(v)
            n_cache += 1
        print(f"{cache}: {n_cache} generations")

    names = feature_names(n_heads)
    X = np.vstack(rows)
    assert X.shape[1] == len(names), (X.shape, len(names))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X, y=np.asarray(labels), ccp=np.asarray(ccp),
        grasp0=np.asarray(grasp0), feature_names=np.asarray(names),
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    y = np.asarray(labels)
    sp = np.asarray(meta["split"])
    print(f"wrote {args.out}: {len(y)} claims "
          f"(train {int((sp == 'train').sum())}, test {int((sp == 'test').sum())}), "
          f"{y.mean():.3f} hallucinated, {len(names)} features, {n_heads} head(s)/layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
