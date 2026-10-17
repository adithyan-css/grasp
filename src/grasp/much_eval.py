"""
An exact replication of MUCH's evaluation protocol.

Transcribed from `much/src/evaluation/supervized_evaluator.py::compute_evaluation`
and `much/src/evaluation/signal_evaluation.py`. Getting any of this wrong silently
produces numbers that cannot be compared with the published table, so the four
non-obvious rules are spelled out here:

1. **Label polarity.** Labels are in {-1, +1} and **-1 is the hallucination
   class** (`if chunk_label == -1: update_values_in`). Treating them as 0/1
   inverts the positive class and hands you an AUROC below 0.5.

2. **The last chunk is dropped.** MUCH iterates
   `token_chunks[:-1]` / `labels[:-1]` because the final chunk is EOS. Its label
   is auto-set during annotation and carries no information.

3. **Stopwords and punctuation are zeroed first.** Every generated token whose
   decoded string (lowercased, stripped) is an NLTK stopword for en/es/fr/de, a
   punctuation mark, or one of a handful of clitics has its signal set to 0
   *before* claim aggregation. This needs the generating model's tokenizer.

4. **Annotator consensus.** Only annotations where gpt-4o and gpt-4.1 agree on
   the whole label vector are evaluated.

Rule 3 is worth noticing for its own sake: the benchmark has to hand-zero
function words because token-probability signals are dominated by them. GRASP's
attention saliency weighting derives the same thing from the model itself,
without a fixed word list — which is a point the paper should make.
"""

from __future__ import annotations

import string
import typing as t
from functools import lru_cache

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

AGGREGATORS = ("mean", "max", "geometric", "product")
HALLUCINATION_LABEL = -1
NLTK_LANGS = ("english", "spanish", "french", "german")


# ---------------------------------------------------------------------------
# stopword masking (rule 3)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def punctuation_set() -> frozenset[str]:
    """Transcribed verbatim from MUCH's `get_punctuation_set`."""
    return frozenset(
        set(string.punctuation)
        | {
            "،", "؛", "؟", "，", "。", "、", "？", "！", "：", "；",
            "（", "）", "《", "》", "“", "”", "¿", "¡", "«", "»", "…",
            "''", "``", "\n", "", "''", '""',
        }
    )


@lru_cache(maxsize=1)
def nltk_stopwords_set() -> frozenset[str]:
    out: set[str] = set()
    try:
        import nltk
        from nltk.corpus import stopwords

        for lang in NLTK_LANGS:
            try:
                out |= set(stopwords.words(lang))
            except (OSError, LookupError):
                nltk.download("stopwords")
                out |= set(stopwords.words(lang))
    except (ImportError, ModuleNotFoundError):
        # Fallback basic stopword set when nltk is not installed locally
        out = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "these", "those", "then", "just", "so", "than",
            "such", "both", "through", "about", "for", "is", "of", "while", "during",
            "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
            "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
            "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
            "wasn", "weren", "won", "wouldn", "le", "la", "les", "un", "une", "des",
            "der", "die", "das", "ein", "eine", "el", "los", "las",
        }
    return frozenset(out)


@lru_cache(maxsize=1)
def extended_stopwords_set() -> frozenset[str]:
    extra = {"'s", "'re", "'ve", "'d", "'ll", "n't", "''", "``", " ", ""}
    return frozenset(nltk_stopwords_set() | punctuation_set() | extra)


def mask_stopwords(
    values: np.ndarray, output_tokens: t.Sequence[int], tokenizer
) -> np.ndarray:
    """Zero the signal on stopword and punctuation tokens, as MUCH does."""
    out = np.asarray(values, dtype=float).copy()
    stop = extended_stopwords_set()
    for idx, tok in enumerate(output_tokens):
        if idx >= out.shape[0]:
            break
        if tokenizer.decode(tok).lower().strip() in stop:
            out[idx] = 0.0
    return out


# ---------------------------------------------------------------------------
# claim aggregation (rule: exact formulas, including the 1e-5)
# ---------------------------------------------------------------------------


def aggregate(chunk: np.ndarray, how: str) -> float:
    x = np.asarray(chunk, dtype=float)
    if x.size == 0:
        return 0.0
    if how == "mean":
        return float(np.mean(x))
    if how == "max":
        return float(np.max(x))
    if how == "geometric":
        # note the +1e-5 — MUCH includes it, so leaving it out shifts the score
        return float(1 - (np.prod(1 - x) + 1e-5) ** (1 / x.shape[0]))
    if how == "product":
        return float(1 - np.prod(1 - x))
    raise ValueError(f"unknown aggregator: {how}")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, budget: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= budget
    return float(tpr[ok].max()) if ok.any() else 0.0


def metrics(y: t.Sequence[int], s: t.Sequence[float]) -> dict:
    """MUCH reports ROC-AUC as auc(fpr, tpr) and PR-AUC as auc(recall, precision).

    `average_precision` is reported alongside because the two PR numbers differ
    and papers are not always explicit about which one they quote.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if y.size == 0 or len(np.unique(y)) < 2:
        return {"n": int(y.size), "positives": int(y.sum()) if y.size else 0}
    fpr, tpr, _ = roc_curve(y, s)
    prc, rec, _ = precision_recall_curve(y, s)
    return {
        "n": int(y.size),
        "positives": int(y.sum()),
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(rec, prc)),
        "average_precision": float(average_precision_score(y, s)),
        "tpr_at_10fpr": tpr_at_fpr(y, s, 0.10),
        "tpr_at_05fpr": tpr_at_fpr(y, s, 0.05),
    }


# ---------------------------------------------------------------------------
# the evaluation loop
# ---------------------------------------------------------------------------


def has_consensus(labels: dict) -> bool:
    """Rule 4: both automatic annotators present and in full agreement."""
    return (
        "gpt-4o" in labels
        and "gpt-4.1" in labels
        and labels["gpt-4o"] == labels["gpt-4.1"]
    )


def evaluate(
    annotations: t.Iterable,
    signal_for: t.Callable[[str], np.ndarray | None],
    tokenizer_for: t.Callable[[str], t.Any] | None = None,
    lang: str | None = None,
    aggregators: t.Sequence[str] = AGGREGATORS,
) -> dict:
    """Score every claim under MUCH's protocol.

    Parameters
    ----------
    annotations
        MUCH `Annotation` objects (or anything with `.generation`,
        `.token_chunks`, `.labels`).
    signal_for
        generation_id -> per-token signal array, or None when absent.
    tokenizer_for
        model_name -> tokenizer. Pass None to SKIP stopword masking — the result
        then does NOT match the published table and is flagged as such.
    lang
        restrict to one language, or None for all.

    Returns
    -------
    {"per_aggregator": {agg: metrics}, "counts": {...},
     "stopword_masking": bool}
    """
    scores: dict[str, list[float]] = {a: [] for a in aggregators}
    labels_out: list[int] = []
    n_annot = n_chunks = n_missing = 0

    for annot in annotations:
        gen = annot.generation
        if lang is not None and gen.lang != lang:
            continue
        if not has_consensus(annot.labels):
            continue

        values = signal_for(gen.id)
        if values is None:
            n_missing += 1
            continue
        values = np.asarray(values, dtype=float)

        if tokenizer_for is not None:
            values = mask_stopwords(
                values, gen.output_tokens, tokenizer_for(gen.generation_cfg.model_name)
            )

        n_annot += 1
        # rule 2: drop the trailing EOS chunk
        for chunk, label in zip(annot.token_chunks[:-1], annot.labels["gpt-4o"][:-1]):
            start, stop = chunk[0], chunk[-1] + 1
            piece = values[start:stop]
            if piece.size == 0:
                continue
            n_chunks += 1
            # rule 1: -1 is the hallucination class
            labels_out.append(1 if label == HALLUCINATION_LABEL else 0)
            for agg in aggregators:
                scores[agg].append(aggregate(piece, agg))

    return {
        "per_aggregator": {a: metrics(labels_out, scores[a]) for a in aggregators},
        "counts": {
            "annotations": n_annot,
            "chunks": n_chunks,
            "missing_signal": n_missing,
        },
        "stopword_masking": tokenizer_for is not None,
        "lang": lang,
    }
