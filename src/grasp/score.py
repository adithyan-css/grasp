"""
Turning per-token attention statistics into a per-token hallucination signal,
and per-token signals into per-claim scores.

MUCH's evaluator expects a per-token signal in [0, 1] where HIGHER means MORE
likely hallucinated, then aggregates it over each claim's token span with one of
{mean, max, geometric, product}. We therefore emit a per-token signal so that
GRASP is comparable to every baseline under the identical protocol, and we
additionally provide `softmin_claim_score`, the saliency-weighted aggregator the
paper proposes, reported alongside the four standard ones.
"""

from __future__ import annotations

import typing as t

import numpy as np

from .heads import DISPERSION, GROUNDING, SALIENCY, SINK, middle_layers

EPS = 1e-9


# ----------------------------------------------------------------------------
# corpus statistics (fit on unlabelled data only)
# ----------------------------------------------------------------------------


class CorpusNormaliser:
    """Per-layer z-scores for each of the three signals.

    Fitted on unlabelled generations. No hallucination label is ever consulted,
    so this stays inside the training-free claim.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None  # (L, 3)
        self.std_: np.ndarray | None = None

    def fit(self, features: t.Sequence[np.ndarray]) -> "CorpusNormaliser":
        cols = [GROUNDING, SINK, DISPERSION]
        acc = [[] for _ in cols]
        for f in features:
            if f.shape[2] == 0:
                continue
            for j, c in enumerate(cols):
                acc[j].append(f[:, 0, :, c])  # (L,T), head axis already collapsed
        if not acc[0]:
            raise ValueError("Nothing to fit the normaliser on.")
        stacked = [np.concatenate(a, axis=1) for a in acc]  # each (L, total_T)
        self.mean_ = np.stack([s.mean(axis=1) for s in stacked], axis=1)  # (L,3)
        self.std_ = np.stack([s.std(axis=1) + EPS for s in stacked], axis=1)
        return self

    def transform(self, feats: np.ndarray) -> np.ndarray:
        """feats (L,1,T,4) -> z-scored (L,T,3) in order G, S, D."""
        cols = [GROUNDING, SINK, DISPERSION]
        x = feats[:, 0, :, :][:, :, cols]  # (L,T,3)
        return (x - self.mean_[:, None, :]) / self.std_[:, None, :]


# ----------------------------------------------------------------------------
# cache-aware scoring options
# ----------------------------------------------------------------------------


def resolve_scoring_options(
    cache_meta: dict, saliency_norm: str, align_shift: int
) -> tuple[str, int, list[str]]:
    """Reconcile requested scoring options with how the cache was extracted.

    `cache_meta` holds the extraction settings ('align', 'saliency_norm'); an
    empty dict means a pre-metadata cache, which was always legacy/none.

    - Saliency already divided by (T-1-i) at extraction must not be divided
      again at scoring, so the scoring-side norm drops to 'none'.
    - align_shift exists to repair legacy caches; on an align='current' cache
      it would mis-align features, so it is refused.

    Returns (saliency_norm, align_shift, notes).
    """
    notes: list[str] = []
    feat_sal = cache_meta.get("saliency_norm", "none")
    feat_align = cache_meta.get("align", "legacy")

    if feat_sal == "per_later_token" and saliency_norm == "per_later_token":
        saliency_norm = "none"
        notes.append(
            "cache saliency already per_later_token-normalised at extraction; "
            "scoring-side saliency_norm set to 'none' to avoid dividing twice"
        )
    if feat_align == "current" and align_shift > 0:
        raise ValueError(
            "align_shift > 0 on an align='current' cache would mis-align features; "
            "the shift only repairs align='legacy' caches."
        )
    return saliency_norm, align_shift, notes


# ----------------------------------------------------------------------------
# per-token signal
# ----------------------------------------------------------------------------


def token_signal(
    feats: np.ndarray,
    normaliser: CorpusNormaliser,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    layers: t.Sequence[int] | None = None,
    use_signals: tuple[bool, bool, bool] = (True, True, True),
    norm_mode: str = "per_gen_rank",
    align_shift: int = 0,
) -> np.ndarray:
    """feats (L, 1, T, 4) -> per-token hallucination score in [0,1], shape (T,).

    Hallucination is high when grounding is LOW and sink leakage and dispersion
    are HIGH, hence the sign pattern below.

    Parameters
    ----------
    feats : np.ndarray
        Attention features of shape (L, 1, T, 4).
    normaliser : CorpusNormaliser
        Fitted corpus normaliser providing per-layer z-scoring statistics.
    weights : tuple[float, float, float]
        Signal weights (wg, ws, wd) for grounding, sink, dispersion.
    layers : Sequence[int] | None
        Layers to aggregate over (default: middle layers).
    use_signals : tuple[bool, bool, bool]
        Booleans toggling (grounding, sink, dispersion).
    norm_mode : str
        "per_gen_rank" (legacy): Rank-normalises across tokens within the single
        generation before sigmoid. Makes layers commensurable within-generation
        but erases absolute risk differences across generations.
        "corpus" (corrected): Uses corpus-level z-scores across layers directly,
        averaging layer risks without within-generation ranking, then squashes with
        sigmoid. Preserves absolute risk differences across generations.
    align_shift : int
        Only for caches extracted with align='legacy'. A legacy cache stores, at
        index i, query row P+i — the attention that PREDICTS token i+1. The row
        that generated token i is P+i-1, i.e. legacy index i-1. So the fix moves
        features FORWARD: token i takes legacy features from index i - align_shift.
        The first `align_shift` tokens have no legacy row (row P-1 was never
        stored) and reuse index 0, so they are approximate; re-extract with
        align='current' for exact alignment. Never combine with an
        align='current' cache, which is already aligned.
    """
    z = normaliser.transform(feats)  # (L,T,3)
    if align_shift > 0:
        # token i <- legacy index i - align_shift; head of the sequence padded
        # with legacy index 0 (the generating row for token 0 is not cached)
        L, T, C = z.shape
        k = min(align_shift, T)
        pad = np.repeat(z[:, :1, :], k, axis=1)
        z = np.concatenate([pad, z[:, : T - k, :]], axis=1)

    L = z.shape[0]
    layers = list(layers) if layers is not None else middle_layers(L)
    layers = [l for l in layers if 0 <= l < L] or list(range(L))

    wg, ws, wd = weights
    ug, us, ud = use_signals
    risk = (
        (-wg * z[:, :, 0] if ug else 0.0)
        + (ws * z[:, :, 1] if us else 0.0)
        + (wd * z[:, :, 2] if ud else 0.0)
    )  # (L,T), higher = riskier

    band = risk[layers]  # (len(layers), T)
    if norm_mode == "corpus":
        # Direct corpus z-score fusion: average layer risk and squash
        fused = band.mean(axis=0)  # (T,)
        return _sigmoid(fused)
    elif norm_mode == "per_gen_rank":
        # Legacy: rank within generation
        fused = _rank_normalise(band).mean(axis=0)  # (T,)
        return _sigmoid(fused)
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}. Expected 'corpus' or 'per_gen_rank'.")


def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """Rank-normalise each row to [0,1]. Makes layers commensurable without
    fitting anything to labels."""
    out = np.empty_like(x, dtype=np.float64)
    n = x.shape[1]
    if n == 0:
        return out
    for i in range(x.shape[0]):
        order = np.argsort(np.argsort(x[i]))
        out[i] = order / max(n - 1, 1)
    return (out - 0.5) * 4.0  # spread before the sigmoid


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------
# per-claim aggregation
# ----------------------------------------------------------------------------


def saliency_weights(
    feats: np.ndarray,
    chunk: t.Sequence[int],
    layers: t.Sequence[int] | None = None,
    saliency_norm: str = "none",
) -> np.ndarray:
    """Normalised saliency weights for the tokens of one claim.

    A token's weight is how much attention LATER tokens send back to it, which
    is high for names, dates and numbers and low for function words. This is the
    free replacement for CCP's per-token NLI filtering.

    Parameters
    ----------
    feats : np.ndarray
        Feature array (L, 1, T, 4).
    chunk : Sequence[int]
        Token indices for this claim.
    layers : Sequence[int] | None
        Layers to average saliency over.
    saliency_norm : str
        "none" (legacy): raw accumulated attention mass.
        "per_later_token": normalized by the number of strictly later tokens (T - 1 - i).
    """
    L = feats.shape[0]
    T = feats.shape[2]
    layers = list(layers) if layers is not None else middle_layers(L)
    layers = [l for l in layers if 0 <= l < L] or list(range(L))
    idx = np.asarray(list(chunk), dtype=int)
    idx = idx[idx < T]
    if idx.size == 0:
        return np.zeros(0)

    raw_w = feats[layers][:, 0, :, SALIENCY].mean(axis=0)  # (T,)
    if saliency_norm == "per_later_token":
        # Denominator is strictly later tokens available: T - 1 - i for token at index i
        denom = np.maximum(T - 1 - np.arange(T), 1.0)
        w = (raw_w / denom)[idx]
    else:
        w = raw_w[idx]

    w = np.clip(w, 0, None)
    if w.sum() <= EPS:
        return np.full(idx.size, 1.0 / idx.size)
    return w / w.sum()


def softmin_claim_score(
    token_scores: np.ndarray,
    chunk: t.Sequence[int],
    weights: np.ndarray | None = None,
    tau: float = 8.0,
    claim_agg: str = "softmin",
) -> float:
    r"""Aggregate token hallucination risks $s_i \in [0, 1]$ into a claim-level score.

    Mathematical Definition
    -----------------------
    In the literature, when scoring confidence $c_i \in [0, 1]$, an unsupported claim
    is governed by its least confident token (soft-minimum over confidence).
    Because MUCH uses hallucination risk $s_i = 1 - c_i$, a soft-minimum over confidence
    is mathematically identical to a soft-MAXIMUM over risk $s_i$:

    $$\text{score}(C) = \text{LogSumExp}_\tau(\{s_i\}_{i \in C}; \mathbf{w}) = m + \frac{1}{\tau} \ln \sum_{i \in C} w_i \exp\left(\tau (s_i - m)\right)$$

    where $m = \max_{i \in C} s_i$ stabilizes computation against numerical overflow,
    $\mathbf{w}$ are normalized saliency weights ($\sum_i w_i = 1$), and $\tau > 0$
    is the inverse temperature:
      - As $\tau \to 0$, $\text{score}(C) \to \sum_{i \in C} w_i s_i$ (saliency-weighted mean).
      - As $\tau \to \infty$, $\text{score}(C) \to \max_{i \in C} s_i$ (maximum risk token).

    Parameters
    ----------
    token_scores : np.ndarray
        Per-token risk values $s_i \in [0, 1]$.
    chunk : Sequence[int]
        Token indices belonging to the claim.
    weights : np.ndarray | None
        Normalized token weights $w_i \ge 0, \sum w_i = 1$. If None, uniform weights $1/|C|$.
    tau : float
        Softmax temperature (default 8.0).
    claim_agg : str
        "softmin": LogSumExp soft-max over risk as defined above.
        "mean": Weighted mean $\sum w_i s_i$.
        "max": Maximum $\max_{i} s_i$.
        "softmin_len_norm": Length-normalized variant.
    """
    idx = np.asarray(list(chunk), dtype=int)
    idx = idx[idx < token_scores.shape[0]]
    if idx.size == 0:
        return 0.0
    s = token_scores[idx]
    w = weights if weights is not None else np.full(idx.size, 1.0 / idx.size)
    w = w[: s.shape[0]]
    if w.sum() > 0:
        w = w / w.sum()

    if claim_agg == "mean":
        return float(np.sum(w * s))
    elif claim_agg == "max":
        return float(s.max())
    elif claim_agg in ("softmin", "softmin_len_norm"):
        m = s.max()
        lse = float(m + np.log(np.sum(w * np.exp(tau * (s - m))) + EPS) / tau)
        if claim_agg == "softmin_len_norm" and idx.size > 1:
            # Length penalty correction
            lse = lse / (1.0 + 0.05 * np.log(idx.size))
        return lse
    else:
        raise ValueError(f"Unknown claim_agg: {claim_agg}")


def claim_scores(
    token_scores: np.ndarray,
    feats: np.ndarray,
    token_chunks: t.Sequence[t.Sequence[int]],
    tau: float = 8.0,
    use_saliency: bool = True,
    layers: t.Sequence[int] | None = None,
    saliency_norm: str = "none",
    claim_agg: str = "softmin",
) -> np.ndarray:
    """Computes per-claim risk scores for all chunks in a generation."""
    out = np.zeros(len(token_chunks))
    for i, chunk in enumerate(token_chunks):
        w = (
            saliency_weights(feats, chunk, layers=layers, saliency_norm=saliency_norm)
            if use_saliency
            else None
        )
        out[i] = softmin_claim_score(
            token_scores, chunk, weights=w, tau=tau, claim_agg=claim_agg
        )
    return out


# ----------------------------------------------------------------------------
# claim-boundary reset
# ----------------------------------------------------------------------------


def apply_boundary_reset(
    token_scores: np.ndarray,
    token_chunks: t.Sequence[t.Sequence[int]],
    alpha: float = 0.6,
    reset_mode: str = "reset",
) -> np.ndarray:
    r"""Recurrence that accumulates risk and handles cross-claim boundaries.

    Parameters
    ----------
    token_scores : np.ndarray
        Per-token risk array (T,).
    token_chunks : Sequence[Sequence[int]]
        List of claim token spans.
    alpha : float
        EMA smoothing factor ($0 < \alpha \le 1$). Higher means less memory of prior tokens.
    reset_mode : str
        "reset" (legacy): Accumulates EMA within each chunk independently and resets
        at the start of each claim.
        "none": No recurrence; returns a copy of raw `token_scores`.
        "cross_claim_ema": Continuous sequence-level EMA across the entire generation
        without resetting at claim boundaries (the un-reset ablation baseline).
    """
    if reset_mode == "none":
        return token_scores.copy()

    if reset_mode == "cross_claim_ema":
        out = np.empty_like(token_scores)
        carry = 0.0
        for i in range(len(token_scores)):
            carry = token_scores[i] if i == 0 else (alpha * token_scores[i] + (1 - alpha) * carry)
            out[i] = carry
        return out

    if reset_mode == "reset":
        out = token_scores.copy()
        for chunk in token_chunks:
            idx = [i for i in chunk if i < len(out)]
            carry = 0.0
            for j, i in enumerate(idx):
                carry = token_scores[i] if j == 0 else (
                    alpha * token_scores[i] + (1 - alpha) * carry
                )
                out[i] = carry
        return out

    raise ValueError(f"Unknown reset_mode: {reset_mode}. Expected 'reset', 'none', or 'cross_claim_ema'.")
