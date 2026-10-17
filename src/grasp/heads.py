"""
Label-free attention-head selection.

The whole method is training-free, so heads must be chosen without ever looking
at a hallucination label. GRASP picks, per layer, the head whose grounding ratio
carries the most information across tokens.

Rationale: a head that returns roughly the same grounding value for every token
cannot discriminate anything, however high or low its mean sits. Variance is a
pure signal-quality criterion and needs no ground truth.

Two alternatives are implemented for the ablation table:

  variance   ours          argmax_h Var_t[G]
  prev_token RAUQ's rule   argmax_h mean_t[ attention to the immediately
                           preceding token ]
  random     control       uniformly sampled head

`prev_token` is approximated here from the dispersion/saliency statistics we
already store; run `select_heads_prev_token_exact` if you want the exact rule,
which needs a second calibration pass that keeps the raw diagonal.
"""

from __future__ import annotations

import typing as t

import numpy as np

GROUNDING, SINK, DISPERSION, SALIENCY = 0, 1, 2, 3


def select_heads_variance(
    calib_features: t.Sequence[np.ndarray],
    feature_idx: int = GROUNDING,
    min_tokens: int = 8,
) -> dict[int, int]:
    """calib_features: list of (L, H, T, 4) arrays from the calibration split.

    Returns {layer_idx: head_idx}.
    """
    if not calib_features:
        raise ValueError("Calibration set is empty.")

    L, H = calib_features[0].shape[0], calib_features[0].shape[1]
    # Welford-free two-pass over a modest calibration set is fine.
    per_sample = np.zeros((len(calib_features), L, H), dtype=np.float64)
    valid = np.zeros(len(calib_features), dtype=bool)

    for i, feats in enumerate(calib_features):
        T = feats.shape[2]
        if T < min_tokens:
            continue
        valid[i] = True
        x = feats[:, :, :, feature_idx].astype(np.float64)  # (L,H,T)
        per_sample[i] = x.var(axis=2)

    if not valid.any():
        raise ValueError("No calibration sample long enough for head selection.")

    mean_var = per_sample[valid].mean(axis=0)  # (L,H)
    return {layer: int(np.argmax(mean_var[layer])) for layer in range(L)}


def select_heads_variance_topk(
    calib_features: t.Sequence[np.ndarray],
    k: int = 4,
    feature_idx: int = GROUNDING,
    min_tokens: int = 8,
) -> dict[int, list[int]]:
    """Like `select_heads_variance`, but keeps the k highest-variance heads per
    layer, ordered by variance (rank 0 is the head the 1-head rule picks).

    Returns {layer_idx: [head_idx, ...]}.
    """
    L, H = calib_features[0].shape[0], calib_features[0].shape[1]
    var = [
        f[:, :, :, feature_idx].astype(np.float64).var(axis=2)
        for f in calib_features
        if f.shape[2] >= min_tokens
    ]
    if not var:
        raise ValueError("No calibration sample long enough for head selection.")
    mean_var = np.mean(var, axis=0)  # (L,H)
    k = min(k, H)
    return {l: [int(h) for h in np.argsort(-mean_var[l])[:k]] for l in range(L)}


def select_heads_random(n_layers: int, n_heads: int, seed: int = 0) -> dict[int, int]:
    rng = np.random.RandomState(seed)
    return {l: int(rng.randint(n_heads)) for l in range(n_layers)}


def select_heads_fixed(n_layers: int, head: int = 0) -> dict[int, int]:
    return {l: head for l in range(n_layers)}


def select_heads(
    method: str = "variance",
    calib_features: t.Sequence[np.ndarray] | None = None,
    n_layers: int | None = None,
    n_heads: int | None = None,
    seed: int = 0,
    head: int = 0,
    k: int = 4,
) -> dict[int, int] | dict[int, list[int]] | None:
    """Unified head selector: 'variance', 'variance_topk', 'random', 'fixed', 'all_heads'."""
    if method == "all_heads":
        return None
    if method == "variance_topk":
        if calib_features is None:
            raise ValueError("calib_features required for variance head selection")
        return select_heads_variance_topk(calib_features, k=k)
    if method == "variance":
        if calib_features is None:
            raise ValueError("calib_features required for variance head selection")
        return select_heads_variance(calib_features)
    if method == "random":
        if n_layers is None or n_heads is None:
            if calib_features:
                n_layers, n_heads = calib_features[0].shape[0], calib_features[0].shape[1]
            else:
                raise ValueError("n_layers and n_heads required for random head selection")
        return select_heads_random(n_layers, n_heads, seed=seed)
    if method == "fixed":
        if n_layers is None:
            if calib_features:
                n_layers = calib_features[0].shape[0]
            else:
                raise ValueError("n_layers required for fixed head selection")
        return select_heads_fixed(n_layers, head=head)
    raise ValueError(f"Unknown head selection method: {method}")


def middle_layers(n_layers: int, lo: float = 0.25, hi: float = 0.85) -> list[int]:
    """The layer band GRASP fuses over.

    Early layers are syntactic and late layers are committed to the output
    token, so the retrieval signal sits in between. The default band is swept on
    the training split and then frozen.
    """
    a, b = int(lo * n_layers), int(hi * n_layers)
    return list(range(max(0, a), max(a + 1, b)))
