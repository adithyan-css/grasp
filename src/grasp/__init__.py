"""GRASP — training-free, claim-level, attention-based uncertainty for LLMs."""

from .features import FEATURE_NAMES, FeatureConfig, extract_features
from .much_eval import AGGREGATORS, aggregate, evaluate as much_evaluate, mask_stopwords, metrics
from .heads import (
    DISPERSION,
    GROUNDING,
    SALIENCY,
    SINK,
    middle_layers,
    select_heads,
    select_heads_fixed,
    select_heads_random,
    select_heads_variance,
)
from .score import (
    CorpusNormaliser,
    apply_boundary_reset,
    claim_scores,
    saliency_weights,
    softmin_claim_score,
    token_signal,
)

__all__ = [
    "AGGREGATORS",
    "aggregate",
    "much_evaluate",
    "mask_stopwords",
    "metrics",
    "FeatureConfig",
    "extract_features",
    "FEATURE_NAMES",
    "select_heads",
    "select_heads_variance",
    "select_heads_random",
    "select_heads_fixed",
    "middle_layers",
    "GROUNDING",
    "SINK",
    "DISPERSION",
    "SALIENCY",
    "CorpusNormaliser",
    "token_signal",
    "claim_scores",
    "softmin_claim_score",
    "saliency_weights",
    "apply_boundary_reset",
]

__version__ = "0.1.0"
