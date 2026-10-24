"""
GRASP as a MUCH `UQSignal`.

Drop this file into the MUCH repository at `src/signals/sig_grasp.py`, add
`SignalGRASP` to `src/signals/__init__.py`, and point `GRASP_SRC` at this
project's `src/` directory so the `grasp` package can be imported. It then
evaluates through the benchmark's own protocol:

    GRASP_SRC=/path/to/ai_project/src python -m src evaluate-signal grasp_gsd_corpus --lang en

Because attention is not shipped with MUCH, extraction has to run once per
generation config on a GPU. `scripts/02_extract_attention.py` does that and
writes a cache; this class reads the cache, so scoring itself is CPU-only and
takes seconds.

MUCH aggregates the per-token signal over each claim with its own aggregators,
so this plugin emits token-level values only. GRASP's saliency-weighted soft-min
claim aggregator is not available through this route; use
`scripts/03_score_and_eval.py` for the native claim score.
"""

from __future__ import annotations

import os
import sys
import typing as t
from functools import cached_property
from pathlib import Path

import numpy as np

try:  # only available when running inside the MUCH repo
    from ..generation import Generation
    from .abc_signal import UQSignal
except ImportError:  # standalone use
    Generation = t.Any  # type: ignore

    class UQSignal:  # type: ignore
        def __init__(self):
            self._all_signal_values_cache = []


try:  # imported as part of the grasp package
    from .score import CorpusNormaliser, apply_boundary_reset, token_signal
except ImportError:  # copied into MUCH's src/signals/: load grasp from GRASP_SRC
    _grasp_src = os.getenv("GRASP_SRC")
    if _grasp_src and _grasp_src not in sys.path:
        sys.path.insert(0, _grasp_src)
    try:
        from grasp.score import CorpusNormaliser, apply_boundary_reset, token_signal
    except ImportError as exc:
        raise ImportError(
            "sig_grasp.py needs the grasp package. Set GRASP_SRC to the project's "
            "src/ directory (the one containing grasp/)."
        ) from exc

DEFAULT_CACHE = os.getenv("GRASP_CACHE", "output/grasp_cache")


class SignalGRASP(UQSignal):
    """GRounding, Attention Sink and disPersion — a training-free, claim-level,
    attention-based uncertainty signal.

    Parameters
    ----------
    alpha
        Weight of the current token in the within-claim recurrence. Set
        `reset=False` to disable the recurrence entirely (ablation).
    use_signals
        (grounding, sink, dispersion) toggles for the ablation table.
    norm_mode
        Token risk normalisation passed to `token_signal`: "corpus" (corpus
        z-scores, as in the GRASP-0 column of scripts/08_claim_table.py) or the
        legacy "per_gen_rank".
    """

    def __init__(
        self,
        alpha: float = 0.6,
        reset: bool = True,
        use_signals: tuple[bool, bool, bool] = (True, True, True),
        norm_mode: str = "corpus",
        cache_dir: str = DEFAULT_CACHE,
    ):
        super().__init__()
        self.alpha = alpha
        self.reset = reset
        self.use_signals = use_signals
        self.norm_mode = norm_mode
        self.cache_dir = Path(cache_dir)

    @classmethod
    def from_env(cls) -> "SignalGRASP":
        return cls(
            alpha=float(os.getenv("PARAM_ALPHA", 0.6)),
            reset=os.getenv("PARAM_RESET", "1") == "1",
            norm_mode=os.getenv("PARAM_NORM", "corpus"),
        )

    @property
    def signal_name(self) -> str:
        flags = "".join(
            c for c, on in zip("gsd", self.use_signals) if on
        ) or "none"
        bits = [f"grasp_{flags}", self.norm_mode]
        if not self.reset:
            bits.append("noreset")
        return "_".join(bits)

    # -- cache -------------------------------------------------------------

    @cached_property
    def normaliser(self) -> CorpusNormaliser:
        path = self.cache_dir / "normaliser.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/02_extract_attention.py first."
            )
        d = np.load(path)
        n = CorpusNormaliser()
        n.mean_, n.std_ = d["mean"], d["std"]
        return n

    def _features(self, generation_id: str) -> np.ndarray:
        path = self.cache_dir / "features" / f"{generation_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"No cached attention features for {generation_id} at {path}."
            )
        return np.load(path)

    # -- the signal --------------------------------------------------------

    def compute_signal_value(self, generation) -> list:
        feats = self._features(generation.id)  # (L, 1, T, 4)
        scores = token_signal(
            feats, self.normaliser, use_signals=self.use_signals, norm_mode=self.norm_mode
        )  # (T,)

        n_tokens = len(generation.output_tokens)
        if scores.shape[0] < n_tokens:
            scores = np.pad(scores, (0, n_tokens - scores.shape[0]), constant_values=0.5)
        scores = scores[:n_tokens]

        if self.reset:
            chunks = getattr(generation, "_token_chunks", None)
            if chunks:
                scores = apply_boundary_reset(scores, chunks, alpha=self.alpha)

        return np.clip(scores, 0.0, 1.0).tolist()
