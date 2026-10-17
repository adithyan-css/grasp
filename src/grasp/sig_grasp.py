"""
GRASP as a MUCH `UQSignal`.

Drop this file into the MUCH repository at `src/signals/sig_grasp.py` and add
`SignalGRASP` to `src/signals/__init__.py`. It then evaluates through the
benchmark's own protocol:

    python -m src evaluate-signal grasp --lang en

Because attention is not shipped with MUCH, extraction has to run once per
generation config on a GPU. `scripts/02_extract_attention.py` does that and
writes a cache; this class reads the cache, so scoring itself is CPU-only and
takes seconds.
"""

from __future__ import annotations

import os
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


from .score import CorpusNormaliser, apply_boundary_reset, token_signal

DEFAULT_CACHE = os.getenv("GRASP_CACHE", "output/grasp_cache")


class SignalGRASP(UQSignal):
    """GRounding, Attention Sink and disPersion — a training-free, claim-level,
    attention-based uncertainty signal.

    Parameters
    ----------
    tau
        Sharpness of the saliency-weighted soft-min claim aggregator. Swept on
        the training split and then frozen.
    alpha
        Weight of the current token in the within-claim recurrence. Set
        `reset=False` to disable the recurrence entirely (ablation).
    use_signals
        (grounding, sink, dispersion) toggles for the ablation table.
    """

    def __init__(
        self,
        tau: float = 8.0,
        alpha: float = 0.6,
        reset: bool = True,
        use_saliency: bool = True,
        use_signals: tuple[bool, bool, bool] = (True, True, True),
        cache_dir: str = DEFAULT_CACHE,
    ):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.reset = reset
        self.use_saliency = use_saliency
        self.use_signals = use_signals
        self.cache_dir = Path(cache_dir)

    @classmethod
    def from_env(cls) -> "SignalGRASP":
        return cls(
            tau=float(os.getenv("PARAM_TAU", 8.0)),
            alpha=float(os.getenv("PARAM_ALPHA", 0.6)),
            reset=os.getenv("PARAM_RESET", "1") == "1",
            use_saliency=os.getenv("PARAM_SALIENCY", "1") == "1",
        )

    @property
    def signal_name(self) -> str:
        flags = "".join(
            c for c, on in zip("gsd", self.use_signals) if on
        ) or "none"
        bits = [f"grasp_{flags}", f"t{self.tau:g}"]
        if not self.reset:
            bits.append("noreset")
        if not self.use_saliency:
            bits.append("nosal")
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
            feats, self.normaliser, use_signals=self.use_signals
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
