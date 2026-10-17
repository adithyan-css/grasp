"""
GRASP attention-feature extraction.

Given a MUCH `Generation` (prompt + exact generated token ids), run ONE
teacher-forced forward pass with eager attention and reduce the attention maps
to four per-token statistics, per (layer, head):

    G  grounding ratio   attention mass sent back into the prompt region,
                         as a fraction of all non-sink mass
    S  sink leakage      attention mass dumped on the sink positions
    P  dispersion        normalised entropy of the non-sink attention row
    W  saliency          attention mass this token RECEIVES from later tokens

Nothing is sampled: the generated token ids are replayed, so the pass is exact
and deterministic.

Memory: attention for a full model is (L, H, N, N) which does not fit for long
sequences. We install forward hooks that reduce each layer's attention to the
four statistics and then drop the tensor before the next layer runs.
"""

from __future__ import annotations

import contextlib
import typing as t
from dataclasses import dataclass

import numpy as np
import torch

EPS = 1e-9


@dataclass
class FeatureConfig:
    """Knobs for feature extraction. Defaults are the ones reported in the paper."""

    sink_size: int = 1
    """Number of leading positions treated as attention sinks."""

    keep_heads: t.Optional[dict[int, int] | dict[int, list[int]]] = None
    """{layer_idx: head_idx} or {layer_idx: [head_idx, ...]}. When set, only those
    heads are kept per layer (in the given order), which is
    what makes full-corpus extraction cheap. When None, every head is kept
    (used for the head-selection calibration pass only)."""

    dtype: np.dtype = np.float32

    align: str = "legacy"
    """Attention alignment mode:
    - 'legacy': Row P+i (attention FROM token i, which predicts token i+1).
    - 'current': Row P+i-1 (attention predicting token i).
    """

    saliency_norm: str = "none"
    """Saliency normalisation mode:
    - 'none': Raw sum of attention mass from strictly later tokens.
    - 'per_later_token': Divided by the number of strictly later tokens (T - 1 - i).
    """


class _AttentionReducer:
    """Forward hooks that reduce attention to per-token statistics in-flight."""

    def __init__(self, model, cfg: FeatureConfig):
        self.cfg = cfg
        self.model = model
        self.buffers: dict[int, np.ndarray] = {}
        self._handles = []
        self._prompt_len = 0

    # -- lifecycle ---------------------------------------------------------

    def _layers(self):
        for attr in (
            "model.layers",
            "model.model.layers",
            "model.language_model.layers",  # Gemma-3 multimodal wrapper
            "model.language_model.model.layers",
            "language_model.model.layers",
            "transformer.h",
        ):
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        # fallback: the longest non-vision ModuleList of attention-bearing layers
        best = None
        for name, mod in self.model.named_modules():
            if (
                isinstance(mod, torch.nn.ModuleList)
                and name.endswith("layers")
                and "vision" not in name
                and len(mod)
                and any(hasattr(mod[0], n) for n in ("self_attn", "attn", "attention"))
                and (best is None or len(mod) > len(best))
            ):
                best = mod
        if best is None:
            raise RuntimeError("Could not locate decoder layers on this model.")
        return best

    def _attn_module(self, layer):
        for name in ("self_attn", "attn", "attention"):
            if hasattr(layer, name):
                return getattr(layer, name)
        raise RuntimeError("Could not locate the attention module on a decoder layer.")

    @contextlib.contextmanager
    def capture(self, prompt_len: int):
        self.buffers = {}
        self._prompt_len = prompt_len
        for idx, layer in enumerate(self._layers()):
            module = self._attn_module(layer)
            self._handles.append(
                module.register_forward_hook(self._make_hook(idx), with_kwargs=False)
            )
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []

    # -- the hook ----------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            attn = None
            if isinstance(output, (tuple, list)):
                for item in output[1:]:
                    if torch.is_tensor(item) and item.dim() == 4:
                        attn = item
                        break
            if attn is None:
                return output

            self.buffers[layer_idx] = self._reduce(attn, layer_idx)

            # Drop the big tensor so the model does not accumulate it.
            if isinstance(output, tuple):
                return tuple(
                    None if (torch.is_tensor(o) and o.dim() == 4 and i > 0) else o
                    for i, o in enumerate(output)
                )
            return output

        return hook

    # -- the maths ---------------------------------------------------------

    def _reduce(self, attn: torch.Tensor, layer_idx: int) -> np.ndarray:
        """attn: (1, H, N, N) -> (H_kept, T, 4) float32 numpy.

        Rows are restricted to generated positions [P, N) (or [P-1, N-1) if aligned).
        Columns are split into sink [0, s), prompt [s, P) and generated [P, .].
        """
        cfg = self.cfg
        s = cfg.sink_size
        P = self._prompt_len

        a = attn[0]  # (H, N, N); select heads before upcasting to save memory
        if cfg.keep_heads is not None:
            head = cfg.keep_heads.get(layer_idx)
            if head is None:
                return np.zeros((0, 0, 4), dtype=cfg.dtype)
            if isinstance(head, (list, tuple)):
                a = a[list(head)]  # k heads, kept in the given (variance) order
            else:
                a = a[head : head + 1]
        a = a.float()

        H, N, _ = a.shape
        T = N - P
        if T <= 0:
            return np.zeros((H, 0, 4), dtype=cfg.dtype)

        idx = torch.arange(N, device=a.device)

        if cfg.align == "current" and P > 0:
            # Query row P+i-1 is the attention distribution when generating token i
            rows = a[:, (P - 1) : (N - 1), :]  # (H, T, N)
            abs_pos = torch.arange(P - 1, N - 1, device=a.device).unsqueeze(1)  # (T,1)
        else:
            # Legacy: Query row P+i
            rows = a[:, P:, :]  # (H, T, N) attention FROM generated tokens
            abs_pos = torch.arange(P, N, device=a.device).unsqueeze(1)  # (T,1)

        valid = (idx.unsqueeze(0) <= abs_pos).unsqueeze(0)  # (1,T,N)

        sink_m = valid & (idx < s).view(1, 1, -1)
        prompt_m = valid & ((idx >= s) & (idx < P)).view(1, 1, -1)
        nonsink_m = valid & (idx >= s).view(1, 1, -1)

        sink = (rows * sink_m).sum(-1)  # (H,T)
        prompt = (rows * prompt_m).sum(-1)  # (H,T)
        nonsink = (rows * nonsink_m).sum(-1)  # (H,T)

        G = prompt / (nonsink + EPS)
        S = sink

        # normalised entropy over the non-sink support
        p = (rows * nonsink_m) / (nonsink.unsqueeze(-1) + EPS)
        ent = -(p * torch.log(p + EPS)).sum(-1)  # (H,T)
        support = nonsink_m.sum(-1).clamp(min=2).float()  # (1,T)
        D = ent / torch.log(support)

        # saliency: mass each generated token RECEIVES from strictly later
        # generated tokens. cols [P, N) of the generated-row block.
        gen_block = a[:, P:, P:]  # (H, T, T) attention among generated tokens
        strictly_later = torch.tril(
            torch.ones(T, T, device=a.device, dtype=torch.bool), diagonal=-1
        )
        W = (gen_block * strictly_later.unsqueeze(0)).sum(1)  # (H,T)

        if cfg.saliency_norm == "per_later_token":
            denom = (
                torch.arange(T - 1, -1, -1, device=a.device)
                .clamp(min=1)
                .float()
                .unsqueeze(0)
            )
            W = W / denom

        out = torch.stack([G, S, D, W], dim=-1)  # (H,T,4)
        return out.detach().cpu().numpy().astype(cfg.dtype)


FEATURE_NAMES = ("grounding", "sink", "dispersion", "saliency")


@torch.no_grad()
def extract_features(
    model,
    tokenizer,
    prompt_text: str,
    output_token_ids: t.Sequence[int],
    cfg: FeatureConfig | None = None,
) -> np.ndarray:
    """Return an array of shape (n_layers, n_heads_kept, T, 4).

    `prompt_text` must already be chat-templated exactly as MUCH did it — use
    `much.src.generation.template_question` so the prompt length matches the
    generation the logits were recorded for.
    """
    cfg = cfg or FeatureConfig()

    prompt_ids = tokenizer.encode(
        prompt_text, add_special_tokens=False, return_tensors="pt"
    )
    P = prompt_ids.size(-1)
    out_ids = torch.tensor([list(output_token_ids)], dtype=prompt_ids.dtype)
    input_ids = torch.cat([prompt_ids, out_ids], dim=-1).to(model.device)

    reducer = _AttentionReducer(model, cfg)
    with reducer.capture(prompt_len=P):
        model(input_ids=input_ids, output_attentions=True, use_cache=False)

    layers = sorted(reducer.buffers)
    if not layers:
        raise RuntimeError(
            "No attention captured. Load the model with attn_implementation='eager'."
        )
    stacked = [reducer.buffers[i] for i in layers]
    n_heads = max(x.shape[0] for x in stacked)
    T = max(x.shape[1] for x in stacked)
    result = np.zeros((len(stacked), n_heads, T, 4), dtype=cfg.dtype)
    for i, arr in enumerate(stacked):
        if arr.size:
            result[i, : arr.shape[0], : arr.shape[1]] = arr
    return result
