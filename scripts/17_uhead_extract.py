#!/usr/bin/env python3
"""
UHeads (Shelmanov et al., EMNLP 2025) — run the authors' PRE-TRAINED claim-level
uncertainty heads on exactly the claims GRASP is evaluated on.

  MUCH      head `llm-uncertainty-head/uhead_claim_Llama-3.1-8B-Instruct` on the two
            Llama-3.1-8B-Instruct configs: all test generations, plus `--n-train`
            train generations used ONLY to fix the score's sign.
  RAGTruth  heads `uhead_claim_exp5/exp6_Mistral-7B-Instruct-v0.2` (their only
            Mistral heads; backbone v0.2) on RAGTruth's Mistral responses: all test
            responses, plus `--rt-n-train` train responses used only to pick the
            sign and, of the two heads, the one with higher train ROC.

Inference is teacher-forced, mirroring the authors' training path
(`CausalLMWithUncertaintyLayerClaim.forward` + `run_train_uhead` collator):
features = [attention of every head of every layer from token i to tokens i..i-h+1]
          ++ [top-4 next-token probabilities, zero inside the prompt]
for positions 0..N-2, and each claim is a 0/1 mask over those positions
(`mask[context_len + token_idx] = 1`, then `mask[1:]`). The attention band is
captured with forward hooks so the full N x N maps are never kept;
`test_uhead_features` (in this file) checks it against the authors' own extractor.

The claim head returns one logit per claim (its transformer pools the claim's tokens);
raw logits are saved. No GRASP or CCP score is computed here; comparison is local.

    python scripts/17_uhead_extract.py --much-root /kaggle/tmp/much \
        --uhead-src /kaggle/tmp/llm-uncertainty-head --rt-data /kaggle/tmp/ragtruth \
        --out /kaggle/working/results/uhead
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
from grasp.features import _AttentionReducer  # noqa: E402  (reused for layer lookup)


# ----------------------------------------------------------------------------- UHead import


def import_luh(uhead_src: Path):
    """Import the authors' package without lm-polygraph: its __init__ imports four
    lm-polygraph names used only by generation-time components we do not call."""
    names = {
        "lm_polygraph.estimators.estimator": "Estimator",
        "lm_polygraph.stat_calculators.stat_calculator": "StatCalculator",
        "lm_polygraph.utils.model": "Model",
        "lm_polygraph.stat_calculators.extract_claims": "Claim",
    }
    for full, cls in names.items():
        parts = full.split(".")
        for k in range(1, len(parts) + 1):
            mod = ".".join(parts[:k])
            sys.modules.setdefault(mod, types.ModuleType(mod))
        setattr(sys.modules[full], cls, type(cls, (), {"__init__": lambda self, *a, **k: None}))
    import huggingface_hub  # newer versions dropped `Repository`, used only by push_to_hub
    if not hasattr(huggingface_hub, "Repository"):
        huggingface_hub.Repository = type("Repository", (), {})
    _dl = huggingface_hub.hf_hub_download  # newer versions dropped `use_auth_token`

    def _hf_hub_download(*a, use_auth_token=None, **k):
        return _dl(*a, **k)

    huggingface_hub.hf_hub_download = _hf_hub_download
    sys.path.insert(0, str(uhead_src))
    from luh.auto_uncertainty_head import AutoUncertaintyHead  # noqa: E402
    return AutoUncertaintyHead


# ----------------------------------------------------------------------------- features


class BandCapture:
    """Forward hooks keeping, per layer, attention from each query i to keys
    i, i-1, ..., i-h+1 (zeros where the key index is negative): (H, N, h)."""

    def __init__(self, model, h: int):
        self.h = h
        self.bands: dict[int, torch.Tensor] = {}
        red = _AttentionReducer.__new__(_AttentionReducer)
        red.model = model
        self.layers = list(red._layers())
        self._attn = red._attn_module
        self._handles = []

    def __enter__(self):
        self.bands = {}
        for li, layer in enumerate(self.layers):
            self._handles.append(self._attn(layer).register_forward_hook(self._hook(li)))
        return self

    def __exit__(self, *exc):
        for hd in self._handles:
            hd.remove()
        self._handles = []

    def _hook(self, li):
        def hook(module, args, output):
            attn = next((o for o in output[1:] if torch.is_tensor(o) and o.dim() == 4), None)
            if attn is None:
                return output
            a = attn[0]  # (H, N, N)
            H, N, _ = a.shape
            band = torch.zeros(H, N, self.h, dtype=torch.float32, device=a.device)
            for k in range(self.h):
                if k < N:
                    band[:, k:, k] = torch.diagonal(a, offset=-k, dim1=-2, dim2=-1).float()
            self.bands[li] = band.cpu()
            return tuple(None if (torch.is_tensor(o) and o.dim() == 4 and i > 0) else o
                         for i, o in enumerate(output))
        return hook


def uhead_features(model, ids: torch.Tensor, context_len: int, h: int, top_n: int,
                   layer_nums, head_nums) -> torch.Tensor:
    """Teacher-forced features (1, N-1, D) in the authors' layout:
    attention block reshaped from (N-1, h, heads, layers), then top-n probabilities."""
    with torch.no_grad(), BandCapture(model, h) as cap:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True,
                    use_cache=False)
    N = ids.shape[-1]
    per_layer = []
    for l in layer_nums:
        b = cap.bands[l][head_nums[l]]          # (heads, N, h)
        per_layer.append(b[:, : N - 1, :].permute(1, 2, 0))  # (N-1, h, heads)
    attn_feat = torch.stack(per_layer, dim=-1).reshape(1, N - 1, -1)  # (1, N-1, h*heads*layers)
    logits = out.logits[0, :-1].float()
    probs = torch.zeros(N - 1, top_n)
    rows = slice(context_len - 1, N - 1)
    probs[rows] = torch.softmax(logits[rows], dim=-1).topk(top_n, dim=-1)[0].cpu()
    return torch.cat([attn_feat, probs[None]], dim=-1)


def claim_masks(N: int, context_len: int, chunks) -> torch.Tensor:
    m = torch.zeros(len(chunks), N, dtype=torch.long)
    for j, c in enumerate(chunks):
        idx = [context_len + t for t in c if context_len + t < N]
        m[j, idx] = 1
    return m[:, 1:]


def head_cfg(uhead):
    fe = uhead.feature_extractor._feature_extractors
    att = next(f for f in fe if hasattr(f, "_attn_history_sz"))
    tp = next(f for f in fe if hasattr(f, "top_n"))
    return att._attn_history_sz, tp.top_n, att._layer_nums, att._head_nums


@torch.no_grad()
def score_claims(uhead, model, ids, context_len, chunks):
    h, top_n, layers, heads = head_cfg(uhead)
    X = uhead_features(model, ids, context_len, h, top_n, layers, heads)
    dev = next(uhead.parameters()).device
    llm_inputs = {"claims": [claim_masks(ids.shape[-1], context_len, chunks).to(dev)]}
    logits = uhead._compute_tensors(llm_inputs, X.to(dev), torch.ones(1, X.shape[1], device=dev))
    return [float(x) for x in logits[0, : len(chunks), 0].cpu()]


# ----------------------------------------------------------------------------- runs


def load_llm(name, eager=True):
    from transformers import AutoModelForCausalLM
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    if torch.cuda.device_count() > 1:
        kw["max_memory"] = {0: "7GiB", **{i: "14GiB" for i in range(1, torch.cuda.device_count())}}
    m = AutoModelForCausalLM.from_pretrained(name, **kw)
    m.eval()
    return m


def run_much(args, Auto, out: Path):
    sys.path.insert(0, str(args.much_root))
    from src.annotation import get_all_annotations  # type: ignore
    from src.generation import get_tokenizer, template_question  # type: ignore

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    anns = [a for a in get_all_annotations().values()
            if a.generation.generation_cfg.model_name == model_name
            and "gpt-4o" in a.labels and "gpt-4.1" in a.labels and a.labels["gpt-4o"] == a.labels["gpt-4.1"]]
    is_test = lambda a: any(not k.startswith("gpt-") for k in a.labels)
    test = [a for a in anns if is_test(a)]
    train = sorted([a for a in anns if not is_test(a)], key=lambda a: a.generation.id)
    rng = np.random.RandomState(0)
    train = [train[i] for i in sorted(rng.choice(len(train), min(args.n_train, len(train)), replace=False))]
    print(f"MUCH Llama-3.1-8B: {len(test)} test, {len(train)} train (sign only)", flush=True)

    tok = get_tokenizer(model_name)
    model = load_llm(model_name)
    uhead = Auto.from_pretrained("llm-uncertainty-head/uhead_claim_Llama-3.1-8B-Instruct", base_model=model)
    uhead = uhead.to("cuda:0" if torch.cuda.is_available() else "cpu").eval()
    rows = []
    t0 = time.time()
    for split, group in (("train", train), ("test", test)):
        for a in group:
            g = a.generation
            prompt = template_question(g.prompt, tok, brief=g.generation_cfg.brief)
            p_ids = tok.encode(prompt, add_special_tokens=False)
            ids = torch.tensor([p_ids + list(g.output_tokens)]).to(model.device)
            chunks = [list(c) for c in a.token_chunks][:-1]  # MUCH: drop trailing EOS chunk
            try:
                logits = score_claims(uhead, model, ids, len(p_ids), chunks)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); print(f"  skip {g.id}: OOM", flush=True); continue
            lab = a.labels["gpt-4o"][:-1]
            for i, (lg, y) in enumerate(zip(logits, lab)):
                rows.append({"gen_id": g.id, "chunk": i, "split": split, "label": int(y == -1), "uhead_logit": lg})
    (out / "uhead_much_llama31.json").write_text(json.dumps(rows))
    print(f"UHEAD-MUCH {len(rows)} claims in {time.time() - t0:.0f}s", flush=True)
    del model, uhead; gc.collect(); torch.cuda.empty_cache()


def run_ragtruth(args, Auto, out: Path):
    spec = importlib.util.spec_from_file_location("rt15", HERE / "15_ragtruth_extract.py")
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    from transformers import AutoTokenizer
    from much_segmenter import much_segmentation

    R, S = rt.load_ragtruth(args.rt_data, "mistral-7B-instruct")
    seg_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")  # as in 15
    test = [r for r in R if r["split"] == "test"]
    tr = [r for r in R if r["split"] == "train"]
    rng = np.random.RandomState(1)
    tr = [tr[i] for i in sorted(rng.choice(len(tr), min(args.rt_n_train, len(tr)), replace=False))]

    backbone = "mistralai/Mistral-7B-Instruct-v0.2"
    tok = AutoTokenizer.from_pretrained(backbone)
    model = load_llm(backbone)
    rows = []
    for head_name in ("uhead_claim_exp5_Mistral-7B-Instruct-v0.2", "uhead_claim_exp6_Mistral-7B-Instruct-v0.2"):
        uhead = Auto.from_pretrained(f"llm-uncertainty-head/{head_name}", base_model=model)
        uhead = uhead.to("cuda:0" if torch.cuda.is_available() else "cpu").eval()
        t0 = time.time()
        for split, group in (("train", tr), ("test", test)):
            for r in group:
                out_ids, chunks, labels, _ = rt.claims_and_labels(seg_tok, r["response"], r["labels"], much_segmentation)
                prompt = rt.chat_prompt(tok, S[r["source_id"]]["prompt"])
                p_ids = tok.encode(prompt, add_special_tokens=False)
                if len(p_ids) + len(out_ids) > args.max_tokens:
                    continue
                ids = torch.tensor([p_ids + list(out_ids)]).to(model.device)
                keep = [(i, c) for i, (c, l) in enumerate(zip(chunks, labels)) if l >= 0 and c]
                if not keep:
                    continue
                try:
                    logits = score_claims(uhead, model, ids, len(p_ids), [c for _, c in keep])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); print(f"  skip {r['id']}: OOM", flush=True); continue
                for (i, _), lg in zip(keep, logits):
                    rows.append({"head": head_name, "id": r["id"], "chunk": i, "split": split,
                                 "label": labels[i], "uhead_logit": lg})
        print(f"UHEAD-RAGTRUTH {head_name}: done in {time.time() - t0:.0f}s", flush=True)
        del uhead; gc.collect(); torch.cuda.empty_cache()
    (out / "uhead_ragtruth_mistral.json").write_text(json.dumps(rows))
    print(f"UHEAD-RAGTRUTH {len(rows)} claim scores", flush=True)


def test_uhead_features(uhead_src: Path, model_name="HuggingFaceTB/SmolLM2-135M-Instruct"):
    """CPU check: our band features == the authors' extractor (training path)."""
    import_luh(uhead_src)
    from luh.feature_extractors.basic_attention import FeatureExtractorBasicAttention
    from luh.feature_extractors.combined import FeatureExtractorCombined
    from luh.feature_extractors.token_probabilities import FeatureExtractorTokenProbabilities
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, attn_implementation="eager").eval()
    ids = torch.randint(5, 1000, (1, 23))
    ctx = 9
    for h in (2, 5):
        fe = FeatureExtractorCombined(FeatureExtractorBasicAttention(m, "all", h, False),
                                      FeatureExtractorTokenProbabilities(m, 4))
        with torch.no_grad():
            o = m(input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True)
        o.context_lengths = torch.tensor([ctx])
        ref = fe({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, o)
        att = fe._feature_extractors[0]
        mine = uhead_features(m, ids, ctx, h, 4, att._layer_nums, att._head_nums)
        assert ref.shape == mine.shape, (ref.shape, mine.shape)
        assert torch.allclose(ref.float(), mine.float(), atol=1e-5), float((ref - mine).abs().max())
    print("ok  UHead features match the authors' extractor (h=2 and h=5)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path)
    ap.add_argument("--uhead-src", type=Path, required=True)
    ap.add_argument("--rt-data", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--rt-n-train", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--skip-much", action="store_true")
    ap.add_argument("--skip-ragtruth", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        test_uhead_features(args.uhead_src)
        return 0
    Auto = import_luh(args.uhead_src)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_much:
        run_much(args, Auto, args.out)
    if not args.skip_ragtruth:
        run_ragtruth(args, Auto, args.out)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
