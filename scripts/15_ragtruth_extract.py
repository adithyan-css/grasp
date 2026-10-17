#!/usr/bin/env python3
"""
GRASP on RAGTruth — GPU extraction for the fresh-benchmark confirmation.

RAGTruth (Niu et al., ACL 2024; MIT) ships responses from several generators
with character-level hallucination spans. Only Mistral-7B-Instruct is open and
accessible here, so this uses its responses (`model == "mistral-7B-instruct"`),
replayed through `mistralai/Mistral-7B-Instruct-v0.1` (assumed to be the version
RAGTruth generated with; v0.2 postdates most of its collection).

For every selected response, one teacher-forced pass with eager attention gives:
  - GRASP attention features for the top-k variance heads per layer, chosen on
    `--calib` unlabelled training responses with the same label-free rule as MUCH;
  - the top-24 next-token probabilities plus the probability of the actual token,
    in MUCH's `topn_prob_dicts` format (so MUCH's CCP code runs on them unchanged).
Responses are split into claims with MUCH's own segmenter (`much_segmentation`);
a claim is hallucinated if its character span overlaps any annotated span.

Then, with the LLM unloaded, MUCH's CCP (delta=24, sigma=8) is computed and timed
on RAGTruth test responses and on MUCH generations (same T4), and real generation
is timed for `--gen-n` RAGTruth prompts against GRASP extraction on the same output.

Nothing here looks at labels except to store them.

    python scripts/15_ragtruth_extract.py --much-root /kaggle/tmp/much \
        --data-dir /kaggle/tmp/ragtruth --out /kaggle/working/results/ragtruth
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp.features import FeatureConfig, _AttentionReducer  # noqa: E402
from grasp.heads import select_heads_variance_topk  # noqa: E402

RAGTRUTH_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{}"
TOPN = 24


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_ragtruth(data_dir: Path, generator: str):
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in ("response.jsonl", "source_info.jsonl"):
        if not (data_dir / f).is_file():
            urllib.request.urlretrieve(RAGTRUTH_URL.format(f), data_dir / f)
    R = [json.loads(l) for l in (data_dir / "response.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    S = {d["source_id"]: d for d in
         (json.loads(l) for l in (data_dir / "source_info.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    R = [r for r in R if r["model"] == generator and r.get("quality", "good") != "truncated"]
    return R, S


def load_llm(name, eager: bool):
    """Shard over all GPUs, leaving headroom on GPU 0: with device_map='auto',
    accelerate moves each layer's full attention tensor back to the input device
    (GPU 0) before our hooks can reduce it."""
    from transformers import AutoModelForCausalLM
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    n = torch.cuda.device_count()
    if n > 1:
        kw["max_memory"] = {0: "7GiB", **{i: "14GiB" for i in range(1, n)}}
    if eager:
        kw["attn_implementation"] = "eager"
    m = AutoModelForCausalLM.from_pretrained(name, **kw)
    m.eval()
    return m


def chat_prompt(tok, text: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def teacher_forced(model, tok, prompt_text, out_ids, cfg):
    """One pass: GRASP features (L, H_kept, T, 4) and top-N probabilities per token."""
    p_ids = tok.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    P = p_ids.size(-1)
    ids = torch.cat([p_ids, torch.tensor([out_ids], dtype=p_ids.dtype)], dim=-1).to(model.device)
    red = _AttentionReducer(model, cfg)
    with red.capture(prompt_len=P):
        out = model(input_ids=ids, output_attentions=True, use_cache=False)
    layers = sorted(red.buffers)
    stacked = [red.buffers[i] for i in layers]
    H = max(x.shape[0] for x in stacked)
    T = max(x.shape[1] for x in stacked)
    feats = np.zeros((len(stacked), H, T, 4), dtype=np.float32)
    for i, a in enumerate(stacked):
        if a.size:
            feats[i, : a.shape[0], : a.shape[1]] = a
    logits = out.logits[0, P - 1 : ids.size(-1) - 1].float()  # row i predicts output token i
    probs = torch.softmax(logits, dim=-1)
    top_p, top_i = probs.topk(TOPN, dim=-1)
    chosen = probs.gather(1, torch.tensor(out_ids, device=probs.device)[:, None])[:, 0]
    return feats, top_i.cpu().numpy().astype(np.int32), top_p.cpu().numpy().astype(np.float32), \
        chosen.cpu().numpy().astype(np.float32)


def topn_dicts(top_i, top_p, chosen, out_ids):
    """MUCH's format: one {token_id: prob} per position, always containing the chosen token."""
    res = []
    for k, tokid in enumerate(out_ids):
        d = {int(a): float(b) for a, b in zip(top_i[k], top_p[k])}
        d.setdefault(int(tokid), float(chosen[k]))
        res.append(d)
    return res


def claims_and_labels(tok, response, spans, segment):
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    out_ids, offs = enc["input_ids"], enc["offset_mapping"]
    chunks = segment(response, llm_tokenizer=tok, precomputed_tokens=out_ids)
    labels, char_spans = [], []
    for c in chunks:
        c = [i for i in c if i < len(offs)]
        if not c:
            labels.append(-1); char_spans.append((0, 0)); continue
        a, b = offs[c[0]][0], offs[c[-1]][1]
        hit = any(a < s["end"] and b > s["start"] for s in spans)
        labels.append(1 if hit else 0)
        char_spans.append((a, b))
    return out_ids, chunks, labels, char_spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generator", default="mistral-7B-instruct")
    ap.add_argument("--hf-model", default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=0, help="0 = all test responses (debug aid)")
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--gen-n", type=int, default=15)
    ap.add_argument("--ccp-test-n", type=int, default=450)
    ap.add_argument("--ccp-much-n", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=2048, help="skip prompt+response longer than this")
    ap.add_argument("--calib-max-tokens", type=int, default=1024,
                    help="calibration (unlabelled head selection) uses only responses up to this length")
    ap.add_argument("--skip-ccp", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from much_segmenter import much_segmentation

    out = args.out
    (out / "features").mkdir(parents=True, exist_ok=True)
    R, S = load_ragtruth(args.data_dir, args.generator)
    rng = np.random.RandomState(0)
    train = [r for r in R if r["split"] == "train"]
    train = [train[i] for i in sorted(rng.choice(len(train), min(args.n_train, len(train)), replace=False))]
    test = [r for r in R if r["split"] == "test"]
    if args.n_test:
        test = test[: args.n_test]
    print(f"RAGTruth {args.generator}: {len(train)} train (sampled), {len(test)} test", flush=True)
    tok = AutoTokenizer.from_pretrained(args.hf_model)

    # ---------------- 1. generation timing (default attention) ----------------
    timing = {}
    gen_outputs = []
    if args.gen_n:
        model = load_llm(args.hf_model, eager=False)
        for k, r in enumerate(test[: args.gen_n + 1]):
            ids = tok.encode(chat_prompt(tok, S[r["source_id"]]["prompt"]), add_special_tokens=False,
                             return_tensors="pt").to(model.device)
            sync(); t0 = time.perf_counter()
            g = model.generate(ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
                               pad_token_id=tok.eos_token_id)
            sync(); dt = time.perf_counter() - t0
            if k:  # first call is warm-up
                gen_outputs.append((r, g[0, ids.size(-1):].tolist(), dt))
        del model; gc.collect(); torch.cuda.empty_cache()
        timing["generation_s"] = [o[2] for o in gen_outputs]
        timing["generation_tokens"] = [len(o[1]) for o in gen_outputs]

    # ---------------- 2. extraction with eager attention ----------------
    model = load_llm(args.hf_model, eager=True)

    # 2a. label-free head selection on unlabelled training responses
    calib_feats = []
    all_heads = FeatureConfig(keep_heads=None, align="current", saliency_norm="per_later_token")
    for r in train:
        if len(calib_feats) >= args.calib:
            break
        out_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        prompt = chat_prompt(tok, S[r["source_id"]]["prompt"])
        if len(tok.encode(prompt, add_special_tokens=False)) + len(out_ids) > args.calib_max_tokens:
            continue
        try:
            f, *_ = teacher_forced(model, tok, prompt, out_ids, all_heads)
            calib_feats.append(f)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
    print(f"head calibration on {len(calib_feats)} unlabelled responses", flush=True)
    heads = select_heads_variance_topk(calib_feats, k=args.top_k)
    del calib_feats; gc.collect()
    (out / "selected_heads.json").write_text(json.dumps({str(k): v for k, v in heads.items()}))
    cfg = FeatureConfig(keep_heads=heads, align="current", saliency_norm="per_later_token")
    (out / "cache_meta.json").write_text(json.dumps({
        "model": args.hf_model, "generator": args.generator, "align": "current",
        "saliency_norm": "per_later_token", "head_select": "variance_topk", "top_k": args.top_k}))

    # 2b. GRASP extraction timing on the generated outputs (overhead on long answers)
    ext = []
    for r, gids, _ in gen_outputs:
        try:
            sync(); t0 = time.perf_counter()
            teacher_forced(model, tok, chat_prompt(tok, S[r["source_id"]]["prompt"]), gids, cfg)
            sync(); ext.append(time.perf_counter() - t0)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); ext.append(float("nan"))
    ext = [e for e in ext if e == e]
    if ext:
        timing["grasp_extract_s"] = ext
        timing["overhead_pct"] = 100 * float(np.mean(ext)) / float(np.mean(timing["generation_s"]))
        print(f"GRASP overhead on RAGTruth-length outputs: {timing['overhead_pct']:.1f}% "
              f"(gen {np.mean(timing['generation_s']):.2f}s, {np.mean(timing['generation_tokens']):.0f} tok; "
              f"extract {np.mean(ext):.2f}s)", flush=True)

    # 2c. features + probabilities for every selected response
    meta = []
    t_start = time.time()
    for k, r in enumerate(train + test):
        fp = out / "features" / f"{r['id']}.npz"
        try:
            out_ids, chunks, labels, cspans = claims_and_labels(tok, r["response"], r["labels"], much_segmentation)
            prompt = chat_prompt(tok, S[r["source_id"]]["prompt"])
            if len(tok.encode(prompt, add_special_tokens=False)) + len(out_ids) > args.max_tokens:
                print(f"  skip {r['id']}: too long", flush=True); continue
            if not fp.is_file():
                f, ti, tp, ch = teacher_forced(model, tok, prompt, out_ids, cfg)
                flat = np.concatenate([np.asarray(c, dtype=np.int32) for c in chunks]) if chunks else np.zeros(0, np.int32)
                np.savez_compressed(fp, feats=f.astype(np.float16), top_i=ti, top_p=tp.astype(np.float16),
                                    chosen=ch, out_ids=np.asarray(out_ids, np.int32),
                                    chunk_flat=flat, chunk_len=np.asarray([len(c) for c in chunks], np.int32))
            meta.append({"id": r["id"], "split": r["split"], "task": S[r["source_id"]]["task_type"],
                         "n_tokens": len(out_ids), "labels": labels, "char_spans": cspans,
                         "response_hallucinated": int(bool(r["labels"]))})
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {r['id']}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if (k + 1) % 100 == 0:
            el = time.time() - t_start
            print(f"  {k + 1}/{len(train) + len(test)}  ({el:.0f}s, {el / (k + 1):.2f}s/resp)", flush=True)
    (out / "meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta), encoding="utf-8")
    n_cl = sum(len(m["labels"]) for m in meta)
    n_h = sum(sum(1 for x in m["labels"] if x == 1) for m in meta)
    print(f"RAGTRUTH-EXTRACT {len(meta)} responses, {n_cl} claims, {n_h} hallucinated", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3. CCP (MUCH's implementation) on the same T4 ----------------
    if not args.skip_ccp:
        sys.path.insert(0, str(args.much_root))
        from src.generation import get_all_generations  # type: ignore
        import src.signals.sig_ccp as sig_ccp_mod  # type: ignore
        from src.signals.sig_ccp import SignalCCP  # type: ignore

        if not torch.cuda.is_available():  # MUCH hard-codes DEVICE=cuda; allow CPU smoke tests
            sig_ccp_mod.DEVICE = torch.device("cpu")
        ccp = SignalCCP(delta=24, sigma=8)
        # 3a. MUCH generations: seconds per generation and per token, per model (T=0.7 configs)
        much_t = {}
        by_cfg = {}
        for g in get_all_generations().values():
            c = g.generation_cfg
            if abs(c.temperature - 0.7) < 1e-6:
                by_cfg.setdefault(c.model_name, []).append(g)
        for m, gl in sorted(by_cfg.items()):
            gl = sorted(gl, key=lambda g: g.id)[: args.ccp_much_n + 1]
            ccp.compute_signal_value(gl[0])  # warm-up (loads NLI model)
            secs, toks = [], []
            for g in gl[1:]:
                sync(); t0 = time.perf_counter()
                ccp.compute_signal_value(g)
                sync(); secs.append(time.perf_counter() - t0); toks.append(len(g.output_tokens))
            much_t[m] = {"n": len(secs), "s_per_gen": float(np.mean(secs)),
                         "s_per_token": float(np.sum(secs) / np.sum(toks)), "mean_tokens": float(np.mean(toks))}
            print(f"CCP-TIMING MUCH {m}: {much_t[m]}", flush=True)
        timing["ccp_much"] = much_t

        # 3b. RAGTruth test responses: CCP values + time
        ccp_vals, ccp_s, ccp_tok = {}, [], []
        test_ids = {m["id"] for m in meta if m["split"] == "test"}
        for r in [r for r in test if r["id"] in test_ids][: args.ccp_test_n]:
            z = np.load(out / "features" / f"{r['id']}.npz")
            out_ids = z["out_ids"].tolist()
            gen = SimpleNamespace(output_tokens=out_ids,
                                  topn_prob_dicts=topn_dicts(z["top_i"], z["top_p"], z["chosen"], out_ids),
                                  generation_cfg=SimpleNamespace(model_name=args.hf_model))
            sync(); t0 = time.perf_counter()
            v = ccp.compute_signal_value(gen)
            sync(); ccp_s.append(time.perf_counter() - t0); ccp_tok.append(len(out_ids))
            ccp_vals[r["id"]] = [float(x) for x in v]
            if len(ccp_s) % 50 == 0:
                print(f"  CCP {len(ccp_s)} responses, {np.mean(ccp_s):.2f}s/resp", flush=True)
        (out / "ccp_test.json").write_text(json.dumps(ccp_vals))
        timing["ccp_ragtruth"] = {"n": len(ccp_s), "s_per_resp": float(np.mean(ccp_s)) if ccp_s else None,
                                  "s_per_token": float(np.sum(ccp_s) / max(np.sum(ccp_tok), 1))}
        if gen_outputs and ccp_s:
            gen_tok_s = float(np.sum(timing["generation_s"]) / np.sum(timing["generation_tokens"]))
            timing["ccp_overhead_pct_ragtruth"] = 100 * timing["ccp_ragtruth"]["s_per_token"] / gen_tok_s
            print(f"CCP-TIMING RAGTruth: {timing['ccp_ragtruth']}  overhead vs generation "
                  f"{timing['ccp_overhead_pct_ragtruth']:.0f}%", flush=True)

    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
