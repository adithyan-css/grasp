#!/usr/bin/env python3
"""
UHeads RETRAINED on the same training data as GRASP (identical-data comparison).

The authors' claim-head architecture and default training recipe
(`configs/run_train_uhead.yaml`: attention history 3 + top-4 probabilities,
head_dim 768, 2 layers, 8 heads, dropout 0.1, pos_weight 3, AdamW lr 1e-4,
weight decay 0.1, 5% linear warm-up, 10 epochs, batch 32, grad-norm 1.0), with one
deviation: the epoch is chosen on a 10% split of TRAINING generations (their
config validates on the test split).

UHeads are model-specific, so one head is trained per backbone:
  MUCH      one head per MUCH model, on that model's train claims (consensus
            GPT-4o labels, EOS chunk dropped); scored on its test claims.
  RAGTruth  one head on RAGTruth's Mistral train responses (same sample as GRASP,
            backbone v0.1 as for GRASP); scored on its test claims.
Teacher-forced features use the band capture verified against the authors'
extractor (scripts/17_uhead_extract.py). Per-claim test logits are saved;
comparison is local (scripts/22_uhead_retrained_compare.py — NOT YET WRITTEN; see
README "Open items").

    python scripts/21_uhead_train.py --much-root /kaggle/tmp/much \
        --uhead-src /kaggle/tmp/llm-uncertainty-head --out /kaggle/working/results/uhead_rt \
        [--ragtruth --rt-data /kaggle/tmp/ragtruth]
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


u17 = _load("u17", "17_uhead_extract.py")

HEAD_CFG = {
    "head_type": "claim",
    "feature_extractor": [
        {"name": "luh.feature_extractors.basic_attention", "layer_nums": "all", "attn_history_sz": 3, "pool": False},
        {"name": "luh.feature_extractors.token_probabilities", "top_n": 4},
    ],
    "uncertainty_head": {"head_dim": 768, "n_layers": 2, "n_heads": 8, "dropout": 0.1},
}
TRAIN = dict(epochs=10, lr=1e-4, weight_decay=0.1, warmup=0.05, batch=32, clip=1.0, pos_weight=3.0, val_frac=0.1)


def text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def build_head(Auto, model):
    from omegaconf import OmegaConf
    return Auto.from_config(OmegaConf.create(HEAD_CFG), SimpleNamespace(config=text_config(model)))


def featurize(model, head, ids, ctx, chunks):
    h, top_n, layers, heads = u17.head_cfg(head)
    X = u17.uhead_features(model, ids, ctx, h, top_n, layers, heads)[0].to(torch.float16)  # (N-1, D)
    M = u17.claim_masks(ids.shape[-1], ctx, chunks).to(torch.uint8)                       # (C, N-1)
    return X, M


def roc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def batches(items, bs, shuffle, rng):
    idx = np.arange(len(items))
    if shuffle:
        rng.shuffle(idx)
    for k in range(0, len(idx), bs):
        yield [items[i] for i in idx[k : k + bs]]


def forward(head, batch, dev):
    """batch: list of dicts {X, M, y?}. Returns list of per-instance claim logits."""
    xm = [(b["X"], b["M"]) for b in batch]  # one access each (lazy items load from disk)
    L = max(x.shape[0] for x, _ in xm)
    D = xm[0][0].shape[1]
    X = torch.zeros(len(batch), L, D, dtype=torch.float32)
    A = torch.zeros(len(batch), L, dtype=torch.long)
    claims = []
    for i, (x, mk) in enumerate(xm):
        n = x.shape[0]
        X[i, :n] = x.float()
        A[i, :n] = 1
        m = torch.zeros(mk.shape[0], L, dtype=torch.long)
        m[:, :n] = mk.long()
        claims.append(m.to(dev))
    out = head._compute_tensors({"claims": claims}, X.to(dev), A.to(dev))  # (B, maxC, 1)
    return [out[i, : mk.shape[0], 0] for i, (_, mk) in enumerate(xm)]


def train_head(head, train_items, dev, seed=0, log=""):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train_items))
    n_val = max(1, int(len(train_items) * TRAIN["val_frac"]))
    val = [train_items[i] for i in order[:n_val]]
    tr = [train_items[i] for i in order[n_val:]]
    head = head.to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    steps = TRAIN["epochs"] * math.ceil(len(tr) / TRAIN["batch"])
    warm = max(1, int(TRAIN["warmup"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([TRAIN["pos_weight"]], device=dev))
    best, best_state, hist = -1.0, None, []
    for ep in range(TRAIN["epochs"]):
        head.train()
        tot = 0.0
        for b in batches(tr, TRAIN["batch"], True, rng):
            logits = torch.cat(forward(head, b, dev))
            y = torch.cat([x["y"] for x in b]).float().to(dev)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), TRAIN["clip"])
            opt.step(); sched.step()
            tot += float(loss) * len(b)
        head.eval()
        with torch.no_grad():
            vs, vy = [], []
            for b in batches(val, TRAIN["batch"], False, rng):
                vs += [float(v) for v in torch.cat(forward(head, b, dev)).cpu()]
                vy += [int(v) for v in torch.cat([x["y"] for x in b])]
        r = roc(vy, vs)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_roc": r})
        print(f"  {log} epoch {ep + 1}: loss {tot / len(tr):.4f}  val ROC {r:.4f}", flush=True)
        if r == r and r > best:
            best = r
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, {"best_val_roc": best, "history": hist, "n_train": len(tr), "n_val": len(val)}


@torch.no_grad()
def predict(head, items, dev):
    head.eval()
    out = []
    for b in batches(items, 16, False, np.random.RandomState(0)):
        for x, lg in zip(b, forward(head, b, dev)):
            out.append([float(v) for v in lg.cpu()])
    return out


# ----------------------------------------------------------------------------- MUCH


def run_much(args, Auto, out: Path):
    sys.path.insert(0, str(args.much_root))
    from src.annotation import get_all_annotations  # type: ignore
    from src.generation import get_tokenizer, template_question  # type: ignore

    anns = [a for a in get_all_annotations().values()
            if "gpt-4o" in a.labels and "gpt-4.1" in a.labels and a.labels["gpt-4o"] == a.labels["gpt-4.1"]]
    is_test = lambda a: any(not k.startswith("gpt-") for k in a.labels)
    models = sorted({a.generation.generation_cfg.model_name for a in anns})
    results = []
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    for mname in models:
        t0 = time.time()
        group = [a for a in anns if a.generation.generation_cfg.model_name == mname]
        if args.limit:
            group = group[: args.limit]
        tok = get_tokenizer(mname)
        model = u17.load_llm(mname)
        head = build_head(Auto, model)
        items = {"train": [], "test": []}
        for a in group:
            g = a.generation
            chunks = [list(c) for c in a.token_chunks][:-1]
            lab = [int(x == -1) for x in a.labels["gpt-4o"][:-1]]
            keep = [(i, c, y) for i, (c, y) in enumerate(zip(chunks, lab)) if c]
            if not keep:
                continue
            p_ids = tok.encode(template_question(g.prompt, tok, brief=g.generation_cfg.brief), add_special_tokens=False)
            ids = torch.tensor([p_ids + list(g.output_tokens)]).to(model.device)
            try:
                X, M = featurize(model, head, ids, len(p_ids), [c for _, c, _ in keep])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            items["test" if is_test(a) else "train"].append(
                {"gen_id": g.id, "chunk": [i for i, _, _ in keep], "X": X, "M": M,
                 "y": torch.tensor([y for _, _, y in keep])})
        del model; gc.collect(); torch.cuda.empty_cache()
        print(f"{mname}: features for {len(items['train'])} train / {len(items['test'])} test "
              f"generations in {time.time() - t0:.0f}s", flush=True)
        head, info = train_head(head, items["train"], dev, log=mname.split("/")[-1])
        preds = predict(head, items["test"], dev)
        for it, p in zip(items["test"], preds):
            for ch, lg, y in zip(it["chunk"], p, it["y"].tolist()):
                results.append({"model": mname, "gen_id": it["gen_id"], "chunk": ch, "label": y, "logit": lg})
        info["test_roc"] = roc([r["label"] for r in results if r["model"] == mname],
                               [r["logit"] for r in results if r["model"] == mname])
        print(f"UHEAD-RETRAINED {mname}: val ROC {info['best_val_roc']:.4f}  test claims "
              f"{sum(r['model'] == mname for r in results)}", flush=True)
        (out / f"train_log_{mname.split('/')[-1]}.json").write_text(json.dumps(info, indent=2))
        del head, items; gc.collect(); torch.cuda.empty_cache()
    (out / "uhead_retrained_much.json").write_text(json.dumps(results))


# ----------------------------------------------------------------------------- RAGTruth


def run_ragtruth(args, Auto, out: Path):
    rt = _load("rt15", "15_ragtruth_extract.py")
    from transformers import AutoTokenizer
    from much_segmenter import much_segmentation

    R, S = rt.load_ragtruth(args.rt_data, "mistral-7B-instruct")
    rng = np.random.RandomState(0)  # the same train sample as 15_ragtruth_extract.py
    train = [r for r in R if r["split"] == "train"]
    train = [train[i] for i in sorted(rng.choice(len(train), min(1500, len(train)), replace=False))]
    test = [r for r in R if r["split"] == "test"]
    if args.limit:
        train, test = train[: args.limit], test[: args.limit]
    name = "mistralai/Mistral-7B-Instruct-v0.1"
    tok = AutoTokenizer.from_pretrained(name)
    model = u17.load_llm(name)
    head = build_head(Auto, model)
    cache = args.cache_dir or (out / "rt_cache")
    cache.mkdir(parents=True, exist_ok=True)
    items = {"train": [], "test": []}
    t0 = time.time()
    for split, group in (("train", train), ("test", test)):
        for r in group:
            out_ids, chunks, labels, _ = rt.claims_and_labels(tok, r["response"], r["labels"], much_segmentation)
            p_ids = tok.encode(rt.chat_prompt(tok, S[r["source_id"]]["prompt"]), add_special_tokens=False)
            if len(p_ids) + len(out_ids) > args.max_tokens:
                continue
            keep = [(i, c, l) for i, (c, l) in enumerate(zip(chunks, labels)) if l >= 0 and c]
            if not keep:
                continue
            f = cache / f"{r['id']}.pt"
            if not f.is_file():
                try:
                    X, M = featurize(model, head, torch.tensor([p_ids + list(out_ids)]).to(model.device),
                                     len(p_ids), [c for _, c, _ in keep])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); continue
                torch.save({"X": X, "M": M}, f)
            items[split].append({"id": r["id"], "chunk": [i for i, _, _ in keep], "file": f,
                                 "y": torch.tensor([l for _, _, l in keep])})
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"RAGTruth: features for {len(items['train'])} train / {len(items['test'])} test in {time.time() - t0:.0f}s",
          flush=True)

    class Lazy(dict):  # loads features from disk when the trainer touches X / M (cached per item pair)
        def __getitem__(self, k):
            if k in ("X", "M") and k not in self.keys():
                if "_buf" not in self.keys() or self["_buf"] is None:
                    dict.__setitem__(self, "_buf", torch.load(super().__getitem__("file")))
                v = super().__getitem__("_buf")[k]
                if k == "M":  # X is always read first in forward(); release after M
                    dict.__setitem__(self, "_buf", None)
                return v
            return super().__getitem__(k)

    tr_items = [Lazy(x) for x in items["train"]]
    te_items = [Lazy(x) for x in items["test"]]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    head, info = train_head(head, tr_items, dev, log="RAGTruth")
    preds = predict(head, te_items, dev)
    rows = []
    for it, p in zip(te_items, preds):
        for ch, lg, y in zip(it["chunk"], p, it["y"].tolist()):
            rows.append({"id": it["id"], "chunk": ch, "label": y, "logit": lg})
    (out / "uhead_retrained_ragtruth.json").write_text(json.dumps(rows))
    (out / "train_log_ragtruth.json").write_text(json.dumps(info, indent=2))
    print(f"UHEAD-RETRAINED RAGTruth: val ROC {info['best_val_roc']:.4f}, test claims {len(rows)}", flush=True)


def selftest(uhead_src: Path):
    """CPU: build a claim head on Llama-3.1-8B dims and check the training loop learns a
    planted signal (no LLM needed)."""
    Auto = u17.import_luh(uhead_src)
    from omegaconf import OmegaConf
    from transformers import AutoConfig
    cfg = SimpleNamespace(config=AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct"))
    head = Auto.from_config(OmegaConf.create(HEAD_CFG), cfg)
    D = head.feature_extractor.feature_dim()
    g = torch.Generator().manual_seed(0)
    items = []
    for _ in range(120):
        N = int(torch.randint(20, 40, (1,), generator=g))
        X = torch.randn(N, D, generator=g) * 0.1
        M = torch.zeros(3, N, dtype=torch.uint8)
        y = torch.randint(0, 2, (3,), generator=g)
        for c in range(3):
            pos = slice(5 + 4 * c, 8 + 4 * c)
            M[c, pos] = 1
            X[pos, 0] += 2.0 * y[c]  # planted signal
        items.append({"X": X.half(), "M": M, "y": y})
    global TRAIN
    TRAIN = dict(TRAIN, epochs=4)
    head, info = train_head(head, items[:100], "cpu", log="selftest")
    preds = predict(head, items[100:], "cpu")
    r = roc([int(v) for it in items[100:] for v in it["y"]], [v for p in preds for v in p])
    assert r > 0.9, r
    print(f"ok  retraining loop learns a planted signal (held-out ROC {r:.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path)
    ap.add_argument("--uhead-src", type=Path, required=True)
    ap.add_argument("--rt-data", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ragtruth", action="store_true", help="also train/score the RAGTruth head")
    ap.add_argument("--skip-much", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--cache-dir", type=Path, default=None, help="RAGTruth feature cache (keep off the output dir)")
    ap.add_argument("--limit", type=int, default=0, help="debug aid")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        selftest(args.uhead_src)
        return 0
    Auto = u17.import_luh(args.uhead_src)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({"head": HEAD_CFG, "train": TRAIN}, indent=2))
    if not args.skip_much:
        run_much(args, Auto, args.out)
    if args.ragtruth:
        run_ragtruth(args, Auto, args.out)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
