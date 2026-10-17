#!/usr/bin/env python3
"""
GRASP on RAGTruth — fresh-benchmark confirmation of the frozen GRASP-v2.

Pre-registered (written before any RAGTruth feature was extracted):

  Data      RAGTruth Mistral-7B-Instruct responses, claims from MUCH's segmenter,
            label = claim overlaps an annotated hallucination span. Train rows are
            a seeded sample of RAGTruth's train split; test = RAGTruth's test split.
  Variants  (no tuning of any kind on RAGTruth)
    GRASP-v2 (MUCH-trained)  exactly the frozen v2 recipe and fit: trained on MUCH's
                             train split only, applied zero-shot — new generator,
                             new task family (RAG), new labelling scheme.
    GRASP-v2-free            the frozen training-free score; per-model ranks fitted
                             on RAGTruth train features (unlabelled).
    GRASP-v2 (RAGTruth-trained)  the same frozen recipe (feature set, learner,
                             hyper-parameters) refitted on RAGTruth train claims.
  Baselines CCP (MUCH's implementation, delta=24, sigma=8, stopword-masked,
            `product` aggregator); max token surprisal; mean token entropy;
            claim length. All on the identical claims.
  Metrics   claim-level ROC / PR / TPR@10%FPR with 95% bootstrap CIs resampled by
            response and paired ΔROC vs CCP; per task type; response level
            (response score = max over its claims, label = any annotated span).

    python scripts/16_ragtruth_eval.py --rt-dir results/ragtruth \
        --much-table results/dev/claims_k4.npz --frozen results/dev/frozen_v2_k4.json \
        --much-root ../much --out results/ragtruth_eval.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
from grasp.much_eval import aggregate, extended_stopwords_set, mask_stopwords  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fe2 = _load("final_eval_v2", "13_final_eval_v2.py")
dev, fe1 = fe2.dev, fe2.fe1
ct = _load("claim_table", "08_claim_table.py")
RT_MODEL = "RAGTruth/mistral-7B-instruct"


def build_ragtruth_rows(rt_dir: Path, tok, n_heads: int):
    meta = [json.loads(l) for l in (rt_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ccp = json.loads((rt_dir / "ccp_test.json").read_text()) if (rt_dir / "ccp_test.json").is_file() else {}
    files = [rt_dir / "features" / f"{m['id']}.npz" for m in meta]
    # unlabelled per-(layer, head) normaliser over all RAGTruth responses, as 08 does per config
    s1 = s2 = None
    n = 0
    for f in files:
        x = np.load(f)["feats"].astype(np.float64)[..., :3]
        if x.shape[2] == 0:
            continue
        a, b = x.sum(axis=2), (x ** 2).sum(axis=2)
        s1 = a if s1 is None else s1 + a
        s2 = b if s2 is None else s2 + b
        n += x.shape[2]
    hmean = s1 / n
    hstd = np.sqrt(np.maximum(s2 / n - hmean ** 2, 0)) + 1e-9
    stop = extended_stopwords_set()

    rows, y, ccp_claim, rid, split, task, chunk_idx, resp_lab = [], [], [], [], [], [], [], []
    for m, f in zip(meta, files):
        z = np.load(f)
        feats = z["feats"].astype(np.float64)
        assert feats.shape[1] == n_heads, (feats.shape, n_heads)
        out_ids = z["out_ids"].tolist()
        L, T = feats.shape[0], feats.shape[2]
        lens = z["chunk_len"].tolist()
        flat = z["chunk_flat"].tolist()
        chunks, k = [], 0
        for ln in lens:
            chunks.append(flat[k : k + ln]); k += ln
        zz = (feats[..., :3] - hmean[:, :, None, :]) / hstd[:, :, None, :]
        W = feats[:, 0, :, 3].mean(axis=0)
        topn = []
        for i, t in enumerate(out_ids):
            d = {int(a): float(b) for a, b in zip(z["top_i"][i], z["top_p"][i])}
            d.setdefault(int(t), float(z["chosen"][i]))
            topn.append(d)
        tp = ct.token_prob_stats(out_ids, topn)
        keep = ct.content_mask(out_ids[:T], tok, stop)
        bins = ct.depth_bins(L)
        cv = None
        if m["id"] in ccp:
            cv = mask_stopwords(np.asarray(ccp[m["id"]]), out_ids, tok)
        for i, (c, lab) in enumerate(zip(chunks, m["labels"])):
            if lab < 0 or not c:
                continue
            fr = ct.claim_features(zz, W, tp, c, keep, bins, T)
            if fr is None:
                continue
            rows.append(fr)
            y.append(lab)
            ccp_claim.append(aggregate(cv[c[0] : c[-1] + 1], "product") if cv is not None else np.nan)
            rid.append(m["id"]); split.append("rt_" + m["split"]); task.append(m["task"])
            chunk_idx.append(i); resp_lab.append(m["response_hallucinated"])
    return {
        "X": np.vstack(rows), "y": np.asarray(y), "ccp": np.asarray(ccp_claim, dtype=float),
        "gen_id": np.asarray(rid), "split": np.asarray(split), "task": np.asarray(task),
        "chunk": np.asarray(chunk_idx), "resp_label": np.asarray(resp_lab),
    }


def response_level(y_resp_by_id, scores, rid):
    ids = np.unique(rid)
    lab = np.asarray([y_resp_by_id[i] for i in ids])
    out = {}
    for name, s in scores.items():
        agg = np.asarray([np.nanmax(s[rid == i]) for i in ids])
        out[name] = round(fe1.roc(lab, agg), 4)
    out["_n_responses"] = int(len(ids))
    out["_hallucinated"] = int(lab.sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt-dir", type=Path, required=True)
    ap.add_argument("--much-table", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--hf-model", default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    frozen = json.loads(args.frozen.read_text())
    key = frozen["probe_no_ccp"]["key"]
    much = dev.load(args.much_table, split=None)
    n_heads = sum(1 for n in much["names"] if n.startswith("b0_") and n.endswith("_grounding_mean"))
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    rt = build_ragtruth_rows(args.rt_dir, tok, n_heads)
    print(f"RAGTruth claims: {len(rt['y'])} ({(rt['split'] == 'rt_train').sum()} train, "
          f"{(rt['split'] == 'rt_test').sum()} test), hallucinated {rt['y'].mean():.3f}")

    # one combined table so the frozen recipe (build_X / fit_score) applies unchanged
    d = {
        "names": much["names"],
        "X": np.vstack([much["X"], rt["X"]]),
        "y": np.concatenate([much["y"], rt["y"]]),
        "ccp": np.concatenate([much["ccp"], rt["ccp"]]),
        "model": np.concatenate([much["model"], np.full(len(rt["y"]), RT_MODEL)]),
        "split": np.concatenate([np.where(much["split"] == "train", "train", "much_test"), rt["split"]]),
        "gen_id": np.concatenate([much["gen_id"], rt["gen_id"]]),
    }
    much_tr = d["split"] == "train"
    rt_tr, rt_te = d["split"] == "rt_train", d["split"] == "rt_test"
    names = d["names"]
    col = {n: names.index(n) for n in ("tp_nll_max", "tp_ent_mean", "len_tokens")}

    s_all = {
        "CCP": d["ccp"][rt_te],
        "max surprisal": d["X"][rt_te, col["tp_nll_max"]],
        "mean entropy": d["X"][rt_te, col["tp_ent_mean"]],
        "claim length": d["X"][rt_te, col["len_tokens"]],
        "GRASP-v2-free": fe2.training_free_score(d, frozen["training_free_features"], much_tr | rt_tr)[rt_te],
        "GRASP-v2 (MUCH-trained, zero-shot)": fe2.fit_score(d, key, much_tr, rt_te),
        "GRASP-v2 (RAGTruth-trained)": fe2.fit_score(d, key, rt_tr, rt_te),
    }
    y = d["y"][rt_te]
    g = d["gen_id"][rt_te]
    task = rt["task"][rt["split"] == "rt_test"]
    has_ccp = ~np.isnan(s_all["CCP"])
    print(f"CCP available for {has_ccp.sum()} of {len(y)} test claims")
    sub = {k: v[has_ccp] for k, v in s_all.items()}

    out = {"frozen_key": key, "n_heads": n_heads, "claim_level": fe1.boot(y[has_ccp], sub, g[has_ccp]),
           "by_task": {}, "response_level": {}}
    for t in sorted(set(task)):
        mk = has_ccp & (task == t)
        if y[mk].min() != y[mk].max():
            out["by_task"][t] = fe1.boot(y[mk], {k: v[mk] for k, v in s_all.items()}, g[mk])
    resp_lab = {i: l for i, l in zip(rt["gen_id"], rt["resp_label"])}
    out["response_level"] = response_level(resp_lab, sub, g[has_ccp])

    n = out["claim_level"]["_n"]
    print(f"\nRAGTruth TEST (claim level): {n['claims']} claims, {n['hallucinated']} hallucinated, "
          f"{n['generations']} responses")
    for k in s_all:
        r = out["claim_level"][k]
        extra = (f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}"
                 if "roc_diff_vs_ccp" in r else "")
        print(f"  {k:36} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}  "
              f"TPR@10={r['tpr_at_10fpr']:.3f}{extra}")
    print("\nby task (ROC):")
    for t, rr in out["by_task"].items():
        short = {"GRASP-v2 (MUCH-trained, zero-shot)": "GRASP-v2[MUCH]",
                 "GRASP-v2 (RAGTruth-trained)": "GRASP-v2[RT]"}
        print(f"  {t:10} n={rr['_n']['claims']:5} " + "  ".join(
            f"{short.get(k, k)}={rr[k]['roc_auc']:.3f}" for k in s_all))
    print("\nresponse level (ROC, max over claims):", out["response_level"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
