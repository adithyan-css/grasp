#!/usr/bin/env python3
"""
GRASP vs the authors' pre-trained UHeads, on identical claims.

Pre-registered (written before any UHead test score was computed):

  Sign      UHead logits are oriented on TRAIN claims only (flip if train ROC < 0.5).
  Head      on RAGTruth, of the two pre-trained Mistral heads (exp5, exp6), the one with
            the higher oriented train ROC is used.
  Claims    only claims UHead scored (both runs skip over-long inputs), so every method
            is compared on the identical set.
  MUCH      Llama-3.1-8B-Instruct test claims: CCP, GRASP-v2 (MUCH-trained, in-domain),
            GRASP-v2 trained on the OTHER three models (no Llama-3.1-8B labels, like a
            pre-trained head), UHead (pre-trained, no MUCH labels).
  RAGTruth  Mistral test claims: CCP, GRASP-v2 zero-shot (MUCH-trained), GRASP-v2
            zero-shot with the domain adaptation selected in 20_domain_adapt.py,
            GRASP-v2 RAGTruth-trained, UHead (pre-trained, no RAGTruth labels).
  Report    ROC/PR/TPR@10% with bootstrap CIs; paired ΔROC vs CCP and vs UHead.

    python scripts/18_uhead_compare.py --uhead-dir results/kaggle_run10/out/uhead \
        --much-table results/dev/claims_k4.npz --frozen results/dev/frozen_v2_k4.json \
        --rt-dir results/kaggle_run9/out/ragtruth --domain-adapt results/domain_adapt.json \
        --out results/uhead_compare.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LLAMA8B = "meta-llama/Llama-3.1-8B-Instruct"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt16 = _load("rt16", "16_ragtruth_eval.py")
da20 = _load("da20", "20_domain_adapt.py")
fe2, dev, fe1 = rt16.fe2, rt16.dev, rt16.fe1


def orient(train_y, train_s):
    r = fe1.roc(train_y, train_s)
    return (1.0 if r >= 0.5 else -1.0), round(max(r, 1 - r), 4)


def report(title, y, scores, g, refs=("CCP", "UHead")):
    out = {}
    for ref in refs:
        out[f"vs_{ref}"] = fe1.boot(y, {ref: scores[ref], **{k: v for k, v in scores.items() if k != ref}}, g, ref=ref)
    base = out[f"vs_{refs[0]}"]
    n = base["_n"]
    print(f"\n{title}: {n['claims']} claims, {n['hallucinated']} hallucinated, {n['generations']} generations")
    for k in scores:
        r = base[k]
        d_ccp = (f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}" if "roc_diff_vs_ccp" in r else "")
        ru = out["vs_UHead"].get(k, {})
        d_uh = (f"  vs UHead {ru['roc_diff_vs_ccp']:+.4f} {ru['roc_diff_vs_ccp_ci95']}" if "roc_diff_vs_ccp" in ru else "")
        print(f"  {k:36} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}  TPR@10={r['tpr_at_10fpr']:.3f}{d_ccp}{d_uh}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uhead-dir", type=Path, required=True)
    ap.add_argument("--much-table", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--rt-dir", type=Path, required=True)
    ap.add_argument("--domain-adapt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    frozen = json.loads(args.frozen.read_text())
    key = frozen["probe_no_ccp"]["key"]
    res = {}

    # ---------------- MUCH, Llama-3.1-8B ----------------
    U = json.loads((args.uhead_dir / "uhead_much_llama31.json").read_text())
    tr_rows = [u for u in U if u["split"] == "train"]
    sign, tr_roc = orient(np.array([u["label"] for u in tr_rows]), np.array([u["uhead_logit"] for u in tr_rows]))
    print(f"MUCH UHead sign from {len(tr_rows)} train claims: {sign:+.0f} (oriented train ROC {tr_roc})")
    uh = {(u["gen_id"], u["chunk"]): sign * u["uhead_logit"] for u in U if u["split"] == "test"}

    d = dev.load(args.much_table, split=None)
    tr, te = d["split"] == "train", d["split"] == "test"
    rows = np.flatnonzero(te & (d["model"] == LLAMA8B))
    keys = [(d["gen_id"][i], int(d["chunk"][i])) for i in rows]
    keep = np.array([k in uh for k in keys])
    rows = rows[keep]
    sel = np.zeros(len(d["y"]), bool); sel[rows] = True
    in_dom = fe2.fit_score(d, key, tr, sel)
    unseen = fe2.fit_score(d, key, tr & (d["model"] != LLAMA8B), sel)
    scores = {"CCP": d["ccp"][rows], "UHead": np.array([uh[(d["gen_id"][i], int(d["chunk"][i]))] for i in rows]),
              "GRASP-v2 (in-domain)": in_dom, "GRASP-v2 (other 3 models only)": unseen}
    res["much_llama31"] = {"uhead_sign": sign, "uhead_train_roc": tr_roc, "coverage": f"{len(rows)}/{len(keep)}",
                           **report("MUCH Llama-3.1-8B test", d["y"][rows], scores, d["gen_id"][rows])}

    # ---------------- RAGTruth, Mistral ----------------
    R = json.loads((args.uhead_dir / "uhead_ragtruth_mistral.json").read_text())
    heads = {}
    for h in sorted({r["head"] for r in R}):
        t = [r for r in R if r["head"] == h and r["split"] == "train"]
        heads[h] = orient(np.array([r["label"] for r in t]), np.array([r["uhead_logit"] for r in t]))
    best = max(heads, key=lambda h: heads[h][1])
    sign = heads[best][0]
    print(f"\nRAGTruth heads (sign, oriented train ROC): {heads} -> {best}")
    uh = {(r["id"], r["chunk"]): sign * r["uhead_logit"] for r in R if r["head"] == best and r["split"] == "test"}

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    n_heads = sum(1 for n in d["names"] if n.startswith("b0_") and n.endswith("_grounding_mean"))
    rt = rt16.build_ragtruth_rows(args.rt_dir, tok, n_heads)
    D = {"names": d["names"], "X": np.vstack([d["X"], rt["X"]]), "y": np.r_[d["y"], rt["y"]],
         "ccp": np.r_[d["ccp"], rt["ccp"]], "model": np.r_[d["model"], np.full(len(rt["y"]), rt16.RT_MODEL)],
         "split": np.r_[np.where(tr, "train", "much_test"), rt["split"]], "gen_id": np.r_[d["gen_id"], rt["gen_id"]],
         "chunk": np.r_[d["chunk"], rt["chunk"]]}
    src, rt_tr = D["split"] == "train", D["split"] == "rt_train"
    cand = np.flatnonzero(D["split"] == "rt_test")
    have = np.array([(D["gen_id"][i], int(D["chunk"][i])) in uh for i in cand])
    rows = cand[have]
    sel = np.zeros(len(D["y"]), bool); sel[rows] = True
    da = json.loads(args.domain_adapt.read_text())
    fset, _, lname = key.split("|")
    cols = dev.columns(D["names"], fset)
    learner = dev.logreg if lname == "logreg" else dev.gbm(lname)
    scores = {"CCP": D["ccp"][rows],
              "UHead": np.array([uh[(D["gen_id"][i], int(D["chunk"][i]))] for i in rows]),
              "GRASP-v2 zero-shot (MUCH)": fe2.fit_score(D, key, src, sel),
              f"GRASP-v2 zero-shot + {da['selected']}": da20.fit_predict(da["selected"], D["X"][:, cols], D["y"], D["model"], src, rt_tr, sel, learner),
              "GRASP-v2 (RAGTruth-trained)": fe2.fit_score(D, key, rt_tr, sel)}
    res["ragtruth_mistral"] = {"heads": heads, "selected_head": best, "coverage": f"{len(rows)}/{len(cand)}",
                               **report("RAGTruth Mistral test", D["y"][rows], scores, D["gen_id"][rows])}

    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
