#!/usr/bin/env python3
"""
Label-free domain adaptation for zero-shot transfer of GRASP-v2.

Pre-registered (written before any adapted score on RAGTruth was computed):

  Candidates  (all fitted on MUCH train labels only; the target domain contributes
               UNLABELLED features only)
    frozen      the frozen v2 recipe ATL|none|gbm_l31 (reference)
    rank        per-domain rank normalisation: every feature mapped to its empirical
                CDF within its own domain (domain = generation model), the target's
                CDF fitted on its own unlabelled features
    rank+iw     rank, plus importance weights for source claims from a logistic
                domain classifier (source vs target, unlabelled), w = p/(1-p),
                clipped to [0.1, 10] and normalised to mean 1
  Selection   mean ROC over MUCH leave-one-model-out folds on the TRAIN split
              (target = held-out model; its labels used only to score the fold).
  Final       the selected candidate, trained on all MUCH train claims, applied once
              to RAGTruth test claims (target CDF / domain classifier fitted on
              RAGTruth TRAIN features, unlabelled). Second read of RAGTruth test;
              reported with the frozen zero-shot result and CCP on identical claims.

    python scripts/20_domain_adapt.py --much-table results/dev/claims_k4.npz \
        --rt-dir results/kaggle_run9/out/ragtruth --frozen results/dev/frozen_v2_k4.json \
        --out results/domain_adapt.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt16 = _load("rt16", "16_ragtruth_eval.py")
dev, fe1 = rt16.dev, rt16.fe1
CANDIDATES = ("frozen", "rank", "rank+iw")


def importance_weights(Xs, Xt, seed=0):
    rng = np.random.RandomState(seed)
    nt = min(len(Xt), len(Xs))
    it = rng.choice(len(Xt), nt, replace=False)
    Z = np.vstack([Xs, Xt[it]])
    lab = np.r_[np.zeros(len(Xs)), np.ones(nt)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Z, lab)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    w = np.clip(p / (1 - p), 0.1, 10.0)
    return w / w.mean()


def fit_predict(cand, X, y, model, src, tgt_fit, tgt_eval, learner):
    """Fit on rows `src` (labelled), predict rows `tgt_eval`; `tgt_fit` are the
    unlabelled target rows used for its CDF / domain classifier."""
    if cand == "frozen":
        Xa = X
    else:
        Xa = dev.per_model_rank(X, model, src | tgt_fit)
    w = None
    if cand == "rank+iw":
        w = importance_weights(Xa[src], Xa[tgt_fit])
    m = learner()
    m.fit(Xa[src], y[src], **({"sample_weight": w} if w is not None else {}))
    return m.predict_proba(Xa[tgt_eval])[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-table", type=Path, required=True)
    ap.add_argument("--rt-dir", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--hf-model", default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    frozen = json.loads(args.frozen.read_text())
    fset, _, lname = frozen["probe_no_ccp"]["key"].split("|")
    learner = dev.logreg if lname == "logreg" else dev.gbm(lname)
    much = dev.load(args.much_table, split=None)
    cols = dev.columns(much["names"], fset)

    # ---------------- selection: MUCH leave-one-model-out on TRAIN ----------------
    tr = much["split"] == "train"
    X, y, model = much["X"][:, cols], much["y"], much["model"]
    sel = {c: {} for c in CANDIDATES}
    for held in sorted(set(model)):
        src = tr & (model != held)
        tgt = tr & (model == held)
        for c in CANDIDATES:
            s = fit_predict(c, X, y, model, src, tgt, tgt, learner)
            sel[c][held] = round(fe1.roc(y[tgt], s), 4)
        print(f"  LOMO {held.split('/')[-1]:28} " + "  ".join(f"{c}={sel[c][held]:.4f}" for c in CANDIDATES), flush=True)
    mean = {c: round(float(np.mean(list(v.values()))), 4) for c, v in sel.items()}
    best = max(mean, key=mean.get)
    print("LOMO mean ROC:", mean, "-> selected:", best, flush=True)

    # ---------------- final: MUCH train -> RAGTruth test ----------------
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    n_heads = sum(1 for n in much["names"] if n.startswith("b0_") and n.endswith("_grounding_mean"))
    rt = rt16.build_ragtruth_rows(args.rt_dir, tok, n_heads)
    XA = np.vstack([much["X"][:, cols], rt["X"][:, cols]])
    yA = np.r_[much["y"], rt["y"]]
    mA = np.r_[much["model"], np.full(len(rt["y"]), rt16.RT_MODEL)]
    sA = np.r_[np.where(tr, "train", "much_test"), rt["split"]]
    src, rt_tr, rt_te = sA == "train", sA == "rt_train", sA == "rt_test"
    scores = {"CCP": rt["ccp"][rt["split"] == "rt_test"],
              "GRASP-v2 frozen (zero-shot)": fit_predict("frozen", XA, yA, mA, src, rt_tr, rt_te, learner)}
    if best != "frozen":
        scores[f"GRASP-v2 {best} (zero-shot, selected)"] = fit_predict(best, XA, yA, mA, src, rt_tr, rt_te, learner)
    g = rt["gen_id"][rt["split"] == "rt_test"]
    yt = yA[rt_te]
    res = {"lomo_selection": sel, "lomo_mean": mean, "selected": best, "ragtruth_test": fe1.boot(yt, scores, g)}
    n = res["ragtruth_test"]["_n"]
    print(f"\nRAGTruth TEST zero-shot: {n['claims']} claims")
    for k in scores:
        r = res["ragtruth_test"][k]
        extra = f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}" if "roc_diff_vs_ccp" in r else ""
        print(f"  {k:42} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}{extra}")
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
