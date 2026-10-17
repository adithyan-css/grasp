#!/usr/bin/env python3
"""
GRASP stage 6 — final, single-shot evaluation on MUCH's human-annotated test split.

Pre-registered (written before any test score was looked at):

  Every choice below is made from TRAINING claims only, by 5-fold cross-validation
  grouped by generation. The test split is read only after all choices are frozen,
  and is scored exactly once.

  Variants reported
    CCP              `ccp_24_8`, stopword-masked, `product` aggregator (frozen)
    GRASP-0          original training-free score (tau/alpha frozen from the sweep)
    GRASP-0v2        training-free: the single attention statistic with the best
                     pooled-train ROC; its sign is fixed a priori
                     (risk = -grounding, +sink, +dispersion), no weights fitted
    GRASP-probe      classifier on the attention features only; the model family
                     and setting are chosen by train CV ROC from
                     {L2 logistic regression C in {0.01, 0.1, 1}, gradient boosting}
    GRASP-probe+CCP  same, with the CCP claim score added as an input feature

  Amendment (2026-10-16, before any test score was seen): the original rule
  searched logistic regression only. Pooled 4-model train CV (09_develop.py,
  develop_all.json) showed gradient boosting well ahead (probe+CCP 0.874 vs 0.833),
  so the search space was widened to include it. The selection is still made from
  train CV alone. The superseded logistic-only run is kept as final_eval.json from
  Kaggle run v5 and reported alongside, for transparency.

  Metrics: ROC-AUC, PR-AUC, TPR@10%FPR, each with a 95% bootstrap CI resampled
  by generation; and the paired bootstrap CI of the ROC difference vs CCP on the
  identical test claims. Reported pooled and per model.

    python scripts/10_final_eval.py --table results/dev/claims_all.npz --out results/final_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp.much_eval import metrics  # noqa: E402

C_GRID = (0.01, 0.1, 1.0)
SIGN = {"grounding": -1.0, "sink": 1.0, "dispersion": 1.0}
N_BOOT = 2000


def logreg(C):
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    )


def gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
        min_samples_leaf=40, class_weight="balanced", random_state=0,
    )


CANDIDATES = {**{f"logreg_C{C}": (lambda C=C: logreg(C)) for C in C_GRID}, "gbm": gbm}


def roc(y, s):
    return float(metrics(list(y), list(s)).get("roc_auc", float("nan")))


def cv_roc(X, y, groups, make):
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        oof[va] = make().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return roc(y, oof)


def boot(y, scores: dict, groups, ref="CCP", seed=0):
    """Generation-level bootstrap: CIs per method and paired ROC diff vs ref."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    keys = ("roc_auc", "pr_auc", "tpr_at_10fpr")
    draws = {m: {k: [] for k in keys} for m in scores}
    diffs = {m: [] for m in scores if m != ref}
    for _ in range(N_BOOT):
        pick = np.concatenate([idx_of[g] for g in rng.choice(uniq, size=len(uniq))])
        yb = y[pick]
        if yb.min() == yb.max():
            continue
        r = {}
        for m, s in scores.items():
            mm = metrics(list(yb), list(s[pick]))
            for k in keys:
                draws[m][k].append(mm[k])
            r[m] = mm["roc_auc"]
        for m in diffs:
            diffs[m].append(r[m] - r[ref])
    out = {}
    for m, s in scores.items():
        point = metrics(list(y), list(s))
        out[m] = {k: round(float(point[k]), 4) for k in keys}
        out[m].update({f"{k}_ci95": [round(float(np.percentile(draws[m][k], q)), 4) for q in (2.5, 97.5)]
                       for k in keys})
        if m in diffs:
            dv = np.asarray(diffs[m])
            out[m]["roc_diff_vs_ccp"] = round(float(point["roc_auc"] - metrics(list(y), list(scores[ref]))["roc_auc"]), 4)
            out[m]["roc_diff_vs_ccp_ci95"] = [round(float(np.percentile(dv, q)), 4) for q in (2.5, 97.5)]
            out[m]["p_diff_le_0"] = round(float((dv <= 0).mean()), 4)
    out["_n"] = {"claims": int(len(y)), "hallucinated": int(y.sum()), "generations": int(len(uniq))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = [dict(np.load(p, allow_pickle=False)) for p in args.table]
    names = list(parts[0]["feature_names"])
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "feature_names"}
    tr, te = d["split"] == "train", d["split"] == "test"
    X, y, g = d["X"], d["y"], d["gen_id"]
    Xc = np.column_stack([X, d["ccp"]])

    # ---------------- all selection on TRAIN only ----------------
    sel: dict = {}
    stat_roc = {}
    for j, n in enumerate(names[:-4]):
        stat_roc[n] = roc(y[tr], SIGN[n.split("_")[1]] * X[tr, j])
    best_stat = max(stat_roc, key=stat_roc.get)
    sel["grasp0v2_statistic"] = best_stat
    sel["grasp0v2_train_roc"] = round(stat_roc[best_stat], 4)
    for key, M in (("probe", X), ("probe_ccp", Xc)):
        cv = {name: round(cv_roc(M[tr], y[tr], g[tr], make), 4) for name, make in CANDIDATES.items()}
        sel[f"{key}_train_cv_roc_by_candidate"] = cv
        sel[f"{key}_model"] = max(cv, key=cv.get)
        sel[f"{key}_train_cv_roc"] = cv[sel[f"{key}_model"]]
    print("frozen selection (train only):", json.dumps(sel, indent=2))

    probe = CANDIDATES[sel["probe_model"]]().fit(X[tr], y[tr])
    probe_ccp = CANDIDATES[sel["probe_ccp_model"]]().fit(Xc[tr], y[tr])

    # ---------------- the one look at TEST ----------------
    j = names.index(best_stat)
    scores_all = {
        "CCP": d["ccp"],
        "GRASP-0": d["grasp0"],
        "GRASP-0v2": SIGN[best_stat.split("_")[1]] * X[:, j],
        "GRASP-probe": probe.predict_proba(X)[:, 1],
        "GRASP-probe+CCP": probe_ccp.predict_proba(Xc)[:, 1],
    }
    result = {"selection": sel, "test": {}, "test_by_model": {}}
    st = {m: s[te] for m, s in scores_all.items()}
    result["test"] = boot(y[te], st, g[te])
    for model in sorted(set(d["model"][te])):
        mk = d["model"][te] == model
        if y[te][mk].min() == y[te][mk].max():
            continue
        result["test_by_model"][model] = boot(y[te][mk], {m: s[mk] for m, s in st.items()}, g[te][mk])

    n = result["test"]["_n"]
    print(f"\nTEST: {n['claims']} claims, {n['hallucinated']} hallucinated, {n['generations']} generations")
    for m in scores_all:
        r = result["test"][m]
        extra = (f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}"
                 if "roc_diff_vs_ccp" in r else "")
        print(f"  {m:16} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}  "
              f"TPR@10={r['tpr_at_10fpr']:.3f}{extra}")
    for model, rr in result["test_by_model"].items():
        print(f"  [{model.split('/')[-1]}] n={rr['_n']['claims']} " + "  ".join(
            f"{m}={rr[m]['roc_auc']:.3f}" for m in scores_all))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
