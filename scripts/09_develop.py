#!/usr/bin/env python3
"""
GRASP stage 5b — development experiments on the TRAINING split only.

Every candidate is scored with 5-fold cross-validation grouped by generation, so
claims of one generation never sit on both sides of a fold. Test rows are dropped
on load and an assertion guards it: the human-annotated test split is touched
exactly once, by 10_final_eval.py, with the configuration chosen here.

    python scripts/09_develop.py --table results/dev/claims_3b07.npz --out results/dev/develop.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp.much_eval import metrics  # noqa: E402


def load_train(path: Path) -> dict:
    d = dict(np.load(path, allow_pickle=False))
    keep = d["split"] == "train"
    out = {k: (v[keep] if v.ndim and len(v) == len(keep) else v) for k, v in d.items()}
    assert (out["split"] == "train").all()
    return out


def rnk(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def summary(y, s):
    m = metrics(list(y), list(s))
    return {k: round(float(m[k]), 4) for k in ("roc_auc", "pr_auc", "tpr_at_10fpr") if k in m}


def cv_predict(make, X, y, groups, n_splits=5):
    """Out-of-fold predicted probabilities, grouped by generation."""
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make()
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
    return oof


def logreg(C):
    return lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    )


def gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
        min_samples_leaf=40, class_weight="balanced", random_state=0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = [load_train(p) for p in args.table]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "feature_names"}
    names = list(parts[0]["feature_names"])
    X, y, groups = d["X"], d["y"], d["gen_id"]
    ccp, g0 = d["ccp"], d["grasp0"]
    length = X[:, names.index("len_tokens")]
    print(f"train claims {len(y)}, generations {len(set(groups))}, "
          f"hallucinated {y.mean():.3f}, configs {sorted(set(d['config']))}")

    res: dict = {"n_claims": int(len(y)), "n_generations": int(len(set(groups)))}

    def report(name, s):
        res[name] = summary(y, s)
        r = res[name]
        print(f"  {name:38} ROC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} TPR@10={r['tpr_at_10fpr']:.3f}")

    print("\n-- references --")
    report("ccp_24_8/product", ccp)
    report("grasp0 (tau16 a0.9)", g0)
    report("claim length only", length)
    report("rank-fuse grasp0+ccp (50/50)", 0.5 * (rnk(g0) + rnk(ccp)))

    print("\n-- training-free single statistics (sign fixed a priori: risk = +sink, +disp, -ground) --")
    sign = {"grounding": -1.0, "sink": 1.0, "dispersion": 1.0}
    singles = {}
    for j, n in enumerate(names[:-4]):
        sig = n.split("_")[1]
        singles[n] = summary(y, sign[sig] * X[:, j])["roc_auc"]
    top = sorted(singles.items(), key=lambda kv: -kv[1])[:8]
    for n, r in top:
        print(f"  {n:38} ROC={r:.4f}")
    res["single_stats_top"] = top
    print("  spearman(length, grasp0) =", round(float(spearmanr(length, g0)[0]), 3),
          "| spearman(length, label) =", round(float(spearmanr(length, y)[0]), 3))

    print("\n-- supervised probes, 5-fold grouped CV (out-of-fold scores) --")
    Xc = np.column_stack([X, ccp, rnk(ccp)])
    oof = {}
    for C in (0.01, 0.1, 1.0):
        oof[f"probe_lr_C{C}"] = cv_predict(logreg(C), X, y, groups)
        report(f"GRASP-probe LR C={C}", oof[f"probe_lr_C{C}"])
    oof["probe_gbm"] = cv_predict(gbm, X, y, groups)
    report("GRASP-probe GBM", oof["probe_gbm"])
    for C in (0.01, 0.1, 1.0):
        oof[f"probe_lr_ccp_C{C}"] = cv_predict(logreg(C), Xc, y, groups)
        report(f"GRASP-probe+CCP LR C={C}", oof[f"probe_lr_ccp_C{C}"])
    oof["probe_gbm_ccp"] = cv_predict(gbm, Xc, y, groups)
    report("GRASP-probe+CCP GBM", oof["probe_gbm_ccp"])

    print("\n-- rank fusion of out-of-fold probe with CCP --")
    for k in ("probe_lr_C0.1", "probe_gbm"):
        for w in (0.3, 0.5, 0.7):
            report(f"rank {w}*{k} + {1-w:.1f}*ccp", w * rnk(oof[k]) + (1 - w) * rnk(ccp))

    models = sorted(set(d["model"]))
    if len(models) > 1:
        print("\n-- leave-one-model-out (train on 3 models' train claims, score the 4th) --")
        res["lomo"] = {}
        for held in models:
            te, trm = d["model"] == held, d["model"] != held
            row = {"n": int(te.sum()), "ccp": summary(y[te], ccp[te])["roc_auc"]}
            for name, M in (("probe", X), ("probe+ccp", Xc)):
                m = logreg(0.1)().fit(M[trm], y[trm])
                row[name] = summary(y[te], m.predict_proba(M[te])[:, 1])["roc_auc"]
            res["lomo"][held] = row
            print(f"  {held.split('/')[-1]:28} n={row['n']:5}  CCP={row['ccp']:.4f}  "
                  f"probe={row['probe']:.4f}  probe+CCP={row['probe+ccp']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
