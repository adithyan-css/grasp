#!/usr/bin/env python3
"""
GRASP v2 — second (and last) evaluation on MUCH's human-annotated test split.

Pre-registered before the test split was read:

  1. Feature cache: for each candidate table (1 head/layer, top-4 heads/layer),
     12_develop_v2.py froze its best configuration by mean per-model train CV
     ROC. The table whose frozen GRASP-v2 probe (no CCP) has the higher train CV
     score is used. Ties go to the cheaper 1-head table.
  2. Variants, all fitted on the full train split:
       GRASP-v2-free    training-free score frozen in 12_develop_v2.py
       GRASP-v2         probe on attention + token probabilities + claim length
                        (no CCP, so no NLI model is run)
       GRASP-v2+CCP     the same with the CCP claim score as an extra input
     References: CCP (`ccp_24_8`/product), GRASP-0, and the v1 probe+CCP result.
  3. Metrics: pooled and per model, ROC/PR/TPR@10%FPR with 95% bootstrap CIs
     resampled by generation, and paired ROC difference vs CCP.
  4. Transfer on test: for each model, fit on the other three models' TRAIN
     claims and score this model's TEST claims (leave-one-model-out).

This is the second look at the test split; the first was v1 (final_eval.json).
Report both.

    python scripts/13_final_eval_v2.py \
        --candidate results/dev/claims_v2_1head.npz results/dev/frozen_v2_1head.json \
        --candidate results/dev/claims_k4.npz results/dev/frozen_v2_k4.json \
        --out results/final_eval_v2.json
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dev = _load("develop_v2", "12_develop_v2.py")
fe1 = _load("final_eval_v1", "10_final_eval.py")


def make_learner(lname):
    return dev.logreg if lname == "logreg" else dev.gbm(lname)


def fit_score(d, key, tr, rows):
    """Fit `key` (fset|norm|learner) on rows `tr`, score rows `rows`."""
    fset, norm, lname = key.split("|")
    X = dev.build_X(d, fset, norm, fit_mask=tr)
    model = make_learner(lname)().fit(X[tr], d["y"][tr])
    return model.predict_proba(X[rows])[:, 1]


def training_free_score(d, features, fit_mask):
    names = d["names"]
    cols = [names.index(f) for f in features]
    Xr = dev.per_model_rank(d["X"][:, cols], d["model"], fit_mask)
    # Same definition as 12_develop_v2.training_free_candidates: average within
    # each feature group (attention 'b*', token-probability 'tp_*'), then average
    # the group means — so a combined candidate weights its two parts equally.
    groups: dict[str, list[np.ndarray]] = {}
    for k, f in enumerate(features):
        sgn = dev.SIGN[f.split("_")[2]] if f.startswith("b") else 1.0
        groups.setdefault("attn" if f.startswith("b") else "tp", []).append(
            Xr[:, k] if sgn > 0 else 1 - Xr[:, k])
    return np.mean([np.mean(v, axis=0) for v in groups.values()], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", nargs=2, action="append", required=True,
                    metavar=("TABLE", "FROZEN_JSON"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # ---- rule 1: choose the table by train CV only (frozen JSONs) ----
    cands = []
    for table, fz in args.candidate:
        frozen = json.loads(Path(fz).read_text())
        cands.append((frozen["probe_no_ccp"]["train_cv"], table, frozen))
    cands.sort(key=lambda c: -c[0])  # stable: 1-head listed first wins ties
    best_cv, table, frozen = cands[0]
    print("candidates (train CV of GRASP-v2):", [(round(c[0], 4), c[1]) for c in cands])
    print("chosen table:", table)
    print("frozen:", json.dumps({k: v for k, v in frozen.items() if k != "training_free_features"}))

    d = dev.load(Path(table), split=None)
    tr, te = d["split"] == "train", d["split"] == "test"
    y, g, model = d["y"], d["gen_id"], d["model"]

    # ---- the one v2 look at TEST ----
    s = {
        "CCP": d["ccp"][te],
        "GRASP-0": d["grasp0"][te],
        "GRASP-v2-free": training_free_score(d, frozen["training_free_features"], tr)[te],
        "GRASP-v2": fit_score(d, frozen["probe_no_ccp"]["key"], tr, te),
        "GRASP-v2+CCP": fit_score(d, frozen["probe_with_ccp"]["key"], tr, te),
    }
    out = {"chosen_table": table, "frozen": frozen, "test": fe1.boot(y[te], s, g[te]),
           "test_by_model": {}, "transfer_on_test": {}}
    for m in sorted(set(model[te])):
        mk = model[te] == m
        out["test_by_model"][m] = fe1.boot(y[te][mk], {k: v[mk] for k, v in s.items()}, g[te][mk])

    # ---- rule 4: leave-one-model-out on test ----
    key = frozen["probe_no_ccp"]["key"]
    for m in sorted(set(model)):
        train_rows = tr & (model != m)
        test_rows = te & (model == m)
        score = fit_score(d, key, train_rows, test_rows)
        out["transfer_on_test"][m] = {
            "n": int(test_rows.sum()),
            "GRASP-v2 (unseen model)": round(fe1.roc(y[test_rows], score), 4),
            "CCP": round(fe1.roc(y[test_rows], d["ccp"][test_rows]), 4),
        }

    n = out["test"]["_n"]
    print(f"\nTEST v2: {n['claims']} claims, {n['hallucinated']} hallucinated, {n['generations']} generations")
    for k in s:
        r = out["test"][k]
        extra = (f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}"
                 if "roc_diff_vs_ccp" in r else "")
        print(f"  {k:14} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}  "
              f"TPR@10={r['tpr_at_10fpr']:.3f}{extra}")
    print("\nper model (ROC):")
    for m, rr in out["test_by_model"].items():
        print(f"  {m.split('/')[-1]:28} n={rr['_n']['claims']:4} " + "  ".join(
            f"{k}={rr[k]['roc_auc']:.3f}" for k in s))
    print("\ntransfer on test (fit on the other 3 models):")
    for m, r in out["transfer_on_test"].items():
        print(f"  {m.split('/')[-1]:28} n={r['n']:4}  GRASP-v2={r['GRASP-v2 (unseen model)']:.3f}  CCP={r['CCP']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
