#!/usr/bin/env python3
"""
Response-level detection on MUCH: does a whole answer contain a hallucination?

Pre-registered (written before any response-level MUCH test score was computed):

  Unit      a generation; label = at least one of its scored claims is hallucinated
            (MUCH protocol: consensus labels, trailing EOS chunk dropped).
  Score     max over the generation's claims of a claim-level score.
  Training-free candidates (no labels used to build them)
            GRASP-v2-free, mean token entropy, max token surprisal, and their
            rank average (GRASP-v2-free + mean entropy).
  Selection the training-free candidate with the best mean per-model response-level
            ROC on MUCH's TRAIN split is the one reported as "best training-free".
  Reported  on MUCH's test split (200 generations): CCP, all training-free
            candidates, GRASP-v2 (supervised), with bootstrap CIs over generations
            and paired ΔROC vs CCP. This is a new granularity, not a new tuning step.

    python scripts/19_response_level.py --table results/dev/claims_k4.npz \
        --frozen results/dev/frozen_v2_k4.json --out results/response_level_much.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fe2 = _load("final_eval_v2", "13_final_eval_v2.py")
dev, fe1 = fe2.dev, fe2.fe1


def per_response(gen_id, y, scores: dict):
    ids, inv = np.unique(gen_id, return_inverse=True)
    lab = np.zeros(len(ids), dtype=int)
    np.maximum.at(lab, inv, y)
    out = {}
    for k, s in scores.items():
        r = np.full(len(ids), -np.inf)
        np.maximum.at(r, inv, s)
        out[k] = r
    return ids, lab, out


def rnk(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frozen = json.loads(args.frozen.read_text())
    d = dev.load(args.table, split=None)
    tr, te = d["split"] == "train", d["split"] == "test"
    names = d["names"]
    X, y, g, model = d["X"], d["y"], d["gen_id"], d["model"]
    free = fe2.training_free_score(d, frozen["training_free_features"], tr)
    ent = X[:, names.index("tp_ent_mean")]
    nll = X[:, names.index("tp_nll_max")]
    combo = 0.5 * (dev.per_model_rank(free[:, None], model, tr)[:, 0]
                   + dev.per_model_rank(ent[:, None], model, tr)[:, 0])
    candidates = {"GRASP-v2-free": free, "mean entropy": ent, "max surprisal": nll,
                  "GRASP-v2-free + mean entropy": combo}

    # ---------------- selection on TRAIN (response level, mean per-model ROC) ----------------
    gm = {}
    for gid, m in zip(g, model):
        gm[gid] = m
    ids_tr, lab_tr, sc_tr = per_response(g[tr], y[tr], {k: v[tr] for k, v in candidates.items()})
    m_tr = np.asarray([gm[i] for i in ids_tr])
    train_cv = {}
    for k, s in sc_tr.items():
        rocs = [fe1.roc(lab_tr[m_tr == m], s[m_tr == m]) for m in np.unique(m_tr)]
        train_cv[k] = round(float(np.mean(rocs)), 4)
    best = max(train_cv, key=train_cv.get)
    print("train response-level mean per-model ROC:", train_cv, "-> best:", best)

    # ---------------- TEST ----------------
    sup = fe2.fit_score(d, frozen["probe_no_ccp"]["key"], tr, te)
    test_scores = {"CCP": d["ccp"][te], **{k: v[te] for k, v in candidates.items()}, "GRASP-v2": sup}
    ids_te, lab_te, sc_te = per_response(g[te], y[te], test_scores)
    res = {"unit": "generation", "train_selection": train_cv, "best_training_free": best,
           "test": fe1.boot(lab_te, sc_te, ids_te)}
    n = res["test"]["_n"]
    print(f"\nMUCH TEST, response level: {n['claims']} generations, {n['hallucinated']} with a hallucination")
    for k in test_scores:
        r = res["test"][k]
        extra = f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}" if "roc_diff_vs_ccp" in r else ""
        print(f"  {k:30} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}{extra}")
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
