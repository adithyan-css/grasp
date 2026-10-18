#!/usr/bin/env python3
"""
Does more diverse training data fix zero-shot transfer? Held-out-task test on RAGTruth.

Pre-registered (written before any score of this experiment was computed):

  For each RAGTruth task T in {QA, Summary, Data2txt} (T's labels never used):
    GRASP-MUCH     trained on MUCH train only
    GRASP-diverse  trained on MUCH train + RAGTruth train claims of the OTHER two tasks
  Both use the same recipe: the frozen v2 learner/feature set with the per-domain rank
  normalisation selected in 20_domain_adapt.py (target CDF from unlabelled RAGTruth
  train features), so the only difference is the training data.
  Scored on RAGTruth TEST claims of task T, against CCP and the authors' pre-trained
  UHead (exp6, sign fixed on train in 18_uhead_compare.py), on identical claims
  (those UHead scored). Pooled over the three held-out tasks as well.
  This is a further read of RAGTruth's test split, reported as such.

    python scripts/23_diverse_training.py --much-table results/dev/claims_k4.npz \
        --frozen results/dev/frozen_v2_k4.json --rt-dir results/kaggle_run9/out/ragtruth \
        --uhead results/kaggle_run10/out/uhead/uhead_ragtruth_mistral.json \
        --uhead-compare results/uhead_compare.json --out results/diverse_training.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt16 = _load("rt16", "16_ragtruth_eval.py")
da20 = _load("da20", "20_domain_adapt.py")
dev, fe1 = rt16.dev, rt16.fe1
TASKS = ("QA", "Summary", "Data2txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-table", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--rt-dir", type=Path, required=True)
    ap.add_argument("--uhead", type=Path, required=True)
    ap.add_argument("--uhead-compare", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    frozen = json.loads(args.frozen.read_text())
    fset, _, lname = frozen["probe_no_ccp"]["key"].split("|")
    learner = dev.logreg if lname == "logreg" else dev.gbm(lname)
    much = dev.load(args.much_table, split=None)
    cols = dev.columns(much["names"], fset)
    n_heads = sum(1 for n in much["names"] if n.startswith("b0_") and n.endswith("_grounding_mean"))
    rt = rt16.build_ragtruth_rows(args.rt_dir, AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1"), n_heads)

    X = np.vstack([much["X"][:, cols], rt["X"][:, cols]])
    y = np.r_[much["y"], rt["y"]]
    model = np.r_[much["model"], np.full(len(rt["y"]), rt16.RT_MODEL)]
    split = np.r_[np.where(much["split"] == "train", "train", "much_test"), rt["split"]]
    task = np.r_[np.full(len(much["y"]), ""), rt["task"]]
    gid = np.r_[much["gen_id"], rt["gen_id"]]
    chunk = np.r_[much["chunk"], rt["chunk"]]
    much_tr, rt_tr, rt_te = split == "train", split == "rt_train", split == "rt_test"

    uc = json.loads(args.uhead_compare.read_text())["ragtruth_mistral"]
    head = uc["selected_head"]; sign = uc["heads"][head][0]
    U = {(r["id"], r["chunk"]): sign * r["uhead_logit"] for r in json.loads(args.uhead.read_text())
         if r["head"] == head and r["split"] == "test"}

    res = {"per_task": {}}
    pooled = {k: [] for k in ("y", "g", "CCP", "UHead", "GRASP-MUCH", "GRASP-diverse")}
    for T in TASKS:
        rows = np.flatnonzero(rt_te & (task == T))
        rows = rows[[(gid[i], int(chunk[i])) in U for i in rows]]
        sel = np.zeros(len(y), bool); sel[rows] = True
        src_div = much_tr | (rt_tr & (task != T))
        s = {"CCP": rt["ccp"][rows - len(much["y"])],
             "UHead": np.array([U[(gid[i], int(chunk[i]))] for i in rows]),
             "GRASP-MUCH": da20.fit_predict("rank", X, y, model, much_tr, rt_tr, sel, learner),
             "GRASP-diverse": da20.fit_predict("rank", X, y, model, src_div, rt_tr, sel, learner)}
        res["per_task"][T] = fe1.boot(y[rows], s, gid[rows])
        r_u = fe1.boot(y[rows], {"UHead": s["UHead"], "GRASP-diverse": s["GRASP-diverse"]}, gid[rows], ref="UHead")
        res["per_task"][T]["GRASP-diverse_vs_UHead"] = {k: r_u["GRASP-diverse"][k] for k in ("roc_diff_vs_ccp", "roc_diff_vs_ccp_ci95")}
        n = res["per_task"][T]["_n"]
        print(f"\nheld-out {T}: {n['claims']} claims, {n['hallucinated']} hallucinated")
        for k in s:
            r = res["per_task"][T][k]
            d = f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}" if "roc_diff_vs_ccp" in r else ""
            print(f"  {k:14} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}{d}")
        dv = res["per_task"][T]["GRASP-diverse_vs_UHead"]
        print(f"  GRASP-diverse vs UHead: {dv['roc_diff_vs_ccp']:+.4f} {dv['roc_diff_vs_ccp_ci95']}")
        pooled["y"].append(y[rows]); pooled["g"].append(gid[rows])
        for k in s:
            pooled[k].append(s[k])

    # pooled over held-out tasks: each claim scored by the model that did not see its task
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    sc = {k: P[k] for k in ("CCP", "UHead", "GRASP-MUCH", "GRASP-diverse")}
    res["pooled"] = fe1.boot(P["y"], sc, P["g"])
    r_u = fe1.boot(P["y"], {"UHead": sc["UHead"], "GRASP-diverse": sc["GRASP-diverse"], "GRASP-MUCH": sc["GRASP-MUCH"]}, P["g"], ref="UHead")
    res["pooled_vs_UHead"] = {k: {kk: r_u[k][kk] for kk in ("roc_diff_vs_ccp", "roc_diff_vs_ccp_ci95")} for k in ("GRASP-diverse", "GRASP-MUCH")}
    print("\npooled over held-out tasks:")
    for k in sc:
        r = res["pooled"][k]
        d = f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}" if "roc_diff_vs_ccp" in r else ""
        print(f"  {k:14} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}{d}")
    print("  vs UHead:", res["pooled_vs_UHead"])
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
