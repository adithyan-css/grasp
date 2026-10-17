#!/usr/bin/env python3
"""
GRASP v2 — abstention analysis (Topic 16) and per-claim test scores.

Re-creates exactly the frozen v2 predictions of 13_final_eval_v2.py (same table,
same frozen configuration, same fit on the full train split) and reports
selective prediction on the test split: if a system withholds the riskiest x%
of claims, what is the hallucination rate among the claims it keeps?

No choice is made from these numbers; they are an additional view of the
already-frozen predictions. Writes the per-claim scores to CSV for audit.

    python scripts/14_abstention_v2.py --table results/dev/claims_k4.npz \
        --frozen results/dev/frozen_v2_k4.json --out results/abstention_v2.json \
        --csv results/test_scores_v2.csv
"""

from __future__ import annotations

import argparse
import csv
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


fe2 = _load("final_eval_v2", "13_final_eval_v2.py")
dev = fe2.dev


def selective(y, risk, budgets=(0.05, 0.10, 0.20, 0.30)):
    order = np.argsort(risk)  # keep the least risky first
    n = len(y)
    base = float(y.mean())
    out = {"base_hallucination_rate": round(base, 4)}
    for b in budgets:
        keep = order[: int(round(n * (1 - b)))]
        err = float(y[keep].mean())
        out[f"reject_{int(b * 100)}pct"] = {
            "coverage": round(len(keep) / n, 4),
            "hallucination_rate_kept": round(err, 4),
            "relative_reduction_pct": round(100 * (base - err) / base, 1),
        }
    # area under the risk-coverage curve (lower is better)
    errs = np.cumsum(y[order]) / np.arange(1, n + 1)
    out["aurc"] = round(float(errs.mean()), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    frozen = json.loads(args.frozen.read_text())
    d = dev.load(args.table, split=None)
    tr, te = d["split"] == "train", d["split"] == "test"
    y = d["y"][te]
    scores = {
        "CCP": d["ccp"][te],
        "GRASP-v2-free": fe2.training_free_score(d, frozen["training_free_features"], tr)[te],
        "GRASP-v2": fe2.fit_score(d, frozen["probe_no_ccp"]["key"], tr, te),
    }
    res = {m: selective(y, s) for m, s in scores.items()}
    res["_check_roc"] = {m: round(fe2.fe1.roc(y, s), 4) for m, s in scores.items()}
    for m, r in res.items():
        if m.startswith("_"):
            continue
        print(f"{m:14} base={r['base_hallucination_rate']:.3f}  AURC={r['aurc']:.4f}  " + "  ".join(
            f"reject{k.split('_')[1]}: {v['hallucination_rate_kept']:.3f} (-{v['relative_reduction_pct']}%)"
            for k, v in r.items() if k.startswith("reject_")))
    print("ROC check (must match final_eval_v2.json):", res["_check_roc"])

    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    idx = np.flatnonzero(te)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gen_id", "chunk", "model", "lang", "label_hallucinated", *scores])
        for k, i in enumerate(idx):
            w.writerow([d["gen_id"][i], d["chunk"][i], d["model"][i], d["lang"][i], int(d["y"][i]),
                        *(f"{scores[m][k]:.6f}" for m in scores)])
    print(f"wrote {args.out} and {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
