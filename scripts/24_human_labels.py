#!/usr/bin/env python3
"""
MUCH test split scored against the HUMAN annotators instead of GPT-4o.

MUCH's 200 test generations carry labels from two human experts (`an0`, `an1`).
The frozen per-claim scores of 14_abstention_v2.py (results/test_scores_v2.csv:
CCP, GRASP-v2-free, GRASP-v2) are evaluated, unchanged, against:
  human consensus   claims where an0 and an1 agree (label = their shared label)
  an0, an1          each annotator separately (all scored claims)
  gpt-4o            the label used so far, for reference
Nothing is refitted; this is a change of reference labels only.

    python scripts/24_human_labels.py --much-root ../much --scores results/test_scores_v2.csv \
        --out results/human_labels.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fe1", HERE / "10_final_eval.py")
fe1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe1)
METHODS = ("CCP", "GRASP-v2-free", "GRASP-v2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.much_root))
    from src.annotation import get_all_annotations  # type: ignore

    rows = list(csv.DictReader(args.scores.open(encoding="utf-8")))
    A = get_all_annotations()
    lab = {}
    for r in rows:
        a = A[r["gen_id"]]
        i = int(r["chunk"])
        lab[(r["gen_id"], i)] = {k: int(a.labels[k][i] == -1) for k in ("an0", "an1", "gpt-4o")}

    def scores(sub):
        return {m: np.array([float(r[m]) for r in sub]) for m in METHODS}

    res = {}
    refs = {
        "human consensus (an0 = an1)": [r for r in rows if lab[(r["gen_id"], int(r["chunk"]))]["an0"]
                                         == lab[(r["gen_id"], int(r["chunk"]))]["an1"]],
        "an0": rows, "an1": rows, "gpt-4o (as before)": rows,
    }
    key = {"human consensus (an0 = an1)": "an0", "an0": "an0", "an1": "an1", "gpt-4o (as before)": "gpt-4o"}
    for name, sub in refs.items():
        y = np.array([lab[(r["gen_id"], int(r["chunk"]))][key[name]] for r in sub])
        g = np.array([r["gen_id"] for r in sub])
        res[name] = fe1.boot(y, scores(sub), g)
        n = res[name]["_n"]
        print(f"\n{name}: {n['claims']} claims, {n['hallucinated']} hallucinated")
        for m in METHODS:
            r = res[name][m]
            d = f"  dROC vs CCP {r['roc_diff_vs_ccp']:+.4f} {r['roc_diff_vs_ccp_ci95']}" if "roc_diff_vs_ccp" in r else ""
            print(f"  {m:14} ROC={r['roc_auc']:.4f} {r['roc_auc_ci95']}  PR={r['pr_auc']:.4f}  TPR@10={r['tpr_at_10fpr']:.3f}{d}")
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
