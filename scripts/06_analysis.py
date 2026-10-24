#!/usr/bin/env python3
"""
GRASP stage 3 — In-depth Evaluation & Topic 16 Uncertainty-Aware Abstention Analysis.

Performs rigorous statistical and diagnostic evaluation:
1. Disaggregated Evaluation: Per-language and per-model performance.
2. Grouped (Within-Generation) AUROC: Distinguishes cross-generation ranking from within-gen token localization.
3. Confound Analysis: Correlation between claim length (in tokens) and risk scores.
4. Non-parametric Bootstrap Confidence Intervals (95% CI on ROC-AUC, PR-AUC, TPR@10%FPR).
5. Topic 16 Abstention Analysis: Risk-Coverage curve, Area Under Risk-Coverage (AURC),
   and selective accuracy at standard rejection budgets (5%, 10%, 20%).

Usage:
    python scripts/06_analysis.py --csv results/claims_eval.csv --out results/analysis_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute point estimate and non-parametric bootstrap confidence interval."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    point = metric_fn(y_true, y_score)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        ys_b = y_score[idx]
        try:
            boot_scores.append(metric_fn(yt_b, ys_b))
        except Exception:
            continue

    if not boot_scores:
        return point, point, point

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_scores, 100 * alpha))
    hi = float(np.percentile(boot_scores, 100 * (1.0 - alpha)))
    return float(point), lo, hi


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"n": len(y_true), "positives": int(np.sum(y_true))}

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prc, rec, _ = precision_recall_curve(y_true, y_score)
    roc_val = float(auc(fpr, tpr))
    pr_val = float(auc(rec, prc))

    # TPR at 10% FPR
    ok = fpr <= 0.10
    tpr10 = float(tpr[ok].max()) if ok.any() else 0.0

    return {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "roc_auc": roc_val,
        "pr_auc": pr_val,
        "tpr_at_10fpr": tpr10,
    }


def compute_metrics_with_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000) -> dict:
    m = compute_metrics(y_true, y_score)
    if "roc_auc" not in m:
        return m

    def roc_fn(yt, ys):
        return float(roc_auc_score(yt, ys))

    def pr_fn(yt, ys):
        p, r, _ = precision_recall_curve(yt, ys)
        return float(auc(r, p))

    def tpr10_fn(yt, ys):
        f, t, _ = roc_curve(yt, ys)
        ok = f <= 0.10
        return float(t[ok].max()) if ok.any() else 0.0

    _, roc_lo, roc_hi = bootstrap_ci(y_true, y_score, roc_fn, n_bootstraps=n_boot)
    _, pr_lo, pr_hi = bootstrap_ci(y_true, y_score, pr_fn, n_bootstraps=n_boot)
    _, tpr_lo, tpr_hi = bootstrap_ci(y_true, y_score, tpr10_fn, n_bootstraps=n_boot)

    m["roc_ci_95"] = [roc_lo, roc_hi]
    m["pr_ci_95"] = [pr_lo, pr_hi]
    m["tpr10_ci_95"] = [tpr_lo, tpr_hi]
    return m


def within_generation_roc(records: list[dict], score_key: str) -> dict:
    """Compute average AUROC evaluated solely within generations that contain both classes."""
    by_gen: dict[str, list[dict]] = {}
    for r in records:
        by_gen.setdefault(r["gen_id"], []).append(r)

    gen_rocs = []
    n_mixed_gens = 0
    for gid, items in by_gen.items():
        y = np.array([int(x["label"]) for x in items])
        s = np.array([float(x[score_key]) for x in items])
        if len(np.unique(y)) == 2:
            try:
                gen_rocs.append(roc_auc_score(y, s))
                n_mixed_gens += 1
            except Exception:
                pass

    return {
        "n_total_gens": len(by_gen),
        "n_mixed_gens": n_mixed_gens,
        "mean_within_gen_roc": float(np.mean(gen_rocs)) if gen_rocs else float("nan"),
        "std_within_gen_roc": float(np.std(gen_rocs)) if gen_rocs else float("nan"),
    }


def length_confound_analysis(records: list[dict], score_key: str) -> dict:
    """Analyze correlation between claim token count and predicted risk / label."""
    lens = np.array([float(r.get("chunk_len", 1)) for r in records])
    scores = np.array([float(r[score_key]) for r in records])
    labels = np.array([float(r["label"]) for r in records])

    def pearson_r(x, y):
        if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    # Rank correlation (Spearman)
    def spearman_r(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return pearson_r(rx, ry)

    return {
        "length_mean": float(np.mean(lens)),
        "length_std": float(np.std(lens)),
        "pearson_length_vs_score": pearson_r(lens, scores),
        "spearman_length_vs_score": spearman_r(lens, scores),
        "pearson_length_vs_label": pearson_r(lens, labels),
        "spearman_length_vs_label": spearman_r(lens, labels),
    }


def abstention_evaluation(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Topic 16: Uncertainty-Aware Abstention / Selective Classification.

    In selective classification, the system accepts claims with risk <= threshold
    and abstains on claims with risk > threshold.
    - Coverage: fraction of claims accepted (not abstained).
    - Selective Risk: error rate (fraction of accepted claims that are hallucinations).
    - AURC: Area Under Risk-Coverage curve.
    """
    n = len(y_true)
    if n == 0:
        return {}

    order = np.argsort(y_score)  # lowest risk first (accepted first)
    y_sorted = y_true[order]

    # Cumulative errors as coverage expands from 1/n to 1.0
    cum_errors = np.cumsum(y_sorted)
    k_range = np.arange(1, n + 1)
    selective_risk = cum_errors / k_range
    coverage = k_range / n

    aurc = float(np.mean(selective_risk))

    # Error rate at standard rejection budgets
    budgets = [0.05, 0.10, 0.20, 0.30]
    rejection_stats = {}
    base_error = float(np.mean(y_true))

    for b in budgets:
        accepted_count = int(math.floor(n * (1.0 - b)))
        if accepted_count > 0:
            err_at_b = float(cum_errors[accepted_count - 1] / accepted_count)
            reduction = float((base_error - err_at_b) / max(base_error, 1e-9) * 100.0)
        else:
            err_at_b = 0.0
            reduction = 100.0
        rejection_stats[f"reject_{int(b*100)}pct"] = {
            "rejection_budget": b,
            "coverage": float(accepted_count / n),
            "selective_error_rate": err_at_b,
            "error_reduction_pct": reduction,
        }

    return {
        "base_hallucination_rate": base_error,
        "aurc": aurc,
        "normalized_aurc": aurc / max(base_error, 1e-9),
        "rejection_budgets": rejection_stats,
    }


def run_full_analysis(records: list[dict], out_path: Path | None = None) -> dict:
    if not records:
        print("No records to analyze.")
        return {}

    y_true = np.array([int(r["label"]) for r in records])
    score_keys = [k for k in ["native_score", "much_mean", "much_max", "much_geometric", "much_product", "fused_score"] if k in records[0]]

    out: dict = {"overall": {}, "by_language": {}, "by_model": {}, "within_gen": {}, "length_confound": {}, "abstention": {}}

    def scored(rows, sk):
        """Rows with a finite value for sk (fused_score is NaN where fusion skipped a claim)."""
        return [r for r in rows if r.get(sk) not in (None, "") and np.isfinite(float(r[sk]))]

    for sk in score_keys:
        rs = scored(records, sk)
        yt = np.array([int(r["label"]) for r in rs])
        ys = np.array([float(r[sk]) for r in rs])
        out["overall"][sk] = compute_metrics_with_ci(yt, ys)
        out["within_gen"][sk] = within_generation_roc(rs, sk)
        out["length_confound"][sk] = length_confound_analysis(rs, sk)
        out["abstention"][sk] = abstention_evaluation(yt, ys)

    # Disaggregate by Language
    langs = sorted(list(set(r.get("lang", "unknown") for r in records if r.get("lang"))))
    for l in langs:
        sub = [r for r in records if r.get("lang") == l]
        out["by_language"][l] = {}
        for sk in score_keys:
            rs = scored(sub, sk)
            yt_sub = np.array([int(r["label"]) for r in rs])
            ys_sub = np.array([float(r[sk]) for r in rs])
            out["by_language"][l][sk] = compute_metrics_with_ci(yt_sub, ys_sub)

    # Disaggregate by Model
    models = sorted(list(set(r.get("model", "unknown") for r in records if r.get("model"))))
    for m in models:
        sub = [r for r in records if r.get("model") == m]
        out["by_model"][m] = {}
        for sk in score_keys:
            rs = scored(sub, sk)
            yt_sub = np.array([int(r["label"]) for r in rs])
            ys_sub = np.array([float(r[sk]) for r in rs])
            out["by_model"][m][sk] = compute_metrics_with_ci(yt_sub, ys_sub)

    # Print clean summary
    print("\n" + "=" * 70)
    print("GRASP & TOPIC 16 ABSTENTION ANALYSIS SUMMARY")
    print("=" * 70)
    for sk in score_keys:
        m = out["overall"][sk]
        abst = out["abstention"][sk]
        r_ci = m.get("roc_ci_95", [0, 0])
        print(f"\n[{sk}]")
        print(f"  ROC-AUC: {m.get('roc_auc', 0):.4f}  (95% CI: [{r_ci[0]:.4f}, {r_ci[1]:.4f}])")
        print(f"  PR-AUC:  {m.get('pr_auc', 0):.4f}  TPR@10%FPR: {m.get('tpr_at_10fpr', 0):.3f}")
        print(f"  Within-Gen AUROC: {out['within_gen'][sk]['mean_within_gen_roc']:.4f}")
        print(f"  Length Confound (Spearman r): {out['length_confound'][sk]['spearman_length_vs_score']:.3f}")
        print(f"  Abstention AURC: {abst.get('aurc', 0):.4f} (Base Error: {abst.get('base_hallucination_rate', 0):.4f})")
        r10 = abst.get("rejection_budgets", {}).get("reject_10pct", {})
        print(f"  Reject 10% Risk Claims -> Error Reduction: {r10.get('error_reduction_pct', 0):.1f}%")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote full analysis to {out_path}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Path to per-claim CSV")
    ap.add_argument("--out", type=Path, default=Path("results/analysis_summary.json"))
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"Error: CSV not found at {args.csv}")
        return 1

    records = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    print(f"Loaded {len(records)} claim records from {args.csv}")
    run_full_analysis(records, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
