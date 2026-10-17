#!/usr/bin/env python3
"""
GRASP v2 development — TRAINING split only. Writes the frozen v2 configuration.

Searches, by 5-fold cross-validation grouped by generation:

  feature sets   A attention, T token probabilities (MUCH ships them; no extra
                 model), L claim length (len_tokens, len_content; generation-level
                 fingerprints gen_len / rel_start excluded), C the CCP claim score
  normalisation  none | per-model rank (each feature mapped to its empirical CDF
                 within its own generation model, fitted on unlabelled features)
  learner        L2 logistic regression | gradient boosting (3 settings) |
                 rank-average ensemble of both
  training-free  sign-fixed, unweighted averages of per-model-normalised
                 statistics (no label is used to weight anything)

Selection criterion, fixed before running: the mean of the per-model CV ROC-AUCs
(each of the 4 generation models scored separately, then averaged). Pooled ROC is
reported too but not used for selection, because pooled ROC rewards recognising
which model wrote a claim (base rates differ: Gemma ~54% vs Llama ~22%).

Cross-model transfer is reported by leave-one-model-out for every candidate.

    python scripts/12_develop_v2.py --table results/dev/claims_v2_1head.npz \
        --out results/dev/develop_v2_1head.json --frozen results/dev/frozen_v2_1head.json
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

SIGN = {"grounding": -1.0, "sink": 1.0, "dispersion": 1.0}
LEN_KEEP = ("len_tokens", "len_content")


# ----------------------------------------------------------------------------- data


def load(path: Path, split: str | None = "train") -> dict:
    d = dict(np.load(path, allow_pickle=False))
    names = [str(n) for n in d.pop("feature_names")]
    if split is not None:
        keep = d["split"] == split
        d = {k: v[keep] for k, v in d.items()}
        assert (d["split"] == split).all()
    d["names"] = names
    return d


def columns(names: list[str], fset: str) -> list[int]:
    cols = []
    for j, n in enumerate(names):
        if "A" in fset and n.startswith("b"):
            cols.append(j)
        elif "T" in fset and n.startswith("tp_"):
            cols.append(j)
        elif "L" in fset and n in LEN_KEEP:
            cols.append(j)
    return cols


def per_model_rank(X: np.ndarray, model: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Map each feature to its empirical CDF within its generation model.
    Fitted on the rows in fit_mask (features only — no labels)."""
    out = np.empty_like(X, dtype=np.float64)
    for m in np.unique(model):
        rows = model == m
        ref_rows = rows & fit_mask
        if not ref_rows.any():
            ref_rows = rows
        ref = np.sort(X[ref_rows], axis=0)
        n = ref.shape[0]
        for j in range(X.shape[1]):
            out[rows, j] = np.searchsorted(ref[:, j], X[rows, j], side="right") / n
    return out


# ----------------------------------------------------------------------------- learners


def logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))


GBM_SETTINGS = {
    "gbm_d3": dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40),
    "gbm_d5": dict(max_depth=5, learning_rate=0.05, max_iter=500, min_samples_leaf=40),
    "gbm_l31": dict(max_depth=None, max_leaf_nodes=31, learning_rate=0.03, max_iter=800,
                    min_samples_leaf=60),
}


def gbm(setting="gbm_d3"):
    kw = dict(GBM_SETTINGS[setting])
    return lambda: HistGradientBoostingClassifier(
        l2_regularization=1.0, class_weight="balanced", random_state=0, **kw)


def rnk(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


# ----------------------------------------------------------------------------- scoring


def roc(y, s):
    return float(metrics(list(y), list(s)).get("roc_auc", float("nan")))


def per_model_roc(y, s, model) -> dict:
    return {str(m): round(roc(y[model == m], s[model == m]), 4) for m in np.unique(model)}


def summarise(y, s, model) -> dict:
    m = metrics(list(y), list(s))
    pm = per_model_roc(y, s, model)
    return {
        "pooled_roc": round(float(m["roc_auc"]), 4),
        "pooled_pr": round(float(m["pr_auc"]), 4),
        "pooled_tpr10": round(float(m["tpr_at_10fpr"]), 4),
        "per_model_roc": pm,
        "mean_per_model_roc": round(float(np.mean(list(pm.values()))), 4),
    }


def oof(make, X, y, groups):
    out = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        out[va] = make().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return out


def lomo(make, X, y, model) -> dict:
    res = {}
    for m in np.unique(model):
        te = model == m
        s = make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        res[str(m)] = round(roc(y[te], s), 4)
    return res


# ----------------------------------------------------------------------------- main


def build_X(d, fset, norm, fit_mask=None):
    cols = columns(d["names"], fset)
    X = d["X"][:, cols]
    if norm == "permodel":
        X = per_model_rank(X, d["model"], np.ones(len(X), bool) if fit_mask is None else fit_mask)
    if "C" in fset:
        c = d["ccp"][:, None]
        if norm == "permodel":
            c = per_model_rank(c, d["model"], np.ones(len(c), bool) if fit_mask is None else fit_mask)
        X = np.column_stack([X, c])
    return X


def training_free_candidates(d) -> dict:
    """Sign-fixed, unweighted averages of per-model-rank statistics."""
    names = d["names"]
    Xr = per_model_rank(d["X"], d["model"], np.ones(len(d["X"]), bool))

    def avg(pred):
        cols = [j for j, n in enumerate(names) if pred(n)]
        signed = []
        for j in cols:
            n = names[j]
            sgn = SIGN[n.split("_")[2]] if n.startswith("b") else 1.0
            signed.append(Xr[:, j] if sgn > 0 else 1 - Xr[:, j])
        return np.mean(signed, axis=0), [names[j] for j in cols]

    cands = {}
    cands["tf_ground_min_mid"] = avg(
        lambda n: n.startswith("b") and n.split("_")[0] in ("b1", "b2", "b3", "b4", "b5", "b6")
        and n.endswith("grounding_min"))
    cands["tf_ground_all_mid"] = avg(
        lambda n: n.startswith("b") and n.split("_")[0] in ("b1", "b2", "b3", "b4", "b5", "b6")
        and "_grounding_" in n)
    cands["tf_attn_all_signed"] = avg(lambda n: n.startswith("b"))
    cands["tf_tokenprob_nll"] = avg(lambda n: n in ("tp_nll_mean", "tp_nll_max", "tp_ent_mean"))
    gm, gn = cands["tf_ground_min_mid"]
    tp, tn = cands["tf_tokenprob_nll"]
    cands["tf_ground_min_mid+tokenprob"] = ((gm + tp) / 2, gn + tn)
    return cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="skip LOMO and GBM tuning")
    args = ap.parse_args()

    d = load(args.table, split="train")
    y, g, model = d["y"], d["gen_id"], d["model"]
    print(f"train claims {len(y)}, generations {len(set(g))}, models {sorted(set(model))}")
    res = {"references": {}, "training_free": {}, "supervised": {}}

    def show(tag, r, extra=""):
        print(f"  {tag:46} meanPM={r['mean_per_model_roc']:.4f} pooled={r['pooled_roc']:.4f} "
              f"PR={r['pooled_pr']:.4f} TPR10={r['pooled_tpr10']:.3f}{extra}")

    print("\n-- references --")
    for name, s in (("CCP", d["ccp"]), ("GRASP-0", d["grasp0"])):
        res["references"][name] = summarise(y, s, model)
        show(name, res["references"][name])

    print("\n-- training-free (no labels used) --")
    for name, (s, used) in training_free_candidates(d).items():
        r = summarise(y, s, model)
        r["features"] = used
        res["training_free"][name] = r
        show(name, r)

    print("\n-- supervised, 5-fold grouped CV --")
    fsets = ["A", "AL", "T", "TL", "AT", "ATL", "ATLC", "ALC"]
    for fset in fsets:
        for norm in ("none", "permodel"):
            X = build_X(d, fset, norm)
            for lname, make in (("logreg", logreg), ("gbm_d3", gbm("gbm_d3"))):
                key = f"{fset}|{norm}|{lname}"
                s = oof(make, X, y, g)
                r = summarise(y, s, model)
                res["supervised"][key] = r
                show(key, r)

    # tune the learner on the two best feature/norm pairs, plus a rank ensemble
    if not args.quick:
        best_pairs = sorted(
            {k.rsplit("|", 1)[0] for k in res["supervised"]},
            key=lambda p: -max(res["supervised"][f"{p}|{l}"]["mean_per_model_roc"]
                               for l in ("logreg", "gbm_d3")))[:3]
        print("\n-- learner tuning + ensemble on", best_pairs, "--")
        for pair in best_pairs:
            fset, norm = pair.split("|")
            X = build_X(d, fset, norm)
            scores = {"logreg": oof(logreg, X, y, g)}
            for st in GBM_SETTINGS:
                scores[st] = oof(gbm(st), X, y, g)
                if st != "gbm_d3":
                    key = f"{pair}|{st}"
                    res["supervised"][key] = summarise(y, scores[st], model)
                    show(key, res["supervised"][key])
            best_gbm = max(GBM_SETTINGS, key=lambda st: summarise(y, scores[st], model)["mean_per_model_roc"])
            ens = 0.5 * rnk(scores["logreg"]) + 0.5 * rnk(scores[best_gbm])
            key = f"{pair}|ens(logreg+{best_gbm})"
            res["supervised"][key] = summarise(y, ens, model)
            show(key, res["supervised"][key])

        print("\n-- leave-one-model-out (transfer to an unseen model) --")
        res["lomo"] = {}
        for key in sorted(res["supervised"], key=lambda k: -res["supervised"][k]["mean_per_model_roc"])[:8]:
            fset, norm, lname = key.split("|")
            if lname.startswith("ens"):
                continue
            X = build_X(d, fset, norm)
            make = logreg if lname == "logreg" else gbm(lname)
            res["lomo"][key] = lomo(make, X, y, model)
            print(f"  {key:46} " + "  ".join(f"{m.split('/')[-1][:12]}={v:.3f}"
                                             for m, v in res["lomo"][key].items()))
        res["lomo_ccp"] = per_model_roc(y, d["ccp"], model)
        print("  CCP (per model)                                " + "  ".join(
            f"{m.split('/')[-1][:12]}={v:.3f}" for m, v in res["lomo_ccp"].items()))

    # ---------------- freeze: best by mean per-model CV ROC ----------------
    def pick(pool, allow):
        ok = {k: v for k, v in pool.items() if allow(k)}
        k = max(ok, key=lambda k: ok[k]["mean_per_model_roc"])
        return k, ok[k]["mean_per_model_roc"]

    frozen = {
        "criterion": "mean per-model 5-fold grouped CV ROC on the train split",
        "training_free": dict(zip(("name", "train_cv"), pick(res["training_free"], lambda k: True))),
        "probe_no_ccp": dict(zip(("key", "train_cv"), pick(res["supervised"], lambda k: "C" not in k.split("|")[0]))),
        "probe_with_ccp": dict(zip(("key", "train_cv"), pick(res["supervised"], lambda k: "C" in k.split("|")[0]))),
        "training_free_features": None,
    }
    frozen["training_free_features"] = res["training_free"][frozen["training_free"]["name"]]["features"]
    print("\nFROZEN v2 (train only):", json.dumps({k: v for k, v in frozen.items()
                                                    if k != "training_free_features"}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    args.frozen.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"wrote {args.out} and {args.frozen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
