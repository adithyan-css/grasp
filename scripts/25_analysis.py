#!/usr/bin/env python3
"""
GRASP analyses for the paper: variance, error analysis, why it works, RAGTruth ablations.

  variance     5 runs of the frozen GRASP-v2 recipe, each trained on a different random
               90% of MUCH's training generations (HistGradientBoosting is deterministic
               given its data, so the variance comes from the training sample); MUCH test
               ROC/PR reported as mean ± sd, against GPT-4o and human-consensus labels.
  error types  MUCH test claims split by content: contains a digit (numbers, dates),
               contains a capitalised non-initial word (named entities), other; ROC of
               GRASP-v2 and CCP per type (frozen scores, results/test_scores_v2.csv).
  importance   group permutation importance of the frozen recipe, fitted on 80% of MUCH
               train generations and measured on the other 20% (train split only), for
               groups signal x depth-bin (attention), token probabilities and length.
  ragtruth     feature-group ablation on RAGTruth by 5-fold grouped CV on its TRAIN split.
  figures      results/figures/*.png

    python scripts/25_analysis.py --much-root ../much --much-table results/dev/claims_k4.npz \
        --frozen results/dev/frozen_v2_k4.json --rt-dir results/kaggle_run9/out/ragtruth \
        --cache-1head results/kaggle_run5/out --out results/analysis.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt16 = _load("rt16", "16_ragtruth_eval.py")
fe2, dev, fe1 = rt16.fe2, rt16.dev, rt16.fe1
SIGNALS = ("grounding", "sink", "dispersion")


def variance(d, key, human, out):
    tr, te = d["split"] == "train", d["split"] == "test"
    gens = np.unique(d["gen_id"][tr])
    res = {"gpt4o_roc": [], "gpt4o_pr": [], "human_roc": []}
    for seed in range(5):
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(gens, int(0.9 * len(gens)), replace=False))
        sub = tr & np.isin(d["gen_id"], list(keep))
        s = fe2.fit_score(d, key, sub, te)
        m = fe1.metrics(list(d["y"][te]), list(s))
        res["gpt4o_roc"].append(m["roc_auc"]); res["gpt4o_pr"].append(m["pr_auc"])
        hy, hs = [], []
        for i, sc in zip(np.flatnonzero(te), s):
            h = human.get((d["gen_id"][i], int(d["chunk"][i])))
            if h is not None:
                hy.append(h); hs.append(sc)
        res["human_roc"].append(fe1.roc(np.array(hy), np.array(hs)))
        print(f"  run {seed}: ROC {res['gpt4o_roc'][-1]:.4f}  human ROC {res['human_roc'][-1]:.4f}", flush=True)
    out["variance"] = {k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4),
                           "runs": [round(float(x), 4) for x in v]} for k, v in res.items()}
    print("variance:", out["variance"], flush=True)


def claim_type(text: str) -> str:
    if re.search(r"\d", text):
        return "number/date"
    words = re.findall(r"[^\W\d_]+", text)
    if any(w[0].isupper() for w in words[1:]):
        return "named entity"
    return "other"


def error_types(much_root, scores_csv, human, out):
    sys.path.insert(0, str(much_root))
    from src.annotation import get_all_annotations  # type: ignore
    from src.generation import get_tokenizer  # type: ignore
    A = get_all_annotations()
    rows = list(csv.DictReader(scores_csv.open(encoding="utf-8")))
    toks = {}
    types = []
    for r in rows:
        a = A[r["gen_id"]]
        m = a.generation.generation_cfg.model_name
        if m not in toks:
            toks[m] = get_tokenizer(m)
        c = a.token_chunks[int(r["chunk"])]
        types.append(claim_type(toks[m].decode([a.generation.output_tokens[i] for i in c])))
    types = np.array(types)
    y = np.array([int(r["label_hallucinated"]) for r in rows])
    res = {}
    for t in ("number/date", "named entity", "other"):
        mk = types == t
        res[t] = {"n": int(mk.sum()), "hallucinated": int(y[mk].sum()),
                  "rate": round(float(y[mk].mean()), 3)}
        for meth in ("CCP", "GRASP-v2"):
            s = np.array([float(r[meth]) for r in rows])
            res[t][meth] = round(fe1.roc(y[mk], s[mk]), 4)
        print(f"  {t:13} n={res[t]['n']:4} rate={res[t]['rate']:.3f}  CCP={res[t]['CCP']:.4f}  GRASP-v2={res[t]['GRASP-v2']:.4f}")
    out["error_types"] = res


def importance(d, key, out, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = d["split"] == "train"
    gens = np.unique(d["gen_id"][tr])
    rng = np.random.RandomState(0)
    val_g = set(rng.choice(gens, int(0.2 * len(gens)), replace=False))
    va = tr & np.isin(d["gen_id"], list(val_g))
    fit = tr & ~va
    fset, norm, lname = key.split("|")
    cols = dev.columns(d["names"], fset)
    names = [d["names"][c] for c in cols]
    X = dev.build_X(d, fset, norm, fit_mask=fit)
    model = (dev.logreg if lname == "logreg" else dev.gbm(lname))().fit(X[fit], d["y"][fit])
    base = fe1.roc(d["y"][va], model.predict_proba(X[va])[:, 1])
    groups = {}
    for j, n in enumerate(names):
        if n.startswith("b"):
            b, _, sig = n.split("_")[:3]
            groups.setdefault(f"{sig}|{b}", []).append(j)
        elif n.startswith("tp_"):
            groups.setdefault("token probabilities", []).append(j)
        else:
            groups.setdefault("claim length", []).append(j)
    imp = {}
    for gname, js in groups.items():
        drops = []
        for rep in range(3):
            Xp = X[va].copy()
            perm = np.random.RandomState(rep).permutation(Xp.shape[0])
            Xp[:, js] = Xp[perm][:, js]
            drops.append(base - fe1.roc(d["y"][va], model.predict_proba(Xp)[:, 1]))
        imp[gname] = round(float(np.mean(drops)), 4)
    out["importance"] = {"val_roc": round(base, 4), "roc_drop_when_permuted": imp}
    sig_tot = {s: round(sum(v for k, v in imp.items() if k.startswith(s + "|")), 4) for s in SIGNALS}
    out["importance"]["by_signal_sum"] = sig_tot
    print("importance (ROC drop):", {k: v for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:8]}, sig_tot)

    M = np.array([[imp.get(f"{s}|b{b}", 0.0) for b in range(8)] for s in SIGNALS])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_yticks(range(3), ["grounding", "sink", "dispersion"])
    ax.set_xticks(range(8), [f"{i * 12.5:.0f}–{(i + 1) * 12.5:.0f}%" for i in range(8)], fontsize=8)
    ax.set_xlabel("relative layer depth")
    for i in range(3):
        for j in range(8):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > M.max() * 0.6 else "#14213D")
    ax.set_title(f"ROC drop when a group is permuted (validation ROC {base:.3f}); "
                 f"token probabilities {imp['token probabilities']:.3f}, length {imp['claim length']:.3f}", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout(); fig.savefig(figdir / "importance_signal_depth.png", dpi=200); plt.close(fig)


def signal_profile(much_root, cache_root: Path, figdir, out):
    """Per-token mid-depth grounding and sink for one real test answer with both
    hallucinated and correct claims (1-head cache, layer-z-scored)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(much_root))
    from src.annotation import get_all_annotations  # type: ignore
    from src.generation import get_tokenizer  # type: ignore
    A = get_all_annotations()
    cand = []
    for a in A.values():
        if "an0" not in a.labels or a.labels["an0"] != a.labels["an1"]:
            continue
        lab = a.labels["an0"][:-1]
        if -1 in lab and 1 in lab and 4 <= len(lab) <= 9:
            for c in cache_root.glob(f"cache_*/features/{a.generation.id}.npy"):
                cand.append((a, c))
    cand.sort(key=lambda t: (t[0].generation.lang != "en", t[0].generation.id))
    if not cand:
        return
    a, path = cand[0]
    f = np.load(path).astype(float)[:, 0]  # (L, T, 4)
    L = f.shape[0]
    mid = slice(L // 4, 3 * L // 4)
    z = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-9)
    G, S = z[mid, :, 0].mean(0), z[mid, :, 1].mean(0)
    tok = get_tokenizer(a.generation.generation_cfg.model_name)
    T = f.shape[1]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for c, lab in zip(a.token_chunks[:-1], a.labels["an0"][:-1]):
        ax.axvspan(c[0] - 0.5, c[-1] + 0.5, color="#F2C49B" if lab == -1 else "#DCE6F2", alpha=0.8, lw=0)
    ax.plot(range(T), G[:T], color="#2F5F9E", lw=1.6, label="grounding (z, mid layers)")
    ax.plot(range(T), S[:T], color="#A14A12", lw=1.2, ls="--", label="sink (z, mid layers)")
    ax.set_xlabel("generated token"); ax.set_ylabel("z-score")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_title("One test answer: orange spans = claims both annotators marked hallucinated", fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "signal_profile_example.png", dpi=200); plt.close(fig)
    out["profile_example"] = {"gen_id": a.generation.id, "model": a.generation.generation_cfg.model_name,
                              "lang": a.generation.lang,
                              "text": tok.decode(a.generation.output_tokens)[:300]}


def ragtruth_ablation(d_much, rt, out):
    names = d_much["names"]
    y, g = rt["y"][rt["split"] == "rt_train"], rt["gen_id"][rt["split"] == "rt_train"]
    Xall = rt["X"][rt["split"] == "rt_train"]
    model = np.full(len(y), rt16.RT_MODEL)
    res = {}
    for fset in ("T", "A", "AT", "ATL"):
        cols = dev.columns(names, fset)
        s = dev.oof(dev.gbm("gbm_d3"), Xall[:, cols], y, g)
        res[fset] = round(fe1.roc(y, s), 4)
        print(f"  RAGTruth train CV  {fset:4} ROC={res[fset]:.4f}", flush=True)
    out["ragtruth_ablation_train_cv"] = res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--much-table", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--rt-dir", type=Path, required=True)
    ap.add_argument("--cache-1head", type=Path, required=True)
    ap.add_argument("--scores", type=Path, default=Path("results/test_scores_v2.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    figdir = args.out.parent / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    key = json.loads(args.frozen.read_text())["probe_no_ccp"]["key"]
    d = dev.load(args.much_table, split=None)

    sys.path.insert(0, str(args.much_root))
    from src.annotation import get_all_annotations  # type: ignore
    A = get_all_annotations()
    human = {}
    for a in A.values():
        if "an0" in a.labels:
            for i, (x, z) in enumerate(zip(a.labels["an0"], a.labels["an1"])):
                if x == z:
                    human[(a.generation_id, i)] = int(x == -1)

    out = {}
    print("== error types"); error_types(args.much_root, args.scores, human, out)
    print("== signal profile"); signal_profile(args.much_root, args.cache_1head, figdir, out)
    print("== importance"); importance(d, key, out, figdir)
    print("== variance"); variance(d, key, human, out)
    from transformers import AutoTokenizer
    n_heads = sum(1 for n in d["names"] if n.startswith("b0_") and n.endswith("_grounding_mean"))
    rt = rt16.build_ragtruth_rows(args.rt_dir, AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1"), n_heads)
    print("== RAGTruth ablation"); ragtruth_ablation(d, rt, out)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
