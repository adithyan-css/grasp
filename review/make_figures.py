"""Regenerate review figures 2-5 from Review_Workbook_GRASP.xlsx.

Figure 1 (flow diagram) is drawn by hand from PRISMA_Counts and is not
rebuilt here. Every number in figures 2-5 is counted from the workbook, so
re-running this after a workbook edit keeps the figures and tables in step.

    python review/make_figures.py
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

HERE = Path(__file__).parent
WB = HERE / "Review_Workbook_GRASP.xlsx"
OUT = HERE / "figures"

FAMILIES = ["Sampled consistency", "Attention", "Hidden states",
            "External verifier / LLM judge", "Output probabilities", "Self-report"]
COLOURS = ["#2f6690", "#3a7ca5", "#81c3d7", "#d9a441", "#b5525c", "#6a8f6f", "#8e7cc3"]
GRAN = ["Response", "Sentence", "Span / token / entity", "Claim"]

# keyword rules used for Figure 4 (dataset: columns J+K, metric: column M)
DATASETS = {
    "TriviaQA": r"trivia",
    "Biographies (FActScore / WikiBio / Bio)": r"factscore|wikibio|biograph|\bbios?\b",
    "GSM8K / MATH": r"gsm8k|\bmath\b|math-500",
    "Natural Questions / NQ-Open": r"\bnq\b|natural questions|nq-open|nq open",
    "SQuAD": r"squad",
    "CoQA": r"coqa",
    "RAGTruth": r"ragtruth",
    "CNN/DM / XSum": r"cnn|xsum",
    "HaluEval": r"halueval",
    "BioASQ": r"bioasq",
    "TruthfulQA": r"truthfulqa",
    "LongFact": r"longfact",
}
METRICS = {
    "AUROC / ROC-AUC / AUC": r"auroc|roc|\bauc\b",
    "F1 / precision / accuracy": r"\bf1\b|precision|accuracy",
    "PR-AUC / AUPRC / AP": r"pr-auc|auprc|auc-pr|aupr\b|\bap\b",
    "PRR": r"\bprr\b",
    "Calibration (ECE / Brier / coverage)": r"ece|brier|calibration|coverage",
}


def load():
    wb = openpyxl.load_workbook(WB, data_only=True)
    em = [r for r in wb["Extraction_Matrix"].iter_rows(min_row=6, values_only=True) if r[0]]
    tax = {r[0]: r for r in wb["Taxonomy_Draft"].iter_rows(min_row=3, values_only=True)
           if r[0] and str(r[0]).startswith("S")}
    gaps = {r[0]: r for r in wb["Gap_Table"].iter_rows(min_row=4, values_only=True) if r[0]}
    return em, tax, gaps


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig2_years(em, tax):
    years = sorted({r[2] for r in em})
    groups = FAMILIES + ["Benchmark / analysis"]
    fam = lambda sid: tax[sid][2] if tax[sid][2] in FAMILIES else "Benchmark / analysis"
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    bottom = np.zeros(len(years))
    for g, c in zip(groups, COLOURS):
        vals = np.array([sum(1 for r in em if r[2] == y and fam(r[0]) == g) for y in years])
        ax.bar([str(y) for y in years], vals, bottom=bottom, color=c, label=g)
        bottom += vals
    for i, t in enumerate(bottom):
        ax.text(i, t + 0.3, str(int(t)), ha="center")
    ax.set_ylabel("Included studies")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_years.png", dpi=200)


def fig3_taxonomy(tax):
    m = np.array([[sum(1 for t in tax.values() if t[2] == f and t[4] == g) for g in GRAN]
                  for f in FAMILIES])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    im = ax.imshow(m, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(GRAN)), ["Response", "Sentence", "Span / token /\nentity", "Claim"],
                  rotation=25, ha="right")
    ax.set_yticks(range(len(FAMILIES)), FAMILIES)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, m[i, j], ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xlabel("Granularity of the score")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"Detectors (n = {m.sum()})")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_taxonomy.png", dpi=200)


def count(em, rules, cols):
    text = lambda r: " ".join(str(r[c]) for c in cols).lower()
    return {k: sum(1 for r in em if re.search(p, text(r))) for k, p in rules.items()}


def fig4_datasets_metrics(em):
    d = count(em, DATASETS, (9, 10))
    m = count(em, METRICS, (12,))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    for ax, data, colour, title in ((a, d, "#6a8f6f", "Datasets (studies using each)"),
                                    (b, m, "#d9a441", "Metrics (studies reporting each)")):
        items = sorted(data.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color=colour)
        for i, (_, v) in enumerate(items):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
        ax.set_title(title, fontsize=10)
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_datasets_metrics.png", dpi=200)
    return d, m


def ids_in(text):
    return re.findall(r"S\d\d", text.split(";")[0])


def fig5_evaluation(em, tax, gaps):
    n = len(em)
    rows = [
        ("Peer-reviewed", sum(r[5] == "Peer-reviewed" for r in em)),
        ("Code linked", sum(r[16] == "Yes" for r in em)),
        ("Own data released", sum(r[17] == "Yes" for r in em)),
        ("Any shift / transfer test", sum(r[15] == "Yes" for r in em)),
        ("Variance / tests reported", sum(r[18] == "Yes" for r in em)),
        ("Cost measured", len(ids_in(gaps["G3"][2]))),
        ("Beyond English", sum(t[6] == "Multilingual" for t in tax.values())),
        ("Directly comparable", sum(r[21] == "Direct" for r in em)),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels, vals = zip(*rows[::-1])
    ax.barh(labels, vals, color="#3a7ca5")
    for i, v in enumerate(vals):
        ax.text(v + 0.6, i, f"{v} ({100 * v / n:.0f}%)", va="center", fontsize=9)
    ax.set_xlim(0, n)
    ax.set_xlabel(f"Studies (of {n})")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_evaluation.png", dpi=200)
    return rows


if __name__ == "__main__":
    em, tax, gaps = load()
    fig2_years(em, tax)
    fig3_taxonomy(tax)
    d, m = fig4_datasets_metrics(em)
    rows = fig5_evaluation(em, tax, gaps)
    print("datasets", d)
    print("metrics", m)
    print("evaluation", rows)
