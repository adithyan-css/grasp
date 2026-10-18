# GRASP

**GR**ounding, **A**ttention **S**ink and dis**P**ersion — a training-free, claim-level,
attention-based uncertainty estimator for long-form LLM generation.

Companion implementation for the BCSE306L case study, Topic 16
(*Uncertainty-Aware Abstention in LLM Applications*).

Start with [`docs/01_concept.md`](docs/01_concept.md) — it explains the problem, the
existing solutions, the method, and the novelty check in plain language.
[`docs/02_reading_list.md`](docs/02_reading_list.md) has the 33 papers behind it, with
direct links; `python scripts/05_download_papers.py --out papers/` fetches them all plus
publisher-issued BibTeX.

---

## The one-sentence claim

The best training-free claim-level hallucination detector (CCP) reaches **0.772 ROC-AUC**
on the MUCH benchmark at **≈124% runtime overhead** — detecting the hallucination costs more
than generating the text did. **GRASP-v2** — a small gradient-boosted classifier on
attention statistics from 4 heads per layer plus the token probabilities MUCH already ships —
reaches **0.924 ROC-AUC vs CCP's 0.770** on MUCH's human-annotated test split
(ΔROC +0.154, 95% CI 0.110–0.199), beats CCP on all four models, needs no CCP/NLI model,
and costs **33–38%** of generation time on a T4, where CCP measured on the same GPU costs
**192–430%**. On a second benchmark, RAGTruth, the same recipe retrained on RAGTruth's
labels reaches **0.827 vs CCP's 0.623**.

With in-domain labels it also beats the authors' pre-trained **UHeads** (closest prior method)
on both benchmarks. It is **supervised**: trained on MUCH and applied to RAGTruth without its
labels, it only reaches CCP's level (0.639 with label-free adaptation), and there pre-trained
UHeads transfer better (0.680). The training-free variant does not beat CCP. Full results and
limits:
[`results/FINAL_RESULTS_v2.md`](results/FINAL_RESULTS_v2.md).

---

## Layout

```
ai_project/
├── docs/
│   ├── 01_concept.md            problem, prior work, method, novelty check
│   └── 02_reading_list.md       33 papers with direct PDF links, by tier
├── src/
│   ├── grasp/
│   │   ├── features.py          teacher-forced attention extraction (G, S, P, W)
│   │   ├── heads.py             label-free head selection by grounding variance
│   │   ├── score.py             token signal, saliency weighting, soft-min, reset
│   │   ├── much_eval.py         exact replication of MUCH's evaluation protocol
│   │   └── sig_grasp.py         MUCH `UQSignal` plugin
│   └── tests/
│       ├── test_grasp.py        20 unit tests, no GPU or data needed
│       └── test_much_eval.py    10 checks, incl. cross-checks vs MUCH's own code
├── scripts/
│   ├── 01_setup_much.py         clone + deps + Python 3.11 compat patches
│   ├── 02_extract_attention.py  the only GPU step
│   ├── 03_score_and_eval.py     scoring, sweeps, ablations, GRASP-fuse
│   ├── 04_reproduce_ccp.py      the go/no-go checkpoint
│   ├── 05_download_papers.py    fetch every PDF + official BibTeX
│   ├── 06_analysis.py           bootstrap CIs, length confound, abstention curves
│   ├── 07_timing.py             extraction timing (estimated denominator; see 11)
│   ├── 08_claim_table.py        per-claim features: attention, token probs, length
│   ├── 09_develop.py            v1 development, train split only (grouped CV)
│   ├── 10_final_eval.py         v1 single-shot test evaluation
│   ├── 11_overhead.py           measured overhead vs real generation, same GPU
│   ├── 12_develop_v2.py         v2 development + frozen config, train split only
│   └── 13_final_eval_v2.py      v2 single-shot test evaluation
├── kaggle/
│   ├── build_notebook.py        builds the self-contained Kaggle notebook
│   └── grasp-full-run/          notebook + kernel-metadata.json (Kaggle CLI push)
├── notebooks/
│   └── GRASP_colab.ipynb        original Colab runner (superseded by kaggle/)
└── results/                     FINAL_RESULTS.md, run outputs, dev tables
```

## Verify the maths locally (no GPU, no data)

```bash
python src/tests/test_grasp.py
```

20 tests covering: exact grounding and sink arithmetic, causal masking, saliency counting
only later tokens, attention alignment (and the legacy-cache shift), saliency normalised
exactly once, head filtering, variance and top-k head selection, soft-min interpolating
between mean and max, boundary reset isolating claims, MUCH's human-annotated test split,
and recovery of a planted signal. All pass.

---

## Running the real thing

Everything downstream needs HuggingFace, which is blocked from this session's sandbox, so
the pipeline runs on **Colab or Kaggle**. Open `notebooks/GRASP_colab.ipynb` and work
through it in order.

### Setup gotchas already handled

`scripts/01_setup_much.py` clones MUCH, installs its dependencies and patches it.
Two traps it removes:

- MUCH's `src/__init__.py` imports **every** submodule, so importing anything pulls in
  the annotation and generation stacks. The full import-time dependency set is installed,
  not just the obvious ones — `python-decouple` is the first one that bites.
- MUCH targets **Python 3.12**. Two f-strings in `annotation/human_evaluator.py` put a
  backslash inside the expression part, which is PEP 701 and legal only on 3.12+. On a
  3.11 runtime the package dies with a SyntaxError in a file you never meant to use. The
  patch hoists those expressions out; no-op on 3.12, safe to re-run.

Also note the baseline signals are imported with `python -m src import-signals`, not
`import-baselines`.

### Gated models

Llama and Gemma are gated on HuggingFace: a token is not enough, your account must be
approved on each model page. The notebook checks this explicitly before the GPU step.

Two things worth knowing:

- **Never `print()` an `Annotation`.** Its `__repr__` decodes claim text, which loads the
  tokenizer, which hits HuggingFace and raises `GatedRepoError` for a model you are not
  authorised for. The notebook inspects samples tokenizer-free instead.
- **The CCP checkpoint does need the tokenizers.** MUCH zeroes stopwords and punctuation
  before aggregating, which means decoding every generated token. An earlier version of
  this README claimed otherwise; that was wrong. `04_reproduce_ccp.py` checks all four
  tokenizers up front and tells you which to request. `--no-stopword-mask` skips the rule
  for a plumbing check, but those numbers do not match the published table.

### Prerequisites

- A free T4 (Colab) or P100/T4 (Kaggle). Start with Llama-3.2-3B.
- A HuggingFace account with the Llama and Gemma licences accepted (they are gated).

### The three stages

**Stage 0 — reproduce CCP.** CPU only (but needs the tokenizers).

```bash
python src/tests/test_much_eval.py --much-root /content/much   # verify the protocol
python scripts/04_reproduce_ccp.py --much-root /content/much --skip-unavailable
```

`--skip-unavailable` evaluates only the models whose tokenizers load, so a pending gated
approval does not block the checkpoint. It reports which models were included and warns
that a partial run will land near, but not exactly on, the published figure.

### The four protocol rules

Transcribed into `grasp/much_eval.py` from `much/src/evaluation/`. Each one silently
corrupts the numbers if missed:

1. Labels are `{-1, +1}` and **`-1` is the hallucination class**.
2. The **trailing EOS chunk is dropped** (`token_chunks[:-1]`).
3. **Stopwords and punctuation are zeroed** before aggregating — needs the tokenizer.
4. Only annotations where **gpt-4o and gpt-4.1 agree** are scored.

Rule 3 is also a motivation for the method: the benchmark hand-zeroes function words
because token-probability signals are dominated by them. GRASP's attention saliency
derives the same thing from the model, with no fixed word list. Worth a sentence in the
paper.

> This is a **go/no-go checkpoint**. If CCP does not land near 0.772, the data loading or
> the claim aggregation is wrong and nothing downstream means anything. Fix it first.

**Stage 1 — extract attention.** The only GPU step, ~2–4 hours for all four configs.

```bash
python scripts/02_extract_attention.py \
    --much-root /content/much --config <generation_config_id> \
    --out results/cache_<cfg> --calib 200
```

Two passes: a calibration pass over 200 unlabelled generations keeping every head, which
picks one head per layer by grounding variance and fits the corpus normaliser; then a full
pass keeping only the selected heads. Resumable — rerun after a disconnect and it skips
what is already cached.

**Stage 2 — score and evaluate.** CPU only, seconds per run, so iterate freely.

```bash
python scripts/03_score_and_eval.py \
    --much-root /content/much --cache results/cache_<cfg> \
    --lang en --sweep                       # tune on the training split
python scripts/03_score_and_eval.py \
    --much-root /content/much --cache results/cache_<cfg> \
    --lang en --tau 8 --alpha 0.6 --ablate --fuse-with ccp_24_8
```

---

## Three variants

| Variant | What it is | Risk |
|---|---|---|
| **GRASP-0** | Fully training-free | Moderate — must beat 0.772 unaided |
| **GRASP-fuse** | Rank-average with CCP | Low — `--fuse-with ccp_24_8` |
| **GRASP-probe** | Logistic regression on 5 GRASP features | Very low — near-certain to beat CCP |

The paper is structured so that **any one of the three winning is publishable**. Write it
that way from the start.

---

## Protocol discipline

Three rules that protect the result from the most common reviewer objections:

1. **Freeze the evaluation script early** and do not touch it again. Every credibility
   collapse in this literature traces back to protocols that shifted while results were
   being collected.
2. **Sweep on the training split only.** Score the 200 human-annotated test samples once,
   with the frozen configuration. State this explicitly in the paper.
3. **Report the runtime column in the main table**, not an appendix. Normalise against
   generation time measured on your own GPU, using MUCH's
   `python -m src evaluate-generation-time`.

---

## Using GRASP inside MUCH's own harness

Copy `src/grasp/sig_grasp.py` into the MUCH repo at `src/signals/sig_grasp.py`, add
`SignalGRASP` to `src/signals/__init__.py`, then:

```bash
python -m src evaluate-signal grasp_gsd_t8 --lang en
```

This routes GRASP through the benchmark's own evaluator, so the comparison against every
baseline is apples-to-apples under an identical protocol.

---

## Status

| Piece | State |
|---|---|
| Concept document and novelty check | done |
| Core method implementation | done, 20 unit tests pass |
| MUCH evaluation protocol | replicated; cross-checks vs MUCH's own code pass |
| CCP reproduction | **passed** — `ccp_24_8`: ROC 0.7747 / PR 0.6426 vs published 0.772 / 0.639 |
| Feature extraction | all 8 configs, 4,873 generations, 1 and 4 heads/layer (Kaggle, 2x T4) |
| v1 evaluation (first test look) | GRASP-probe+CCP 0.891 vs CCP 0.770 — `results/FINAL_RESULTS.md` |
| v2 evaluation (second test look) | GRASP-v2 0.924 vs CCP 0.770, no CCP needed — `results/FINAL_RESULTS_v2.md` |
| Runtime overhead | measured on T4: GRASP-v2 33–38%, CCP 192–430% (same GPU); the old 1–3% claim is retracted |
| Second benchmark (RAGTruth, Mistral-7B) | retrained in-domain: 0.827 vs CCP 0.623; zero-shot from MUCH: ties CCP (0.625) |
| UHeads head-to-head (pre-trained heads, same claims) | in-domain GRASP wins on both; zero-shot, UHeads transfer better to RAGTruth |
| Label-free adaptation / response level | rank normalisation 0.625 → 0.639 zero-shot (n.s.); response level GRASP 0.911 vs CCP 0.762 |
| Paper draft / slides | `paper/grasp_paper.pdf` (6 pp.), `presentation/` (15-slide deck) |
| `report.html` / PDF | rewritten from the audited results (18 Oct 2026) |

Superseded numbers (pre-audit `report.html`, `results/RESULTS.md`) are kept for the record
and are marked as such.

---

## Licences

MUCH benchmark and repository: Apache-2.0 (École Polytechnique). `much-segmenter`: see its
own repo. RAGTruth, if used as the second evaluation setting: MIT.

## Key references

- **MUCH** — Dentan et al., *A Multilingual Claim Hallucination Benchmark*, arXiv 2511.17081
- **CCP** — Fadeeva et al., *Fact-Checking the Output of LLMs via Token-Level Uncertainty
  Quantification*, ACL Findings 2024
- **RAUQ** — Vazhentsev et al., *Efficient Hallucination Detection Using Uncertainty-Aware
  Attention Heads*, arXiv 2505.20045
- **UHeads** — *A Head to Predict and a Head to Question*, EMNLP 2025, arXiv 2505.08200
- **SinkProbe** — *Attention Sinks as Internal Signals for Hallucination Detection*,
  arXiv 2604.10697

Verify every venue, year and DOI against the publisher's page before citing.
