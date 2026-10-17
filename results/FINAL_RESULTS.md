# GRASP — final results (all 4 models, all 8 configs)

> **Superseded by [FINAL_RESULTS_v2.md](FINAL_RESULTS_v2.md)** (GRASP v2: 4 heads + token probabilities, no CCP needed, test ROC 0.924 vs CCP 0.770).

Kaggle run `adithyancss/grasp-full-run` v5 (2x Tesla T4), 2026-10-16. Raw outputs:
`results/kaggle_run5/out/`. Supersedes the single-config numbers in `RESULTS.md`.

## Protocol

- **Data:** MUCH, all 4,873 generations across 8 configs: Llama-3.2-3B, Llama-3.1-8B,
  Gemma-3-4B and Ministral-8B, each at T=0.7 and T=1.0. 20,751 claims, matching the
  benchmark count exactly.
- **Split:** MUCH's own. Test is the 200 human-annotated generations (867 claims, 229
  hallucinated). Train is the other 4,673 (19,884 claims). Labels follow MUCH's evaluator
  (gpt-4o, consensus rule).
- **Features:** extracted with `--profile corrected` (aligned attention rows, saliency
  normalised once, one variance-selected head per layer). Claims are summarised over 8
  relative-depth bins (`scripts/08_claim_table.py`).
- **Selection:** every choice was made by 5-fold cross-validation on the train split,
  grouped by generation (`scripts/09_develop.py`, `scripts/10_final_eval.py`). The test
  split was scored once.
- **Amendment, disclosed:** the selection rule first covered logistic regression only. Pooled
  train CV showed gradient boosting well ahead, so the search space was widened to include it
  **before any test number was seen**. The superseded logistic-only result is reported below
  and leads to the same conclusion.

## Main result — human-annotated test split (867 claims, 200 generations)

| Method | Trained? | ROC-AUC [95% CI] | PR-AUC | TPR@10%FPR | ΔROC vs CCP [95% CI] |
|---|---|---|---|---|---|
| CCP (`ccp_24_8`) | no | 0.770 [0.724, 0.817] | 0.637 | 0.537 | — |
| GRASP-0 (original) | no | 0.629 [0.569, 0.687] | 0.388 | 0.188 | −0.141 [−0.215, −0.072] |
| GRASP-0v2 (min grounding, mid-depth) | no | 0.667 [0.615, 0.720] | 0.403 | 0.271 | −0.103 [−0.175, −0.036] |
| **GRASP-probe** (attention only) | yes | **0.863** [0.829, 0.898] | **0.691** | **0.598** | **+0.093** [0.043, 0.143] |
| **GRASP-probe+CCP** | yes | **0.891** [0.859, 0.921] | **0.776** | **0.699** | **+0.121** [0.083, 0.162] |

CIs come from a 2,000-sample bootstrap resampled by generation. ΔROC is paired, on
identical claims.

## Per model (the fair comparison: base rates differ, e.g. Gemma ≈ 54% hallucinated)

| Model (test claims) | CCP | GRASP-probe | GRASP-probe+CCP | ΔROC probe+CCP vs CCP |
|---|---|---|---|---|
| Gemma-3-4B (134) | 0.726 | **0.864** | **0.871** | +0.145 [0.049, 0.219] |
| Llama-3.1-8B (282) | 0.831 | 0.866 | **0.903** | +0.071 [0.030, 0.107] |
| Llama-3.2-3B (279) | 0.767 | 0.838 | **0.841** | +0.074 [0.017, 0.136] |
| Ministral-8B (172) | **0.830** | 0.822 | **0.885** | +0.055 [0.012, 0.107] |

GRASP-probe+CCP beats CCP on **every model**, and every CI excludes zero. GRASP-probe
with attention alone beats CCP on three of four models; its per-model CI excludes zero
only on Gemma.

## Superseded logistic-only rule (same test set, for transparency)

| Method | ROC | PR | TPR@10 | ΔROC vs CCP |
|---|---|---|---|---|
| GRASP-probe (LR, C=0.1) | 0.800 | 0.574 | 0.437 | +0.031 [−0.018, 0.081] |
| GRASP-probe+CCP (LR, C=0.1) | 0.845 | 0.706 | 0.607 | +0.075 [0.036, 0.116] |

## Limits — state these in the paper

1. **The winners are supervised.** GRASP-probe is trained on the train split's GPT labels
   (the planned "GRASP-probe" variant). The **training-free** variants do *not* beat CCP;
   GRASP-0v2 reaches 0.667.
2. **No cross-model transfer.** In leave-one-model-out on train, the probe trained on three
   models falls below CCP on the fourth (Gemma: 0.65 vs 0.78; Ministral: 0.65 vs 0.81).
   The probe needs labelled data from the model it will monitor.
3. **The test labels are GPT labels.** These are gpt-4o consensus labels on the
   human-annotated samples, as in MUCH's evaluator, not the human labels themselves.
4. **Cost.** Extraction averages about 0.4–1.4 s per generation on 2x T4 (3B to 8B),
   one teacher-forced pass with no sampling and no NLI model. The probe is a small
   gradient-boosted tree on 100 features. The overhead relative to generation was not
   measured against MUCH's generation timer.

## Reproduce

```
python kaggle/build_notebook.py        # then push kaggle/grasp-full-run (Kaggle CLI)
python scripts/08_claim_table.py --much-root <much> --cache <cache_cfg>... --out claims_all.npz
python scripts/09_develop.py   --table claims_all.npz --out develop_all.json
python scripts/10_final_eval.py --table claims_all.npz --out final_eval.json
```

Scoring the test split locally from the Kaggle claim table reproduces these numbers.
