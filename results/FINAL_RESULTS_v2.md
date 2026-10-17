# GRASP v2 — final results

2026-10-17. Supersedes `FINAL_RESULTS.md` (v1), which stays in the repo as the first
evaluation. Raw outputs: `results/final_eval_v2.json` (test) and
`results/dev/develop_v2_{1head,k4}.json` with `frozen_v2_*.json` (development). Features and
timings come from Kaggle runs v5 (1 head/layer) and v6 (top-4 heads/layer), on 2x Tesla T4.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| Attention heads | 1 per layer (top variance) | **top 4 per layer**, by variance, no labels |
| Token probabilities | not used | **surprisal, entropy, top-1, top-1/2 gap** from MUCH's shipped top-24 probabilities; no extra model |
| Transfer across models | none | per-model rank normalisation, fitted on unlabelled features (selected only where it helps) |
| Learner | logistic regression → gradient boosting | tuned gradient boosting (31 leaves, 800 rounds) |
| Selection criterion | pooled train CV ROC | **mean per-model** train CV ROC, which can't be inflated by recognising which model wrote a claim |
| CCP needed? | yes, for the best variant | **no** |

## Protocol

- **Data and split:** MUCH, all 8 configs: 20,751 claims. Train is 19,884 claims (4,673
  generations); test is MUCH's 200 human-annotated generations (867 claims, 229
  hallucinated). Labels follow MUCH's evaluator (gpt-4o, consensus rule).
- **Selection:** everything was chosen by 5-fold cross-validation on train, grouped by
  generation (`scripts/12_develop_v2.py`). Its frozen output was written before the test
  split was read.
- **Single evaluation:** `scripts/13_final_eval_v2.py` reads those files, picks the 4-head
  table by the pre-registered rule (train CV 0.885 vs 0.881 for 1-head), fits on all of
  train, and scores test once.
- **Disclosure:** this is the **second** look at the test split; the first was v1. A
  plumbing bug found on a pretend split carved from train (training-free score weighting)
  was fixed before the real run.

## Main result — human-annotated test split (867 claims)

| Method | Trained? | Needs CCP? | ROC-AUC [95% CI] | PR-AUC [95% CI] | TPR@10%FPR [95% CI] | ΔROC vs CCP [95% CI] |
|---|---|---|---|---|---|---|
| CCP (`ccp_24_8`) | no | — | 0.770 [0.724, 0.817] | 0.637 [0.565, 0.708] | 0.537 [0.438, 0.610] | — |
| GRASP-0 (original) | no | no | 0.629 [0.569, 0.687] | 0.388 | 0.188 | −0.141 [−0.215, −0.072] |
| **GRASP-v2-free** | **no** | no | 0.764 [0.713, 0.815] | 0.501 [0.381, 0.628] | 0.354 [0.233, 0.498] | −0.006 [−0.056, +0.041] |
| **GRASP-v2** | yes | **no** | **0.924** [0.896, 0.948] | **0.845** [0.788, 0.893] | **0.790** [0.717, 0.864] | **+0.154** [0.110, 0.199] |
| GRASP-v2+CCP | yes | yes | 0.920 [0.895, 0.944] | 0.838 [0.786, 0.883] | 0.729 [0.652, 0.839] | +0.150 [0.111, 0.191] |
| *v1 GRASP-probe+CCP (for reference)* | yes | yes | *0.891* | *0.776* | *0.699* | *+0.121* |

Bootstrap: 2,000 samples resampled by generation; ΔROC is paired on identical claims.

## Per model

| Model (test claims) | CCP | GRASP-v2 | ΔROC [95% CI] | GRASP-v2-free |
|---|---|---|---|---|
| Gemma-3-4B (134) | 0.726 | **0.917** | +0.191 [0.073, 0.284] | 0.764 |
| Llama-3.1-8B (282) | 0.831 | **0.937** | +0.106 [0.046, 0.155] | 0.841 |
| Llama-3.2-3B (279) | 0.767 | **0.889** | +0.122 [0.047, 0.193] | 0.758 |
| Ministral-8B (172) | 0.830 | **0.897** | +0.067 [0.013, 0.128] | 0.708 |

GRASP-v2 beats CCP on every model, and every CI excludes zero. PR-AUC and TPR@10% also
beat CCP on every model (e.g. Llama-3.2-3B PR 0.728 vs 0.519).

## Transfer to an unseen model (test claims; probe fitted on the other 3 models)

| Held-out model | GRASP-v2 | CCP | v1 probe (train-CV LOMO) |
|---|---|---|---|
| Gemma-3-4B | **0.808** | 0.726 | 0.647 |
| Llama-3.1-8B | **0.882** | 0.831 | 0.749 |
| Llama-3.2-3B | 0.759 | **0.767** | 0.732 |
| Ministral-8B | **0.853** | 0.830 | 0.654 |

The probe now beats CCP on 3 of 4 models it has never seen labels for, and is within 0.01 on
the fourth. In v1 it lost on all four. These per-model transfer numbers have no CIs and
small samples (134–282 claims); read them as "comparable to or better than CCP".

## Measured runtime overhead (2x Tesla T4, same GPU, `scripts/11_overhead.py`)

| Model | tokens/answer | generation | GRASP (extract + score) | overhead |
|---|---|---|---|---|
| Gemma-3-4B | 8.3 | 0.75 s | 0.28 s | 37% |
| Llama-3.2-3B | 21.4 | 1.11 s | 0.38 s | 34% |
| Llama-3.1-8B | 23.9 | 2.28 s | 0.83 s | 36% |
| Ministral-8B | 14.9 | 1.40 s | 0.53 s | 38% |

**34–38% overhead**, against CCP's published ~124% (MUCH paper, A100; not re-measured
here, so the comparison crosses hardware). GRASP's cost is one teacher-forced forward pass
with eager attention, so its share shrinks for longer answers. The old "1–3%" claim in the
README and report was wrong and is retracted.

## Limits — state these in the paper

1. **GRASP-v2 is supervised.** It is trained on the train split's GPT-4o labels. The
   **training-free** GRASP-v2-free **ties** CCP (ΔROC −0.006, CI −0.056 to +0.041) but does
   not beat it; it is weaker on PR and TPR.
2. **This is the second look at the test split.** v1 was the first. All v2 choices were
   frozen on train before the look, and the result (+0.154) is far outside v1's margin
   (+0.121), but a fully fresh confirmation needs another benchmark (e.g. RAGTruth).
3. **Test labels are GPT-4o's**, on the human-annotated samples, as in MUCH's evaluator.
4. **Test scores sit above train CV** (0.924 vs about 0.885–0.897). The final model trains on
   100% of train rather than 80% folds, and the test set is small (200 generations), so
   expect some regression on new data. The lower CI bound is 0.896.
5. **Overhead is not apples-to-apples** with CCP's 124% (different GPU, CCP not re-timed).

## Reproduce

```
python kaggle/build_notebook.py   # push kaggle/grasp-full-run; v6 extracts top-4 heads + timing
python scripts/08_claim_table.py --much-root <much> --cache <cache_k4_cfg> ... --out claims_k4.npz
python scripts/12_develop_v2.py --table claims_k4.npz --out develop_v2_k4.json --frozen frozen_v2_k4.json
python scripts/13_final_eval_v2.py --candidate claims_v2_1head.npz frozen_v2_1head.json \
                                   --candidate claims_k4.npz frozen_v2_k4.json --out final_eval_v2.json
```
