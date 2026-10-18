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
   (+0.121). **Update:** the RAGTruth check (below) confirms the in-domain-trained result (+0.203) but shows no zero-shot advantage.
3. **Test labels are GPT-4o's**, on the human-annotated samples, as in MUCH's evaluator.
4. **Test scores sit above train CV** (0.924 vs about 0.885–0.897). The final model trains on
   100% of train rather than 80% folds, and the test set is small (200 generations), so
   expect some regression on new data. The lower CI bound is 0.896.
5. **Overhead:** resolved below. CCP re-timed on the same T4 costs 192–430% of generation time, against 33–38% for GRASP-v2.

## Reproduce

```
python kaggle/build_notebook.py   # push kaggle/grasp-full-run; v6 extracts top-4 heads + timing
python scripts/08_claim_table.py --much-root <much> --cache <cache_k4_cfg> ... --out claims_k4.npz
python scripts/12_develop_v2.py --table claims_k4.npz --out develop_v2_k4.json --frozen frozen_v2_k4.json
python scripts/13_final_eval_v2.py --candidate claims_v2_1head.npz frozen_v2_1head.json \
                                   --candidate claims_k4.npz frozen_v2_k4.json --out final_eval_v2.json
```

---

## Second benchmark — RAGTruth (added 2026-10-18)

Pre-registered in `scripts/16_ragtruth_eval.py` before any RAGTruth feature existed;
nothing was tuned on RAGTruth. Raw outputs: `results/ragtruth_eval.json`,
`results/kaggle_run9/out/ragtruth/` (Kaggle run v9, 2x Tesla T4).

**Setup.** RAGTruth (Niu et al., ACL 2024) responses from Mistral-7B-Instruct, the only
open generator accessible here, replayed through `mistralai/Mistral-7B-Instruct-v0.1`.
There are 1,475 sampled train responses (59,352 claims) and 445 of the 450 test responses
(17,226 claims, 1,323 hallucinated). The 5 missing test responses exceeded the pre-set
2,048-token cap. Claims come from MUCH's own segmenter; a claim is hallucinated if it
overlaps an annotated span. Heads were chosen with the same label-free rule.

| Method (claim level, test) | Trained on | ROC [95% CI] | PR [95% CI] | TPR@10% | ΔROC vs CCP [95% CI] |
|---|---|---|---|---|---|
| CCP (MUCH implementation, δ=24, σ=8) | — | 0.623 [0.602, 0.644] | 0.150 [0.126, 0.180] | 0.268 | — |
| max token surprisal | — | 0.613 | 0.105 | 0.161 | −0.010 [−0.038, +0.019] |
| mean token entropy | — | 0.623 | 0.114 | 0.178 | −0.000 [−0.028, +0.030] |
| GRASP-v2-free | — | 0.595 [0.573, 0.616] | 0.098 | 0.135 | −0.028 [−0.057, +0.001] |
| **GRASP-v2, zero-shot** | MUCH only | 0.625 [0.606, 0.645] | 0.118 | 0.187 | **+0.002 [−0.025, +0.029]** |
| **GRASP-v2, same recipe** | RAGTruth train | **0.827** [0.803, 0.849] | **0.383** [0.329, 0.435] | **0.556** | **+0.203 [+0.177, +0.233]** |

**By task (ROC; CCP → zero-shot GRASP-v2 → RAGTruth-trained GRASP-v2):**

- QA: 0.626 → 0.677 → **0.859**
- Summary: 0.617 → 0.652 → **0.765**
- Data-to-text: 0.624 → **0.568** (worse than CCP, CI −0.089 to −0.020) → **0.853**

**Response level** (response score = max over its claims; 445 responses, 248 hallucinated):
CCP 0.647, mean entropy 0.701, GRASP-v2-free 0.679, zero-shot GRASP-v2 0.619,
**RAGTruth-trained GRASP-v2 0.832**.

**Reading it honestly.**

1. **Trained in-domain, GRASP-v2 beats CCP on a second benchmark by a wide margin**
   (+0.203 ROC), with the recipe frozen on MUCH and no tuning.
2. **Zero-shot, it does not.** Trained only on MUCH and moved to a new generator, a new task
   family (RAG) and a new labelling scheme, it ties CCP overall. It is better on QA and
   summaries and worse on data-to-text. The detector is supervised and needs in-domain labels.
3. **The training-free score does not beat CCP** at claim level. At response level it does
   (0.679 vs 0.647), though plain mean entropy is better still (0.701).
4. Absolute claim-level numbers on RAGTruth are low for every method: claims from long RAG
   answers are short, and only 7.7% of test claims are hallucinated.

## CCP timed on the same GPU (resolves the cross-hardware caveat)

MUCH's own CCP code (`ccp_24_8`) on Tesla T4, 30 generations per model. CCP and
generation timings come from different sample generations, so overheads compare per token.

| Setting | CCP s/token | generation s/token | **CCP overhead** | **GRASP-v2 overhead** |
|---|---|---|---|---|
| MUCH, Gemma-3-4B | 0.222 | 0.090 | ~245% | 37% |
| MUCH, Llama-3.1-8B | 0.228 | 0.095 | ~240% | 36% |
| MUCH, Llama-3.2-3B | 0.225 | 0.052 | ~430% | 34% |
| MUCH, Ministral-8B | 0.220 | 0.094 | ~235% | 38% |
| RAGTruth, Mistral-7B (195-token answers) | 0.204 | 0.106 | **192%** | **33%** |

On the same GPU, GRASP-v2 costs roughly **one-sixth to one-twelfth** of CCP.

## Bottom line after both benchmarks

| Claim | Status |
|---|---|
| Supervised GRASP-v2 beats CCP when trained on the target setting's labels | **Yes, on 2 benchmarks** (MUCH +0.154, RAGTruth +0.203; both CIs well above 0) |
| …at a fraction of CCP's cost | **Yes**: 33–38% vs 192–430% overhead, same GPU |
| Beats CCP zero-shot on a new benchmark/task | **No**: ties on RAGTruth; better on QA and summaries, worse on data-to-text |
| Training-free GRASP beats CCP | **No**: ties on MUCH, loses on RAGTruth at claim level |

---

## Round 3 (added 2026-10-18): UHeads, zero-shot adaptation, response level

### Head-to-head with UHeads (Shelmanov et al., EMNLP 2025)

The authors' **pre-trained** claim heads, run teacher-forced on exactly the claims GRASP is
scored on (`scripts/17_uhead_extract.py`, Kaggle run v10; `scripts/18_uhead_compare.py`,
`results/uhead_compare.json`). The reimplemented band-attention features were checked equal
to the authors' extractor. Score sign, and on RAGTruth the choice of head (exp6 over exp5),
were fixed on training claims only.

**MUCH, Llama-3.1-8B-Instruct test** (282 claims; UHead covers all):

| Method | Labels | ROC [95% CI] | PR | TPR@10% | ΔROC vs UHead |
|---|---|---|---|---|---|
| UHead (pre-trained) | authors' data | 0.570 [0.486, 0.653] | 0.241 | 0.046 | — |
| CCP | — | 0.831 | 0.682 | 0.631 | +0.261 |
| GRASP-v2, other 3 models only | MUCH, not this model | 0.883 [0.817, 0.938] | 0.734 | 0.677 | +0.313 [+0.234, +0.397] |
| GRASP-v2, in-domain | MUCH | **0.937** | **0.871** | **0.785** | **+0.367** [+0.289, +0.449] |

The pre-trained Llama head is weak on MUCH even on training claims (oriented ROC 0.526). The
same code gives a sensible result on RAGTruth (below), so this is most likely a mismatch
between the head's training claims and MUCH's very short claims (about 4.6 tokens), not an
integration error. Treat the margin over UHead on MUCH as indicative, not decisive.

**RAGTruth, Mistral test** (17,181 of 17,226 claims; UHead uses its v0.2 backbone):

| Method | Labels | ROC [95% CI] | PR | TPR@10% | ΔROC vs UHead |
|---|---|---|---|---|---|
| CCP | — | 0.623 | 0.149 | 0.267 | −0.057 [−0.083, −0.029] |
| GRASP-v2 zero-shot | MUCH | 0.625 | 0.118 | 0.186 | −0.055 [−0.090, −0.017] |
| GRASP-v2 zero-shot + rank normalisation | MUCH + unlabelled RAGTruth | 0.639 | 0.131 | 0.219 | **−0.041 [−0.076, −0.004]** |
| **UHead (pre-trained)** | authors' data | **0.680** [0.649, 0.708] | 0.202 | 0.317 | — |
| GRASP-v2 retrained | RAGTruth | **0.827** | **0.383** | **0.556** | **+0.147 [+0.116, +0.180]** |

**Reading.** With in-domain labels, GRASP beats UHead clearly on both benchmarks. Without them,
UHead's pre-trained head transfers to RAGTruth better than GRASP trained on MUCH alone; it was
trained on much larger and more varied data.

### Label-free domain adaptation (`scripts/20_domain_adapt.py`, `results/domain_adapt.json`)

Candidates were selected by MUCH leave-one-model-out on the train split. Mean held-out ROC:
frozen 0.808, **per-domain rank normalisation 0.819** (better on all 4 models),
rank + importance weighting 0.818. Applied once to RAGTruth test (its second read), rank
normalisation lifts zero-shot GRASP from 0.625 to **0.639** vs CCP 0.623
(Δ +0.015, CI −0.011 to +0.042). It is closer, but **still not a significant win** over CCP.

### Response level on MUCH (`scripts/19_response_level.py`, `results/response_level_much.json`)

Unit: does an answer contain any hallucinated claim (200 test generations, 119 positive)?
Score: the max over its claims.

| Method | ROC [95% CI] | ΔROC vs CCP |
|---|---|---|
| CCP | 0.762 [0.695, 0.827] | — |
| best training-free by train selection (max surprisal) | 0.644 | −0.118 [−0.175, −0.063] |
| GRASP-v2-free | 0.616 | −0.146 |
| **GRASP-v2** | **0.911** [0.871, 0.949] | **+0.149 [+0.086, +0.214]** |

The training-free response-level edge seen on RAGTruth (0.679 vs 0.647) does **not** hold on
MUCH.

### Updated bottom line

| Claim | Status |
|---|---|
| GRASP-v2 beats CCP with in-domain labels | **Yes**, MUCH +0.154 and RAGTruth +0.203; also at response level (+0.149) |
| GRASP-v2 beats UHeads with in-domain labels | **Yes**, MUCH +0.367 and RAGTruth +0.147 (the MUCH margin has the caveat above) |
| Cheaper than CCP | **Yes**: 33–38% vs 192–430%, same GPU |
| Zero-shot to a new benchmark | **No**: ties CCP; rank normalisation narrows the gap (+0.015, n.s.); **loses to pre-trained UHeads** (−0.041) |
| Training-free beats CCP | **No**, at claim or response level, on either benchmark |
