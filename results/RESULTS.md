# GRASP results — corrected protocol, run of 2026-10-16

> **Superseded by [FINAL_RESULTS.md](FINAL_RESULTS.md)** (all 8 configs, 867 test claims, improved GRASP).

Kaggle notebook `adithyancss/grasp-full-run`, version 3 (Tesla T4), built by
`kaggle/build_notebook.py`. Raw outputs: `results/kaggle_run3/out/`. Every headline
number below was re-scored independently on the local machine from the same cached
features (`results/local/`) and matches to four decimals.

## Setup

| | |
|---|---|
| Model / config | Llama-3.2-3B-Instruct, T=0.7 (`bESSWQcqIaYZslTy7sy1Hg`), 586 generations |
| Extraction | `--profile corrected`: `align=current`, `saliency_norm=per_later_token`, variance head selection, 200 calibration generations |
| Split | MUCH's own: **test** = the 200 human-annotated samples (labels `an0`/`an1`), **train** = the other 4,673. Labels scored are `gpt-4o` under the consensus rule, as in MUCH's evaluator |
| Tuning | τ/α swept on **train only** → frozen at **τ=16, α=0.9** (train ROC 0.7356). Note: the optimum sits at the edge of the τ grid |
| Fusion | rank-average with `ccp_24_8`, aggregator frozen to `product` in advance |

## Checkpoint — CCP reproduction (all 4 models, 20,751 claims)

`ccp_24_8` / product: **ROC 0.7747, PR 0.6426, TPR@10%FPR 0.482** vs published
0.772 / 0.639 → reproduced (+0.003 / +0.004). Identical locally and on Kaggle.

## Main result — human-annotated test split (28 generations, 136 claims, 13 hallucinated)

| Method | ROC-AUC | 95% CI | PR-AUC | TPR@10%FPR |
|---|---|---|---|---|
| GRASP-0 (native soft-max) | 0.635 | 0.436 – 0.828 | 0.169 | 0.308 |
| CCP `ccp_24_8` (same claims) | **0.776** | — | **0.367** | 0.385 |
| GRASP-fuse (GRASP + CCP) | 0.737 | 0.577 – 0.882 | 0.335 | **0.538** |

On the held-out test claims, CCP beats GRASP-0. Fusion lowers ROC and PR but raises
the catch rate at 10% false alarms. The GRASP-0 interval includes 0.5. Only 28 of the
200 test samples belong to this model config, so the test split cannot separate the
methods with confidence.

## Reference — train split, same frozen config (558 generations, 3,041 claims)

τ/α were tuned on this split, so these are optimistic.

| Method | ROC-AUC | PR-AUC | TPR@10%FPR |
|---|---|---|---|
| GRASP-0 | 0.736 | 0.395 | 0.272 |
| CCP `ccp_24_8` | 0.807 | **0.601** | **0.514** |
| GRASP-fuse | **0.817** | 0.585 | 0.501 |

On the larger sample, fusion adds +0.010 ROC over CCP, with slightly lower PR and TPR.

## Per-language test slices

5–9 generations and 28–42 claims each; one language has no hallucinated claim at all.
These numbers range from 0.06 to 0.98 ROC and are noise — do not report them as findings.
Files: `grasp_test_{en,fr,es,de}.json`.

## Ablation (test split, native aggregator — 136 claims, read with the CI above)

| Variant | ROC | | Variant | ROC |
|---|---|---|---|---|
| full | 0.635 | | sink only | **0.733** |
| no grounding | 0.630 | | grounding only | 0.567 |
| no sink | 0.576 | | dispersion only | 0.584 |
| no dispersion | 0.679 | | no saliency | 0.635 |
| no boundary reset | 0.641 | | mean aggregation | 0.607 |
| cross-claim EMA | 0.635 | | max aggregation | 0.634 |

The sink signal carries most of the information here; dispersion appears to hurt. The
test sample is too small to settle this — rerun the ablation on the train split before
drawing conclusions.

## Diagnostics (test split, `analysis_test.json`)

- **Length confound:** GRASP score correlates with claim length (Spearman 0.44) far more
  than the label does (0.22). Part of the signal is "longer claim → higher risk".
- **Abstention (Topic 16):** rejecting the riskiest 10% of claims cuts the error rate on
  the rest by 22.8% (base hallucination rate 9.6%, AURC 0.078).

## Runtime

- Extraction, full pass: **231.9 s for 586 generations = 0.40 s/generation** on a T4
  (`cache_corrected/timing.json`; excludes model load and the calibration pass).
- `07_timing.py`: 0.34 s/generation extraction + 0.2 ms CPU scoring.
- The "40% overhead" in `timing_profile.json` is **not a measurement**: the denominator is
  a hard-coded 30 ms/token generation estimate (`scripts/07_timing.py:116`). A real
  overhead figure needs MUCH's `python -m src evaluate-generation-time` on the same GPU.

## What this replaces

`report.html` and `GRASP_Project_Report.pdf` were built from the earlier run (no split,
legacy alignment, test-selected fusion aggregator). Their headline — GRASP-fuse 0.8343
beating CCP 0.8251, and the German "outright win" — is not supported by this run and
should be rewritten from the tables above.
