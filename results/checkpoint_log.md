# Checkpoint log

## 2026-09-09 — CCP reproduction: PASSED

Environment: Colab, Python 3.13, CPU only. MUCH imported via `01_setup_much.py`.

**Protocol verification** — `test_much_eval.py`, 10/10 checks. Aggregators match MUCH's
own implementation across 200 random chunks × 4 aggregators; ROC/PR match exactly
(0.895429 / 0.874051 on the shared fixture).

**Tokenizer availability** — all four models loaded: `google/gemma-3-4b-it`,
`meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`,
`mistralai/Ministral-8B-Instruct-2410`. Llama-3.1-8B works despite its page still
showing "awaiting review", so no model is excluded.

**Coverage** — 4,873 annotations, 20,751 chunks scored, 0 missing signals, stopword
masking on. Chunk count matches the MUCH paper's stated 20,751 exactly.

### Result — signal `ccp_10_3`

| Aggregator | ROC-AUC | PR-AUC | AP | TPR@10%FPR |
|---|---|---|---|---|
| mean | 0.7575 | 0.5715 | 0.5713 | 0.417 |
| max | 0.7632 | 0.5947 | 0.5945 | 0.436 |
| geometric | 0.7598 | 0.5815 | 0.5815 | 0.427 |
| **product** | **0.7660** | **0.6160** | 0.6156 | **0.446** |

Published: ROC 0.772 / PR 0.639. Ours: **-0.0060 / -0.0230**. Verdict: reproduced.

`product` is the winning aggregator throughout — fix it for all later comparisons.

### Per language (product)

| Lang | n | pos | ROC | PR | TPR@10%FPR |
|---|---|---|---|---|---|
| en | 4,204 | 1,164 | 0.8030 | 0.6867 | 0.564 |
| de | 4,790 | 1,626 | 0.7799 | 0.6807 | 0.458 |
| fr | 5,998 | 1,779 | 0.7594 | 0.5951 | 0.404 |
| es | 5,759 | 1,393 | 0.7190 | 0.5080 | 0.378 |

English is 0.084 ROC above Spanish. Any per-language table must be read against this
spread, not against the pooled number.

### Per model (product)

| Model | n | pos | base rate | ROC | PR |
|---|---|---|---|---|---|
| Ministral-8B-2410 | 4,836 | 1,396 | 29% | 0.8114 | 0.7074 |
| Llama-3.2-3B | 6,220 | 1,373 | 22% | 0.7982 | 0.5713 |
| gemma-3-4b-it | 3,482 | 1,858 | **53%** | 0.7799 | 0.8022 |
| Llama-3.1-8B | 6,213 | 1,335 | 21% | 0.7746 | 0.5289 |

**Gemma hallucinates at more than twice the rate of the Llamas.** PR-AUC is base-rate
dependent, so Gemma's 0.8022 is not evidence the detector works better there — its ROC
is in fact the second lowest. Compare PR-AUC only within a model, never across.

### Caveat on this run

`04_reproduce_ccp.py` picked `ccp_10_3` (alphabetically first). MUCH's own CCP docstring
specifies **δ=24, σ=8**, so `ccp_24_8` is the variant behind the published headline and
should land closer still. The script now defaults to `ccp_24_8`; re-run with
`--all-signals` for the complete table.

### Targets for GRASP

Beat, on the same protocol and the `product` aggregator:

- pooled ROC ≥ 0.772, PR ≥ 0.639, TPR@10%FPR ≥ 0.446
- and do it at ~1-3% runtime overhead against CCP's ~124%
