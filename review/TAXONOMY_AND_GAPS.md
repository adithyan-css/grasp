# Taxonomy and gap draft (from the 29-row matrix)

**Status:** draft. The three taxonomy axes below are my coding from each paper's task and method text, not columns in the workbook; the full-text read may move papers between cells. Gaps marked *Weak* depend on attributes that are not workbook columns (cost, language) or were checked in main text only. All `TO VERIFY` cells in the matrix are now filled from main-text reads; appendices were not read.

## 1. Taxonomy rule set

| Axis | Categories | Rule |
|---|---|---|
| Signal source | Output probabilities; Sampled consistency; Hidden states; Attention; Self-report | What the detector reads: logits, several sampled answers, layer activations, attention maps, or the model's own stated confidence |
| Supervision | Training-free; Supervised | Supervised if a classifier or head is fit on labelled hallucination data (including pre-trained heads) |
| Granularity | Response; Sentence; Claim; Span/token/entity | The unit that receives a score |
| Outside the axes | Evaluation resource (S01, S22, S24, S28); Mechanistic analysis (S17) | Papers that supply data/metrics or explain mechanisms, not detectors |

## 2. Counts (24 detector papers)

| Signal source | Training-free | Supervised | Total |
|---|---|---|---|
| Output probabilities | 2 | 0 | 2 |
| Sampled consistency | 9 | 0 | 9 |
| Hidden states | 1 | 4 | 5 |
| Attention | 1 | 6 | 7 |
| Self-report | 0 | 1 | 1 |
| **Total** | 13 | 11 | 24 |

Granularity: Claim 5, Response 13, Sentence 2, Span / token / entity 4.

## 3. Paper assignments

| ID | Paper | Signal | Supervision | Granularity |
|---|---|---|---|---|
| S02 | Fadeeva, E. et al. 2024 | Output probabilities | Training-free | Claim |
| S03 | Vazhentsev, A. et al. 2026 | Attention | Training-free | Response |
| S04 | Shelmanov, A. et al. 2025 | Attention | Supervised | Claim |
| S05 | Binkowski, J. et al. 2026 | Attention | Supervised | Response |
| S06 | Kuhn, L. et al. 2023 | Sampled consistency | Training-free | Response |
| S07 | Manakul, P. et al. 2023 | Sampled consistency | Training-free | Sentence |
| S08 | Kadavath, S. et al. 2022 | Self-report | Supervised | Response |
| S09 | Duan, J. et al. 2024 | Output probabilities | Training-free | Response |
| S10 | Nikitin, A. et al. 2024 | Sampled consistency | Training-free | Response |
| S11 | Qiu, X. & Miikkulainen, R. 2024 | Sampled consistency | Training-free | Response |
| S12 | Vashurin, R. et al. 2025 | Sampled consistency | Training-free | Response |
| S13 | Azaria, A. & Mitchell, T. 2023 | Hidden states | Supervised | Sentence |
| S14 | Chen, C. et al. 2024 | Hidden states | Training-free | Response |
| S15 | Chuang, Y.-S. et al. 2024 | Attention | Supervised | Span / token / entity |
| S16 | Kossen, J. et al. 2024 | Hidden states | Supervised | Response |
| S18 | Sun, Z. et al. 2025 | Attention | Supervised | Span / token / entity |
| S19 | Ogasa, Y. & Arase, Y. 2025 | Attention | Supervised | Span / token / entity |
| S20 | Obeso, O. et al. 2026 | Hidden states | Supervised | Span / token / entity |
| S21 | van Dijk, G. 2026 | Attention | Supervised | Response |
| S23 | Zhang, C. et al. 2024 | Sampled consistency | Training-free | Response |
| S25 | Bouchard, D. et al. 2026 | Sampled consistency | Training-free | Claim |
| S26 | Sawczyn, A. et al. 2026 | Sampled consistency | Training-free | Claim |
| S27 | Han, J. et al. 2025 | Hidden states | Supervised | Claim |
| S29 | Abbasi-Yadkori, Y. et al. 2024 | Sampled consistency | Training-free | Response |

## 4. Gap table (also in the workbook, sheet Gap_Table)

| ID | Gap | Evidence | Why it matters | Future study | Strength |
|---|---|---|---|---|---|
| G1 | Claim-level detection is rare: only 5 of the 24 method papers score at claim/fact level; 13 score whole responses or short answers | 5 of 29 (17%) (IDs S02, S04, S25, S26, S27). Response-level: 13 of 24 method papers. Granularity is MY classification from the task/method text, not a matrix column | A paragraph with one false claim is rejected or accepted as a whole; users cannot see where to check | Evaluate response-level methods (S03, S06, S10, S12, S14) on a claim-level benchmark such as MUCH with one shared protocol | Moderate (counts rest on my granularity coding; confirm after full reads) |
| G2 | No claim-level method is both training-free and non-sampling and attention-based: attention methods are 7 papers, only 1 is training-free (S03) and only 1 works at claim level (S04, supervised) | Attention-based 7 of 24 method papers; training-free: S03; claim-level: S04. Training-free claim-level methods are 1 non-sampling (S02, probabilities plus NLI); the other claim-level methods are sampling-based (S25, S26) or supervised (S04, S27) | Sampling costs 5-20 extra generations (S06, S09, S10 report it); supervised heads need labelled data per model | Test a label-free attention score at claim level against CCP (S02) and UHeads (S04) on MUCH under S01's protocol | Moderate (classification-dependent; sampling counts need a recheck) |
| G3 | Cost is rarely measured: only 5 of 29 papers report a runtime, FLOP or sample-count comparison | 5 of 29 (17%) (IDs S01, S03, S09, S16, S27). Cost is only stated qualitatively elsewhere. Cost is not a workbook column; add one so this count can be audited | Deployment decisions need overhead relative to generation time; S01 shows CCP 124% on its hardware, a number no other paper in the set can be compared with | Report overhead as % of generation time on a stated GPU for all methods, per token and per answer | Weak until verified (cost wording not fully scanned) |
| G4 | Transfer is seldom tested: only 6 of 29 papers test cross-domain, cross-model, cross-language or cross-generator transfer | 6 of 29 (21%) (IDs S04, S08, S15, S16, S20, S27). The 'Robustness' column says Yes for 21 rows, but most of those only mean several LLMs, not a distribution shift; I used the narrower definition here | Supervised probes and heads (S04, S15, S16, S20, S27) are the strongest on paper but need labels for each new setting | Train on one benchmark and evaluate on another with the same protocol; report in-domain vs zero-shot side by side | Moderate (the narrow definition of transfer is mine) |
| G5 | Statistical reporting is thin: only 9 of 29 papers report variance, intervals or a significance test for their main results | 9 of 29 (31%) (IDs S02, S05, S10, S11, S13, S20, S21, S27, S28). Others report single numbers (N/R or No). Checked by searching each paper's main text for ±, SD, CI, seeds, bootstrap and significance tests; appendices were not read, so a few may report variance there | Differences of 0.01-0.03 AUROC between methods cannot be interpreted without intervals (S05's gains are often within one SD; S18's are under 0.02 with no seeds), and test sets are small (e.g. S07 has 238 passages) | Bootstrap intervals by generation and paired deltas for every compared method | Moderate (main-text check; appendices unread) |
| G6 | Data release lags code release: 23 of 29 link code but only 9 release their own data | Code 23 of 29 (79%); own data 9 of 29 (31%) (IDs S01, S04, S06, S07, S13, S20, S24, S25, S26). N/R means no link found in my scan, not proven absence | Reproduction of supervised methods needs the training labels, not just the code | Release claim-level labels and per-token features alongside code, as S01 does | Moderate |
| G7 | Results cannot be ranked across papers: no shared protocol | Comparability coded Direct 5, Conditional 21, Non-comparable 3 of 29. Metrics: AUROC/ROC-AUC 22, PR-AUC family 8, PRR 3, accuracy/F1 6 (papers can appear in several) | Reviews that rank methods from mixed tables mislead; the handout forbids leaderboards from mixed settings | Adopt MUCH (S01) or LM-Polygraph (S28) as a shared harness and re-run published methods | Strong (visible across the whole matrix) |
| G8 | Multilingual evaluation is almost absent: only 4 of 29 evaluate beyond English | 4 of 29 (14%) (IDs S01, S02, S04, S28). Language is not a workbook column; the other 25 were judged English-only from their dataset descriptions, so add a language column to confirm | Deployment is multilingual; S01 itself lists limited languages as a limitation | Evaluate on MUCH's four languages and add at least one non-European language | Weak until verified |

## 5. Handout's four gap tests

- **Can I state a count?** Yes for all eight; G3 (cost) and G8 (language) rest on attributes that are not workbook columns, so add those columns to make them auditable.
- **Visible in three or more papers?** G1, G4, G6, G7 yes. G2 rests on the absence of a combination, so it is a structural gap, not a frequency.
- **Concrete study named?** Yes in each row.
- **Do the same papers show the opposite?** Not checked yet. Known counter-evidence to look at: S27 and S20 show supervised probes transfer reasonably to other generators, which weakens G4 for hidden-state probes. S15 reports Lookback Lens AUROC up to 91 in-domain, while S19 finds its token-level F1 near 0 on RAGTruth at a 0.5 threshold: the conflict comes from metric and threshold choice, which supports G7.
- **Coding change after full reads:** S23 (LUQ) moved from sentence to response level, since it is evaluated by per-response correlation; S28 added to the multilingual list (G8), since its claim-level benchmark covers English, Arabic, Russian and Chinese.
