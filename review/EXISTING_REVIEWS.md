# Existing reviews: comparison draft (v2, after full read of R1 and R2)

**Status:** R1 and R2 were read in full from the PDFs in `papers/` (every body section; reference lists skipped).
R3-R5 are candidates I have not opened and are still `TO FILL`. A correction from v1 is listed at the end.

## Comparison table

| Review | Year / venue | Scope and period | Taxonomy? | Evaluation analysis? | What is missing / what this paper adds |
|---|---|---|---|---|---|
| **R1** Wen et al., *Know Your Limits: A Survey of Abstention in LLMs* (arXiv 2407.18418) | 2025. PDF is arXiv v3 dated 12 Feb 2025; **no journal or conference name appears in the PDF**. Published version, if any: verify | Abstention as refusal to answer, seen from three views: query answerability, model confidence, human-value alignment. General-purpose chatbots. Methods organised by lifecycle (pretraining, alignment, inference). **No covered period, databases, search terms or inclusion dates are stated**; section 2.3 only says papers are included if they fall under one of the three views. Pretraining stage: "no existing research" found | Yes: lifecycle (alignment: SFT, preference learning; inference: input-, in-, output-processing) crossed with the three perspectives | **Surveys** benchmarks (query-, model-knowledge- and human-value-centric datasets) and metrics (abstention confusion-matrix metrics, coverage, AURCC, AUACC, AUROC, model-based and human evaluation). **No experiments of its own.** Notes that no single metric captures abstention and that benchmarks mostly cover one perspective | Safety refusal dominates. Hallucination detectors appear only as "probing inner state" and "uncertainty estimation" (it cites Kadavath, Azaria and Mitchell, Chen/EigenScore, Kuhn, Duan). No detection granularity (claim, span), no attention-head methods, no claim-level benchmarks, no comparability or reproducibility audit |
| **R2** Kang et al., *Uncertainty Quantification for Hallucination Detection in LLMs: Foundations, Methodology, and Future Directions* (arXiv 2510.12040) | 2025. **Preprint**: v1, 14 Oct 2025, header "A Preprint, October 15, 2025". No peer-reviewed venue in the PDF | UQ as a hallucination detector on open-ended QA, short-form and long-form. Explicitly excludes interactive agents and mitigation. **No covered period, databases or search method stated**; its method table (Table 1) lists 35 methods, newest 2025 (e.g. RAUQ) | Yes, four independent axes: conceptual approach (token probability, output consistency, internal state, self-checking), sampling requirement, model accessibility (black/gray/white box), training reliance (supervised/unsupervised) | **Yes, with its own experiments**: TriviaQA (1,000), GSM8K (1,000), FactScore-Bio; LLaMA-3-8B and GPT-4o-mini; 16 methods; AUROC and PRR (AUPRC discussed). Short-form: supervised LARS and SAPLMA best (AUROC about 0.85-0.86 on TriviaQA with LLaMA-3-8B); SAR best unsupervised white-box. FactScore-Bio: all methods drop, verbalised confidence best (AUROC 0.698 / 0.717). The one attention method tested (Attention Scores) is near chance on TriviaQA (AUROC 0.523) | Claim-level scoring is raised only as a **future direction** ("UQ in long-form QA"); granularity is not a taxonomy axis and the unit scored on FactScore-Bio is not stated (verify). Attention methods are folded into "internal state" and only one is evaluated. Two models, three datasets. No audit of code/data release, variance or cross-paper comparability. States as limitations: knowledge boundary/temporal drift, non-interpretable score scales, cost of training ensembles |
| **R3** Huang et al., *A Survey on Hallucination in LLMs*, ACM TOIS (candidate) | verify | TO FILL | TO FILL | TO FILL | TO FILL |
| **R4** Shorinwa et al., *A Survey on Uncertainty Quantification of LLMs*, ACM CSUR (candidate; cited in S01) | verify | TO FILL | TO FILL | TO FILL | TO FILL |
| **R5** Geng et al., *A Survey of Confidence Estimation and Calibration in LLMs*, NAACL 2024 (candidate) | verify | TO FILL | TO FILL | TO FILL | TO FILL |

Optional: Ji et al., *Survey of Hallucination in NLG*, ACM CSUR 2023 (pre-LLM scope; background only).
R3-R5 come from memory and from citations in your PDFs; titles, venues, years and DOIs are unverified.

## What this review can claim to add (each point checked against R1 and R2)

1. **Detection granularity as an axis.** Neither R1 nor R2 classifies methods by the unit they score. R2 names claim-level scoring
   as future work. Your matrix: 5 of 24 detector papers are claim-level; 12 score whole responses.
2. **A split between attention and hidden-state signals.** R2 *does* cover attention methods (Attention Scores, Lookback Lens, RAUQ) but
   groups them with hidden states under "internal state". Your Taxonomy_Draft separates them (7 attention vs 5 hidden-state papers).
   This is a finer split, **not** a new topic. Do not claim attention methods are uncovered.
3. **Comparability audit.** Each study is coded Direct / Conditional / Non-comparable (5 / 21 / 3 of 29). Neither R1 nor R2 reports such an audit.
4. **Reproducibility audit.** Code, data and variance reporting are counted per study (Gap_Table G5, G6). Neither survey does this.
5. **Overlap to acknowledge.** R2's four axes (sampling, accessibility, training reliance) overlap your supervision axis. Cite R2 for that part
   instead of presenting it as new.

**Do not write "first review".** No systematic search for competing reviews has been done. Use: "complements R1 and R2 by ...".

## Correction to v1 of this file
v1 said R2's method families "do not list attention separately" and implied attention methods were missing. After the full read: R2 includes
them under internal-state methods. The wording above replaces it.

## Still open
- Read R3-R5 (or replace them); keep at least three fully filled reviews.
- Neither R1 nor R2 states a covered period; write "not stated" in the manuscript, not a guessed range.
- Check whether R1 has a published version (the PDF shows none) and whether R2 has moved beyond preprint.
