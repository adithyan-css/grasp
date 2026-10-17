# Reading list — 33 papers with direct links

Every paper referenced while choosing the topic, checking novelty and designing GRASP.

**Status column:** ✅ = I opened this source during the research and the identifier is
confirmed. ⚠️ = title and venue are right but I did not re-confirm the arXiv ID this
session — check it before citing.

Bulk download: `python scripts/05_download_papers.py --out papers/` fetches every PDF and
pulls the **official** BibTeX from arXiv and ACL Anthology.

---

## Tier 1 — read these in full (5)

These five carry the argument. Everything else is supporting.

| # | Paper | Venue | PDF | Status |
|---|---|---|---|---|
| 1 | **MUCH: A Multilingual Claim Hallucination Benchmark** — Dentan, Canesse, Buscaldi, Shabou, Vanier | preprint 2025 | [arxiv.org/pdf/2511.17081](https://arxiv.org/pdf/2511.17081) | ✅ |
| 2 | **Fact-Checking the Output of LLMs via Token-Level Uncertainty Quantification** (CCP) — Fadeeva et al. | ACL Findings 2024 | [aclanthology.org/2024.findings-acl.558.pdf](https://aclanthology.org/2024.findings-acl.558.pdf) · [arXiv](https://arxiv.org/pdf/2403.04696) | ✅ |
| 3 | **Efficient Hallucination Detection Using Uncertainty-Aware Attention Heads** (RAUQ) — Vazhentsev et al. | ICML 2026 | [arxiv.org/pdf/2505.20045](https://arxiv.org/pdf/2505.20045) | ✅ |
| 4 | **A Head to Predict and a Head to Question** (UHeads) | EMNLP 2025 | [aclanthology.org/2025.emnlp-main.1809](https://aclanthology.org/2025.emnlp-main.1809/) · [arXiv](https://arxiv.org/pdf/2505.08200) | ✅ |
| 5 | **Attention Sinks as Internal Signals for Hallucination Detection** (SinkProbe) | ICML 2026 | [arxiv.org/pdf/2604.10697](https://arxiv.org/pdf/2604.10697) | ✅ |

**Why these five.** 1 is the benchmark and the number to beat. 2 is the method to beat.
3, 4 and 5 are the three papers that each solved a different two-thirds of the problem —
together they are the whole novelty argument, so you must be able to say precisely what
each one does and does not do.

---

## Tier 2 — foundations of LLM uncertainty (10)

| # | Paper | Venue | PDF | Status |
|---|---|---|---|---|
| 6 | **Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation** — Kuhn, Gal, Farquhar | ICLR 2023 | [arxiv.org/pdf/2302.09664](https://arxiv.org/pdf/2302.09664) | ✅ |
| 7 | **Detecting hallucinations in LLMs using semantic entropy** — Farquhar et al. | Nature 630, 2024 | [nature.com](https://www.nature.com/articles/s41586-024-07421-0) · [free PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750) | ✅ |
| 8 | **SelfCheckGPT** — Manakul, Liusie, Gales | EMNLP 2023 | [aclanthology.org/2023.emnlp-main.557.pdf](https://aclanthology.org/2023.emnlp-main.557.pdf) | ✅ |
| 9 | **Language Models (Mostly) Know What They Know** — Kadavath et al. | preprint 2022 | [arxiv.org/pdf/2207.05221](https://arxiv.org/pdf/2207.05221) | ✅ |
| 10 | **Shifting Attention to Relevance** (SAR) — Duan et al. | ACL 2024 | [arxiv.org/pdf/2307.01379](https://arxiv.org/pdf/2307.01379) | ✅ |
| 11 | **Kernel Language Entropy** — Nikitin et al. | NeurIPS 2024 | [arxiv.org/pdf/2405.20003](https://arxiv.org/pdf/2405.20003) | ✅ |
| 12 | **Semantic Density** — Qiu, Miikkulainen | NeurIPS 2024 | [arxiv.org/pdf/2405.13845](https://arxiv.org/pdf/2405.13845) | ✅ |
| 13 | **CoCoA: bridging confidence and consistency** — Vashurin et al. | NeurIPS 2025 | [arxiv.org/pdf/2502.04964](https://arxiv.org/pdf/2502.04964) | ✅ |
| 14 | **Generating with Confidence** — Lin, Trivedi, Sun | TMLR 2024 | [search arXiv](https://arxiv.org/a/search?searchtype=all&query=Generating+with+Confidence+uncertainty+black-box+LLM) | ⚠️ ID unverified |
| 15 | **Just Ask for Calibration** — Tian et al. | EMNLP 2023 | [search ACL Anthology](https://aclanthology.org/search/?q=Just+Ask+for+Calibration) | ⚠️ ID unverified |

SAR (10) is worth close reading: it also reweights tokens by relevance, but from the
output distribution rather than from attention. It is the nearest conceptual cousin to
GRASP's saliency weighting and a reviewer will ask you to distinguish them.

---

## Tier 3 — internal states and attention (9)

The mechanistic evidence that GRASP's signal exists.

| # | Paper | Venue | PDF | Status |
|---|---|---|---|---|
| 16 | **The Internal State of an LLM Knows When It's Lying** (SAPLMA) — Azaria, Mitchell | EMNLP Findings 2023 | [arxiv.org/pdf/2304.13734](https://arxiv.org/pdf/2304.13734) | ✅ |
| 17 | **INSIDE: LLMs' internal states retain the power of hallucination detection** (EigenScore) — Chen et al. | ICLR 2024 | [arxiv.org/pdf/2402.03744](https://arxiv.org/pdf/2402.03744) | ✅ |
| 18 | **Lookback Lens** — Chuang et al. | EMNLP 2024 | [arxiv.org/pdf/2407.07071](https://arxiv.org/pdf/2407.07071) · [MIT copy](https://sls.csail.mit.edu/publications/2024/YChuang_EMNLP_2024.pdf) | ✅ |
| 19 | **Semantic Entropy Probes** — Kossen et al. | preprint 2024 | [arxiv.org/pdf/2406.15927](https://arxiv.org/pdf/2406.15927) | ✅ |
| 20 | **Retrieval Head Mechanistically Explains Long-Context Factuality** — Wu et al. | ICLR 2025 | [arxiv.org/pdf/2404.15574](https://arxiv.org/pdf/2404.15574) | ✅ |
| 21 | **ReDeEP: Detecting Hallucination in RAG via Mechanistic Interpretability** | ICLR 2025 | [arxiv.org/pdf/2410.11414](https://arxiv.org/pdf/2410.11414) | ✅ |
| 22 | **Hallucination Detection using Multi-View Attention Features** | preprint 2025 | [arxiv.org/pdf/2504.04335](https://arxiv.org/pdf/2504.04335) | ✅ |
| 23 | **Real-Time Detection of Hallucinated Entities in Long-Form Generation** | preprint 2025 | [arxiv.org/pdf/2509.03531](https://arxiv.org/pdf/2509.03531) | ✅ |
| 24 | **Detecting Hallucinations via Internal Attention Divergence Signals** | preprint 2026 | [arxiv.org/pdf/2605.05025](https://arxiv.org/pdf/2605.05025) | ✅ |

Paper 20 is the mechanistic basis for the grounding signal. Paper 24's future-work
section explicitly asks for "methods that eliminate probe dependency entirely" — quote it
in your introduction, it is a published invitation to build GRASP.

---

## Tier 4 — claim-level detection and benchmarks (6)

| # | Paper | Venue | PDF | Status |
|---|---|---|---|---|
| 25 | **FActScore** — Min et al. | EMNLP 2023 | [aclanthology.org/2023.emnlp-main.741.pdf](https://aclanthology.org/2023.emnlp-main.741.pdf) | ✅ |
| 26 | **LUQ: Long-text Uncertainty Quantification for LLMs** — Zhang et al. | EMNLP 2024 | [aclanthology.org/2024.emnlp-main.299.pdf](https://aclanthology.org/2024.emnlp-main.299.pdf) | ✅ |
| 27 | **RAGTruth** — Niu et al. | ACL 2024 | [aclanthology.org/2024.acl-long.585](https://aclanthology.org/2024.acl-long.585/) · [arXiv](https://arxiv.org/pdf/2401.00396) | ✅ |
| 28 | **Fine-Grained UQ for Long-Form LM Outputs: A Comparative Study** | preprint 2026 | [arxiv.org/pdf/2602.17431](https://arxiv.org/pdf/2602.17431) | ✅ |
| 29 | **FactSelfCheck: Fact-Level Black-Box Hallucination Detection** | EACL Findings 2026 | [aclanthology.org/2026.findings-eacl.296.pdf](https://aclanthology.org/2026.findings-eacl.296.pdf) | ✅ |
| 30 | **Simple Factuality Probes Detect Hallucinations in Long-Form Generation** | EMNLP Findings 2025 | [aclanthology.org/2025.findings-emnlp.880.pdf](https://aclanthology.org/2025.findings-emnlp.880.pdf) | ✅ |
| 31 | **HaluEval** — Li et al. | EMNLP 2023 | [search ACL Anthology](https://aclanthology.org/search/?q=HaluEval) | ⚠️ ID unverified |

29 and 30 are the two nearest competitors found in the second novelty sweep. Both are
claim-level; 29 needs 20 samples, 30 needs labelled training data. Neither uses attention.
That contrast is your novelty table.

---

## Tier 5 — abstention, benchmarking, and one cautionary tale (5)

| # | Paper | Venue | PDF | Status |
|---|---|---|---|---|
| 32 | **Know Your Limits: A Survey of Abstention in LLMs** — Wen et al. | TACL 2025 | [arxiv.org/pdf/2407.18418](https://arxiv.org/pdf/2407.18418) | ✅ |
| 33 | **Benchmarking UQ Methods for LLMs with LM-Polygraph** — Vashurin et al. | TACL 2025 | [aclanthology.org/2025.tacl-1.11](https://aclanthology.org/2025.tacl-1.11/) · [arXiv](https://arxiv.org/pdf/2406.15627) | ✅ |
| 34 | **Mitigating LLM Hallucinations via Conformal Abstention** — Yadkori et al. | NeurIPS 2024 | [arxiv.org/pdf/2405.01563](https://arxiv.org/pdf/2405.01563) | ✅ |
| 35 | **UQ for Hallucination Detection: Foundations, Methodology, Future Directions** | preprint 2025 | [arxiv.org/pdf/2510.12040](https://arxiv.org/pdf/2510.12040) | ✅ |
| 36 | **The Mirage of Performance Gains: Why Contrastive Decoding Fails** | preprint 2025 | [arxiv.org/pdf/2504.10020](https://arxiv.org/pdf/2504.10020) | ✅ |

Paper 32 frames the whole review (Topic 16 is abstention). Paper 35 is where the
long-form performance gap is quantified — that quantified gap is your motivation
paragraph.

Paper 36 is not about your method at all, and it is the most instructive item on the
list. It shows an entire subfield's reported gains were artefacts of a biased benchmark
and a decoding side-effect. Read it as a warning about your own protocol discipline, and
cite it in your limitations section — reviewers respect an author who has clearly read it.

---

## Assignment arithmetic

The brief wants **at least 40 references, majority primary studies**. This list has 36,
of which roughly 30 are primary. To close the gap, pull from the citation chains of Tier 1
— specifically the related-work sections of MUCH, RAUQ and UHeads. Ten more comes easily
and they will be the right ten, because they are what the papers you are extending
consider their own context.

Two rules from the brief that matter here:

- **Cite the original study for a method, the survey for the shape of a field.** So cite
  CCP for CCP, and 35 for "the field is organised into four signal families".
- **Do not cite a paper you have not read sufficiently to verify the claim.** Tier 1 in
  full, Tiers 2–5 at abstract-plus-results depth is a defensible standard — but if you
  make a specific numerical claim about a paper, open that paper's table first.

## Verification

Three IDs (14, 15, 31) are unverified — I have given you search links rather than a
possibly-wrong number. Everything else was opened during the research.

Even so: **re-check every venue, year and DOI against the publisher's page before
submitting.** Several of these appeared as preprints a year or more before their
conference version, and preprint-versus-published status is a distinction the brief
explicitly asks you to handle. The download script pulls official BibTeX, which is the
cheapest way to get this right.
