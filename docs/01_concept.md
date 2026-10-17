# GRASP — the idea, in simple words

**Project:** claim-level hallucination detection without training
**Course:** BCSE306L Artificial Intelligence — Case Study, Topic 16
**Written:** 9 September 2026

---

## 1. The problem

A large language model always answers. It has no internal "I don't know."

Ask it for a biography and it will write six fluent sentences. Five may be true and one may be
completely invented — and the invented one is written with exactly the same confidence as the
true ones. Nothing in the output tells you which is which.

This matters because people are now putting these models in places where a wrong sentence has a
cost: medical summaries, legal drafting, customer support, financial reports. Before you can
build a safe system, you need a component that looks at what the model just wrote and says
*"this part is probably wrong, check it."*

That component is what this project builds.

### Why "claim level" and not "answer level"

Most existing work gives one score for the whole answer. That is not useful for a paragraph.
If the paragraph is 90% correct, a single score of "70% confident" tells you nothing about
*where* the problem is. You would have to re-check everything, which defeats the purpose.

What is actually needed is a score **per atomic claim**:

> "Marie Curie was born in Warsaw"  → score 0.08 (safe)
> "She won the Nobel Prize in 1899" → score 0.91 (check this)

An atomic claim is the smallest complete factual statement you can pull out of a sentence.
The benchmark we use already splits text into these claims for us.

### How hard is this right now

| Setting | Best published detection quality |
|---|---|
| Short answers (one fact) | ~0.85 ROC-AUC — close to solved |
| Long paragraphs, per claim | **~0.77 ROC-AUC** — not solved |

And the benchmark authors state the practical version of that number plainly: the best method
catches only **48% of hallucinations when you allow a 10% false-alarm rate**. Roughly half the
fabrications slip through. Nobody would ship that.

---

## 2. What people do today, and why each approach is limited

There are four families of existing solutions. Understanding why each one is weak is what
motivates our design.

### 2.1 Look at the token probabilities

The simplest idea: when the model wrote this word, how confident was it?

**Why it is weak.** Probability is mostly driven by grammar and word choice, not by facts. The
model is highly confident about "was", "the", "in" — and it is also frequently confident while
being completely wrong. Confidence and correctness are only loosely related.

*Methods in this family:* Maximum Likelihood, Token Likelihood, Token Entropy.

### 2.2 CCP — filter the probabilities using meaning

This is the current best method, and the idea is genuinely clever.

At each word, the model considered several alternatives. CCP asks: **do the alternatives mean
the same thing?**

- Model hesitated between "art" and "painting" → same meaning → just wording, ignore it.
- Model hesitated between "1943" and "1951" → different meaning → real factual doubt, count it.

To decide "same meaning or not", CCP runs a natural language inference (NLI) model on every
single token, comparing the chosen word against each alternative.

**Why it is limited.** It works, but the cost is enormous. On the benchmark's own hardware,
generating all the text took 2,758 seconds; computing CCP on it took **3,230–5,429 seconds**.
Detecting the hallucination costs *more than writing the text did*. That is roughly **124%
overhead**, and it rules the method out for real deployment. It also still only ever looks at
the output probabilities — it never looks inside the model.

*Reference:* Fadeeva et al., ACL Findings 2024.

### 2.3 Generate the answer many times and check for agreement

Ask the same question 10 or 20 times. Facts that are real tend to stay stable across samples;
things the model invented tend to change.

**Why it is limited.** Two reasons. First, it multiplies the cost by 10–20×, which is worse than
CCP. Second, and more seriously: **a model can be consistently wrong**. If it has a wrong fact
baked into its weights, it will repeat that wrong fact every single time, and this method will
score it as highly reliable.

*Methods in this family:* SelfCheckGPT, semantic entropy, SAR, FactSelfCheck, claim–response
entailment.

### 2.4 Train a small classifier on the model's internals

Take the model's hidden states or attention weights while it writes, and train a small
classifier to predict "hallucinated / not hallucinated."

This works well — it is the best-performing family. But it needs **labelled training data**, and
producing that data means paying GPT-4o to fact-check thousands of claims against Wikipedia.
The UHeads paper reports roughly **$100 per base model**, and the resulting classifier does not
transfer well: trained on English, it drops from 0.66 to 0.44 when moved to German.

*Methods in this family:* SAPLMA, Lookback Lens, UHeads, SinkProbe, Factuality Probes.

### 2.5 Summary of the gap

| Family | Cheap? | Works per claim? | Needs labels? | Looks inside the model? |
|---|---|---|---|---|
| Token probability | Yes | Yes | No | No |
| CCP | **No** (124%) | Yes | No | No |
| Multi-sample | **No** (10–20×) | Yes | No | No |
| Trained probes | Yes | Yes | **Yes** | Yes |
| **What is missing** | **Yes** | **Yes** | **No** | **Yes** |

Nobody has built the bottom row. That is the space this project occupies.

---

## 3. Our solution — GRASP

**GRASP = GRounding, Attention Sink, disPersion**

### 3.1 The core idea in one sentence

> Stop looking at *what* the model said. Look at *where it was looking* when it said it.

### 3.2 Why that should work

Inside a transformer, "attention" is literally a record of which earlier words the model was
reading while producing the current word. Two well-established findings make this readable:

1. **Retrieval heads.** Certain attention heads act as retrievers — they reach back and ground
   what is being written in something concrete (the question, the entity name, an earlier
   established fact).

2. **The attention sink.** When a head has nothing relevant to attend to, it dumps its attention
   on the very first token of the sequence. This is a documented behaviour and it functions as
   the transformer's "no-operation" signal.

Put those together and you get a testable hypothesis:

> When the model states a fact it genuinely holds, retrieval heads reach back to the subject and
> to the context that set the fact up. When it fabricates, those heads find nothing — attention
> flattens out and leaks onto the sink. The model is riffing off its own previous words instead
> of retrieving anything.

### 3.3 The three signals

For every generated token we measure three numbers, all read from one forward pass:

| Signal | The question it asks | Hallucination looks like |
|---|---|---|
| **G — Grounding ratio** | Is attention going back to the prompt, or to the model's own freshly written text? | Reading its own output — improvising |
| **S — Sink leakage** | How much attention landed on the "no-op" first token? | High — the head found nothing to retrieve |
| **P — disPersion** | Is attention sharp (one clear source) or smeared across everything? | Smeared — no specific source exists |

### 3.4 The two ideas that make it work per claim

The three signals above are per token. Turning them into a per-claim score is where the real
contribution sits.

**(a) Saliency weighting.** In "was born in 1943", the fact lives entirely in "1943". The rest is
scaffolding. So which tokens actually carry the factual load? *The ones that later tokens attend
back to.* Content words — names, dates, numbers — receive heavy downstream attention; function
words do not.

This matters a lot: it does the same job as CCP's NLI filtering (separating real factual doubt
from mere word choice) but reads it straight out of the attention matrix, for free. **This is the
part that replaces CCP's 124% overhead with roughly 1%.**

**(b) Soft-min aggregation.** A claim is wrong if *any* of its load-bearing tokens is ungrounded.
Averaging over the whole span drowns one bad date in four perfectly fine words, so we use a
soft-minimum over confidence instead of a mean.

### 3.5 Choosing attention heads without any labels

A model has hundreds of heads and only some are useful. Existing methods pick them either by a
fixed rule (RAUQ takes the head that attends most to the previous token) or by training on
labels (UHeads learns the weights).

We pick, for each layer, **the head whose grounding ratio varies most across tokens**, measured
on a small unlabelled sample.

The reasoning: a head that always returns roughly the same value carries no information,
whatever its average. Variance is a pure signal-quality criterion and it needs no ground truth —
which is exactly what keeps the whole method training-free.

### 3.6 One extra detail: reset at claim boundaries

RAUQ propagates confidence through a recurrence that runs across the entire generation. Applied
to a paragraph, doubt about claim 3 bleeds into claim 4 and the scores become correlated
garbage. We reset the recurrence at every claim boundary. This single change is the direct
technical answer to *why sequence-level methods cannot simply be pointed at long text*, and it
gives us a clean ablation.

### 3.7 Three variants, so the project cannot fail

| Variant | What it is | Risk |
|---|---|---|
| **GRASP-0** | Fully training-free, exactly as described above | Moderate — must beat 0.772 unaided |
| **GRASP-fuse** | Rank-average of GRASP-0 with CCP | Low — combining independent signals usually gains |
| **GRASP-probe** | Logistic regression on 5 GRASP features, fit on the training split | Very low — near-certain to beat CCP |

The paper is written so that **any one of the three winning is publishable**:

- GRASP-0 wins → "training-free and 100× cheaper than the previous best."
- Only GRASP-fuse wins → "attention carries information that token probabilities do not."
- Only GRASP-probe wins → "five interpretable features match a trained transformer head, and
  transfer across languages."

---

## 4. Is it novel? — the honest check

### 4.1 What we searched

Prior art was searched under: attention-based claim-level detection, training-free claim-level
UQ, RAUQ extensions, attention-sink factuality signals, retrieval heads for hallucination, and
papers citing the MUCH benchmark. Ten near neighbours were examined in detail.

### 4.2 The nearest work, and why none of it blocks us

| Paper | Venue | Claim level? | Training free? | Uses attention? | Long form? |
|---|---|---|---|---|---|
| CCP (Fadeeva et al.) | ACL Findings 2024 | Yes | Yes | **No** | Yes |
| UHeads | EMNLP 2025 | Yes | **No** | Yes | Yes |
| RAUQ | ICML 2026 | **No** (sequence) | Yes | Yes | Partly |
| SinkProbe | ICML 2026 | **No** (sequence) | **No** | Yes | **No** |
| Attention Divergence (2605.05025) | 2026 preprint | **No** (answer) | **No** | Yes | **No** |
| Factuality Probes | EMNLP Findings 2025 | Yes | **No** | **No** (hidden states) | Yes |
| Real-time entity detection (2509.03531) | 2025 preprint | Token | **No** | **No** (hidden states) | Yes |
| FactSelfCheck | EACL Findings 2026 | Yes | Yes | **No** (20 samples) | Yes |
| Claim–response entailment (2602.17431) | 2026 preprint | Yes | Yes | **No** (K samples) | Yes |
| Lookback Lens | EMNLP 2024 | **No** | **No** | Yes | Partly |
| **GRASP (ours)** | — | **Yes** | **Yes** | **Yes** | **Yes** |

**Verdict: the combination is novel.** No published method is simultaneously claim-level,
training-free, and attention-based.

### 4.3 Why this is a *good* kind of novelty

This is not "nobody tried it, probably because it does not work." It is the opposite:

- **UHeads proves** attention features predict hallucination *at claim level*.
- **SinkProbe proves** attention-sink mass predicts hallucination.
- **RAUQ proves** attention heads can be selected *without labels*.

Three independent papers have each demonstrated one third of what we need. The signal is
established. The only open question is whether it can be extracted without labels at claim
level — and that question is worth a paper whichever way it resolves.

There is even an explicit invitation: the Attention Divergence paper's own future-work section
asks for "methods that eliminate probe dependency entirely by establishing direct relationships
between specific heads and hallucinations." That is a description of GRASP.

### 4.4 Is there a base paper?

There is no single paper we are extending. There are three, with different roles:

| Role | Paper | What we take from it |
|---|---|---|
| **Method base** | RAUQ — *Efficient Hallucination Detection Using Uncertainty-Aware Attention Heads*, arXiv 2505.20045 | The idea of label-free attention-head selection. We move it from sequence level to claim level and change the selection criterion. |
| **Benchmark base** | MUCH — *A Multilingual Claim Hallucination Benchmark*, arXiv 2511.17081 | The dataset, the claim segmentation, the evaluation protocol, the runtime accounting. |
| **Baseline to beat** | CCP — Fadeeva et al., ACL Findings 2024 | The number: 0.772 ROC-AUC at 124% overhead. |

Being explicit about this in the paper is a strength, not a weakness. Reviewers respond well to
"we combine X's mechanism with Y's granularity and here is exactly what is new"; they respond
badly to vague claims of total originality.

---

## 5. How we will show it beats the existing work

### 5.1 The benchmark

**MUCH** (Apache-2.0, École Polytechnique). 4,873 samples, 20,751 claims, four languages
(en / fr / es / de), four open-weight models (Llama-3.2-3B, Llama-3.1-8B, Ministral-8B,
Gemma-3-4B). Training split of 4,673 auto-annotated samples; test split of 200 samples annotated
by two human experts.

It was chosen because it removes almost all of the usual work:

- Claim boundaries are already computed and shipped (`token_chunks`).
- Labels are already there (`labels`).
- The top-24 token probabilities are shipped, so every probability baseline runs on CPU.
- The exact generated token IDs are shipped, so we can push text back through the model and get
  attention **without any sampling** — completely deterministic.
- Baseline signals are downloadable, so we do not recompute CCP on an A100 we do not have.
- The repository explicitly invites new white-box claim-level methods and defines the API.

### 5.2 What "beating it" means concretely

| Metric | CCP (to beat) | GRASP target |
|---|---|---|
| ROC-AUC | 0.772 | ≥ 0.78 |
| PR-AUC | 0.639 | ≥ 0.66 |
| TPR at 10% FPR | 0.48 | ≥ 0.55 |
| Runtime overhead vs generation | ≈ 124% | ≈ 1–3% |

The runtime column is half the contribution. Even *matching* CCP at one-hundredth of the cost is
a publishable result on a benchmark whose stated purpose is realistic deployment constraints.

### 5.3 Tables the paper will contain

1. Main results — all baselines plus three GRASP variants, with the runtime column.
2. Per-language — en / fr / es / de, showing the transfer penalty that supervised methods pay.
3. Per-model — four base models, showing it is not tuned to one architecture.
4. Ablation — each signal, the saliency weighting, the aggregation, the head-selection rule, the
   boundary reset, one at a time.
5. Risk–coverage curves — the abstention framing that ties the paper back to Topic 16.

### 5.4 Honest risks

| Risk | What we do about it |
|---|---|
| GRASP-0 does not beat CCP unaided | Lead with GRASP-fuse; the story becomes complementarity |
| Gemma-3's sliding-window attention breaks the prompt/generation split | Report Gemma separately or drop it; three models is enough |
| Automatic labels are noisy | Headline the 200 human-annotated test samples; tune only on the training split |
| Someone publishes the same idea first | Post the preprint the week the main table is stable |
| Reviewer says head selection is tuned on test | Freeze all hyperparameters on the training split, state it explicitly |

---

## 6. Where this gets submitted

- **arXiv (cs.CL)** — immediately, to establish priority.
- **UncertaiNLP @ EMNLP 2026** — the 3rd Workshop on Uncertainty-Aware NLP. Exact topical match,
  ACL Anthology proceedings, the most realistic first acceptance.
- **ACL / EMNLP Student Research Workshop** — designed for a student first-author paper with one
  clean contribution.
- **ACL / EMNLP short paper or Findings** — reachable if GRASP-0 wins outright.
- **Scopus journals** as fallback — Knowledge-Based Systems, Neurocomputing, Expert Systems with
  Applications, IEEE Access.

---

## 7. What must be verified before submission

Everything below was collected from paper text and must be checked against the published PDFs
before it enters the manuscript. This is also required by the course checklist.

- CCP's exact ROC-AUC and PR-AUC on MUCH, per language.
- The 2,758 s generation time and the CCP timings used for the overhead figure.
- Venue and year for every citation — several of these were preprints a year before publication.
- Re-run the novelty search on the day writing begins. This subfield moves in weeks.
