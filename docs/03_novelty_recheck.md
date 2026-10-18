# Novelty re-check — 17 October 2026

`01_concept.md` §7 asked for the prior-art search to be re-run on the day writing begins.
This is that re-run, made against the method as it now stands (GRASP-v2), not the
original training-free GRASP-0.

## What was searched

- claim-level hallucination detection with attention heads, long-form, 2026;
- work citing or evaluating on MUCH (arXiv 2511.17081), and anything reported to beat CCP there;
- attention-sink / grounding probes combined with token probabilities, per claim.

## What came back

| Work | Signals | Granularity | Supervised | Relation to GRASP-v2 |
|---|---|---|---|---|
| **UHeads / LUH** — Shelmanov et al., *A Head to Predict and a Head to Question*, EMNLP 2025 (arXiv 2505.08200) | attention maps + token probabilities | claim | yes (trained transformer head) | **Closest prior work.** Same signal families and granularity. |
| Single-Pass Uncertainty Heads, Persian medical — arXiv [2610.03482](https://arxiv.org/abs/2610.03482), 2 Oct 2026 | attention maps + token probabilities (LUH adapted) | claim | yes | Applies LUH to Persian medical models; ROC ≈ 0.78. Not on MUCH, no CCP comparison. |
| RAUQ — arXiv [2505.20045](https://arxiv.org/abs/2505.20045), ICML 2026 | attention (recurrent) | sequence | no | Training-free head selection; not claim-level. |
| SinkProbe — arXiv [2604.10697](https://arxiv.org/abs/2604.10697), ICML 2026 | attention-sink mass, all heads | response | yes (logistic regression) | Sink signal, but response-level. |
| Temporal Multi-Signal Fusion — arXiv [2608.18115](https://arxiv.org/abs/2608.18115), Jun 2026 | text statistics + NLI + surprisal; **no internals** | token | yes (BiGRU) | RAGTruth AUC ≈ 0.84; black-box, not attention. |
| Cross-Layer Attention Probing (CLAP) — arXiv [2509.09700](https://arxiv.org/abs/2509.09700), ECAI-W 2025 | residual-stream activations | response | yes | Activation probe, response-level. |
| Real-time entity detection — arXiv [2509.03531](https://arxiv.org/abs/2509.03531) | hidden states | entity/token | yes | Long-form, but entity-level and hidden states. |
| MUCH itself — arXiv [2511.17081](https://arxiv.org/abs/2511.17081) | benchmark | claim | — | CCP is still its best reported baseline (0.772); no follow-up beating it was found. |

## Consequence for the claims we can make

**Drop:** "no published method is claim-level, attention-based and cheap" — UHeads/LUH is
claim-level and uses attention plus token probabilities. GRASP-v2 is in the same family as
UHeads, not a new family.

**Keep, and state precisely:**

1. **First reported result beating CCP on MUCH** — 0.924 vs 0.770 ROC on the human-annotated
   test split, on all four models, without running CCP's NLI model (`results/FINAL_RESULTS_v2.md`).
2. **A much lighter learner than UHeads:** three interpretable attention statistics
   (grounding, sink, dispersion) from 4 heads per layer, chosen with no labels, pooled over
   relative depth, plus the token probabilities the benchmark already ships, fed to a
   gradient-boosted tree on a CPU. UHeads trains a transformer head on attention features.
3. **Cross-model transfer:** fitted on three models and scored on the fourth, GRASP-v2 beats
   CCP on 3 of 4 unseen models and is within 0.01 on the fourth. The concept doc cites
   UHeads losing 0.66 → 0.44 across languages. That is a different transfer setting, so this is
   suggestive, not a head-to-head comparison.
4. **Measured cost:** 34–38% of generation time on a T4.

**Update (18 Oct 2026): comparison done** with the authors' released pre-trained heads, on
identical claims (`results/uhead_compare.json`). With in-domain labels GRASP-v2 beats them on MUCH
(Llama-3.1-8B: 0.937 vs 0.570) and on RAGTruth (0.827 vs 0.680). Zero-shot, the pre-trained
UHead transfers to RAGTruth better than MUCH-trained GRASP (0.680 vs 0.639). Still owed: UHeads
retrained on MUCH's training split, for an identical-data comparison.

Search date: 2026-10-17. Re-run again the week the paper is submitted.
