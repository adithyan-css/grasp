"""Unit tests for the GRASP maths and new remediation features.

These do not need a GPU, a model, or the MUCH dataset. Run with:

    python -m pytest src/tests/test_grasp.py -q
or  python src/tests/test_grasp.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp.features import FeatureConfig, _AttentionReducer  # noqa: E402
from grasp.heads import select_heads, select_heads_variance  # noqa: E402
from grasp.score import (  # noqa: E402
    CorpusNormaliser,
    apply_boundary_reset,
    claim_scores,
    resolve_scoring_options,
    saliency_weights,
    softmin_claim_score,
    token_signal,
)


# ---------------------------------------------------------------------------
# 1. attention reduction & alignment
# ---------------------------------------------------------------------------


def _reducer(prompt_len, cfg=None):
    r = _AttentionReducer.__new__(_AttentionReducer)
    r.cfg = cfg or FeatureConfig()
    r.buffers = {}
    r._handles = []
    r._prompt_len = prompt_len
    return r


def test_grounding_and_sink_are_exact():
    """Hand-built attention with a known split between sink, prompt and self."""
    P, T, N = 3, 2, 5  # sink=pos0, prompt=pos1..2, generated=pos3..4
    attn = torch.zeros(1, 1, N, N)

    # generated token 0 (abs pos 3): 0.5 sink, 0.4 prompt, 0.1 self
    attn[0, 0, 3, 0] = 0.5
    attn[0, 0, 3, 1] = 0.2
    attn[0, 0, 3, 2] = 0.2
    attn[0, 0, 3, 3] = 0.1
    # generated token 1 (abs pos 4): 0.1 sink, 0.1 prompt, 0.8 self-ish
    attn[0, 0, 4, 0] = 0.1
    attn[0, 0, 4, 1] = 0.1
    attn[0, 0, 4, 3] = 0.4
    attn[0, 0, 4, 4] = 0.4

    out = _reducer(P)._reduce(attn, 0)  # (H=1, T=2, 4)
    G, S = out[0, :, 0], out[0, :, 1]

    # token 0: prompt 0.4 / non-sink 0.5 = 0.8 ; sink 0.5
    assert np.isclose(G[0], 0.8, atol=1e-5), G[0]
    assert np.isclose(S[0], 0.5, atol=1e-5), S[0]
    # token 1: prompt 0.1 / non-sink 0.9 = 0.1111 ; sink 0.1
    assert np.isclose(G[1], 0.1 / 0.9, atol=1e-5), G[1]
    assert np.isclose(S[1], 0.1, atol=1e-5), S[1]
    print("ok  grounding/sink exact")


def test_dispersion_is_high_when_flat_and_low_when_peaked():
    P, T, N = 2, 1, 8
    flat = torch.zeros(1, 1, N, N)
    flat[0, 0, 2, 1:3] = 0.5  # spread over the available non-sink support
    peaked = torch.zeros(1, 1, N, N)
    peaked[0, 0, 2, 1] = 1.0  # all mass on one position

    d_flat = _reducer(P)._reduce(flat, 0)[0, 0, 2]
    d_peak = _reducer(P)._reduce(peaked, 0)[0, 0, 2]
    assert d_flat > d_peak, (d_flat, d_peak)
    assert np.isclose(d_peak, 0.0, atol=1e-4)
    print(f"ok  dispersion flat={d_flat:.3f} > peaked={d_peak:.3f}")


def test_saliency_counts_only_later_tokens():
    """Token A is attended to by a later token; token B is not."""
    P, N = 2, 5  # generated = pos 2,3,4
    attn = torch.zeros(1, 1, N, N)
    attn[0, 0, 3, 2] = 0.7  # pos3 looks back at pos2
    attn[0, 0, 4, 2] = 0.3  # pos4 looks back at pos2
    attn[0, 0, 4, 3] = 0.0  # nobody looks at pos3

    W = _reducer(P)._reduce(attn, 0)[0, :, 3]
    assert np.isclose(W[0], 1.0, atol=1e-5), W  # generated token 0 == abs pos 2
    assert np.isclose(W[1], 0.0, atol=1e-5), W
    assert np.isclose(W[2], 0.0, atol=1e-5), W  # last token receives nothing
    print("ok  saliency counts only later tokens")


def test_saliency_norm_per_later_token():
    P, N = 2, 5  # T=3 (pos 2, 3, 4)
    attn = torch.zeros(1, 1, N, N)
    attn[0, 0, 3, 2] = 0.6  # pos 3 looks at pos 2
    attn[0, 0, 4, 2] = 0.6  # pos 4 looks at pos 2
    attn[0, 0, 4, 3] = 0.6  # pos 4 looks at pos 3

    # Without norm: pos 2 gets 1.2, pos 3 gets 0.6
    cfg_raw = FeatureConfig(saliency_norm="none")
    w_raw = _reducer(P, cfg_raw)._reduce(attn, 0)[0, :, 3]
    assert np.isclose(w_raw[0], 1.2, atol=1e-5)
    assert np.isclose(w_raw[1], 0.6, atol=1e-5)

    # With per_later_token norm: pos 2 divided by 2 -> 0.6; pos 3 divided by 1 -> 0.6
    cfg_norm = FeatureConfig(saliency_norm="per_later_token")
    w_norm = _reducer(P, cfg_norm)._reduce(attn, 0)[0, :, 3]
    assert np.isclose(w_norm[0], 0.6, atol=1e-5)
    assert np.isclose(w_norm[1], 0.6, atol=1e-5)
    print("ok  saliency length normalisation eliminates early position bias")


def test_alignment_modes_shift_query_rows():
    P, N = 2, 4  # T=2 (generated tokens at pos 2, 3)
    attn = torch.zeros(1, 1, N, N)
    # Query row 1 (pos 1, prompt end): 0.9 prompt (pos 1), 0.1 self (pos 1) -> G = 0.9 / 0.9 = 1.0 (or with self 0.9)
    attn[0, 0, 1, 1] = 1.0  # G = 1.0 / 1.0 = 1.0
    # Query row 2 (pos 2): 0.1 prompt (pos 1), 0.9 self (pos 2) -> G = 0.1 / 1.0 = 0.1
    attn[0, 0, 2, 1] = 0.1
    attn[0, 0, 2, 2] = 0.9

    cfg_legacy = FeatureConfig(align="legacy")
    out_legacy = _reducer(P, cfg_legacy)._reduce(attn, 0)
    # Legacy takes query row 2 for token 0 -> G = 0.1
    assert np.isclose(out_legacy[0, 0, 0], 0.1, atol=1e-4)

    cfg_current = FeatureConfig(align="current")
    out_current = _reducer(P, cfg_current)._reduce(attn, 0)
    # Current takes query row 1 for token 0 -> G = 1.0
    assert np.isclose(out_current[0, 0, 0], 1.0, atol=1e-4)
    print("ok  alignment modes correctly assign generating query rows")


def test_causal_mask_respected():
    """Mass placed above the diagonal must be ignored."""
    P, N = 2, 5
    attn = torch.zeros(1, 1, N, N)
    attn[0, 0, 2, 4] = 1.0  # illegal: pos2 attending to pos4
    out = _reducer(P)._reduce(attn, 0)
    assert np.isclose(out[0, 0, 1], 0.0, atol=1e-6)  # no sink mass
    print("ok  causal mask respected")


def test_head_filtering_keeps_only_requested_head():
    P, N, H = 2, 5, 4
    attn = torch.rand(1, H, N, N)
    cfg = FeatureConfig(keep_heads={0: 2})
    out = _reducer(P, cfg)._reduce(attn, 0)
    assert out.shape[0] == 1, out.shape
    full = _reducer(P)._reduce(attn, 0)
    assert np.allclose(out[0], full[2], atol=1e-6)
    print("ok  head filtering")


# ---------------------------------------------------------------------------
# 2. head selection
# ---------------------------------------------------------------------------


def test_variance_selection_picks_the_informative_head():
    rng = np.random.RandomState(0)
    L, H, T = 3, 5, 40
    calib = []
    for _ in range(12):
        f = np.zeros((L, H, T, 4), dtype=np.float32)
        f[:, :, :, 0] = 0.5  # constant grounding everywhere...
        f[:, 3, :, 0] = rng.rand(T)  # ...except head 3, which varies
        calib.append(f)
    sel = select_heads_variance(calib)
    assert set(sel.values()) == {3}, sel
    print("ok  variance head selection")


def test_select_heads_dispatcher():
    rng = np.random.RandomState(0)
    L, H, T = 4, 8, 20
    calib = [rng.randn(L, H, T, 4).astype(np.float32) for _ in range(5)]

    sel_var = select_heads("variance", calib_features=calib)
    assert len(sel_var) == L and all(0 <= v < H for v in sel_var.values())

    sel_rand = select_heads("random", calib_features=calib, seed=42)
    assert len(sel_rand) == L and all(0 <= v < H for v in sel_rand.values())

    sel_fix = select_heads("fixed", calib_features=calib, head=3)
    assert sel_fix == {l: 3 for l in range(L)}

    sel_all = select_heads("all_heads", calib_features=calib)
    assert sel_all is None
    print("ok  select_heads dispatcher handles all methods")


# ---------------------------------------------------------------------------
# 3. scoring, normalisation & reset
# ---------------------------------------------------------------------------


def _synthetic(n=250, L=6, T=32, seed=0, effect=1.4):
    """Half the claims are 'hallucinated': low grounding, high sink, high
    dispersion on their tokens. Recovering them is the sanity check."""
    rng = np.random.RandomState(seed)
    feats, chunks, labels = [], [], []
    for i in range(n):
        f = rng.randn(L, 1, T, 4).astype(np.float32) * 0.4
        f[:, 0, :, 0] += 1.0  # grounding baseline
        f[:, 0, :, 3] = np.abs(rng.randn(L, T)) + 0.1  # saliency positive
        halluc = i % 2 == 0
        span = list(range(8, 16))
        if halluc:
            f[:, 0, span, 0] -= effect  # less grounded
            f[:, 0, span, 1] += effect  # more sink leakage
            f[:, 0, span, 2] += effect  # more dispersion
        feats.append(f)
        chunks.append([span])
        labels.append(1 if halluc else 0)
    return feats, chunks, np.array(labels)


def test_pipeline_recovers_planted_signal():
    feats, chunks, labels = _synthetic()
    norm = CorpusNormaliser().fit(feats)

    scores = []
    for f, ch in zip(feats, chunks):
        ts = token_signal(f, norm)
        assert ts.shape == (f.shape[2],), ts.shape
        assert ts.min() >= 0.0 and ts.max() <= 1.0
        cs = claim_scores(ts, f, ch)
        scores.append(cs[0])
    scores = np.array(scores)

    auc = roc_auc_score(labels, scores)
    assert auc > 0.75, f"planted signal not recovered, AUROC={auc:.3f}"
    print(f"ok  pipeline recovers planted signal, AUROC={auc:.3f}")


def test_norm_mode_corpus_preserves_cross_generation_distinction():
    """Demonstrate that norm_mode='corpus' preserves whole-generation differences
    while 'per_gen_rank' standardizes them away."""
    L, T = 4, 30
    rng = np.random.RandomState(42)

    # Reference corpus to establish mean/std
    corpus = [rng.randn(L, 1, T, 4).astype(np.float32) for _ in range(50)]
    norm = CorpusNormaliser().fit(corpus)

    # Gen A: universally highly grounded (factual text)
    gen_a = rng.randn(L, 1, T, 4).astype(np.float32) * 0.2
    gen_a[:, 0, :, 0] += 3.0  # Very high grounding
    gen_a[:, 0, :, 1] -= 1.0  # Low sink
    gen_a[:, 0, :, 2] -= 1.0  # Low dispersion

    # Gen B: universally ungrounded (hallucinated text)
    gen_b = rng.randn(L, 1, T, 4).astype(np.float32) * 0.2
    gen_b[:, 0, :, 0] -= 2.0  # Very low grounding
    gen_b[:, 0, :, 1] += 2.0  # High sink
    gen_b[:, 0, :, 2] += 2.0  # High dispersion

    # Under legacy per_gen_rank:
    ts_a_rank = token_signal(gen_a, norm, norm_mode="per_gen_rank")
    ts_b_rank = token_signal(gen_b, norm, norm_mode="per_gen_rank")
    # Means are virtually identical (~0.5) because both are rank-normalised internally!
    assert abs(ts_a_rank.mean() - ts_b_rank.mean()) < 0.05, (ts_a_rank.mean(), ts_b_rank.mean())

    # Under corpus norm_mode:
    ts_a_corp = token_signal(gen_a, norm, norm_mode="corpus")
    ts_b_corp = token_signal(gen_b, norm, norm_mode="corpus")
    # Gen A has much lower risk than Gen B
    assert ts_a_corp.mean() < 0.15, ts_a_corp.mean()
    assert ts_b_corp.mean() > 0.85, ts_b_corp.mean()
    assert ts_b_corp.mean() - ts_a_corp.mean() > 0.70
    print("ok  norm_mode='corpus' preserves cross-generation risk separation")


def test_softmin_between_mean_and_max():
    s = np.array([0.1, 0.2, 0.9, 0.15])
    chunk = [0, 1, 2, 3]
    w = np.full(4, 0.25)
    lo = softmin_claim_score(s, chunk, w, tau=0.01)
    hi = softmin_claim_score(s, chunk, w, tau=200.0)
    assert abs(lo - s.mean()) < 0.05, (lo, s.mean())
    assert abs(hi - s.max()) < 0.02, (hi, s.max())
    mid = softmin_claim_score(s, chunk, w, tau=8.0)
    assert lo < mid < hi + 1e-6
    print(f"ok  softmin interpolates mean={lo:.3f} -> {mid:.3f} -> max={hi:.3f}")


def test_claim_agg_options():
    s = np.array([0.1, 0.2, 0.9, 0.15])
    chunk = [0, 1, 2, 3]
    w = np.array([0.1, 0.2, 0.6, 0.1])
    s_mean = softmin_claim_score(s, chunk, w, claim_agg="mean")
    s_max = softmin_claim_score(s, chunk, w, claim_agg="max")
    s_soft = softmin_claim_score(s, chunk, w, tau=8.0, claim_agg="softmin")
    s_len = softmin_claim_score(s, chunk, w, tau=8.0, claim_agg="softmin_len_norm")

    assert np.isclose(s_mean, np.sum(w * s), atol=1e-5)
    assert np.isclose(s_max, 0.9, atol=1e-5)
    assert s_mean < s_soft <= s_max
    assert s_len < s_soft  # Length normalized reduces length inflation
    print("ok  claim_agg options (mean, max, softmin, softmin_len_norm)")


def test_saliency_weights_sum_to_one_and_favour_salient_tokens():
    L, T = 4, 10
    f = np.zeros((L, 1, T, 4), dtype=np.float32)
    f[:, 0, :, 3] = 0.01
    f[:, 0, 5, 3] = 5.0  # token 5 is heavily attended to later
    w = saliency_weights(f, list(range(T)))
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert w.argmax() == 5, w
    print("ok  saliency weights")


def test_boundary_reset_modes():
    s = np.array([0.9, 0.9, 0.9, 0.0, 0.0, 0.0])
    chunks = [[0, 1, 2], [3, 4, 5]]

    # reset mode: resets carry at start of chunk [3,4,5]
    out_reset = apply_boundary_reset(s, chunks, alpha=0.5, reset_mode="reset")
    assert np.isclose(out_reset[3], 0.0, atol=1e-9)

    # none mode: leaves s untouched
    out_none = apply_boundary_reset(s, chunks, alpha=0.5, reset_mode="none")
    assert np.allclose(out_none, s)

    # cross_claim_ema mode: carry from index 2 leaks into index 3
    out_cross = apply_boundary_reset(s, chunks, alpha=0.5, reset_mode="cross_claim_ema")
    assert out_cross[3] > 0.3  # Leakage occurred
    print("ok  boundary reset modes ('reset', 'none', 'cross_claim_ema')")


def test_align_shift_at_scoring_time():
    """A legacy cache + align_shift=1 must reproduce an align='current' cache.

    Legacy index i holds query row P+i (predicts token i+1); the generating row
    for token i is P+i-1 = legacy index i-1. So the shift moves features forward.
    """
    P, N = 3, 9  # T = 6
    torch.manual_seed(0)
    attn = torch.tril(torch.rand(1, 1, N, N))
    attn = attn / attn.sum(-1, keepdim=True)

    legacy = _reducer(P, FeatureConfig(align="legacy"))._reduce(attn, 0)
    current = _reducer(P, FeatureConfig(align="current"))._reduce(attn, 0)
    f_legacy = legacy[None]  # (L=1, H=1, T, 4)
    f_current = current[None]
    norm = CorpusNormaliser().fit([f_legacy])

    ts_fixed = token_signal(f_legacy, norm, norm_mode="corpus", align_shift=1)
    ts_true = token_signal(f_current, norm, norm_mode="corpus", align_shift=0)
    ts_raw = token_signal(f_legacy, norm, norm_mode="corpus", align_shift=0)

    # every token except the first (whose generating row is not cached) matches
    assert np.allclose(ts_fixed[1:], ts_true[1:], atol=1e-5), (ts_fixed, ts_true)
    # and the shift moved features forward, not backward
    assert np.allclose(ts_fixed[1:], ts_raw[:-1], atol=1e-6)
    assert ts_fixed.shape == ts_raw.shape
    print("ok  align_shift on a legacy cache reproduces align='current'")


def test_ablation_toggles_change_the_signal():
    feats, chunks, _ = _synthetic(n=20)
    norm = CorpusNormaliser().fit(feats)
    f = feats[0]
    full = token_signal(f, norm, use_signals=(True, True, True))
    no_sink = token_signal(f, norm, use_signals=(True, False, True))
    assert not np.allclose(full, no_sink), "ablation toggle had no effect"
    print("ok  ablation toggles wired up")


def test_saliency_is_normalised_exactly_once():
    """A per_later_token cache must not be divided by (T-1-i) again at scoring."""
    P, N = 2, 6  # T = 4
    torch.manual_seed(1)
    attn = torch.tril(torch.rand(1, 1, N, N))
    attn = attn / attn.sum(-1, keepdim=True)
    raw = _reducer(P, FeatureConfig(saliency_norm="none"))._reduce(attn, 0)[None]
    pre = _reducer(P, FeatureConfig(saliency_norm="per_later_token"))._reduce(attn, 0)[None]
    chunk = [0, 1, 2]

    # reference: raw cache normalised once, at scoring
    once = saliency_weights(raw, chunk, saliency_norm="per_later_token")

    # pre-normalised cache with the 'corrected' scoring request
    sal, _, notes = resolve_scoring_options(
        {"align": "current", "saliency_norm": "per_later_token"}, "per_later_token", 0
    )
    assert sal == "none" and notes
    assert np.allclose(saliency_weights(pre, chunk, saliency_norm=sal), once, atol=1e-6)

    # the bug this guards against: dividing twice gives different weights
    twice = saliency_weights(pre, chunk, saliency_norm="per_later_token")
    assert not np.allclose(twice, once, atol=1e-3)

    # legacy cache (no metadata) keeps the scoring-side normalisation
    assert resolve_scoring_options({}, "per_later_token", 1)[:2] == ("per_later_token", 1)
    # align_shift on an already-aligned cache is refused
    try:
        resolve_scoring_options({"align": "current"}, "none", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("align_shift on an align='current' cache was accepted")
    print("ok  saliency normalised exactly once; align_shift refused on aligned cache")


def test_split_uses_human_annotator_keys():
    """MUCH marks its 200-sample test set by human label keys, not a field."""
    import importlib.util
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[2] / "scripts" / "03_score_and_eval.py"
    spec = importlib.util.spec_from_file_location("score_and_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    auto = SimpleNamespace(labels={"gpt-4o": [1, 0], "gpt-4.1": [1, 0]})
    human = SimpleNamespace(labels={"gpt-4o": [1, 0], "gpt-4.1": [1, 0], "an0": [1, 0]})
    unlabelled = SimpleNamespace(labels={})
    assert mod.get_split(auto) == "train"
    assert mod.get_split(human) == "test"
    assert mod.get_split(unlabelled) == "train"
    print("ok  split: human-annotated -> test, auto-only -> train")


def test_topk_heads_ordered_and_reduced():
    """variance_topk keeps k heads per layer, rank 0 = the 1-head choice, and the
    reducer returns them in that order."""
    from grasp.heads import select_heads_variance_topk

    rng = np.random.RandomState(0)
    L, H, T = 2, 5, 20
    calib = []
    for _ in range(10):
        f = np.zeros((L, H, T, 4), dtype=np.float32)
        for h, scale in enumerate([0.1, 3.0, 0.5, 2.0, 1.0]):  # variance order: 1,3,4,2,0
            f[:, h, :, 0] = rng.randn(L, T) * scale
        calib.append(f)
    top = select_heads_variance_topk(calib, k=3)
    assert top == {0: [1, 3, 4], 1: [1, 3, 4]}, top
    assert all(top[l][0] == select_heads_variance(calib)[l] for l in range(L))

    P, N = 2, 6
    attn = torch.rand(1, H, N, N)
    attn = torch.tril(attn) / torch.tril(attn).sum(-1, keepdim=True)
    full = _reducer(P)._reduce(attn, 0)
    kept = _reducer(P, FeatureConfig(keep_heads={0: [3, 1]}))._reduce(attn, 0)
    assert kept.shape[0] == 2 and np.allclose(kept[0], full[3]) and np.allclose(kept[1], full[1])
    print("ok  variance_topk keeps k heads per layer in variance order")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} tests passed successfully.")
