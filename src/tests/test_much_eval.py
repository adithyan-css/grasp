"""Cross-check our evaluator transcription against MUCH's own code.

The point of these tests is not that our maths is self-consistent — it is that it
produces byte-for-byte the same numbers as `much/src/evaluation/`. If MUCH is not
importable the cross-checks skip and only the self-contained rules are asserted.

    python src/tests/test_much_eval.py --much-root /content/much
    python src/tests/test_much_eval.py                 # skips cross-checks
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp.much_eval import (  # noqa: E402
    AGGREGATORS,
    HALLUCINATION_LABEL,
    aggregate,
    evaluate,
    extended_stopwords_set,
    has_consensus,
    mask_stopwords,
    metrics,
)

MUCH_ROOT = None


# ---------------------------------------------------------------------------
# cross-checks against MUCH's own implementation
# ---------------------------------------------------------------------------


def _much_signal_evaluation():
    if MUCH_ROOT is None:
        return None
    sys.path.insert(0, str(MUCH_ROOT))
    try:
        from src.evaluation.signal_evaluation import SignalEvaluation

        return SignalEvaluation
    except Exception as exc:  # noqa: BLE001
        print(f"  (cross-check skipped: {type(exc).__name__}: {exc})")
        return None


def test_aggregators_match_much():
    SE = _much_signal_evaluation()
    if SE is None:
        print("skip  aggregator cross-check (MUCH not importable)")
        return

    rng = np.random.RandomState(0)
    for trial in range(200):
        n = rng.randint(1, 12)
        chunk = rng.rand(n)
        ev = SE(signal_name="x")
        ev.update_values_in(chunk)
        theirs = {
            "mean": ev.mean_in[0],
            "max": ev.max_in[0],
            "geometric": ev.geometric_in[0],
            "product": ev.product_in[0],
        }
        for agg in AGGREGATORS:
            ours = aggregate(chunk, agg)
            assert np.isclose(ours, theirs[agg], atol=1e-12), (
                f"{agg} mismatch on trial {trial}: ours={ours} theirs={theirs[agg]}"
            )
    print("ok    aggregators match MUCH exactly (200 random chunks x 4 aggregators)")


def test_metrics_match_much():
    SE = _much_signal_evaluation()
    if SE is None:
        print("skip  metric cross-check (MUCH not importable)")
        return

    rng = np.random.RandomState(1)
    ev = SE(signal_name="x")
    ours_y, ours_s = [], []
    for _ in range(120):
        chunk = rng.rand(rng.randint(1, 9))
        if rng.rand() < 0.4:
            chunk = np.clip(chunk + 0.3, 0, 1)
            ev.update_values_in(chunk)
            ours_y.append(1)
        else:
            ev.update_values_out(chunk)
            ours_y.append(0)
        ours_s.append(aggregate(chunk, "mean"))

    _, _, their_roc, _, _, their_pr, their_ap = ev.get_rocauc_prauc("mean")
    ours = metrics(ours_y, ours_s)
    assert np.isclose(ours["roc_auc"], their_roc, atol=1e-12), (ours["roc_auc"], their_roc)
    assert np.isclose(ours["pr_auc"], their_pr, atol=1e-12), (ours["pr_auc"], their_pr)
    assert np.isclose(ours["average_precision"], their_ap, atol=1e-12)
    print(f"ok    metrics match MUCH exactly (ROC={their_roc:.6f}, PR={their_pr:.6f})")


# ---------------------------------------------------------------------------
# the four protocol rules
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, model_name="m"):
        self.model_name = model_name


class _Gen:
    def __init__(self, gid, tokens, lang="en"):
        self.id = gid
        self.lang = lang
        self.output_tokens = list(tokens)
        self.generation_cfg = _Cfg()


class _Annot:
    def __init__(self, gid, chunks, labels, tokens, lang="en"):
        self.generation = _Gen(gid, tokens, lang)
        self.token_chunks = chunks
        self.labels = labels


class _Tok:
    """Decodes even token ids to a stopword, odd ids to a content word."""

    def decode(self, tok):
        return "the" if tok % 2 == 0 else f"w{tok}"


def test_label_polarity_minus_one_is_positive():
    assert HALLUCINATION_LABEL == -1
    # one hallucinated claim (-1) with a high signal, one clean claim (+1) low
    anns = []
    for i in range(40):
        halluc = i % 2 == 0
        lab = -1 if halluc else 1
        anns.append(
            _Annot(f"g{i}", [[0, 1], [2, 3]], {"gpt-4o": [lab, 1], "gpt-4.1": [lab, 1]},
                   tokens=[1, 3, 5, 7])
        )
    sig = {f"g{i}": np.array([0.9, 0.9, 0.1, 0.1]) if i % 2 == 0
           else np.array([0.1, 0.1, 0.1, 0.1]) for i in range(40)}
    r = evaluate(anns, lambda g: sig.get(g), tokenizer_for=None)
    roc = r["per_aggregator"]["mean"]["roc_auc"]
    assert roc > 0.9, f"polarity is inverted: ROC={roc}"
    print(f"ok    label polarity: -1 is the positive class (ROC={roc:.3f})")


def test_last_chunk_is_dropped():
    ann = _Annot("g", [[0, 1], [2, 3], [4, 5]],
                 {"gpt-4o": [-1, 1, -1], "gpt-4.1": [-1, 1, -1]},
                 tokens=[1, 3, 5, 7, 9, 11])
    r = evaluate([ann], lambda g: np.array([0.5] * 6), tokenizer_for=None)
    assert r["counts"]["chunks"] == 2, r["counts"]
    print("ok    trailing EOS chunk dropped (3 chunks -> 2 scored)")


def test_stopword_masking_zeroes_the_right_tokens():
    values = np.array([0.9, 0.9, 0.9, 0.9])
    tokens = [2, 3, 4, 5]  # even -> "the" (stopword), odd -> content
    out = mask_stopwords(values, tokens, _Tok())
    assert list(out) == [0.0, 0.9, 0.0, 0.9], out
    assert "the" in extended_stopwords_set()
    assert "." in extended_stopwords_set()
    assert "w3" not in extended_stopwords_set()
    print("ok    stopword masking zeroes stopwords and punctuation only")


def test_stopword_masking_changes_the_aggregated_score():
    """Masking must move the claim score. Note it cannot be detected through
    AUROC alone when every sample is transformed identically — ranking is
    preserved — so assert on the score itself."""
    values = np.array([0.95, 0.2, 0.95, 0.2])
    tokens = [2, 3, 4, 5]  # even ids decode to a stopword
    plain = aggregate(values, "mean")
    masked = aggregate(mask_stopwords(values, tokens, _Tok()), "mean")
    assert abs(plain - 0.575) < 1e-12, plain
    assert abs(masked - 0.10) < 1e-12, masked
    print(f"ok    masking moves the claim score ({plain:.3f} -> {masked:.3f})")


def test_masking_can_flip_the_ranking():
    """A case where masking actually changes AUROC: the hallucinated claim's
    signal sits on content words, the clean claim's on stopwords."""
    anns, sig = [], {}
    for i in range(40):
        halluc = i % 2 == 0
        lab = -1 if halluc else 1
        anns.append(
            _Annot(f"g{i}", [[0, 1, 2, 3], [4, 5]],
                   {"gpt-4o": [lab, 1], "gpt-4.1": [lab, 1]},
                   tokens=[3, 5, 7, 9, 11, 13] if halluc else [2, 4, 6, 8, 10, 12])
        )
        # identical raw signal for both classes -> unmasked is pure chance
        sig[f"g{i}"] = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])

    plain = evaluate(anns, lambda g: sig.get(g), tokenizer_for=None)
    masked = evaluate(anns, lambda g: sig.get(g), tokenizer_for=lambda m: _Tok())
    p = plain["per_aggregator"]["mean"]["roc_auc"]
    m = masked["per_aggregator"]["mean"]["roc_auc"]
    assert abs(p - 0.5) < 1e-9, f"unmasked should be chance, got {p}"
    assert m > 0.99, f"masking should make it separable, got {m}"
    print(f"ok    masking changes AUROC when it changes ranking ({p:.3f} -> {m:.3f})")


def test_consensus_filter():
    assert has_consensus({"gpt-4o": [1, -1], "gpt-4.1": [1, -1]})
    assert not has_consensus({"gpt-4o": [1, -1], "gpt-4.1": [1, 1]})
    assert not has_consensus({"gpt-4o": [1, -1]})
    ann_bad = _Annot("g", [[0, 1], [2, 3]],
                     {"gpt-4o": [-1, 1], "gpt-4.1": [1, 1]}, tokens=[1, 3, 5, 7])
    r = evaluate([ann_bad], lambda g: np.array([0.5] * 4), tokenizer_for=None)
    assert r["counts"]["annotations"] == 0, r["counts"]
    print("ok    annotator-disagreement filter")


def test_missing_signal_is_counted_not_crashed():
    ann = _Annot("g", [[0, 1], [2, 3]], {"gpt-4o": [-1, 1], "gpt-4.1": [-1, 1]},
                 tokens=[1, 3, 5, 7])
    r = evaluate([ann], lambda g: None, tokenizer_for=None)
    assert r["counts"]["missing_signal"] == 1 and r["counts"]["chunks"] == 0
    print("ok    missing signal rows counted, not fatal")


def test_geometric_includes_the_epsilon():
    x = np.array([0.0, 0.0])
    # without the +1e-5 this would be exactly 0.0
    assert aggregate(x, "geometric") != 0.0
    assert abs(aggregate(x, "geometric") - (1 - (1.0 + 1e-5) ** 0.5)) < 1e-15
    print("ok    geometric aggregator keeps MUCH's +1e-5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, default=None)
    args = ap.parse_args()
    MUCH_ROOT = args.much_root

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} checks completed.")
