#!/usr/bin/env python3
"""
GRASP stage 0 — the go/no-go checkpoint.

Reproduce a published MUCH baseline (CCP by default) from the shipped signal
file, under MUCH's exact evaluation protocol. If this does not land near the
published number, nothing downstream is comparable and the pipeline is wrong.

The protocol has four rules that are easy to get wrong; all four live in
`grasp.much_eval`, which is cross-checked against MUCH's own code by
`src/tests/test_much_eval.py`:

  1. labels are {-1, +1} and -1 is the hallucination class
  2. the trailing EOS chunk is dropped
  3. stopwords and punctuation are zeroed before aggregating (needs the tokenizer)
  4. only annotations where gpt-4o and gpt-4.1 agree are scored

Published reference (arXiv 2511.17081): CCP at ROC-AUC 0.772 / PR-AUC 0.639.
Verify those figures against the PDF before quoting them.

Usage
-----
    python scripts/04_reproduce_ccp.py --much-root /content/much
    python scripts/04_reproduce_ccp.py --much-root /content/much --signal ccp_24_8
    python scripts/04_reproduce_ccp.py --much-root /content/much --all-signals
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp.much_eval import AGGREGATORS, evaluate  # noqa: E402

PUBLISHED = {"roc_auc": 0.772, "pr_auc": 0.639}
LANGS = ("en", "fr", "es", "de")


def load_much(much_root: Path):
    sys.path.insert(0, str(much_root))
    from src.annotation import get_all_annotations  # type: ignore
    from src.generation import get_tokenizer  # type: ignore

    return get_all_annotations, get_tokenizer


def load_signal(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["generation_id"]] = np.asarray(d["values"], dtype=float)
    return out


def pick_signal(sig_dir: Path, wanted: str | None) -> Path:
    available = sorted(sig_dir.glob("*.jsonl"))
    if not available:
        raise SystemExit(
            f"No signal files in {sig_dir}. Run `python -m src import-signals` first."
        )
    if wanted:
        exact = sig_dir / f"{wanted}.jsonl"
        if exact.is_file():
            return exact
        raise SystemExit(
            f"{wanted}.jsonl not found. Available: {[p.stem for p in available]}"
        )
    # MUCH's own CCP docstring specifies delta=24 and sigma=8 ("we provide the
    # model with the sigma=8 preceding and succeeding tokens ... we evaluated the
    # delta=24 possible alternative for each token"). That is the variant behind
    # the published headline, so prefer it over the alphabetically-first one.
    preferred = sig_dir / "ccp_24_8.jsonl"
    if preferred.is_file():
        return preferred
    ccp = [p for p in available if p.stem.startswith("ccp")]
    if ccp:
        return sorted(ccp)[0]
    return available[0]


def fmt(m: dict) -> str:
    if "roc_auc" not in m:
        return f"n={m.get('n', 0):6}  (single class — nothing to score)"
    return (
        f"n={m['n']:6} pos={m['positives']:5}  "
        f"ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  "
        f"AP={m['average_precision']:.4f}  TPR@10%FPR={m['tpr_at_10fpr']:.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--much-root", type=Path, required=True)
    ap.add_argument("--signal", default=None, help="signal stem, e.g. ccp_24_8")
    ap.add_argument("--all-signals", action="store_true")
    ap.add_argument("--per-lang", action="store_true", default=True)
    ap.add_argument("--no-stopword-mask", action="store_true",
                    help="skip rule 3. Faster and needs no tokenizer, but the "
                         "numbers will NOT match the published table.")
    ap.add_argument("--skip-unavailable", action="store_true",
                    help="evaluate only the models whose tokenizers load, instead "
                         "of refusing to run. Useful while gated access is pending.")
    ap.add_argument("--per-model", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=Path("results/ccp_reproduction.json"))
    args = ap.parse_args()

    get_all_annotations, get_tokenizer = load_much(args.much_root)
    annotations = list(get_all_annotations().values())
    print(f"{len(annotations)} annotations loaded")

    sig_dir = args.much_root / "output" / "signals"
    targets = (
        sorted(sig_dir.glob("*.jsonl"))
        if args.all_signals
        else [pick_signal(sig_dir, args.signal)]
    )
    print(f"available signals: {[p.stem for p in sorted(sig_dir.glob('*.jsonl'))]}")

    # Rule 3 needs the tokenizer of whichever model generated each sample, and
    # Llama/Gemma tokenizers are gated. Fail loudly rather than silently
    # producing numbers that cannot be compared.
    tokenizer_for = None
    if not args.no_stopword_mask:
        cache: dict[str, object] = {}

        def tokenizer_for(model_name: str):  # noqa: F811
            if model_name not in cache:
                cache[model_name] = get_tokenizer(model_name)
            return cache[model_name]

        needed = sorted({a.generation.generation_cfg.model_name for a in annotations})
        available, blocked = [], []
        print("\ntokenizer availability:")
        for m in needed:
            try:
                tokenizer_for(m)
                available.append(m)
                print(f"  OK      {m}")
            except Exception as exc:  # noqa: BLE001
                blocked.append((m, type(exc).__name__))
                print(f"  BLOCKED {m}  ({type(exc).__name__})")
                print(f"          https://huggingface.co/{m}")

        if blocked and not args.skip_unavailable:
            print(
                "\nRule 3 needs every model's tokenizer, so the full published number "
                "is\nout of reach until access lands. Your options:\n"
                "  --skip-unavailable   evaluate the models you DO have (recommended)\n"
                "  --no-stopword-mask   plumbing check only, not a reproduction"
            )
            return 2

        if blocked and args.skip_unavailable:
            keep = set(available)
            before = len(annotations)
            annotations = [
                a for a in annotations
                if a.generation.generation_cfg.model_name in keep
            ]
            print(
                f"\n--skip-unavailable: evaluating {len(annotations)}/{before} "
                f"annotations from {len(keep)} of {len(needed)} models."
            )
            print(
                "  This is a PARTIAL reproduction. The published 0.772 is over all\n"
                "  models, so expect a nearby but not identical number. Re-run once\n"
                "  the remaining access is granted."
            )

    results: dict = {}
    for path in targets:
        name = path.stem
        values = load_signal(path)
        print(f"\n=== {name}  ({len(values)} signal rows) ===")
        results[name] = {}

        overall = evaluate(annotations, lambda g: values.get(g),
                           tokenizer_for=tokenizer_for, lang=None)
        results[name]["all"] = overall
        c = overall["counts"]
        print(f"  {c['annotations']} annotations, {c['chunks']} chunks scored, "
              f"{c['missing_signal']} missing  |  stopword masking: "
              f"{overall['stopword_masking']}")
        for agg in AGGREGATORS:
            print(f"    all  {agg:<10} {fmt(overall['per_aggregator'][agg])}")

        if args.per_lang:
            for lang in LANGS:
                r = evaluate(annotations, lambda g: values.get(g),
                             tokenizer_for=tokenizer_for, lang=lang)
                results[name][lang] = r
                best = max(
                    AGGREGATORS,
                    key=lambda a: r["per_aggregator"][a].get("roc_auc", -1),
                )
                print(f"    {lang}   best={best:<10} "
                      f"{fmt(r['per_aggregator'][best])}")

        if args.per_model:
            models = sorted({a.generation.generation_cfg.model_name
                             for a in annotations})
            for m in models:
                subset = [a for a in annotations
                          if a.generation.generation_cfg.model_name == m]
                r = evaluate(subset, lambda g: values.get(g),
                             tokenizer_for=tokenizer_for, lang=None)
                results[name][f"model:{m}"] = r
                best = max(
                    AGGREGATORS,
                    key=lambda a: r["per_aggregator"][a].get("roc_auc", -1),
                )
                short = m.split("/")[-1]
                print(f"    {short:<28} best={best:<10} "
                      f"{fmt(r['per_aggregator'][best])}")

        # verdict against the published figure
        if name.startswith("ccp"):
            best_agg = max(
                AGGREGATORS,
                key=lambda a: overall["per_aggregator"][a].get("roc_auc", -1),
            )
            got = overall["per_aggregator"][best_agg]
            if "roc_auc" in got:
                d_roc = got["roc_auc"] - PUBLISHED["roc_auc"]
                d_pr = got["pr_auc"] - PUBLISHED["pr_auc"]
                print(f"\n  published: ROC={PUBLISHED['roc_auc']:.3f} "
                      f"PR={PUBLISHED['pr_auc']:.3f}")
                print(f"  ours     : ROC={got['roc_auc']:.4f} ({d_roc:+.4f})  "
                      f"PR={got['pr_auc']:.4f} ({d_pr:+.4f})  "
                      f"[aggregator={best_agg}]")
                if abs(d_roc) <= 0.02:
                    print("  VERDICT: reproduced. Proceed to attention extraction.")
                elif abs(d_roc) <= 0.05:
                    print("  VERDICT: close. Check the aggregator and the signal "
                          "variant before proceeding.")
                else:
                    print("  VERDICT: NOT reproduced. Do not spend GPU time until "
                          "this is understood.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
