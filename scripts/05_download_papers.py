#!/usr/bin/env python3
"""
Download every paper in the reading list, and fetch OFFICIAL BibTeX for each.

The BibTeX comes from arXiv's own API and from ACL Anthology's `.bib` endpoints,
never from anything hand-written here — hand-typed entries are how wrong years
and missing DOIs get into a manuscript, and the brief requires every reference to
be verified.

Usage
-----
    python scripts/05_download_papers.py --out papers/
    python scripts/05_download_papers.py --out papers/ --bibtex-only
    python scripts/05_download_papers.py --out papers/ --tier 1

Output
------
    papers/NN_short-name.pdf
    papers/references.bib
    papers/download_report.md      what succeeded, what failed, what to check
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (compatible; grasp-reading-list/1.0)"

# (n, tier, short_name, arxiv_id or None, acl_id or None, note)
PAPERS: list[tuple[int, int, str, str | None, str | None, str]] = [
    (1, 1, "much-benchmark", "2511.17081", None, "the benchmark and the target number"),
    (2, 1, "ccp", "2403.04696", "2024.findings-acl.558", "the method to beat"),
    (3, 1, "rauq", "2505.20045", None, "unsupervised attention head selection"),
    (4, 1, "uheads", "2505.08200", "2025.emnlp-main.1809", "supervised claim-level attention"),
    (5, 1, "sinkprobe", "2604.10697", None, "attention sinks predict hallucination"),

    (6, 2, "semantic-uncertainty", "2302.09664", None, "Kuhn et al., ICLR 2023"),
    (7, 2, "semantic-entropy-nature", None, None, "Nature 630 2024 - see PMC11186750"),
    (8, 2, "selfcheckgpt", None, "2023.emnlp-main.557", "zero-resource black-box detection"),
    (9, 2, "know-what-they-know", "2207.05221", None, "Kadavath et al."),
    (10, 2, "sar", "2307.01379", None, "relevance-weighted tokens, from probabilities"),
    (11, 2, "kernel-language-entropy", "2405.20003", None, "NeurIPS 2024"),
    (12, 2, "semantic-density", "2405.13845", None, "NeurIPS 2024"),
    (13, 2, "cocoa", "2502.04964", None, "confidence + consistency, NeurIPS 2025"),

    (16, 3, "saplma", "2304.13734", None, "hidden-state probe"),
    (17, 3, "inside-eigenscore", "2402.03744", None, "ICLR 2024"),
    (18, 3, "lookback-lens", "2407.07071", None, "attention ratios, EMNLP 2024"),
    (19, 3, "semantic-entropy-probes", "2406.15927", None, "cheap SE approximation"),
    (20, 3, "retrieval-head", "2404.15574", None, "mechanistic basis for grounding"),
    (21, 3, "redeep", "2410.11414", None, "ICLR 2025"),
    (22, 3, "multi-view-attention", "2504.04335", None, "supervised token-level"),
    (23, 3, "realtime-entity-detection", "2509.03531", None, "LoRA probes, long-form"),
    (24, 3, "attention-divergence", "2605.05025", None, "asks for probe-free methods"),

    (25, 4, "factscore", None, "2023.emnlp-main.741", "atomic claim evaluation"),
    (26, 4, "luq", None, "2024.emnlp-main.299", "long-text UQ"),
    (27, 4, "ragtruth", "2401.00396", "2024.acl-long.585", "span-level corpus, MIT licence"),
    (28, 4, "finegrained-uq-study", "2602.17431", None, "comparative study"),
    (29, 4, "factselfcheck", None, "2026.findings-eacl.296", "claim-level, 20 samples"),
    (30, 4, "factuality-probes", None, "2025.findings-emnlp.880", "claim-level, supervised"),

    (32, 5, "abstention-survey", "2407.18418", None, "frames the review, Topic 16"),
    (33, 5, "lm-polygraph-benchmark", "2406.15627", "2025.tacl-1.11", "40+ baselines"),
    (34, 5, "conformal-abstention", "2405.01563", None, "NeurIPS 2024"),
    (35, 5, "uq-hallucination-survey", "2510.12040", None, "quantifies the long-form gap"),
    (36, 5, "mirage-contrastive-decoding", "2504.10020", None, "cautionary tale on protocols"),
]

UNVERIFIED = {
    14: ("Generating with Confidence (Lin, Trivedi, Sun, TMLR 2024)",
         "https://arxiv.org/a/search?searchtype=all&query=Generating+with+Confidence"),
    15: ("Just Ask for Calibration (Tian et al., EMNLP 2023)",
         "https://aclanthology.org/search/?q=Just+Ask+for+Calibration"),
    31: ("HaluEval (Li et al., EMNLP 2023)",
         "https://aclanthology.org/search/?q=HaluEval"),
}


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def arxiv_bibtex(arxiv_id: str) -> str | None:
    """arXiv exposes BibTeX at /bibtex/<id>."""
    try:
        return fetch(f"https://arxiv.org/bibtex/{arxiv_id}").decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return None


def acl_bibtex(acl_id: str) -> str | None:
    try:
        return fetch(f"https://aclanthology.org/{acl_id}.bib").decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("papers"))
    ap.add_argument("--tier", type=int, default=0, help="0 = every tier")
    ap.add_argument("--bibtex-only", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="pause between requests; be polite to arXiv and ACL")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in PAPERS if args.tier in (0, p[1])]
    print(f"{len(todo)} papers to fetch into {args.out}/\n")

    bib_entries: list[str] = []
    ok_pdf, no_pdf, ok_bib, no_bib = [], [], [], []

    for n, tier, name, arx, acl, note in todo:
        stem = f"{n:02d}_{name}"
        print(f"[{n:02d}] {name}  (tier {tier})  — {note}")

        # ---- PDF ----
        if not args.bibtex_only:
            dest = args.out / f"{stem}.pdf"
            if dest.exists() and dest.stat().st_size > 10_000:
                print("     pdf: already have it")
                ok_pdf.append(n)
            else:
                urls = []
                if arx:
                    urls.append(f"https://arxiv.org/pdf/{arx}")
                if acl:
                    urls.append(f"https://aclanthology.org/{acl}.pdf")
                got = False
                for u in urls:
                    try:
                        data = fetch(u)
                        if data[:4] == b"%PDF":
                            dest.write_bytes(data)
                            print(f"     pdf: {len(data)/1e6:.1f} MB  <- {u}")
                            ok_pdf.append(n)
                            got = True
                            break
                        print(f"     pdf: not a PDF from {u}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"     pdf: {type(exc).__name__} from {u}")
                    time.sleep(args.sleep)
                if not got:
                    no_pdf.append((n, name, note))

        # ---- BibTeX: publisher first, arXiv as fallback ----
        entry = acl_bibtex(acl) if acl else None
        source = "ACL Anthology" if entry else None
        if entry is None and arx:
            entry = arxiv_bibtex(arx)
            source = "arXiv" if entry else None
        if entry:
            bib_entries.append(f"% [{n}] {note}  (via {source})\n{entry}")
            print(f"     bib: {source}")
            ok_bib.append(n)
        else:
            no_bib.append((n, name))
            print("     bib: FAILED — fetch it by hand")
        time.sleep(args.sleep)

    # ---- write outputs ----
    bib_path = args.out / "references.bib"
    header = (
        "% GRASP reading list — BibTeX fetched from ACL Anthology and arXiv.\n"
        "% Nothing here was hand-typed, but VERIFY venue and year anyway: several\n"
        "% of these appeared as preprints well before their conference version, and\n"
        "% arXiv BibTeX always reports the preprint, not the published record.\n\n"
    )
    bib_path.write_text(header + "\n\n".join(bib_entries) + "\n", encoding="utf-8")

    report = [
        "# Download report", "",
        f"- PDFs retrieved: **{len(ok_pdf)}**",
        f"- BibTeX entries: **{len(ok_bib)}** -> `references.bib`", "",
    ]
    if no_pdf:
        report += ["## PDFs that failed", ""]
        report += [f"- [{n}] {name} — {note}" for n, name, note in no_pdf]
        report += ["", "Paper 7 (Nature) is paywalled by design; use the free PMC copy:",
                   "<https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750>", ""]
    if no_bib:
        report += ["## BibTeX that failed", ""]
        report += [f"- [{n}] {name}" for n, name in no_bib]
        report += [""]
    report += ["## Identifiers I could not verify", "",
               "Search these yourself rather than trusting a guessed ID:", ""]
    report += [f"- [{n}] {title} — <{url}>" for n, (title, url) in UNVERIFIED.items()]
    report += ["", "## Before submitting", "",
               "Re-check every venue, year and DOI against the publisher's page.",
               "arXiv BibTeX reports the preprint even when a published version exists."]
    (args.out / "download_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"\nPDFs: {len(ok_pdf)} ok, {len(no_pdf)} failed")
    print(f"BibTeX: {len(ok_bib)} ok, {len(no_bib)} failed")
    print(f"wrote {bib_path} and {args.out / 'download_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
