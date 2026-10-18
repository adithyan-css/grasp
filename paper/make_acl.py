#!/usr/bin/env python3
"""Build paper/acl/grasp_acl.tex (ACL review format) from paper/grasp_paper.tex.

Re-run after editing grasp_paper.tex:  python paper/make_acl.py
"""

import re
from pathlib import Path

here = Path(__file__).resolve().parent
src = (here / "grasp_paper.tex").read_text(encoding="utf-8")

preamble = r"""\documentclass[11pt]{article}
\usepackage[review]{acl}
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}
\usepackage{booktabs}
\usepackage{amsmath}

\title{Where Was the Model Looking? Cheap Claim-Level Hallucination\\ Detection from Attention and Token Probabilities}
\author{Anonymous ACL submission}
"""
body = src[src.index(r"\begin{document}"):]

# wide tables span both columns
body = body.replace(r"\begin{table}[h]", r"\begin{table*}[t]").replace(r"\end{table}", r"\end{table*}")

# Limitations: unnumbered, after the conclusion (ACL requirement)
m = re.search(r"\\section\{Limitations\}\s*\\label\{sec:limits\}(.*?)(?=\\section\{Conclusion\})", body, re.S)
assert m, "limitations section not found"
lim_text = m.group(1).strip()
body = body[: m.start()] + body[m.end():]
body = body.replace(r"(\S\ref{sec:limits})", "(Limitations)")
body = body.replace(r"\bibliographystyle{plainnat}",
                    "\\section*{Limitations}\n" + lim_text + "\n\n\\bibliographystyle{acl_natbib}")
body = body.replace(r"\bibliography{../papers/references}", r"\bibliography{../../papers/references}")
assert r"\pending" not in body

out = here / "acl" / "grasp_acl.tex"
out.write_text(preamble + "\n" + body, encoding="utf-8")
print(f"wrote {out}")
