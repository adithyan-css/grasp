"""Build Review_Manuscript_Hallucination_Detection.docx from the review workbook.

Every count in the text and every table cell is read from
Review_Workbook_GRASP.xlsx (Extraction_Matrix, Quality_Check, Taxonomy_Draft,
Gap_Table, Existing_Reviews, PRISMA_Counts). Run make_figures.py first so the
figures match the same workbook.

    python review/make_figures.py
    python review/build_manuscript.py
"""
import re
from collections import Counter
from pathlib import Path

import openpyxl
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).parent
WB = HERE / "Review_Workbook_GRASP.xlsx"
FIG = HERE / "figures"
OUT = HERE / "Review_Manuscript_Hallucination_Detection.docx"
FONT = "Times New Roman"


# ---------------------------------------------------------------- data

def load():
    wb = openpyxl.load_workbook(WB, data_only=True)
    em = [r for r in wb["Extraction_Matrix"].iter_rows(min_row=6, values_only=True) if r[0]]
    qc = [r for r in wb["Quality_Check"].iter_rows(min_row=6, values_only=True) if r[0]]
    tax = [r for r in wb["Taxonomy_Draft"].iter_rows(min_row=3, values_only=True)
           if r[0] and str(r[0]).startswith("S")]
    gaps = [r for r in wb["Gap_Table"].iter_rows(min_row=4, values_only=True)
            if r[0] and re.fullmatch(r"G\d+", str(r[0]))]
    rev = [r for r in wb["Existing_Reviews"].iter_rows(min_row=2, values_only=True) if r[0]]
    prisma = {r[0]: r[1] for r in wb["PRISMA_Counts"].iter_rows(min_row=4, values_only=True)
              if r[0] and isinstance(r[1], int)}
    return em, qc, tax, gaps, rev, prisma


em, qc, tax, gaps, rev, prisma = load()
N = len(em)
det = [t for t in tax if t[3] in ("Supervised", "Training-free")]
ND = len(det)
sig = Counter(t[2] for t in det)
sup = Counter(t[3] for t in det)
gran = Counter(t[4] for t in det)
cost = Counter(t[5] for t in det)
gap = {g[0]: g for g in gaps}


def ids(text):
    return re.findall(r"S\d\d", text.split(";")[0])


def pct(n, d=N):
    return f"{n} of {d} ({100 * n / d:.1f}%)"


def refs(sids):
    nums = sorted(int(s[1:]) for s in sids)
    return "[" + ", ".join(str(n) for n in nums) + "]"


code = sum(r[16] == "Yes" for r in em)
data = sum(r[17] == "Yes" for r in em)
var = sum(r[18] == "Yes" for r in em)
shift = [r[0] for r in em if r[15] == "Yes"]
transfer = ids(gap["G4"][2])
costed = ids(gap["G3"][2])
multiling = [t[0] for t in tax if t[6] == "Multilingual"]
comp = Counter(r[21] for r in em)
peer = sum(r[5] == "Peer-reviewed" for r in em)
years = Counter(r[2] for r in em)
claim = [t[0] for t in det if t[4] == "Claim"]
tf_single_claim = [t[0] for t in det if t[4] == "Claim" and t[3] == "Training-free"
                   and t[5].startswith("Single pass")]
attn = [t[0] for t in det if t[2] == "Attention"]
attn_sup = [s for s in attn if dict((t[0], t[3]) for t in det)[s] == "Supervised"]
whitebox = [t[0] for t in det if t[2] in ("Attention", "Hidden states", "Output probabilities")]
judge = ids(gap["G9"][2].replace("S01 train", "S01"))
judge_n = len(re.findall(r"S\d\d", gap["G9"][2]))
Q = {}
for i, name in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"], start=1):
    Q[name] = Counter(r[i] for r in qc)

assert N == prisma["Studies included in review"] == 54
assert sum(sig.values()) == ND
assert len(transfer) + 1 == len(shift), (transfer, shift)


# ---------------------------------------------------------------- docx helpers

doc = Document()
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for name, size in (("Heading 1", 14), ("Heading 2", 12)):
    h = doc.styles[name]
    h.font.name = FONT
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0)
    rf = h.element.get_or_add_rPr().get_or_add_rFonts()
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    rf.set(qn("w:ascii"), FONT)
    rf.set(qn("w:hAnsi"), FONT)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(2.54))


def page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


page_number(sec)


def para(text, bold_lead=None, align=None, size=None, italic=False, after=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        if size:
            r.font.size = Pt(size)
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align == "j":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def body(text):
    return para(text, align="j")


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def bullet(text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def caption(label, text):
    p = doc.add_paragraph()
    p.add_run(label + " ").bold = True
    p.add_run(text)
    p.paragraph_format.keep_with_next = label.startswith("Table")
    p.paragraph_format.space_before = Pt(6)
    return p


def figure(path, width_cm, label, text):
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    caption(label, text)


def table(header, rows, widths_cm, size=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    trpr = t.rows[0]._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trpr.append(el)
    for row_cells, values, bold in [(t.rows[0].cells, header, True)] + \
            [(None, r, False) for r in rows]:
        cells = row_cells or t.add_row().cells
        for c, v, w in zip(cells, values, widths_cm):
            c.width = Cm(w)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run("" if v is None else str(v))
            run.font.size = Pt(size)
            run.bold = bold
            if bold:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:fill"), "D9E2F3")
                c._tc.get_or_add_tcPr().append(shd)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ---------------------------------------------------------------- title page

para("Hallucination detection for large language models at response, sentence, span and "
     "claim level: a structured review of signals, evaluation practice and evidence gaps "
     "(2022-2026)", align="c", size=15, after=12).runs[0].bold = True
para("Adithyan C S S (Reg. No. 24BCE2820)", align="c", after=2)
para("School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India",
     align="c", after=2)
para("Course BCSE306L, Topic 16. Supervisor: Dr. K. Ragavan", align="c", after=2)
para("Corresponding author: adithyan.css2024@vitstudent.ac.in", align="c", after=14)

# ---------------------------------------------------------------- abstract

h1("Abstract")
para(f"Large language models (LLMs) write false or unsupported statements in fluent text, and "
     f"many automatic detectors have been proposed, each evaluated in its own way.",
     bold_lead="Background: ", align="j")
para("To map which signals hallucination detectors use, at which granularity they score "
     "output, and how well the published evidence supports comparison between them.",
     bold_lead="Objective: ", align="j")
para(f"OpenAlex (four queries) and the arXiv API (two queries) were searched on 24 October "
     f"2026 and 33 records were added from other sources. Of "
     f"{prisma['Total records identified']:,} records identified, "
     f"{prisma['Records after duplicates removed (screened)']} were screened, "
     f"{prisma['Full texts assessed for eligibility']} full texts were assessed and {N} "
     f"primary studies ({min(years)}-{max(years)}) were included. Each study was coded in a "
     f"31-field extraction matrix and an eight-item quality checklist by a single reviewer.",
     bold_lead="Methods: ", align="j")
para(f"Of {ND} detectors, attention-based ({sig['Attention']}), sampled-consistency "
     f"({sig['Sampled consistency']}) and hidden-state ({sig['Hidden states']}) methods were "
     f"the largest groups. {gran['Response']} of {ND} scored whole responses and "
     f"{gran['Claim']} scored individual claims; only {len(tf_single_claim)} claim-level "
     f"detector was both training-free and single-pass, and it needs an auxiliary NLI model. "
     f"Code was linked by {pct(code)} studies and own data released by {pct(data)}. Variance or "
     f"significance tests were reported by {pct(var)}, transfer to new data, models or languages "
     f"was tested by {pct(len(transfer))}, compute cost was measured by {pct(len(costed))} and "
     f"evaluation beyond English appeared in {pct(len(multiling))}. Only {comp['Direct']} "
     f"studies used a protocol that allows direct comparison with other studies.",
     bold_lead="Results: ", align="j")
para("Detection methods are numerous but their evidence is hard to compare, so no ranking "
     "across papers is given. Shared claim-level benchmarks, cost reported against generation "
     "time, interval estimates and multilingual test sets are the main needs; nine gaps with "
     "counts and proposed studies are listed.",
     bold_lead="Conclusion: ", align="j")
para("hallucination detection; large language models; uncertainty quantification; "
     "claim-level verification; attention; evaluation methodology; structured review",
     bold_lead="Keywords: ", align="j")

# ---------------------------------------------------------------- 1 introduction

h1("1. Introduction")
body("LLMs are now used to answer questions, summarise documents and draft long reports. They "
     "also produce statements that read well but are false, or that are not supported by the "
     "source they were given. These outputs are usually called hallucinations. Because a "
     "hallucinated sentence looks the same as a correct one, a reader cannot catch it by style "
     "alone, and automatic detection is needed before LLM output is trusted in search, "
     "medicine, law or education.")
body(f"Work on detection has grown quickly. In the included set the number of studies rises "
     f"from {years[2022]} in 2022 and {years[2023]} in 2023 to {years[2024]} in 2024 and "
     f"{years[2025]} in 2025, with {years[2026]} already in 2026 (Figure 2). The methods differ "
     f"in nearly every design choice. Some sample the model several times and measure "
     f"agreement [6, 7, 54]. Some read attention maps or hidden states from the single forward "
     f"pass that produced the text [3, 15, 16]. Others call an external verifier or a stronger "
     f"LLM [22, 30, 41]. They also score different units: a whole answer, a sentence, a token "
     f"span or an atomic claim. Since each paper chooses its own datasets, labels and metrics, "
     f"the reported numbers are hard to put side by side.")
body("Existing surveys (Section 2) cover hallucination in general, uncertainty quantification "
     "in general, or abstention. None of the five examined here codes each study for score "
     "granularity, compute cost, transfer testing and reproducibility, or counts how often "
     "these are reported. This review does that. It is a structured review with a documented "
     "database search. It is not presented as a full systematic review, because one reviewer "
     "screened the records and not every eligible full text was retrieved (Sections 4 and 10).")
body("The review answers five research questions:")
for rq in (
        "RQ1. Which signal families are used to detect hallucinations, and how do they divide "
        "by supervision, granularity and cost?",
        "RQ2. Which datasets, labelling procedures and metrics are used, and how comparable are "
        "the reported results?",
        "RQ3. How often do studies measure the compute cost of detection, and which cost classes "
        "exist?",
        "RQ4. How often is generalisation tested across datasets, models and languages?",
        "RQ5. How reproducible is the evidence, judged by code release, data release and "
        "statistical reporting?"):
    bullet(rq)
body(f"The contributions are: (i) a documented search and screening ledger whose counts can be "
     f"traced to a workbook; (ii) a four-axis taxonomy of {ND} detectors with an assignment rule "
     f"for every category; (iii) a comparison table of all {N} studies without cross-setting "
     f"ranking; (iv) counts of evaluation and reproducibility practice; and (v) nine "
     f"evidence-based gaps with proposed studies. Section 2 compares existing reviews, Section 3 "
     f"defines terms, Section 4 gives the method, Section 5 the taxonomy, Section 6 the "
     f"comparative analysis, Section 7 datasets, metrics and evaluation, Section 8 challenges "
     f"and limitations of the methods, Section 9 gaps and future directions, Section 10 the "
     f"limitations of this review and Section 11 the answers to the research questions.")

# ---------------------------------------------------------------- 2 existing reviews

h1("2. Existing reviews and how this review differs")
body("Five recent reviews [55-59] were compared on scope, period, taxonomy, evaluation analysis "
     "and what they leave out (Table 1). Huang et al. [57] give the broadest account of "
     "hallucination, covering causes, detection, benchmarks and mitigation; detection is one "
     "section among many. Shorinwa et al. [58] and Geng et al. [59] sort uncertainty and "
     "confidence methods into white-box and black-box classes. Geng et al. appeared before most "
     "attention-based and claim-level detectors, which in this set date from 2024 onward. Wen "
     "et al. [55] cover abstention, where detection is only one input to a refusal decision. "
     "Kang et al. [56] are closest to this topic and run their own experiments on two models, "
     "but they place attention methods inside a general internal-state group and leave "
     "claim-level detection to future work.")
body("None of the five states its search strategy or search period. None counts how often "
     "primary studies report variance, compute cost, transfer or non-English evaluation. This "
     "review adds those counts, a granularity axis, a cost axis and a comparability code for "
     "each study.")
caption("Table 1.", "Comparison of existing reviews with this review.")
table(["Review", "Year", "Scope and period", "Taxonomy", "Evaluation analysis",
       "Missing / what this review adds"],
      [r[:6] for r in rev],
      [3.0, 1.9, 3.3, 2.3, 3.0, 3.0], size=8)

# ---------------------------------------------------------------- 3 background

h1("3. Background and terminology")
body("This section fixes the terms used in the rest of the review.")
for lead, text in (
        ("Hallucination. ", "Generated content that is false with respect to world knowledge "
         "(a factuality error) or not supported by a given source such as a retrieved passage "
         "(a faithfulness error) [57]."),
        ("Detector. ", "A method that assigns a score to a unit of LLM output so that "
                       "hallucinated units score higher (or lower) than correct ones. Scores "
                       "are compared with labels from humans, from an LLM judge, or from "
                       "string matching against a reference answer."),
        ("Granularity. ", "The unit that receives a score: a whole response, a sentence, a "
                          "token span or named entity, or an atomic claim (a short statement "
                          "with one fact) [22]."),
        ("Signal source. ", "What the detector reads. Output probabilities are the token "
                            "probabilities assigned during generation; claim-conditioned "
                            "variants such as CCP [2] adjust them for alternatives with the "
                            "same meaning. Sampled consistency draws several answers and "
                            "measures agreement, often in meaning space, as in semantic entropy "
                            "[6, 54]. Hidden-state probes train a classifier on intermediate "
                            "activations [13, 16]. Attention-based methods use attention "
                            "weights, for example the share of attention on the source context "
                            "[15] or on sink tokens [5]. External verifiers split an answer into "
                            "claims and check each against a reference with another model [22, "
                            "41]. Self-report asks the model for its own confidence [8]."),
        ("White-box and black-box. ", "A white-box detector needs model internals (weights, "
                                      "activations or token probabilities); a black-box "
                                      "detector needs only generated text."),
        ("Training-free and supervised. ", "A supervised detector is fitted on labelled "
                                           "hallucinations; a training-free one uses a fixed "
                                           "formula or an existing model without new labels."),
        ("Metrics. ", "AUROC is the probability that a random hallucinated unit scores above a "
                      "random correct one. PR-AUC (area under the precision-recall curve) is "
                      "more informative when hallucinations are rare. The prediction-rejection "
                      "ratio (PRR) measures how much output quality improves when the "
                      "highest-scoring outputs are rejected [28]. Calibration error (ECE) "
                      "compares stated confidence with observed accuracy."),
        ("Transfer. ", "Testing a detector on data, models, tasks or languages different from "
                       "those used to fit, tune or calibrate it.")):
    bullet(text, lead)

# ---------------------------------------------------------------- 4 methodology

h1("4. Review methodology")
h2("4.1 Design")
body("The review follows the reporting items of PRISMA 2020 [60] where they apply to a "
     "single-reviewer course review. All decisions and counts are recorded in a workbook "
     "(Search_Log, Screening_Sheet, PRISMA_Counts, Extraction_Matrix, Quality_Check, "
     "Evidence_Summary, Gap_Table), which is supplied as supplementary material together with "
     "the exports of the verification re-run described in Section 4.2.")
h2("4.2 Sources and search")
body(f"OpenAlex [61], an open index that covers ACL Anthology, IEEE, ACM, Springer and arXiv "
     f"records, and the arXiv API were searched on 24 October 2026. Scopus and Web of Science "
     f"were not available to the author through an automatable interface; this is listed as a "
     f"limitation. OpenAlex queries searched title and abstract from 1 January 2020 to the "
     f"search date. Table 2 gives the exact strings and record counts. For each OpenAlex query "
     f"the 200 most relevant records were exported; "
     f"{prisma['Records not exported (ranked below the top 200 of a query; not screened)']} "
     f"lower-ranked records were not exported. Thirty-three records came from other sources: "
     f"29 studies collected before the search, one found by citation chasing (Farquhar et al. "
     f"[54]), two surveys and one off-topic paper from the prior collection.")
body("All six queries were run again on 26 October 2026 to check that the search can be "
     "repeated. OpenAlex returned 391, 337, 409 and 420 records for OA1-OA4 (two days earlier: "
     "389, 336, 401 and 413; the index grows daily), and 197, 199, 195 and 195 of the 200 "
     "originally exported records appeared again in the top 200. The arXiv queries returned "
     "the same 132 and 84 records. The re-run exports are kept with the workbook.")
caption("Table 2.", "Search strings as executed and records returned (24 October 2026).")
table(["ID", "Source", "Query", "Records"], [
    ["OA1", "OpenAlex", '("hallucination detection" OR "uncertainty quantification" OR "confidence '
     'estimation") AND ("large language model" OR LLM) AND (claim OR "fine-grained" OR '
     '"long-form" OR "token-level" OR "atomic fact")', 389],
    ["OA2", "OpenAlex", '(attention OR "attention head" OR "hidden state" OR probe) AND '
     'hallucination AND ("large language model" OR LLM) AND detection', 336],
    ["OA3", "OpenAlex", '(abstention OR "selective generation" OR "selective prediction" OR '
     'conformal) AND ("large language model" OR LLM) AND (hallucination OR factuality)', 401],
    ["OA4", "OpenAlex", '(benchmark OR dataset) AND hallucination AND (claim OR span OR '
     '"retrieval-augmented") AND detection AND ("large language model" OR LLM)', 413],
    ["AX1", "arXiv", 'abs:"hallucination detection" AND (abs:claim OR abs:"fine-grained" OR '
     'abs:"long-form" OR abs:"token-level")', 132],
    ["AX2", "arXiv", 'abs:hallucination AND (abs:"attention heads" OR abs:"attention maps" OR '
     'abs:"hidden states") AND abs:detection', 84],
    ["OS1", "Other sources", "No query: 29 primary studies already collected, Farquhar et al. "
     "2024 (Nature) from citation chasing, 2 surveys and 1 off-topic paper", 33],
], [1.2, 2.0, 11.0, 1.6])
h2("4.3 Eligibility criteria and quality questions")
body("Table 3 lists the inclusion and exclusion criteria applied at screening and the eight "
     "quality questions applied to every included study, with the rule used to answer each.")
caption("Table 3.", "Eligibility criteria and quality-appraisal questions.")
table(["Code", "Criterion or question", "How it was applied"], [
    ["I1", "Proposes, evaluates or benchmarks a method that detects, scores or abstains on "
           "hallucinated or incorrect content in LLM-generated text", "Title, abstract, then "
                                                                     "full text"],
    ["I2", "Reports a quantitative evaluation", "Full text"],
    ["I3", "Published from 2020 onward, in English", "Search filter and record date"],
    ["I4", "Peer-reviewed paper or arXiv preprint", "Venue field; preprints flagged"],
    ["E1", "Multimodal, speech or diffusion-model output", "Title and abstract"],
    ["E2", "Secondary study (review, survey, editorial)", "Title and abstract; used in "
                                                          "Section 2 instead"],
    ["E3", "Non-archival grey literature (Zenodo, SSRN, Preprints.org, Research Square)",
     "Source field"],
    ["E4", "Not about detection or uncertainty (mitigation, application or theory only)",
     "Title and abstract"],
    ["Q1", "Is the objective clear?", "Yes if the task and claim are stated"],
    ["Q2", "Is the method described well enough to re-implement?", "Yes if inputs, features and "
                                                                   "training are given"],
    ["Q3", "Are the datasets described?", "Yes if name, size, split and label source are given"],
    ["Q4", "Are baselines fair?", "Partial if strong recent baselines are missing or tuned "
                                  "differently"],
    ["Q5", "Is the evaluation valid (no leakage)?", "Partial if labels come from the same model "
                                                    "family or test data overlap training"],
    ["Q6", "Are variance or statistics reported?", "Yes for SD, standard error, intervals or "
                                                   "significance tests"],
    ["Q7", "Are code or data available?", "Yes if a working public link is given"],
    ["Q8", "Are limitations stated?", "Yes for a limitations section or explicit paragraph"],
], [1.2, 8.6, 6.0])
h2("4.4 Screening and selection")
body(f"After merging the exports and removing {prisma['Duplicates removed']} duplicates by "
     f"normalised title, {prisma['Records after duplicates removed (screened)']} records were "
     f"screened on title and abstract (Figure 1). "
     f"{prisma['Excluded at title/abstract stage']} were excluded (E1 73, E2 32, E3 94, E4 120). "
     f"Of {prisma['Full texts sought']} records eligible for full-text reading, "
     f"{prisma['Full texts assessed for eligibility']} were retrieved and assessed. The other "
     f"{prisma['Full texts not retrieved']} were not retrieved because one reviewer could not "
     f"read them all; the retrieved set was chosen by relevance rank, peer-review status and "
     f"coverage of method families that were thin in the prior collection. Two full texts were "
     f"excluded (one off-topic, one secondary study used in Section 2), leaving {N} included "
     f"primary studies. Title and abstract decisions were proposed with keyword rules for E1-E4 "
     f"and checked by hand; the rules and every decision are in the Screening_Sheet.")
figure(FIG / "fig1_flow.png", 13.5, "Figure 1.",
       "Flow of records through identification, screening and inclusion. All counts come from "
       "the PRISMA_Counts sheet of the review workbook.")
body("Figure 1 shows that the largest losses occur before screening (records ranked below the "
     "export cut-off) and at full-text retrieval. Both are capacity limits rather than "
     "eligibility decisions, so the included set is a prioritised sample of the eligible "
     "literature.")
h2("4.5 Data extraction and quality appraisal")
body("For each study 31 fields were extracted: bibliographic data, task, method family, method "
     "details, datasets, baselines, metrics, main result, evaluation setting, robustness "
     "testing, code and data availability, variance reporting, author-stated and "
     "reviewer-identified limitations, comparability, evidence strength and RQ mapping. Fields "
     "a paper did not report were coded N/R. Every study was read from its full text (abstract, "
     "method, experimental setup, result tables and limitations section); appendices were "
     "opened only to check a specific item, so the workbook marks every study as read "
     "'Partly'. On 26 October 2026, 19 quality ratings first made from HTML versions were "
     "checked again against the PDFs. Every DOI was resolved at doi.org and its registered "
     "title compared with the cited title (all 61 matched); studies without a publisher DOI "
     "are cited with their arXiv DOI.")
body("Quality was judged with the eight questions in Table 3. The answers describe reporting "
     "quality; they are not added into a score and studies are not ranked by them.")
caption("Table 4.", f"Quality appraisal of the {N} included studies (number of studies per "
                    f"answer).")
qnames = ["Q1 Objective clear", "Q2 Method described", "Q3 Dataset described",
          "Q4 Baselines fair", "Q5 Evaluation valid", "Q6 Variance / statistics",
          "Q7 Code or data available", "Q8 Limitations stated"]
table(["Question", "Yes", "Partial", "No", "Not reported"],
      [[qn_, Q[k]["Yes"], Q[k]["Partial"], Q[k]["No"], Q[k]["Not reported"]]
       for qn_, k in zip(qnames, Q)], [5.6, 2.2, 2.2, 2.2, 2.6])
body(f"Reporting of objectives, methods and data is good: Q1-Q3 are met by at least "
     f"{min(Q[k]['Yes'] for k in ('Q1', 'Q2', 'Q3'))} of {N} studies. The weak item is Q6: only "
     f"{Q['Q6']['Yes']} studies report variance, intervals or significance tests, and "
     f"{Q['Q6']['No'] + Q['Q6']['Not reported']} report none. {Q['Q5']['Partial']} studies were "
     f"rated Partial on Q5, mostly because labels came from the same LLM family that generated "
     f"the text or because chunk labels were derived from token labels. Q4 is Partial for "
     f"{Q['Q4']['Partial']} studies, usually because the newest baselines of the same family "
     f"were not run.")

# ---------------------------------------------------------------- 5 taxonomy

h1("5. Taxonomy of hallucination detectors")
bench = [t[0] for t in tax if t[2] == "Benchmark / resource"]
analysis = [t[0] for t in tax if t[2] == "Analysis / methodology"]
body(f"Of the {N} studies, {ND} propose or evaluate a detector. {len(bench)} are benchmarks or "
     f"evaluation resources {refs(bench)} and {len(analysis)} are analysis or methodology papers "
     f"{refs(analysis)}. The taxonomy was built from the extraction matrix, not taken from an "
     f"earlier survey: the method-detail column of each detector was coded on four axes (signal "
     f"source, supervision, granularity and cost class), and categories were merged until each "
     f"held at least two detectors. Table 5 gives the rule used to assign each category and the "
     f"number of detectors in it.")
caption("Table 5.", f"Taxonomy axes, assignment rules and number of detectors per category "
                    f"(n = {ND}).")
table(["Axis", "Category", "Assignment rule", "Detectors"], [
    ["Signal source", "Attention", "Main feature is computed from attention weights or "
                                   "attention-head outputs", sig["Attention"]],
    ["", "Sampled consistency", "Score depends on agreement among two or more sampled "
                                "outputs", sig["Sampled consistency"]],
    ["", "Hidden states", "Main feature is residual-stream or layer activations",
     sig["Hidden states"]],
    ["", "External verifier / LLM judge", "A separate model checks the output against a "
                                          "reference or retrieved text",
     sig["External verifier / LLM judge"]],
    ["", "Output probabilities", "Score uses only token probabilities of the generating model",
     sig["Output probabilities"]],
    ["", "Self-report", "The model states its own confidence or P(True)", sig["Self-report"]],
    ["Supervision", "Supervised", "Fitted on labelled hallucinations (including few-shot head "
                                  "selection)", sup["Supervised"]],
    ["", "Training-free", "No new labels needed", sup["Training-free"]],
    ["Granularity", "Response", "One score per answer", gran["Response"]],
    ["", "Sentence", "One score per sentence", gran["Sentence"]],
    ["", "Span / token / entity", "Scores for tokens, spans or named entities",
     gran["Span / token / entity"]],
    ["", "Claim", "Scores for atomic claims or claim triplets", gran["Claim"]],
    ["Cost class", "Single pass", "Reuses the forward pass that produced the text",
     cost["Single pass"]],
    ["", "Single pass + auxiliary model", "As above plus a small fixed model (for example NLI)",
     cost["Single pass + auxiliary model"]],
    ["", "Multi-pass", "Needs extra generations or forward passes", cost["Multi-pass"]],
    ["", "External model", "Calls a separate verifier, encoder or LLM", cost["External model"]],
], [2.6, 3.8, 7.4, 2.0])
figure(FIG / "fig2_years.png", 15.5, "Figure 2.",
       f"Included studies by publication year and signal source (n = {N}).")
body(f"Figure 2 shows that attention-based detectors are recent: "
     f"{sum(1 for r in em if r[0] in attn and r[2] >= 2025)} of the {len(attn)} appeared in 2025 "
     f"or 2026, while sampling methods were the main family in 2023 and 2024.")
figure(FIG / "fig3_taxonomy.png", 13.0, "Figure 3.",
       f"Detectors by signal source and granularity of the score (n = {ND}).")
body("Figure 3 crosses signal source with granularity. Response-level scoring dominates the "
     "three largest families, attention methods are the main group at span level, and the "
     "claim column is thin for every family.")
h2("5.1 Signal source")
fam_ids = lambda f: [t[0] for t in det if t[2] == f]
body(f"Attention ({sig['Attention']} detectors {refs(attn)}) is the largest family in this "
     f"set. Most attention methods train a probe on attention features such as lookback ratios "
     f"[15], Laplacian eigenvalues [32], topological divergence [33], sink scores [5] or "
     f"high-frequency energy [48]. Sampled consistency ({sig['Sampled consistency']} "
     f"{refs(fam_ids('Sampled consistency'))}) is the oldest family and is mostly "
     f"training-free. Hidden-state methods ({sig['Hidden states']} "
     f"{refs(fam_ids('Hidden states'))}) train probes on activations; recent work moves from "
     f"single layer-token pairs to full activation tensors [37] or residual-stream dynamics "
     f"[36]. External verifiers ({sig['External verifier / LLM judge']} "
     f"{refs(fam_ids('External verifier / LLM judge'))}) are used mainly for RAG and "
     f"fine-grained checking. Output-probability methods ({sig['Output probabilities']}) and "
     f"self-report methods ({sig['Self-report']}) are small groups.")
h2("5.2 Supervision")
body(f"Supervised detectors ({sup['Supervised']}) slightly outnumber training-free ones "
     f"({sup['Training-free']}). Among attention methods, {len(attn_sup)} of {len(attn)} need "
     f"labelled data, at least to choose heads [33]. Training-free methods are mostly "
     f"sampling-based. Three studies reduce the need for labels without removing it: "
     f"pseudo-labels from uncertainty [53], a probe set of about 100 examples [33] and "
     f"multi-LLM training with adapters for new models [37].")
h2("5.3 Granularity")
body(f"Response-level scoring dominates ({gran['Response']} of {ND}). {gran['Claim']} "
     f"detectors score claims {refs(claim)}, {gran['Span / token / entity']} score spans, "
     f"tokens or entities and {gran['Sentence']} score sentences. Claim-level detectors are "
     f"sampling-based [25, 26, 51], supervised on hidden states or attention [4, 27, 43], an "
     f"external checker [41], calibrated self-report [31], or probability-based with an "
     f"auxiliary NLI model [2].")
h2("5.4 Cost class")
body(f"Single-pass detectors ({cost['Single pass']}, plus "
     f"{cost['Single pass + auxiliary model']} with an auxiliary model) reuse the forward pass "
     f"that produced the text. Multi-pass detectors ({cost['Multi-pass']}) need extra "
     f"generations or passes; semantic entropy, for example, uses about ten samples [54]. "
     f"{cost['External model']} detectors call an external model. Cost class is not the same as "
     f"measured cost, which Section 7.3 discusses.")
h2("5.5 Least-studied combinations and limits of the taxonomy")
body(f"Crossing the axes shows where evidence is thin. Only {len(tf_single_claim)} detector "
     f"{refs(tf_single_claim)} is training-free, single-pass and claim-level, and it relies on "
     f"an auxiliary NLI model. Only one attention detector scores claims [4], and it is "
     f"supervised. No external verifier scores whole responses in this set, and no sampling "
     f"method scores spans. The taxonomy has two limits. First, the codes are the reviewer's "
     f"reading of each method description, and hybrid methods (for example attention-guided "
     f"resampling [35] or attention plus feed-forward signals [18]) were placed by their main "
     f"signal. Second, the axes describe design, not performance: a category with many "
     f"detectors is not shown to work better.")

# ---------------------------------------------------------------- 6 comparative analysis

h1("6. Comparative analysis")
body(f"Table A1 (Appendix A) lists all {N} studies with author, year, problem or domain, method, "
     f"dataset, baseline, metrics, main result and limitations. Results are copied as reported "
     f"by the authors. Each study was coded for comparability: {comp['Direct']} use a shared "
     f"protocol that allows direct comparison with other studies (for example the RAGTruth "
     f"split [24, 40], the LM-Polygraph protocol [28, 43] or MUCH [1]), {comp['Conditional']} "
     f"are comparable only under conditions (same dataset but different models, labels or "
     f"splits) and {comp['Non-comparable']} are not comparable. For this reason families are "
     f"compared by design trade-offs and within-paper results, and no cross-paper leaderboard "
     f"is built. Numbers from different datasets, settings or hardware are never ranked against "
     f"each other.")
h2("6.1 Single-pass internal signals versus sampling")
body("Within-paper comparisons repeatedly show that cheap internal-signal probes can match or "
     "beat sampling methods on the data they were trained on. Semantic entropy probes recover "
     "much of semantic entropy's performance from one forward pass [16]. TOHA reports results "
     "competitive with SelfCheckGPT and semantic entropy at much lower cost [33]. LettuceDetect, "
     "a 150-400M parameter encoder, reports 79.22% example-level F1 on RAGTruth at 30 to 60 "
     "examples per second on one GPU [40]. The price is supervision: these detectors need labels "
     "for each model, and several authors state that heads or probes do not transfer across "
     "architectures [32, 33].")
h2("6.2 Attention-based detectors")
body("Attention methods share one idea: a model that is not reading its context or its earlier "
     "answer is more likely to hallucinate [15, 17, 18]. They differ in which statistic they "
     "read, and the reported gains between them are small. Binkowski et al. [5] report "
     "sink-score probes best in 23 of 28 model-dataset pairs, but most gains over Lookback Lens "
     "and spectral features are within one standard deviation. Only one attention detector is "
     "training-free and single-pass [3], and it scores responses. Only one attention detector "
     "scores claims [4], and it is supervised.")
h2("6.3 Claim-level and fine-grained detection")
body("Fine-grained evaluation changes conclusions. RefChecker reports that checking claim "
     "triplets beats response-level checking by about 10 macro-F1 points on the same responses "
     "[41]. Zhang et al. find that models that look well calibrated per response are poorly "
     "calibrated per atomic claim [31]. HalluEntity shows that token-probability scores "
     "over-predict hallucinations at entity level (AUROC about 0.54-0.62) while context-aware "
     "scores reach about 0.77-0.78 across 17 LLMs [42]. MUCH [1] and LM-Polygraph [28] are the "
     "only harnesses in this set that evaluate claim-level detectors in more than one language "
     "under one protocol.")
h2("6.4 External verifiers")
body("Verifier pipelines (FActScore [22], FAVA [30], RefChecker [41], ANAH-v2 [38]) can say "
     "which claim is wrong and why, and they work with closed models. They depend on a "
     "reference source and on a strong LLM. ANAH-v2 reports that a 7B annotator exceeds GPT-4 "
     "on its own test set (F1 89.30 vs 87.11), but the gold labels began as GPT-4 "
     "pre-annotations, which the authors note can bias similarity metrics [38].")

# ---------------------------------------------------------------- 7 datasets, metrics, evaluation

h1("7. Datasets, metrics and evaluation analysis")
h2("7.1 Datasets")
body("Short-answer QA sets dominate. TriviaQA is used by 16 studies, followed by biography sets "
     "built on FActScore, WikiBio or similar entity lists (12), SQuAD (8), Natural Questions "
     "(8), maths reasoning (8), CoQA (7) and RAGTruth (7) (Figure 4). Long-form factuality sets "
     "such as LongFact appear in only 2 studies. Biographies are the main long-form test bed, so "
     "long-form evidence is concentrated on one genre.")
figure(FIG / "fig4_datasets_metrics.png", 16.0, "Figure 4.",
       "Datasets and metrics used by the included studies (a study can use several). Counts "
       "come from keyword rules over the dataset and metric columns of the extraction matrix.")
h2("7.2 Labels and metrics")
body(f"AUROC is the default metric (38 of {N} studies). F1, precision or accuracy is reported "
     f"by 14, PR-AUC or average precision by 12, PRR by 4 and calibration measures by 4 "
     f"(Figure 4). Labels come from very different sources: substring match with a reference "
     f"answer [45], ROUGE thresholds [6], human annotation [22, 24, 30] and, in {judge_n} "
     f"studies, GPT-4-class judges or automatic pipelines {refs(set(judge))}. At least three "
     f"studies check judge labels against humans or a second judge [1, 15, 32]. Sanity tests by "
     f"Zollicoffer et al. show that reasoning-trace detectors can score well by reading "
     f"answer-level artefacts [52], so label and test design affect the measured skill of a "
     f"detector.")
h2("7.3 Evaluation practice")
body(f"Figure 5 summarises how the {N} studies report evidence. Code is linked by {pct(code)}, "
     f"but own data is released by only {pct(data)}. Variance, intervals or significance tests "
     f"appear in {pct(var)}. Transfer to data, models, tasks or languages other than those used "
     f"for fitting is tested by {pct(len(transfer))}; counting one further study that tests "
     f"controlled perturbations [52], {pct(len(shift))} report some shift test. Compute cost "
     f"(runtime, throughput, FLOPs or overhead against generation) is measured by "
     f"{pct(len(costed))}, although {cost['Multi-pass']} of the {ND} detectors need extra "
     f"generations. Only {pct(len(multiling))} evaluate beyond English, and "
     f"{comp['Direct']} of {N} ({100 * comp['Direct'] / N:.1f}%) are directly comparable with "
     f"other studies.")
figure(FIG / "fig5_evaluation.png", 14.5, "Figure 5.",
       f"Evaluation and reproducibility indicators across the {N} included studies.")
body("The pattern in Figure 5 is that what is cheap to report (a code link, a peer-reviewed "
     "venue) is common, while what costs extra experiments (repeated runs, transfer tests, "
     "timing, other languages) is rare.")

# ---------------------------------------------------------------- 8 challenges

h1("8. Challenges and limitations of current methods")
body("Each challenge below is backed by counts from the extraction matrix and by two or three "
     "example studies.")
bullet(f" {len(whitebox)} of {ND} detectors read attention, hidden states or token "
       f"probabilities and so need open weights. Attention methods also need eager attention, "
       f"which is slower than fused kernels [5]; LapEigvals works only on open LLMs [32] and "
       f"semantic density needs token probabilities that some APIs hide [11]. Closed models "
       f"allow only sampling or external verifiers.", "Access to model internals.")
bullet(f" {sup['Supervised']} of {ND} detectors are supervised and need labelled "
       f"hallucinations for each model and task, and {judge_n} of {N} studies take labels from "
       f"LLM judges or automatic pipelines. Detectors may then learn the judge's errors: "
       f"LapEigvals uses gpt-4o-mini labels [32], PsiloQA uses GPT-4o for spans [39] and ANAH-v2 "
       f"starts from GPT-4 pre-annotations [38].", "Labels.")
bullet(f" {cost['Multi-pass']} detectors need several passes and {cost['External model']} "
       f"call an external model, but only {len(costed)} of {N} studies measure cost. Semantic "
       f"entropy uses about ten samples [54], conformal abstention needs O(k^2) extra inferences "
       f"[29] and Finch-Zk samples several different models [46].", "Compute cost.")
bullet(f" {gran['Response']} of {ND} detectors score whole responses, yet long answers mix true "
       f"and false claims. Claim triplets change results by about 10 macro-F1 points [41], "
       f"atomic-level calibration is worse than response-level calibration [31], and claim "
       f"segmentation adds its own error [1].", "Granularity mismatch.")
bullet(f" {comp['Conditional']} studies are comparable only under conditions and "
       f"{comp['Non-comparable']} not at all. A gain of 0.01-0.03 AUROC within one standard "
       f"deviation [5], or RAGTruth results on different base models [18, 33, 40], cannot be "
       f"ranked across papers.", "Comparability.")
bullet(f" Only {len(transfer)} of {N} test transfer. Probes trained on one model's output lose "
       f"accuracy on another's (AUROC 0.726 in-domain vs 0.625 out of domain for a 3B model "
       f"[27]); heads chosen for one architecture do not carry over [32, 33].",
       "Generalisation.")

# ---------------------------------------------------------------- 9 gaps

h1("9. Research gaps and future directions")
body("Table 6 lists the gaps. Each is supported by counts from the extraction matrix and comes "
     "with a proposed study. Strong means the pattern holds across many studies; Moderate means "
     "it depends partly on the reviewer's coding.")
caption("Table 6.", "Evidence-based research gaps (counts from the review workbook).")
table(["ID", "Gap", "Evidence (n of N, study IDs)", "Why it matters", "Proposed study",
       "Strength"], [g[:6] for g in gaps], [0.9, 2.6, 5.0, 2.7, 3.2, 1.6], size=8)
body("Three directions follow from the gaps. First, claim-level evaluation of cheap detectors: "
     "G1 and G2 show that the lowest-cost detectors score whole responses, while claim-level "
     "detectors are supervised or need sampling. A training-free, single-pass claim-level "
     "detector evaluated on MUCH [1] against CCP [2] and UHeads [4] would test whether this "
     "trade-off is necessary. Second, shared reporting: overhead as a percentage of generation "
     "time, bootstrap intervals with paired differences, and judge-human agreement (G3, G5, "
     "G9). Third, transfer and language coverage: train on one benchmark and test zero-shot on "
     "another, and include non-English sets such as MUCH and PsiloQA [39] (G4, G8).")

# ---------------------------------------------------------------- 10 limitations

h1("10. Limitations of this review")
body(f"One reviewer screened and extracted all records. Screening decisions were proposed with "
     f"keyword rules and software assistance and are marked in the workbook for confirmation "
     f"by the author; no second reviewer checked them. Scopus and Web of Science were not "
     f"searched. Only the top 200 records per OpenAlex query were exported, and "
     f"{prisma['Full texts not retrieved']} eligible records were not retrieved in full text, "
     f"so the included set is a prioritised sample of the eligible literature and the counts "
     f"describe that sample. Appendices were not read in full, so some variance or cost "
     f"reporting placed only in appendices may be missed; the counts in Section 7.3 are "
     f"therefore lower bounds for those items. Taxonomy codes are the reviewer's judgments from "
     f"method descriptions. {N - peer} of {N} studies are preprints. Finally, the author "
     f"develops a claim-level attention detector as a separate course project; that work is not "
     f"among the included studies, and gap G2 was derived from the counts, not from that "
     f"project.")

# ---------------------------------------------------------------- 11 conclusion

h1("11. Conclusion")
body(f"RQ1. Six signal families appear across {ND} detectors. Attention ({sig['Attention']}) "
     f"and sampled consistency ({sig['Sampled consistency']}) are the largest, followed by "
     f"hidden states ({sig['Hidden states']}). Supervised ({sup['Supervised']}) and "
     f"training-free ({sup['Training-free']}) detectors are close in number, {gran['Response']} "
     f"of {ND} score whole responses and {gran['Claim']} score claims.")
body(f"RQ2. Short-answer QA (TriviaQA in 16 studies) and biographies (12) dominate, AUROC is "
     f"reported by 38 of {N}, and {judge_n} studies take labels from LLM judges or automatic "
     f"pipelines. Only {comp['Direct']} of {N} studies are directly comparable with others, so "
     f"results should not be ranked across papers.")
body(f"RQ3. {cost['Multi-pass']} detectors need several passes and {cost['External model']} "
     f"need an external model, but only {pct(len(costed))} studies measure cost.")
body(f"RQ4. Transfer is tested by {pct(len(transfer))} studies and non-English evaluation by "
     f"{pct(len(multiling))}. Where transfer is tested, supervised probes lose accuracy outside "
     f"their training data [27, 43], while multi-LLM training helps [37].")
body(f"RQ5. Code is often released ({pct(code)}), data less often ({pct(data)}), and "
     f"statistical uncertainty is reported by only {pct(var)}.")
body(f"The largest gap is in evaluation rather than in method design: only {comp['Direct']} "
     f"of {N} studies can be compared directly and only {var} report uncertainty, so most "
     f"reported improvements cannot be confirmed across papers. The most useful next step is "
     f"to re-run existing detectors, including cheap single-pass ones, at claim level on a "
     f"shared multilingual harness such as MUCH, reporting overhead against generation time and "
     f"bootstrap intervals for every comparison.")

# ---------------------------------------------------------------- declarations

h1("Declarations")
para("Adithyan C S S: conceptualisation, search, screening, data extraction, quality "
     "appraisal, analysis, figures and writing.", bold_lead="Author contributions. ", align="j")
para("Dr. K. Ragavan supervised the course project.", bold_lead="Acknowledgements. ",
     align="j")
para("None.", bold_lead="Funding. ", align="j")
para("The author is developing a claim-level hallucination detector in a separate course "
     "project (see Section 10).", bold_lead="Competing interests. ", align="j")
para("The review workbook (search log, screening sheet, PRISMA counts, extraction matrix, "
     "quality check, evidence summary and gap table, listing every screened record) and the "
     "exports of the 26 October 2026 search re-run are supplied as supplementary material.", bold_lead="Data availability. ", align="j")
para("See the separate AI-use declaration submitted with this manuscript.",
     bold_lead="Use of AI tools. ", align="j")

# ---------------------------------------------------------------- references

h1("References")
for line in (HERE / "references.txt").read_text(encoding="utf-8").splitlines():
    if line.strip():
        p = para(line, size=10, after=3)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)

# ---------------------------------------------------------------- appendix (landscape)

land = doc.add_section(WD_SECTION.NEW_PAGE)
land.orientation = WD_ORIENT.LANDSCAPE
land.page_width, land.page_height = Cm(27.94), Cm(21.59)
for side in ("left_margin", "right_margin"):
    setattr(land, side, Cm(1.8))
h1("Appendix A. Comparison table of the included studies")
body("Results are as reported by the authors under each study's own setting. They are not "
     "comparable across rows unless the Comparability column of the workbook says Direct, and "
     "rows are ordered by study ID, not by performance. Limitations are author-stated; where a "
     "paper states none, the reviewer-identified limitation is given after 'Reviewer:'.")
caption("Table A1.", f"Comparison of the {N} included studies.")


def lim(r):
    a, b = str(r[19] or "N/R"), str(r[20] or "")
    if a.startswith("N/R") and b and b != "N/R":
        return f"{a}. Reviewer: {b}"
    return a


table(["Author", "Year", "Problem / domain", "Method", "Dataset", "Baseline", "Metrics",
       "Results (as reported)", "Limitations"],
      [[f"{r[1]} [{int(r[0][1:])}]", r[2], r[6], f"{r[7]}: {r[8]}", r[9], r[11], r[12],
        r[13], lim(r)] for r in em],
      [2.0, 1.0, 2.4, 3.6, 3.0, 2.6, 2.2, 4.0, 3.5], size=7)

doc.save(OUT)
print("saved", OUT)
print(dict(N=N, ND=ND, sig=dict(sig), sup=dict(sup), gran=dict(gran), cost=dict(cost),
           code=code, data=data, var=var, transfer=len(transfer), shift=len(shift),
           costed=len(costed), multiling=len(multiling), comp=dict(comp), judge=judge_n,
           tf_single_claim=tf_single_claim))
