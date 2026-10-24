# Review manuscript: status (24-10-2026)

## Files to submit
- `Review_Manuscript_Hallucination_Detection.docx`: the manuscript, in the order the professor asked for. Comparison table is Appendix A (Table A1).
- `Review_Workbook_GRASP.xlsx`: every count in the manuscript comes from this workbook.
- `figures/`: the 5 figures as PNG files (also embedded in the .docx).

`TAXONOMY_AND_GAPS.md` and `EXISTING_REVIEWS.md` are older drafts based on 29 papers. The workbook sheets `Taxonomy_Draft`, `Gap_Table` and `Existing_Reviews` replace them.

## What was done
- Search run on 24-10-2026: OpenAlex, 4 queries, 1,539 records; arXiv API, 2 queries, 216 records. Another 33 records came from other sources. Exact strings are in `Search_Log`.
- Screening: 746 records screened, 319 excluded with reasons E1 to E4, 56 full texts assessed, 54 primary studies included. The ledger is in `PRISMA_Counts`; open the file in Excel so the formulas calculate.
- Extraction_Matrix and Quality_Check now have 54 rows (S01-S54). The 25 new rows (S30-S54) were read from full-text HTML: abstract, setup, results and limitations, but not the appendices.
- DOIs: every publisher DOI and every arXiv DOI was resolved at doi.org on 24-10-2026. Two studies have no DOI and are cited by arXiv ID: S42 (TMLR) and S49.
- Existing reviews: R1-R5 are filled in. R1's venue is now TACL 2025.
- Taxonomy: 4 axes (signal, supervision, granularity, cost class) plus a language column.
- Gap table: G1-G9, each with counts.

## What you still have to do
1. Put your name and register number on the title page; it currently shows `[Author name], [Register number]`.
2. Check the Screening_Sheet decisions marked "AI-assisted; confirm" and replace that with your initials once you agree with them. The professor asked for screening decisions you made yourself.
3. Check these items, which I could not fully confirm:
   - S30 FAVA venue: the manuscript says COLM 2024, but Semantic Scholar lists the paper as arXiv only.
   - S33 year: it is ACL 2026 with an ACL 2026 DOI; the workbook says 2026.
   - The co-author lists in the references. I trimmed any names I could not confirm to "et al.".
4. If your department gives you Scopus access, run the same strings there and add the rows to Search_Log. If you do, the methodology section can drop that limitation.
5. Run the similarity check (Turnitin or similar) on the .docx.
6. Choose the target journal and move the text into its template. Change the reference style if the journal requires it; the current style is numbered, APA-like.
7. `papers/references.bib` entries `lee2026meanshift` and `yu2026rt4chart` are real arXiv papers: they appeared as records R0252 and R0129 in the search. They are not in the included set.
