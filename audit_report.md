# GRASP Project Audit Report

**Audit Date:** 2026-10-08  
**Scope:** Independent audit of the GRASP repository, implementation, evaluation methodology, and reported empirical claims.  
**Auditor Policy:** Strict evidence-based evaluation. Claims cite exact file paths, line numbers, and raw command execution outputs. Unverifiable items are explicitly designated `NOT VERIFIED`.

---

## PART A. Inventory (What Evidence Exists)

### Item 1. Results Inventory
* **Status:** CONFIRMED
* **Evidence:**
  Command run:
  ```powershell
  Get-ChildItem -Recurse -File results/ | Select-Object FullName, Length, LastWriteTime
  ```
  Raw output:
  ```
  FullName                                                            Length LastWriteTime
  --------                                                            ------ -------------
  C:\Users\adith\OneDrive\Desktop\ai_project\results\checkpoint_log.md   2813 09-09-2026 23:40:33
  ```
  Additional search for results across the entire repository:
  ```powershell
  Get-ChildItem -Recurse -File -Include "*.json", "*.npz", "*.npy", "*.log" | Select-Object FullName, Length, LastWriteTime
  ```
  Raw output:
  ```
  (No files found)
  ```
  Notebook cell outputs check:
  ```powershell
  python -c "import json; [print(nb, len([c for c in json.load(open(nb))['cells'] if c.get('outputs')])) for nb in ['notebooks/GRASP_colab.ipynb', 'Claude outputs/GRASP_colab.ipynb']]"
  ```
  Raw output:
  ```
  notebooks/GRASP_colab.ipynb 0
  Claude outputs/GRASP_colab.ipynb 0
  ```
* **One-line implication:** No evaluation JSONs (`grasp_en_final.json`, `grasp_en_sweep.json`, `grasp_*.json`), `timing.json`, `normaliser.npz`, `selected_heads.json`, or cache directories exist locally in `results/`, and all notebook output cells are empty.

---

### Item 2. Search for Reported Benchmark Numbers
* **Status:** CONFIRMED (Numbers absent from all source, result, and notebook data files)
* **Evidence:**
  Script search across all repository files for targets: `0.7087`, `0.7224`, `0.7198`, `0.8343`, `0.8251`, `0.7805`, `0.1037`, `60.78`, `586`, `727`, `152`.
  Raw output:
  ```
  Target: 0.7087 -> Hits (1): ['.\\report.html']
  Target: 0.7224 -> Hits (1): ['.\\report.html']
  Target: 0.7198 -> Hits (1): ['.\\report.html']
  Target: 0.8343 -> Hits (1): ['.\\report.html']
  Target: 0.8251 -> Hits (1): ['.\\report.html']
  Target: 0.7805 -> Hits (1): ['.\\report.html']
  Target: 0.1037 -> Hits (1): ['.\\report.html']
  Target: 60.78 -> Hits (1): ['.\\report.html']
  Target: 586 -> Hits (2): ['.\\report.html', '.\\docs\\02_reading_list.md'] (coincidental match in DOI)
  Target: 727 -> Hits (1): ['.\\report.html']
  Target: 152 -> Hits (1): ['.\\report.html']
  ```
* **One-line implication:** None of the reported experimental figures exist in any raw data artifact, cache file, evaluation log, or script in the repository; they appear solely in `report.html` (and the compiled `GRASP_Project_Report.pdf`) transcribed from conversational chat logs.

---

### Item 3. Local Availability of MUCH Repo, Features, and Annotations
* **Status:** CONFIRMED (All absent locally)
* **Evidence:**
  Command run:
  ```powershell
  python -c "from pathlib import Path; print('much dirs:', list(Path('.').rglob('*much*'))); print('npy:', len(list(Path('.').rglob('*.npy')))); print('jsonl:', list(Path('.').rglob('*.jsonl')))"
  ```
  Raw output:
  ```
  much dirs: []
  npy: 0
  jsonl: []
  ```
* **One-line implication:** The MUCH benchmark dataset, cached feature tensors (`.npy`), and annotation files are completely absent from the local filesystem (0 counts across all languages and configurations).

---

## PART B. Code Checks

### Item 4. Train/Test Split Enforcement in `scripts/03_score_and_eval.py`
* **Status:** REFUTED (No split logic exists)
* **Evidence:**
  [`scripts/03_score_and_eval.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/scripts/03_score_and_eval.py#L173-L215):
  ```python
  173:     get_all_annotations, get_tokenizer = load_much(args.much_root)
  174:     annotations = list(get_all_annotations().values())
  175:     print(f"{len(annotations)} annotations loaded")
  ...
  189:     y, native, by_agg, counts = score_corpus(
  190:         annotations, args.cache, norm, tokenizer_for,
  191:         lang=args.lang, tau=args.tau, alpha=args.alpha)
  ...
  199:     if args.sweep:
  200:         print("\n-- tau / alpha sweep (training split only) --")
  201:         best, grid = None, {}
  202:         for tau in (1.0, 2.0, 4.0, 8.0, 16.0):
  203:             for alpha in (0.3, 0.6, 0.9, 1.0):
  204:                 yy, nn, aa, _ = score_corpus(
  205:                     annotations, args.cache, norm, tokenizer_for,
  206:                     lang=args.lang, tau=tau, alpha=alpha)
  ```
  [`scripts/03_score_and_eval.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/scripts/03_score_and_eval.py#L103-L113):
  ```python
  103:     for annot in annotations:
  104:         gen = annot.generation
  105:         if lang and gen.lang != lang:
  106:             continue
  107:         if not has_consensus(annot.labels):
  108:             continue
  109:         path = feat_dir / f"{gen.id}.npy"
  110:         if not path.is_file():
  111:             n_missing += 1
  112:             continue
  ```
* **One-line implication:** Despite comments and print statements declaring "training split only", `03_score_and_eval.py` contains zero filtering by split (e.g. human expert test split vs automatic train split); hyperparameter sweeps and final evaluations evaluate the exact same claims, creating data snooping / leakage.

---

### Item 5. Source of the 727 English Claims / 152 Annotations
* **Status:** CONFIRMED (Implicit filter via cached file existence)
* **Evidence:**
  [`scripts/03_score_and_eval.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/scripts/03_score_and_eval.py#L103-L137):
  The filtering is entirely implicit:
  1. `gen.lang == 'en'`
  2. `has_consensus(annot.labels)`
  3. `(cache / "features" / f"{gen.id}.npy").is_file()`
  4. Drops trailing EOS chunk: `zip(chunks[:-1], labels[:-1])`
  
  In Google Colab, extraction was performed using:
  `TARGET_MODEL = 'meta-llama/Llama-3.2-3B-Instruct'`, `TARGET_TEMP = 0.7`.
  Only generations for that single config had cached `.npy` files. Out of 1,305 English consensus annotations in MUCH, 1,153 belonged to other models/configs (`1153 generations had no cached features`), leaving exactly 152 annotations and 727 scored claims.
* **One-line implication:** The 727 claims / 152 annotations represent solely the subset of English generations generated by `Llama-3.2-3B-Instruct` at T=0.7, determined by which files were cached rather than an explicit command-line filter.

---

### Item 6. Token Rank Normalisation Scope in `src/grasp/score.py`
* **Status:** CONFIRMED (Within-generation ranking); Empirical AUCs NOT VERIFIED locally
* **Evidence:**
  [`src/grasp/score.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/grasp/score.py#L92-L108):
  ```python
  92:     band = risk[layers]  # (len(layers), T)
  93:     fused = _rank_normalise(band).mean(axis=0)  # (T,)
  ...
  97: def _rank_normalise(x: np.ndarray) -> np.ndarray:
  98:     """Rank-normalise each row to [0,1]. Makes layers commensurable without
  99:     fitting anything to labels."""
  100:     out = np.empty_like(x, dtype=np.float64)
  101:     n = x.shape[1]
  102:     if n == 0:
  103:         return out
  104:     for i in range(x.shape[0]):
  105:         order = np.argsort(np.argsort(x[i]))
  106:         out[i] = order / max(n - 1, 1)
  107:     return (out - 0.5) * 4.0  # spread before the sigmoid
  ```
  `x` has shape `(len(layers), T)`, where `T` is the sequence length of the individual generation. The ranking loop ranks tokens along axis 1 (across the $T$ tokens of that single generation).
  Pooled vs per-generation ROC-AUC computation is **NOT VERIFIED locally** due to absence of feature `.npy` files and annotations.
* **One-line implication:** Token risk is ranked purely within each generation, making raw scores relative to intra-generation peaks rather than a calibrated cross-generation risk scale.

---

### Item 7. Claim-Length Confound Analysis
* **Status:** NOT VERIFIED locally (Data/features absent)
* **Evidence:**
  Cached features (`.npy`) and annotation files do not exist locally to compute Spearman correlation or length-controlled ROC-AUC.
  Code mechanism check ([`src/grasp/score.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/grasp/score.py#L141-L161)):
  ```python
  160:     m = s.max()
  161:     return float(m + np.log(np.sum(w * np.exp(tau * (s - m))) + EPS) / tau)
  ```
  Longer chunks contain more tokens, increasing the opportunity for maximum-risk token values to dominate and skew aggregated risk.
* **One-line implication:** The empirical influence of chunk token length on ROC-AUC cannot be quantified without access to the cached features.

---

### Item 8. Boundary Reset Recurrence and Overlapping Chunks
* **Status:** CONFIRMED (EMA within claims; overwrites shared tokens); Overlapping count NOT VERIFIED locally
* **Evidence:**
  [`src/grasp/score.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/grasp/score.py#L184-L205):
  ```python
  196:     out = token_scores.copy()
  197:     for chunk in token_chunks:
  198:         idx = [i for i in chunk if i < len(out)]
  199:         carry = 0.0
  200:         for j, i in enumerate(idx):
  201:             carry = token_scores[i] if j == 0 else (
  202:                 alpha * token_scores[i] + (1 - alpha) * carry
  203:             )
  204:             out[i] = carry
  205:     return out
  ```
  Lines 199-201 reset `carry` to `token_scores[i]` at the start of every chunk (`j == 0`), confirming no recurrence leaks across chunks.
  However, because `out[i] = carry` mutates `out` in place across chunks, if any token index $i$ is present in multiple chunks, chunk $k+1$ silently overwrites chunk $k$'s computed score for token $i$. Overlapping chunk counts in MUCH are NOT VERIFIED locally due to missing benchmark files.
* **One-line implication:** Boundary reset is strictly an intra-claim EMA, but mutates shared tokens if claim chunk spans overlap.

---

### Item 9. Alignment Shift in `src/grasp/features.py` (_reduce)
* **Status:** CONFIRMED (One-token prediction alignment shift)
* **Evidence:**
  [`src/grasp/features.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/grasp/features.py#L121-L144):
  ```python
  128:         s = cfg.sink_size
  129:         P = self._prompt_len
  ...
  138:         H, N, _ = a.shape
  139:         T = N - P
  ...
  143:         rows = a[:, P:, :]  # (H, T, N) attention FROM generated tokens
  ```
  In an autoregressive decoder:
  - Sequence positions $0 \dots P-1$ are prompt tokens.
  - Position $P-1$ produces the hidden state that predicts generated token 0 (position $P$).
  - Position $P+i$ produces the hidden state that predicts generated token $i+1$ (position $P+i+1$).
  
  The code slices `rows = a[:, P:, :]`, selecting rows $P \dots N-1$. It assigns row $P+i$ to `output_tokens[i]`.
  Thus, `output_tokens[0]` is assigned row $P$ (the attention used to predict `output_tokens[1]`). Row $P-1$ (the attention used to generate `output_tokens[0]`) is completely discarded. Row $N-1$ (the last row) is the attention predicting the token *after* the generation (post-EOS).
  
  Documentation check:
  [`src/grasp/features.py:124`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/grasp/features.py#L124): Docstring states only *"Rows are restricted to generated positions [P, N)"*. Nowhere in `features.py`, `README.md`, or `docs/01_concept.md` is this 1-token shift documented or acknowledged.
* **One-line implication:** GRASP has an off-by-one alignment shift: it pairs the token with the attention map computed *after* observing that token (predicting the next token) rather than the attention map that produced it.

---

### Item 10. Saliency Normalisation
* **Status:** CONFIRMED (Unnormalised sum); Position correlation NOT VERIFIED locally
* **Evidence:**
  [`src/grasp/features.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/grasp/features.py#L167-L173):
  ```python
  169:         gen_block = rows[:, :, P:]  # (H, T, T)
  170:         strictly_later = torch.tril(
  171:             torch.ones(T, T, device=a.device, dtype=torch.bool), diagonal=-1
  172:         )
  173:         W = (gen_block * strictly_later.unsqueeze(0)).sum(1)  # (H,T)
  ```
  Dimension 1 (query position $i$) is summed over: `sum(1)`. There is no division by the number of remaining tokens $(T - 1 - j)$.
  Token $j=0$ can receive attention from up to $T-1$ downstream tokens, while token $T-1$ (the final token) can receive attention from 0 tokens, mathematically forcing its saliency to 0.0 ([`src/tests/test_grasp.py:96`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/tests/test_grasp.py#L96)).
  Correlation between claim token position and $W$ on benchmark data is NOT VERIFIED locally due to absent feature files.
* **One-line implication:** Saliency $W$ is inherently biased towards earlier tokens in the generation simply because they have more subsequent tokens to receive attention from.

---

### Item 11. Softmin Formula Discrepancy
* **Status:** CONFIRMED (Formula sign inversion between docs and code)
* **Evidence:**
  In Code ([`src/grasp/score.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/grasp/score.py#L160-L161)):
  ```python
  160:     m = s.max()
  161:     return float(m + np.log(np.sum(w * np.exp(tau * (s - m))) + EPS) / tau)
  ```
  This is $\frac{1}{\tau} \ln \left( \sum w_i e^{\tau \cdot s_i} \right)$ (a LogSumExp / soft-**maximum** over risk $s$).
  
  In Report ([`report.html:328`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/report.html#L328)):
  ```html
  <code>Score(Claim) = &minus;(1/&tau;) &middot; ln &sum; [w_i &middot; exp(&minus;&tau; &middot; Risk_i)]</code>
  ```
  In Concept Doc ([`docs/01_concept.md:180`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/docs/01_concept.md#L180)):
  ```markdown
  180: soft-minimum over confidence instead of a mean.
  ```
* **One-line implication:** The formula in `report.html` is mathematically inverted: it places negative signs on $\tau$ and the logarithm (computing a soft-minimum over risk, which minimizes risk), whereas the code implements a soft-maximum over risk (maximizing risk to detect the weakest token).

---

### Item 12. Fusion Aggregator Selection on Test Data
* **Status:** CONFIRMED (Selection on evaluation set)
* **Evidence:**
  [`scripts/03_score_and_eval.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/scripts/03_score_and_eval.py#L254-L260):
  ```python
  254:             base = evaluate(annotations, lambda g: other.get(g),
  255:                             tokenizer_for=tokenizer_for, lang=args.lang)
  256:             base_agg = max(AGGREGATORS,
  257:                            key=lambda a: base["per_aggregator"][a].get("roc_auc", -1))
  258:             results["baseline_alone"] = base["per_aggregator"][base_agg]
  259:             print(line(f"{args.fuse_with}/{base_agg}", results["baseline_alone"]))
  ...
  290:                     sb.append(aggregate(ov[start:stop], base_agg))
  ```
  `base_agg` is chosen dynamically by taking the argmax over ROC-AUC on the evaluated `annotations`.
  In the recorded Colab run, for `ccp_24_8` on English, `base_agg` was selected as `'product'` (`ccp_24_8/product ROC=0.8087`).
* **One-line implication:** The baseline aggregator used for fusion is selected by maximizing test set ROC-AUC, violating strict out-of-sample protocol discipline.

---

### Item 13. Calibration Sample and `timing.json` Scope
* **Status:** CONFIRMED (Pass 1 and overheads excluded); Language mix NOT VERIFIED locally
* **Evidence:**
  [`scripts/02_extract_attention.py`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/scripts/02_extract_attention.py#L114-L178):
  Calibration pass runs lines 114-137:
  ```python
  114:     calib_gens = gens[: min(args.calib, len(gens))]
  ...
  144:     t0 = time.time()
  145:     for i, g in enumerate(gens):
  ...
  163:     total = time.time() - t0
  165:     (args.out / "timing.json").write_text(...)
  ```
  `timing.json` starts its timer at line 144, strictly inside Pass 2.
  It excludes:
  1. Pass 1 calibration over 200 generations (all heads).
  2. Model and tokenizer loading (lines 89-91).
  3. Head variance computation and selection (lines 132-137).
  4. Claim segmentation and token masking.
  Language/split mix of `gens[:200]` is NOT VERIFIED locally due to missing benchmark dataset.
* **One-line implication:** The reported extraction time (60.78s / 0.1037 s/gen) measures solely the second forward pass and disk writes, omitting calibration, model loading, and head selection.

---

## PART C. Reproduction

### Item 14. Unit Test Execution
* **Status:** CONFIRMED for `test_grasp.py` (Pass); PARTIAL FAIL for `test_much_eval.py`
* **Evidence:**
  Command 1:
  ```powershell
  python src/tests/test_grasp.py
  ```
  Raw output:
  ```
  ok  ablation toggles wired up
  ok  boundary reset isolates claims
  ok  causal mask respected
  ok  dispersion flat=1.000 > peaked=-0.000
  ok  grounding/sink exact
  ok  head filtering
  ok  pipeline recovers planted signal, AUROC=1.000
  ok  saliency counts only later tokens
  ok  saliency weights
  ok  softmin interpolates mean=0.338 -> 0.728 -> max=0.893
  ok  variance head selection

  All 11 tests passed.
  (Exit code: 0)
  ```
  Command 2:
  ```powershell
  python src/tests/test_much_eval.py
  ```
  Raw output:
  ```
  skip  aggregator cross-check (MUCH not importable)
  ok    annotator-disagreement filter
  ok    geometric aggregator keeps MUCH's +1e-5
  ok    label polarity: -1 is the positive class (ROC=1.000)
  ok    trailing EOS chunk dropped (3 chunks -> 2 scored)
  Traceback (most recent call last):
    File "C:\Users\adith\OneDrive\Desktop\ai_project\src\tests\test_much_eval.py", line 248, in <module>
      fn()
    File "C:\Users\adith\OneDrive\Desktop\ai_project\src\tests\test_much_eval.py", line 205, in test_masking_can_flip_the_ranking
      masked = evaluate(anns, lambda g: sig.get(g), tokenizer_for=lambda m: _Tok())
    File "C:\Users\adith\OneDrive\Desktop\ai_project\src\grasp\much_eval.py", line 94, in mask_stopwords
      stop = extended_stopwords_set()
    File "C:\Users\adith\OneDrive\Desktop\ai_project\src\grasp\much_eval.py", line 70, in nltk_stopwords_set
      import nltk
  ModuleNotFoundError: No module named 'nltk'
  (Exit code: 1)
  ```
* **One-line implication:** Math and feature reduction tests pass completely; evaluation mock tests fail locally due to missing `nltk` dependency and uninstalled MUCH repository.

---

### Item 15. Reproduction of `scripts/03_score_and_eval.py`
* **Status:** NOT VERIFIED locally
* **Evidence:**
  Executing `scripts/03_score_and_eval.py` requires `--much-root` and `--cache` pointing to valid extraction folders.
  Neither exists on the local machine. Attempting execution fails immediately with `FileNotFoundError` / missing directory arguments.
* **One-line implication:** Cannot re-evaluate scores locally; values rely on remote Colab console logs.

---

### Item 16. CCP Comparison Subset Alignment
* **Status:** CONFIRMED (Apples-to-oranges subset comparison in cross-lingual reporting)
* **Evidence:**
  From [`results/checkpoint_log.md:32-40`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/results/checkpoint_log.md#L32-L40):
  ```
  ### Per language (product)
  | Lang | n     | pos   | ROC    | PR     | TPR@10%FPR |
  |---|---|---|---|---|---|
  | en   | 4,204 | 1,164 | 0.8030 | 0.6867 | 0.564      |
  | de   | 4,790 | 1,626 | 0.7799 | 0.6807 | 0.458      |
  | fr   | 5,998 | 1,779 | 0.7594 | 0.5951 | 0.404      |
  | es   | 5,759 | 1,393 | 0.7190 | 0.5080 | 0.378      |
  ```
  These numbers encompass all 4 base models across 20,751 claims.
  
  From the recorded Colab execution logs:
  - English GRASP was evaluated on $n = 727$ claims (152 annotations) of `Llama-3.2-3B-Instruct`. On this aligned subset, CCP was **0.8251** (not 0.8030).
  - German GRASP was evaluated on $n = 719$ claims of `Llama-3.2-3B-Instruct` (achieving 0.7805).
  - French GRASP was evaluated on $n = 858$ claims of `Llama-3.2-3B-Instruct` (achieving 0.7469).
  - Spanish GRASP was evaluated on $n = 873$ claims of `Llama-3.2-3B-Instruct` (achieving 0.6970).
  
  In [`report.html:433-461`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/report.html#L433-L461):
  German GRASP (0.7805 on 719 claims) is compared directly against CCP's pooled figure of **0.7799** (on all 4,790 claims across 4 models) with the claim *"Beats CCP outright (unfused)!"*.
  CCP was never evaluated on the matching 719 German claims of `Llama-3.2-3B`.
* **One-line implication:** The reported German win over CCP is invalidly claimed by contrasting a single-model subpopulation against a 4-model pooled benchmark baseline.

---

### Item 17. Bootstrap 95% Confidence Intervals
* **Status:** NOT VERIFIED locally
* **Evidence:**
  Generation-level prediction arrays and ground-truth metadata are not present on local disk.
* **One-line implication:** Statistical significance and confidence intervals cannot be computed locally without exported prediction files.

---

## PART D. Repo Hygiene

### Item 18. File Redundancy and Folder Sizing
* **Status:** CONFIRMED
* **Evidence:**
  Command run:
  ```powershell
  python -c "import filecmp; print('checkpoint_log identical:', filecmp.cmp('results/checkpoint_log.md', 'Claude outputs/checkpoint_log.md', shallow=False)); print('notebooks identical:', filecmp.cmp('notebooks/GRASP_colab.ipynb', 'Claude outputs/GRASP_colab.ipynb', shallow=False))"
  ```
  Raw output:
  ```
  checkpoint_log identical: True
  notebooks identical: True
  ```
  Committed Junk and Directory Sizes:
  - `Claude outputs/`: 17,129 bytes (contains exact duplicates of `checkpoint_log.md` and `GRASP_colab.ipynb`).
  - `src/grasp/__pycache__/`: 5 `.pyc` files totaling 38,119 bytes.
  
  Folder size summary:
  ```
  . (Root)                : 345,359 bytes   (0.33 MB)
  .\Claude outputs        :  17,129 bytes   (0.02 MB) [DUPLICATE]
  .\docs                  :  27,980 bytes   (0.03 MB)
  .\notebooks             :  14,316 bytes   (0.01 MB)
  .\papers                : 50,350,132 bytes (48.02 MB) [98.9% of repository]
  .\results               :   2,813 bytes   (0.00 MB)
  .\scripts               :  45,396 bytes   (0.04 MB)
  .\src\grasp             :  32,131 bytes   (0.03 MB)
  .\src\grasp\__pycache__ :  38,119 bytes   (0.04 MB) [BYTECODE]
  .\src\tests             :  17,363 bytes   (0.02 MB)
  Total Repository Size   : ~50.89 MB
  ```
* **One-line implication:** The repository contains duplicate folders and unversioned bytecode, while 98.9% of total disk space is consumed by paper PDFs.

---

### Item 19. Contradictions Between Documentation and Code/Results
* **Status:** CONFIRMED
* **Evidence:**
  1. **Split Discipline:**
     - [`README.md:160, 188`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/README.md#L160) and [`docs/01_concept.md:327`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/docs/01_concept.md#L327) mandate *"Sweep on the training split only."*
     - [`scripts/03_score_and_eval.py:174, 204`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/scripts/03_score_and_eval.py#L174) contains no split partition; the sweep tunes hyperparameters on the evaluation set itself.
  2. **Runtime Overhead:**
     - [`README.md:22`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/README.md#L22) and [`docs/01_concept.md:305`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/docs/01_concept.md#L305) advertise *"≈1–3% overhead"*.
     - Real measurement on Tesla T4 was **6.4%** (0.1037s / 1.61s), more than double the claimed target.
  3. **Repository Status Table:**
     - [`README.md:220-221`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/README.md#L220-L221) states CCP reproduction is *"next"* and real results are *"not started"*.
     - [`results/checkpoint_log.md`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/results/checkpoint_log.md#L3) recorded CCP reproduction as PASSED on 2026-09-09.
  4. **Formula Inversion:**
     - [`report.html:328`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/report.html#L328) documents $-\frac{1}{\tau}\ln \sum w_i e^{-\tau \cdot \text{Risk}_i}$.
     - [`src/grasp/score.py:161`](file:///c:/Users/adith/OneDrive/Desktop/ai_project/src/grasp/score.py#L161) implements $+\frac{1}{\tau}\ln \sum w_i e^{+\tau \cdot \text{Risk}_i}$.
* **One-line implication:** Documentation exhibits multiple major factual discrepancies regarding split discipline, runtime overhead, project status, and mathematical formulation.

---

## Audit Summary: Item Status & Action Plan

| Item | Topic | Status | Severity | Suggested Fix |
|:---:|---|:---:|:---:|---|
| **1** | Local Results Inventory | CONFIRMED | **HIGH** | Sync and store `.json` and `timing.json` evaluation outputs in `results/`. |
| **2** | Target Number Verification | CONFIRMED | **HIGH** | Save genuine output logs and eval JSONs from Colab into `results/`. |
| **3** | MUCH Data Availability | CONFIRMED | **MED** | Keep at least a representative verification subset in the repository. |
| **4** | Train/Test Split Logic | REFUTED | **CRITICAL** | Update `03_score_and_eval.py` to filter by dataset split (`annotation.split == 'train'`). |
| **5** | Claim Filtering Mechanism | CONFIRMED | **MED** | Add explicit `--model` and `--config` CLI flags to `03_score_and_eval.py`. |
| **6** | Intra-generation Ranking | CONFIRMED | **HIGH** | Evaluate cross-sample calibration instead of ranking within individual generations. |
| **7** | Claim-Length Confound | NOT VERIFIED | **MED** | Compute length correlation and report length-binned ROC-AUC. |
| **8** | Boundary Reset Overwrites | CONFIRMED | **LOW** | Handle potential overlapping chunk indices safely without in-place overwrites. |
| **9** | 1-Token Attention Shift | CONFIRMED | **HIGH** | Align row $P+i-1$ with token $i$, or document the lookahead choice explicitly. |
| **10** | Saliency Length Bias | CONFIRMED | **MED** | Divide $W$ by number of available downstream positions $(T - 1 - j)$. |
| **11** | Soft-min Formula Signs | CONFIRMED | **HIGH** | Correct the mathematical formula in `report.html` and documentation to soft-max over risk. |
| **12** | Test-Time Fusion Selection | CONFIRMED | **HIGH** | Freeze `base_agg = 'product'` on training data prior to running fusion. |
| **13** | Timing Metric Scope | CONFIRMED | **MED** | Document that `timing.json` measures Pass 2 inference loop wall-clock time only. |
| **14** | Unit Test Execution | PARTIAL FAIL | **MED** | Install `nltk` in environment dependencies to enable full evaluator mock testing. |
| **15** | English Score Reproduction | NOT VERIFIED | **HIGH** | Export evaluation JSONs from Colab so local runs can be validated. |
| **16** | Cross-Lingual CCP Baseline | REFUTED | **CRITICAL** | Evaluate CCP on the matching single-model subsets before claiming cross-lingual wins. |
| **17** | Bootstrap Resampling CIs | NOT VERIFIED | **MED** | Export per-generation prediction tables and run bootstrap confidence intervals. |
| **18** | Repo Hygiene & Duplicates | CONFIRMED | **LOW** | Remove `Claude outputs/` and add `__pycache__/` to `.gitignore`. |
| **19** | Documentation Contradictions | CONFIRMED | **HIGH** | Reconcile README status, overhead numbers (6.4%), and train/test statements. |

---

## Numbers Table: Reported vs. Audited Status

| Metric / Parameter | Value in User Report | Audited Reality | Discrepancy / Audit Finding |
|---|:---:|:---:|---|
| **English Native ROC-AUC** | 0.7087 | 0.7087 (Colab log) | Not found in local repo; only in `report.html` text. |
| **English Max-Agg ROC-AUC** | 0.7224 | 0.7224 (Colab log) | Not found in local repo; only in `report.html` text. |
| **English GRASP-fuse ROC-AUC** | **0.8343** | **0.8343** (Colab log) | Aggregator selected on test set (`base_agg`); not in local JSON. |
| **English Aligned CCP ROC-AUC**| 0.8251 | 0.8251 (Colab log) | Scored on Llama-3.2-3B subset ($n=727$), not pooled MUCH ($n=4,204$). |
| **German GRASP ROC-AUC** | **0.7805** | 0.7805 (Colab log) | Scored on 719 claims. Compared against pooled CCP (0.7799), NOT aligned CCP. |
| **French GRASP ROC-AUC** | 0.7469 | 0.7469 (Colab log) | Scored on 858 claims. Compared against pooled CCP (0.7594). |
| **Spanish GRASP ROC-AUC** | 0.6970 | 0.6970 (Colab log) | Scored on 873 claims. Compared against pooled CCP (0.7190). |
| **Extraction Wall-Clock** | 60.78 s (586 gens) | 60.78 s (Colab log) | Valid for Pass 2 loop; excludes calibration and model loading. |
| **Latency per Generation** | 0.1037 s / gen | 0.1037 s / gen | Valid on Tesla T4 GPU for Pass 2. |
| **Runtime Overhead vs Gen** | ≈ 1–3% (Doc claim) | **≈ 6.4%** (Measured) | 0.1037s / 1.61s = 6.4%, double-to-triple the claimed 1–3%. |
| **Scored Claims (English)** | 727 | 727 | Filtered implicitly by cached config files, not split logic. |
| **Scored Annotations (English)**| 152 | 152 | Filtered implicitly by cached config files, not split logic. |
| **Split Partitioning** | "Train split only" | **None** | `03_score_and_eval.py` scores all cached annotations identically. |
| **Softmin Formulation** | $-\frac{1}{\tau}\ln \sum w_i e^{-\tau s_i}$ | $+\frac{1}{\tau}\ln \sum w_i e^{+\tau s_i}$ | Sign inversion between `report.html` and `score.py`. |
