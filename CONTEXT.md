# CONTEXT.md

### 1. Project Overview
* **Objective:** Amazon ML Challenge 2026 — Entity Resolution.
* **Input files:** `dataset/train/train_source1.tsv`, `dataset/train/train_source2.tsv`, `dataset/train/train_source3.tsv` (and corresponding test splits).
* **Expected output files:** `candidate_pairs.tsv` and `matching_results.tsv` (bundled into `antigravity_team_submission.zip`).
* **Evaluation metric:** Macro F0.5.

### 2. Current Repository Structure
```text
D:\AMAZON_ML
├── artifacts/
│   ├── blocked/
│   ├── cleaned/
│   ├── features/
│   ├── models/
│   ├── val_split/
│   ├── manifest.json
│   └── progress.json
├── dataset/
│   ├── test/ (source1, source2, source3)
│   └── train/ (ground_truth, source1, source2, source3)
├── info/
│   └── data/student_resource/
├── notebooks/
│   ├── kaggle_runner.ipynb
│   └── kernel-metadata.json
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── scripts/
│   ├── deploy.py
│   └── package_submission.py
├── src/
│   ├── __init__.py
│   ├── blocking.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── normalizer.py
│   ├── run_pipeline.py
│   ├── state_manager.py
│   └── train.py
├── Makefile
├── requirements.txt
└── run.ps1
```

### 3. Pipeline Architecture
* **Flow:** Cleaning → Normalization → Country Partitioning → Blocking → Feature Engineering → LightGBM → Inference → Validation → Kaggle Deployment.
* **Entry file:** Managed via PowerShell `run.ps1` wrapping `src/run_pipeline.py` and `scripts/deploy.py`.
* **Output artifacts:** Stored systematically under `artifacts/` (e.g., `artifacts/blocked/`, `artifacts/features/`).
* **Checkpoint behavior:** Managed by `src/state_manager.py`, persisting status to `artifacts/progress.json`.
* **Implementation status:** Full pipeline scaffolding is complete (verified from commits). Blocking logic is currently undergoing revisions to resolve scaling issues.

### 4. Blocking System (Very Detailed)
* **Layer 1:** Dynamic Country Partition Filtering.
* **Layer 2:** Rare-token inverted index (keeps tokens appearing in < 2% of documents, max 100 hits).
* **Layer 3:** Numeric anchor blocking (keeps numeric sequences >= 2 chars, max 50 hits).
* **Layer 4 (TF-IDF Retrieval):** Currently implemented as a **Global Hybrid Chunked Sparse Top-K Retrieval**. It pre-builds a sparse TF-IDF matrix for all targets, chunks S1 (2,000 rows) and Targets (200,000 rows), and computes dot products to extract Top-K candidates. 
* **Layer 5:** Candidate union, scoring, and deduplication (`composite_score = (2.0 * tfidf) + (1.2 * tokens) + (1.5 * numbers)`).
* **Parameters:** `TOP_K = 20`. `TFIDF_MIN_SIMILARITY = 0.35`. `TFIDF_NGRAM_RANGE = (3, 3)`. 
* **Memory Strategy:** Polars streaming for IO; chunked sparse matrix products for Layer 4 to avoid OOM.
* **Current limitations:** Pure Python `for` loops slicing the chunked matrices in Layer 4 are a severe computational bottleneck, causing runtimes exceeding several hours.

### 5. Feature Engineering
* **Verified extracted from code:** `label` (ground truth matching).
* **Planned/Partially Implemented (Unverified locally due to Regex limits):** RapidFuzz ratio, token sort ratio, Jaro-Winkler, Jaccard similarity, address overlap, numeric agreement, same-country, legal-suffix.

### 6. Model
(Verified strictly from `src/config.py`)
* **learning rate:** 0.05
* **max depth:** 8
* **num leaves:** 63
* **boosting rounds (`n_estimators`):** 1500
* **early stopping:** 50 rounds
* **threads (`n_jobs`):** -1 (All cores)
* **device behavior:** Not explicitly bound to GPU in `config.py` (defaults to CPU unless Kaggle runtime injects GPU LightGBM).

### 7. Kaggle Integration
* **Kaggle username:** `ashash77`
* **Kernel slug:** `amazon-ml-2026-gbdt`
* **Dataset slug:** `amazon-ml-2026-dataset` (Confirmed mapped in `kernel-metadata.json`).
* **Private/Public status:** `is_private = true`.
* **Deployment flow:** `scripts/deploy.py --push` pushes the Kaggle kernel containing the notebook.
* **Download flow:** `scripts/deploy.py --download-only` retrieves outputs from Kaggle.
* **Dataset Mount:** The raw TSV data is explicitly mounted at `/kaggle/input/amazon-ml-2026-dataset` rather than uploaded inside the notebook codebase.

### 8. Commands Reference
* `.\run.ps1 setup`: Installs dependencies.
* `.\run.ps1 clean`: Purges `artifacts/` and `output/`.
* `.\run.ps1 block`: Runs Data Loading, Normalization, and Blocking.
* `.\run.ps1 train`: Runs Feature Engineering and LightGBM model training.
* `.\run.ps1 infer`: Runs inference on the test dataset.
* `.\run.ps1 validate`: Validates submission format.
* `.\run.ps1 deploy`: Triggers the `scripts/deploy.py` workflow.
* `python scripts/deploy.py --dry-run`: Checks git status and Kaggle authentication without pushing.
* `python scripts/deploy.py --push`: Pushes kernel to Kaggle.
* `python scripts/deploy.py --download-only`: Downloads results from a finished Kaggle run.

### 9. Execution Evidence
* **Total Runtime:** > 8.5 Hours (Terminated).
* **Peak RAM:** 7,330.6 MB (from background task logs before termination).
* **India blocking metrics:** Layer 4 took ~45 minutes (2,718s). Layer 5 bugged and reported 20 total candidates.
* **US blocking metrics:** *Unverified (Task cancelled before completion).*
* **Recall measurements:** *Unverified (Dropped to 30.57% in a previous iteration, but current hybrid approach benchmark was cancelled before generating metrics).*

### 10. Git Status
* **Current branch:** `main` (Up to date with origin).
* **Uncommitted changes:**
  * `M OPERATOR_GUIDE.md`
  * `M notebooks/kaggle_runner.ipynb`
  * `M notebooks/kernel-metadata.json`
  * `M scripts/deploy.py`
  * `M src/blocking.py`
  * `?? antigravity_team_submission.zip`
* **Latest commits:**
  * `3bf0caa` feat: update kaggle_runner.ipynb with runtime GPU audit and output path verification
  * `58af850` feat: add CLI flags (--push, --download-only, --dry-run) to deploy.py
  * `01aae62` Documentation: Complete Operator Guide with step-by-step execution instructions and command reference
  * `2abbfc9` Commit 8: Full inference pipeline, macro F0.5 evaluator, submission generator, validator check, and zip packager

### 11. Current Problems
* **Layer 4 Bottleneck:** The hybrid global TF-IDF retrieval successfully bounds memory but is far too slow (taking 8+ hours) due to slicing chunked sparse matrices in pure Python `for` loops.
* **Layer 5 Bug:** Indentation was broken during a previous patch to export layer-specific hits, causing it to discard almost all candidates (resulting in only 20 candidates for the entire India partition).
* **Blocking recall:** Needs to be re-measured once Layer 4 and 5 are fixed.
* **Training intentionally not started.** `progress.json` shows `stage_5_training` as `PENDING`.

### 12. Next Recommended Actions
* **Immediate next step:** Fix the indentation in `src/blocking.py` Layer 5. Replace the pure Python loop in Layer 4 with vectorized operations (e.g., using `scipy.sparse` vectorized thresholding) to fix the runtime bottleneck.
* **Before Kaggle deployment:** Run `.\run.ps1 block` over the validation set and verify the overall macro recall is > 90%.
* **Before training:** Run a small smoke test for feature engineering to verify parquet sizes and `pl.concat` memory usage.
* **Future improvements:** Migrate TF-IDF sparse dot products to CuPy for GPU acceleration.

---

### Audit Confidence

| Section | Status | Source |
| :--- | :--- | :--- |
| Repository Structure | Verified | File System / Script |
| Pipeline Architecture | Verified | Codebase / Git |
| Blocking System | Verified | `src/blocking.py` |
| Feature Engineering | Partially Verified | `src/feature_engineering.py` (`label` only extracted cleanly via regex) |
| Model Configuration | Verified | `src/config.py` |
| Kaggle Integration | Verified | `kernel-metadata.json`, `deploy.py` |
| Commands Reference | Verified | `run.ps1`, `deploy.py` |
| Execution Metrics | Unverified / Partial | Logs (Benchmark manually cancelled) |
| Git Status | Verified | `git status`, `git log` |
| Current Problems | Verified | Process Logs / Code inspection |
