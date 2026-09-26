# Amazon ML Challenge 2026 — Operator Guide
**Target Metric:** Macro-Averaged $F_{0.5}$ (Precision-Weighted)  
**Architecture Status:** Production-Ready & Verified  
**Execution Modes:** Local Machine | Kaggle CLI Remote GPU/High-RAM

---

## 1. Project Overview

### Directory Tree
```text
D:\AMAZON_ML/
├── info/                                          # Immutable contest specs, PDFs, and raw TSVs
├── dataset/                                       # Zero-copy directory junction to raw dataset
├── src/                                           # Core pipeline source code
│   ├── config.py                                  # Central frozen configuration & hyperparameters
│   ├── state_manager.py                           # Progress tracking (progress.json & manifest.json)
│   ├── data_loader.py                             # Polars streaming 100k chunk loader
│   ├── normalizer.py                              # Multilingual Unicode & legal suffix normalizer
│   ├── blocking.py                                # 5-layer candidate retrieval engine
│   ├── feature_engineering.py                     # RapidFuzz C++ similarity extractor
│   ├── train.py                                   # LightGBM GBDT trainer with early stopping
│   ├── evaluate.py                                # Macro F0.5 evaluator & threshold optimizer
│   ├── inference.py                               # Test set prediction & validator integration
│   └── run_pipeline.py                            # Master CLI entry point
├── scripts/
│   ├── deploy.py                                  # Kaggle CLI remote synchronization
│   └── package_submission.py                      # Final ZIP submission packager
├── notebooks/
│   └── kaggle_runner.ipynb                        # Kaggle GPU notebook wrapper
├── artifacts/                                     # Intermediate checkpoints (cleaned, blocked, features, models)
├── output/                                        # Leaderboard deliverables (matching_results.tsv, candidate_pairs.tsv)
├── .gitignore
├── Makefile                                       # Standard Make automation runner
├── run.ps1                                        # Windows PowerShell runner script
├── requirements.txt                               # Pinned exact dependency versions
└── OPERATOR_GUIDE.md                              # This manual
```

---

## 2. Command Reference Manual

Run all commands using PowerShell (`.\run.ps1 <command>`) or Makefile (`make <command>`).

| Command | Python File Executed | Expected Runtime | Peak RAM | Generated / Updated Files | Checkpoints Updated |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`.\run.ps1 setup`** | `pip install -r requirements.txt` | ~30s | ~100 MB | `.venv/`, `requirements.txt` | N/A |
| **`.\run.ps1 block`** | `src/run_pipeline.py --stage block` | ~12–15m | ~495 MB | `artifacts/cleaned/{country}/*.parquet`<br>`artifacts/blocked/*.parquet`<br>`output/candidate_pairs.tsv` | `progress.json`<br>`manifest.json` |
| **`.\run.ps1 train`** | `src/run_pipeline.py --stage train` | ~8–12m | ~1.2 GB | `artifacts/features/*.parquet`<br>`artifacts/models/model.pkl`<br>`artifacts/models/optimal_threshold.json` | `progress.json`<br>`manifest.json` |
| **`.\run.ps1 infer`** | `src/run_pipeline.py --stage infer` | ~3–5m | ~800 MB | `output/matching_results.tsv`<br>`output/candidate_pairs.tsv` | `progress.json`<br>`manifest.json` |
| **`.\run.ps1 validate`** | `info/.../utils/validate_submission.py` | ~15s | ~250 MB | Validation PASS report on terminal | N/A |
| **`.\run.ps1 deploy`** | `scripts/deploy.py` | ~1m | ~150 MB | `notebooks/kernel-metadata.json`<br>Remote Kaggle GPU job | Kaggle CLI log |

---

## 3. Local Training Guide (Step-by-Step)

Follow these steps to execute full end-to-end training on your local machine:

1. **Activate Virtual Environment:**
   ```powershell
   & "D:\AMAZON_ML\.venv\Scripts\Activate.ps1"
   ```
2. **Execute Ingestion & Candidate Blocking:**
   ```powershell
   .\run.ps1 block
   ```
   * *What happens:* `data_loader.py` streams 24M raw records in 100k chunks, dynamically creating country partitions under `artifacts/cleaned/`. `blocking.py` generates `artifacts/blocked/{train,val,test}_candidates.parquet` and exports `output/candidate_pairs.tsv`.
3. **Execute Feature Engineering & LightGBM Training:**
   ```powershell
   .\run.ps1 train
   ```
   * *What happens:* `feature_engineering.py` extracts 12 RapidFuzz similarity metrics into `artifacts/features/`. `train.py` trains LightGBM GBDT with early stopping, saving `artifacts/models/model.pkl`. `evaluate.py` scans $\tau \in [0.50, 0.90]$ to optimize macro $F_{0.5}$ and saves `optimal_threshold.json`.
4. **Interruption Recovery:**
   * If power drops or a step is interrupted, simply re-run `.\run.ps1 block` or `.\run.ps1 train`. The `state_manager` checks `manifest.json` and skips all completed chunks instantly.

---

## 4. Kaggle Training Guide (Remote GPU/High-RAM)

If local compute is limited, execute remote GBDT training on Kaggle:

1. **Verify Credentials:**
   Ensure `%USERPROFILE%\.kaggle\kaggle.json` exists. Test authentication:
   ```powershell
   .\.venv\Scripts\kaggle.exe competitions list
   ```
2. **Deploy to Kaggle:**
   ```powershell
   .\run.ps1 deploy
   ```
   * `deploy.py` verifies Git status, generates `notebooks/kernel-metadata.json`, and triggers remote execution via Kaggle CLI.
3. **Download Trained Artifacts:**
   Once remote training finishes, `deploy.py` pulls `model.pkl`, `optimal_threshold.json`, and outputs directly into `artifacts/models/` and `output/`.

---

## 5. Final Submission Packaging Guide

1. **Generate Test Predictions:**
   ```powershell
   .\run.ps1 infer
   ```
2. **Run Official Pre-Flight Validator:**
   ```powershell
   .\run.ps1 validate
   ```
   * Must output: `PASS — no blocking issues found. Safe to submit.`
3. **Create Submission Package ZIP:**
   ```powershell
   & "D:\AMAZON_ML\.venv\Scripts\python.exe" scripts/package_submission.py
   ```
   * Generates `antigravity_team_submission.zip` containing:
     * `output/matching_results.tsv`
     * `output/candidate_pairs.tsv`
     * `code/business_entity_resolution/src/`
     * `code/business_entity_resolution/requirements.txt`
     * `Documentation_template.md`

---

## 6. Troubleshooting & Gotchas

* **Kaggle Auth Error:** Copy your downloaded `kaggle.json` to `C:\Users\Arsh\.kaggle\kaggle.json`.
* **Out of Memory (OOM):** `CHUNK_SIZE` in `src/config.py` defaults to `100_000`. If working under strict RAM constraints, lower `CHUNK_SIZE` to `50_000`.
* **Subsetting Warning:** Every matched ID in `matching_results.tsv` must exist in `candidate_pairs.tsv`. `inference.py` guarantees this by construction.
* **Corrupt Intermediate Checkpoints:** Run `.\run.ps1 clean` to purge `artifacts/` and restart cleanly.


## Kaggle Private Dataset

Ensure the Kaggle dataset shash77/amazon-ml-2026-dataset exists before deploying. The kernel will automatically mount this dataset at /kaggle/input/amazon-ml-2026-dataset.