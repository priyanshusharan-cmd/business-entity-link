<#
.SYNOPSIS
  PowerShell pipeline runner for Amazon ML Challenge 2026.
.EXAMPLE
  .\run.ps1 setup
  .\run.ps1 block
  .\run.ps1 train
  .\run.ps1 infer
  .\run.ps1 validate
  .\run.ps1 deploy
  .\run.ps1 clean
#>

param (
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet("setup", "clean", "block", "train", "infer", "validate", "deploy", "help")]
    [string]$Command
)

$PYTHON = ".\.venv\Scripts\python.exe"

switch ($Command) {
    "setup" {
        Write-Host "[SETUP] Installing pinned requirements..." -ForegroundColor Cyan
        & $PYTHON -m pip install -r requirements.txt
    }
    "clean" {
        Write-Host "[CLEAN] Purging artifacts and outputs..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue artifacts\cleaned, artifacts\blocked, artifacts\features, artifacts\models, output
        Write-Host "[CLEAN] Complete." -ForegroundColor Green
    }
    "block" {
        Write-Host "[BLOCK] Running loading, normalization, and candidate blocking..." -ForegroundColor Cyan
        & $PYTHON -m src.run_pipeline --stage block
    }
    "train" {
        Write-Host "[TRAIN] Running feature extraction and model training..." -ForegroundColor Cyan
        & $PYTHON -m src.run_pipeline --stage train
    }
    "infer" {
        Write-Host "[INFER] Generating matching_results.tsv and candidate_pairs.tsv..." -ForegroundColor Cyan
        & $PYTHON -m src.run_pipeline --stage infer
    }
    "validate" {
        Write-Host "[VALIDATE] Running official submission validator..." -ForegroundColor Cyan
        & $PYTHON info\data\student_resource\utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
    }
    "deploy" {
        Write-Host "[DEPLOY] Launching Kaggle deployment..." -ForegroundColor Cyan
        & $PYTHON scripts\deploy.py
    }
    "help" {
        Write-Host "Usage: .\run.ps1 [setup | clean | block | train | infer | validate | deploy]"
    }
}
