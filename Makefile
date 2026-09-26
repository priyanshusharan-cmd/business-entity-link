# Amazon ML Challenge 2026 Makefile
PYTHON = .venv/Scripts/python.exe

.PHONY: setup clean block train infer deploy validate help

help:
	@echo "Amazon ML Challenge 2026 Pipeline Automation"
	@echo "  make setup    - Install dependencies and bootstrap environment"
	@echo "  make clean    - Remove intermediate artifacts and outputs"
	@echo "  make block    - Execute data loading, normalization, and 5-layer blocking"
	@echo "  make train    - Run feature engineering and LightGBM model training"
	@echo "  make infer    - Generate test predictions and candidate pairs"
	@echo "  make validate - Run official submission validator against generated TSVs"
	@echo "  make deploy   - Trigger remote Kaggle CLI deployment"

setup:
	$(PYTHON) -m pip install -r requirements.txt

clean:
	rmdir /S /Q artifacts\cleaned artifacts\blocked artifacts\features artifacts\models output

block:
	$(PYTHON) -m src.run_pipeline --stage block

train:
	$(PYTHON) -m src.run_pipeline --stage train

infer:
	$(PYTHON) -m src.run_pipeline --stage infer

validate:
	$(PYTHON) info/data/student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

deploy:
	$(PYTHON) scripts/deploy.py
