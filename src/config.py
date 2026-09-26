from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, List

# ==============================================================================
# AMAZON ML CHALLENGE 2026 — CENTRAL FROZEN CONFIGURATION
# All tunable parameters are centralized here. Never edit parameters in other files.
# ==============================================================================

@dataclass
class PipelineConfig:
    # --- Paths ---
    BASE_DIR: Path = Path(r'D:\AMAZON_ML')
    DATASET_DIR: Path = BASE_DIR / 'dataset'
    ARTIFACTS_DIR: Path = BASE_DIR / 'artifacts'
    OUTPUT_DIR: Path = BASE_DIR / 'output'
    CLEANED_DIR: Path = ARTIFACTS_DIR / 'cleaned'
    VAL_SPLIT_DIR: Path = ARTIFACTS_DIR / 'val_split'
    BLOCKED_DIR: Path = ARTIFACTS_DIR / 'blocked'
    FEATURES_DIR: Path = ARTIFACTS_DIR / 'features'
    MODELS_DIR: Path = ARTIFACTS_DIR / 'models'
    PROGRESS_FILE: Path = ARTIFACTS_DIR / 'progress.json'
    MANIFEST_FILE: Path = ARTIFACTS_DIR / 'manifest.json'
    
    # Official Validator Path
    VALIDATOR_SCRIPT: Path = BASE_DIR / 'info' / 'data' / 'student_resource' / 'utils' / 'validate_submission.py'
    
    # --- Core Tunables ---
    RANDOM_SEED: int = 42
    CHUNK_SIZE: int = 100_000               # Polars streaming chunk size (records per chunk)
    VALIDATION_SPLIT: int = 100_000         # Hold-out validation S1 entities (60k US, 40k India)
    
    # --- Stage 3: Blocking Tunables ---
    TOP_K: int = 20                         # Number of candidate matches to retain per Source 1 entity
    TFIDF_NGRAM_RANGE: Tuple[int, int] = (3, 3) # Character n-grams for typo-resilient similarity
    TFIDF_MIN_SIMILARITY: float = 0.35      # Cosine similarity cutoff for sparse candidate retrieval
    TFIDF_MAX_FEATURES: int = 50_000        # Vocabulary ceiling for sparse matrix
    MAX_TOKEN_DOC_FREQ: float = 0.02        # Ignore tokens appearing in > 2% of records (stop words)
    MIN_TOKEN_LEN: int = 3                  # Ignore single/double character noise tokens
    
    # --- Stage 4 & 5: Model & Training Tunables ---
    EARLY_STOPPING_ROUNDS: int = 50
    DEFAULT_THRESHOLD: float = 0.72         # Conservative precision-weighted threshold for F0.5
    THRESHOLD_SCAN_RANGE: Tuple[float, float, float] = (0.50, 0.90, 0.01) # (min, max, step)
    
    LIGHTGBM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'n_estimators': 1500,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })

    # --- Kaggle CLI Integration Settings ---
    KAGGLE_USERNAME: str = 'ashash77'
    KAGGLE_KERNEL_SLUG: str = 'amazon-ml-2026-gbdt'
    KAGGLE_DATASET_SLUG: str = 'amazon-ml-2026-features'

config = PipelineConfig()
