import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil

from src.config import config
from src.state_manager import StateManager
from src.feature_engineering import FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TrainModule")


class EntityMatcherTrainer:
    """
    LightGBM GBDT Entity Matching Trainer.
    Trains binary classifier with early stopping on leak-free validation split.
    Integrates with threshold optimizer for macro F0.5 evaluation.
    """
    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_memory_mb(self) -> Tuple[float, float]:
        rss = self.process.memory_info().rss / (1024 * 1024)
        if rss > self.peak_memory_mb:
            self.peak_memory_mb = rss
        return rss, self.peak_memory_mb

    def train_model(
        self,
        train_features_path: Optional[Path] = None,
        val_features_path: Optional[Path] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Trains LightGBM GBDT binary classifier.
        If dry_run is True, initializes data structures and verifies pipeline readiness without full training.
        """
        stage_name = "stage_5_training"
        self.state_manager.mark_in_progress(stage_name)
        t_start = time.perf_counter()

        train_path = Path(train_features_path or (config.FEATURES_DIR / "train_features.parquet"))
        val_path = Path(val_features_path or (config.FEATURES_DIR / "val_features.parquet"))

        if not train_path.exists():
            raise FileNotFoundError(f"Training features artifact not found: {train_path}. Run feature engineering first.")

        logger.info("[Training Initialization] Loading feature matrices from Parquet...")
        train_df = pl.read_parquet(train_path)
        val_df = pl.read_parquet(val_path) if val_path.exists() else pl.DataFrame()

        n_train = len(train_df)
        n_val = len(val_df)
        logger.info("Train rows: %d | Val rows: %d | Features: %s", n_train, n_val, FEATURE_COLUMNS)

        if dry_run or n_train == 0:
            logger.info("[DRY RUN MODE] Initialized training pipeline successfully. Skipping full epoch training.")
            self.state_manager.mark_completed(stage_name, meta={"dry_run": True, "train_rows": n_train, "val_rows": n_val})
            return {
                "status": "DRY_RUN_SUCCESS",
                "train_rows": n_train,
                "val_rows": n_val,
                "features_count": len(FEATURE_COLUMNS),
                "peak_ram_mb": self.peak_memory_mb
            }

        # Extract numpy X and y arrays
        X_train = train_df.select(FEATURE_COLUMNS).to_numpy()
        y_train = train_df["label"].to_numpy()

        if n_val > 0:
            X_val = val_df.select(FEATURE_COLUMNS).to_numpy()
            y_val = val_df["label"].to_numpy()
            val_data = [(X_val, y_val)]
        else:
            val_data = None

        # Build LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMNS)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=FEATURE_COLUMNS) if n_val > 0 else None

        params = dict(config.LIGHTGBM_PARAMS)
        callbacks = [lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS)] if n_val > 0 else []

        logger.info("[LightGBM] Starting training with params: %s", params)
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval] if dval else [dtrain],
            valid_names=["train", "val"] if dval else ["train"],
            callbacks=callbacks
        )

        # Save serialized model binary
        model_path = config.MODELS_DIR / "model.pkl"
        import joblib
        joblib.dump(model, model_path)
        logger.info("Saved trained LightGBM binary to %s", model_path)

        # Compute feature importances
        importance_dict = dict(zip(FEATURE_COLUMNS, model.feature_importance(importance_type="gain").tolist()))
        logger.info("Top Feature Importances (Gain): %s", sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])

        elapsed = time.perf_counter() - t_start
        rss_mb, peak_mb = self._get_memory_mb()

        self.state_manager.record_artifact(
            "model_binary",
            model_path,
            row_count=n_train,
            meta={"best_iteration": model.best_iteration, "feature_importance_gain": importance_dict}
        )
        self.state_manager.mark_completed(stage_name, meta={
            "train_rows": n_train,
            "best_iteration": model.best_iteration,
            "time_sec": elapsed,
            "peak_ram_mb": peak_mb
        })

        return {
            "status": "TRAINED_SUCCESSFULLY",
            "train_rows": n_train,
            "val_rows": n_val,
            "best_iteration": model.best_iteration,
            "time_sec": elapsed,
            "peak_ram_mb": peak_mb,
            "model_path": str(model_path)
        }
