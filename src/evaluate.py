import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple, List

import numpy as np
import polars as pl

from src.config import config
from src.state_manager import StateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Evaluator")


def compute_macro_f05(
    ground_truth_dict: Dict[str, Set[str]],
    predictions_dict: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes official competition Macro F0.5 Score across all Source 1 entities.
    Precision-weighted (beta = 0.5): false merges penalized 2x over misses.
    Singletons: empty prediction = 1.0; false merge = 0.0.
    """
    total_f05 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n = len(ground_truth_dict)

    if n == 0:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    for s1_id, true_matches in ground_truth_dict.items():
        pred_matches = predictions_dict.get(s1_id, set())

        # Singleton handling (official rule)
        if len(true_matches) == 0:
            if len(pred_matches) == 0:
                total_f05 += 1.0
                total_precision += 1.0
                total_recall += 1.0
            else:
                total_f05 += 0.0  # False merge on singleton drops score to 0
                total_precision += 0.0
                total_recall += 1.0
            continue

        if len(pred_matches) == 0:
            total_f05 += 0.0
            total_precision += 1.0
            total_recall += 0.0
            continue

        tp = len(true_matches & pred_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)

        if tp == 0:
            total_f05 += 0.0
            total_precision += 0.0
            total_recall += 0.0
            continue

        p = tp / (tp + fp)
        r = tp / (tp + fn)

        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
        total_f05 += f05
        total_precision += p
        total_recall += r

    return {
        "macro_f05": total_f05 / n,
        "macro_precision": total_precision / n,
        "macro_recall": total_recall / n
    }


class ThresholdOptimizer:
    """
    1D Grid Search Optimizer for Precision-Weighted Threshold (tau).
    """
    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)

    def optimize_threshold(
        self,
        val_probs: np.ndarray,
        val_pairs_df: pl.DataFrame,
        gt_parquet_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Scans thresholds tau in [0.50, 0.90] to find the threshold maximizing macro F0.5.
        """
        logger.info("[Threshold Optimizer] Starting 1D grid search over tau in [%.2f, %.2f]...", config.THRESHOLD_SCAN_RANGE[0], config.THRESHOLD_SCAN_RANGE[1])
        
        # Load validation ground truth
        gt_dir = gt_parquet_dir or config.VAL_SPLIT_DIR
        gt_files = list(gt_dir.glob("val_ground_truth_part_*.parquet"))
        
        gt_dict = {}
        if gt_files:
            gt_df = pl.concat([pl.read_parquet(f) for f in gt_files])
            for s1, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
                gt_dict[s1] = set(mids.split(",")) if (mids and mids.strip()) else set()

        min_t, max_t, step_t = config.THRESHOLD_SCAN_RANGE
        best_tau = config.DEFAULT_THRESHOLD
        best_metrics = {"macro_f05": -1.0}

        s1_ids = val_pairs_df["source1_entity_id"].to_list()
        c_ids = val_pairs_df["candidate_entity_id"].to_list()

        for tau in np.arange(min_t, max_t + step_t, step_t):
            tau = float(np.round(tau, 2))
            
            # Predict pairs exceeding tau
            pred_dict = defaultdict(set)
            for i, prob in enumerate(val_probs):
                if prob >= tau:
                    pred_dict[s1_ids[i]].add(c_ids[i])

            metrics = compute_macro_f05(gt_dict, pred_dict)
            logger.info("  tau = %.2f -> Macro F0.5: %.4f | Precision: %.4f | Recall: %.4f", tau, metrics["macro_f05"], metrics["macro_precision"], metrics["macro_recall"])

            if metrics["macro_f05"] > best_metrics["macro_f05"]:
                best_metrics = metrics
                best_tau = tau

        logger.info("[Threshold Search Complete] Optimal tau = %.2f -> Macro F0.5 = %.4f", best_tau, best_metrics["macro_f05"])

        out_json = config.MODELS_DIR / "optimal_threshold.json"
        res_data = {"optimal_threshold": best_tau, "validation_metrics": best_metrics}
        
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res_data, f, indent=2)
            
        self.state_manager.record_artifact("optimal_threshold", out_json, row_count=1, meta=res_data)
        return res_data
