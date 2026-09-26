import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple

import numpy as np
import polars as pl
import psutil
from rapidfuzz import fuzz, distance

from src.config import config
from src.state_manager import StateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FeatureEngineering")

FEATURE_COLUMNS = [
    "name_rapidfuzz_ratio",
    "name_token_sort_ratio",
    "name_jaro_winkler",
    "name_jaccard_similarity",
    "name_len_ratio",
    "address_token_overlap",
    "address_numeric_agreement",
    "same_country",
    "legal_suffix_match",
    "missing_address_flag",
    "heuristic_score",
    "layers_matched"
]


class FeatureExtractor:
    """
    Vectorized C++ Accelerated Pairwise Feature Engineering Engine.
    Generates exact similarity metrics on candidate pairs without extra external dependencies.
    """
    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    def _get_memory_mb(self) -> Tuple[float, float]:
        rss = self.process.memory_info().rss / (1024 * 1024)
        if rss > self.peak_memory_mb:
            self.peak_memory_mb = rss
        return rss, self.peak_memory_mb

    def _atomic_write_parquet(self, df: pl.DataFrame, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f"{out_path.name}.tmp")
        df.write_parquet(tmp_path, compression="snappy")
        os.replace(tmp_path, out_path)

    # --------------------------------------------------------------------------
    # RapidFuzz & Numeric Vectorized Metric Computation
    # --------------------------------------------------------------------------
    def compute_pair_features(
        self,
        s1_names: List[str],
        s1_addrs: List[str],
        s1_suffixes: List[str],
        cand_names: List[str],
        cand_addrs: List[str],
        cand_suffixes: List[str],
        heuristic_scores: List[float],
        layers_matched: List[int]
    ) -> Dict[str, List[Any]]:
        """
        Computes 12 pairwise feature vectors for candidate pairs.
        """
        n_pairs = len(s1_names)
        
        ratio_feat = np.zeros(n_pairs, dtype=np.float32)
        token_sort_feat = np.zeros(n_pairs, dtype=np.float32)
        jaro_feat = np.zeros(n_pairs, dtype=np.float32)
        jaccard_feat = np.zeros(n_pairs, dtype=np.float32)
        name_len_feat = np.zeros(n_pairs, dtype=np.float32)
        
        addr_overlap_feat = np.zeros(n_pairs, dtype=np.float32)
        num_agree_feat = np.zeros(n_pairs, dtype=np.float32)
        suffix_match_feat = np.zeros(n_pairs, dtype=np.float32)
        missing_addr_feat = np.zeros(n_pairs, dtype=np.float32)

        for i in range(n_pairs):
            n1 = s1_names[i] or ""
            n2 = cand_names[i] or ""
            a1 = s1_addrs[i] or ""
            a2 = cand_addrs[i] or ""
            suf1 = s1_suffixes[i] or "none"
            suf2 = cand_suffixes[i] or "none"

            # 1. RapidFuzz Name Similarity
            ratio_feat[i] = fuzz.ratio(n1, n2) / 100.0
            token_sort_feat[i] = fuzz.token_sort_ratio(n1, n2) / 100.0
            jaro_feat[i] = distance.JaroWinkler.similarity(n1, n2)

            # 2. Token Jaccard Similarity
            toks1 = set(n1.split())
            toks2 = set(n2.split())
            union_len = len(toks1 | toks2)
            jaccard_feat[i] = (len(toks1 & toks2) / union_len) if union_len > 0 else 0.0

            # 3. Name Length Ratio
            l1, l2 = len(n1), len(n2)
            name_len_feat[i] = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

            # 4. Address Features
            has_a1 = len(a1.strip()) > 0
            has_a2 = len(a2.strip()) > 0

            if not has_a1 or not has_a2:
                missing_addr_feat[i] = 1.0
                addr_overlap_feat[i] = 0.0
                num_agree_feat[i] = 0.0
            else:
                missing_addr_feat[i] = 0.0
                atok1 = set(a1.split())
                atok2 = set(a2.split())
                a_union = len(atok1 | atok2)
                addr_overlap_feat[i] = (len(atok1 & atok2) / a_union) if a_union > 0 else 0.0

                # Numeric Token Agreement
                nums1 = set(re.findall(r'\b\d+\b', a1))
                nums2 = set(re.findall(r'\b\d+\b', a2))
                num_union = len(nums1 | nums2)
                num_agree_feat[i] = (len(nums1 & nums2) / num_union) if num_union > 0 else 0.0

            # 5. Legal Suffix Match
            if suf1 != "none" and suf2 != "none":
                suffix_match_feat[i] = 1.0 if suf1 == suf2 else 0.0
            elif suf1 == "none" and suf2 == "none":
                suffix_match_feat[i] = 0.5
            else:
                suffix_match_feat[i] = 0.25

        return {
            "name_rapidfuzz_ratio": ratio_feat,
            "name_token_sort_ratio": token_sort_feat,
            "name_jaro_winkler": jaro_feat,
            "name_jaccard_similarity": jaccard_feat,
            "name_len_ratio": name_len_feat,
            "address_token_overlap": addr_overlap_feat,
            "address_numeric_agreement": num_agree_feat,
            "same_country": np.ones(n_pairs, dtype=np.float32),  # Country partitioned
            "legal_suffix_match": suffix_match_feat,
            "missing_address_flag": missing_addr_feat,
            "heuristic_score": np.array(heuristic_scores, dtype=np.float32),
            "layers_matched": np.array(layers_matched, dtype=np.int8)
        }

    # --------------------------------------------------------------------------
    # Pipeline Runner for Feature Extraction
    # --------------------------------------------------------------------------
    def extract_features_for_mode(self, mode: str = "train") -> Dict[str, Any]:
        """
        Loads blocked candidate pairs, attaches cleaned text attributes,
        computes feature matrix, attaches ground-truth binary labels (if train/val),
        and saves artifacts/features/{mode}_features.parquet.
        """
        stage_name = f"stage_4_feature_engineering_{mode}"
        self.state_manager.mark_in_progress(stage_name)
        t_start = time.perf_counter()

        cand_path = config.BLOCKED_DIR / f"{mode}_candidates.parquet"
        if not cand_path.exists():
            raise FileNotFoundError(f"Candidate pairs parquet not found: {cand_path}")

        cand_df = pl.read_parquet(cand_path)
        n_pairs = len(cand_df)
        logger.info("[%s Feature Extraction] Processing %d candidate pairs...", mode.upper(), n_pairs)

        if n_pairs == 0:
            empty_dict = {"source1_entity_id": [], "candidate_entity_id": []}
            for col in FEATURE_COLUMNS:
                empty_dict[col] = []
            if mode in ["train", "val"]:
                empty_dict["label"] = []
            feat_df = pl.DataFrame(empty_dict)
            out_parquet = config.FEATURES_DIR / f"{mode}_features.parquet"
            self._atomic_write_parquet(feat_df, out_parquet)
            return {"mode": mode, "total_pairs": 0, "time_sec": 0.0}

        # Build entity lookup dictionary across all cleaned partitions
        s1_files = list(config.VAL_SPLIT_DIR.glob("val_s1_part_*.parquet")) if mode == "val" else list(config.CLEANED_DIR.rglob("*_source1_part_*.parquet"))
        s2_s3_files = list(config.CLEANED_DIR.rglob("*_source2_part_*.parquet")) + list(config.CLEANED_DIR.rglob("*_source3_part_*.parquet"))
        
        if mode == "test":
            s1_files = list(config.CLEANED_DIR.rglob("test_source1_part_*.parquet"))
            s2_s3_files = list(config.CLEANED_DIR.rglob("test_source2_part_*.parquet")) + list(config.CLEANED_DIR.rglob("test_source3_part_*.parquet"))

        logger.info("[%s] Loading cleaned entity lookup tables...", mode.upper())
        s1_entities_df = pl.concat([pl.read_parquet(f) for f in s1_files]) if s1_files else pl.DataFrame()
        target_entities_df = pl.concat([pl.read_parquet(f) for f in s2_s3_files]) if s2_s3_files else pl.DataFrame()

        # Build fast index dicts
        s1_dict = {}
        for r in s1_entities_df.iter_rows(named=True):
            s1_dict[r["entity_id"]] = (
                r.get("name_clean", r.get("business_name", "")),
                r.get("address_clean", r.get("business_address", "")),
                r.get("legal_suffix", "none")
            )

        target_dict = {}
        for r in target_entities_df.iter_rows(named=True):
            target_dict[r["entity_id"]] = (
                r.get("name_clean", r.get("business_name", "")),
                r.get("address_clean", r.get("business_address", "")),
                r.get("legal_suffix", "none")
            )

        del s1_entities_df, target_entities_df
        gc.collect()

        # Gather arrays for feature computation
        s1_ids = cand_df["source1_entity_id"].to_list()
        cand_ids = cand_df["candidate_entity_id"].to_list()
        scores = cand_df["heuristic_score"].to_list()
        layers = cand_df["layers_matched"].to_list()

        s1_names, s1_addrs, s1_sufs = [], [], []
        cand_names, cand_addrs, cand_sufs = [], [], []

        for s1_id, c_id in zip(s1_ids, cand_ids):
            n1, a1, suf1 = s1_dict.get(s1_id, ("", "", "none"))
            n2, a2, suf2 = target_dict.get(c_id, ("", "", "none"))
            s1_names.append(n1)
            s1_addrs.append(a1)
            s1_sufs.append(suf1)
            cand_names.append(n2)
            cand_addrs.append(a2)
            cand_sufs.append(suf2)

        del s1_dict, target_dict
        gc.collect()

        # Compute pairwise features in memory-safe batches of 50,000
        batch_size = 50_000
        feature_dict = {col: [] for col in FEATURE_COLUMNS}

        for i in range(0, n_pairs, batch_size):
            b_end = min(i + batch_size, n_pairs)
            sub_feats = self.compute_pair_features(
                s1_names[i:b_end], s1_addrs[i:b_end], s1_sufs[i:b_end],
                cand_names[i:b_end], cand_addrs[i:b_end], cand_sufs[i:b_end],
                scores[i:b_end], layers[i:b_end]
            )
            for col in FEATURE_COLUMNS:
                feature_dict[col].append(sub_feats[col])

        # Concatenate array batches into final Polars DataFrame
        df_cols = {
            "source1_entity_id": s1_ids,
            "candidate_entity_id": cand_ids
        }
        for col in FEATURE_COLUMNS:
            df_cols[col] = np.concatenate(feature_dict[col])

        feat_df = pl.DataFrame(df_cols)

        # Attach binary ground-truth target labels for training/validation
        if mode in ["train", "val"]:
            gt_files = list(config.VAL_SPLIT_DIR.glob("val_ground_truth_part_*.parquet")) if mode == "val" else list(config.CLEANED_DIR.glob("train_ground_truth_part_*.parquet"))
            gt_set = set()
            if gt_files:
                gt_df = pl.concat([pl.read_parquet(f) for f in gt_files])
                for s1, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
                    if mids and mids.strip():
                        for m in mids.split(","):
                            gt_set.add((s1, m))

            labels = np.array([1 if (s1, c) in gt_set else 0 for s1, c in zip(s1_ids, cand_ids)], dtype=np.int8)
            feat_df = feat_df.with_columns(pl.Series("label", labels, dtype=pl.Int8))
            pos_count = int(np.sum(labels))
            logger.info("[%s Features] Attached binary labels: %d positive matches out of %d candidate pairs", mode.upper(), pos_count, n_pairs)

        # Save feature parquet artifact
        out_parquet = config.FEATURES_DIR / f"{mode}_features.parquet"
        self._atomic_write_parquet(feat_df, out_parquet)

        elapsed = time.perf_counter() - t_start
        rss_mb, peak_mb = self._get_memory_mb()

        self.state_manager.record_artifact(
            f"{mode}_features",
            out_parquet,
            len(feat_df),
            meta={"features": FEATURE_COLUMNS, "time_sec": elapsed, "peak_ram_mb": peak_mb}
        )
        self.state_manager.mark_completed(stage_name, meta={"total_pairs": len(feat_df), "time_sec": elapsed})

        logger.info(
            "[%s Feature Extraction Complete] Total Pairs: %d | Time: %.2fs (%.0f pairs/s) | RAM: %.1f MB",
            mode.upper(), len(feat_df), elapsed, len(feat_df) / elapsed if elapsed > 0 else 0, rss_mb
        )

        return {
            "mode": mode,
            "total_pairs": len(feat_df),
            "throughput_pairs_per_sec": len(feat_df) / elapsed if elapsed > 0 else 0,
            "time_sec": elapsed,
            "peak_ram_mb": peak_mb,
            "feature_columns": FEATURE_COLUMNS
        }
