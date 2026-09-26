import gc
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple

import numpy as np
import polars as pl
import psutil
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import config
from src.state_manager import StateManager
from src.normalizer import EntityNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BlockingEngine")

GLOBAL_STOP_WORDS = {
    'and', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an', '&',
    'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'sarl', 'sas', 'sci', 'sa', 'eurl',
    'services', 'enterprises', 'solutions', 'traders', 'trading', 'group',
    'center', 'centre', 'industries', 'international', 'india', 'us', 'france'
}

class MultiLayerBlocker:
    def __init__(self, state_manager: Optional[StateManager] = None, top_k: int = config.TOP_K):
        self.top_k = top_k
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.normalizer = EntityNormalizer()
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        config.BLOCKED_DIR.mkdir(parents=True, exist_ok=True)

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

    def block_country_partition(
        self,
        country: str,
        s1_df: pl.DataFrame,
        target_df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        t0 = time.perf_counter()
        
        if "name_clean" not in s1_df.columns:
            s1_df = self.normalizer.normalize_polars_df(s1_df)
        if "name_clean" not in target_df.columns:
            target_df = self.normalizer.normalize_polars_df(target_df)

        n_s1 = len(s1_df)
        n_target = len(target_df)
        
        if n_s1 == 0 or n_target == 0:
            empty_df = pl.DataFrame({
                "source1_entity_id": pl.Series([], dtype=pl.String),
                "candidate_entity_id": pl.Series([], dtype=pl.String),
                "heuristic_score": pl.Series([], dtype=pl.Float32),
                "layers_matched": pl.Series([], dtype=pl.Int8)
            })
            return empty_df, {"candidates_generated": 0, "time_sec": 0.0}

        logger.info("[%s] Blocking %d S1 entities against %d target (S2/S3) records...", country, n_s1, n_target)

        s1_ids = s1_df["entity_id"].to_list()
        s1_names = s1_df["name_clean"].to_list()
        s1_addrs = s1_df["address_clean"].to_list()
        
        target_ids = target_df["entity_id"].to_list()
        target_names = target_df["name_clean"].to_list()
        target_addrs = target_df["address_clean"].to_list()

        candidates_map = defaultdict(lambda: defaultdict(lambda: {"tfidf": 0.0, "token_hit": 0, "num_hit": 0}))

        # Layer 2: Rare Token Index
        t_l2 = time.perf_counter()
        token_doc_freq = defaultdict(int)
        for name in target_names:
            if not name: continue
            for tok in set(name.split()):
                if len(tok) >= config.MIN_TOKEN_LEN and tok not in GLOBAL_STOP_WORDS:
                    token_doc_freq[tok] += 1

        max_freq = max(1, int(n_target * config.MAX_TOKEN_DOC_FREQ))
        valid_rare_tokens = {tok for tok, freq in token_doc_freq.items() if freq <= max_freq}

        target_token_index = defaultdict(list)
        for t_idx, name in enumerate(target_names):
            if not name: continue
            for tok in set(name.split()):
                if tok in valid_rare_tokens:
                    target_token_index[tok].append(t_idx)

        for s1_idx, name in enumerate(s1_names):
            if not name: continue
            for tok in set(name.split()):
                if tok in valid_rare_tokens:
                    for t_idx in target_token_index[tok][:100]:
                        candidates_map[s1_idx][t_idx]["token_hit"] += 1

        logger.info("[%s | Layer 2] Rare token index completed in %.2fs", country, time.perf_counter() - t_l2)

        # Layer 3: Numeric Anchor Blocking
        t_l3 = time.perf_counter()
        target_num_index = defaultdict(list)
        for t_idx, (name, addr) in enumerate(zip(target_names, target_addrs)):
            comb = f"{name} {addr}"
            nums = set(re.findall(r'\b\d+\b', comb))
            for num in nums:
                if len(num) >= 2:
                    target_num_index[num].append(t_idx)

        for s1_idx, (name, addr) in enumerate(zip(s1_names, s1_addrs)):
            comb = f"{name} {addr}"
            nums = set(re.findall(r'\b\d+\b', comb))
            for num in nums:
                if len(num) >= 2 and num in target_num_index:
                    for t_idx in target_num_index[num][:50]:
                        candidates_map[s1_idx][t_idx]["num_hit"] += 1

        logger.info("[%s | Layer 3] Numeric anchor blocking completed in %.2fs", country, time.perf_counter() - t_l3)

        # Layer 4: Global Hybrid Chunked Sparse Top-K Retrieval
        t_l4 = time.perf_counter()
        
        if n_target > 0 and n_s1 > 0:
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=config.TFIDF_NGRAM_RANGE,
                max_features=config.TFIDF_MAX_FEATURES,
                sublinear_tf=True
            )
            all_names_for_vocab = target_names + [n for n in s1_names if n]
            vectorizer.fit(all_names_for_vocab)
            X_target = vectorizer.transform(target_names).tocsr()
            X_s1 = vectorizer.transform(s1_names).tocsr()
            
            # Pre-transpose Target for fast column slicing (O(1) CSC slicing)
            X_target_T = X_target.T.tocsc()
            
            s1_batch_size = 2_000
            t_batch_size = 200_000
            
            for s1_start in range(0, n_s1, s1_batch_size):
                s1_end = min(s1_start + s1_batch_size, n_s1)
                X_s1_chunk = X_s1[s1_start:s1_end]
                
                # Maintain local top-K heap for this S1 chunk
                top_cands_for_chunk = [[] for _ in range(s1_end - s1_start)]
                
                for t_start in range(0, n_target, t_batch_size):
                    t_end = min(t_start + t_batch_size, n_target)
                    X_target_chunk_T = X_target_T[:, t_start:t_end]
                    
                    # Sparse dot product -> shape: (s1_batch, t_batch)
                    sim_chunk = X_s1_chunk.dot(X_target_chunk_T)
                    
                    for local_r in range(sim_chunk.shape[0]):
                        row = sim_chunk.getrow(local_r)
                        if row.nnz == 0:
                            continue
                            
                        cols = row.indices
                        vals = row.data
                        
                        mask = vals >= config.TFIDF_MIN_SIMILARITY
                        if not mask.any():
                            continue
                            
                        cols_filt = cols[mask]
                        vals_filt = vals[mask]
                        
                        if len(vals_filt) > self.top_k:
                            top_k_idx = np.argpartition(vals_filt, -self.top_k)[-self.top_k:]
                            cols_filt = cols_filt[top_k_idx]
                            vals_filt = vals_filt[top_k_idx]
                            
                        # Store candidates internally
                        for c, val in zip(cols_filt, vals_filt):
                            actual_t_idx = t_start + int(c)
                            top_cands_for_chunk[local_r].append((float(val), actual_t_idx))
                            
                    del sim_chunk
                
                # Finalize top-K for this S1 chunk & merge into global candidates_map
                for local_r in range(s1_end - s1_start):
                    actual_s1_idx = s1_start + local_r
                    cands = top_cands_for_chunk[local_r]
                    if cands:
                        cands.sort(key=lambda x: x[0], reverse=True)
                        for val, t_idx in cands[:self.top_k]:
                            candidates_map[actual_s1_idx][t_idx]["tfidf"] = max(
                                candidates_map[actual_s1_idx][t_idx].get("tfidf", 0.0), 
                                val
                            )
                
            del X_target_T
            gc.collect()

        logger.info("[%s | Layer 4] Hybrid global TF-IDF retrieval completed in %.2fs", country, time.perf_counter() - t_l4)

        # Layer 5: Union, Scoring, Capping to TOP_K
        t_l5 = time.perf_counter()
        pair_s1_ids, pair_target_ids, pair_scores, pair_layers = [], [], [], []

        for s1_idx in range(n_s1):
            s1_id = s1_ids[s1_idx]
            cand_dict = candidates_map.get(s1_idx, {})
            if not cand_dict:
                continue

            scored_cands = []
            for t_idx, hits in cand_dict.items():
                tfidf_score = hits["tfidf"]
                token_hit = hits["token_hit"]
                num_hit = hits["num_hit"]
                
                composite_score = (2.0 * tfidf_score) + (1.2 * min(token_hit, 3)) + (1.5 * min(num_hit, 2))
                layers_count = (1 if tfidf_score > 0 else 0) + (1 if token_hit > 0 else 0) + (1 if num_hit > 0 else 0)
                scored_cands.append((t_idx, composite_score, layers_count))

            scored_cands.sort(key=lambda x: x[1], reverse=True)
            top_cands = scored_cands[:self.top_k]

            l2_hits, l3_hits, l4_hits = [], [], []
        
        for t_idx, score, layers in top_cands:
                pair_s1_ids.append(s1_id)
                pair_target_ids.append(target_ids[t_idx])
                pair_scores.append(score)
                pair_layers.append(layers)
                
                hits = cand_dict[t_idx]
                l2_hits.append(True if hits["token_hit"] > 0 else False)
                l3_hits.append(True if hits["num_hit"] > 0 else False)
                l4_hits.append(True if hits["tfidf"] > 0 else False)

        res_df = pl.DataFrame({
            "source1_entity_id": pl.Series(pair_s1_ids, dtype=pl.String),
            "candidate_entity_id": pl.Series(pair_target_ids, dtype=pl.String),
            "heuristic_score": pl.Series(pair_scores, dtype=pl.Float32),
            "layers_matched": pl.Series(pair_layers, dtype=pl.Int8),
            "l2": pl.Series(l2_hits, dtype=pl.Boolean),
            "l3": pl.Series(l3_hits, dtype=pl.Boolean),
            "l4": pl.Series(l4_hits, dtype=pl.Boolean)
        })

        elapsed_total = time.perf_counter() - t0
        avg_cands = len(res_df) / max(1, n_s1)
        rss_mb, peak_mb = self._get_memory_mb()

        logger.info(
            "[%s | Layer 5] Blocking complete in %.2fs | Candidates: %d (Avg %.1f/entity) | RAM: %.1f MB",
            country, elapsed_total, len(res_df), avg_cands, rss_mb
        )

        stats = {
            "country": country,
            "s1_entities": n_s1,
            "target_entities": n_target,
            "candidates_generated": len(res_df),
            "avg_candidates_per_entity": avg_cands,
            "time_sec": elapsed_total,
            "peak_ram_mb": peak_mb
        }
        return res_df, stats

    def run_blocking_pipeline(self, mode: str = "train") -> Dict[str, Any]:
        stage_name = f"stage_3_blocking_{mode}"
        self.state_manager.mark_in_progress(stage_name)
        t_start = time.perf_counter()

        country_dirs = [d for d in config.CLEANED_DIR.iterdir() if d.is_dir()]
        logger.info("Found %d dynamic country partitions under %s", len(country_dirs), config.CLEANED_DIR)

        all_candidate_dfs = []
        country_stats = {}

        for c_dir in country_dirs:
            country = c_dir.name
            s1_pattern = "val_s1_part_*.parquet" if mode == "val" else ("test_source1_part_*.parquet" if mode == "test" else "train_source1_part_*.parquet")
            search_dir = config.VAL_SPLIT_DIR if mode == "val" else c_dir
            
            s1_files = list(search_dir.glob(s1_pattern))
            if not s1_files:
                continue

            s1_df = pl.concat([pl.read_parquet(f) for f in s1_files])
            
            s2_files = list(c_dir.glob("test_source2_part_*.parquet" if mode == "test" else "train_source2_part_*.parquet"))
            s3_files = list(c_dir.glob("test_source3_part_*.parquet" if mode == "test" else "train_source3_part_*.parquet"))
            
            target_files = s2_files + s3_files
            if not target_files:
                continue

            target_df = pl.concat([pl.read_parquet(f) for f in target_files])

            cand_df, stats = self.block_country_partition(country, s1_df, target_df)
            all_candidate_dfs.append(cand_df)
            country_stats[country] = stats

            del s1_df, target_df
            gc.collect()

        if all_candidate_dfs:
            final_cand_df = pl.concat(all_candidate_dfs)
        else:
            final_cand_df = pl.DataFrame({
                "source1_entity_id": pl.Series([], dtype=pl.String),
                "candidate_entity_id": pl.Series([], dtype=pl.String),
                "heuristic_score": pl.Series([], dtype=pl.Float32),
                "layers_matched": pl.Series([], dtype=pl.Int8),
                "l2": pl.Series([], dtype=pl.Boolean),
                "l3": pl.Series([], dtype=pl.Boolean),
                "l4": pl.Series([], dtype=pl.Boolean)
            })

        out_parquet = config.BLOCKED_DIR / f"{mode}_candidates.parquet"
        self._atomic_write_parquet(final_cand_df, out_parquet)
        self.state_manager.record_artifact(f"{mode}_candidates", out_parquet, len(final_cand_df), meta=country_stats)

        if mode == "test":
            self._export_official_candidate_pairs_tsv(final_cand_df)

        recall_est = 0.0
        if mode == "val":
            recall_est = self._estimate_validation_recall(final_cand_df)

        elapsed = time.perf_counter() - t_start
        self.state_manager.mark_completed(stage_name, meta={
            "total_candidates": len(final_cand_df),
            "time_sec": elapsed,
            "recall_estimate": recall_est,
            "country_stats": country_stats
        })

        logger.info(
            "[%s Blocking Complete] Total Candidates: %d | Time: %.2fs | Recall Est: %.2f%% | Peak RAM: %.1f MB",
            mode.upper(), len(final_cand_df), elapsed, recall_est * 100.0, self.peak_memory_mb
        )

        return {
            "mode": mode,
            "total_candidates": len(final_cand_df),
            "recall_estimate": recall_est,
            "time_sec": elapsed,
            "peak_ram_mb": self.peak_memory_mb,
            "country_stats": country_stats
        }

    def _export_official_candidate_pairs_tsv(self, cand_df: pl.DataFrame):
        out_tsv = config.OUTPUT_DIR / "candidate_pairs.tsv"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        grouped = cand_df.group_by("source1_entity_id").agg(
            pl.col("candidate_entity_id").implode().map_elements(lambda ids: ",".join(ids), return_dtype=pl.String).alias("candidate_entity_ids")
        )
        
        test_s1_files = list(config.CLEANED_DIR.rglob("test_source1_part_*.parquet"))
        if test_s1_files:
            all_s1_df = pl.concat([pl.read_parquet(f).select("entity_id") for f in test_s1_files]).rename({"entity_id": "source1_entity_id"})
            merged = all_s1_df.join(grouped, on="source1_entity_id", how="left").with_columns(
                pl.col("candidate_entity_ids").fill_null("")
            )
        else:
            merged = grouped

        tmp_tsv = out_tsv.with_name("candidate_pairs.tsv.tmp")
        with open(tmp_tsv, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1, c_ids in zip(merged["source1_entity_id"], merged["candidate_entity_ids"]):
                f.write(f"{s1}\t{c_ids}\n")
        os.replace(tmp_tsv, out_tsv)
        logger.info("Exported official candidate pairs to %s (%d rows)", out_tsv, len(merged))

    def _estimate_validation_recall(self, val_cand_df: pl.DataFrame) -> float:
        val_gt_files = list(config.VAL_SPLIT_DIR.glob("val_ground_truth_part_*.parquet"))
        if not val_gt_files:
            return 0.0
        
        val_gt = pl.concat([pl.read_parquet(f) for f in val_gt_files])
        
        pairs_set = set()
        for s1, mids in zip(val_gt["source1_entity_id"], val_gt["matched_entity_ids"]):
            if mids and mids.strip():
                for m in mids.split(","):
                    pairs_set.add((s1, m))
                    
        if not pairs_set:
            return 1.0
            
        cand_set = set(zip(val_cand_df["source1_entity_id"], val_cand_df["candidate_entity_id"]))
        found = len(pairs_set & cand_set)
        return found / len(pairs_set)
