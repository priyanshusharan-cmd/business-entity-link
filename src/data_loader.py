import gc
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple

import polars as pl
import psutil

from src.config import config
from src.state_manager import StateManager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DataLoader")


class PolarsChunkLoader:
    """
    Memory-safe, restartable Polars streaming data loader.
    Partitions datasets dynamically by country without full-file loads.
    Builds a leak-free validation split for macro F0.5 evaluation.
    """
    def __init__(self, state_manager: Optional[StateManager] = None, chunk_size: int = config.CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        
        # Ensure directories exist
        config.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        config.VAL_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    def _get_memory_mb(self) -> Tuple[float, float]:
        """Returns (current_rss_mb, peak_rss_mb)."""
        rss = self.process.memory_info().rss / (1024 * 1024)
        if rss > self.peak_memory_mb:
            self.peak_memory_mb = rss
        return rss, self.peak_memory_mb

    def _atomic_write_parquet(self, df: pl.DataFrame, out_path: Path):
        """Atomically writes a Polars DataFrame to Parquet using a temp file."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f"{out_path.name}.tmp")
        df.write_parquet(tmp_path, compression="snappy")
        os.replace(tmp_path, out_path)

    # --------------------------------------------------------------------------
    # Adjustment 3: Leak-Free Validation Split Selection
    # --------------------------------------------------------------------------
    def get_or_create_val_ids(self) -> Set[str]:
        """
        Deterministically selects 100,000 Source 1 IDs stratified by country
        from train_source1.tsv, with zero ground-truth leakage.
        """
        val_ids_path = config.VAL_SPLIT_DIR / "val_s1_ids.parquet"
        if val_ids_path.exists():
            df = pl.read_parquet(val_ids_path)
            return set(df["entity_id"].to_list())

        logger.info("Sampling 100,000 stratified validation Source 1 IDs (seed=%d)...", config.RANDOM_SEED)
        s1_path = config.DATASET_DIR / "train" / "train_source1.tsv"
        
        # Fast 2-column scan to sample without loading addresses/names
        id_country_df = pl.scan_csv(
            s1_path,
            separator="\t",
            schema_overrides={"entity_id": pl.String, "country": pl.String}
        ).select(["entity_id", "country"]).collect()

        # Stratified sampling: 60,000 US, 40,000 India
        val_us = id_country_df.filter(pl.col("country") == "US").sample(
            n=min(60_000, len(id_country_df)),
            seed=config.RANDOM_SEED
        ).select("entity_id")
        
        val_in = id_country_df.filter(pl.col("country") == "India").sample(
            n=min(40_000, len(id_country_df)),
            seed=config.RANDOM_SEED
        ).select("entity_id")

        val_df = pl.concat([val_us, val_in])
        self._atomic_write_parquet(val_df, val_ids_path)
        
        val_ids = set(val_df["entity_id"].to_list())
        logger.info("Validation split initialized with %d disjoint S1 entities.", len(val_ids))
        return val_ids

    # --------------------------------------------------------------------------
    # Core Streaming & Dynamic Country Partitioning
    # --------------------------------------------------------------------------
    def stream_and_partition_source(
        self,
        source_key: str,
        tsv_path: Path,
        val_ids: Optional[Set[str]] = None,
        max_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Streams a TSV in chunks of config.CHUNK_SIZE rows.
        Dynamically partitions by country to artifacts/cleaned/{country}/.
        Routes validation S1 rows to artifacts/val_split/.
        Resumes seamlessly if interrupted.
        """
        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            raise FileNotFoundError(f"Source TSV not found: {tsv_path}")

        # Register source file fingerprint in manifest
        src_info = self.state_manager.register_source_file(source_key, tsv_path)
        completed_chunks = set(src_info.get("completed_chunks", []))

        logger.info(
            "[%s] Starting streaming load from %s (size: %.1f MB, completed chunks: %d)",
            source_key, tsv_path.name, tsv_path.stat().st_size / (1024 * 1024), len(completed_chunks)
        )

        lf = pl.scan_csv(
            tsv_path,
            separator="\t",
            schema_overrides={
                "entity_id": pl.String,
                "business_name": pl.String,
                "business_address": pl.String,
                "country": pl.String
            },
            null_values=["", "NULL", "null", "None"]
        )

        batch_iter = lf.collect_batches(chunk_size=self.chunk_size)
        chunk_idx = 0
        total_rows_processed = 0
        val_id_list = list(val_ids) if val_ids else None

        for chunk_df in batch_iter:
            chunk_rows = len(chunk_df)
            
            # Check if this chunk was already processed and verified
            if chunk_idx in completed_chunks:
                logger.info("[%s | Chunk %04d] Already completed — skipping.", source_key, chunk_idx)
                chunk_idx += 1
                total_rows_processed += chunk_rows
                if max_chunks and chunk_idx >= max_chunks:
                    break
                continue

            t0 = time.perf_counter()
            partition_files = {}

            # Handle validation split separation for train_source1
            train_chunk = chunk_df
            if val_id_list and source_key == "train_source1":
                is_val = chunk_df["entity_id"].is_in(val_id_list)
                val_rows = chunk_df.filter(is_val)
                train_chunk = chunk_df.filter(~is_val)
                
                if len(val_rows) > 0:
                    val_out = config.VAL_SPLIT_DIR / f"val_s1_part_{chunk_idx:05d}.parquet"
                    self._atomic_write_parquet(val_rows, val_out)
                    partition_files["val_split"] = str(val_out)

            # Adjustment 1: Dynamic Country Partitioning
            partitions = train_chunk.partition_by("country", as_dict=True)
            for country_key, part_df in partitions.items():
                raw_c = country_key[0] if isinstance(country_key, tuple) else country_key
                c_str = str(raw_c).strip() if (raw_c is not None and str(raw_c).strip()) else "UNKNOWN"
                
                c_dir = config.CLEANED_DIR / c_str
                c_dir.mkdir(parents=True, exist_ok=True)
                
                out_path = c_dir / f"{source_key}_part_{chunk_idx:05d}.parquet"
                self._atomic_write_parquet(part_df, out_path)
                partition_files[c_str] = str(out_path)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            throughput = chunk_rows / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0
            rss_mb, peak_mb = self._get_memory_mb()

            # Record completed chunk atomically in manifest.json
            self.state_manager.mark_chunk_completed(
                source_key=source_key,
                chunk_idx=chunk_idx,
                rows_in_chunk=chunk_rows,
                partition_files=partition_files
            )

            logger.info(
                "[%s | Chunk %04d] %d rows in %.1f ms | Throughput: %.0f rows/s | RAM: %.1f MB (Peak: %.1f MB) | Partitions: %s",
                source_key, chunk_idx, chunk_rows, elapsed_ms, throughput, rss_mb, peak_mb, list(partitions.keys())
            )

            total_rows_processed += chunk_rows
            chunk_idx += 1
            
            # Explicit garbage collection to enforce memory ceiling
            del chunk_df, train_chunk, partitions
            gc.collect()

            if max_chunks and chunk_idx >= max_chunks:
                break

        return {
            "source_key": source_key,
            "chunks_processed": chunk_idx,
            "total_rows": total_rows_processed,
            "peak_ram_mb": self.peak_memory_mb
        }

    # --------------------------------------------------------------------------
    # Ground Truth Streaming (Train Ground Truth)
    # --------------------------------------------------------------------------
    def stream_and_partition_ground_truth(
        self,
        val_ids: Set[str],
        tsv_path: Optional[Path] = None,
        max_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Streams train_ground_truth.tsv in chunks.
        Splits into validation ground truth and training ground truth
        based strictly on the disjoint val_ids.
        """
        tsv_path = Path(tsv_path or (config.DATASET_DIR / "train" / "train_ground_truth.tsv"))
        source_key = "train_ground_truth"
        
        src_info = self.state_manager.register_source_file(source_key, tsv_path)
        completed_chunks = set(src_info.get("completed_chunks", []))
        
        lf = pl.scan_csv(
            tsv_path,
            separator="\t",
            schema_overrides={
                "source1_entity_id": pl.String,
                "matched_entity_ids": pl.String
            },
            null_values=["", "NULL", "null", "None"]
        )

        batch_iter = lf.collect_batches(chunk_size=self.chunk_size)
        chunk_idx = 0
        total_rows = 0
        val_id_list = list(val_ids)

        for chunk_df in batch_iter:
            chunk_rows = len(chunk_df)
            if chunk_idx in completed_chunks:
                chunk_idx += 1
                total_rows += chunk_rows
                if max_chunks and chunk_idx >= max_chunks:
                    break
                continue

            t0 = time.perf_counter()
            is_val = chunk_df["source1_entity_id"].is_in(val_id_list)
            val_gt = chunk_df.filter(is_val)
            train_gt = chunk_df.filter(~is_val)

            partition_files = {}
            if len(val_gt) > 0:
                val_out = config.VAL_SPLIT_DIR / f"val_ground_truth_part_{chunk_idx:05d}.parquet"
                self._atomic_write_parquet(val_gt, val_out)
                partition_files["val_ground_truth"] = str(val_out)

            train_out = config.CLEANED_DIR / f"train_ground_truth_part_{chunk_idx:05d}.parquet"
            self._atomic_write_parquet(train_gt, train_out)
            partition_files["train_ground_truth"] = str(train_out)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            rss_mb, peak_mb = self._get_memory_mb()

            self.state_manager.mark_chunk_completed(
                source_key=source_key,
                chunk_idx=chunk_idx,
                rows_in_chunk=chunk_rows,
                partition_files=partition_files
            )

            logger.info(
                "[%s | Chunk %04d] %d rows in %.1f ms | RAM: %.1f MB (Peak: %.1f MB) | Val matches: %d, Train matches: %d",
                source_key, chunk_idx, chunk_rows, elapsed_ms, rss_mb, peak_mb, len(val_gt), len(train_gt)
            )

            total_rows += chunk_rows
            chunk_idx += 1
            del chunk_df, val_gt, train_gt
            gc.collect()

            if max_chunks and chunk_idx >= max_chunks:
                break

        return {
            "source_key": source_key,
            "chunks_processed": chunk_idx,
            "total_rows": total_rows,
            "peak_ram_mb": self.peak_memory_mb
        }
