import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.state_manager import StateManager
from src.data_loader import PolarsChunkLoader
from src.blocking import MultiLayerBlocker
from src.feature_engineering import FeatureExtractor
from src.train import EntityMatcherTrainer
from src.inference import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PipelineOrchestrator")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Master Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        choices=["setup", "block", "train", "infer", "validate", "all"],
        default="all",
        help="Stage to execute"
    )
    parser.add_argument("--dry-run", action="store_true", help="Execute lightweight verification without heavy compute")
    args = parser.parse_args()

    sm = StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
    logger.info("=== Amazon ML Challenge 2026 Pipeline Orchestrator (Stage: %s) ===", args.stage)

    if args.stage in ["block", "all"]:
        logger.info("--- STAGE 1..3: DATA LOADING & BLOCKING ---")
        loader = PolarsChunkLoader(state_manager=sm)
        val_ids = loader.get_or_create_val_ids()
        
        # Stream raw TSVs
        loader.stream_and_partition_source("train_source1", config.DATASET_DIR / "train" / "train_source1.tsv", val_ids)
        loader.stream_and_partition_source("train_source2", config.DATASET_DIR / "train" / "train_source2.tsv", val_ids)
        loader.stream_and_partition_source("train_source3", config.DATASET_DIR / "train" / "train_source3.tsv", val_ids)
        loader.stream_and_partition_ground_truth(val_ids)

        if not args.dry_run:
            blocker = MultiLayerBlocker(state_manager=sm)
            blocker.run_blocking_pipeline(mode="val")
            blocker.run_blocking_pipeline(mode="train")
            blocker.run_blocking_pipeline(mode="test")

    if args.stage in ["train", "all"] and not args.dry_run:
        logger.info("--- STAGE 4..5: FEATURE ENGINEERING & MODEL TRAINING ---")
        fe = FeatureExtractor(state_manager=sm)
        fe.extract_features_for_mode("val")
        fe.extract_features_for_mode("train")

        trainer = EntityMatcherTrainer(state_manager=sm)
        trainer.train_model(dry_run=args.dry_run)

    if args.stage in ["infer", "all"] or args.dry_run:
        logger.info("--- STAGE 6: INFERENCE & SUBMISSION GENERATION ---")
        infer = InferencePipeline(state_manager=sm)
        infer.run_inference(dry_run=args.dry_run)

    if args.stage == "validate":
        infer = InferencePipeline(state_manager=sm)
        infer.run_official_validator()

    logger.info("=== Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
