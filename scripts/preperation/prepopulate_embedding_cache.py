#!/usr/bin/env python3
"""
Pre-populate Embedding Cache for ESCO Skills

This script pre-computes and caches embeddings for all ESCO skills,
so subsequent runs of generateForRareLables.py can benefit from fast
embedding lookup instead of recomputing embeddings every time.

Usage:
    python prepopulate_embedding_cache.py \\
        --esco-csv ../../data/ESCO/sources/skills_as_documents_v120.csv \\
        --model-path ../../models/finetuned_esco_model_115640 \\
        --cache-dir ./chroma_embedding_cache
        
This will take ~10-20 minutes depending on the model and number of skills,
but only needs to be done once. Future runs will be much faster.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding_cache import create_embedding_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_esco_skills(esco_path: Path) -> Dict[str, Dict[str, object]]:
    """Load ESCO skill documents from csv file."""
    logger.info(f"Loading ESCO skills from {esco_path}")

    skill_csv = pd.read_csv(esco_path)
    skill_docs = {}
    for _, row in skill_csv.iterrows():
        skill_name = row["preferredLabel"]
        if pd.isna(skill_name):
            continue

        # Safely parse occupation columns
        essential_occs = []
        optional_occs = []

        try:
            essential_val = row.get("isEssentialForOccupations", "[]")
            if not pd.isna(essential_val) and essential_val:
                essential_occs = json.loads(essential_val)
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            optional_val = row.get("isOptionalForOccupations", "[]")
            if not pd.isna(optional_val) and optional_val:
                optional_occs = json.loads(optional_val)
        except (json.JSONDecodeError, TypeError):
            pass

        # Get description and strip whitespace to match generation script format
        desc_raw = row.get("description", "")
        description = desc_raw.strip() if not pd.isna(desc_raw) and desc_raw else ""

        skill_docs[skill_name] = {
            "description": description,
            "occupations": essential_occs + optional_occs,
            "skillType": (
                row.get("skillType", "")
                if not pd.isna(row.get("skillType", ""))
                else ""
            ),
        }
    logger.info(f"Loaded {len(skill_docs)} ESCO skills")
    return skill_docs


def main():
    parser = argparse.ArgumentParser(
        description="Pre-populate embedding cache for ESCO skills",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--esco-csv", required=True, type=Path, help="Path to ESCO skills JSON file"
    )

    parser.add_argument(
        "--model-path", required=True, help="Path to sentence transformer model"
    )

    parser.add_argument(
        "--cache-dir",
        default="./chroma_embedding_cache",
        help="Directory for embedding cache",
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for encoding"
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache before populating",
    )

    args = parser.parse_args()

    # Load ESCO skills
    try:
        skill_docs = load_esco_skills(args.esco_csv)
    except Exception as e:
        logger.error(f"Failed to load ESCO skills: {e}")
        return 1

    # Create embedding cache
    try:
        logger.info(f"Creating embedding cache with model: {args.model_path}")
        cache = create_embedding_cache(
            model_name_or_path=args.model_path, cache_dir=args.cache_dir
        )

        if cache is None:
            logger.error("Failed to create embedding cache")
            return 1

    except Exception as e:
        logger.error(f"Failed to create embedding cache: {e}")
        return 1

    # Clear cache if requested
    if args.clear_cache:
        logger.info("Clearing existing cache...")
        cache.clear_cache()

    # Check existing cache status
    stats = cache.get_cache_stats()
    logger.info(f"Cache status: {stats['total_embeddings']} embeddings already cached")

    # Pre-populate cache
    try:
        logger.info(
            f"Pre-populating cache with {len(skill_docs)} skills "
            f"(batch_size={args.batch_size})..."
        )
        logger.info("This may take 10-20 minutes depending on model and hardware.")

        new_embeddings = cache.populate_from_skill_docs(
            skill_docs, batch_size=args.batch_size
        )

        # Get final statistics
        final_stats = cache.get_cache_stats()

        logger.info("=" * 60)
        logger.info("Cache population complete!")
        logger.info(f"  New embeddings added: {new_embeddings}")
        logger.info(f"  Total embeddings in cache: {final_stats['total_embeddings']}")
        logger.info(f"  Cache directory: {args.cache_dir}")
        logger.info("=" * 60)
        logger.info(
            "You can now use generateForRareLables.py with --skip-cache-prepopulation "
            "for much faster execution!"
        )

        return 0

    except Exception as e:
        logger.error(f"Failed to populate cache: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
