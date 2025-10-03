#!/usr/bin/env python3
"""Enrich skill labels with their full descriptions from ESCO skill documents."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def load_skill_documents(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load skill documents from CSV file and create a lookup dictionary."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Skill documents CSV not found: {csv_path}")

    skill_docs: Dict[str, Dict[str, str]] = {}
    
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Try different possible column names for the skill label
            preferred_label = (
                row.get("preferredLabel") or 
                row.get("preferred_label") or 
                row.get("title") or
                row.get("label")
            )
            
            if not preferred_label:
                continue
                
            description = (row.get("description") or "").strip()
            
            # Create the enriched label combining preferredLabel and description
            enriched_label = preferred_label.strip()
            if description:
                enriched_label = f"{preferred_label.strip()}: {description}"
            
            skill_docs[preferred_label.strip()] = {
                "original": preferred_label.strip(),
                "enriched": enriched_label,
                "description": description
            }
    
    logger.info("Loaded %d skill documents from %s", len(skill_docs), csv_path)
    return skill_docs


def load_dataset(dataset_path: Path) -> List[Dict[str, object]]:
    """Load the training dataset from JSON file."""
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of records")
        return data


def enrich_labels(
    labels: List[str], 
    skill_docs: Dict[str, Dict[str, str]],
    fallback_strategy: str = "keep_original"
) -> List[str]:
    """
    Enrich skill labels with descriptions from skill documents.
    
    Args:
        labels: List of skill labels to enrich
        skill_docs: Dictionary mapping labels to their enriched versions
        fallback_strategy: What to do for labels not found in skill_docs
                          "keep_original" - keep the original label
                          "skip" - omit labels not found in skill_docs
    
    Returns:
        List of enriched labels
    """
    enriched = []
    
    for label in labels:
        if not isinstance(label, str):
            continue
            
        label_info = skill_docs.get(label.strip())
        if label_info:
            enriched.append(label_info["enriched"])
        else:
            if fallback_strategy == "keep_original":
                enriched.append(label)
                logger.debug(f"No enrichment found for label: '{label}', keeping original")
            elif fallback_strategy == "skip":
                logger.debug(f"Skipping label not found in skill docs: '{label}'")
            # For other strategies, we simply don't add the label
    
    return enriched


def process_record(
    record: Dict[str, object], 
    skill_docs: Dict[str, Dict[str, str]],
    pos_fallback: str = "keep_original",
    neg_fallback: str = "keep_original"
) -> Dict[str, object]:
    """Process a single dataset record, enriching all skill labels."""
    # Create a copy of the record
    enriched_record = record.copy()
    
    # Enrich positive labels
    pos_labels = record.get("pos", [])
    if isinstance(pos_labels, list):
        enriched_pos = enrich_labels(pos_labels, skill_docs, pos_fallback)
        enriched_record["pos"] = enriched_pos
    
    # Enrich negative labels
    neg_labels = record.get("neg", [])
    if isinstance(neg_labels, list):
        enriched_neg = enrich_labels(neg_labels, skill_docs, neg_fallback)
        enriched_record["neg"] = enriched_neg
    
    # Update metadata to indicate enrichment
    meta = record.get("meta", {})
    if isinstance(meta, dict):
        enriched_meta = meta.copy()
        enriched_meta["enriched"] = True
        enriched_meta["pos_fallback"] = pos_fallback
        enriched_meta["neg_fallback"] = neg_fallback
        enriched_record["meta"] = enriched_meta
    
    return enriched_record


def analyze_dataset(
    dataset: List[Dict[str, object]], 
    skill_docs: Dict[str, Dict[str, str]]
) -> Dict[str, object]:
    """Analyze the dataset to provide statistics about label coverage."""
    stats = {
        "total_records": len(dataset),
        "total_pos_labels": 0,
        "total_neg_labels": 0,
        "enrichable_pos_labels": 0,
        "enrichable_neg_labels": 0,
        "unique_pos_labels": set(),
        "unique_neg_labels": set(),
        "missing_pos_labels": set(),
        "missing_neg_labels": set()
    }
    
    for record in dataset:
        pos_labels = record.get("pos", [])
        neg_labels = record.get("neg", [])
        
        if isinstance(pos_labels, list):
            for label in pos_labels:
                if isinstance(label, str):
                    stats["total_pos_labels"] += 1
                    stats["unique_pos_labels"].add(label.strip())
                    if label.strip() in skill_docs:
                        stats["enrichable_pos_labels"] += 1
                    else:
                        stats["missing_pos_labels"].add(label.strip())
        
        if isinstance(neg_labels, list):
            for label in neg_labels:
                if isinstance(label, str):
                    stats["total_neg_labels"] += 1
                    stats["unique_neg_labels"].add(label.strip())
                    if label.strip() in skill_docs:
                        stats["enrichable_neg_labels"] += 1
                    else:
                        stats["missing_neg_labels"].add(label.strip())
    
    # Convert sets to counts for final stats
    stats["unique_pos_count"] = len(stats["unique_pos_labels"])
    stats["unique_neg_count"] = len(stats["unique_neg_labels"])
    stats["missing_pos_count"] = len(stats["missing_pos_labels"])
    stats["missing_neg_count"] = len(stats["missing_neg_labels"])
    
    return stats


def write_dataset(output_path: Path, dataset: List[Dict[str, object]]) -> None:
    """Write the enriched dataset to a JSON file."""
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote %d enriched records to %s", len(dataset), output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich skill labels with descriptions from ESCO skill documents"
    )
    parser.add_argument("dataset", help="Path to the source dataset JSON file")
    parser.add_argument("output", help="Path to write the enriched dataset JSON file")
    parser.add_argument(
        "--skill-docs",
        default="../../data/ESCO/sources/skills_as_documents_v120.csv",
        help="CSV file containing ESCO skill descriptions (default: %(default)s)",
    )
    parser.add_argument(
        "--pos-fallback",
        choices=["keep_original", "skip"],
        default="keep_original",
        help="Strategy for positive labels not found in skill docs (default: %(default)s)",
    )
    parser.add_argument(
        "--neg-fallback", 
        choices=["keep_original", "skip"],
        default="keep_original",
        help="Strategy for negative labels not found in skill docs (default: %(default)s)",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    skill_docs_path = Path(args.skill_docs)

    # Load skill documents
    logger.info("Loading skill documents from %s", skill_docs_path)
    try:
        skill_docs = load_skill_documents(skill_docs_path)
    except FileNotFoundError as exc:
        logger.error("Failed to load skill documents: %s", exc)
        return 1

    # Load dataset
    logger.info("Loading dataset from %s", dataset_path)
    try:
        dataset = load_dataset(dataset_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        return 1

    # Analyze dataset before enrichment
    logger.info("=== Dataset Analysis ===")
    stats = analyze_dataset(dataset, skill_docs)
    
    logger.info("Records: %d", stats["total_records"])
    logger.info("Positive labels: %d total, %d unique", 
                stats["total_pos_labels"], stats["unique_pos_count"])
    logger.info("Negative labels: %d total, %d unique", 
                stats["total_neg_labels"], stats["unique_neg_count"])
    logger.info("Enrichable positive labels: %d/%d (%.1f%%)", 
                stats["enrichable_pos_labels"], stats["total_pos_labels"],
                100 * stats["enrichable_pos_labels"] / max(stats["total_pos_labels"], 1))
    logger.info("Enrichable negative labels: %d/%d (%.1f%%)", 
                stats["enrichable_neg_labels"], stats["total_neg_labels"],
                100 * stats["enrichable_neg_labels"] / max(stats["total_neg_labels"], 1))
    
    if stats["missing_pos_count"] > 0:
        logger.info("Missing positive labels: %d unique", stats["missing_pos_count"])
        if args.log.upper() == "DEBUG" and stats["missing_pos_count"] <= 20:
            for label in sorted(stats["missing_pos_labels"]):
                logger.debug("Missing positive label: '%s'", label)
    
    if stats["missing_neg_count"] > 0:
        logger.info("Missing negative labels: %d unique", stats["missing_neg_count"])
        if args.log.upper() == "DEBUG" and stats["missing_neg_count"] <= 20:
            for label in sorted(stats["missing_neg_labels"]):
                logger.debug("Missing negative label: '%s'", label)

    # Process dataset
    logger.info("=== Enriching Labels ===")
    enriched_dataset = []
    
    for idx, record in enumerate(dataset):
        if len(dataset) > 10 and (idx + 1) % max(1, len(dataset) // 20) == 0:
            progress = (idx + 1) / len(dataset) * 100
            logger.info(f"Progress: {idx + 1}/{len(dataset)} ({progress:.1f}%)")
        
        enriched_record = process_record(
            record, 
            skill_docs, 
            args.pos_fallback, 
            args.neg_fallback
        )
        enriched_dataset.append(enriched_record)

    # Calculate enrichment statistics
    logger.info("=== Enrichment Results ===")
    total_original_length = 0
    total_enriched_length = 0
    enriched_count = 0
    
    for i, (original, enriched) in enumerate(zip(dataset, enriched_dataset)):
        orig_pos = original.get("pos", [])
        orig_neg = original.get("neg", [])
        enr_pos = enriched.get("pos", [])
        enr_neg = enriched.get("neg", [])
        
        if isinstance(orig_pos, list) and isinstance(enr_pos, list):
            total_original_length += sum(len(str(label)) for label in orig_pos)
            total_enriched_length += sum(len(str(label)) for label in enr_pos)
            enriched_count += sum(1 for orig, enr in zip(orig_pos, enr_pos) if len(str(enr)) > len(str(orig)))
        
        if isinstance(orig_neg, list) and isinstance(enr_neg, list):
            total_original_length += sum(len(str(label)) for label in orig_neg)
            total_enriched_length += sum(len(str(label)) for label in enr_neg)
            enriched_count += sum(1 for orig, enr in zip(orig_neg, enr_neg) if len(str(enr)) > len(str(orig)))

    if total_original_length > 0:
        avg_expansion = total_enriched_length / total_original_length
        logger.info("Average label length expansion: %.2fx", avg_expansion)
        logger.info("Labels actually enriched: %d", enriched_count)

    # Write enriched dataset
    try:
        write_dataset(output_path, enriched_dataset)
    except Exception as exc:
        logger.error("Failed to write enriched dataset: %s", exc)
        return 1

    logger.info("Label enrichment completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        sys.exit(130)