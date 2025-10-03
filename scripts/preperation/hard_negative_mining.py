#!/usr/bin/env python3
"""
🎯 Hard Negative Mining Script for ESCO Skill Retrieval

This script performs hard negative mining using an existing model to find
challenging negatives that will improve training effectiveness.

Usage:
    python hard_negative_mining.py --model_path intfloat/multilingual-e5-base --train_data data/eval_split/train_dataset_remaining.jsonl
    python hard_negative_mining.py --model_path ./models/finetuned_model --train_data data/combined.jsonl --top_k 10
    
Features:
    - Uses existing model to find hard negatives
    - Configurable number of hard negatives per query
    - Avoids including existing positives as negatives
    - Preserves original data structure
    - Creates augmented training dataset
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set
import numpy as np
from tqdm import tqdm
import time

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ sentence-transformers not available. Install with: pip install sentence-transformers")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load JSONL training data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data

def save_jsonl_data(data: List[Dict], file_path: Path):
    """Save data to JSONL format."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} samples to {file_path}")

def extract_all_labels(data: List[Dict]) -> List[str]:
    """Extract all unique labels from the dataset."""
    all_labels = set()
    
    for sample in data:
        pos_labels = sample.get('pos', [])
        neg_labels = sample.get('neg', [])
        
        for label in pos_labels + neg_labels:
            label = label.strip()
            if label:
                all_labels.add(label)
    
    unique_labels = list(all_labels)
    logger.info(f"Extracted {len(unique_labels)} unique labels")
    return unique_labels

def mine_hard_negatives(
    model: SentenceTransformer,
    data: List[Dict],
    all_labels: List[str],
    top_k: int = 10,
    batch_size: int = 32
) -> List[Dict]:
    """
    Mine hard negatives using the provided model.
    
    Hard negatives are labels that are similar to the query but not in the positive set.
    """
    logger.info(f"🔍 Mining hard negatives with top_k={top_k}")
    
    # Encode all labels once
    logger.info(f"🧮 Encoding {len(all_labels)} labels...")
    label_embeddings = model.encode(
        all_labels,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    
    # Create label lookup
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    augmented_data = []
    
    logger.info(f"⛏️  Mining hard negatives for {len(data)} samples...")
    for sample in tqdm(data, desc="Mining negatives"):
        query = sample.get('query', '').strip()
        pos_labels = set(sample.get('pos', []))
        existing_neg_labels = set(sample.get('neg', []))
        
        if not query or not pos_labels:
            augmented_data.append(sample)
            continue
        
        # Encode query
        query_embedding = model.encode([query], convert_to_tensor=True)
        
        # Calculate similarities to all labels
        similarities = model.similarity(query_embedding, label_embeddings)[0]
        
        # Get top similar labels
        top_indices = similarities.argsort(descending=True)
        
        # Find hard negatives (high similarity but not positive)
        hard_negatives = []
        for idx in top_indices:
            candidate_label = all_labels[idx.item()]
            
            # Skip if it's already a positive or existing negative
            if (candidate_label in pos_labels or 
                candidate_label in existing_neg_labels):
                continue
            
            hard_negatives.append(candidate_label)
            if len(hard_negatives) >= top_k:
                break
        
        # Create augmented sample
        augmented_sample = sample.copy()
        
        # Combine existing negatives with hard negatives
        all_negatives = list(existing_neg_labels) + hard_negatives
        augmented_sample['neg'] = all_negatives
        
        # Add metadata about mining
        if 'meta' not in augmented_sample:
            augmented_sample['meta'] = {}
        augmented_sample['meta']['hard_negatives_added'] = len(hard_negatives)
        augmented_sample['meta']['total_negatives'] = len(all_negatives)
        
        augmented_data.append(augmented_sample)
    
    logger.info(f"✅ Hard negative mining completed")
    return augmented_data

def analyze_mining_results(original_data: List[Dict], augmented_data: List[Dict]):
    """Analyze the results of hard negative mining."""
    original_neg_counts = []
    augmented_neg_counts = []
    hard_neg_added = []
    
    for orig, aug in zip(original_data, augmented_data):
        orig_negs = len(orig.get('neg', []))
        aug_negs = len(aug.get('neg', []))
        added = aug.get('meta', {}).get('hard_negatives_added', 0)
        
        original_neg_counts.append(orig_negs)
        augmented_neg_counts.append(aug_negs)
        hard_neg_added.append(added)
    
    logger.info(f"📊 Mining Analysis:")
    logger.info(f"   Average original negatives: {np.mean(original_neg_counts):.2f}")
    logger.info(f"   Average augmented negatives: {np.mean(augmented_neg_counts):.2f}")
    logger.info(f"   Average hard negatives added: {np.mean(hard_neg_added):.2f}")
    logger.info(f"   Samples with hard negatives: {sum(1 for x in hard_neg_added if x > 0)}")

def main():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers is required!")
        logger.error("Install with: pip install sentence-transformers")
        return 1
    
    parser = argparse.ArgumentParser(description='Mine hard negatives for embedding training')
    parser.add_argument('--model_path', '-m', type=str, required=True,
                       help='Path to model for mining (local path or HuggingFace identifier)')
    parser.add_argument('--train_data', '-t', type=str, required=True,
                       help='Path to training data (JSONL format)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--top_k', '-k', type=int, default=8,
                       help='Number of hard negatives to mine per query (default: 8)')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                       help='Batch size for encoding (default: 32)')
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"🤖 Loading model: {args.model_path}")
    try:
        model = SentenceTransformer(args.model_path)
        logger.info(f"✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return 1
    
    # Load training data
    train_path = Path(args.train_data)
    if not train_path.exists():
        logger.error(f"Training data {train_path} does not exist!")
        return 1
    
    original_data = load_jsonl_data(train_path)
    if not original_data:
        logger.error("No valid training data found!")
        return 1
    
    # Extract all labels for candidate pool
    all_labels = extract_all_labels(original_data)
    if not all_labels:
        logger.error("No labels found in training data!")
        return 1
    
    # Mine hard negatives
    logger.info(f"🚀 Starting hard negative mining...")
    start_time = time.time()
    
    augmented_data = mine_hard_negatives(
        model=model,
        data=original_data,
        all_labels=all_labels,
        top_k=args.top_k,
        batch_size=args.batch_size
    )
    
    mining_time = time.time() - start_time
    logger.info(f"⏱️  Mining completed in {mining_time:.2f}s")
    
    # Analyze results
    analyze_mining_results(original_data, augmented_data)
    
    # Save augmented data
    if args.output:
        output_path = Path(args.output)
    else:
        # Create output filename
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        train_stem = train_path.stem
        output_path = train_path.parent / f"{train_stem}_hard_negatives_{timestamp}.jsonl"
    
    save_jsonl_data(augmented_data, output_path)
    
    logger.info(f"✨ Hard negative mining completed successfully!")
    logger.info(f"   Original samples: {len(original_data)}")
    logger.info(f"   Augmented file: {output_path}")
    logger.info(f"   Next step: Use {output_path} for training")
    
    return 0

if __name__ == "__main__":
    exit(main())