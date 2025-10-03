#!/usr/bin/env python3
"""
Script to reduce overfitting risk in bi-encoder training datasets by limiting
the maximum occurrence of positive labels while preserving dataset diversity
and semantic coherence.

This script identifies labels that are at risk of causing overfitting and then
chooses samples to remove based on semantic similarity, preserving diversity
in the remaining dataset.
"""

import json
import argparse
import logging
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OverfittingReducer:
    """
    Class to handle reduction of overfitting risk in training datasets.
    """
    
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-base'):
        """
        Initialize the overfitting reducer.
        
        Args:
            model_name: Name of the sentence transformer model to use for similarity calculations
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def analyze_label_frequency(self, dataset: List[Dict]) -> Dict[str, int]:
        """
        Analyze the frequency of positive labels in the dataset.
        
        Args:
            dataset: List of training samples with 'query', 'pos', and 'neg' keys
            
        Returns:
            Dictionary mapping labels to their occurrence counts
        """
        logger.info("Analyzing label frequencies...")
        label_counts = Counter()
        
        for sample in dataset:
            for label in sample.get('pos', []):
                label_counts[label] += 1
                
        return dict(label_counts)
    
    def identify_overfitting_labels(self, label_counts: Dict[str, int], max_occurrences: int) -> Set[str]:
        """
        Identify labels that exceed the maximum allowed occurrences.
        
        Args:
            label_counts: Dictionary mapping labels to their counts
            max_occurrences: Maximum allowed occurrences for a label
            
        Returns:
            Set of labels that exceed the maximum occurrences
        """
        overfitting_labels = {
            label for label, count in label_counts.items() 
            if count > max_occurrences
        }
        
        logger.info(f"Found {len(overfitting_labels)} labels exceeding max occurrences of {max_occurrences}")
        return overfitting_labels
    
    def group_samples_by_labels(self, dataset: List[Dict], overfitting_labels: Set[str]) -> Dict[str, List[int]]:
        """
        Group sample indices by the overfitting labels they contain.
        
        Args:
            dataset: List of training samples
            overfitting_labels: Set of labels that need to be reduced
            
        Returns:
            Dictionary mapping labels to lists of sample indices
        """
        label_to_samples = defaultdict(list)
        
        for idx, sample in enumerate(dataset):
            for label in sample.get('pos', []):
                if label in overfitting_labels:
                    label_to_samples[label].append(idx)
                    
        return dict(label_to_samples)
    
    def find_multi_label_samples(self, dataset: List[Dict], overfitting_labels: Set[str]) -> Dict[int, List[str]]:
        """
        Find samples that contain multiple overfitting labels.
        These samples are high-priority for removal as they can reduce overfitting for multiple labels.
        
        Args:
            dataset: List of training samples
            overfitting_labels: Set of labels that need to be reduced
            
        Returns:
            Dictionary mapping sample indices to lists of overfitting labels they contain
        """
        multi_label_samples = {}
        
        for idx, sample in enumerate(dataset):
            sample_overfitting_labels = [
                label for label in sample.get('pos', [])
                if label in overfitting_labels
            ]
            
            if len(sample_overfitting_labels) > 1:
                multi_label_samples[idx] = sample_overfitting_labels
        
        logger.info(f"Found {len(multi_label_samples)} samples with multiple overfitting labels")
        return multi_label_samples
    
    def calculate_query_embeddings(self, queries: List[str]) -> np.ndarray:
        """
        Calculate embeddings for the queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Numpy array of embeddings
        """
        logger.info("Calculating query embeddings...")
        embeddings = self.model.encode(queries, show_progress_bar=True)
        return embeddings
    
    def select_samples_to_keep(self, 
                             dataset: List[Dict], 
                             label_to_samples: Dict[str, List[int]], 
                             overfitting_labels: Set[str],
                             max_occurrences: int,
                             diversity_weight: float = 0.3) -> Tuple[Set[int], Dict]:
        """
        Select which samples to keep using smart prioritization.
        Prioritizes removal of samples with multiple overfitting labels.
        
        Args:
            dataset: List of training samples
            label_to_samples: Mapping from labels to sample indices
            overfitting_labels: Set of labels that need to be reduced
            max_occurrences: Maximum allowed occurrences for each label
            diversity_weight: Weight for diversity preservation (0-1)
            
        Returns:
            Tuple of (samples_to_keep, removal_info)
        """
        samples_to_keep = set(range(len(dataset)))  # Start with all samples
        removal_info = {
            'removed_samples': [],  # List of (sample_idx, reasons)
            'multi_label_impact': {},  # Label -> count of how many samples with multiple labels were removed
        }
        
        # Find samples with multiple overfitting labels
        multi_label_samples = self.find_multi_label_samples(dataset, overfitting_labels)
        
        # Track current label counts after removals
        current_label_counts = {label: len(samples) for label, samples in label_to_samples.items()}
        
        # Phase 1: Prioritize removal of multi-label samples
        logger.info("Phase 1: Removing samples with multiple overfitting labels")
        multi_label_removed = 0
        
        # Sort multi-label samples by number of overfitting labels (descending)
        sorted_multi_label = sorted(multi_label_samples.items(), 
                                  key=lambda x: len(x[1]), reverse=True)
        
        for sample_idx, labels in sorted_multi_label:
            if sample_idx not in samples_to_keep:
                continue  # Already removed
                
            # Check if removing this sample helps with any overfitting labels
            can_remove = False
            affected_labels = []
            
            for label in labels:
                if current_label_counts[label] > max_occurrences:
                    can_remove = True
                    affected_labels.append(label)
            
            if can_remove:
                # Remove the sample
                samples_to_keep.discard(sample_idx)
                multi_label_removed += 1
                
                # Update counts and track removal info
                for label in labels:
                    if label in current_label_counts:
                        current_label_counts[label] -= 1
                        removal_info['multi_label_impact'][label] = removal_info['multi_label_impact'].get(label, 0) + 1
                
                removal_info['removed_samples'].append((sample_idx, f"Multiple overfitting labels: {', '.join(affected_labels)}"))
        
        logger.info(f"Removed {multi_label_removed} samples with multiple overfitting labels")
        
        # Phase 2: Handle remaining overfitting labels with diversity-based selection
        logger.info("Phase 2: Handling remaining overfitting labels with diversity preservation")
        
        for label, sample_indices in label_to_samples.items():
            if current_label_counts[label] <= max_occurrences:
                continue
                
            # Filter to samples that haven't been removed yet
            remaining_samples = [idx for idx in sample_indices if idx in samples_to_keep]
            samples_to_remove_count = current_label_counts[label] - max_occurrences
            
            if samples_to_remove_count <= 0 or len(remaining_samples) <= max_occurrences:
                continue
                
            logger.info(f"Processing label '{label}': {len(remaining_samples)} remaining samples, need to remove {samples_to_remove_count}")
            
            # Get queries for remaining samples
            queries = [dataset[idx]['query'] for idx in remaining_samples]
            embeddings = self.calculate_query_embeddings(queries)
            
            # Select samples to remove (least diverse ones)
            samples_to_remove = self._select_samples_to_remove(
                remaining_samples, embeddings, samples_to_remove_count, diversity_weight
            )
            
            # Remove selected samples
            for sample_idx in samples_to_remove:
                samples_to_keep.discard(sample_idx)
                removal_info['removed_samples'].append((sample_idx, f"Overfitting label: {label}"))
            
            # Update count
            current_label_counts[label] -= len(samples_to_remove)
            logger.info(f"Removed {len(samples_to_remove)} additional samples for label '{label}'")
        
        return samples_to_keep, removal_info
    
    def _select_samples_to_remove(self, 
                                sample_indices: List[int], 
                                embeddings: np.ndarray, 
                                num_to_remove: int,
                                diversity_weight: float) -> List[int]:
        """
        Select samples to remove, prioritizing less diverse ones.
        
        Args:
            sample_indices: List of sample indices to select from
            embeddings: Embeddings for the samples
            num_to_remove: Number of samples to remove
            diversity_weight: Weight for diversity consideration
            
        Returns:
            List of sample indices to remove
        """
        if len(sample_indices) <= num_to_remove:
            return sample_indices
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Calculate diversity score for each sample (lower = more similar to others)
        diversity_scores = []
        for i in range(len(embeddings)):
            # Average similarity to all other samples (lower is more diverse)
            avg_similarity = np.mean([similarity_matrix[i][j] for j in range(len(embeddings)) if i != j])
            diversity_scores.append((i, avg_similarity))
        
        # Sort by diversity score (highest similarity first = least diverse)
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select samples to remove (mix of least diverse and random)
        samples_to_remove = []
        remaining_candidates = list(range(len(sample_indices)))
        
        for _ in range(num_to_remove):
            if not remaining_candidates:
                break
                
            if random.random() < diversity_weight and len(diversity_scores) > 0:
                # Select least diverse sample
                for score_idx, _ in diversity_scores:
                    if score_idx in remaining_candidates:
                        samples_to_remove.append(sample_indices[score_idx])
                        remaining_candidates.remove(score_idx)
                        diversity_scores = [(i, s) for i, s in diversity_scores if i != score_idx]
                        break
            else:
                # Random selection from remaining
                if remaining_candidates:
                    selected_idx = random.choice(remaining_candidates)
                    samples_to_remove.append(sample_indices[selected_idx])
                    remaining_candidates.remove(selected_idx)
                    diversity_scores = [(i, s) for i, s in diversity_scores if i != selected_idx]
        
        return samples_to_remove
    
    def _select_diverse_samples(self, 
                              sample_indices: List[int], 
                              embeddings: np.ndarray, 
                              max_samples: int,
                              diversity_weight: float) -> List[int]:
        """
        Select diverse samples using a combination of random sampling and diversity maximization.
        
        Args:
            sample_indices: List of sample indices to select from
            embeddings: Embeddings for the samples
            max_samples: Maximum number of samples to select
            diversity_weight: Weight for diversity vs randomness
            
        Returns:
            List of selected sample indices
        """
        if len(sample_indices) <= max_samples:
            return sample_indices
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Start with a random sample
        selected_indices = [0]  # Index in the embeddings array
        remaining_indices = list(range(1, len(sample_indices)))
        
        # Greedily select samples that are most different from already selected ones
        while len(selected_indices) < max_samples and remaining_indices:
            if random.random() < diversity_weight:
                # Diversity-based selection
                best_candidate = None
                best_score = -1
                
                for candidate_idx in remaining_indices:
                    # Calculate minimum similarity to already selected samples
                    min_similarity = min([
                        similarity_matrix[candidate_idx][selected_idx] 
                        for selected_idx in selected_indices
                    ])
                    
                    if min_similarity > best_score:
                        best_score = min_similarity
                        best_candidate = candidate_idx
                
                if best_candidate is not None:
                    selected_indices.append(best_candidate)
                    remaining_indices.remove(best_candidate)
            else:
                # Random selection
                candidate_idx = random.choice(remaining_indices)
                selected_indices.append(candidate_idx)
                remaining_indices.remove(candidate_idx)
        
        # Convert back to original sample indices
        return [sample_indices[i] for i in selected_indices]
    
    def reduce_overfitting_risk(self, 
                               dataset: List[Dict], 
                               max_occurrences: int,
                               diversity_weight: float = 0.3) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Reduce overfitting risk in the dataset.
        
        Args:
            dataset: Original dataset
            max_occurrences: Maximum allowed occurrences for any positive label
            diversity_weight: Weight for diversity preservation
            
        Returns:
            Tuple of (reduced_dataset, filtered_samples, statistics_dict)
        """
        logger.info(f"Starting overfitting reduction with max_occurrences={max_occurrences}")
        
        # Analyze current label frequencies
        original_label_counts = self.analyze_label_frequency(dataset)
        
        # Identify overfitting labels
        overfitting_labels = self.identify_overfitting_labels(original_label_counts, max_occurrences)
        
        if not overfitting_labels:
            logger.info("No overfitting labels found. Returning original dataset.")
            return dataset, [], {
                'original_size': len(dataset),
                'final_size': len(dataset),
                'removed_samples': 0,
                'overfitting_labels': 0
            }
        
        # Group samples by overfitting labels
        label_to_samples = self.group_samples_by_labels(dataset, overfitting_labels)
        
        # Select samples to keep
        samples_to_keep, removal_info = self.select_samples_to_keep(
            dataset, label_to_samples, overfitting_labels, max_occurrences, diversity_weight
        )
        
        # Create reduced dataset
        reduced_dataset = [dataset[i] for i in sorted(samples_to_keep)]
        
        # Create filtered samples with removal reasons
        filtered_samples = []
        removal_dict = dict(removal_info['removed_samples'])
        
        for idx, sample in enumerate(dataset):
            if idx not in samples_to_keep:
                filtered_sample = sample.copy()
                filtered_sample['removal_reason'] = removal_dict.get(idx, "Unknown reason")
                filtered_samples.append(filtered_sample)
        
        # Calculate final statistics
        final_label_counts = self.analyze_label_frequency(reduced_dataset)
        
        statistics = {
            'original_size': len(dataset),
            'final_size': len(reduced_dataset),
            'removed_samples': len(dataset) - len(reduced_dataset),
            'overfitting_labels': len(overfitting_labels),
            'multi_label_impact': removal_info['multi_label_impact'],
            'original_label_counts': original_label_counts,
            'final_label_counts': final_label_counts
        }
        
        logger.info(f"Overfitting reduction completed. Reduced from {len(dataset)} to {len(reduced_dataset)} samples")
        
        return reduced_dataset, filtered_samples, statistics


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSON file.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        List of training samples
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def save_dataset(dataset: List[Dict], output_path: str):
    """
    Save dataset to JSON file.
    
    Args:
        dataset: List of training samples
        output_path: Path to save the dataset
    """
    logger.info(f"Saving dataset to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(dataset)} samples")


def save_statistics(statistics: Dict, stats_path: str):
    """
    Save statistics to JSON file.
    
    Args:
        statistics: Statistics dictionary
        stats_path: Path to save statistics
    """
    logger.info(f"Saving statistics to: {stats_path}")
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)


def main():
    """Main function to run the overfitting reduction script."""
    parser = argparse.ArgumentParser(
        description="Reduce overfitting risk in bi-encoder training datasets"
    )
    
    parser.add_argument(
        'dataset_name',
        help='Name of the dataset file (e.g., "combinedEsco.json", "combinedGreta.json")'
    )
    
    parser.add_argument(
        '--max-occurrences', '-m',
        type=int,
        default=10,
        help='Maximum allowed occurrences for any positive label (default: 10)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output path for the reduced dataset (default: adds "_reduced" to original name)'
    )
    
    parser.add_argument(
        '--diversity-weight', '-d',
        type=float,
        default=0.3,
        help='Weight for diversity preservation (0.0-1.0, default: 0.3)'
    )
    
    parser.add_argument(
        '--model', 
        default='intfloat/multilingual-e5-base',
        help='Sentence transformer model to use for similarity calculations'
    )
    
    parser.add_argument(
        '--save-stats',
        action='store_true',
        help='Save statistics about the reduction process'
    )
    
    parser.add_argument(
        '--save-filtered',
        action='store_true',
        help='Save filtered-out samples with removal reasons'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate arguments
    if not os.path.exists(args.dataset_name):
        logger.error(f"Dataset file not found: {args.dataset_name}")
        return 1
    
    if not 0.0 <= args.diversity_weight <= 1.0:
        logger.error("Diversity weight must be between 0.0 and 1.0")
        return 1
    
    # Generate output path if not provided
    if args.output is None:
        base_name = os.path.splitext(args.dataset_name)[0]
        args.output = f"{base_name}_reduced_max{args.max_occurrences}.json"
    
    try:
        # Load dataset
        dataset = load_dataset(args.dataset_name)
        
        # Initialize overfitting reducer
        reducer = OverfittingReducer(args.model)
        
        # Reduce overfitting risk
        reduced_dataset, filtered_samples, statistics = reducer.reduce_overfitting_risk(
            dataset, 
            args.max_occurrences,
            args.diversity_weight
        )
        
        # Save reduced dataset
        save_dataset(reduced_dataset, args.output)
        
        # Save statistics if requested
        if args.save_stats:
            stats_path = os.path.splitext(args.output)[0] + "_stats.json"
            save_statistics(statistics, stats_path)
        
        # Save filtered samples if requested
        if args.save_filtered and filtered_samples:
            filtered_path = os.path.splitext(args.output)[0] + "_filtered.json"
            save_dataset(filtered_samples, filtered_path)
            logger.info(f"Saved {len(filtered_samples)} filtered samples to: {filtered_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"OVERFITTING REDUCTION SUMMARY")
        print(f"{'='*60}")
        print(f"Original dataset size: {statistics['original_size']:,}")
        print(f"Final dataset size: {statistics['final_size']:,}")
        print(f"Samples removed: {statistics['removed_samples']:,}")
        print(f"Reduction percentage: {statistics['removed_samples']/statistics['original_size']*100:.1f}%")
        print(f"Overfitting labels found: {statistics['overfitting_labels']}")
        print(f"Output saved to: {args.output}")
        
        # Show multi-label impact statistics
        if statistics.get('multi_label_impact'):
            multi_label_total = sum(statistics['multi_label_impact'].values())
            print(f"\nMulti-label sample removal impact:")
            print(f"Total multi-label removals: {multi_label_total}")
            for label, count in sorted(statistics['multi_label_impact'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                print(f"  '{label}': {count} samples")
        
        if statistics['overfitting_labels'] > 0:
            print(f"\nTop labels that were reduced:")
            original_counts = statistics['original_label_counts']
            final_counts = statistics['final_label_counts']
            
            # Show top labels that were reduced
            reduced_labels = [
                (label, original_counts[label], final_counts.get(label, 0))
                for label in sorted(original_counts.keys(), key=lambda x: original_counts[x], reverse=True)
                if original_counts[label] > args.max_occurrences
            ][:10]
            
            for label, original, final in reduced_labels:
                print(f"  '{label}': {original} → {final}")
        
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())