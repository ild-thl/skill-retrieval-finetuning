#!/usr/bin/env python3
"""
🎯 Evaluation Dataset Creator for ESCO Skill Retrieval

This script creates a representative evaluation dataset from training data,
ensuring good coverage across ESCO categories and different topics.

Usage:
    # Fixed size mode (proportional sampling)
    python create_evaluation_dataset.py --input data/combined.jsonl --eval_size 400

    # Ratio mode (consistent percentage per category) - RECOMMENDED
    python create_evaluation_dataset.py --input data/combined.jsonl --target_ratio 0.15
    python create_evaluation_dataset.py --input data/combined.jsonl --target_ratio 0.10 --max_per_category 50

    # Utility: Convert JSON to JSONL
    python create_evaluation_dataset.py --input data/training_data.json --convert_only

Features:
    - Two sampling modes: fixed size OR consistent ratio per category
    - Supports both JSON and JSONL input formats (auto-detection)
    - Representative sampling across all ESCO categories
    - Length-stratified sampling for diverse query lengths
    - Balanced coverage - prevents over/under sampling categories
    - Detailed statistics and visualizations
    - JSON to JSONL conversion utility
"""

import json
import random
import argparse
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Enhanced ESCO category mapping for better classification
ENHANCED_ESCO_CATEGORIES = {
    # Transversal Skills (T)
    "T1": {
        "name": "Lernkompetenz",
        "description": "Lernen, Weiterbildung, Fortbildung, Schulung, Training, Kompetenzentwicklung, Wissensaufbau, lebenslanges Lernen",
        "keywords": [
            "lernen",
            "weiterbildung",
            "fortbildung",
            "schulung",
            "training",
            "entwicklung",
            "wissen",
            "kompetenz",
        ],
    },
    "T2": {
        "name": "Kognitive Fähigkeiten",
        "keywords": [
            "analyse",
            "problem",
            "lösung",
            "kritisch",
            "denken",
            "bewertung",
            "entscheidung",
            "strategie",
        ],
    },
    "T3": {
        "name": "Selbstmanagement",
        "keywords": [
            "selbst",
            "organisation",
            "zeit",
            "planung",
            "priorität",
            "struktur",
            "eigenständig",
            "verantwortung",
        ],
    },
    "T4": {
        "name": "Soziale und kommunikative Fähigkeiten",
        "keywords": [
            "kommunikation",
            "präsentation",
            "team",
            "sozial",
            "gespräch",
            "verhandlung",
            "zusammenarbeit",
        ],
    },
    "T5": {
        "name": "Körperliche und manuelle Fähigkeiten",
        "keywords": [
            "körperlich",
            "manuell",
            "handwerk",
            "bewegung",
            "motorik",
            "geschick",
        ],
    },
    "T6": {
        "name": "Aktive Bürgerschaft",
        "keywords": [
            "bürger",
            "gesellschaft",
            "politik",
            "demokratie",
            "engagement",
            "partizipation",
        ],
    },
    # Skills (S)
    "S1": {
        "name": "Kommunikation, Zusammenarbeit und Kreativität",
        "keywords": [
            "kommunikation",
            "kreativität",
            "innovation",
            "zusammenarbeit",
            "kollaboration",
        ],
    },
    "S2": {
        "name": "Informationskompetenzen",
        "keywords": [
            "information",
            "recherche",
            "daten",
            "analyse",
            "bewertung",
            "quellen",
        ],
    },
    "S3": {
        "name": "Unterstützung und Pflege",
        "keywords": [
            "pflege",
            "betreuung",
            "unterstützung",
            "hilfe",
            "sozial",
            "gesundheit",
        ],
    },
    "S4": {
        "name": "Managementfähigkeiten",
        "keywords": [
            "management",
            "führung",
            "leitung",
            "projekt",
            "team",
            "koordination",
            "controlling",
        ],
    },
    "S5": {
        "name": "Arbeiten mit Computern",
        "keywords": [
            "computer",
            "software",
            "programm",
            "datenbank",
            "excel",
            "word",
            "powerpoint",
            "office",
            "digitale werkzeuge",
        ],
    },
    "S6": {
        "name": "Handhabung/Transport und Bewegung",
        "keywords": ["transport", "logistik", "bewegung", "handhabung", "fahrzeug"],
    },
    "S7": {
        "name": "Bau",
        "keywords": ["bau", "konstruktion", "gebäude", "architektur", "handwerk"],
    },
    "S8": {
        "name": "Arbeiten mit Maschinen und Spezialgeräten",
        "keywords": [
            "maschine",
            "gerät",
            "technik",
            "bedienung",
            "wartung",
            "reparatur",
        ],
    },
    # Knowledge (K)
    "K01": {
        "name": "Erziehungswissenschaften",
        "keywords": [
            "bildung",
            "erziehung",
            "pädagogik",
            "didaktik",
            "unterricht",
            "schule",
        ],
    },
    "K02": {
        "name": "Kunst und Geisteswissenschaften",
        "keywords": [
            "kunst",
            "kultur",
            "geschichte",
            "philosophie",
            "literatur",
            "sprache",
            "musik",
        ],
    },
    "K03": {
        "name": "Sozialwissenschaften, Journalismus und Informationswesen",
        "keywords": [
            "sozial",
            "journalismus",
            "medien",
            "information",
            "gesellschaft",
            "politik",
        ],
    },
    "K04": {
        "name": "Wirtschaft, Verwaltung und Rechtswissenschaft",
        "keywords": [
            "wirtschaft",
            "verwaltung",
            "recht",
            "jura",
            "finanzen",
            "buchhaltung",
            "steuer",
        ],
    },
    "K05": {
        "name": "Naturwissenschaften, Mathematik und Statistik",
        "keywords": [
            "mathematik",
            "statistik",
            "physik",
            "chemie",
            "biologie",
            "naturwissenschaft",
        ],
    },
    "K06": {
        "name": "Informations- und Kommunikationstechnologien",
        "keywords": [
            "informatik",
            "programmierung",
            "softwareentwicklung",
            "webentwicklung",
            "datenbank",
            "netzwerk",
            "server",
            "cloud",
            "coding",
            "algorithmus",
        ],
    },
    "K07": {
        "name": "Ingenieurwesen, Fertigung und Baugewerbe",
        "keywords": [
            "ingenieur",
            "technik",
            "fertigung",
            "produktion",
            "maschinenbau",
            "elektrotechnik",
        ],
    },
    "K08": {
        "name": "Agrarwissenschaft, Forstwissenschaft, Fischerei und Tiermedizin",
        "keywords": ["landwirtschaft", "forst", "tier", "veterinär", "agrar", "natur"],
    },
    "K09": {
        "name": "Gesundheit und soziale Dienste",
        "keywords": [
            "gesundheit",
            "medizin",
            "pflege",
            "therapie",
            "sozial",
            "betreuung",
        ],
    },
    "K10": {
        "name": "Dienstleistungen",
        "keywords": [
            "service",
            "dienstleistung",
            "kunde",
            "beratung",
            "verkauf",
            "gastronomie",
        ],
    },
    # Language (L)
    "L": {
        "name": "Sprachliche Fähigkeiten und Kenntnisse",
        "keywords": [
            "sprache",
            "fremdsprache",
            "english",
            "deutsch",
            "französisch",
            "übersetzen",
        ],
    },
}


def load_data(file_path: Path) -> List[Dict]:
    """Load data from JSON or JSONL file."""
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try to parse as regular JSON first
    try:
        parsed_data = json.loads(content)
        if isinstance(parsed_data, list):
            data = parsed_data
            print(f"   📋 Loaded as JSON array with {len(data)} items")
        elif isinstance(parsed_data, dict):
            data = [parsed_data]
            print(f"   📋 Loaded as single JSON object")
        else:
            raise ValueError("JSON content is not a list or object")
    except json.JSONDecodeError:
        # If JSON parsing fails, try JSONL format
        print(f"   📋 JSON parsing failed, trying JSONL format...")
        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Warning: Invalid JSON on line {line_num}: {e}")
                    continue
        print(f"   📋 Loaded as JSONL with {len(data)} items")

    if not data:
        raise ValueError(f"No valid data found in {file_path}")

    return data


def save_as_jsonl(data: List[Dict], output_file: Path):
    """Save data as JSONL format."""
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def classify_sample_semantic(
    sample: Dict,
    category_embeddings: Dict[str, np.ndarray],
    model: SentenceTransformer,
    threshold: float = 0.3,
) -> Set[str]:
    """
    Classify a sample into ESCO categories using semantic similarity.

    Args:
        sample: Training sample with 'query' and 'pos' fields
        category_embeddings: Pre-computed embeddings for each category description
        model: SentenceTransformer model for encoding
        threshold: Minimum similarity score to assign category

    Returns:
        Set of category codes that match
    """
    # Combine query and positive labels for classification
    query_text = sample.get("query", "")
    pos_labels = sample.get("pos", [])

    # Create comprehensive text for embedding
    sample_text = f"{query_text} {' '.join(pos_labels)}"

    # Encode sample
    sample_embedding = model.encode([sample_text], normalize_embeddings=True)[0]

    # Calculate similarities
    category_scores = {}
    for cat_code, cat_embedding in category_embeddings.items():
        similarity = float(np.dot(sample_embedding, cat_embedding))
        if similarity >= threshold:
            category_scores[cat_code] = similarity

    # Return top categories (max 2)
    if not category_scores:
        return {"UNCLASSIFIED"}

    sorted_categories = sorted(
        category_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Take top category and second if close enough
    result = {sorted_categories[0][0]}
    if (
        len(sorted_categories) > 1
        and sorted_categories[1][1] >= sorted_categories[0][1] * 0.85
    ):
        result.add(sorted_categories[1][0])

    return result


def calculate_eval_statistics(eval_samples, remaining_train_data, category_stats):
    """Calculate comprehensive evaluation dataset statistics."""

    # Length statistics
    eval_query_lengths = [
        len(sample.get("query", "").split()) for sample in eval_samples
    ]
    train_query_lengths = [
        len(sample.get("query", "").split()) for sample in remaining_train_data
    ]

    # Label statistics
    eval_pos_labels = [len(sample.get("pos", [])) for sample in eval_samples]
    eval_neg_labels = [len(sample.get("neg", [])) for sample in eval_samples]
    train_pos_labels = [len(sample.get("pos", [])) for sample in remaining_train_data]
    train_neg_labels = [len(sample.get("neg", [])) for sample in remaining_train_data]

    eval_stats = {
        "dataset_sizes": {
            "eval_samples": len(eval_samples),
            "train_samples": len(remaining_train_data),
            "eval_percentage": len(eval_samples)
            / (len(eval_samples) + len(remaining_train_data))
            * 100,
        },
        "category_distribution": category_stats,
        "length_comparison": {
            "eval": {
                "mean": float(np.mean(eval_query_lengths)) if eval_query_lengths else 0,
                "std": float(np.std(eval_query_lengths)) if eval_query_lengths else 0,
                "min": int(np.min(eval_query_lengths)) if eval_query_lengths else 0,
                "max": int(np.max(eval_query_lengths)) if eval_query_lengths else 0,
                "median": (
                    float(np.median(eval_query_lengths)) if eval_query_lengths else 0
                ),
            },
            "train": {
                "mean": (
                    float(np.mean(train_query_lengths)) if train_query_lengths else 0
                ),
                "std": float(np.std(train_query_lengths)) if train_query_lengths else 0,
                "min": int(np.min(train_query_lengths)) if train_query_lengths else 0,
                "max": int(np.max(train_query_lengths)) if train_query_lengths else 0,
                "median": (
                    float(np.median(train_query_lengths)) if train_query_lengths else 0
                ),
            },
        },
        "label_comparison": {
            "eval_positive": {
                "mean": float(np.mean(eval_pos_labels)) if eval_pos_labels else 0,
                "std": float(np.std(eval_pos_labels)) if eval_pos_labels else 0,
                "median": float(np.median(eval_pos_labels)) if eval_pos_labels else 0,
            },
            "eval_negative": {
                "mean": float(np.mean(eval_neg_labels)) if eval_neg_labels else 0,
                "std": float(np.std(eval_neg_labels)) if eval_neg_labels else 0,
                "median": float(np.median(eval_neg_labels)) if eval_neg_labels else 0,
            },
            "train_positive": {
                "mean": float(np.mean(train_pos_labels)) if train_pos_labels else 0,
                "std": float(np.std(train_pos_labels)) if train_pos_labels else 0,
                "median": float(np.median(train_pos_labels)) if train_pos_labels else 0,
            },
            "train_negative": {
                "mean": float(np.mean(train_neg_labels)) if train_neg_labels else 0,
                "std": float(np.std(train_neg_labels)) if train_neg_labels else 0,
                "median": float(np.median(train_neg_labels)) if train_neg_labels else 0,
            },
        },
    }

    return eval_stats


def create_representative_eval_dataset(
    data: List[Dict],
    eval_size: int = None,
    target_ratio: float = None,
    min_samples_per_category: int = 2,
    max_samples_per_category: int = None,
    stratify_by_length: bool = True,
    seed: int = 42,
    model: Optional[SentenceTransformer] = None,
    use_semantic: bool = True,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Create a representative evaluation dataset with proper ESCO category classification.

    Args:
        data: Input dataset
        eval_size: Target total evaluation size (mutually exclusive with target_ratio)
        target_ratio: Target ratio of eval/total per category (e.g., 0.15 for 15%)
        min_samples_per_category: Minimum samples to take from each category
        max_samples_per_category: Maximum samples to take from each category (optional)
        stratify_by_length: Whether to stratify by query length
        seed: Random seed
        model: SentenceTransformer model for semantic classification (optional)
        use_semantic: Use semantic similarity instead of keywords (default: True)
    """
    if eval_size is None and target_ratio is None:
        raise ValueError("Must specify either eval_size or target_ratio")
    if eval_size is not None and target_ratio is not None:
        raise ValueError("Cannot specify both eval_size and target_ratio")

    random.seed(seed)
    np.random.seed(seed)

    # Prepare category embeddings for semantic classification
    category_embeddings = {}
    if use_semantic and model is not None:
        print("   🧠 Computing category embeddings for semantic classification...")
        category_texts = []
        category_codes = []
        for cat_code, cat_info in ENHANCED_ESCO_CATEGORIES.items():
            # Use description if available, otherwise use name + keywords
            if "description" in cat_info:
                text = cat_info["description"]
            else:
                text = f"{cat_info['name']} {' '.join(cat_info.get('keywords', []))}"
            category_texts.append(text)
            category_codes.append(cat_code)

        # Compute embeddings
        embeddings = model.encode(
            category_texts, normalize_embeddings=True, show_progress_bar=False
        )
        category_embeddings = dict(zip(category_codes, embeddings))

    # Classify samples by ESCO categories
    category_samples = defaultdict(list)

    for i, sample in enumerate(data):
        if use_semantic and model is not None:
            # Semantic classification
            sample_categories = classify_sample_semantic(
                sample, category_embeddings, model
            )
        else:
            # Keyword-based classification (legacy)
            sample_categories = set()
            query_text = sample.get("query", "").lower()
            pos_labels = sample.get("pos", [])
            all_label_text = " ".join(pos_labels).lower()

            # Score each category based on keyword matches
            category_scores = {}
            for cat_code, cat_info in ENHANCED_ESCO_CATEGORIES.items():
                score = 0
                keywords = cat_info.get("keywords", [])

                # Score from query text
                for keyword in keywords:
                    if keyword in query_text:
                        score += 2

                # Score from label text
                for keyword in keywords:
                    if keyword in all_label_text:
                        score += 3

                if score > 0:
                    category_scores[cat_code] = score

            # Assign to best matching categories (top 2 max)
            if category_scores:
                sorted_categories = sorted(
                    category_scores.items(), key=lambda x: x[1], reverse=True
                )
                sample_categories.add(sorted_categories[0][0])
                if (
                    len(sorted_categories) > 1
                    and sorted_categories[1][1] >= sorted_categories[0][1] * 0.7
                ):
                    sample_categories.add(sorted_categories[1][0])
            else:
                sample_categories.add("UNCLASSIFIED")

        # Add sample to all its categories
        for category in sample_categories:
            category_samples[category].append((i, sample))

    # Length-based stratification setup
    if stratify_by_length:
        query_lengths = [len(sample.get("query", "").split()) for sample in data]
        if query_lengths:
            length_percentiles = np.percentile(
                query_lengths, [25, 75]
            )  # More balanced split

            def get_length_bucket(query_len):
                if query_len <= length_percentiles[0]:
                    return "short"
                elif query_len <= length_percentiles[1]:
                    return "medium"
                else:
                    return "long"

    selected_indices = set()
    eval_samples = []
    category_stats = {}

    print(f"🎯 Creating representative evaluation dataset:")
    if target_ratio is not None:
        print(f"   Target ratio: {target_ratio*100:.1f}% per category")
    else:
        print(f"   Target size: {eval_size}")
    print(f"   Found {len(category_samples)} categories")
    print(f"   Stratify by length: {stratify_by_length}")
    print()

    # Sort categories by size (smallest first to ensure representation)
    sorted_categories = sorted(
        [
            (cat, samples)
            for cat, samples in category_samples.items()
            if cat != "UNCLASSIFIED"
        ],
        key=lambda x: len(x[1]),
    )

    # Calculate targets per category based on mode
    if target_ratio is not None:
        # Ratio-based: take target_ratio from each category
        category_targets = {}
        for category, samples in sorted_categories:
            category_size = len(samples)
            target_samples = max(
                min_samples_per_category, int(category_size * target_ratio)
            )
            if max_samples_per_category is not None:
                target_samples = min(target_samples, max_samples_per_category)
            category_targets[category] = target_samples

        # Calculate total eval size based on targets
        total_eval_size = sum(category_targets.values())
        remaining_eval_slots = None  # Not used in ratio mode
    else:
        # Size-based: distribute eval_size proportionally
        total_available = sum(len(samples) for _, samples in sorted_categories)
        remaining_eval_slots = eval_size
        category_targets = None  # Calculated dynamically

    for category, samples in sorted_categories:
        # Calculate target for this category
        category_size = len(samples)

        if target_ratio is not None:
            # Ratio-based mode: use pre-calculated target
            target_samples = category_targets[category]
        else:
            # Size-based mode: adaptive calculation
            if remaining_eval_slots <= 0:
                break

            # For small categories, take higher percentage
            if category_size <= 50:
                target_samples = min(
                    max_samples_per_category,
                    max(min_samples_per_category, category_size // 2),
                )
            else:
                # For larger categories, use proportional sampling
                proportion = category_size / total_available
                proportional_target = int(eval_size * proportion)
                target_samples = min(
                    max_samples_per_category,
                    max(min_samples_per_category, proportional_target),
                )

            # Don't exceed remaining slots
            target_samples = min(target_samples, remaining_eval_slots)

        # Stratified sampling within category
        selected_for_category = []

        if (
            stratify_by_length and len(samples) >= target_samples * 2 and query_lengths
        ):  # Only stratify if enough samples
            # Group by length buckets
            length_buckets = defaultdict(list)
            for idx, sample in samples:
                if idx not in selected_indices:
                    query_len = len(sample.get("query", "").split())
                    bucket = get_length_bucket(query_len)
                    length_buckets[bucket].append((idx, sample))

            # Sample from each bucket
            buckets_with_samples = [
                (bucket, bucket_samples)
                for bucket, bucket_samples in length_buckets.items()
                if bucket_samples
            ]
            if buckets_with_samples:
                samples_per_bucket = max(1, target_samples // len(buckets_with_samples))

                for bucket, bucket_samples in buckets_with_samples:
                    bucket_target = min(samples_per_bucket, len(bucket_samples))
                    if bucket_target > 0:
                        selected = random.sample(bucket_samples, bucket_target)
                        selected_for_category.extend(selected)
                        if len(selected_for_category) >= target_samples:
                            selected_for_category = selected_for_category[
                                :target_samples
                            ]
                            break

        # If stratified sampling didn't get enough, fill with random sampling
        if len(selected_for_category) < target_samples:
            available_samples = [
                (idx, sample)
                for idx, sample in samples
                if idx not in selected_indices
                and idx not in [s[0] for s in selected_for_category]
            ]
            remaining_needed = target_samples - len(selected_for_category)
            if available_samples and remaining_needed > 0:
                additional = random.sample(
                    available_samples, min(remaining_needed, len(available_samples))
                )
                selected_for_category.extend(additional)

        # Add to evaluation set
        for idx, sample in selected_for_category:
            if idx not in selected_indices:
                selected_indices.add(idx)
                eval_samples.append(sample)

        # Update remaining slots only in size-based mode
        if target_ratio is None:
            remaining_eval_slots -= len(selected_for_category)

        category_stats[category] = {
            "total_available": len(samples),
            "selected": len(selected_for_category),
            "percentage": (
                len(selected_for_category) / len(samples) * 100
                if len(samples) > 0
                else 0
            ),
        }

        print(
            f"   {category:4s}: {len(selected_for_category):3d}/{len(samples):3d} samples "
            f"({len(selected_for_category)/len(samples)*100:.1f}%) - {ENHANCED_ESCO_CATEGORIES.get(category, {}).get('name', 'Unknown')}"
        )

    # Handle unclassified samples
    if "UNCLASSIFIED" in category_samples:
        unclassified_size = len(category_samples["UNCLASSIFIED"])

        if target_ratio is not None:
            # Ratio-based: take target_ratio from unclassified
            unclassified_target = max(
                min_samples_per_category, int(unclassified_size * target_ratio)
            )
            if max_samples_per_category is not None:
                unclassified_target = min(unclassified_target, max_samples_per_category)
        else:
            # Size-based: fill remaining slots
            unclassified_target = (
                remaining_eval_slots if remaining_eval_slots > 0 else 0
            )

        if unclassified_target > 0:
            unclassified_samples = [
                (idx, sample)
                for idx, sample in category_samples["UNCLASSIFIED"]
                if idx not in selected_indices
            ]
            if unclassified_samples:
                additional = random.sample(
                    unclassified_samples,
                    min(unclassified_target, len(unclassified_samples)),
                )
                for idx, sample in additional:
                    selected_indices.add(idx)
                    eval_samples.append(sample)

                category_stats["UNCLASSIFIED"] = {
                    "total_available": len(category_samples["UNCLASSIFIED"]),
                    "selected": len(additional),
                    "percentage": len(additional)
                    / len(category_samples["UNCLASSIFIED"])
                    * 100,
                }

    # Create remaining training data
    remaining_train_data = [
        sample for i, sample in enumerate(data) if i not in selected_indices
    ]

    # Calculate comprehensive statistics
    eval_stats = calculate_eval_statistics(
        eval_samples, remaining_train_data, category_stats
    )

    return eval_samples, remaining_train_data, eval_stats


def save_datasets(
    eval_samples, train_samples, eval_stats, output_dir, convert_to_jsonl=False
):
    """Save evaluation and training datasets with statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation dataset
    eval_file = output_dir / "eval_dataset_representative.jsonl"
    save_as_jsonl(eval_samples, eval_file)

    # Save updated training dataset
    train_file = output_dir / "train_dataset_remaining.jsonl"
    save_as_jsonl(train_samples, train_file)

    # Optionally save as regular JSON as well
    if convert_to_jsonl:
        eval_json_file = output_dir / "eval_dataset_representative.json"
        with open(eval_json_file, "w", encoding="utf-8") as f:
            json.dump(eval_samples, f, indent=2, ensure_ascii=False)

        train_json_file = output_dir / "train_dataset_remaining.json"
        with open(train_json_file, "w", encoding="utf-8") as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)

        print(f"   📄 Also saved as JSON: {eval_json_file}, {train_json_file}")

    # Save statistics
    stats_file = output_dir / "eval_split_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(eval_stats, f, indent=2, ensure_ascii=False)

    return eval_file, train_file, stats_file


def create_visualization(eval_samples, train_samples, eval_stats, output_dir):
    """Create visualization of evaluation dataset distribution."""
    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "🎯 Evaluation Dataset Distribution Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Category distribution in eval set
        categories = []
        selected_counts = []
        percentages = []

        for cat, stats in eval_stats["category_distribution"].items():
            if cat != "UNCLASSIFIED" and stats["selected"] > 0:
                categories.append(cat)
                selected_counts.append(stats["selected"])
                percentages.append(stats["percentage"])

        if categories:
            # Sort by selected count
            sorted_data = sorted(
                zip(categories, selected_counts, percentages),
                key=lambda x: x[1],
                reverse=True,
            )
            categories, selected_counts, percentages = zip(*sorted_data)

            ax1.barh(categories, selected_counts, color="lightblue", alpha=0.7)
            ax1.set_xlabel("Number of Evaluation Samples")
            ax1.set_title("Evaluation Samples per ESCO Category")
            ax1.grid(axis="x", alpha=0.3)

            # Add percentage labels
            for i, (count, pct) in enumerate(zip(selected_counts, percentages)):
                ax1.text(count + 0.5, i, f"{pct:.1f}%", va="center", fontsize=9)

        # 2. Query length distribution comparison
        eval_lengths = [len(sample.get("query", "").split()) for sample in eval_samples]
        train_lengths = [
            len(sample.get("query", "").split()) for sample in train_samples
        ]

        if eval_lengths and train_lengths:
            ax2.hist(
                eval_lengths,
                bins=30,
                alpha=0.7,
                label="Evaluation",
                color="lightcoral",
                density=True,
            )
            ax2.hist(
                train_lengths,
                bins=30,
                alpha=0.7,
                label="Training",
                color="lightblue",
                density=True,
            )
            ax2.set_xlabel("Query Length (words)")
            ax2.set_ylabel("Density")
            ax2.set_title("Query Length Distribution: Eval vs Train")
            ax2.legend()
            ax2.grid(alpha=0.3)

        # 3. Label count distribution comparison
        eval_pos_counts = [len(sample.get("pos", [])) for sample in eval_samples]
        train_pos_counts = [len(sample.get("pos", [])) for sample in train_samples]

        if eval_pos_counts and train_pos_counts:
            max_count = max(
                max(eval_pos_counts) if eval_pos_counts else 0,
                max(train_pos_counts) if train_pos_counts else 0,
            )
            if max_count > 0:
                ax3.hist(
                    eval_pos_counts,
                    bins=range(1, max_count + 2),
                    alpha=0.7,
                    label="Evaluation",
                    color="lightgreen",
                    density=True,
                )
                ax3.hist(
                    train_pos_counts,
                    bins=range(1, max_count + 2),
                    alpha=0.7,
                    label="Training",
                    color="orange",
                    density=True,
                )
                ax3.set_xlabel("Number of Positive Labels")
                ax3.set_ylabel("Density")
                ax3.set_title("Positive Label Count: Eval vs Train")
                ax3.legend()
                ax3.grid(alpha=0.3)

        # 4. Category coverage pie chart
        coverage_categories = []
        coverage_counts = []

        for cat, stats in eval_stats["category_distribution"].items():
            if stats["selected"] > 0:
                coverage_categories.append(f"{cat}\n({stats['selected']})")
                coverage_counts.append(stats["selected"])

        if coverage_categories:
            ax4.pie(
                coverage_counts,
                labels=coverage_categories,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax4.set_title("Evaluation Dataset Category Coverage")

        plt.tight_layout()

        # Save plot
        plot_file = Path(output_dir) / "eval_dataset_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"   📊 Visualization saved: {plot_file}")

        plt.show()

    except ImportError:
        print("   ⚠️  Matplotlib not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(
        description="Create representative evaluation dataset for ESCO skill retrieval"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input training data file (JSON or JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/eval_split",
        help="Output directory for evaluation split (default: data/eval_split)",
    )

    # Sampling mode: either fixed size OR ratio (mutually exclusive)
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--eval_size",
        "-s",
        type=int,
        help="Target size of evaluation dataset (e.g., 400)",
    )
    sampling_group.add_argument(
        "--target_ratio",
        "-r",
        type=float,
        help="Target ratio of eval/total per category (e.g., 0.15 for 15%%)",
    )

    parser.add_argument(
        "--min_per_category",
        type=int,
        default=6,
        help="Minimum samples per ESCO category (default: 6)",
    )
    parser.add_argument(
        "--max_per_category",
        type=int,
        default=None,
        help="Maximum samples per ESCO category (optional, e.g., 50)",
    )
    parser.add_argument(
        "--no_stratify", action="store_true", help="Disable length-based stratification"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_viz", action="store_true", help="Skip creating visualization"
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Also save datasets in regular JSON format (in addition to JSONL)",
    )
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="Only convert input file from JSON to JSONL without creating eval split",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="Path to SentenceTransformer model for semantic classification (recommended)",
    )
    parser.add_argument(
        "--use_keywords",
        action="store_true",
        help="Use keyword-based classification instead of semantic (legacy mode)",
    )

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading training data from: {args.input}")
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file {input_path} does not exist!")
        return 1

    data = load_data(input_path)
    print(f"✅ Loaded {len(data)} training samples")

    # Validate sampling arguments
    if not args.convert_only and args.eval_size is None and args.target_ratio is None:
        print(f"❌ Error: Must specify either --eval_size or --target_ratio")
        return 1

    # If only converting format, do that and exit
    if args.convert_only:
        output_file = input_path.with_suffix(".jsonl")
        if input_path.suffix.lower() == ".jsonl":
            print(f"⚠️  Input file is already JSONL format")
            return 0

        print(f"🔄 Converting to JSONL format...")
        save_as_jsonl(data, output_file)
        print(f"✅ Converted to: {output_file}")
        print(f"✨ Conversion completed successfully!")
        return 0

    # Load embedding model if using semantic classification
    model = None
    use_semantic = not args.use_keywords

    if use_semantic:
        if args.model_path:
            print(f"\n🧠 Loading embedding model: {args.model_path}")
            try:
                model = SentenceTransformer(args.model_path)
                print(f"✅ Model loaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not load model: {e}")
                print(f"   Falling back to keyword-based classification")
                use_semantic = False
        else:
            print(f"\n⚠️  No model specified, using keyword-based classification")
            print(f"   Use --model_path for better semantic classification")
            use_semantic = False

    # Create evaluation dataset
    print(f"\n🚀 Creating evaluation dataset...")
    eval_samples, train_samples, eval_stats = create_representative_eval_dataset(
        data,
        eval_size=args.eval_size,
        target_ratio=args.target_ratio,
        min_samples_per_category=args.min_per_category,
        max_samples_per_category=args.max_per_category,
        stratify_by_length=not args.no_stratify,
        seed=args.seed,
        model=model,
        use_semantic=use_semantic,
    )

    # Save results
    print(f"\n💾 Saving datasets...")
    eval_file, train_file, stats_file = save_datasets(
        eval_samples, train_samples, eval_stats, args.output_dir, args.save_json
    )

    print(f"   📄 Evaluation dataset: {eval_file}")
    print(f"   📄 Training dataset: {train_file}")
    print(f"   📄 Statistics: {stats_file}")

    # Create visualization
    if not args.no_viz:
        print(f"\n📊 Creating visualization...")
        create_visualization(eval_samples, train_samples, eval_stats, args.output_dir)

    # Print summary
    print(f"\n📈 SUMMARY:")
    print(f"   ✅ Evaluation samples: {len(eval_samples)}")
    print(f"   ✅ Training samples: {len(train_samples)}")
    print(
        f"   ✅ Categories covered: {len([c for c in eval_stats['category_distribution'] if eval_stats['category_distribution'][c]['selected'] > 0])}"
    )
    print(
        f"   ✅ Eval/Total ratio: {eval_stats['dataset_sizes']['eval_percentage']:.1f}%"
    )

    print(f"\n✨ Evaluation dataset creation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
