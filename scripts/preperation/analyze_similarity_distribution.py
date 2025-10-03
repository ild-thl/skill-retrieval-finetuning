#!/usr/bin/env python3
"""
Diagnostic tool to analyze similarity score distributions from embedding models.

This tool helps calibrate semantic similarity thresholds by analyzing the actual
score ranges produced by your embedding model.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_skill_docs(csv_path: Path) -> Dict[str, Dict]:
    """Load ESCO skill documents from CSV file."""
    import csv as csv_module

    logger.info(f"Loading skill documents from {csv_path}")

    def parse_occupations(raw: str) -> list:
        if not raw:
            return []
        raw = raw.strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [item.strip() for item in raw.split("|") if item.strip()]

        titles = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    title = entry.get("title")
                    if isinstance(title, str) and title.strip():
                        titles.append(title.strip())
                elif isinstance(entry, str) and entry.strip():
                    titles.append(entry.strip())
        return titles

    skill_docs = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            label = (
                row.get("preferredLabel")
                or row.get("preferred_label")
                or row.get("title")
            )
            if not label:
                continue

            description = (row.get("description") or "").strip() or ""

            essential_occ = parse_occupations(row.get("isEssentialForOccupations", ""))
            optional_occ = parse_occupations(row.get("isOptionalForOccupations", ""))
            occupations = []
            seen = set()
            for title in essential_occ + optional_occ:
                normalized = title.strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    occupations.append(normalized)

            skill_docs[label.strip()] = {
                "description": description,
                "occupations": occupations,
            }

    logger.info(f"Loaded {len(skill_docs)} skill documents")
    return skill_docs


def analyze_similarity_distribution(
    skill_docs: Dict[str, Dict],
    model_path: str,
    num_samples: int = 500,
    output_path: Path = None,
):
    """
    Analyze the distribution of similarity scores produced by the model.

    Args:
        skill_docs: Dictionary of ESCO skill documents
        model_path: Path to the sentence transformer model
        num_samples: Number of random skill pairs to sample
        output_path: Optional path to save results
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers required. Install with: pip install sentence-transformers"
        )

    logger.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)

    # Sample random skill pairs
    skill_names = list(skill_docs.keys())
    logger.info(
        f"Sampling {num_samples} random skill pairs from {len(skill_names)} skills"
    )

    similarities = {
        "same_occupation": [],
        "different_occupation": [],
        "semantically_related": [],
        "all_pairs": [],
    }

    # Group skills by occupation
    occupation_to_skills = {}
    for skill, doc in skill_docs.items():
        for occ in doc.get("occupations", []):
            if occ not in occupation_to_skills:
                occupation_to_skills[occ] = []
            occupation_to_skills[occ].append(skill)

    # Sample pairs from same occupations
    logger.info("Computing similarities for skill pairs from same occupations...")
    occupations_with_multiple = [
        occ for occ, skills in occupation_to_skills.items() if len(skills) >= 2
    ]
    for _ in range(min(num_samples // 2, len(occupations_with_multiple) * 5)):
        occ = random.choice(occupations_with_multiple)
        skills = occupation_to_skills[occ]
        skill1, skill2 = random.sample(skills, 2)

        doc1 = skill_docs[skill1]
        doc2 = skill_docs[skill2]

        text1 = f"{skill1}: {doc1.get('description', '')}"
        text2 = f"{skill2}: {doc2.get('description', '')}"

        emb1 = model.encode([text1], show_progress_bar=False)[0]
        emb2 = model.encode([text2], show_progress_bar=False)[0]

        sim = float(np.dot(emb1, emb2))
        similarities["same_occupation"].append(sim)
        similarities["all_pairs"].append(sim)

    # Sample pairs from different occupations
    logger.info("Computing similarities for skill pairs from different occupations...")
    for _ in range(min(num_samples // 2, len(skill_names) * 2)):
        skill1, skill2 = random.sample(skill_names, 2)

        occ1 = set(skill_docs[skill1].get("occupations", []))
        occ2 = set(skill_docs[skill2].get("occupations", []))

        # Skip if they share occupations
        if len(occ1 & occ2) > 0:
            continue

        doc1 = skill_docs[skill1]
        doc2 = skill_docs[skill2]

        text1 = f"{skill1}: {doc1.get('description', '')}"
        text2 = f"{skill2}: {doc2.get('description', '')}"

        emb1 = model.encode([text1], show_progress_bar=False)[0]
        emb2 = model.encode([text2], show_progress_bar=False)[0]

        sim = float(np.dot(emb1, emb2))
        similarities["different_occupation"].append(sim)
        similarities["all_pairs"].append(sim)

    # Calculate statistics
    stats = {}
    for category, scores in similarities.items():
        if not scores:
            continue
        stats[category] = {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "percentile_10": float(np.percentile(scores, 10)),
            "percentile_25": float(np.percentile(scores, 25)),
            "percentile_50": float(np.percentile(scores, 50)),
            "percentile_75": float(np.percentile(scores, 75)),
            "percentile_90": float(np.percentile(scores, 90)),
        }

    # Print results
    print("\n" + "=" * 80)
    print("SIMILARITY SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for category, category_stats in stats.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Range: [{category_stats['min']:.4f}, {category_stats['max']:.4f}]")
        print(f"  Mean:   {category_stats['mean']:.4f} ± {category_stats['std']:.4f}")
        print(f"  Median: {category_stats['median']:.4f} (50th percentile)")
        print(f"  10th percentile: {category_stats['percentile_10']:.4f}")
        print(f"  25th percentile: {category_stats['percentile_25']:.4f}")
        print(f"  75th percentile: {category_stats['percentile_75']:.4f}")
        print(f"  90th percentile: {category_stats['percentile_90']:.4f}")

    # Recommendations
    print("\n" + "=" * 80)
    print("THRESHOLD RECOMMENDATIONS")
    print("=" * 80)

    if "same_occupation" in stats:
        # For hard negatives, we want skills that are semantically distant
        # but from the same occupation domain
        same_occ_25th = stats["same_occupation"]["percentile_25"]
        same_occ_10th = stats["same_occupation"]["percentile_10"]

        # Distance = 1 - similarity
        recommended_distance_threshold = 1 - stats["same_occupation"]["percentile_75"]

        print(f"\nFor HARD NEGATIVE selection (same occupation, but distinct):")
        print(f"  Current threshold: semantic_distance >= 0.3 (similarity <= 0.7)")
        print(
            f"  Recommended: semantic_distance >= {recommended_distance_threshold:.3f} (similarity <= {1-recommended_distance_threshold:.3f})"
        )
        print(
            f"  Rationale: This excludes the top 75% most similar skills within same occupations"
        )

        # More conservative option
        conservative_threshold = 1 - stats["same_occupation"]["percentile_50"]
        print(
            f"  Conservative: semantic_distance >= {conservative_threshold:.3f} (similarity <= {1-conservative_threshold:.3f})"
        )
        print(
            f"  Rationale: This excludes the top 50% most similar skills (median split)"
        )

    # Save results
    if output_path:
        output_data = {
            "statistics": stats,
            "model_path": model_path,
            "num_samples": len(similarities["all_pairs"]),
            "recommendations": {
                "hard_negative_distance_threshold": (
                    recommended_distance_threshold
                    if "same_occupation" in stats
                    else None
                ),
                "conservative_distance_threshold": (
                    conservative_threshold if "same_occupation" in stats else None
                ),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved analysis results to {output_path}")

    # Plot histograms if matplotlib available
    if MATPLOTLIB_AVAILABLE and any(similarities.values()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Similarity Score Distributions", fontsize=16)

        categories = ["same_occupation", "different_occupation", "all_pairs"]
        colors = ["#2ecc71", "#e74c3c", "#3498db"]

        for idx, (category, color) in enumerate(zip(categories, colors)):
            if not similarities[category]:
                continue

            ax = axes[idx // 2, idx % 2]
            ax.hist(
                similarities[category],
                bins=50,
                alpha=0.7,
                color=color,
                edgecolor="black",
            )
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Frequency")
            ax.set_title(category.replace("_", " ").title())
            ax.axvline(
                stats[category]["mean"],
                color="red",
                linestyle="--",
                label=f"Mean: {stats[category]['mean']:.3f}",
            )
            ax.axvline(
                stats[category]["median"],
                color="orange",
                linestyle="--",
                label=f"Median: {stats[category]['median']:.3f}",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplot
        if len(categories) < 4:
            fig.delaxes(axes[1, 1])

        plt.tight_layout()

        if output_path:
            plot_path = output_path.with_suffix(".png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved distribution plots to {plot_path}")

        plt.show()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze similarity score distributions from embedding models"
    )
    parser.add_argument(
        "esco_csv_path",
        type=Path,
        help="Path to ESCO skills CSV file (e.g., skills_as_documents_v120.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to sentence transformer model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of random skill pairs to sample (default: 500)",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to save analysis results (JSON)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log), format="%(levelname)s:%(name)s:%(message)s"
    )

    skill_docs = load_skill_docs(args.esco_csv_path)

    stats = analyze_similarity_distribution(
        skill_docs=skill_docs,
        model_path=args.model_path,
        num_samples=args.num_samples,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    exit(main())
