#!/usr/bin/env python3
"""
🎯 Model Evaluation Script for ESCO Skill Retrieval

This script evaluates fine-tuned embedding models using the representative
evaluation dataset created by create_evaluation_dataset.py.

Usage:
    python evaluate_model.py --model_path ./multilingual_finetuned_esco_optimized --eval_data data/eval_split/eval_dataset_representative.jsonl
    python evaluate_model.py --model_path intfloat/multilingual-e5-base --eval_data data/evaluation.json

Features:
    - Supports both JSON and JSONL input formats (auto-detection)
    - Embedding-based similarity evaluation
    - Ranking metrics (MRR, NDCG, Recall@K)
    - Category-wise performance analysis
    - Detailed performance breakdowns
"""

import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import time
import re

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "⚠️  sentence-transformers not available. Install with: pip install sentence-transformers"
    )


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
        elif isinstance(parsed_data, dict):
            data = [parsed_data]
        else:
            raise ValueError("JSON content is not a list or object")
    except json.JSONDecodeError:
        # If JSON parsing fails, try JSONL format
        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Warning: Invalid JSON on line {line_num}: {e}")
                    continue

    if not data:
        raise ValueError(f"No valid data found in {file_path}")

    return data


def get_esco_category_name(category_code: str) -> str:
    """Get human-readable name for ESCO category code."""
    hierarchy_data = get_esco_hierarchy_data()
    if category_code in hierarchy_data:
        description = hierarchy_data[category_code]
        # Extract the main category name (before the colon)
        return description.split(":")[0]
    return category_code


# Global cache for ESCO hierarchy embeddings
_esco_hierarchy_embeddings = None
_esco_hierarchy_labels = None


def get_esco_hierarchy_data():
    """Get ESCO hierarchy categories with German descriptions for semantic clustering."""
    return {
        # Transversal skills/competences (T1-T6)
        "T1": "Kernfähigkeiten und -kompetenzen: Grundlegende Fertigkeiten wie Lesen, Schreiben, Rechnen, digitale Grundkompetenz",
        "T2": "Denkfähigkeiten und -kompetenzen: Kritisches Denken, Problemlösung, analytisches Denken, Kreativität",
        "T3": "Fähigkeiten und Kompetenzen im Bereich Selbstmanagement: Zeitmanagement, Selbstreflexion, Eigeninitiative, Selbstorganisation",
        "T4": "Soziale und kommunikative Fähigkeiten und Kompetenzen: Teamarbeit, Kommunikation, Empathie, Konfliktlösung",
        "T5": "Körperliche und manuelle Fähigkeiten und Kompetenzen: Handwerkliche Fertigkeiten, körperliche Koordination, manuelle Geschicklichkeit",
        "T6": "Fähigkeiten und Kompetenzen für eine aktive Bürgerschaft: Bürgerkompetenz, gesellschaftliches Engagement, demokratische Teilhabe",
        # Skills (S1-S8)
        "S1": "Kommunikation, Zusammenarbeit und Kreativität: Kommunikative Fertigkeiten, Kollaboration, kreative Problemlösung",
        "S2": "Informationskompetenzen: Informationsbeschaffung, -bewertung und -verarbeitung, Medienkompetenz",
        "S3": "Unterstützung und Pflege: Betreuung, Pflege, soziale Unterstützung, Hilfeleistung",
        "S4": "Managementfähigkeiten: Führung, Projektmanagement, strategische Planung, Organisationsführung",
        "S5": "Arbeiten mit Computern: IT-Anwendung, Softwarebeherrschung, digitale Technologien",
        "S6": "Handhabung/Transport und Bewegung: Logistik, Transport, physische Handhabung, Mobilität",
        "S7": "Bau: Bauwesen, Konstruktion, Architektur, Gebäudetechnik",
        "S8": "Arbeiten mit Maschinen und Spezialausrüstungen: Maschinenbedienung, technische Ausrüstung, Spezialgeräte",
        # Education fields (00-10, 99)
        "00": "Allgemeine Bildungsprogramme und Qualifikationen: Grundbildung, allgemeine Qualifikationen",
        "01": "Erziehungswissenschaften: Pädagogik, Bildungswissenschaft, Unterrichtsmethoden",
        "02": "Kunst und Geisteswissenschaften: Literatur, Geschichte, Philosophie, Kunst, Musik",
        "03": "Sozialwissenschaften, Journalistik und Informationswissenschaft: Soziologie, Psychologie, Medien, Journalismus",
        "04": "Wirtschaft, Verwaltung und Rechtswissenschaft: Betriebswirtschaft, Volkswirtschaft, Recht, Verwaltung",
        "05": "Naturwissenschaften, Mathematik und Statistik: Physik, Chemie, Biologie, Mathematik, Statistik",
        "06": "Informations- und Kommunikationstechnologien (IKT): Informatik, Programmierung, IT-Systeme",
        "07": "Ingenieurwesen, Fertigung und Bauwesen: Maschinenbau, Elektrotechnik, Bauingenieurwesen",
        "08": "Agrarwissenschaft, Forstwissenschaft, Fischereiwirtschaft und Veterinärwissenschaft: Landwirtschaft, Forstwirtschaft, Tierheilkunde",
        "09": "Gesundheit und soziale Dienste: Medizin, Pflege, Sozialarbeit, Gesundheitswesen",
        "10": "Dienstleistungen: Service, Gastgewerbe, Tourismus, persönliche Dienstleistungen",
        "99": "Bereich nicht bekannt: Unklare oder nicht zuordenbare Bereiche",
        # Language skills (L)
        "L": "Sprachliche Fähigkeiten und Kenntnisse: Fremdsprachen, Übersetzung, sprachliche Kommunikation",
    }


def categorize_esco_topic_semantic(
    skill_label: str, model, use_cache: bool = True
) -> str:
    """
    Categorize ESCO skills using semantic similarity to ESCO hierarchy categories.

    Args:
        skill_label: The skill label to categorize
        model: The sentence transformer model to use for embeddings
        use_cache: Whether to use cached embeddings for hierarchy categories

    Returns:
        ESCO hierarchy category code (e.g., 'T1', 'S5', '06', 'L')
    """
    global _esco_hierarchy_embeddings, _esco_hierarchy_labels

    # Get hierarchy data
    hierarchy_data = get_esco_hierarchy_data()

    # Initialize or use cached embeddings for hierarchy categories
    if _esco_hierarchy_embeddings is None or not use_cache:
        hierarchy_descriptions = list(hierarchy_data.values())
        _esco_hierarchy_labels = list(hierarchy_data.keys())
        _esco_hierarchy_embeddings = model.encode(
            hierarchy_descriptions, show_progress_bar=False
        )

    # Encode the skill label
    skill_embedding = model.encode([skill_label], show_progress_bar=False)[0]

    # Calculate similarities
    similarities = np.dot(skill_embedding, _esco_hierarchy_embeddings.T)

    # Find best match
    best_match_idx = np.argmax(similarities)
    best_category = _esco_hierarchy_labels[best_match_idx]
    best_similarity = similarities[best_match_idx]

    # Optional: Set a similarity threshold (lowered to be more inclusive)
    if best_similarity < 0.15:  # Lowered threshold to reduce "unknown" classifications
        return "99"  # "Bereich nicht bekannt"

    return best_category


def analyze_query_characteristics(query: str) -> Dict:
    """Analyze characteristics of a query for performance analysis."""
    # Length categories
    char_length = len(query)
    word_count = len(query.split())

    if char_length < 100:
        length_category = "Short (50-100 chars)"
    elif char_length < 200:
        length_category = "Medium (100-200 chars)"
    elif char_length < 400:
        length_category = "Long (200-400 chars)"
    else:
        length_category = "Very Long (> 400 chars)"

    # Word count categories
    if word_count < 10:
        word_category = "Few Words (5-10)"
    elif word_count < 20:
        word_category = "Moderate Words (10-20)"
    elif word_count < 40:
        word_category = "Many Words (20-40)"
    else:
        word_category = "Very Many Words (> 40)"

    # Complexity indicators
    has_punctuation = bool(re.search(r"[.!?;:]", query))
    has_numbers = bool(re.search(r"\d", query))
    has_special_chars = bool(re.search(r"[^\w\s.!?;:,-]", query))

    # Language complexity (simple heuristic)
    complex_words = [word for word in query.split() if len(word) > 8]
    complexity_ratio = len(complex_words) / max(1, word_count)

    if complexity_ratio < 0.1:
        complexity_level = "Simple"
    elif complexity_ratio < 0.3:
        complexity_level = "Moderate"
    else:
        complexity_level = "Complex"

    return {
        "char_length": char_length,
        "word_count": word_count,
        "length_category": length_category,
        "word_category": word_category,
        "complexity_level": complexity_level,
        "has_punctuation": has_punctuation,
        "has_numbers": has_numbers,
        "has_special_chars": has_special_chars,
        "complexity_ratio": complexity_ratio,
    }


def calculate_mrr(rankings: List[int]) -> float:
    """Calculate Mean Reciprocal Rank."""
    if not rankings:
        return 0.0
    return np.mean([1.0 / rank if rank > 0 else 0.0 for rank in rankings])


def calculate_recall_at_k(rankings: List[int], k: int) -> float:
    """Calculate Recall@K."""
    if not rankings:
        return 0.0
    return np.mean([1.0 if rank > 0 and rank <= k else 0.0 for rank in rankings])


def calculate_ndcg_at_k(rankings: List[int], k: int) -> float:
    """Calculate NDCG@K (simplified version)."""
    if not rankings:
        return 0.0

    ndcgs = []
    for rank in rankings:
        if rank > 0 and rank <= k:
            # Simplified NDCG: just use position-based discounting
            dcg = 1.0 / np.log2(rank + 1)
            idcg = 1.0  # Ideal DCG for single relevant item
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)

    return np.mean(ndcgs)


def calculate_detailed_metrics(rankings: List[int]) -> Dict:
    """Calculate comprehensive metrics for a list of rankings."""
    if not rankings:
        return {}

    return {
        "count": len(rankings),
        "mrr": calculate_mrr(rankings),
        "recall_at_1": calculate_recall_at_k(rankings, 1),
        "recall_at_5": calculate_recall_at_k(rankings, 5),
        "recall_at_10": calculate_recall_at_k(rankings, 10),
        "recall_at_20": calculate_recall_at_k(rankings, 20),
        "ndcg_at_5": calculate_ndcg_at_k(rankings, 5),
        "ndcg_at_10": calculate_ndcg_at_k(rankings, 10),
        "average_rank": float(np.mean(rankings)),
        "median_rank": float(np.median(rankings)),
        "std_rank": float(np.std(rankings)),
        "min_rank": int(np.min(rankings)),
        "max_rank": int(np.max(rankings)),
        "percentile_25": float(np.percentile(rankings, 25)),
        "percentile_75": float(np.percentile(rankings, 75)),
    }


def evaluate_embeddings(model, eval_data: List[Dict], batch_size: int = 32) -> Dict:
    """
    Enhanced evaluation with detailed performance analysis.

    Returns comprehensive metrics including:
    - Overall performance
    - Query length analysis
    - Positive label count analysis
    - ESCO topic performance
    - Query complexity analysis
    """
    print(f"🔄 Evaluating {len(eval_data)} samples with detailed analysis...")

    # Filter out samples without positive labels
    valid_samples = []
    for sample in eval_data:
        pos_labels = sample.get("pos", sample.get("positive_labels", []))
        if sample.get("query", "") and pos_labels:
            valid_samples.append(sample)

    print(f"   📋 Valid samples for evaluation: {len(valid_samples)}")

    # Extract all unique labels for creating the candidate pool
    all_positive_labels = []
    all_negative_labels = []

    for sample in valid_samples:
        all_positive_labels.extend(sample.get("pos", sample.get("positive_labels", [])))
        all_negative_labels.extend(sample.get("neg", sample.get("negative_labels", [])))

    # Create unique label pool
    unique_labels = list(set(all_positive_labels + all_negative_labels))
    print(f"   📋 Created candidate pool with {len(unique_labels)} unique labels")

    # Encode all labels (candidates) in batches
    print(f"   🧮 Encoding {len(unique_labels)} candidate labels...")
    start_time = time.time()

    label_embeddings = []
    for i in range(0, len(unique_labels), batch_size):
        batch_labels = unique_labels[i : i + batch_size]
        batch_embeddings = model.encode(batch_labels, show_progress_bar=False)
        label_embeddings.append(batch_embeddings)
        if (i // batch_size + 1) % 10 == 0:
            print(
                f"      Progress: {i + len(batch_labels)}/{len(unique_labels)} labels"
            )

    label_embeddings = np.vstack(label_embeddings)
    encoding_time = time.time() - start_time
    print(f"   ✅ Label encoding completed in {encoding_time:.2f}s")

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Initialize detailed tracking
    all_rankings = []
    sample_details = []

    # Categorization tracking
    rankings_by_length = defaultdict(list)
    rankings_by_word_count = defaultdict(list)
    rankings_by_complexity = defaultdict(list)
    rankings_by_pos_count = defaultdict(list)
    rankings_by_topic = defaultdict(list)

    print(f"   🎯 Evaluating queries with detailed analysis...")
    eval_start = time.time()

    for i, sample in enumerate(valid_samples):
        if (i + 1) % 25 == 0:
            print(
                f"      Progress: {i+1}/{len(valid_samples)} ({(i+1)/len(valid_samples)*100:.1f}%)"
            )

        query = sample.get("query", "")
        positive_labels = sample.get("pos", sample.get("positive_labels", []))

        # Analyze query characteristics
        query_analysis = analyze_query_characteristics(query)

        # Analyze positive labels
        pos_count = len(positive_labels)
        if pos_count == 1:
            pos_count_category = "Single (1)"
        elif pos_count <= 3:
            pos_count_category = "Few (2-3)"
        elif pos_count <= 5:
            pos_count_category = "Several (4-5)"
        else:
            pos_count_category = "Many (6-10)"

        # Categorize by dominant ESCO topic using semantic similarity
        topic_counts = defaultdict(int)
        for pos_label in positive_labels:
            topic = categorize_esco_topic_semantic(pos_label, model)
            topic_counts[topic] += 1

        # Get dominant topic (most frequent)
        if topic_counts:
            dominant_topic = max(topic_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant_topic = "Other"

        # Encode query
        query_embedding = model.encode([query], show_progress_bar=False)[0]

        # Calculate similarities
        similarities = np.dot(query_embedding, label_embeddings.T)

        # Get ranking of all labels by similarity
        ranked_indices = np.argsort(similarities)[::-1]  # Descending order

        # Find rank of each positive label
        sample_rankings = []
        positive_details = []

        for pos_label in positive_labels:
            if pos_label in label_to_idx:
                label_idx = label_to_idx[pos_label]
                # Find rank (1-indexed)
                rank = np.where(ranked_indices == label_idx)[0]
                if len(rank) > 0:
                    rank = rank[0] + 1  # Convert to 1-indexed
                    sample_rankings.append(rank)
                    positive_details.append(
                        {
                            "label": pos_label,
                            "rank": rank,
                            "topic": categorize_esco_topic_semantic(pos_label, model),
                            "similarity": similarities[label_idx],
                        }
                    )

        if sample_rankings:
            # Use best ranking for this sample
            best_rank = min(sample_rankings)
            all_rankings.append(best_rank)

            # Store detailed sample information
            sample_detail = {
                "query": query,
                "query_analysis": query_analysis,
                "positive_count": pos_count,
                "pos_count_category": pos_count_category,
                "dominant_topic": dominant_topic,
                "best_rank": best_rank,
                "all_ranks": sample_rankings,
                "positive_details": positive_details,
            }
            sample_details.append(sample_detail)

            # Categorize rankings for analysis
            rankings_by_length[query_analysis["length_category"]].append(best_rank)
            rankings_by_word_count[query_analysis["word_category"]].append(best_rank)
            rankings_by_complexity[query_analysis["complexity_level"]].append(best_rank)
            rankings_by_pos_count[pos_count_category].append(best_rank)
            rankings_by_topic[dominant_topic].append(best_rank)

    eval_time = time.time() - eval_start
    print(f"   ✅ Evaluation completed in {eval_time:.2f}s")

    # Calculate overall metrics
    if not all_rankings:
        print("   ❌ No valid rankings found!")
        return {}

    overall_metrics = calculate_detailed_metrics(all_rankings)
    overall_metrics.update(
        {
            "num_queries": len(all_rankings),
            "num_candidates": len(unique_labels),
            "num_total_samples": len(eval_data),
            "num_valid_samples": len(valid_samples),
            "encoding_time": encoding_time,
            "eval_time": eval_time,
            "total_time": encoding_time + eval_time,
        }
    )

    # Calculate detailed breakdowns
    print(f"   📊 Calculating detailed breakdowns...")

    # Length analysis
    length_analysis = {}
    for category, rankings in rankings_by_length.items():
        if rankings:  # Only include categories with data
            length_analysis[category] = calculate_detailed_metrics(rankings)

    # Word count analysis
    word_count_analysis = {}
    for category, rankings in rankings_by_word_count.items():
        if rankings:
            word_count_analysis[category] = calculate_detailed_metrics(rankings)

    # Complexity analysis
    complexity_analysis = {}
    for category, rankings in rankings_by_complexity.items():
        if rankings:
            complexity_analysis[category] = calculate_detailed_metrics(rankings)

    # Positive count analysis
    pos_count_analysis = {}
    for category, rankings in rankings_by_pos_count.items():
        if rankings:
            pos_count_analysis[category] = calculate_detailed_metrics(rankings)

    # Topic analysis
    topic_analysis = {}
    for topic, rankings in rankings_by_topic.items():
        if rankings:
            topic_analysis[topic] = calculate_detailed_metrics(rankings)

    # Compile comprehensive results
    results = {
        "overall": overall_metrics,
        "detailed_breakdowns": {
            "by_query_length": length_analysis,
            "by_word_count": word_count_analysis,
            "by_complexity": complexity_analysis,
            "by_positive_count": pos_count_analysis,
            "by_esco_topic": topic_analysis,
        },
        "sample_details": (
            sample_details[:100] if len(sample_details) > 100 else sample_details
        ),  # Limit for file size
        "analysis_summary": {
            "total_samples_analyzed": len(sample_details),
            "categories_found": {
                "length_categories": list(length_analysis.keys()),
                "word_count_categories": list(word_count_analysis.keys()),
                "complexity_levels": list(complexity_analysis.keys()),
                "positive_count_categories": list(pos_count_analysis.keys()),
                "esco_topics": list(topic_analysis.keys()),
            },
        },
    }

    return results


def print_results(metrics: Dict, model_name: str):
    """Print comprehensive evaluation results in a formatted way."""
    print(f"\n📊 COMPREHENSIVE EVALUATION RESULTS FOR: {model_name}")
    print("=" * 80)

    if not metrics or "overall" not in metrics:
        print("❌ No metrics available!")
        return

    overall = metrics["overall"]
    breakdowns = metrics.get("detailed_breakdowns", {})

    # Overall Performance
    print(f"🎯 OVERALL PERFORMANCE:")
    print(f"   MRR (Mean Reciprocal Rank): {overall['mrr']:.4f}")
    print(
        f"   Average Rank: {overall['average_rank']:.1f} (median: {overall['median_rank']:.1f})"
    )
    print(f"   Std Dev Rank: {overall['std_rank']:.1f}")

    print(f"\n📈 RECALL METRICS:")
    print(
        f"   Recall@1:  {overall['recall_at_1']:.4f} ({overall['recall_at_1']*100:.2f}%)"
    )
    print(
        f"   Recall@5:  {overall['recall_at_5']:.4f} ({overall['recall_at_5']*100:.2f}%)"
    )
    print(
        f"   Recall@10: {overall['recall_at_10']:.4f} ({overall['recall_at_10']*100:.2f}%)"
    )
    print(
        f"   Recall@20: {overall['recall_at_20']:.4f} ({overall['recall_at_20']*100:.2f}%)"
    )

    print(f"\n📊 NDCG METRICS:")
    print(f"   NDCG@5:  {overall['ndcg_at_5']:.4f}")
    print(f"   NDCG@10: {overall['ndcg_at_10']:.4f}")

    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Queries evaluated: {overall['num_queries']}")
    print(f"   Candidate pool size: {overall['num_candidates']}")
    print(f"   Total time: {overall['total_time']:.2f}s")
    print(f"   Time per query: {overall['total_time']/overall['num_queries']:.3f}s")

    # Detailed Breakdowns
    print(f"\n" + "=" * 80)
    print(f"🔍 DETAILED PERFORMANCE ANALYSIS")
    print(f"=" * 80)

    # Query Length Analysis
    if "by_query_length" in breakdowns:
        print(f"\n📏 PERFORMANCE BY QUERY LENGTH:")
        print(
            f"   {'Category':<25} {'Count':<8} {'MRR':<8} {'Recall@1':<10} {'Recall@5':<11} {'Avg Rank':<10}"
        )
        print(f"   {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*11} {'-'*10}")

        for category, stats in breakdowns["by_query_length"].items():
            print(
                f"   {category:<25} {stats['count']:<8} {stats['mrr']:.3f}    {stats['recall_at_1']:.3f}      {stats['recall_at_5']:.3f}       {stats['average_rank']:.1f}"
            )

    # Positive Count Analysis
    if "by_positive_count" in breakdowns:
        print(f"\n🎯 PERFORMANCE BY NUMBER OF POSITIVE LABELS:")
        print(
            f"   {'Category':<20} {'Count':<8} {'MRR':<8} {'Recall@1':<10} {'Recall@5':<11} {'Avg Rank':<10}"
        )
        print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*11} {'-'*10}")

        for category, stats in breakdowns["by_positive_count"].items():
            print(
                f"   {category:<20} {stats['count']:<8} {stats['mrr']:.3f}    {stats['recall_at_1']:.3f}      {stats['recall_at_5']:.3f}       {stats['average_rank']:.1f}"
            )

    # ESCO Topic Analysis
    if "by_esco_topic" in breakdowns:
        print(f"\n🏷️  PERFORMANCE BY ESCO TOPIC:")
        print(
            f"   {'Topic':<45} {'Count':<8} {'MRR':<8} {'Recall@1':<10} {'Recall@5':<11} {'Avg Rank':<10}"
        )
        print(f"   {'-'*45} {'-'*8} {'-'*8} {'-'*10} {'-'*11} {'-'*10}")

        # Sort by count (descending) to show most common topics first
        topic_items = sorted(
            breakdowns["by_esco_topic"].items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )

        for topic_code, stats in topic_items:
            topic_name = get_esco_category_name(topic_code)
            display_name = f"{topic_code}: {topic_name}"
            print(
                f"   {display_name:<45} {stats['count']:<8} {stats['mrr']:.3f}    {stats['recall_at_1']:.3f}      {stats['recall_at_5']:.3f}       {stats['average_rank']:.1f}"
            )

    # Query Complexity Analysis
    if "by_complexity" in breakdowns:
        print(f"\n🧠 PERFORMANCE BY QUERY COMPLEXITY:")
        print(
            f"   {'Complexity':<15} {'Count':<8} {'MRR':<8} {'Recall@1':<10} {'Recall@5':<11} {'Avg Rank':<10}"
        )
        print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*10} {'-'*11} {'-'*10}")

        for complexity, stats in breakdowns["by_complexity"].items():
            print(
                f"   {complexity:<15} {stats['count']:<8} {stats['mrr']:.3f}    {stats['recall_at_1']:.3f}      {stats['recall_at_5']:.3f}       {stats['average_rank']:.1f}"
            )

    # Key Insights
    print(f"\n" + "=" * 80)
    print(f"💡 KEY INSIGHTS & RECOMMENDATIONS")
    print(f"=" * 80)

    insights = analyze_performance_insights(breakdowns)
    for insight in insights:
        print(f"   {insight}")


def analyze_performance_insights(breakdowns: Dict) -> List[str]:
    """Generate actionable insights from the performance breakdowns."""
    insights = []

    # Query Length Insights
    if "by_query_length" in breakdowns:
        length_data = breakdowns["by_query_length"]
        if len(length_data) > 1:
            # Find best and worst performing length categories
            best_length = max(length_data.items(), key=lambda x: x[1]["mrr"])
            worst_length = min(length_data.items(), key=lambda x: x[1]["mrr"])

            insights.append(
                f"📏 Query Length: {best_length[0]} performs best (MRR: {best_length[1]['mrr']:.3f}), "
                f"{worst_length[0]} performs worst (MRR: {worst_length[1]['mrr']:.3f})"
            )

            if "Short" in [best_length[0], worst_length[0]]:
                insights.append("   💡 Consider query expansion for short queries")
            if "Very Long" in [best_length[0], worst_length[0]]:
                insights.append(
                    "   💡 Consider query summarization for very long queries"
                )

    # Positive Count Insights
    if "by_positive_count" in breakdowns:
        pos_data = breakdowns["by_positive_count"]
        if "Single (1)" in pos_data and "Many (6-10)" in pos_data:
            single_mrr = pos_data["Single (1)"]["mrr"]
            many_mrr = pos_data["Many (6-10)"]["mrr"]
            if single_mrr < many_mrr:
                insights.append(
                    f"🎯 Samples with more positive labels perform better - consider data augmentation"
                )
            else:
                insights.append(
                    f"🎯 Single positive label samples perform well - model handles precision tasks effectively"
                )

    # Topic Insights
    if "by_esco_topic" in breakdowns:
        topic_data = breakdowns["by_esco_topic"]
        if len(topic_data) > 1:
            # Find best and worst performing topics
            best_topic = max(topic_data.items(), key=lambda x: x[1]["mrr"])
            worst_topic = min(topic_data.items(), key=lambda x: x[1]["mrr"])

            best_topic_name = get_esco_category_name(best_topic[0])
            worst_topic_name = get_esco_category_name(worst_topic[0])

            insights.append(
                f"🏷️  ESCO Topics: {best_topic[0]} ({best_topic_name}) performs best (MRR: {best_topic[1]['mrr']:.3f}), "
                f"{worst_topic[0]} ({worst_topic_name}) performs worst (MRR: {worst_topic[1]['mrr']:.3f})"
            )

            # Check for IT/Technology bias (S5 and 06)
            it_categories = ["S5", "06"]  # "Arbeiten mit Computern" and "IKT"
            it_performance = []

            for cat in it_categories:
                if cat in topic_data:
                    it_performance.append(topic_data[cat]["mrr"])

            if it_performance:
                avg_it_mrr = np.mean(it_performance)
                avg_overall_mrr = np.mean(
                    [stats["mrr"] for stats in topic_data.values()]
                )

                if avg_it_mrr > avg_overall_mrr * 1.2:
                    insights.append(
                        "   💡 Strong IT/Technology bias detected - consider balanced training data across domains"
                    )
                elif avg_it_mrr < avg_overall_mrr * 0.8:
                    insights.append(
                        "   💡 IT/Technology topics underperforming - may need more technical training data"
                    )

            # Check for transversal vs specific skills
            transversal_cats = [cat for cat in topic_data.keys() if cat.startswith("T")]
            skill_cats = [cat for cat in topic_data.keys() if cat.startswith("S")]
            education_cats = [cat for cat in topic_data.keys() if cat.isdigit()]

            if transversal_cats and skill_cats:
                trans_mrr = np.mean(
                    [topic_data[cat]["mrr"] for cat in transversal_cats]
                )
                skill_mrr = np.mean([topic_data[cat]["mrr"] for cat in skill_cats])

                if trans_mrr > skill_mrr * 1.1:
                    insights.append(
                        "   💡 Transversal skills (T1-T6) perform better than specific skills (S1-S8)"
                    )
                elif skill_mrr > trans_mrr * 1.1:
                    insights.append(
                        "   💡 Specific skills (S1-S8) perform better than transversal skills (T1-T6)"
                    )

    # Complexity Insights
    if "by_complexity" in breakdowns:
        complexity_data = breakdowns["by_complexity"]
        if "Simple" in complexity_data and "Complex" in complexity_data:
            simple_mrr = complexity_data["Simple"]["mrr"]
            complex_mrr = complexity_data["Complex"]["mrr"]
            if simple_mrr > complex_mrr * 1.2:
                insights.append(
                    "🧠 Model struggles with complex queries - consider query preprocessing"
                )
            elif complex_mrr > simple_mrr * 1.2:
                insights.append(
                    "🧠 Model handles complex queries well - may benefit from query enrichment"
                )

    return insights


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_results(metrics: Dict, model_name: str, output_file: Path):
    """Save comprehensive evaluation results to JSON file."""
    results = {
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_metrics": metrics.get("overall", {}),
        "detailed_breakdowns": metrics.get("detailed_breakdowns", {}),
        "analysis_summary": metrics.get("analysis_summary", {}),
        "sample_details": metrics.get("sample_details", []),
        "evaluation_metadata": {
            "script_version": "2.1_semantic_clustering",
            "esco_categorization": "semantic_similarity_based",
            "esco_hierarchy_categories": len(get_esco_hierarchy_data()),
            "analysis_features": [
                "query_length_analysis",
                "positive_count_analysis",
                "esco_semantic_topic_classification",
                "query_complexity_analysis",
                "performance_insights",
                "transversal_vs_specific_skills_analysis",
            ],
        },
    }

    # Convert NumPy types to native Python types for JSON serialization
    results = convert_numpy_types(results)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {output_file}")


def main():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers is required for model evaluation!")
        print("   Install with: pip install sentence-transformers")
        return 1

    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned embedding models for ESCO skill retrieval"
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--eval_data",
        "-e",
        type=str,
        required=True,
        help="Path to evaluation dataset (JSON or JSONL)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for results (default: auto-generated)",
    )

    args = parser.parse_args()

    # Validate inputs
    eval_data_path = Path(args.eval_data)

    if not eval_data_path.exists():
        print(f"❌ Error: Evaluation data {eval_data_path} does not exist!")
        return 1

    # Load model
    print(f"🤖 Loading model: {args.model_path}")
    try:
        # Check if it's a local path or a Hugging Face model identifier
        local_path = Path(args.model_path)
        if local_path.exists():
            # Local path
            print(f"   📁 Loading from local path: {local_path}")
            model = SentenceTransformer(str(local_path))
            model_name = local_path.name
        else:
            # Assume it's a Hugging Face model identifier
            print(f"   🤗 Loading from Hugging Face: {args.model_path}")
            model = SentenceTransformer(args.model_path)
            model_name = args.model_path.replace("/", "_")
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # Load evaluation data
    print(f"📂 Loading evaluation data from: {eval_data_path}")
    eval_data = load_data(eval_data_path)
    print(f"✅ Loaded {len(eval_data)} evaluation samples")

    # Run evaluation
    print(f"\\n🚀 Starting evaluation...")
    metrics = evaluate_embeddings(model, eval_data, args.batch_size)

    # Print results
    print_results(metrics, model_name)

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        output_file = Path(f"eval_results_{safe_model_name}_{timestamp}.json")

    save_results(metrics, model_name, output_file)

    print(f"\\n✨ Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
