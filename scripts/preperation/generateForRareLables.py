#!/usr/bin/env python3
"""Generate synthetic samples for rare positive labels in bi-encoder datasets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datetime import datetime
from dotenv import load_dotenv

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

# Import embedding cache module
try:
    from embedding_cache import EmbeddingCache, create_embedding_cache
except ImportError:
    # Fallback if embedding_cache is not in path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    try:
        from embedding_cache import EmbeddingCache, create_embedding_cache
    except ImportError:
        EmbeddingCache = None
        create_embedding_cache = None


logger = logging.getLogger(__name__)


DEFAULT_SCENARIOS: Tuple[str, ...] = (
    "Weiterbildungskursbeschreibung",
    "Kursinhalte",
    "Lernzielbeschreibung",
    "Lernergebnis",
    "Jobprofil-Zusammenfassung",
    "CV-Kompetenzenabschnitt",
    "Weiterbildungsberatung",
    "Suchanfrage für Fortbildungen",
    "Praxiserfahrung/Lehren aus beruflicher Erfahrung",
)


SCENARIO_GUIDANCE: Dict[str, List[str]] = {
    "Weiterbildungskursbeschreibung": [
        "Schreibe einen kurzen Fließtext (1-2 Sätze) im Präsens, der das Angebot anschaulich beschreibt.",
        "Betone Praxisnutzen und Zielgruppe klar.",
    ],
    "Kursinhalte": [
        "Beschreibe detailiert welche Themen in dem Kurs behandelt werden.",
    ],
    "Lernzielbeschreibung": [
        "Die Überschrift lautet 'Teilnehmende lernen in diesem Kurs:'. Formuliere den darauf folgenden kruzen und prägnanten Text.",
        "Kann Fließtext mit maximal 1-3 präzisen Sätzen sein oder eine Auflistung begonnen mit '- '",
    ],
    "Lernergebnis": [
        "Nach erfolgreichem Abschluss, wird ein Zertifikat ausgestellt. Darauf steht was die Teilnehmenden gelernt haben.",
        "Formuliere den Satz so, dass klar wird, was die Teilnehmenden am Ende des Kurses können.",
    ],
    "Jobprofil-Zusammenfassung": [
        "Formuliere eine kurze Zusammenfassung (1 Satz) der Kompetenzanforderung für ein Stellenprofil.",
        "Nutze aktive Verben und verzichte auf Füllwörter.",
    ],
    "CV-Kompetenzenabschnitt": [
        "Gib genau einen Bullet-Point zurück, der mit '- ' beginnt.",
        "Schreibe im Stil eines Lebenslauf-Stichpunkts (knapp, ergebnisorientiert).",
    ],
    "Weiterbildungsberatung": [
        "Formuliere einen beratenden Satz, der eine Empfehlung ausspricht, welche Fähigkeiten gefördert werden sollen."
        "Verwende eine freundliche, motivierende Tonalität.",
    ],
    "Suchanfrage für Fortbildungen": [
        "Formuliere eine kurze Suchanfrage, die mit 'Suche nach:' beginnt.",
        "Nutze relevante Schlüsselwörter und bleibe unter 160 Zeichen.",
    ],
    "Praxiserfahrung/Lehren aus beruflicher Erfahrung": [
        "Schreibe einen reflektierenden Satz in der Ich-Form.",
        "Hebe hervor, welches Ergebnis, Fähigkeit oder welche Erkenntnis erzielt wurde.",
    ],
}


PREFIXES_TO_STRIP: Tuple[str, ...] = (
    "Teilnehmende lernen in diesem Kurs:",
    "Suche nach:",
)


@dataclass
class LabelContext:
    """Container holding contextual information for a skill label."""

    label: str
    current_queries: Sequence[str]
    skill_description: Optional[str]
    related_occupations: Sequence[str]


@dataclass
class SyntheticSample:
    """Representation of a generated sample ready to be saved."""

    query: str
    pos: List[str]
    neg: List[str]
    source: str
    scenario: str
    label: str

    def to_record(self) -> Dict[str, object]:
        record = {
            "query": self.query,
            "pos": self.pos,
            "neg": self.neg,
        }
        record["meta"] = {
            "source": self.source,
            "label": self.label,
            "scenario": self.scenario,
        }
        return record


class LLMSampleGenerator:
    """Wrapper around an OpenAI-compatible endpoint for text generation."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str],
        temperature: float,
        max_retries: int,
        dry_run: bool,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.dry_run = dry_run
        self.client: Optional[OpenAI]

        if dry_run:
            self.client = None
            logger.warning(
                "Dry-run enabled: no requests will be sent to the LLM endpoint."
            )
        else:
            if OpenAI is None:
                raise RuntimeError(
                    "The 'openai' package is required. Install it with `pip install openai`."
                )
            if not api_key:
                raise RuntimeError(
                    "An API key must be provided via --api-key or the OPENAI_API_KEY environment variable."
                )
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self, prompt: str, expected: int, label_hint: Optional[str] = None
    ) -> List[str]:
        """Call the LLM and parse bullet-list style responses."""
        if self.dry_run:
            logger.info(
                "Simulating LLM response for prompt (truncated): %s", prompt[:120]
            )
            hint = label_hint or "Kompetenz"
            return [f"[DryRun] Beispieltext {i+1} zu {hint}" for i in range(expected)]

        assert self.client is not None  # for type checkers
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": "Du bist eine Schreibhilfe für deutsche Bildungstexte.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=512,
                )
                text = response.choices[0].message.content or ""
                return self._extract_bullets(text)
            except Exception as exc:  # pragma: no cover - network errors
                logger.warning(
                    "LLM call failed on attempt %s/%s: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= 1.5
        return []

    @staticmethod
    def _extract_bullets(text: str) -> List[str]:
        """Split bullet or newline separated responses into clean strings."""
        candidates: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = stripped.lstrip("-•*1234567890. ")
            if stripped:
                candidates.append(stripped)
        return candidates


def load_dataset(dataset_path: Path) -> List[Dict[str, object]]:
    """Load dataset from JSON or JSONL file."""
    with dataset_path.open("r", encoding="utf-8") as handle:
        content = handle.read().strip()

    # Try to parse as regular JSON first
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of records")
        return data
    except json.JSONDecodeError:
        # If JSON parsing fails, try JSONL format
        data = []
        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue

        if not data:
            raise ValueError(f"No valid data found in {dataset_path}")

        return data


def load_skill_documents(csv_path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if csv_path is None or not csv_path.exists():
        logger.warning(
            "Skill CSV not provided or not found. Continuing without extra context."
        )
        return {}

    def parse_occupations(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        raw = raw.strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [item.strip() for item in raw.split("|") if item.strip()]

        titles: List[str] = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    title = entry.get("title")
                    if isinstance(title, str) and title.strip():
                        titles.append(title.strip())
                elif isinstance(entry, str) and entry.strip():
                    titles.append(entry.strip())
        return titles

    documents: Dict[str, Dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (
                row.get("preferredLabel")
                or row.get("preferred_label")
                or row.get("title")
            )
            if not label:
                continue

            description = (row.get("description") or "").strip() or None

            essential_occ = parse_occupations(row.get("isEssentialForOccupations"))
            optional_occ = parse_occupations(row.get("isOptionalForOccupations"))
            occupations = []
            seen = set()
            for title in essential_occ + optional_occ:
                normalized = title.strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    occupations.append(normalized)

            documents[label.strip()] = {
                "description": description,
                "occupations": occupations,
            }

    logger.info("Loaded %s skill documents from %s", len(documents), csv_path)
    return documents


def build_label_indexes(dataset: Sequence[Dict[str, object]]) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, Counter[str]],
    Dict[str, List[Tuple[int, Tuple[str, ...]]]],
    Dict[int, List[str]],
]:
    label_queries: Dict[str, List[str]] = {}
    label_negatives: Dict[str, set[str]] = {}
    label_cooccurrence: Dict[str, Counter[str]] = {}
    label_sample_groups: Dict[str, List[Tuple[int, Tuple[str, ...]]]] = {}
    record_negatives: Dict[int, List[str]] = {}

    for record_index, record in enumerate(dataset):
        positives = record.get("pos") or []
        negatives = record.get("neg") or []
        query = record.get("query")

        if not isinstance(positives, list) or not isinstance(query, str):
            continue

        pos_labels = [label for label in positives if isinstance(label, str)]
        if not pos_labels:
            continue

        unique_group = tuple(dict.fromkeys(pos_labels))
        if unique_group:
            for label in unique_group:
                label_sample_groups.setdefault(label, []).append(
                    (record_index, unique_group)
                )

        neg_values = (
            [neg for neg in negatives if isinstance(neg, str)]
            if isinstance(negatives, list)
            else []
        )
        record_negatives[record_index] = neg_values

        for label in pos_labels:
            label_queries.setdefault(label, []).append(query)
            if neg_values:
                label_negatives.setdefault(label, set()).update(neg_values)
            co_counter = label_cooccurrence.setdefault(label, Counter())
            for other in pos_labels:
                if other != label:
                    co_counter[other] += 1

    label_negatives_lists = {
        label: sorted(values) for label, values in label_negatives.items()
    }
    return (
        label_queries,
        label_negatives_lists,
        label_cooccurrence,
        label_sample_groups,
        record_negatives,
    )


def find_rare_labels(
    label_queries: Dict[str, List[str]], min_occurrences: int
) -> Dict[str, int]:
    rare: Dict[str, int] = {}
    for label, queries in label_queries.items():
        count = len(queries)
        if count < min_occurrences:
            rare[label] = count
    return rare


def analyze_skill_coverage(
    dataset: Sequence[Dict[str, object]], skill_docs: Dict[str, Dict[str, object]]
) -> Tuple[set[str], set[str], Dict[str, int]]:
    """Analyze which skills are covered in the dataset vs available in ESCO documents.

    Returns:
        - covered_skills: Skills that appear in the dataset
        - uncovered_skills: Skills from ESCO that don't appear in dataset
        - skill_frequency: How often each covered skill appears
    """
    covered_skills: set[str] = set()
    skill_frequency: Dict[str, int] = {}

    for record in dataset:
        positives = record.get("pos") or []
        if isinstance(positives, list):
            for label in positives:
                if isinstance(label, str) and label.strip():
                    skill = label.strip()
                    covered_skills.add(skill)
                    skill_frequency[skill] = skill_frequency.get(skill, 0) + 1

    available_skills = set(skill_docs.keys())
    uncovered_skills = available_skills - covered_skills

    logger.info(
        "Coverage analysis: %d covered skills, %d uncovered skills from %d total ESCO skills (%.1f%% coverage)",
        len(covered_skills),
        len(uncovered_skills),
        len(available_skills),
        (len(covered_skills) / len(available_skills) * 100) if available_skills else 0,
    )

    return covered_skills, uncovered_skills, skill_frequency


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


def categorize_skill_semantic(
    skill_label: str,
    skill_description: str,
    model,
    hierarchy_embeddings=None,
    hierarchy_labels=None,
) -> str:
    """
    Categorize skills using semantic similarity to ESCO hierarchy categories.
    Uses the same approach as evaluate_model.py for consistency.

    Args:
        skill_label: The skill name
        skill_description: The skill description
        model: The sentence transformer model
        hierarchy_embeddings: Pre-computed embeddings for hierarchy categories
        hierarchy_labels: List of hierarchy category codes

    Returns:
        ESCO hierarchy category code (e.g., 'T1', 'S5', '06', 'L')
    """
    if hierarchy_embeddings is None or hierarchy_labels is None:
        hierarchy_data = get_esco_hierarchy_data()
        hierarchy_descriptions = list(hierarchy_data.values())
        hierarchy_labels = list(hierarchy_data.keys())
        hierarchy_embeddings = model.encode(
            hierarchy_descriptions, show_progress_bar=False
        )

    # Create skill text for embedding (similar to how skills are presented in evaluation)
    skill_text = (
        f"{skill_label}: {skill_description}" if skill_description else skill_label
    )

    # Encode the skill
    skill_embedding = model.encode([skill_text], show_progress_bar=False)[0]

    # Calculate similarities
    similarities = np.dot(skill_embedding, hierarchy_embeddings.T)

    # Find best match
    best_match_idx = np.argmax(similarities)
    best_category = hierarchy_labels[best_match_idx]
    best_similarity = similarities[best_match_idx]

    # Use a similarity threshold
    if best_similarity < 0.15:
        return "99"  # "Bereich nicht bekannt"

    return best_category


def suggest_skills_for_expansion(
    uncovered_skills: set[str],
    skill_docs: Dict[str, Dict[str, object]],
    target_expansion_ratio: float = 0.2,
    current_covered_count: int = 0,
    model_path: Optional[str] = None,
    performance_priorities: Optional[Dict[str, float]] = None,
    covered_skills: Optional[set[str]] = None,
    enable_diversity: bool = False,
) -> List[Tuple[str, float]]:
    """Enhanced skill suggestion using semantic ESCO categorization and diversity.

    Prioritizes skills from underrepresented areas and those with worse performance.
    Uses semantic similarity to accurately categorize skills into ESCO hierarchy.
    Favors skills that are most diverse compared to already covered skills.

    Args:
        uncovered_skills: Set of skills not covered in current dataset
        skill_docs: ESCO skill documents with descriptions
        target_expansion_ratio: Target expansion ratio for coverage
        current_covered_count: Number of currently covered skills
        model_path: Path to sentence transformer model for semantic categorization
        performance_priorities: Dict mapping ESCO categories to priority multipliers
                               (higher values = higher priority for expansion)
        covered_skills: Set of currently covered skills (for diversity calculation)
        enable_diversity: Whether to enable diversity calculation (may be slow)

    Returns:
        List of (skill_name, priority_score) tuples sorted by priority
    """
    if not uncovered_skills or not skill_docs:
        return []

    target_additional_skills = int(current_covered_count * target_expansion_ratio)
    if target_additional_skills == 0:
        target_additional_skills = max(10, len(uncovered_skills) // 20)

    # Define performance-based priorities based on evaluation results
    # Higher values = higher priority for expansion (underrepresented or worse performing areas)
    if performance_priorities is None:
        performance_priorities = {
            # Underrepresented areas (from evaluation analysis)
            "00": 5.0,  # Allgemeine Bildungsprogramme - underrepresented
            "T5": 4.5,  # Körperliche und manuelle Fähigkeiten - underrepresented
            "01": 4.0,  # Erziehungswissenschaften - underrepresented
            "04": 4.0,  # Wirtschaft, Verwaltung - underrepresented
            "S5": 4.0,  # Arbeiten mit Computern - underrepresented
            "02": 3.5,  # Kunst und Geisteswissenschaften - underrepresented
            "10": 3.5,  # Dienstleistungen - underrepresented
            "T2": 3.5,  # Denkfähigkeiten - underrepresented
            "05": 3.0,  # Naturwissenschaften - underrepresented
            # Areas with worse performance (many samples but could improve)
            "T6": 3.5,  # Aktive Bürgerschaft - worst performance (MRR: 0.653)
            "S2": 3.0,  # Informationskompetenzen - poor performance (MRR: 0.708)
            "S4": 2.5,  # Managementfähigkeiten - poor performance (MRR: 0.711)
            # Well-performing areas (lower priority)
            "L": 1.5,  # Language skills - excellent performance
            "08": 1.0,  # Agriculture - perfect performance
            "T3": 1.0,  # Self-management - perfect performance
            "09": 1.0,  # Health services - perfect performance
            "S7": 2.0,  # Construction - good performance
            "S1": 2.0,  # Communication - decent performance
            "S3": 2.0,  # Care and support - decent performance
            "S6": 2.0,  # Handling/Transport - decent performance
            "S8": 2.0,  # Machine operation - decent performance
            "T4": 2.0,  # Social/communication skills - decent performance
            "03": 2.0,  # Social sciences - decent performance
            "06": 2.0,  # ICT - decent performance
            "07": 2.0,  # Engineering - decent performance
            "99": 1.0,  # Unknown - lowest priority
        }

    # Load model for semantic categorization if provided
    model = None
    hierarchy_embeddings = None
    hierarchy_labels = None

    if model_path:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_path)

            # Pre-compute hierarchy embeddings for efficiency
            hierarchy_data = get_esco_hierarchy_data()
            hierarchy_descriptions = list(hierarchy_data.values())
            hierarchy_labels = list(hierarchy_data.keys())
            hierarchy_embeddings = model.encode(
                hierarchy_descriptions, show_progress_bar=False
            )

            logger.info(f"Loaded model {model_path} for semantic ESCO categorization")
        except ImportError:
            logger.warning(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
            logger.warning("Falling back to keyword-based categorization.")
            model = None
        except Exception as e:
            logger.warning(
                f"Could not load model {model_path}: {e}. Falling back to keyword-based categorization."
            )
            model = None

    skill_priorities: List[Tuple[str, float]] = []
    category_counts = {}  # Track how many skills we're suggesting per category

    # Pre-compute covered skill embeddings for efficiency (if using diversity calculation)
    covered_embeddings_cache = None
    if covered_skills and model is not None and enable_diversity:
        logger.info(
            "Pre-computing embeddings for %d covered skills for diversity calculation...",
            len(covered_skills),
        )
        # Sample a representative subset of covered skills
        sample_size = min(30, len(covered_skills))  # Further reduced for speed
        sampled_covered = random.sample(list(covered_skills), sample_size)
        covered_embeddings_cache = model.encode(
            list(sampled_covered), show_progress_bar=False
        )
        logger.info("Completed pre-computing covered skill embeddings")

    logger.info("Analyzing %d uncovered skills...", len(uncovered_skills))
    processed_count = 0

    for skill in uncovered_skills:
        processed_count += 1
        if processed_count % 1000 == 0:  # Progress update every 1000 skills
            logger.info(
                "Progress: %d/%d skills analyzed (%.1f%%)",
                processed_count,
                len(uncovered_skills),
                (processed_count / len(uncovered_skills)) * 100,
            )

        doc = skill_docs.get(skill, {})
        skill_description = doc.get("description", "")
        occupations = (
            doc.get("occupations", [])
            if isinstance(doc.get("occupations"), list)
            else []
        )

        # Filter 1: Skip skills with fewer than 3 related occupations
        if len(occupations) < 3:
            continue

        # Filter 2: Skip skills with very short descriptions (less than 50 characters)
        if len(skill_description) < 50:
            continue

        # Filter 3: Lower priority for circus, religious, and military skills
        skill_lower = skill.lower()
        desc_lower = skill_description.lower() if skill_description else ""
        text_combined = f"{skill_lower} {desc_lower}"

        # Check for circus-related skills
        circus_keywords = [
            "zirkus",
            "circus",
            "akrobat",
            "akrobatik",
            "artist",
            "varieté",
            "manege",
            "dompteur",
            "clown",
            "jongleur",
            "jonglieren",
            "seiltanz",
            "trapez",
        ]
        is_circus = any(keyword in text_combined for keyword in circus_keywords)

        # Check for religious/spiritual skills
        religious_keywords = [
            "kirche",
            "religion",
            "religiös",
            "geistlich",
            "pastor",
            "pfarrer",
            "priester",
            "imam",
            "rabbi",
            "mönch",
            "nonne",
            "kloster",
            "gottesdienst",
            "predigt",
            "seelsorge",
            "liturgie",
            "sakrament",
            "bibel",
            "koran",
            "torah",
            "spirituell",
        ]
        is_religious = any(keyword in text_combined for keyword in religious_keywords)

        # Check for military skills
        military_keywords = [
            "militär",
            "armee",
            "soldat",
            "bundeswehr",
            "offizier",
            "marine",
            "luftwaffe",
            "heer",
            "kaserne",
            "waffe",
            "kampf",
            "krieg",
            "verteidigung",
            "strategie militär",
            "einsatz militär",
            "truppe",
            "regiment",
            "bataillon",
            "kommando",
        ]
        is_military = any(keyword in text_combined for keyword in military_keywords)

        priority_score = 0.0

        # Factor 1: Has description (essential for quality generation)
        if skill_description:
            priority_score += 2.0
        else:
            priority_score += 0.3  # Some value for name-only skills

        # Factor 2: Number of related occupations (broader applicability)
        occupation_count = len(occupations)
        if occupation_count >= 10:
            priority_score += 2.0  # High applicability
        elif occupation_count >= 6:
            priority_score += 1.5  # Good applicability
        elif occupation_count >= 3:
            priority_score += 1.0  # Moderate applicability
        # Note: Skills with < 3 occupations and < 50 char descriptions are already filtered out

        # Factor 3: Description quality bonus (longer descriptions = better training data)
        description_length = len(skill_description)
        if description_length >= 200:
            priority_score += 1.0  # Excellent detail
        elif description_length >= 120:
            priority_score += 0.6  # Good detail
        elif description_length >= 80:
            priority_score += 0.3  # Adequate detail
        # Note: Descriptions < 50 chars are already filtered out

        # Factor 4: Apply penalties for specialized domains
        if is_circus:
            priority_score *= 0.3  # Strong penalty for circus skills
        elif is_religious:
            priority_score *= 0.4  # Strong penalty for religious skills
        elif is_military:
            priority_score *= 0.5  # Moderate penalty for military skills

        # Factor 5: Semantic ESCO categorization (main priority factor)
        if model is not None:
            category = categorize_skill_semantic(
                skill, skill_description, model, hierarchy_embeddings, hierarchy_labels
            )
        else:
            # Fallback to basic keyword matching if no model available
            category = categorize_skill_fallback(skill, skill_description)

        # Apply performance-based priority multiplier
        performance_multiplier = performance_priorities.get(category, 1.0)
        priority_score *= performance_multiplier

        # Factor 6: Diversity bonus (favor skills different from already covered ones)
        if covered_skills and model is not None and enable_diversity:
            diversity_score = calculate_skill_diversity(
                skill,
                skill_description,
                covered_skills,
                model,
                covered_embeddings_cache,
            )
            priority_score += diversity_score * 1.5  # Diversity bonus up to 1.5 points

        # Track category distribution
        category_counts[category] = category_counts.get(category, 0) + 1

        skill_priorities.append((skill, priority_score, category))

    # Sort by priority score (descending)
    skill_priorities.sort(key=lambda x: x[1], reverse=True)

    # Balance selection across priority categories to avoid over-concentration
    final_suggestions: List[Tuple[str, float]] = []
    category_limits = {}

    # Set limits per category based on priority
    for category, multiplier in performance_priorities.items():
        if multiplier >= 4.0:  # High priority categories
            category_limits[category] = max(5, target_additional_skills // 4)
        elif multiplier >= 3.0:  # Medium priority categories
            category_limits[category] = max(3, target_additional_skills // 6)
        else:  # Lower priority categories
            category_limits[category] = max(2, target_additional_skills // 10)

    category_selected = {}

    for skill, score, category in skill_priorities:
        if (
            len(final_suggestions) >= target_additional_skills * 2
        ):  # Provide more options
            break

        current_count = category_selected.get(category, 0)
        limit = category_limits.get(category, 3)

        if current_count < limit:
            final_suggestions.append((skill, score))
            category_selected[category] = current_count + 1

    # If we don't have enough suggestions, fill remaining slots from top remaining skills
    remaining_needed = max(0, target_additional_skills - len(final_suggestions))
    if remaining_needed > 0:
        remaining_skills = [
            (skill, score)
            for skill, score, category in skill_priorities
            if skill not in [s[0] for s in final_suggestions]
        ]
        final_suggestions.extend(remaining_skills[:remaining_needed])

    # Log category distribution
    final_categories = {}
    for skill, score in final_suggestions:
        # Find category for this skill
        for s, sc, cat in skill_priorities:
            if s == skill:
                final_categories[cat] = final_categories.get(cat, 0) + 1
                break

    logger.info(
        "Suggested %d skills for expansion (target: %d for %.0f%% growth)",
        len(final_suggestions),
        target_additional_skills,
        target_expansion_ratio * 100,
    )

    logger.info("Category distribution in suggestions:")
    for category, count in sorted(
        final_categories.items(),
        key=lambda x: performance_priorities.get(x[0], 0),
        reverse=True,
    ):
        category_name = get_esco_hierarchy_data().get(category, category)
        priority = performance_priorities.get(category, 1.0)
        logger.info(
            f"  {category}: {count} skills (priority: {priority:.1f}) - {category_name[:50]}..."
        )

    return final_suggestions


def calculate_skill_diversity(
    skill: str,
    skill_description: str,
    covered_skills: set[str],
    model=None,
    covered_embeddings_cache=None,
) -> float:
    """
    Calculate how diverse/different a skill is compared to already covered skills.
    Returns a diversity score (0-1.0) where higher values mean more diverse.
    Uses cached embeddings for efficiency.
    """
    if not covered_skills or model is None:
        return 0.5  # Default moderate diversity score if no comparison possible

    try:
        # Create skill representation
        skill_text = f"{skill}: {skill_description}" if skill_description else skill
        skill_embedding = model.encode([skill_text], show_progress_bar=False)[0]

        # Use cached covered embeddings if available
        if covered_embeddings_cache is not None:
            covered_embeddings = covered_embeddings_cache
        else:
            # Sample a smaller subset of covered skills for comparison (for efficiency)
            sample_size = min(20, len(covered_skills))  # Reduced from 50 to 20
            sampled_covered = random.sample(list(covered_skills), sample_size)
            covered_embeddings = model.encode(
                list(sampled_covered), show_progress_bar=False
            )

        # Calculate similarities to existing skills
        similarities = np.dot(skill_embedding, covered_embeddings.T)

        # Diversity score is inverse of maximum similarity
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
        diversity_score = 1.0 - min(max_similarity, 1.0)

        return max(0.0, min(diversity_score, 1.0))

    except Exception as e:
        # If calculation fails, return moderate diversity
        return 0.5


def categorize_skill_fallback(skill: str, description: str) -> str:
    """Fallback keyword-based categorization when no model is available."""
    text = f"{skill} {description}".lower()

    # Simple keyword mapping to major categories
    if any(
        kw in text for kw in ["computer", "software", "programming", "digital", "it"]
    ):
        return "S5"
    elif any(
        kw in text for kw in ["management", "führung", "leitung", "projektmanagement"]
    ):
        return "S4"
    elif any(kw in text for kw in ["kommunikation", "sprache", "language"]):
        return "L"
    elif any(kw in text for kw in ["gesundheit", "medizin", "pflege"]):
        return "09"
    elif any(kw in text for kw in ["mathematik", "physik", "chemie", "wissenschaft"]):
        return "05"
    elif any(kw in text for kw in ["kunst", "design", "kreativ"]):
        return "02"
    elif any(kw in text for kw in ["bildung", "unterricht", "lehre"]):
        return "01"
    elif any(kw in text for kw in ["wirtschaft", "recht", "verwaltung"]):
        return "04"
    else:
        return "99"


def select_focus_labels(
    primary_label: str,
    deficits: Dict[str, int],
    label_sample_groups: Dict[str, List[Tuple[int, Tuple[str, ...]]]],
    max_group_size: int,
) -> Tuple[List[str], Tuple[str, ...], Optional[int]]:
    groups = label_sample_groups.get(primary_label, [])
    if not groups:
        return [primary_label], (primary_label,), None

    max_group_size = max(1, max_group_size)

    chosen_group: Optional[Tuple[str, ...]] = None
    chosen_record_index: Optional[int] = None
    best_score = (-1, -1)
    for record_index, labels in groups:
        unique_labels = tuple(dict.fromkeys(labels))
        other_deficits = sum(
            1
            for label in unique_labels
            if label != primary_label and deficits.get(label, 0) > 0
        )
        score = (other_deficits, len(unique_labels))
        if score > best_score:
            chosen_group = unique_labels
            best_score = score
            chosen_record_index = record_index

    if chosen_group is None:
        first_record_index, first_labels = groups[0]
        chosen_group = tuple(dict.fromkeys(first_labels))
        chosen_record_index = first_record_index

    selectable_labels = [label for label in chosen_group if deficits.get(label, 0) > 0]
    if primary_label not in selectable_labels:
        selectable_labels.insert(0, primary_label)

    focus_labels = [primary_label]
    others = [label for label in selectable_labels if label != primary_label]
    random.shuffle(others)
    max_extra = min(len(others), max_group_size - 1)
    if max_extra > 0:
        extra_count = max_extra if max_extra == 1 else random.randint(1, max_extra)
        focus_labels.extend(others[:extra_count])

    focus_labels = list(dict.fromkeys(focus_labels))
    return focus_labels, chosen_group, chosen_record_index


def select_negative_labels(
    focus_labels: Sequence[str],
    unused_group_labels: Sequence[str],
    record_negative_labels: Sequence[str],
    max_negatives: int,
) -> List[str]:
    max_negatives = max(0, max_negatives)
    prioritized: List[str] = []
    seen: set[str] = set()

    for label in unused_group_labels:
        if not isinstance(label, str):
            continue
        if label in focus_labels or label in seen:
            continue
        prioritized.append(label)
        seen.add(label)
        if len(prioritized) >= max_negatives:
            return prioritized[:max_negatives]

    for label in record_negative_labels:
        if not isinstance(label, str):
            continue
        if label in focus_labels or label in seen:
            continue
        prioritized.append(label)
        seen.add(label)
        if len(prioritized) >= max_negatives:
            break

    return prioritized[:max_negatives]


def build_prompt(
    label_context: LabelContext,
    scenario: str,
    focus_labels: Sequence[str],
    language: str,
    max_chars: int,
    avoid_terms: Sequence[str],
) -> str:
    existing_snippets = (
        "\n".join(f"- {q[:180]}" for q in label_context.current_queries[:2])
        or "- [keine Beispieldaten verfügbar]"
    )
    # Sample maximum of 3 occupations randomly if more than 3 are available
    available_occupations = label_context.related_occupations
    if len(available_occupations) > 3:
        sampled_occupations = random.sample(available_occupations, 3)
    else:
        sampled_occupations = available_occupations
    occupations = ", ".join(sampled_occupations) or "(keine Angaben)"
    description = label_context.skill_description or "(keine Beschreibung verfügbar)"

    focus_text = ", ".join(sorted(dict.fromkeys(focus_labels)))
    avoid_text = ", ".join(sorted({term for term in avoid_terms if term}))

    target_length = max(150, max_chars - 100)
    instructions: List[str] = [
        f"Erzeuge genau einen kurzen deutschen Text (<= {target_length} Zeichen) und nutze höchstens zwei Sätze.",
        f"Der Text muss die Kompetenzen {focus_text} klar hervorheben.",
        f"Verwende die Perspektive '{scenario}' und passe den Stil an.",
        "Formuliere natürlich, ohne Markdown, Aufzählungen oder Fettdruck.",
        "Der Satz soll als Trainingsbeispiel für semantische Suche dienen (prägnant, faktenorientiert).",
    ]

    scenario_guidance = SCENARIO_GUIDANCE.get(scenario, [])
    instructions.extend(scenario_guidance)

    instructions.extend(
        [
            "Gib nur den fertigen Text ohne zusätzliche Einordnung zurück.",
            "Vermeide Wiederholungen und halte den Text natürlich klingend.",
        ]
    )

    if avoid_text:
        instructions.append(f"Erwähne auf keinen Fall: {avoid_text}.")

    prompt = "\n".join(
        [
            # f"Sprache: {language}",
            # "Kontext aus vorhandenen Datensätzen:",
            # existing_snippets,
            "\nESCO-Beschreibung:",
            description,
            "\nVerwandte Berufe:",
            occupations,
            "\nAufgabe:",
            *instructions,
        ]
    )
    return prompt


def strip_known_prefixes(text: str) -> str:
    for prefix in PREFIXES_TO_STRIP:
        if text.startswith(prefix):
            return text[len(prefix) :].lstrip()
    return text


def enforce_constraints(
    candidates: Iterable[str],
    max_length: int,
    existing_texts: Sequence[str],
    forbidden_terms: Sequence[str],
) -> List[str]:
    approved: List[str] = []
    seen_lower = {text.lower() for text in existing_texts}
    forbidden_lower = {term.lower() for term in forbidden_terms if term}

    for candidate in candidates:
        normalized = strip_known_prefixes(candidate.strip())
        if not normalized:
            continue
        normalized_lower = normalized.lower()
        if not normalized:
            continue
        if len(normalized) > max_length:
            trimmed = normalized[:max_length].rstrip()
            if " " in trimmed:
                trimmed = trimmed.rsplit(" ", 1)[0]
            trimmed = trimmed.rstrip(",;:-")
            if trimmed and len(trimmed) >= max_length * 0.6:
                candidate_after_trim = trimmed
                if not candidate_after_trim.endswith("."):
                    candidate_after_trim += "."
                normalized = candidate_after_trim.strip()
                normalized_lower = normalized.lower()
            if len(normalized) > max_length:
                logger.warning(
                    "Candidate rejected for length (%s > %s): %s",
                    len(normalized),
                    max_length,
                    normalized,
                )
                continue
        if forbidden_lower and any(
            term in normalized_lower for term in forbidden_lower
        ):
            logger.warning("Candidate rejected due to forbidden term: %s", normalized)
            continue
        if normalized_lower in seen_lower:
            logger.warning("Candidate rejected as duplicate: %s", normalized)
            continue
        seen_lower.add(normalized_lower)
        approved.append(normalized)
    return approved


def build_label_context(
    label: str,
    existing_queries: Sequence[str],
    skill_docs: Dict[str, Dict[str, object]],
) -> LabelContext:
    doc = skill_docs.get(label, {})
    return LabelContext(
        label=label,
        current_queries=existing_queries,
        skill_description=(
            doc.get("description") if isinstance(doc.get("description"), str) else None
        ),
        related_occupations=(
            doc.get("occupations") if isinstance(doc.get("occupations"), list) else []
        ),
    )


def write_samples(path: Path, samples: Sequence[SyntheticSample]) -> None:
    output = [sample.to_record() for sample in samples]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    logger.info("Saved %s synthetic samples to %s", len(samples), path)


def log_prompt(
    prompt_log_path: Optional[Path],
    *,
    label: str,
    scenario: str,
    attempt: int,
    focus_labels: Sequence[str],
    avoid_terms: Sequence[str],
    prompt: str,
) -> None:
    if prompt_log_path is None:
        return

    try:
        with prompt_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "=" * 80,
                        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                        f"label: {label}",
                        f"scenario: {scenario}",
                        f"attempt: {attempt}",
                        f"focus_labels: {', '.join(focus_labels)}",
                        f"forbidden_terms: {', '.join(avoid_terms) if avoid_terms else '—'}",
                        "prompt:",
                        prompt,
                        "",
                    ]
                )
            )
    except Exception as exc:  # pragma: no cover - logging should not break generation
        logger.warning("Could not write prompt log: %s", exc)


def find_cooccurring_skills_by_occupation(
    target_skill: str,
    skill_docs: Dict[str, Dict[str, object]],
    min_shared_occupations: int = 1,
    max_results: int = 10,
) -> List[Tuple[str, int]]:
    """Find skills that co-occur with the target skill in the same occupations.

    Args:
        target_skill: The skill to find co-occurring skills for
        skill_docs: ESCO skill documents
        min_shared_occupations: Minimum number of shared occupations required
        max_results: Maximum number of co-occurring skills to return

    Returns:
        List of (skill_name, shared_occupation_count) tuples, sorted by overlap
    """
    target_doc = skill_docs.get(target_skill, {})
    target_occupations = set(target_doc.get("occupations", []))

    if not target_occupations:
        return []

    cooccurrence_scores: List[Tuple[str, int]] = []

    for skill, doc in skill_docs.items():
        if skill == target_skill:
            continue

        skill_occupations = set(doc.get("occupations", []))
        shared_occupations = target_occupations & skill_occupations
        overlap_count = len(shared_occupations)

        if overlap_count >= min_shared_occupations:
            cooccurrence_scores.append((skill, overlap_count))

    # Sort by overlap count (descending) and return top results
    cooccurrence_scores.sort(key=lambda x: x[1], reverse=True)
    return cooccurrence_scores[:max_results]


def find_smart_hard_negatives(
    positive_skills: List[str],
    skill_docs: Dict[str, Dict[str, object]],
    covered_skills: set[str],
    max_negatives: int = 2,
    model=None,
    embedding_cache=None,
) -> List[str]:
    """Find hard negatives from the same occupations but semantically distant.

    NEW Strategy for smart hard negatives:
    1. Sample from the SAME occupations as positive skills (domain relevance)
    2. Ensure semantic distance to avoid synonyms/similar skills
    3. Use lexical distance as fallback when no model available
    4. Prefer skills with occupation overlap for harder negatives

    This creates challenging negatives that are domain-relevant but clearly different,
    which is more realistic and useful for training than completely unrelated negatives.

    Args:
        positive_skills: List of positive skill labels
        skill_docs: ESCO skill documents
        covered_skills: Set of well-covered skills (used as preference, not requirement)
        max_negatives: Maximum number of negatives to return
        model: Optional sentence transformer model for semantic distance calculation
        embedding_cache: Optional EmbeddingCache instance for faster embedding retrieval

    Returns:
        List of hard negative skill names
    """
    # Get all occupations for positive skills
    positive_occupations = set()
    positive_skill_words = set()
    positive_skill_normalized = set()  # Store normalized forms for substring matching
    positive_skill_embeddings = []

    for skill in positive_skills:
        doc = skill_docs.get(skill, {})
        occupations = doc.get("occupations", [])
        positive_occupations.update(occupations)

        # Extract key words from skill name (for lexical distance fallback)
        skill_words = set(skill.lower().split())
        positive_skill_words.update(skill_words)

        # Store normalized version (lowercase, remove punctuation) for substring matching
        normalized = (
            skill.lower().replace("(", "").replace(")", "").replace("/", " ").strip()
        )
        positive_skill_normalized.add(normalized)
        # Also add individual significant words (length > 3) for matching
        significant_words = {w for w in normalized.split() if len(w) > 3}
        positive_skill_normalized.update(significant_words)

    if not positive_occupations:
        # Fallback: if no occupations, return empty
        logger.debug(
            "No occupations found for positive skills, cannot find hard negatives"
        )
        return []

    # Compute embeddings for positive skills if model available
    if model is not None or embedding_cache is not None:
        try:
            positive_skill_names = []
            positive_descriptions = []
            for skill in positive_skills:
                doc = skill_docs.get(skill, {})
                desc = doc.get("description", "")
                positive_skill_names.append(skill)
                positive_descriptions.append(desc if isinstance(desc, str) else "")

            # Use cache if available, otherwise encode directly
            if embedding_cache is not None:
                positive_skill_embeddings = embedding_cache.get_embeddings(
                    positive_skill_names, positive_descriptions
                )
            else:
                positive_texts = [
                    f"{skill}: {desc}" if desc else skill
                    for skill, desc in zip(positive_skill_names, positive_descriptions)
                ]
                positive_skill_embeddings = model.encode(
                    positive_texts, show_progress_bar=False
                )
        except Exception as e:
            logger.debug(f"Could not compute embeddings for positive skills: {e}")
            positive_skill_embeddings = []

    # Find candidate negatives from the SAME occupations
    negative_candidates: List[Tuple[str, float]] = []

    for skill, doc in skill_docs.items():
        if skill in positive_skills:
            continue

        skill_occupations = set(doc.get("occupations", []))

        # Calculate overlap with positive skills' occupations
        occupation_overlap = len(positive_occupations & skill_occupations)

        # REQUIRE at least 1 shared occupation (domain relevance)
        if occupation_overlap < 1:
            continue

        # Calculate lexical similarity (for filtering obvious synonyms)
        skill_words = set(skill.lower().split())
        word_overlap = len(positive_skill_words & skill_words)

        # Skip if too lexically similar (likely synonyms)
        if word_overlap > 2:
            continue

        # NEW: Check for substring/token overlap with positive skills
        # Normalize candidate skill for comparison
        candidate_normalized = (
            skill.lower().replace("(", "").replace(")", "").replace("/", " ").strip()
        )
        candidate_words = {w for w in candidate_normalized.split() if len(w) > 3}

        # # Skip if candidate contains significant words from positive labels
        # # This catches cases like "Computerprogrammierung" appearing in "Python (Computerprogrammierung)"
        # significant_overlap = len(candidate_words & positive_skill_normalized)
        # if significant_overlap > 0:
        #     logger.debug(
        #         f"Skipping '{skill}' as hard negative: shares significant words with positive labels"
        #     )
        #     continue

        # # Also check if any positive skill normalized form is a substring of candidate or vice versa
        # skip_substring = False
        # for pos_norm in positive_skill_normalized:
        #     if len(pos_norm) > 5:  # Only check meaningful strings
        #         if pos_norm in candidate_normalized or candidate_normalized in pos_norm:
        #             logger.debug(
        #                 f"Skipping '{skill}' as hard negative: substring overlap with positive labels"
        #             )
        #             skip_substring = True
        #             break
        # if skip_substring:
        #     continue

        # Calculate semantic distance if model available
        semantic_distance = 0.0
        if (model is not None or embedding_cache is not None) and len(
            positive_skill_embeddings
        ) > 0:
            try:
                desc = doc.get("description", "")

                # Use cache if available, otherwise encode directly
                if embedding_cache is not None:
                    candidate_embedding = embedding_cache.get_embedding(
                        skill, desc if isinstance(desc, str) else ""
                    )
                else:
                    candidate_text = f"{skill}: {desc}" if desc else skill
                    candidate_embedding = model.encode(
                        [candidate_text], show_progress_bar=False
                    )[0]

                # Calculate minimum similarity to any positive skill
                import numpy as np

                similarities = [
                    np.dot(candidate_embedding, pos_emb)
                    for pos_emb in positive_skill_embeddings
                ]
                max_similarity = max(similarities) if similarities else 0.0

                # Convert to distance (higher = more distant = better)
                semantic_distance = 1.0 - max_similarity

            except Exception as e:
                logger.debug(f"Could not compute semantic distance for {skill}: {e}")
                semantic_distance = 0.5  # Default moderate distance
        else:
            # Fallback: use lexical distance as proxy for semantic distance
            # Fewer shared words = more distant
            semantic_distance = 1.0 - (
                word_overlap / max(len(positive_skill_words), len(skill_words))
            )

        # Score: higher occupation overlap + higher semantic distance = better hard negative
        score = 0.0

        # Factor 1: Occupation overlap (more shared occupations = harder negative)
        score += occupation_overlap * 2.0

        # Factor 2: Semantic distance (more distant = better, but not too distant)
        # We want semantically distant skills, but still from same domain
        #
        # ADAPTIVE THRESHOLD: Calibrated based on similarity distribution analysis.
        #
        # Model: finetuned_esco_model_115640
        # Analysis results (same occupation pairs):
        #   - Mean similarity: 0.24, Median: 0.22
        #   - 75th percentile: 0.37 (top 25% most similar)
        #   - 90th percentile: 0.46
        #   - Max observed: 0.71
        #
        # This model produces LOW similarity scores compared to generic embeddings!
        # Using threshold of 0.367 (75th percentile) to exclude top 25% most similar skills.
        #
        # To recalibrate for your specific model, run:
        #   python analyze_similarity_distribution.py ../../data/ESCO/sources/skills_as_documents_v120.csv \
        #       --model-path <model> --output analysis.json
        #
        MINIMUM_DISTANCE_THRESHOLD = (
            0.63  # semantic_distance >= 0.63 means similarity <= 0.37
        )

        if semantic_distance >= MINIMUM_DISTANCE_THRESHOLD:
            # Give higher score to more distant skills (but cap to avoid completely unrelated skills)
            # Use a sigmoid-like function to prefer distance around 0.3-0.5 range
            distance_score = min(semantic_distance * 5.0, 3.0)
            score += distance_score
        else:
            score += 0.0  # Too similar, low score

        # Factor 3: Lexical distance (avoid very similar names)
        if word_overlap <= 1:
            score += 1.0

        # Small bonus for being in covered_skills (shows it's well-established)
        if skill in covered_skills:
            score += 0.5

        if score > 0:
            negative_candidates.append((skill, score))

    if not negative_candidates:
        logger.debug(
            "No suitable hard negatives found from shared occupations for skills: %s",
            positive_skills,
        )
        return []

    # Sort by score and return top candidates
    negative_candidates.sort(key=lambda x: x[1], reverse=True)
    selected_negatives = [skill for skill, _ in negative_candidates[:max_negatives]]

    logger.debug(
        "Selected %d hard negatives from %d candidates (shared occupations: %s)",
        len(selected_negatives),
        len(negative_candidates),
        ", ".join(list(positive_occupations)[:3])
        + ("..." if len(positive_occupations) > 3 else ""),
    )

    return selected_negatives


def generate_new_skill_samples(
    target_skills: List[str],
    skill_docs: Dict[str, Dict[str, object]],
    generator: LLMSampleGenerator,
    samples_per_skill: int = 4,
    max_chars: int = 300,
    language: str = "Deutsch",
    prompt_log_path: Optional[Path] = None,
    covered_skills: Optional[set[str]] = None,
    model=None,
    embedding_cache=None,
    single_positive_label: bool = False,
    max_additional_skills: int = 3,
) -> List[SyntheticSample]:
    """Generate synthetic samples for completely new skills not present in the dataset.

    Args:
        target_skills: List of skill names to generate samples for
        skill_docs: ESCO skill documents for context
        generator: LLM sample generator instance
        samples_per_skill: Number of samples to generate per skill
        max_chars: Maximum character length per sample
        language: Language hint for generation
        prompt_log_path: Optional path to log prompts
        model: Optional sentence transformer model for semantic hard negative selection
        embedding_cache: Optional EmbeddingCache instance for faster embedding retrieval
        covered_skills: Set of skills already covered (used for negative selection)

    Returns:
        List of synthetic samples for the new skills
    """
    if not target_skills:
        logger.warning("No target skills provided for new sample generation")
        return []

    synthetic_samples: List[SyntheticSample] = []
    covered_skills = covered_skills or set()

    # Prepare potential negative skills (well-established skills that are different)
    potential_negatives = list(covered_skills) if covered_skills else []

    for skill_idx, skill in enumerate(target_skills):
        logger.info(
            "Generating samples for new skill '%s' (%d/%d)",
            skill,
            skill_idx + 1,
            len(target_skills),
        )

        # Get skill context from ESCO documents
        doc = skill_docs.get(skill, {})
        skill_context = LabelContext(
            label=skill,
            current_queries=[],  # No existing queries for new skills
            skill_description=(
                doc.get("description")
                if isinstance(doc.get("description"), str)
                else None
            ),
            related_occupations=(
                doc.get("occupations")
                if isinstance(doc.get("occupations"), list)
                else []
            ),
        )

        # Find co-occurring skills based on shared occupations
        cooccurring_skills = find_cooccurring_skills_by_occupation(
            skill, skill_docs, min_shared_occupations=1, max_results=20
        )

        logger.debug(
            "Found %d co-occurring skills for '%s' based on occupation overlap",
            len(cooccurring_skills),
            skill,
        )

        # Generate multiple samples with different scenarios and skill combinations
        skill_samples: List[SyntheticSample] = []
        scenarios_used: set[str] = set()

        for sample_idx in range(samples_per_skill):
            # Select scenario, avoiding repeats if possible
            available_scenarios = [
                s for s in DEFAULT_SCENARIOS if s not in scenarios_used
            ]
            if not available_scenarios:
                available_scenarios = list(DEFAULT_SCENARIOS)
                scenarios_used.clear()

            scenario = random.choice(available_scenarios)
            scenarios_used.add(scenario)

            # Build positive label list starting with target skill
            positive_skills = [skill]

            # Determine number of additional skills based on flags
            if single_positive_label:
                # Single positive label mode: no additional skills
                num_additional_skills = 0
            else:
                # Multi-label mode: randomly decide how many additional skills (1-max)
                # Weight towards fewer skills (more realistic)
                max_add = min(max_additional_skills, 5)
                choices = list(range(1, max_add + 1))
                # Weight exponentially towards fewer skills
                weights = [2 ** (max_add - i) for i in choices]
                num_additional_skills = random.choices(choices, weights=weights, k=1)[0]

            if num_additional_skills > 0 and cooccurring_skills:
                # Sample from co-occurring skills, weighted by overlap
                # Higher overlap = more likely to co-occur
                available_cooccurring = [
                    (s, count)
                    for s, count in cooccurring_skills
                    if s not in positive_skills
                ]

                if available_cooccurring:
                    # Weight by square root of overlap to balance
                    weights = [count**0.5 for _, count in available_cooccurring]
                    total_weight = sum(weights)
                    probabilities = [w / total_weight for w in weights]

                    # Sample without replacement
                    num_to_select = min(
                        num_additional_skills, len(available_cooccurring)
                    )
                    selected_indices = []
                    for _ in range(num_to_select):
                        if not probabilities:
                            break
                        idx = random.choices(
                            range(len(probabilities)), weights=probabilities, k=1
                        )[0]
                        selected_indices.append(idx)
                        # Remove selected item
                        probabilities.pop(idx)
                        available_cooccurring.pop(idx)

                    additional_skills = [
                        available_cooccurring[i][0]
                        for i in selected_indices
                        if i < len(available_cooccurring)
                    ]
                    positive_skills.extend(additional_skills)

            logger.debug(
                "Sample %d for '%s': %d positive labels (including %d co-occurring)",
                sample_idx + 1,
                skill,
                len(positive_skills),
                len(positive_skills) - 1,
            )

            # Select smart hard negatives that avoid confusion
            # These are from the SAME occupation but semantically distant
            negative_skills = find_smart_hard_negatives(
                positive_skills,
                skill_docs,
                covered_skills,
                max_negatives=2,
                model=model,
                embedding_cache=embedding_cache,
            )

            # Build combined context for all positive skills
            combined_description_parts = []
            combined_occupations = []

            for pos_skill in positive_skills:
                pos_doc = skill_docs.get(pos_skill, {})
                desc = pos_doc.get("description")
                if desc and isinstance(desc, str):
                    combined_description_parts.append(f"{pos_skill}: {desc}")

                occs = pos_doc.get("occupations", [])
                if isinstance(occs, list):
                    combined_occupations.extend(occs)

            # Create combined context
            combined_description = (
                " | ".join(combined_description_parts)
                if combined_description_parts
                else None
            )
            combined_occupations = list(
                dict.fromkeys(combined_occupations)
            )  # Remove duplicates

            multi_skill_context = LabelContext(
                label=", ".join(positive_skills[:3])
                + ("..." if len(positive_skills) > 3 else ""),  # Truncate for display
                current_queries=[],
                skill_description=combined_description,
                related_occupations=combined_occupations,
            )

            # Build prompt for multi-skill generation
            prompt = build_new_skill_prompt(
                multi_skill_context,
                scenario=scenario,
                language=language,
                max_chars=max_chars,
                avoid_terms=negative_skills,
                positive_skills=positive_skills,  # Pass full list
            )

            # Log the prompt
            log_prompt(
                prompt_log_path,
                label=skill,
                scenario=scenario,
                attempt=sample_idx + 1,
                focus_labels=positive_skills,
                avoid_terms=negative_skills,
                prompt=prompt,
            )

            # Generate sample
            try:
                label_hint = " + ".join(positive_skills[:3])  # Hint for multiple skills
                raw_candidates = generator.generate(
                    prompt, expected=1, label_hint=label_hint
                )

                # Enforce constraints (no existing texts to avoid since this is a new skill)
                approved = enforce_constraints(
                    raw_candidates,
                    max_length=max_chars,
                    existing_texts=[],
                    forbidden_terms=negative_skills,
                )

                if approved:
                    sample = SyntheticSample(
                        query=approved[0],
                        pos=positive_skills,  # Multiple positive labels
                        neg=negative_skills,
                        source=(
                            "llm_new_skill_multi"
                            if len(positive_skills) > 1
                            else "llm_new_skill"
                        ),
                        scenario=scenario,
                        label=skill,  # Primary skill for tracking
                    )
                    skill_samples.append(sample)
                    logger.debug(
                        "Generated sample %d/%d for skill '%s'",
                        len(skill_samples),
                        samples_per_skill,
                        skill,
                    )
                else:
                    logger.warning(
                        "No valid sample generated for skill '%s', attempt %d",
                        skill,
                        sample_idx + 1,
                    )

            except Exception as exc:
                logger.error(
                    "Failed to generate sample for skill '%s', attempt %d: %s",
                    skill,
                    sample_idx + 1,
                    exc,
                )

        # Add successful samples
        synthetic_samples.extend(skill_samples)
        logger.info(
            "Successfully generated %d/%d samples for skill '%s'",
            len(skill_samples),
            samples_per_skill,
            skill,
        )

    logger.info(
        "Generated %d total samples for %d new skills",
        len(synthetic_samples),
        len(target_skills),
    )
    return synthetic_samples


def build_new_skill_prompt(
    skill_context: LabelContext,
    scenario: str,
    language: str,
    max_chars: int,
    avoid_terms: Sequence[str],
    positive_skills: Optional[List[str]] = None,
) -> str:
    """Build a prompt specifically for generating samples for completely new skills.

    Args:
        skill_context: Context information for the skill(s)
        scenario: Generation scenario
        language: Target language
        max_chars: Maximum character length
        avoid_terms: Skills to avoid mentioning
        positive_skills: List of positive skills to include (supports multi-label)
    """
    description = skill_context.skill_description or "(keine Beschreibung verfügbar)"
    # Sample maximum of 3 occupations randomly if more than 3 are available
    available_occupations = skill_context.related_occupations
    if len(available_occupations) > 3:
        sampled_occupations = random.sample(available_occupations, 3)
    else:
        sampled_occupations = available_occupations
    occupations = ", ".join(sampled_occupations) or "(keine Angaben)"
    avoid_text = ", ".join(sorted({term for term in avoid_terms if term}))

    target_length = max(150, max_chars - 100)

    # Handle single or multiple positive skills
    if positive_skills and len(positive_skills) > 1:
        skills_text = ", ".join(positive_skills[:-1]) + f" und {positive_skills[-1]}"
        skill_instruction = (
            f"Der Text muss alle folgenden Kompetenzen klar vermitteln: {skills_text}."
        )
        context_prefix = (
            "Du generierst Trainingsbeispiele für mehrere zusammenhängende Kompetenzen."
        )
    else:
        skill = positive_skills[0] if positive_skills else skill_context.label
        skills_text = skill
        skill_instruction = (
            f"Der Text muss die Kompetenz '{skill}' eindeutig und sinnvoll vermitteln."
        )
        context_prefix = "Du generierst Trainingsbeispiele für eine neue, bisher nicht abgedeckte Kompetenz."

    instructions: List[str] = [
        f"Erzeuge genau einen kurzen deutschen Text (<= {target_length} Zeichen) und nutze höchstens zwei Sätze.",
        skill_instruction,
        f"Verwende die Perspektive '{scenario}' und passe den Stil entsprechend an.",
        "Formuliere natürlich und praxisnah, ohne Markdown, Aufzählungen oder Fettdruck.",
        "Der Text soll als realistisches Trainingsbeispiel für semantische Suche dienen.",
    ]

    # Add specific guidance for multi-skill samples
    if positive_skills and len(positive_skills) > 1:
        instructions.append(
            "Die Kompetenzen sollten auf natürliche Weise zusammen erwähnt werden, "
            "wie sie in einem realen Kontext gemeinsam auftreten würden."
        )

    # Add scenario-specific guidance
    scenario_guidance = SCENARIO_GUIDANCE.get(scenario, [])
    instructions.extend(scenario_guidance)

    instructions.extend(
        [
            "Gib nur den fertigen Text ohne zusätzliche Einordnung zurück.",
            "Vermeide Wiederholungen und halte den Text natürlich klingend.",
        ]
    )

    if avoid_text:
        instructions.append(
            f"Erwähne auf keinen Fall diese anderen Kompetenzen: {avoid_text}."
        )

    prompt = "\n".join(
        [
            context_prefix,
            f"\nKompetenz(en): {skills_text}",
            "\nESCO-Beschreibung:",
            description,
            "\nVerwandte Berufe:",
            occupations,
            "\nAufgabe:",
            *instructions,
        ]
    )

    return prompt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training samples for rare positive labels or completely new skills",
    )
    parser.add_argument("dataset", help="Path to the source dataset JSON file")
    parser.add_argument("output", help="Path to write the synthetic samples JSON file")
    parser.add_argument(
        "--skill-docs",
        default="../../data/ESCO/sources/skills_as_documents_v120.csv",
        help="CSV file containing ESCO skill descriptions (default: %(default)s)",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--generate-new-skills",
        action="store_true",
        help="Generate samples for completely new skills instead of rare existing labels",
    )
    mode_group.add_argument(
        "--analyze-coverage",
        action="store_true",
        help="Analyze skill coverage and suggest skills for expansion, then exit",
    )

    # New skill generation options
    parser.add_argument(
        "--target-skills-json",
        help="JSON file containing list of skill names to generate samples for (required with --generate-new-skills)",
    )
    parser.add_argument(
        "--samples-per-skill",
        type=int,
        default=4,
        help="Number of samples to generate per new skill (default: %(default)s)",
    )
    parser.add_argument(
        "--single-positive-label",
        action="store_true",
        help="Generate samples with only one positive label (recommended for focused training data)",
    )
    parser.add_argument(
        "--max-additional-skills",
        type=int,
        default=3,
        help="Maximum number of additional co-occurring skills to include (default: %(default)s, ignored if --single-positive-label)",
    )
    parser.add_argument(
        "--expansion-ratio",
        type=float,
        default=0.2,
        help="Target expansion ratio for coverage analysis (default: %(default)s means 20%% growth)",
    )
    parser.add_argument(
        "--semantic-model-path",
        help="Path to sentence transformer model for semantic ESCO categorization (improves accuracy)",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        default="./chroma_embedding_cache",
        help="Directory for embedding cache (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-cache-prepopulation",
        action="store_true",
        help="Skip pre-populating the embedding cache (useful if cache is already populated)",
    )
    parser.add_argument(
        "--enable-diversity",
        action="store_true",
        help="Enable diversity calculation (requires semantic model, may be slow for large datasets)",
    )

    # Existing rare label generation options
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=4,
        help="Minimum occurrences required per label before generation (default: %(default)s)",
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=3,
        help="Maximum number of positive labels combined in one synthetic sample (default: %(default)s)",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=8,
        help="Maximum number of negative labels to include per synthetic sample (default: %(default)s)",
    )
    parser.add_argument(
        "--attempt-multiplier",
        type=float,
        default=4.0,
        help="Multiplier applied to the remaining sample deficit to cap generation attempts (default: %(default)s)",
    )

    # LLM configuration
    parser.add_argument(
        "--model",
        default="mistral-medium-2508",
        help="Model name for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://mistral.ai/api/v1"),
        help="Base URL for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLM_API_KEY"),
        help="API key for the endpoint (default: environment variable)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retry attempts for failed LLM calls",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=300,
        help="Maximum character length per generated sample",
    )

    # Deprecated/ignored options
    parser.add_argument(
        "--negatives",
        type=int,
        default=None,
        help="[Deprecated] This value is ignored; alle verfügbaren Negativ-Labels werden genutzt",
    )

    # General options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without contacting the LLM endpoint, generating placeholder outputs",
    )
    parser.add_argument(
        "--language",
        default="Deutsch",
        help="Language hint for the generation prompt (default: Deutsch)",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--prompt-log",
        default="prompt_log.txt",
        help="Path to append generated prompts for debugging (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()

    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))
    random.seed(args.seed)

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    skill_docs_path = Path(args.skill_docs) if args.skill_docs else None

    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_dataset(dataset_path)

    logger.info("Loading skill documents")
    skill_docs = load_skill_documents(skill_docs_path)

    # Analyze current skill coverage
    covered_skills, uncovered_skills, skill_frequency = analyze_skill_coverage(
        dataset, skill_docs
    )

    # Handle coverage analysis mode
    if args.analyze_coverage:
        logger.info("=== SKILL COVERAGE ANALYSIS ===")
        logger.info("Current coverage: %d skills", len(covered_skills))
        logger.info("Available uncovered skills: %d", len(uncovered_skills))

        if uncovered_skills:
            # Create performance priorities based on evaluation insights
            performance_priorities = {
                # Underrepresented areas (high priority)
                "00": 5.0,  # Allgemeine Bildungsprogramme
                "T5": 4.5,  # Körperliche und manuelle Fähigkeiten
                "01": 4.0,  # Erziehungswissenschaften
                "04": 4.0,  # Wirtschaft, Verwaltung
                "S5": 4.0,  # Arbeiten mit Computern
                "02": 3.5,  # Kunst und Geisteswissenschaften
                "10": 3.5,  # Dienstleistungen
                "T2": 3.5,  # Denkfähigkeiten
                "05": 4.0,  # Naturwissenschaften
                # Areas with worse performance (medium-high priority)
                "T6": 3.5,  # Aktive Bürgerschaft - worst performance
                "S2": 4.0,  # Informationskompetenzen - poor performance
                "S4": 1.0,  # Managementfähigkeiten - poor performance
                # Well-performing areas (lower priority)
                "L": 1.0,  # Language skills
                "08": 1.0,  # Agriculture
                "T3": 1.0,  # Self-management
                "09": 1.0,  # Health services
                "S7": 2.0,  # Construction
                "S1": 2.0,  # Communication
                "S3": 2.0,  # Care and support
                "S6": 2.0,  # Handling/Transport
                "S8": 2.0,  # Machine operation
                "T4": 2.0,  # Social/communication skills
                "03": 2.0,  # Social sciences
                "06": 2.0,  # ICT
                "07": 2.0,  # Engineering
                "99": 1.0,  # Unknown
            }

            suggestions = suggest_skills_for_expansion(
                uncovered_skills,
                skill_docs,
                args.expansion_ratio,
                len(covered_skills),
                args.semantic_model_path,
                performance_priorities,
                covered_skills,  # Pass covered skills for diversity calculation
                args.enable_diversity,  # Pass diversity flag
            )

            logger.info("=== TOP SUGGESTED SKILLS FOR EXPANSION ===")
            for i, (skill, score) in enumerate(suggestions[:20], 1):
                doc = skill_docs.get(skill, {})
                desc = (
                    doc.get("description", "")[:100] + "..."
                    if len(doc.get("description", "")) > 100
                    else doc.get("description", "")
                )
                logger.info("%2d. %s (score: %.1f)", i, skill, score)
                if desc:
                    logger.info("    %s", desc)

            # Write suggestions to JSON file
            suggestions_file = output_path.with_suffix(".suggested_skills.json")
            suggestions_data = [
                {
                    "skill": skill,
                    "priority_score": score,
                    "description": skill_docs.get(skill, {}).get("description", ""),
                    "occupations": skill_docs.get(skill, {}).get("occupations", []),
                }
                for skill, score in suggestions
            ]

            with suggestions_file.open("w", encoding="utf-8") as f:
                json.dump(suggestions_data, f, ensure_ascii=False, indent=2)
            logger.info(
                "Wrote %d skill suggestions to %s",
                len(suggestions_data),
                suggestions_file,
            )

        logger.info("Coverage analysis complete.")
        return 0

    # Handle new skill generation mode
    if args.generate_new_skills:
        if not args.target_skills_json:
            logger.error(
                "--target-skills-json is required when using --generate-new-skills"
            )
            return 1

        target_skills_path = Path(args.target_skills_json)
        if not target_skills_path.exists():
            logger.error("Target skills JSON file not found: %s", target_skills_path)
            return 1

        logger.info("Loading target skills from %s", target_skills_path)
        with target_skills_path.open("r", encoding="utf-8") as f:
            target_skills_data = json.load(f)

        if isinstance(target_skills_data, list):
            target_skills = [
                skill for skill in target_skills_data if isinstance(skill, str)
            ]
        elif isinstance(target_skills_data, dict) and "skills" in target_skills_data:
            target_skills = target_skills_data["skills"]
        else:
            logger.error(
                'Invalid target skills JSON format. Expected list of strings or {"skills": [...]}'
            )
            return 1

        if not target_skills:
            logger.error("No valid skills found in target skills JSON")
            return 1

        logger.info("Generating samples for %d new skills", len(target_skills))

        # Validate skills exist in ESCO documents
        missing_skills = [skill for skill in target_skills if skill not in skill_docs]
        if missing_skills:
            logger.warning(
                "The following skills are not found in ESCO documents: %s",
                missing_skills[:5],
            )
            target_skills = [skill for skill in target_skills if skill in skill_docs]
            logger.info(
                "Proceeding with %d skills that have ESCO documentation",
                len(target_skills),
            )

        if not target_skills:
            logger.error("No target skills found in ESCO skill documents")
            return 1

        prompt_log_path = Path(args.prompt_log) if args.prompt_log else None
        if prompt_log_path is not None:
            prompt_log_path.parent.mkdir(parents=True, exist_ok=True)

        generator = LLMSampleGenerator(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_retries=args.max_retries,
            dry_run=args.dry_run,
        )

        # Load semantic model and create embedding cache for hard negative selection
        semantic_model = None
        embedding_cache = None

        if args.semantic_model_path:
            try:
                from sentence_transformers import SentenceTransformer

                # Create embedding cache if available
                cache_dir = args.embedding_cache_dir or "./chroma_embedding_cache"
                if create_embedding_cache is not None:
                    logger.info("Initializing embedding cache at %s", cache_dir)
                    embedding_cache = create_embedding_cache(
                        model_name_or_path=args.semantic_model_path, cache_dir=cache_dir
                    )

                    # Pre-populate cache if requested and not already populated
                    if embedding_cache and not args.skip_cache_prepopulation:
                        cache_stats = embedding_cache.get_cache_stats()
                        if cache_stats["total_embeddings"] == 0:
                            logger.info(
                                "Pre-populating embedding cache with all ESCO skills..."
                            )
                            embedding_cache.populate_from_skill_docs(skill_docs)
                        else:
                            logger.info(
                                "Using existing cache with %d embeddings (hit rate: %.1f%%)",
                                cache_stats["total_embeddings"],
                                cache_stats["hit_rate"] * 100,
                            )
                else:
                    logger.warning(
                        "Embedding cache module not available. "
                        "Embeddings will be computed on-the-fly (slower)."
                    )
                    semantic_model = SentenceTransformer(args.semantic_model_path)

                logger.info(
                    "Loaded semantic model %s for smart hard negative selection",
                    args.semantic_model_path,
                )
            except ImportError as e:
                logger.warning(
                    "sentence-transformers not available. Install with: pip install sentence-transformers"
                )
                logger.warning("Hard negatives will use lexical distance only.")
            except Exception as e:
                logger.warning(
                    f"Could not load model {args.semantic_model_path}: {e}. "
                    "Hard negatives will use lexical distance only."
                )

        synthetic_samples = generate_new_skill_samples(
            target_skills=target_skills,
            skill_docs=skill_docs,
            generator=generator,
            samples_per_skill=args.samples_per_skill,
            max_chars=args.max_chars,
            language=args.language,
            prompt_log_path=prompt_log_path,
            covered_skills=covered_skills,
            model=semantic_model,
            embedding_cache=embedding_cache,
            single_positive_label=args.single_positive_label,
            max_additional_skills=args.max_additional_skills,
        )

        # Log final cache statistics
        if embedding_cache:
            final_stats = embedding_cache.get_cache_stats()
            logger.info(
                "Embedding cache statistics - Total: %d, Hits: %d, Misses: %d, Hit rate: %.1f%%",
                final_stats["total_embeddings"],
                final_stats["cache_hits"],
                final_stats["cache_misses"],
                final_stats["hit_rate"] * 100,
            )

        write_samples(output_path, synthetic_samples)
        logger.info(
            "New skill sample generation complete. Generated %d samples.",
            len(synthetic_samples),
        )
        return 0

    # Handle traditional rare label generation mode
    logger.info("Indexing positives by label")
    (
        label_queries,
        _label_negatives,
        _label_cooccurrence,
        label_sample_groups,
        record_negatives,
    ) = build_label_indexes(dataset)

    rare_labels = find_rare_labels(label_queries, args.min_occurrences)
    if not rare_labels:
        logger.info(
            "No labels found with fewer than %s occurrences", args.min_occurrences
        )
        write_samples(output_path, [])
        return 0

    label_deficits = {
        label: args.min_occurrences - count for label, count in rare_labels.items()
    }
    total_needed = sum(label_deficits.values())
    logger.info(
        "Identified %s rare labels requiring %s synthetic samples in total",
        len(label_deficits),
        total_needed,
    )
    if total_needed <= 0:
        logger.info("All labels already meet the minimum occurrence threshold")
        write_samples(output_path, [])
        return 0

    prompt_log_path = Path(args.prompt_log) if args.prompt_log else None
    if prompt_log_path is not None:
        prompt_log_path.parent.mkdir(parents=True, exist_ok=True)

    generator = LLMSampleGenerator(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )

    synthetic_samples: List[SyntheticSample] = []
    generated_queries_by_label: Dict[str, List[str]] = defaultdict(list)
    attempt_counter: Dict[str, int] = defaultdict(int)

    max_attempts = max(
        int(total_needed * args.attempt_multiplier),
        len(label_deficits) * 3,
        total_needed,
    )
    max_attempts = max(max_attempts, total_needed)
    attempts_total = 0

    while label_deficits and attempts_total < max_attempts:
        primary_label = max(label_deficits, key=label_deficits.get)
        if label_deficits.get(primary_label, 0) <= 0:
            label_deficits.pop(primary_label, None)
            continue

        attempt_counter[primary_label] += 1
        attempts_total += 1

        focus_labels, source_group, source_record_index = select_focus_labels(
            primary_label,
            label_deficits,
            label_sample_groups,
            args.max_group_size,
        )
        unused_group_labels = [
            label for label in source_group if label not in focus_labels
        ]
        avoid_terms = select_negative_labels(
            focus_labels,
            unused_group_labels,
            (
                record_negatives.get(source_record_index, [])
                if source_record_index is not None
                else []
            ),
            args.max_negatives,
        )

        combined_queries: List[str] = []
        descriptions: List[str] = []
        occupations: List[str] = []
        for focus in focus_labels:
            combined_queries.extend(label_queries.get(focus, []))
            doc = skill_docs.get(focus, {})
            desc = (
                doc.get("description")
                if isinstance(doc.get("description"), str)
                else None
            )
            if desc:
                descriptions.append(desc.strip())
            occ_list = (
                doc.get("occupations")
                if isinstance(doc.get("occupations"), list)
                else []
            )
            for occ in occ_list:
                if isinstance(occ, str) and occ.strip():
                    occupations.append(occ.strip())

        combined_queries = list(dict.fromkeys(combined_queries))
        combined_description = " ".join(descriptions) if descriptions else None
        combined_occupations = list(dict.fromkeys(occupations))

        context = LabelContext(
            label=primary_label,
            current_queries=combined_queries,
            skill_description=combined_description,
            related_occupations=combined_occupations,
        )
        scenario = random.choice(DEFAULT_SCENARIOS)

        prompt = build_prompt(
            context,
            scenario=scenario,
            focus_labels=focus_labels,
            language=args.language,
            max_chars=args.max_chars,
            avoid_terms=avoid_terms,
        )
        log_prompt(
            prompt_log_path,
            label=primary_label,
            scenario=scenario,
            attempt=attempt_counter[primary_label],
            focus_labels=focus_labels,
            avoid_terms=avoid_terms,
            prompt=prompt,
        )

        raw_candidates = generator.generate(
            prompt,
            expected=1,
            label_hint=" ".join(focus_labels),
        )

        existing_texts: List[str] = []
        for focus in focus_labels:
            existing_texts.extend(label_queries.get(focus, []))
            existing_texts.extend(generated_queries_by_label.get(focus, []))

        approved = enforce_constraints(
            raw_candidates,
            max_length=args.max_chars,
            existing_texts=existing_texts,
            forbidden_terms=avoid_terms,
        )
        if not approved:
            logger.warning(
                "No valid candidate accepted for focus labels %s (primary '%s', attempt %s/%s).",
                ", ".join(focus_labels),
                primary_label,
                attempt_counter[primary_label],
                max_attempts,
            )
            continue

        query_text = approved[0]
        sample = SyntheticSample(
            query=query_text,
            pos=list(focus_labels),
            neg=avoid_terms[: args.max_negatives],
            source="llm",
            scenario=scenario,
            label=primary_label,
        )
        synthetic_samples.append(sample)

        for focus in focus_labels:
            generated_queries_by_label[focus].append(query_text)
            label_queries.setdefault(focus, []).append(query_text)
            if focus in label_deficits:
                label_deficits[focus] = max(label_deficits[focus] - 1, 0)
                if label_deficits[focus] <= 0:
                    label_deficits.pop(focus, None)

        remaining_deficit = sum(label_deficits.values())
        pending_labels = sum(1 for deficit in label_deficits.values() if deficit > 0)
        logger.info(
            "Progress: generated %s/%s samples; labels still below target: %s (remaining deficit %s)",
            len(synthetic_samples),
            total_needed,
            pending_labels,
            remaining_deficit,
        )

    if label_deficits:
        for label, remaining in sorted(
            label_deficits.items(), key=lambda item: item[0]
        ):
            if remaining > 0:
                logger.warning(
                    "After generation %s additional samples are still required for label '%s'.",
                    remaining,
                    label,
                )

    write_samples(output_path, synthetic_samples)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        sys.exit(130)
