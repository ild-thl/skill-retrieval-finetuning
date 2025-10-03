#!/usr/bin/env python3
"""Compress or split overly long query texts for bi-encoder fine-tuning datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from collections import Counter
from datetime import datetime

from dotenv import load_dotenv

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:  # pragma: no cover - explicit guidance
    raise RuntimeError(
        "The 'sentence-transformers' package is required. Install it with `pip install sentence-transformers`."
    ) from exc

try:  # pragma: no cover - optional dependency for actual calls
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - explicit guidance
    raise RuntimeError(
        "The 'transformers' package is required. Install it with `pip install transformers`."
    ) from exc


logger = logging.getLogger(__name__)

MODEL_NAME_DEFAULT = "intfloat/multilingual-e5-base"
MAX_TOKENS_DEFAULT = 320
TARGET_TOKENS_DEFAULT = 320
TOKEN_SPLIT_THRESHOLD = 700
LABEL_SPLIT_THRESHOLD = 5
MAX_LABELS_PER_SPLIT = 3
DEFAULT_SIMILARITY_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIMILARITY_THRESHOLD = 0.88


@dataclass(frozen=True)
class DatasetRecord:
    query: str
    pos: List[str]
    neg: List[str]
    meta: Dict[str, object]

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "DatasetRecord":
        query = data.get("query")
        pos = data.get("pos")
        neg = data.get("neg")
        meta = data.get("meta") or {}
        if not isinstance(query, str):
            raise ValueError("Record missing text query")
        if not isinstance(pos, list) or not all(isinstance(item, str) for item in pos):
            raise ValueError("Record positives must be list of strings")
        if not isinstance(neg, list) or not all(isinstance(item, str) for item in neg):
            raise ValueError("Record negatives must be list of strings")
        if not isinstance(meta, dict):
            raise ValueError("Record meta must be a dict")
        return DatasetRecord(query=query, pos=list(pos), neg=list(neg), meta=dict(meta))

    def to_dict(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "pos": list(self.pos),
            "neg": list(self.neg),
            "meta": dict(self.meta),
        }


class QueryCompressor:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str],
        temperature: float,
        dry_run: bool,
        max_retries: int,
        prompt_log_path: Optional[Path],
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.prompt_log_path = prompt_log_path
        self.client: Optional[OpenAI]
        if dry_run:
            self.client = None
            logger.warning("Dry-run enabled: no LLM requests will be sent.")
        else:
            if OpenAI is None:
                raise RuntimeError("The 'openai' package is required for live compression.")
            if not api_key:
                raise RuntimeError("LLM API key must be provided via flag or environment variable.")
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def compress(self, prompt: str, *, attempt: int, meta: Dict[str, object]) -> Optional[str]:
        self._log_prompt(prompt, attempt=attempt, meta=meta)
        if self.dry_run:
            strategy = str(meta.get("strategy", "compress"))
            labels = meta.get("labels", "?")
            if strategy == "llm_clustering":
                # Return a proper JSON structure for clustering
                return '''```json
{
    "groups": [
        {"labels": ["[DryRun] Cluster 1"]},
        {"labels": ["[DryRun] Cluster 2"]}
    ]
}
```'''
            elif strategy in {"split", "label_group"}:
                return f"[DryRun] Kurztext für {labels}"
            return f"[DryRun] komprimiert: {labels}"

        assert self.client is not None
        delay = 2.0
        for idx in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=512,
                    messages=[
                        {"role": "system", "content": "Du bist eine professionelle Kurstext-Redakteurin."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = response.choices[0].message.content or ""
                text = text.strip()
                return text or None
            except Exception as exc:  # pragma: no cover - network errors
                logger.warning("Compression attempt %s/%s failed: %s", idx, self.max_retries, exc)
                if idx == self.max_retries:
                    raise
                import time

                time.sleep(delay)
                delay *= 1.6
        return None

    def _log_prompt(self, prompt: str, *, attempt: int, meta: Dict[str, object]) -> None:
        if self.prompt_log_path is None:
            return
        try:
            with self.prompt_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n".join(
                        [
                            "=" * 80,
                            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                            f"strategy: {meta.get('strategy')}",
                            f"labels: {meta.get('labels')}",
                            f"token_count: {meta.get('token_count')}",
                            f"attempt: {attempt}",
                            "prompt:",
                            prompt,
                            "",
                        ]
                    )
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not write prompt log: %s", exc)


class EmbeddingHelper:
    def __init__(self, model_name: str, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, np.ndarray] = {}

    def _ensure_model(self) -> None:
        if not self.enabled:
            return
        if self._model is None:
            logger.info("Loading similarity model '%s'", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> Optional[np.ndarray]:
        if not self.enabled or not text:
            return None
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        self._ensure_model()
        assert self._model is not None
        vector = self._model.encode(text, normalize_embeddings=True)
        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        self._cache[text] = vector
        return vector

    @staticmethod
    def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None:
            return None
        if a.shape != b.shape:
            return None
        value = float(np.dot(a, b))
        return value


def load_records(path: Path) -> List[DatasetRecord]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of records")
    records: List[DatasetRecord] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry at index %s", idx)
            continue
        try:
            records.append(DatasetRecord.from_dict(entry))
        except ValueError as exc:
            logger.warning("Skipping invalid record at index %s: %s", idx, exc)
    logger.info("Loaded %s valid records from %s", len(records), path)
    return records


def tokenize_length(tokenizer: AutoTokenizer, text: str) -> int:
    # Use truncation to avoid warnings about sequences longer than max length
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=tokenizer.model_max_length or 512)
    return len(tokens)


def should_split(
    record: DatasetRecord,
    token_count: int,
    *,
    token_split_threshold: int,
    label_split_threshold: int,
    max_labels_per_split: int,
) -> bool:
    if len(record.pos) > max_labels_per_split:
        return True
    return (token_count >= token_split_threshold and len(record.pos) > 1) or len(record.pos) > label_split_threshold


def chunk_labels(labels: Sequence[str], *, max_per_chunk: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    for idx in range(0, len(labels), max_per_chunk):
        chunk = [label for label in labels[idx : idx + max_per_chunk] if isinstance(label, str)]
        if chunk:
            chunks.append(chunk)
    return chunks or [[]]


def chunk_labels_by_similarity(
    labels: Sequence[str], 
    *,
    max_per_chunk: int,
    similarity_threshold: float,
    embedding_helper: EmbeddingHelper
) -> List[List[str]]:
    """Group labels by semantic similarity to avoid confusion between positive and negative sets."""
    clean_labels = [label for label in labels if isinstance(label, str) and label]
    if not clean_labels or not embedding_helper.enabled:
        return chunk_labels(clean_labels, max_per_chunk=max_per_chunk)
    
    # Encode all labels
    embeddings = {}
    for label in clean_labels:
        vec = embedding_helper.encode(label)
        if vec is not None:
            embeddings[label] = vec
    
    if not embeddings:
        return chunk_labels(clean_labels, max_per_chunk=max_per_chunk)
    
    # Build similarity matrix and group similar labels
    processed = set()
    chunks = []
    
    for label in clean_labels:
        if label in processed or label not in embeddings:
            continue
            
        # Start a new cluster with this label
        cluster = [label]
        processed.add(label)
        
        # Find similar labels to include in the same cluster
        label_vec = embeddings[label]
        for other_label in clean_labels:
            if (other_label in processed or 
                other_label not in embeddings or 
                len(cluster) >= max_per_chunk):
                continue
                
            other_vec = embeddings[other_label]
            similarity = embedding_helper.cosine(label_vec, other_vec)
            
            if similarity is not None and similarity >= similarity_threshold:
                cluster.append(other_label)
                processed.add(other_label)
        
        chunks.append(cluster)
    
    return chunks or [[]]


def extract_and_parse_json(text: str) -> dict:
    """Extract and parse JSON from text that may contain markdown code blocks or other formatting."""
    import json
    import re
    
    # First try direct parsing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in markdown code blocks (```json...``` or ```...```)
    json_patterns = [
        r'```json\s*(.*?)\s*```',  # ```json ... ```
        r'```\s*(.*?)\s*```',      # ``` ... ```
        r'`([^`]*)`',              # `...`
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON-like structure by looking for { ... } blocks
    brace_pattern = r'\{.*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to extract JSON from lines that start and end with { }
    lines = text.split('\n')
    json_lines = []
    in_json = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{'):
            in_json = True
            json_lines = [stripped]
        elif in_json:
            json_lines.append(stripped)
            if stripped.endswith('}'):
                try:
                    json_text = '\n'.join(json_lines)
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
                in_json = False
                json_lines = []
    
    # If all else fails, raise the original parsing error
    raise json.JSONDecodeError("No valid JSON found in text", text, 0)


def chunk_labels_with_llm(
    labels: Sequence[str],
    *,
    max_per_chunk: int,
    compressor: QueryCompressor,
) -> List[List[str]]:
    """Use LLM to create semantically meaningful and well-differentiable clusters of labels."""
    clean_labels = [label for label in labels if isinstance(label, str) and label.strip()]
    
    if len(clean_labels) <= max_per_chunk:
        return [clean_labels] if clean_labels else [[]]
    
    prompt = build_clustering_prompt(labels=clean_labels, max_per_group=max_per_chunk)
    
    try:
        response = compressor.compress(
            prompt,
            attempt=1,
            meta={
                "strategy": "llm_clustering",
                "labels": f"{len(clean_labels)} labels",
                "token_count": len(prompt.split()),  # Rough token estimate
            },
        )
        
        if not response:
            logger.warning("LLM clustering returned empty response, skipping clustering")
            return None
        
        # Log the raw response for debugging
        logger.debug(f"LLM clustering raw response: {repr(response[:500])}{'...' if len(response) > 500 else ''}")
        
        # Parse JSON response with markdown handling
        import json
        try:
            result = extract_and_parse_json(response)
            if not isinstance(result, dict):
                logger.warning(f"LLM clustering response is not a JSON object, got {type(result).__name__}: {repr(response[:200])}{'...' if len(response) > 200 else ''}")
                return None
                
            if "groups" not in result:
                logger.warning(f"LLM clustering response missing 'groups' key. Available keys: {list(result.keys())}. Response: {repr(response[:300])}{'...' if len(response) > 300 else ''}")
                return None
            
            clusters = []
            processed_labels = set()
            
            for group_idx, group in enumerate(result["groups"]):
                if not isinstance(group, dict):
                    logger.warning(f"Group {group_idx} is not a dict, got {type(group).__name__}: {repr(group)}")
                    continue
                    
                if "labels" not in group:
                    logger.warning(f"Group {group_idx} missing 'labels' key. Available keys: {list(group.keys())}")
                    continue
                    
                cluster = []
                for label_idx, label in enumerate(group["labels"]):
                    if not isinstance(label, str):
                        logger.warning(f"Label {label_idx} in group {group_idx} is not a string, got {type(label).__name__}: {repr(label)}")
                        continue
                        
                    clean_label = label.strip()
                    if clean_label in clean_labels:
                        if clean_label not in processed_labels:
                            cluster.append(clean_label)
                            processed_labels.add(clean_label)
                    else:
                        logger.warning(f"Label '{clean_label}' from LLM response not found in original labels")
                
                if cluster:
                    clusters.append(cluster)
                    logger.debug(f"Created cluster {len(clusters)-1} with {len(cluster)} labels: {cluster[:3]}{'...' if len(cluster) > 3 else ''}")
            
            # Add any remaining labels that weren't processed
            remaining = [label for label in clean_labels if label not in processed_labels]
            if remaining:
                logger.info(f"Adding {len(remaining)} unprocessed labels to clusters: {remaining[:3]}{'...' if len(remaining) > 3 else ''}")
                if clusters and len(clusters[-1]) + len(remaining) <= max_per_chunk:
                    # Add to last cluster if it fits
                    clusters[-1].extend(remaining)
                else:
                    # Create additional clusters for remaining labels
                    for i in range(0, len(remaining), max_per_chunk):
                        chunk = remaining[i:i + max_per_chunk]
                        if chunk:
                            clusters.append(chunk)
            
            if not clusters:
                logger.warning("LLM clustering produced no valid clusters, skipping clustering")
                return None
            
            logger.info(f"LLM clustering created {len(clusters)} clusters from {len(clean_labels)} labels")
            return clusters
            
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to extract or parse JSON from LLM clustering response: {exc}. Raw response: {repr(response[:400])}{'...' if len(response) > 400 else ''}")
            return None
        except (ValueError, KeyError) as exc:
            logger.warning(f"Failed to process LLM clustering response structure: {exc}. Response: {repr(response[:300])}{'...' if len(response) > 300 else ''}")
            return None
            
    except Exception as exc:
        logger.warning(f"LLM clustering failed with exception: {exc}, skipping clustering")
        return None


def build_compress_prompt(
    *,
    original_text: str,
    labels: Sequence[str],
    target_tokens: int,
    tokenizer_name: str,
) -> str:
    label_list = ", ".join(sorted(dict.fromkeys(labels))) or "(keine)"
    instructions = [
        "Kürze den folgenden Kurstext prägnant, ohne Inhalte zu erfinden.",
        "Übernimm Tonalität, Wortwahl und Satzbau des Originals so weit wie möglich und streiche nur überflüssige Passagen.",
        "Heb alle genannten Kompetenzen klar hervor.",
        "Nutze höchstens zwei Sätze im Präsens und bleibe sachlich.",
        "Vermeide Umschreibungen, Synonyme oder neue Beispiele.",
        "Verzichte auf Markdown, Aufzählungen oder Metakommentare.",
        "Gib ausschließlich den gekürzten Text zurück.",
    ]
    return "\n".join(
        [
            "Kompetenzen: " + label_list,
            "\nAufgabe:",
            *instructions,
            "\nOriginaltext:",
            original_text,
        ]
    )


def build_clustering_prompt(
    *,
    labels: Sequence[str],
    max_per_group: int,
) -> str:
    """Build a prompt to cluster positive labels into semantically differentiable groups."""
    label_list = "\n".join(f"- {label}" for label in labels if label)
    
    instructions = [
        f"Gruppiere die folgenden Kompetenzen in semantisch unterscheidbare Cluster von maximal {max_per_group} Kompetenzen pro Gruppe.",
        "Jede Gruppe soll thematisch kohärent und von anderen Gruppen klar abgrenzbar sein.",
        "Verwende ähnliche oder überlappende Kompetenzen in derselben Gruppe.",
        "Trenne deutlich verschiedene Fachbereiche, Fertigkeiten oder Anwendungsgebiete in separate Gruppen.",
        "Gib die Gruppierung im folgenden JSON-Format zurück:",
        '{"groups": [{"labels": ["Kompetenz 1", "Kompetenz 2"]}, {"labels": ["Kompetenz 3", "Kompetenz 4"]}]}',
        "Verwende keine zusätzlichen Erklärungen oder Kommentare, nur das JSON-Objekt.",
    ]
    
    return "\n".join([
        "Zu gruppierende Kompetenzen:",
        label_list,
        "\nAufgabe:",
        *instructions,
    ])


def build_label_group_prompt(
    *,
    original_text: str,
    focus_labels: Sequence[str],
    avoid_labels: Sequence[str],
    target_tokens: int,
    tokenizer_name: str,
) -> str:
    focus_list = ", ".join(sorted(dict.fromkeys([label for label in focus_labels if label]))) or "(keine)"
    avoid_clean = [label for label in avoid_labels if label]
    avoid_list = ", ".join(sorted(dict.fromkeys(avoid_clean))) if avoid_clean else "—"
    base_char_limit = 200 if len(focus_labels) <= 2 else 260
    token_based_limit = max(220, int(target_tokens * 0.8))
    char_limit = min(base_char_limit, token_based_limit)
    instructions = [
        "Schreibe eine stark verdichtete Fassung des Kurstextes in höchstens zwei kurzen Sätzen.",
        f"Die Verkürzte Fassung sollte sich ausschließlich auf folgende angedeutete Kompetenzen fokussieren: {focus_list}.",
        "Bewahre Tonalität, Perspektive und sachlichen Stil des Originaltextes.",
        "Fasse nur vorhandene Inhalte zusammen und erfinde keine neuen Aspekte.",
        "Gib nur den verdichteten Text zurück.",
    ]
    if avoid_clean:
        instructions.insert(4, f"Erwähne keine der folgenden Kompetenzen oder Begriffe: {avoid_list}.")
    return "\n".join(
        [
            "Fokus-Kompetenzen: " + focus_list,
            "Zu vermeidende Kompetenzen: " + (avoid_list if avoid_clean else "(keine)") ,
            "\nAufgabe:",
            *instructions,
            "\nOriginaltext:",
            original_text,
        ]
    )


@dataclass
class CompressionResult:
    original: DatasetRecord
    replacements: List[DatasetRecord]
    token_counts: List[int]
    strategy: str


def process_record(
    record: DatasetRecord,
    *,
    tokenizer: AutoTokenizer,
    compressor: QueryCompressor,
    embedding_helper: EmbeddingHelper,
    target_tokens: int,
    max_tokens: int,
    token_split_threshold: int,
    label_split_threshold: int,
    max_labels_per_split: int,
    chunk_similarity_threshold: float,
    use_llm_clustering: bool = False,
) -> CompressionResult:
    token_count = tokenize_length(tokenizer, record.query)
    if token_count <= max_tokens:
        return CompressionResult(original=record, replacements=[record], token_counts=[token_count], strategy="keep")

    labels = record.pos
    should_do_split = should_split(
        record,
        token_count,
        token_split_threshold=token_split_threshold,
        label_split_threshold=label_split_threshold,
        max_labels_per_split=max_labels_per_split,
    )
    replacements: List[DatasetRecord] = []
    token_counts: List[int] = []

    if should_do_split:
        # Choose clustering strategy
        if use_llm_clustering:
            label_groups = chunk_labels_with_llm(
                labels,
                max_per_chunk=max_labels_per_split,
                compressor=compressor,
            )
            if label_groups is None:
                # LLM clustering failed, skip splitting and just compress
                logger.info("LLM clustering failed, falling back to compression without splitting")
                should_do_split = False
            else:
                strategy_name = "llm_label_group"
        else:
            label_groups = chunk_labels_by_similarity(
                labels, 
                max_per_chunk=max_labels_per_split,
                similarity_threshold=chunk_similarity_threshold,
                embedding_helper=embedding_helper
            )
            strategy_name = "label_group"
        
        if should_do_split:  # Re-check in case LLM clustering failed
            # Create all other groups as potential negatives
            all_other_labels = []
            for group in label_groups:
                all_other_labels.extend(group)
            
            for idx, group in enumerate(label_groups):
                focus_labels = list(dict.fromkeys(group))
                if not focus_labels:
                    continue
                    
                # Use labels from other groups as avoid_labels to prevent confusion
                avoid_labels = [label for label in all_other_labels if label not in focus_labels]
                
                prompt = build_label_group_prompt(
                    original_text=record.query,
                    focus_labels=focus_labels,
                    avoid_labels=avoid_labels,
                    target_tokens=target_tokens,
                    tokenizer_name=tokenizer.name_or_path,
                )
                response = compressor.compress(
                    prompt,
                    attempt=1,
                    meta={
                        "strategy": strategy_name,
                        "labels": ", ".join(focus_labels),
                        "token_count": token_count,
                        "avoid": ", ".join(avoid_labels[:5]) + ("..." if len(avoid_labels) > 5 else ""),  # Truncate for readability
                    },
                )
                if response is None:
                    raise RuntimeError("Label-group prompt returned no content")
                cleaned = response.strip()
                if not cleaned:
                    raise RuntimeError("Label-group prompt returned empty text")
                
                # Check if generated text mentions any forbidden labels
                lowered = cleaned.casefold()
                for forbidden in avoid_labels:
                    if forbidden and forbidden.casefold() in lowered:
                        logger.warning(
                            "Generated text for label group %s contains forbidden label '%s'",
                            focus_labels,
                            forbidden,
                        )
                        break
                        
                # Use labels from other groups as negatives along with original negatives
                neg_labels = sorted(dict.fromkeys(list(record.neg) + avoid_labels))
                new_record = DatasetRecord(
                    query=cleaned,
                    pos=list(focus_labels),
                    neg=neg_labels,
                    meta={
                        **record.meta,
                        "augmentation": strategy_name,
                        "group_index": idx,
                        "group_size": len(focus_labels),
                        "total_groups": len(label_groups),
                    },
                )
                replacements.append(new_record)
                token_counts.append(tokenize_length(tokenizer, cleaned))
                
            return CompressionResult(
                original=record,
                replacements=replacements,
                token_counts=token_counts,
                strategy=strategy_name,
            )

    prompt = build_compress_prompt(
        original_text=record.query,
        labels=labels,
        target_tokens=target_tokens,
        tokenizer_name=tokenizer.name_or_path,
    )
    response = compressor.compress(
        prompt,
        attempt=1,
        meta={"strategy": "compress", "labels": ", ".join(labels), "token_count": token_count},
    )
    if response is None:
        raise RuntimeError("Compression prompt returned no content")
    cleaned = response.strip()
    new_record = DatasetRecord(
        query=cleaned,
        pos=list(labels),
        neg=list(record.neg),
        meta={**record.meta, "augmentation": "compress"},
    )
    replacements.append(new_record)
    token_counts.append(tokenize_length(tokenizer, cleaned))
    return CompressionResult(
        original=record,
        replacements=replacements,
        token_counts=token_counts,
        strategy="compress",
    )


def aggregate_statistics(results: Iterable[CompressionResult]) -> Dict[str, object]:
    counts = Counter(result.strategy for result in results)
    token_deltas: List[int] = []
    for result in results:
        for count in result.token_counts:
            token_deltas.append(count)
    return {
        "strategy_counts": dict(counts),
        "avg_token_count": sum(token_deltas) / max(len(token_deltas), 1),
        "max_token_count": max(token_deltas) if token_deltas else 0,
        "min_token_count": min(token_deltas) if token_deltas else 0,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress or split overly long query texts.")
    parser.add_argument("dataset", help="Path to the source dataset JSON file")
    parser.add_argument("output", help="Path to write the augmented dataset JSON file")
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "mistral-medium-2508"),
        help="Model name for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://api.mistral.ai/v1"),
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
        default=0.4,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed LLM calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without contacting the LLM endpoint, generating placeholder outputs",
    )
    parser.add_argument(
        "--tokenizer",
        default=MODEL_NAME_DEFAULT,
        help="Tokenizer model to measure token lengths (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS_DEFAULT,
        help="Maximum allowed tokens before compression is triggered",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=TARGET_TOKENS_DEFAULT,
        help="Target token length after compression",
    )
    parser.add_argument(
        "--token-split-threshold",
        type=int,
        default=TOKEN_SPLIT_THRESHOLD,
        help="Token threshold above which splitting is attempted",
    )
    parser.add_argument(
        "--label-split-threshold",
        type=int,
        default=LABEL_SPLIT_THRESHOLD,
        help="Positive Label count above which splitting is attempted",
    )
    parser.add_argument(
        "--max-labels-per-split",
        type=int,
        default=MAX_LABELS_PER_SPLIT,
        help="Maximum number of positive labels per split section",
    )
    parser.add_argument(
        "--similarity-model",
        default=DEFAULT_SIMILARITY_MODEL,
        help="SentenceTransformer model to use for similarity-aware chunking",
    )
    parser.add_argument(
        "--disable-similarity-chunking",
        action="store_true",
        help="Disable embedding-based similarity chunking",
    )
    parser.add_argument(
        "--chunk-similarity-threshold",
        type=float,
        default=CHUNK_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for grouping labels in same chunk",
    )
    parser.add_argument(
        "--use-llm-clustering",
        action="store_true",
        help="Use LLM for intelligent clustering of positive labels (slower but more semantic)",
    )
    parser.add_argument(
        "--prompt-log",
        default=None,
        help="Path to append generated prompts for debugging",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    prompt_log_path = Path(args.prompt_log) if args.prompt_log else None
    if prompt_log_path is not None:
        prompt_log_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info(
        "Using tokenizer '%s' (model max length %s).",
        tokenizer.name_or_path,
        getattr(tokenizer, "model_max_length", "unknown"),
    )

    compressor = QueryCompressor(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        prompt_log_path=prompt_log_path,
    )

    embedding_helper = EmbeddingHelper(
        model_name=args.similarity_model,
        enabled=not args.disable_similarity_chunking,
    )
    
    if embedding_helper.enabled:
        logger.info(
            "Similarity-aware chunking enabled (model=%s, threshold=%.3f)",
            args.similarity_model,
            args.chunk_similarity_threshold,
        )
    else:
        logger.info("Similarity-aware chunking disabled; using sequential chunking.")
    
    if args.use_llm_clustering:
        logger.info("LLM-based clustering enabled for semantic differentiation")
    else:
        logger.info("Using embedding-based similarity clustering")

    records = load_records(dataset_path)
    
    # Analyze dataset and provide overview
    logger.info("=== Dataset Analysis ===")
    token_lengths = []
    records_needing_compression = 0
    records_needing_splitting = 0
    total_pos_labels = 0
    
    for record in records:
        token_count = tokenize_length(tokenizer, record.query)
        token_lengths.append(token_count)
        total_pos_labels += len(record.pos)
        
        if token_count > args.max_tokens:
            records_needing_compression += 1
            
            should_do_split = should_split(
                record,
                token_count,
                token_split_threshold=args.token_split_threshold,
                label_split_threshold=args.label_split_threshold,
                max_labels_per_split=args.max_labels_per_split,
            )
            if should_do_split:
                records_needing_splitting += 1
    
    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0
    min_tokens = min(token_lengths) if token_lengths else 0
    
    logger.info(f"Total records: {len(records)}")
    logger.info(f"Token statistics - avg: {avg_tokens:.1f}, max: {max_tokens}, min: {min_tokens}")
    logger.info(f"Records requiring compression: {records_needing_compression} ({records_needing_compression/len(records)*100:.1f}%)")
    logger.info(f"Records requiring splitting: {records_needing_splitting} ({records_needing_splitting/len(records)*100:.1f}%)")
    logger.info(f"Total positive labels: {total_pos_labels}, avg per record: {total_pos_labels/len(records):.1f}")
    
    if max_tokens > (tokenizer.model_max_length or 512):
        logger.warning(f"Some records ({max_tokens} tokens) exceed tokenizer max length ({tokenizer.model_max_length or 512}). This may cause truncation during processing.")
    
    logger.info("=== Processing Records ===")
    
    results: List[CompressionResult] = []
    augmented: List[DatasetRecord] = []

    for idx, record in enumerate(records):
        # Progress reporting
        if len(records) > 10:  # Only show progress for larger datasets
            if (idx + 1) % max(1, len(records) // 20) == 0 or idx == 0 or idx == len(records) - 1:
                progress = (idx + 1) / len(records) * 100
                logger.info(f"Progress: {idx + 1}/{len(records)} ({progress:.1f}%)")
        
        result = process_record(
            record,
            tokenizer=tokenizer,
            compressor=compressor,
            embedding_helper=embedding_helper,
            target_tokens=args.target_tokens,
            max_tokens=args.max_tokens,
            token_split_threshold=args.token_split_threshold,
            label_split_threshold=args.label_split_threshold,
            max_labels_per_split=args.max_labels_per_split,
            chunk_similarity_threshold=args.chunk_similarity_threshold,
            use_llm_clustering=args.use_llm_clustering,
        )
        results.append(result)
        augmented.extend(result.replacements)
        # Removed individual warnings to reduce log spam - summary will show totals

    stats = aggregate_statistics(results)
    logger.info("=== Final Results ===")
    logger.info("Strategy distribution: %s", stats["strategy_counts"])
    logger.info(
        "Token stats after processing -> avg: %.1f, max: %s, min: %s",
        stats["avg_token_count"],
        stats["max_token_count"],
        stats["min_token_count"],
    )
    
    # Additional statistics
    original_count = len(records)
    final_count = len(augmented)
    expansion_ratio = final_count / original_count if original_count > 0 else 1
    
    logger.info(f"Dataset expansion: {original_count} -> {final_count} records (ratio: {expansion_ratio:.2f}x)")
    
    records_still_too_long = sum(1 for result in results for count in result.token_counts if count > args.max_tokens)
    if records_still_too_long > 0:
        logger.warning(f"{records_still_too_long} records still exceed max token limit after processing")

    output = [record.to_dict() for record in augmented]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote %s records to %s", len(output), output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        sys.exit(130)
