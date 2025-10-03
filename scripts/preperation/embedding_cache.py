"""
Embedding Cache Module using ChromaDB

This module provides efficient caching of skill embeddings to avoid recomputing
embeddings for the same skills repeatedly. Uses ChromaDB for persistent storage.

Key Features:
- Persistent storage across runs
- Fast similarity search
- Automatic batch processing
- Cache hit/miss logging
- Support for multiple embedding models

Usage:
    cache = EmbeddingCache(
        model_name="intfloat/multilingual-e5-base",
        cache_dir="./chroma_cache"
    )

    # Get embeddings (uses cache or computes if missing)
    embeddings = cache.get_embeddings(["Python programming", "Java development"])

    # Pre-populate cache
    cache.populate_from_skill_docs(skill_docs)
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Caches skill embeddings using ChromaDB for fast retrieval."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "./chroma_embedding_cache",
        collection_name: Optional[str] = None,
    ):
        """
        Initialize embedding cache.

        Args:
            model_name: Name or path of the sentence transformer model
            cache_dir: Directory to store ChromaDB data
            collection_name: Name of the collection (derived from model_name if None)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for embedding caching. "
                "Install it with: pip install chromadb"
            )

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a safe collection name from model name
        if collection_name is None:
            # Hash the model name to create a valid collection name
            model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
            collection_name = f"embeddings_{model_hash}"

        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.cache_dir), settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        # Note: ChromaDB requires a distance metric. We use cosine for semantic similarity.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"model_name": model_name, "hnsw:space": "cosine"},
        )

        self.model: Optional[SentenceTransformer] = None
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            f"Initialized embedding cache at {cache_dir} "
            f"(collection: {collection_name}, model: {model_name})"
        )

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _create_skill_id(self, skill_text: str) -> str:
        """Create a unique ID for a skill text."""
        # Use hash of the text as ID
        return hashlib.md5(skill_text.encode("utf-8")).hexdigest()

    def get_embedding(self, skill_text: str, description: str = "") -> np.ndarray:
        """
        Get embedding for a single skill (uses cache or computes).

        Args:
            skill_text: The skill name
            description: Optional skill description to include in embedding

        Returns:
            Embedding vector as numpy array
        """
        embeddings = self.get_embeddings([skill_text], [description])
        return embeddings[0]

    def get_embeddings(
        self,
        skill_texts: List[str],
        descriptions: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        Get embeddings for multiple skills (uses cache or computes).

        Args:
            skill_texts: List of skill names
            descriptions: Optional list of skill descriptions
            batch_size: Batch size for encoding new skills

        Returns:
            List of embedding vectors
        """
        if descriptions is None:
            descriptions = [""] * len(skill_texts)

        if len(skill_texts) != len(descriptions):
            raise ValueError("skill_texts and descriptions must have same length")

        # Prepare full texts for embedding
        full_texts = [
            f"{skill}: {desc}" if desc else skill
            for skill, desc in zip(skill_texts, descriptions)
        ]

        # Create IDs for lookup
        skill_ids = [self._create_skill_id(text) for text in full_texts]

        # Try to retrieve from cache
        cached_results = self.collection.get(ids=skill_ids, include=["embeddings"])

        cached_embeddings = cached_results.get("embeddings", [])
        cached_ids_set = set(cached_results.get("ids", []))

        # Build result list
        embeddings: List[Optional[np.ndarray]] = [None] * len(skill_texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        missing_ids: List[str] = []

        for i, (skill_id, full_text) in enumerate(zip(skill_ids, full_texts)):
            if skill_id in cached_ids_set:
                # Found in cache
                cache_idx = cached_results["ids"].index(skill_id)
                embeddings[i] = np.array(cached_embeddings[cache_idx])
                self._cache_hits += 1
            else:
                # Need to compute
                missing_indices.append(i)
                missing_texts.append(full_text)
                missing_ids.append(skill_id)
                self._cache_misses += 1

        # Compute missing embeddings
        if missing_texts:
            logger.debug(
                f"Cache miss: computing {len(missing_texts)} embeddings "
                f"(hits: {self._cache_hits}, misses: {self._cache_misses})"
            )

            model = self._load_model()
            new_embeddings = model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=len(missing_texts) > 100,
                convert_to_numpy=True,
            )

            # Store in cache in batches to avoid chromadb batch size limit
            batch_size_chroma = 5000
            for i in range(0, len(missing_ids), batch_size_chroma):
                batch_ids = missing_ids[i : i + batch_size_chroma]
                batch_embeddings = new_embeddings[i : i + batch_size_chroma].tolist()
                batch_metadatas = [
                    {"text": text} for text in missing_texts[i : i + batch_size_chroma]
                ]
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )

            # Fill in results
            for i, new_emb in zip(missing_indices, new_embeddings):
                embeddings[i] = new_emb

        # All embeddings should be filled now
        return [emb for emb in embeddings if emb is not None]

    def populate_from_skill_docs(
        self,
        skill_docs: Dict[str, Dict[str, object]],
        batch_size: int = 32,
    ) -> int:
        """
        Pre-populate cache with embeddings for all skills in skill_docs.

        Args:
            skill_docs: Dictionary of skill documents with descriptions
            batch_size: Batch size for encoding

        Returns:
            Number of new embeddings added
        """
        logger.info(f"Pre-populating cache with {len(skill_docs)} skills...")

        skill_texts = []
        descriptions = []

        for skill_name, doc in skill_docs.items():
            skill_texts.append(skill_name)
            desc = doc.get("description", "")
            descriptions.append(desc if isinstance(desc, str) else "")

        initial_misses = self._cache_misses

        # This will compute and cache any missing embeddings
        self.get_embeddings(skill_texts, descriptions, batch_size=batch_size)

        new_embeddings = self._cache_misses - initial_misses

        logger.info(
            f"Cache population complete. Added {new_embeddings} new embeddings. "
            f"Total cache size: {self.collection.count()}"
        )

        return new_embeddings

    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find most similar skills to a query embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_ids: Optional list of IDs to exclude from results

        Returns:
            List of (skill_id, distance) tuples
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["distances"],
        )

        similar_skills = []
        for skill_id, distance in zip(results["ids"][0], results["distances"][0]):
            if filter_ids is None or skill_id not in filter_ids:
                similar_skills.append((skill_id, distance))

        return similar_skills

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "total_embeddings": self.collection.count(),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        logger.warning(f"Clearing embedding cache: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"model_name": self.model_name, "hnsw:space": "cosine"},
        )
        self._cache_hits = 0
        self._cache_misses = 0


def create_embedding_cache(
    model_name_or_path: str,
    cache_dir: str = "./chroma_embedding_cache",
) -> Optional[EmbeddingCache]:
    """
    Create an embedding cache instance.

    Args:
        model_name_or_path: Name or path of the sentence transformer model
        cache_dir: Directory to store ChromaDB data

    Returns:
        EmbeddingCache instance or None if ChromaDB not available
    """
    if not CHROMADB_AVAILABLE:
        logger.warning(
            "ChromaDB not available. Embeddings will not be cached. "
            "Install chromadb with: pip install chromadb"
        )
        return None

    try:
        return EmbeddingCache(
            model_name=model_name_or_path,
            cache_dir=cache_dir,
        )
    except Exception as e:
        logger.error(f"Failed to create embedding cache: {e}")
        return None
