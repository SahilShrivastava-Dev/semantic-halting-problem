"""
semantic_entropy.py

Provides SemanticEntropyCalculator — the mathematical backbone of the
Semantic Halting Problem (SHP) solution.

Core idea (Banach Fixed-Point analogy):
    Every draft is mapped to a high-dimensional vector.  We measure the cosine
    distance between consecutive draft vectors.  When that distance falls below
    CONVERGENCE_THRESHOLD (see config.py), the agent loop has reached a fixed
    point — no new semantic value is being generated — and is halted.

No LLM API calls are made here; this module is pure numerical computation on
top of a Langchain-compatible embedding model passed in at construction time.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class SemanticEntropyCalculator:
    """
    Computes cosine distance between pairs of text embeddings to detect
    semantic convergence (deadlock) in iterative multi-agent loops.

    Attributes:
        embedding_model: Any Langchain-compatible embeddings object that
            implements ``embed_query(text: str) -> List[float]``.
            Example: ``HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")``

    Usage::

        from langchain_huggingface import HuggingFaceEmbeddings
        from semantic_entropy import SemanticEntropyCalculator

        model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        calc  = SemanticEntropyCalculator(embedding_model=model)

        v1 = calc.get_embedding("The foundation uses cement.")
        v2 = calc.get_embedding("The foundation uses concrete.")
        dist = calc.calculate_distance(v1, v2)
        # dist ≈ 0.04  →  below CONVERGENCE_THRESHOLD → halt signal
    """

    def __init__(self, embedding_model) -> None:
        """
        Initialise the calculator with a pre-constructed embedding model.

        Args:
            embedding_model: A Langchain-compatible embeddings object.
                Must not be ``None``; callers are responsible for building
                and configuring the model before passing it in.

        Raises:
            ValueError: If ``embedding_model`` is None.
        """
        if embedding_model is None:
            raise ValueError(
                "SemanticEntropyCalculator requires a non-None embedding_model. "
                "Pass a HuggingFaceEmbeddings (or equivalent) instance."
            )
        self.embedding_model = embedding_model
        logger.debug("SemanticEntropyCalculator initialised with %s", type(embedding_model).__name__)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> List[float]:
        """
        Convert a string into its dense vector representation.

        The vector dimensionality is determined by the underlying model
        (384-dim for ``BAAI/bge-small-en-v1.5``).

        Args:
            text (str): Arbitrary text to embed.  Empty strings are
                embedded as-is; the model returns a zero-like vector.

        Returns:
            List[float]: Dense embedding vector (length = model dimension).
        """
        embedding: List[float] = self.embedding_model.embed_query(text)
        logger.debug("Embedded text (%d chars) → vector dim %d", len(text), len(embedding))
        return embedding

    def calculate_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute the cosine distance between two embedding vectors.

        Cosine distance = 1 − cosine_similarity, which lies in [0, 2]:
            • 0.0  → vectors are identical (same semantic content)
            • ~0.04–0.08 → synonym-level paraphrase (typical deadlock zone)
            • > 0.1  → meaningfully different content

        Args:
            vec1 (List[float]): Embedding of the previous draft.
            vec2 (List[float]): Embedding of the current draft.

        Returns:
            float: Cosine distance in the range [0, 2].
                   Returns 1.0 if either vector is the zero vector, i.e.,
                   we conservatively assume the texts are *different*
                   (preventing a false-positive halt on empty inputs).
        """
        v1 = np.array(vec1, dtype=np.float64)
        v2 = np.array(vec2, dtype=np.float64)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0.0 or norm2 == 0.0:
            logger.warning(
                "calculate_distance called with a zero-norm vector; "
                "returning conservative distance 1.0 to avoid false-positive halt."
            )
            return 1.0

        cosine_similarity: float = float(np.dot(v1, v2) / (norm1 * norm2))
        # Clamp to [-1, 1] to guard against floating-point drift
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        distance = 1.0 - cosine_similarity

        logger.debug("Cosine distance: %.6f", distance)
        return distance
