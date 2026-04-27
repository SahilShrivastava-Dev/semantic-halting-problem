"""
semantic_entropy.py

Provides SemanticEntropyCalculator — the mathematical backbone of the
Semantic Halting Problem (SHP) solution.

Core idea (Banach Fixed-Point analogy):
    Every draft is mapped to a high-dimensional vector. We measure cosine
    distance between consecutive draft vectors. When that distance falls below
    CONVERGENCE_THRESHOLD for CONVERGENCE_PATIENCE consecutive rounds, the
    agent loop has reached a fixed point — no new semantic value is being
    generated — and is halted.

Metric choice rationale:
    Cosine distance measures the ANGLE between vectors, ignoring magnitude.
    This is correct for text: we care whether meaning changed, not whether the
    text got longer. BAAI/bge-small-en-v1.5 is trained with InfoNCE contrastive
    loss on semantic pairs, producing L2-normalised vectors where dot product ==
    cosine similarity — making this both theoretically sound and computationally
    cheap (no LLM API calls required).
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
            implements embed_query(text: str) -> List[float].
    """

    def __init__(self, embedding_model) -> None:
        if embedding_model is None:
            raise ValueError(
                "SemanticEntropyCalculator requires a non-None embedding_model."
            )
        self.embedding_model = embedding_model
        logger.debug(
            "SemanticEntropyCalculator initialised with %s",
            type(embedding_model).__name__,
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Embed text into a dense vector (384-dim for BAAI/bge-small-en-v1.5).

        Args:
            text: Arbitrary text to embed.

        Returns:
            Dense embedding vector as a list of floats.
        """
        embedding: List[float] = self.embedding_model.embed_query(text)
        logger.debug(
            "Embedded text (%d chars) → vector dim %d", len(text), len(embedding)
        )
        return embedding

    def calculate_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine distance between two embedding vectors.

        Cosine distance = 1 − cosine_similarity ∈ [0, 2]:
            0.0      → identical semantic content
            ~0.04–0.08 → synonym-level paraphrase (deadlock zone)
            > 0.10   → meaningfully different content

        Returns 1.0 (conservative) if either vector is zero-norm, preventing
        a false-positive halt on empty or degenerate inputs.
        """
        v1 = np.array(vec1, dtype=np.float64)
        v2 = np.array(vec2, dtype=np.float64)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0.0 or norm2 == 0.0:
            logger.warning(
                "calculate_distance: zero-norm vector detected; "
                "returning conservative distance 1.0 to avoid false-positive halt."
            )
            return 1.0

        cosine_similarity = float(np.dot(v1, v2) / (norm1 * norm2))
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        distance = 1.0 - cosine_similarity

        logger.debug("Cosine distance: %.6f", distance)
        return distance
