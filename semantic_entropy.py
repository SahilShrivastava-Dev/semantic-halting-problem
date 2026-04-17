import numpy as np
from typing import List
from langchain_openai import OpenAIEmbeddings

# For demonstration, we'll allow passing a mock embedding function or a real one.
class SemanticEntropyCalculator:
    def __init__(self, embedding_model=None):
        # Fallback to OpenAI if None provided (requires OPENAI_API_KEY)
        self.embedding_model = embedding_model or OpenAIEmbeddings()

    def get_embedding(self, text: str) -> List[float]:
        """Convert text to a high-dimensional vector."""
        return self.embedding_model.embed_query(text)

    def calculate_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine distance between two vectors. 
        Distance approaches 0 as vectors become identical."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 1.0 # Max distance if empty
        
        cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return 1.0 - cosine_similarity
