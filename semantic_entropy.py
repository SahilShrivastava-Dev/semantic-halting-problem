"""
semantic_entropy.py

This module provides the SemanticEntropyCalculator, which computes the semantic distance
between text sequences using high-dimensional embeddings. This mathematical measurement 
is used to detect convergence and halt iterative loops.
"""
import numpy as np
from typing import List
from langchain_openai import OpenAIEmbeddings

# For demonstration, we'll allow passing a mock embedding function or a real one.
class SemanticEntropyCalculator:
    """
    A utility class to calculate the semantic distance (entropy) between two pieces of text.
    
    This class wraps an embedding model (e.g., OpenAIEmbeddings, HuggingFaceEmbeddings) 
    and uses cosine distance to measure how semantically different two strings are. 
    It is the mathematical backbone for deciding when to halt the multi-agent system.
    """
    def __init__(self, embedding_model=None):
        """
        Initializes the calculator with a specific embedding model.
        
        Args:
            embedding_model: A Langchain-compatible embeddings object. If None, 
                             defaults to OpenAIEmbeddings (requires OPENAI_API_KEY).
        """
        # Fallback to OpenAI if None provided (requires OPENAI_API_KEY)
        self.embedding_model = embedding_model or OpenAIEmbeddings()

    def get_embedding(self, text: str) -> List[float]:
        """
        Converts a string of text into a high-dimensional vector.
        
        Args:
            text (str): The input text to embed.
            
        Returns:
            List[float]: The dense vector representation of the text.
        """
        return self.embedding_model.embed_query(text)

    def calculate_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine distance between two embedding vectors.
        
        The distance approaches 0.0 as the vectors become more semantically identical, 
        and approaches 1.0 (or higher) as they diverge.
        
        Args:
            vec1 (List[float]): The first embedding vector.
            vec2 (List[float]): The second embedding vector.
            
        Returns:
            float: The cosine distance (1.0 - cosine_similarity).
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 1.0 # Max distance if empty
        
        cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return 1.0 - cosine_similarity
