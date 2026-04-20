"""
agent_workflow.py

This module serves as the entry point for the Semantic Halting Problem (SHP) exoskeleton.
It builds and executes a LangGraph StateGraph comprising a Writer agent, a Critic agent, 
and a Semantic Entropy convergence checker to prevent infinite loops (deadlocks) in 
multi-agent workflows.
"""
import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from semantic_entropy import SemanticEntropyCalculator
from agents import writer_node, critic_node

# Define the State for LangGraph
class WorkflowState(TypedDict):
    """
    Represents the shared state of the LangGraph multi-agent workflow.
    
    Attributes:
        current_draft (str): The latest text output from the writer agent.
        history (List[Dict[str, str]]): A log of previous drafts and critic feedback.
        loop_count (int): Counter for the number of writer-critic iterations.
        previous_embedding (List[float]): The high-dimensional vector representation of the previous draft.
        current_embedding (List[float]): The high-dimensional vector representation of the current draft.
    """
    current_draft: str
    history: List[Dict[str, str]]
    loop_count: int
    previous_embedding: List[float]
    current_embedding: List[float]

# Initialize our math component
# NOTE: We use a Mock embedding model here so you don't need an API key to run the exoskeleton.
# In production, use: calculator = SemanticEntropyCalculator()
class MockEmbeddings:
    """
    A lightweight, deterministic mock embedding class for testing the system 
    without making live network calls to an embedding API provider.
    """
    def embed_query(self, text: str) -> List[float]:
        """
        Hashes the input text and scales it to simulate a 10-dimensional embedding vector.
        
        Args:
            text (str): The text to embed.
            
        Returns:
            List[float]: A pseudo-random (but deterministic) vector representation.
        """
        # Simple hash-based mock embedding for demonstration
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in h[:10]] # 10-dimensional vector

calculator = SemanticEntropyCalculator(embedding_model=MockEmbeddings())
CONVERGENCE_THRESHOLD = 0.01

def embed_state_node(state):
    """
    LangGraph Node: Calculates the embedding of the current draft and updates the state.
    
    This node extracts the latest draft from the state, runs it through the 
    SemanticEntropyCalculator, and promotes the old current_embedding to previous_embedding.
    
    Args:
        state (dict): The current workflow state.
        
    Returns:
        dict: A state update containing the new current_embedding and previous_embedding.
    """
    draft = state.get("current_draft", "")
    current_emb = calculator.get_embedding(draft)
    
    prev_emb = state.get("current_embedding", None)
    
    return {
        "previous_embedding": prev_emb,
        "current_embedding": current_emb
    }

def check_convergence(state) -> str:
    """
    LangGraph Conditional Edge: Applies the Banach Fixed-Point Theorem logic.
    
    This function compares the semantic distance (cosine distance) between the 
    current draft's embedding and the previous draft's embedding. If the distance 
    falls below CONVERGENCE_THRESHOLD, it halts the graph to prevent deadlock.
    
    Args:
        state (dict): The current workflow state containing the embeddings.
        
    Returns:
        str: The name of the next node to route to ("critic" or "end").
    """
    prev_emb = state.get("previous_embedding")
    curr_emb = state.get("current_embedding")
    
    if prev_emb is None:
        print("[System] Initial pass, continuing...")
        return "critic"
        
    distance = calculator.calculate_distance(curr_emb, prev_emb)
    print(f"[System] Semantic Distance (Entropy): {distance:.6f}")
    
    if distance < CONVERGENCE_THRESHOLD:
        print(f"\n🚀 [HALT] Semantic Entropy Convergence Detected! (Distance: {distance:.6f} < {CONVERGENCE_THRESHOLD})")
        print("🚀 The agents are no longer generating semantic value. Halting execution.")
        return "end"
    
    # Failsafe
    if state.get("loop_count", 0) > 10:
        print("[System] Hard limit reached (failsafe).")
        return "end"
        
    return "critic"

# Build the Graph
workflow = StateGraph(WorkflowState)

workflow.add_node("writer", writer_node)
workflow.add_node("embed_state", embed_state_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("writer")

workflow.add_edge("writer", "embed_state")

# The dynamic mathematical router
workflow.add_conditional_edges(
    "embed_state",
    check_convergence,
    {
        "critic": "critic",
        "end": END
    }
)

workflow.add_edge("critic", "writer")

# Compile
app = workflow.compile()

if __name__ == "__main__":
    print("Starting Semantic Entropy Exoskeleton...")
    initial_state = {
        "current_draft": "",
        "history": [],
        "loop_count": 0,
        "previous_embedding": None,
        "current_embedding": None
    }
    
    # Run the graph
    app.invoke(initial_state)
