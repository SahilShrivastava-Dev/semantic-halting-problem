import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from semantic_entropy import SemanticEntropyCalculator
from agents import writer_node, critic_node

# Define the State for LangGraph
class WorkflowState(TypedDict):
    current_draft: str
    history: List[Dict[str, str]]
    loop_count: int
    previous_embedding: List[float]
    current_embedding: List[float]

# Initialize our math component
# NOTE: We use a Mock embedding model here so you don't need an API key to run the exoskeleton.
# In production, use: calculator = SemanticEntropyCalculator()
class MockEmbeddings:
    def embed_query(self, text: str) -> List[float]:
        # Simple hash-based mock embedding for demonstration
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in h[:10]] # 10-dimensional vector

calculator = SemanticEntropyCalculator(embedding_model=MockEmbeddings())
CONVERGENCE_THRESHOLD = 0.01

def embed_state_node(state):
    """
    Calculates the embedding of the current draft and updates the state.
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
    The Conditional Edge function.
    Applies the Banach Fixed-Point logic to determine if the graph should halt.
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
