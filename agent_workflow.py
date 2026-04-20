"""
agent_workflow.py

This module builds and runs the LangGraph multi-agent exoskeleton for the
Semantic Halting Problem (SHP). It iterates over ALL scenarios defined in
test_scenarios.json, running a Writer-Critic loop for each and halting using
REAL semantic embeddings (BAAI/bge-small-en-v1.5) so that cosine distance
accurately reflects meaning — not just character differences.

Why real embeddings?
    MockEmbeddings (MD5 hash) treats "cement base" and "concrete foundation"
    as completely unrelated because they hash differently. Real embeddings place
    them close together in vector space because they mean the same thing.
    Only real embeddings can reliably detect semantic convergence (deadlock).
"""
import json
import os
from typing import TypedDict, List, Dict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings

from semantic_entropy import SemanticEntropyCalculator
from agents import writer_node, critic_node

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Real Embeddings — captures semantic meaning, not just character hashes.
# This is the same model downloaded by ragas_eval.py, so no extra downloads.
# ──────────────────────────────────────────────────────────────────────────────
print("[Setup] Loading real semantic embedding model (BAAI/bge-small-en-v1.5)...")
real_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
calculator = SemanticEntropyCalculator(embedding_model=real_embeddings)

# Convergence threshold: if two consecutive drafts are this similar, we halt.
# With real embeddings, semantically identical phrases like "cement base" /
# "concrete foundation" will produce a distance < 0.05, so we use 0.05 here.
CONVERGENCE_THRESHOLD = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Shared State
# ──────────────────────────────────────────────────────────────────────────────
class WorkflowState(TypedDict):
    """
    Represents the shared memory (whiteboard) of the LangGraph multi-agent workflow.

    Attributes:
        scenario (dict):               The active test scenario loaded from JSON.
        current_draft (str):           The latest text produced by the writer agent.
        history (List[Dict[str,str]]): Log of every draft + critic feedback pair.
        loop_count (int):              How many writer-critic rounds have run.
        previous_embedding (Optional[List[float]]): Embedding vector of the previous draft.
        current_embedding (Optional[List[float]]):  Embedding vector of the current draft.
        halt_reason (str):             Why the loop was stopped ("convergence" or "failsafe").
    """
    scenario: dict
    current_draft: str
    history: List[Dict[str, str]]
    loop_count: int
    previous_embedding: Optional[List[float]]
    current_embedding: Optional[List[float]]
    halt_reason: str


# ──────────────────────────────────────────────────────────────────────────────
# Graph Nodes
# ──────────────────────────────────────────────────────────────────────────────
def embed_state_node(state: WorkflowState) -> dict:
    """
    LangGraph Node: Converts the current draft into a real semantic embedding vector.

    This node slides the embeddings forward: the old current_embedding becomes
    previous_embedding, and a fresh embedding of the current draft becomes
    current_embedding. The check_convergence function then compares them.

    Args:
        state (WorkflowState): The current workflow state.

    Returns:
        dict: State update with new previous_embedding and current_embedding.
    """
    draft = state.get("current_draft", "")
    current_emb = calculator.get_embedding(draft)
    prev_emb = state.get("current_embedding", None)

    return {
        "previous_embedding": prev_emb,
        "current_embedding": current_emb
    }


def check_convergence(state: WorkflowState) -> str:
    """
    LangGraph Conditional Edge: Applies the Banach Fixed-Point Theorem logic.

    Compares consecutive draft embeddings using cosine distance:
        - distance ≈ 0  → the drafts mean the same thing → HALT (convergence)
        - distance >> 0 → new semantic information was added → CONTINUE

    Also enforces a hard failsafe at 15 iterations to prevent any edge-case
    runaway loops regardless of embedding distance.

    Args:
        state (WorkflowState): Current workflow state containing embeddings.

    Returns:
        str: "critic" to continue, or "end" to halt the graph.
    """
    prev_emb = state.get("previous_embedding")
    curr_emb = state.get("current_embedding")

    if prev_emb is None:
        print("[System] Initial pass, no previous embedding yet — continuing...")
        return "critic"

    distance = calculator.calculate_distance(curr_emb, prev_emb)
    loop = state.get("loop_count", 0)
    print(f"[System] Round {loop} | Semantic Distance (Entropy): {distance:.6f}  (threshold: {CONVERGENCE_THRESHOLD})")

    if distance < CONVERGENCE_THRESHOLD:
        print(f"\n🛑 [HALT — CONVERGENCE] Distance {distance:.6f} < {CONVERGENCE_THRESHOLD}.")
        print("   Semantic entropy collapsed. Agents are no longer adding new information.\n")
        return "end"

    if loop > 15:
        print("\n⚠️  [HALT — FAILSAFE] Hard iteration limit reached (>15 rounds).")
        print("   Semantic entropy did NOT converge. Consider adjusting the threshold or scenario.\n")
        return "end"

    return "critic"


# ──────────────────────────────────────────────────────────────────────────────
# Graph Assembly
# ──────────────────────────────────────────────────────────────────────────────
def build_graph() -> object:
    """
    Assembles and compiles the LangGraph StateGraph.

    Graph topology:
        writer → embed_state → check_convergence ──► critic → (back to writer)
                                                  └──► END

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    workflow = StateGraph(WorkflowState)
    workflow.add_node("writer", writer_node)
    workflow.add_node("embed_state", embed_state_node)
    workflow.add_node("critic", critic_node)
    workflow.set_entry_point("writer")
    workflow.add_edge("writer", "embed_state")
    workflow.add_conditional_edges(
        "embed_state",
        check_convergence,
        {"critic": "critic", "end": END}
    )
    workflow.add_edge("critic", "writer")
    return workflow.compile()


def run_scenario(app, scenario: dict) -> dict:
    """
    Executes the full Writer-Critic workflow for a single scenario.

    Args:
        app:             The compiled LangGraph application.
        scenario (dict): A scenario dict loaded from test_scenarios.json.

    Returns:
        dict: The final workflow state after halting.
    """
    print(f"\n{'━'*60}")
    print(f"📋 SCENARIO: {scenario['id'].upper().replace('_',' ')}")
    print(f"   Topic: {scenario['topic']}")
    print(f"   {scenario['description']}")
    print(f"{'━'*60}")

    initial_state = {
        "scenario": scenario,
        "current_draft": "",
        "history": [],
        "loop_count": 0,
        "previous_embedding": None,
        "current_embedding": None,
        "halt_reason": ""
    }

    final_state = app.invoke(initial_state)

    print(f"\n✅ Scenario '{scenario['id']}' complete after {final_state['loop_count']} rounds.")
    print(f"   Final draft: {final_state['current_draft'][:120]}...")
    return final_state


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load all scenarios from the JSON file
    with open("test_scenarios.json", "r") as f:
        scenarios = json.load(f)

    app = build_graph()

    print("\n🌐 Starting Semantic Halting Problem — Multi-Scenario Run")
    print(f"   Loaded {len(scenarios)} scenario(s) from test_scenarios.json\n")

    results = []
    for scenario in scenarios:
        final = run_scenario(app, scenario)
        results.append({
            "scenario_id": scenario["id"],
            "rounds": final["loop_count"],
            "final_draft": final["current_draft"]
        })

    print("\n" + "═"*60)
    print("📊 MULTI-SCENARIO SUMMARY")
    print("═"*60)
    for r in results:
        print(f"  [{r['scenario_id']}]  →  halted after {r['rounds']} round(s)")
    print("═"*60)
