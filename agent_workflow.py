"""
agent_workflow.py

Builds and executes the LangGraph multi-agent exoskeleton implementing the
Semantic Halting Problem (SHP) solution.

Halting Logic (two independent signals — either triggers a halt):
    1. Semantic Entropy Convergence: the cosine distance between consecutive
       draft embeddings drops below CONVERGENCE_THRESHOLD, meaning the agents
       are no longer generating new semantic value.
    2. Critic Approval: the Critic LLM explicitly returns "APPROVED", signalling
       it can find no further substantive improvements.
    3. Failsafe: a hard cap of MAX_ROUNDS prevents runaway loops in edge cases.

Data Output:
    Saves agent_results.json — consumed by ragas_eval.py in the next pipeline stage.
"""
import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import TypedDict, List, Dict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings

from semantic_entropy import SemanticEntropyCalculator
from agents import writer_node, critic_node

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
# Semantic distance below which we declare convergence (deadlock).
# With real sentence embeddings, "cement base" vs "concrete foundation"
# scores ~0.04 — well below this threshold.
CONVERGENCE_THRESHOLD = 0.06

# Hard safety cap to prevent unbounded API usage.
MAX_ROUNDS = 12


# ─────────────────────────────────────────────────────────────
# Real Embedding Model
# ─────────────────────────────────────────────────────────────
print("[Setup] Loading BAAI/bge-small-en-v1.5 embedding model...")
_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
calculator  = SemanticEntropyCalculator(embedding_model=_embeddings)


# ─────────────────────────────────────────────────────────────
# Shared State Schema
# ─────────────────────────────────────────────────────────────
class WorkflowState(TypedDict):
    """
    The shared whiteboard that all LangGraph nodes read from and write to.

    Attributes:
        scenario (dict):                The active scenario from test_scenarios.json.
        current_draft (str):            Latest draft text from the Writer LLM.
        history (List[Dict]):           Log of all (draft, feedback) pairs this run.
        loop_count (int):               Number of completed Writer→Critic iterations.
        previous_embedding (Optional):  Embedding vector of the previous draft.
        current_embedding (Optional):   Embedding vector of the current draft.
        halt_reason (str):              Why the loop stopped (logged for analysis).
    """
    scenario:            dict
    current_draft:       str
    history:             List[Dict[str, str]]
    loop_count:          int
    previous_embedding:  Optional[List[float]]
    current_embedding:   Optional[List[float]]
    halt_reason:         str


# ─────────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────────
def embed_state_node(state: WorkflowState) -> dict:
    """
    Converts the current draft to a semantic embedding vector.
    Slides the window: previous_embedding ← current_embedding ← new embedding.

    Args:
        state (WorkflowState): Current workflow state.

    Returns:
        dict: State update with updated embedding vectors.
    """
    draft       = state.get("current_draft", "")
    new_emb     = calculator.get_embedding(draft)
    prev_emb    = state.get("current_embedding", None)
    return {
        "previous_embedding": prev_emb,
        "current_embedding":  new_emb,
    }


def check_convergence(state: WorkflowState) -> str:
    """
    LangGraph Conditional Edge implementing the Banach Fixed-Point halting logic.

    Checks three conditions in priority order:
        1. Critic APPROVED signal in last feedback.
        2. Semantic entropy (cosine distance) < CONVERGENCE_THRESHOLD.
        3. Hard failsafe: loop_count > MAX_ROUNDS.

    Args:
        state (WorkflowState): Current workflow state.

    Returns:
        str: "critic" to continue looping, or "end" to halt.
    """
    prev_emb   = state.get("previous_embedding")
    curr_emb   = state.get("current_embedding")
    loop_count = state.get("loop_count", 0)
    history    = state.get("history", [])

    # ── 1. Initial pass: no previous embedding yet ───────────────────────────
    if prev_emb is None:
        print("[System] Initial pass — no previous embedding yet, continuing...")
        return "critic"

    # ── 2. Check if Critic approved on the last round ────────────────────────
    if history:
        last_feedback = history[-1].get("feedback", "")
        if last_feedback.strip().upper().startswith("APPROVED"):
            print(f"\n✅ [HALT — CRITIC APPROVED] The Critic LLM found no further improvements.")
            print(f"   Halting after {loop_count} round(s).\n")
            return "end"

    # ── 3. Semantic entropy convergence check ────────────────────────────────
    distance = calculator.calculate_distance(curr_emb, prev_emb)
    print(f"[System] Round {loop_count:2d} | Semantic Distance: {distance:.6f}  "
          f"(threshold: {CONVERGENCE_THRESHOLD})")

    if distance < CONVERGENCE_THRESHOLD:
        print(f"\n🛑 [HALT — ENTROPY CONVERGENCE] Distance {distance:.6f} < {CONVERGENCE_THRESHOLD}")
        print(f"   Agents have reached a semantic fixed point. No new information is being generated.")
        print(f"   Halted at round {loop_count}.\n")
        return "end"

    # ── 4. Hard failsafe ─────────────────────────────────────────────────────
    if loop_count >= MAX_ROUNDS:
        print(f"\n⚠️  [HALT — FAILSAFE] Reached maximum of {MAX_ROUNDS} rounds.")
        print(f"   Semantic entropy did not converge within the limit.\n")
        return "end"

    return "critic"


# ─────────────────────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────────────────────
def build_graph():
    """
    Assembles and compiles the LangGraph StateGraph.

    Topology:
        writer → embed_state → check_convergence ──► critic ─► (back to writer)
                                                 └──► END

    Returns:
        Compiled LangGraph application.
    """
    wf = StateGraph(WorkflowState)
    wf.add_node("writer",      writer_node)
    wf.add_node("embed_state", embed_state_node)
    wf.add_node("critic",      critic_node)
    wf.set_entry_point("writer")
    wf.add_edge("writer", "embed_state")
    wf.add_conditional_edges(
        "embed_state",
        check_convergence,
        {"critic": "critic", "end": END}
    )
    wf.add_edge("critic", "writer")
    return wf.compile()


def run_scenario(app, scenario: dict) -> dict:
    """
    Runs the full Writer→Critic loop for one scenario until halted.

    Args:
        app:             Compiled LangGraph application.
        scenario (dict): Scenario loaded from test_scenarios.json.

    Returns:
        dict: Structured result with all data needed for Ragas evaluation.
    """
    print(f"\n{'━'*65}")
    print(f"📋 SCENARIO: {scenario['id'].upper().replace('_', ' ')}")
    print(f"   Topic:   {scenario['topic']}")
    print(f"{'━'*65}")

    final = app.invoke({
        "scenario":            scenario,
        "current_draft":       "",
        "history":             [],
        "loop_count":          0,
        "previous_embedding":  None,
        "current_embedding":   None,
        "halt_reason":         "",
    })

    rounds       = final["loop_count"]
    final_draft  = final["current_draft"]
    history      = final["history"]

    print(f"\n📝 Final draft ({len(final_draft.split())} words) after {rounds} round(s):")
    print(f"   {final_draft[:200]}...")

    # Determine halt reason for the summary log
    halt_reason = "failsafe"
    if history:
        last_fb = history[-1].get("feedback", "")
        if last_fb.strip().upper().startswith("APPROVED"):
            halt_reason = "critic_approved"
    if halt_reason == "failsafe" and rounds < MAX_ROUNDS:
        halt_reason = "entropy_convergence"

    return {
        "scenario_id":  scenario["id"],
        "topic":        scenario["topic"],
        "rounds":       rounds,
        "halt_reason":  halt_reason,
        "question":     scenario["question"],
        "answer":       final_draft,           # ← REAL LLM output evaluated by Ragas
        "contexts":     scenario["contexts"],
        "ground_truth": scenario["ground_truth"],
        "history":      [{"draft": h["draft"][:200], "feedback": h["feedback"]}
                         for h in history],   # truncated for JSON size
    }


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("test_scenarios.json", "r") as f:
        scenarios = json.load(f)

    app = build_graph()

    print(f"\n🌐 Semantic Halting Problem — Live LLM Multi-Agent Run")
    print(f"   {len(scenarios)} scenario(s) | threshold={CONVERGENCE_THRESHOLD} | max={MAX_ROUNDS} rounds\n")

    results = []
    for scenario in scenarios:
        result = run_scenario(app, scenario)
        results.append(result)

    # Save for ragas_eval.py
    with open("agent_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Agent results → agent_results.json")

    # Summary
    print(f"\n{'═'*65}")
    print(f"📊 MULTI-SCENARIO SUMMARY")
    print(f"{'═'*65}")
    for r in results:
        print(f"  [{r['scenario_id']}]")
        print(f"    Rounds:      {r['rounds']}")
        print(f"    Halt reason: {r['halt_reason']}")
        print(f"    Answer:      ...{r['answer'][-100:]}")
    print(f"{'═'*65}")
