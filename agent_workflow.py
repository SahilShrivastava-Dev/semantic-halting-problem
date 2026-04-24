"""
agent_workflow.py

Builds and executes the LangGraph multi-agent exoskeleton implementing the
Semantic Halting Problem (SHP) solution.

Pipeline stages (per scenario)
-------------------------------
    writer  →  evaluator  →  embed_state  →  check_convergence
                                                 │ "critic"
                                                 ▼
                                              critic  →  (back to writer)
                                                 │ "end"
                                                 ▼
                                                END

Halting logic (any one signal triggers a halt)
-----------------------------------------------
    1. Critic approval  — Critic LLM returns ``APPROVED``; it can find no
       further substantive improvements.
    2. Semantic convergence  — cosine distance between consecutive draft
       embeddings drops below ``CONVERGENCE_THRESHOLD`` (config.py).
    3. No Information Gain  — IS score (composite Ragas metric) does not
       improve from one round to the next (after the minimum warm-up period).
    4. Hard failsafe  — ``MAX_ROUNDS`` cap prevents unbounded API usage.

Data output
-----------
    Saves ``agent_results.json`` consumed by ``ragas_eval.py`` in the next
    pipeline stage.

CLI usage
---------
    python agent_workflow.py --split train   # Process training scenarios
    python agent_workflow.py --split val     # Process validation scenarios
    python agent_workflow.py --split test    # Process test scenarios
    python agent_workflow.py --split all     # Process every scenario
"""

import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import TypedDict, List, Dict, Optional

from datasets import Dataset
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper

from config import (
    AGENT_RESULTS_FILE,
    CONVERGENCE_THRESHOLD,
    DEFAULT_IS_WEIGHTS,
    EMBEDDING_MODEL_NAME,
    EVAL_LLM_MAX_RETRIES,
    EVAL_LLM_MAX_TOKENS,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    MAX_ROUNDS,
    MIN_ROUNDS_FOR_GAIN_CHECK,
    SCENARIOS_FILE,
    WEIGHTS_FILE,
)
from semantic_entropy import SemanticEntropyCalculator
from agents import writer_node, critic_node

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Groq compatibility shim for Ragas
# ─────────────────────────────────────────────────────────────
class _FixedChatGroq(ChatGroq):
    """
    Groq's API does not support ``n > 1``.

    Ragas internally requests ``n=3`` for certain metrics (e.g.
    AnswerRelevancy) to sample multiple generations.  This thin subclass
    silently forces ``n=1`` in both the dict-building and generate paths
    so that all Ragas metrics work transparently against the Groq endpoint.
    """

    def _create_message_dicts(self, messages, stop):  # noqa: D401
        message_dicts, params = super()._create_message_dicts(messages, stop)
        params.pop("n", None)  # remove entirely; Groq rejects even n=1 explicitly
        return message_dicts, params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ─────────────────────────────────────────────────────────────
# Load optimised weights (written by optimize_score.py)
# ─────────────────────────────────────────────────────────────
def _load_is_weights() -> dict[str, float]:
    """
    Return the Information Score weights from ``optimized_weights.json``
    if available, otherwise fall back to the equal-weight defaults from
    ``config.py``.

    Returns:
        dict[str, float]: Mapping of metric name → weight (sum ≈ 1.0).
    """
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, "r") as fh:
            data = json.load(fh)
        weights: dict[str, float] = data["weights"]
        logger.info("Loaded learned IS weights from %s: %s", WEIGHTS_FILE, weights)
        return weights
    logger.warning(
        "%s not found — using equal default weights. "
        "Run optimize_score.py to generate learned weights.",
        WEIGHTS_FILE,
    )
    return dict(DEFAULT_IS_WEIGHTS)


IS_WEIGHTS: dict[str, float] = _load_is_weights()

# ─────────────────────────────────────────────────────────────
# Shared model instances (loaded once at module level)
# ─────────────────────────────────────────────────────────────
logger.info("Loading embedding model %s for convergence checking...", EMBEDDING_MODEL_NAME)
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
_calculator = SemanticEntropyCalculator(embedding_model=_embeddings)

logger.info("Initialising Ragas judge LLM (%s) for in-loop IS scoring...", EVAL_LLM_MODEL)
_eval_llm = LangchainLLMWrapper(_FixedChatGroq(
    model=EVAL_LLM_MODEL,
    temperature=EVAL_LLM_TEMPERATURE,
    max_tokens=EVAL_LLM_MAX_TOKENS,
    max_retries=EVAL_LLM_MAX_RETRIES,
))

_RAGAS_METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]


# ─────────────────────────────────────────────────────────────
# Shared State Schema
# ─────────────────────────────────────────────────────────────
class WorkflowState(TypedDict):
    """
    The shared whiteboard that all LangGraph nodes read from and write to.

    Each field is immutable by convention within a single node; nodes return
    partial dicts that LangGraph merges into the state.

    Attributes:
        scenario (dict): The active scenario loaded from test_scenarios.json.
            Contains ``id``, ``split``, ``topic``, ``question``, ``contexts``,
            and ``ground_truth`` keys.
        current_draft (str): Latest draft text produced by the Writer LLM.
        history (List[Dict[str, str]]): Chronological log of all
            ``{draft, feedback}`` pairs for this run.
        loop_count (int): Number of completed Writer → Critic iterations.
            Incremented by ``critic_node`` at the end of each round.
        previous_embedding (Optional[List[float]]): Embedding vector of the
            draft from the *previous* iteration; ``None`` on the first pass.
        current_embedding (Optional[List[float]]): Embedding vector of the
            *current* draft, computed by ``embed_state_node``.
        current_is_score (float): Information Score for the current draft,
            computed by ``evaluator_node``.
        is_score_history (List[float]): Ordered list of IS scores across all
            completed rounds; used to compute per-round IS gain.
        halt_reason (str): Human-readable label set when the loop stops.
            One of: ``"critic_approved"``, ``"entropy_convergence"``,
            ``"no_information_gain"``, or ``"failsafe"``.
    """

    scenario:            dict
    current_draft:       str
    history:             List[Dict[str, str]]
    loop_count:          int
    previous_embedding:  Optional[List[float]]
    current_embedding:   Optional[List[float]]
    current_is_score:    float
    is_score_history:    List[float]
    halt_reason:         str


# ─────────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────────
def embed_state_node(state: WorkflowState) -> dict:
    """
    Compute and store the embedding of the current draft.

    Shifts the current embedding into the ``previous_embedding`` slot and
    stores the freshly computed embedding in ``current_embedding``, enabling
    ``check_convergence`` to compute the round-over-round cosine distance.

    Args:
        state (WorkflowState): Current workflow state.

    Returns:
        dict: Partial state update with ``previous_embedding`` and
            ``current_embedding``.
    """
    draft   = state.get("current_draft", "")
    new_emb = _calculator.get_embedding(draft)
    prev_emb = state.get("current_embedding", None)  # shift current → previous

    logger.debug("embed_state_node: computed embedding for draft (%d chars)", len(draft))
    return {
        "previous_embedding": prev_emb,
        "current_embedding":  new_emb,
    }


def evaluator_node(state: WorkflowState) -> dict:
    """
    Compute the real-time Information Score for the current draft.

    Runs the full set of Ragas metrics against the current draft using the
    in-loop Groq judge LLM and the local BAAI embedding model.  The raw
    metric scores are combined using the learned ``IS_WEIGHTS`` to produce a
    single composite score that summarises RAG quality for this round.

    On evaluation failure (e.g. Groq rate-limit), the last known IS score is
    carried forward so the pipeline continues rather than crashing.

    Args:
        state (WorkflowState): Current workflow state.

    Returns:
        dict: Partial state update with:
            - ``current_is_score`` (float): IS score for this round.
            - ``is_score_history`` (List[float]): Updated history list.
    """
    scenario   = state["scenario"]
    draft      = state["current_draft"]
    round_num  = state.get("loop_count", 0)
    is_history = list(state.get("is_score_history", []))

    logger.info("[Evaluator] Computing Information Score for round %d...", round_num)

    dataset = Dataset.from_dict({
        "question":    [scenario["question"]],
        "answer":      [draft],
        "contexts":    [scenario["contexts"]],
        "ground_truth": [scenario["ground_truth"]],
    })

    try:
        results = evaluate(
            dataset=dataset,
            metrics=_RAGAS_METRICS,
            llm=_eval_llm,
            embeddings=_embeddings,
        )
    except Exception as exc:  # noqa: BLE001
        prev_score = is_history[-1] if is_history else 0.0
        logger.warning(
            "[Evaluator] Ragas evaluation failed (%s: %s). "
            "Carrying forward last IS score %.4f.",
            type(exc).__name__, exc, prev_score,
        )
        return {"current_is_score": prev_score}

    # Apply learned weights to produce a single composite IS score.
    score = 0.0
    for metric in _RAGAS_METRICS:
        m_name = metric.name
        try:
            raw = results[m_name]
            val = float(raw[0]) if isinstance(raw, list) else float(raw)
            # NaN → 0 (metric couldn't be evaluated — treat as absent)
            val = 0.0 if (val != val) else val
        except Exception:  # noqa: BLE001
            val = 0.0
        weight = IS_WEIGHTS.get(m_name, 0.25)
        score += weight * val
        logger.info("  %s: %.4f (weight: %.2f)", m_name, val, weight)

    is_history.append(score)
    logger.info("⭐ Round IS: %.4f", score)

    return {"current_is_score": score, "is_score_history": is_history}


def check_convergence(state: WorkflowState) -> str:
    """
    LangGraph conditional edge — implements the Banach Fixed-Point halting logic.

    Evaluates three independent halt signals in priority order, returning
    ``"end"`` to stop the graph or ``"critic"`` to continue the loop.

    Signal 1 — Critic Approval (highest priority):
        The Critic LLM returned ``APPROVED`` in the previous round's feedback,
        indicating it can find no further substantive improvements.

    Signal 2 — Semantic Entropy Convergence:
        Cosine distance between consecutive draft embeddings is below
        ``CONVERGENCE_THRESHOLD`` (see config.py).  The agents are recycling
        the same semantic content without adding new information.

    Signal 3 — No Information Gain:
        IS score did not improve from the previous round (Δ ≤ 0), after
        a minimum warm-up period (``MIN_ROUNDS_FOR_GAIN_CHECK``).

    Signal 4 — Hard Failsafe (lowest priority):
        ``loop_count ≥ MAX_ROUNDS``.

    Args:
        state (WorkflowState): Current workflow state.

    Returns:
        str: ``"critic"`` to continue, or ``"end"`` to halt.
    """
    prev_emb   = state.get("previous_embedding")
    curr_emb   = state.get("current_embedding")
    loop_count = state.get("loop_count", 0)
    history    = state.get("history", [])

    # ── Initial pass: no previous embedding yet ──────────────────────────────
    if prev_emb is None:
        logger.info("[Convergence] Initial pass — no previous embedding yet, continuing.")
        return "critic"

    # ── Signal 1: Critic approved last round ─────────────────────────────────
    if history:
        last_feedback = history[-1].get("feedback", "")
        if last_feedback.strip().upper().startswith("APPROVED"):
            logger.info(
                "✅ [HALT — CRITIC APPROVED] No further improvements after %d round(s).",
                loop_count,
            )
            return "end"

    # ── Signals 2 & 3: Semantic distance + IS Gain ───────────────────────────
    distance   = _calculator.calculate_distance(curr_emb, prev_emb)
    is_history = state.get("is_score_history", [])

    gain = 0.0
    if len(is_history) >= 2:
        gain = is_history[-1] - is_history[-2]

    logger.info(
        "[Convergence] Round %2d | Distance: %.6f | IS Gain: %+.4f",
        loop_count, distance, gain,
    )

    if distance < CONVERGENCE_THRESHOLD:
        logger.info(
            "🛑 [HALT — ENTROPY] Semantic convergence detected "
            "(distance %.6f < threshold %.4f).",
            distance, CONVERGENCE_THRESHOLD,
        )
        return "end"

    if loop_count >= MIN_ROUNDS_FOR_GAIN_CHECK and gain <= 0.0:
        logger.info(
            "🛑 [HALT — NO IS GAIN] Information Score did not improve (Δ=%.4f).",
            gain,
        )
        return "end"

    # ── Signal 4: Hard failsafe ───────────────────────────────────────────────
    if loop_count >= MAX_ROUNDS:
        logger.warning(
            "⚠️  [HALT — FAILSAFE] Reached maximum of %d rounds "
            "without semantic convergence.",
            MAX_ROUNDS,
        )
        return "end"

    return "critic"


# ─────────────────────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────────────────────
def build_graph():
    """
    Assemble and compile the LangGraph ``StateGraph``.

    Topology::

        writer → evaluator → embed_state → check_convergence ──► critic ─► (back to writer)
                                                              └──► END

    The ``evaluator`` node computes an IS score immediately after each draft,
    enabling real-time Information Gain tracking inside the loop rather than
    as a post-hoc step.

    Returns:
        CompiledStateGraph: Compiled LangGraph application ready for
            ``.invoke()`` calls.
    """
    wf = StateGraph(WorkflowState)
    wf.add_node("writer",      writer_node)
    wf.add_node("evaluator",   evaluator_node)
    wf.add_node("embed_state", embed_state_node)
    wf.add_node("critic",      critic_node)

    wf.set_entry_point("writer")
    wf.add_edge("writer",      "evaluator")
    wf.add_edge("evaluator",   "embed_state")
    wf.add_conditional_edges(
        "embed_state",
        check_convergence,
        {"critic": "critic", "end": END},
    )
    wf.add_edge("critic", "writer")

    logger.info("LangGraph compiled: writer → evaluator → embed_state → [critic|END]")
    return wf.compile()


# ─────────────────────────────────────────────────────────────
# Scenario Runner
# ─────────────────────────────────────────────────────────────
def run_scenario(app, scenario: dict) -> dict:
    """
    Execute the full Writer → Critic loop for one scenario until halted.

    Args:
        app: Compiled LangGraph application returned by ``build_graph()``.
        scenario (dict): A single scenario dict loaded from
            ``test_scenarios.json``.  Must contain ``id``, ``topic``,
            ``question``, ``contexts``, and ``ground_truth``.

    Returns:
        dict: Structured result suitable for serialisation to
            ``agent_results.json``.  Contains all fields required by
            ``ragas_eval.py``:
            ``scenario_id``, ``topic``, ``rounds``, ``halt_reason``,
            ``question``, ``answer``, ``contexts``, ``ground_truth``,
            ``is_score_history``, and ``history`` (truncated to 200 chars
            per draft to keep the file manageable).
    """
    logger.info("=" * 65)
    logger.info("📋 SCENARIO: %s", scenario["id"].upper().replace("_", " "))
    logger.info("   Topic: %s", scenario["topic"])
    logger.info("=" * 65)

    final = app.invoke({
        "scenario":           scenario,
        "current_draft":      "",
        "history":            [],
        "loop_count":         0,
        "previous_embedding": None,
        "current_embedding":  None,
        "current_is_score":   0.0,
        "is_score_history":   [],
        "halt_reason":        "",
    })

    rounds      = final["loop_count"]
    final_draft = final["current_draft"]
    history     = final["history"]
    is_history  = final.get("is_score_history", [])

    logger.info(
        "📝 Final draft (%d words) after %d round(s): ...%s...",
        len(final_draft.split()), rounds, final_draft[:200],
    )

    # Determine the halt reason by inspecting the final state.
    if history and history[-1].get("feedback", "").strip().upper().startswith("APPROVED"):
        halt_reason = "critic_approved"
    elif rounds >= MAX_ROUNDS:
        halt_reason = "failsafe"
    elif (
        len(is_history) >= MIN_ROUNDS_FOR_GAIN_CHECK
        and (is_history[-1] - is_history[-2]) <= 0.0
    ):
        halt_reason = "no_information_gain"
    else:
        halt_reason = "entropy_convergence"

    return {
        "scenario_id":    scenario["id"],
        "topic":          scenario["topic"],
        "rounds":         rounds,
        "halt_reason":    halt_reason,
        "question":       scenario["question"],
        "answer":         final_draft,
        "contexts":       scenario["contexts"],
        "ground_truth":   scenario["ground_truth"],
        "is_score_history": is_history,
        "history": [
            {"draft": h["draft"][:200], "feedback": h["feedback"]}
            for h in history
        ],
    }


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """Parse CLI arguments, load scenarios, run the workflow, and save results."""
    parser = argparse.ArgumentParser(
        description="SHP Multi-Agent Workflow — Semantic Halting Problem demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Which data split to run (default: train).",
    )
    args = parser.parse_args()

    if not os.path.exists(SCENARIOS_FILE):
        logger.error("Scenarios file not found: %s", SCENARIOS_FILE)
        sys.exit(1)

    with open(SCENARIOS_FILE, "r") as fh:
        all_scenarios: list[dict] = json.load(fh)

    scenarios = (
        all_scenarios
        if args.split == "all"
        else [s for s in all_scenarios if s.get("split") == args.split]
    )

    if not scenarios:
        logger.error("No scenarios found for split '%s' in %s.", args.split, SCENARIOS_FILE)
        sys.exit(1)

    logger.info(
        "🌐 SHP Workflow — split=%s | %d scenario(s) | threshold=%.4f",
        args.split, len(scenarios), CONVERGENCE_THRESHOLD,
    )

    app = build_graph()
    results: list[dict] = []

    for scenario in scenarios:
        result = run_scenario(app, scenario)
        results.append(result)

    with open(AGENT_RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("💾 Agent results saved → %s", AGENT_RESULTS_FILE)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("═" * 65)
    logger.info("📊 MULTI-SCENARIO SUMMARY")
    logger.info("═" * 65)
    for r in results:
        logger.info(
            "  [%s]  rounds=%d  halt=%s",
            r["scenario_id"], r["rounds"], r["halt_reason"],
        )
    logger.info("═" * 65)


if __name__ == "__main__":
    main()
