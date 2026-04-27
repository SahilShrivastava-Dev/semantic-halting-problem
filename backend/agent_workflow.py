"""
agent_workflow.py

Builds and executes the LangGraph multi-agent exoskeleton implementing the
Semantic Halting Problem (SHP) solution.

Pipeline stages (per scenario)
-------------------------------
    writer → evaluator → embed_state → check_convergence
                                            │ "critic"
                                            ▼
                                         critic → (back to writer)
                                            │ "end"
                                            ▼
                                           END

Halting signals (any one triggers a halt)
------------------------------------------
    1. Critic approval   — Critic LLM returns "APPROVED".
    2. Semantic entropy  — k=2 patience window: cosine distance < threshold
                           for CONVERGENCE_PATIENCE consecutive rounds.
    3. No IS gain        — Information Score Δ ≤ 0 after MIN_ROUNDS warm-up.
    4. Hard failsafe     — MAX_ROUNDS cap.

Streaming
---------
    All nodes accept an emit callback (dict → None). In WebSocket mode the
    callback routes events to the frontend in real time. In CLI mode it is
    a no-op. This keeps the workflow pure and testable.

CLI usage
---------
    python agent_workflow.py --split train
    python agent_workflow.py --split val
    python agent_workflow.py --split test
    python agent_workflow.py --split all
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypedDict

warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
)

# Ragas RunConfig: serialise metric calls to avoid Groq 429 rate-limit storms.
# All 4 metrics firing in parallel = ~20 simultaneous requests = instant throttle.
try:
    from ragas.run_config import RunConfig as _RagasRunConfig
    _HAS_RUN_CONFIG = True
except ImportError:
    _HAS_RUN_CONFIG = False

from config import (
    AGENT_RESULTS_FILE,
    CONVERGENCE_PATIENCE,
    CONVERGENCE_THRESHOLD,
    DEFAULT_AGENT_MODELS,
    DEFAULT_EVAL_MODELS,
    DEFAULT_IS_WEIGHTS,
    DEFAULT_PROVIDER,
    EMBEDDING_MODEL_NAME,
    GROQ_INTER_ROUND_SLEEP,
    MAX_ROUNDS,
    METRIC_COLS,
    MIN_ROUNDS_FOR_GAIN_CHECK,
    RAGAS_EVAL_TIMEOUT,
    RAGAS_MAX_WORKERS,
    SCENARIOS_FILE,
    WEIGHTS_FILE,
)
from semantic_entropy import SemanticEntropyCalculator
from agents import make_writer_node, make_critic_node
from providers import get_llm, get_ragas_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

Emit = Callable[[Dict[str, Any]], None]


# ─────────────────────────────────────────────────────────────
# Shared State Schema
# ─────────────────────────────────────────────────────────────
class WorkflowState(TypedDict):
    """
    The shared whiteboard that all LangGraph nodes read from and write to.

    Fields
    ------
    scenario            : Active scenario dict from test_scenarios.json.
    current_draft       : Latest draft text produced by the Writer.
    history             : Chronological {draft, feedback} pairs.
    loop_count          : Completed Writer → Critic iterations (0-indexed).
    previous_embedding  : Embedding of the draft from the previous round.
    current_embedding   : Embedding of the current draft.
    distance_history    : Cosine distances per round (for k-patience window).
    current_is_score    : Composite Information Score for this round.
    is_score_history    : IS scores per round (for gain detection).
    halt_reason         : Human-readable stop label set when loop ends.
    """
    scenario:            dict
    current_draft:       str
    history:             List[Dict[str, str]]
    loop_count:          int
    previous_embedding:  Optional[List[float]]
    current_embedding:   Optional[List[float]]
    distance_history:    List[float]
    current_is_score:    float
    is_score_history:    List[float]
    halt_reason:         str


# ─────────────────────────────────────────────────────────────
# Weight loading
# ─────────────────────────────────────────────────────────────
def _load_is_weights() -> dict[str, float]:
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        weights = data["weights"]
        logger.info("Loaded learned IS weights from %s: %s", WEIGHTS_FILE, weights)
        return weights
    logger.warning(
        "%s not found — using equal default weights. Run optimize_score.py first.",
        WEIGHTS_FILE,
    )
    return dict(DEFAULT_IS_WEIGHTS)


# ─────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────
def build_graph(
    provider: str = DEFAULT_PROVIDER,
    agent_model: str | None = None,
    eval_model: str | None = None,
    emit: Emit | None = None,
):
    """
    Assemble and compile the LangGraph StateGraph with provider-specific LLMs.

    Topology:
        writer → evaluator → embed_state → check_convergence ──► critic ─► writer
                                                               └──► END

    Args:
        provider:    LLM provider ("groq" or "openai").
        agent_model: Model for Writer and Critic. Defaults to provider default.
        eval_model:  Model for Ragas judge. Defaults to provider default.
        emit:        Event callback (dict) → None. No-op if None.

    Returns:
        Compiled LangGraph application.
    """
    agent_model = agent_model or DEFAULT_AGENT_MODELS[provider]
    eval_model  = eval_model  or DEFAULT_EVAL_MODELS[provider]
    _emit       = emit or (lambda e: None)

    is_weights = _load_is_weights()

    logger.info(
        "Building graph: provider=%s agent=%s eval=%s", provider, agent_model, eval_model
    )

    # ── Models (loaded once per graph instance) ───────────────────────────────
    agent_llm = get_llm(provider, agent_model)
    ragas_llm = get_ragas_llm(provider, eval_model)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    calculator = SemanticEntropyCalculator(embedding_model=embeddings)
    ragas_metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    # Serialise Ragas metric calls to avoid 429 storms on Groq free tier.
    # OpenAI has higher limits so 2 workers is safe there.
    _n_workers = RAGAS_MAX_WORKERS if provider == "groq" else min(2, RAGAS_MAX_WORKERS + 1)
    _ragas_run_config = (
        _RagasRunConfig(max_workers=_n_workers, timeout=RAGAS_EVAL_TIMEOUT)
        if _HAS_RUN_CONFIG else None
    )
    logger.info(
        "Ragas RunConfig: max_workers=%d timeout=%ds (RunConfig available=%s)",
        _n_workers, RAGAS_EVAL_TIMEOUT, _HAS_RUN_CONFIG,
    )

    # Per-metric fallback scores: when a metric hard-fails (LLMDidNotFinishException,
    # 429 after all retries) we carry forward the last known value instead of
    # defaulting to 0.0, which would corrupt the IS score trajectory.
    _last_metric_scores: dict[str, float] = {}

    # ── Node: embed_state ─────────────────────────────────────────────────────
    def embed_state_node(state: WorkflowState) -> dict:
        draft     = state.get("current_draft", "")
        new_emb   = calculator.get_embedding(draft)
        prev_emb  = state.get("current_embedding", None)
        dist_hist = list(state.get("distance_history", []))

        distance = None
        if prev_emb is not None:
            distance = calculator.calculate_distance(new_emb, prev_emb)
            dist_hist.append(distance)

            is_hist = state.get("is_score_history", [])
            gain = (is_hist[-1] - is_hist[-2]) if len(is_hist) >= 2 else 0.0

            logger.info(
                "[Convergence] Round %2d | Distance: %.6f | IS Gain: %+.4f",
                state.get("loop_count", 0), distance, gain,
            )
            _emit({
                "type":            "convergence_metrics",
                "round":           state.get("loop_count", 0),
                "distance":        round(distance, 6),
                "threshold":       CONVERGENCE_THRESHOLD,
                "is_gain":         round(gain, 4),
                "dist_history":    dist_hist,
                "timestamp":       _now(),
            })

        return {
            "previous_embedding": prev_emb,
            "current_embedding":  new_emb,
            "distance_history":   dist_hist,
        }

    # ── Node: evaluator ───────────────────────────────────────────────────────
    def evaluator_node(state: WorkflowState) -> dict:
        scenario   = state["scenario"]
        draft      = state["current_draft"]
        round_num  = state.get("loop_count", 0)
        is_history = list(state.get("is_score_history", []))

        # Inter-round throttle: give Groq's rate-limit token bucket time to refill.
        # Ragas makes ~20 calls per evaluation; without a pause the next round's
        # calls arrive before the window resets and cause another 429 burst.
        if round_num > 0 and provider == "groq":
            logger.info(
                "[Evaluator] Groq inter-round pause %.1fs to recover rate-limit...",
                GROQ_INTER_ROUND_SLEEP,
            )
            time.sleep(GROQ_INTER_ROUND_SLEEP)

        logger.info("[Evaluator] Computing IS for round %d...", round_num)

        dataset = Dataset.from_dict({
            "question":     [scenario["question"]],
            "answer":       [draft],
            "contexts":     [scenario["contexts"]],
            "ground_truth": [scenario["ground_truth"]],
        })

        eval_kwargs: dict = dict(
            dataset=dataset,
            metrics=ragas_metrics,
            llm=ragas_llm,
            embeddings=embeddings,
        )
        if _ragas_run_config is not None:
            eval_kwargs["run_config"] = _ragas_run_config

        # Emit a heartbeat BEFORE the blocking evaluate() call so the WebSocket
        # queue doesn't go silent for 3-4 minutes and trigger a timeout error.
        _emit({
            "type":      "evaluating",
            "round":     round_num,
            "message":   f"Running Ragas metrics (sequential, max_workers={_n_workers})…",
            "timestamp": _now(),
        })

        try:
            results = evaluate(**eval_kwargs)
        except Exception as exc:
            prev_score = is_history[-1] if is_history else 0.0
            logger.warning(
                "[Evaluator] Ragas evaluation failed entirely (%s). Carrying forward IS=%.4f.",
                exc, prev_score,
            )
            _emit({
                "type":    "is_score",
                "round":   round_num,
                "score":   prev_score,
                "metrics": dict(_last_metric_scores),
                "error":   str(exc),
                "timestamp": _now(),
            })
            return {"current_is_score": prev_score}

        per_metric: dict[str, float] = {}
        score = 0.0
        for metric in ragas_metrics:
            m_name = metric.name
            try:
                raw = results[m_name]
                val = float(raw[0]) if isinstance(raw, list) else float(raw)
                is_nan = val != val
            except Exception:
                is_nan = True
                val = 0.0

            if is_nan:
                # Carry forward the last known value for this metric instead of
                # defaulting to 0.0. A hard-failed Faithfulness of 0.0 drags IS
                # down by 0.25 per round and corrupts the convergence trajectory.
                val = _last_metric_scores.get(m_name, 0.0)
                logger.warning(
                    "  %s: NaN/failed — using last known %.4f", m_name, val
                )
            else:
                _last_metric_scores[m_name] = val

            per_metric[m_name] = round(val, 4)
            weight = is_weights.get(m_name, 0.25)
            score += weight * val
            logger.info("  %s: %.4f (w=%.2f)", m_name, val, weight)

        is_history.append(score)
        logger.info("⭐ Round IS: %.4f", score)

        _emit({
            "type":      "is_score",
            "round":     round_num,
            "score":     round(score, 4),
            "metrics":   per_metric,
            "weights":   is_weights,
            "history":   is_history,
            "timestamp": _now(),
        })

        return {"current_is_score": score, "is_score_history": is_history}

    # ── Conditional edge: check_convergence ───────────────────────────────────
    def check_convergence(state: WorkflowState) -> str:
        loop_count  = state.get("loop_count", 0)
        history     = state.get("history", [])
        dist_history = state.get("distance_history", [])
        is_history  = state.get("is_score_history", [])

        if not dist_history:
            logger.info("[Convergence] Initial pass — no distance yet, continuing.")
            return "critic"

        # Signal 1: Critic APPROVED
        if history:
            last_feedback = history[-1].get("feedback", "")
            if last_feedback.strip().upper().startswith("APPROVED"):
                _halt("critic_approved", loop_count, state, _emit)
                return "end"

        # Signal 2: Semantic entropy convergence (k=2 patience window)
        if len(dist_history) >= CONVERGENCE_PATIENCE:
            recent = dist_history[-CONVERGENCE_PATIENCE:]
            if all(d < CONVERGENCE_THRESHOLD for d in recent):
                logger.info(
                    "🛑 [HALT — ENTROPY] k=%d consecutive distances < %.4f: %s",
                    CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD,
                    [round(d, 6) for d in recent],
                )
                _halt("entropy_convergence", loop_count, state, _emit)
                return "end"

        # Signal 3: No Information Gain
        if loop_count >= MIN_ROUNDS_FOR_GAIN_CHECK and len(is_history) >= 2:
            gain = is_history[-1] - is_history[-2]
            if gain <= 0.0:
                logger.info(
                    "🛑 [HALT — NO IS GAIN] Δ IS = %.4f ≤ 0", gain
                )
                _halt("no_information_gain", loop_count, state, _emit)
                return "end"

        # Signal 4: Hard failsafe
        if loop_count >= MAX_ROUNDS:
            logger.warning(
                "⚠️  [HALT — FAILSAFE] Reached %d rounds.", MAX_ROUNDS
            )
            _halt("failsafe", loop_count, state, _emit)
            return "end"

        return "critic"

    # ── Assemble graph ────────────────────────────────────────────────────────
    writer_fn = make_writer_node(agent_llm, _emit)
    critic_fn = make_critic_node(agent_llm, _emit)

    wf = StateGraph(WorkflowState)
    wf.add_node("writer",      writer_fn)
    wf.add_node("evaluator",   evaluator_node)
    wf.add_node("embed_state", embed_state_node)
    wf.add_node("critic",      critic_fn)

    wf.set_entry_point("writer")
    wf.add_edge("writer",      "evaluator")
    wf.add_edge("evaluator",   "embed_state")
    wf.add_conditional_edges(
        "embed_state",
        check_convergence,
        {"critic": "critic", "end": END},
    )
    wf.add_edge("critic", "writer")

    logger.info("LangGraph compiled: writer→evaluator→embed_state→[critic|END]")
    return wf.compile()


# ─────────────────────────────────────────────────────────────
# Scenario runner
# ─────────────────────────────────────────────────────────────
def run_scenario(
    app,
    scenario: dict,
    emit: Emit | None = None,
) -> dict:
    """
    Execute the full Writer → Critic loop for one scenario until halted.

    Args:
        app:      Compiled LangGraph application from build_graph().
        scenario: Scenario dict from test_scenarios.json.
        emit:     Optional streaming callback.

    Returns:
        Structured result dict compatible with ragas_eval.py input.
    """
    _emit = emit or (lambda e: None)

    logger.info("=" * 65)
    logger.info("📋 SCENARIO: %s", scenario["id"].upper().replace("_", " "))
    logger.info("   Topic: %s", scenario["topic"])
    logger.info("=" * 65)

    _emit({
        "type":        "scenario_start",
        "scenario_id": scenario["id"],
        "topic":       scenario["topic"],
        "question":    scenario["question"],
        "timestamp":   _now(),
    })

    final = app.invoke({
        "scenario":           scenario,
        "current_draft":      "",
        "history":            [],
        "loop_count":         0,
        "previous_embedding": None,
        "current_embedding":  None,
        "distance_history":   [],
        "current_is_score":   0.0,
        "is_score_history":   [],
        "halt_reason":        "",
    })

    rounds      = final["loop_count"]
    final_draft = final["current_draft"]
    history     = final["history"]
    is_history  = final.get("is_score_history", [])
    dist_history = final.get("distance_history", [])

    # Determine halt reason — priority order MUST match check_convergence exactly.
    # Bug fix: old code checked IS gain before entropy distance, so when both
    # signals fired simultaneously the wrong reason was reported (no_information_gain
    # shown even though HALT—ENTROPY triggered the actual stop).
    if history and history[-1].get("feedback", "").strip().upper().startswith("APPROVED"):
        halt_reason = "critic_approved"
    elif rounds >= MAX_ROUNDS:
        halt_reason = "failsafe"
    elif (
        len(dist_history) >= CONVERGENCE_PATIENCE
        and all(d < CONVERGENCE_THRESHOLD for d in dist_history[-CONVERGENCE_PATIENCE:])
    ):
        # Signal 2 (entropy) takes priority over Signal 3 (IS gain) — same order
        # as check_convergence, which evaluates cosine distance before IS gain.
        halt_reason = "entropy_convergence"
    elif (
        len(is_history) >= MIN_ROUNDS_FOR_GAIN_CHECK
        and (is_history[-1] - is_history[-2]) <= 0.0
    ):
        halt_reason = "no_information_gain"
    else:
        halt_reason = "entropy_convergence"

    logger.info(
        "✅ Halted after %d round(s) — reason: %s | Final IS: %.4f",
        rounds, halt_reason, is_history[-1] if is_history else 0.0,
    )

    result = {
        "scenario_id":     scenario["id"],
        "topic":           scenario["topic"],
        "rounds":          rounds,
        "halt_reason":     halt_reason,
        "question":        scenario["question"],
        "answer":          final_draft,
        "contexts":        scenario["contexts"],
        "ground_truth":    scenario["ground_truth"],
        "is_score_history":  is_history,
        "dist_history":      dist_history,
        "history": [
            {"draft": h["draft"][:300], "feedback": h["feedback"]}
            for h in history
        ],
    }

    _emit({
        "type":              "scenario_complete",
        "scenario_id":       scenario["id"],
        "rounds":            rounds,
        "halt_reason":       halt_reason,
        "final_is_score":    is_history[-1] if is_history else 0.0,
        "is_score_history":  is_history,
        "dist_history":      dist_history,
        "timestamp":         _now(),
    })

    return result


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _halt(reason: str, loop_count: int, state: WorkflowState, emit: Emit) -> None:
    is_history = state.get("is_score_history", [])
    dist_history = state.get("distance_history", [])
    emit({
        "type":             "halt_signal",
        "reason":           reason,
        "round":            loop_count,
        "final_is_score":   is_history[-1] if is_history else 0.0,
        "final_distance":   dist_history[-1] if dist_history else None,
        "timestamp":        _now(),
    })


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHP Multi-Agent Workflow — Semantic Halting Problem",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test", "all"],
        help="Data split to process.",
    )
    parser.add_argument(
        "--provider", type=str, default=DEFAULT_PROVIDER,
        choices=["groq", "openai"],
        help="LLM provider.",
    )
    parser.add_argument("--agent-model", type=str, default=None)
    parser.add_argument("--eval-model",  type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(SCENARIOS_FILE):
        logger.error("Scenarios file not found: %s", SCENARIOS_FILE)
        sys.exit(1)

    with open(SCENARIOS_FILE, "r", encoding="utf-8") as fh:
        all_scenarios: list[dict] = json.load(fh)

    scenarios = (
        all_scenarios if args.split == "all"
        else [s for s in all_scenarios if s.get("split") == args.split]
    )

    if not scenarios:
        logger.error("No scenarios for split '%s'.", args.split)
        sys.exit(1)

    logger.info(
        "🌐 SHP — split=%s | %d scenario(s) | provider=%s | threshold=%.4f | patience=%d",
        args.split, len(scenarios), args.provider,
        CONVERGENCE_THRESHOLD, CONVERGENCE_PATIENCE,
    )

    app = build_graph(
        provider=args.provider,
        agent_model=args.agent_model,
        eval_model=args.eval_model,
    )

    results: list[dict] = []
    for scenario in scenarios:
        result = run_scenario(app, scenario)
        results.append(result)

    with open(AGENT_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("💾 Agent results saved → %s", AGENT_RESULTS_FILE)

    logger.info("═" * 65)
    logger.info("📊 SUMMARY")
    logger.info("═" * 65)
    for r in results:
        logger.info(
            "  [%s]  rounds=%d  halt=%s  IS=%.4f",
            r["scenario_id"], r["rounds"], r["halt_reason"],
            r["is_score_history"][-1] if r["is_score_history"] else 0.0,
        )
    logger.info("═" * 65)


if __name__ == "__main__":
    main()
