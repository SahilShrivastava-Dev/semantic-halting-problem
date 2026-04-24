"""
ragas_eval.py

Evaluates the final agent-produced drafts using the Ragas v0.4 framework.

Reads ``agent_results.json`` (written by ``agent_workflow.py``) and runs all
four RAG quality metrics against each scenario's final answer, using real LLM
calls — not cached or mocked data.

Metrics
-------
    Faithfulness       — Did the answer hallucinate, or is every claim
                         grounded in the provided contexts?
    Answer Relevancy   — Does the answer address the question that was asked?
    Context Precision  — Of the retrieved contexts, what fraction was useful?
    Context Recall     — Did the retrieved contexts cover all necessary
                         information?

Output
------
    Prints a per-scenario breakdown to stdout (via logging).
    Saves ``ragas_scores.json`` consumed by ``optimize_score.py``.

Usage
-----
    python ragas_eval.py
"""

import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config import (
    AGENT_RESULTS_FILE,
    EMBEDDING_MODEL_NAME,
    EVAL_LLM_MAX_TOKENS,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    METRIC_COLS,
    RAGAS_SCORES_FILE,
)

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
    Forces ``n=1`` on all Groq API calls.

    Ragas internally requests ``n=3`` for AnswerRelevancy; Groq rejects
    ``n > 1`` with a 400 error.  This subclass strips the ``n`` parameter
    before each request, making all Ragas metrics compatible with Groq.
    """

    def _create_message_dicts(self, messages, stop):
        message_dicts, params = super()._create_message_dicts(messages, stop)
        params.pop("n", None)
        return message_dicts, params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ─────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────
logger.info("Initialising Ragas judge LLM (%s) and embeddings (%s)...",
            EVAL_LLM_MODEL, EMBEDDING_MODEL_NAME)

_ragas_llm = LangchainLLMWrapper(_FixedChatGroq(
    model=EVAL_LLM_MODEL,
    temperature=EVAL_LLM_TEMPERATURE,
    max_tokens=EVAL_LLM_MAX_TOKENS,
))
_ragas_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

_METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]


# ─────────────────────────────────────────────────────────────
# Core evaluation helper
# ─────────────────────────────────────────────────────────────
def _evaluate_scenario(entry: dict) -> dict:
    """
    Run all four Ragas metrics against a single scenario result entry.

    Args:
        entry (dict): A single item from ``agent_results.json`` with keys
            ``question``, ``answer``, ``contexts``, and ``ground_truth``.

    Returns:
        dict: Metric name → float score mapping.  NaN values (metric
            evaluation failures) are normalised to 0.0.
    """
    dataset = Dataset.from_dict({
        "question":    [entry["question"]],
        "answer":      [entry["answer"]],
        "contexts":    [entry["contexts"]],
        "ground_truth": [entry["ground_truth"]],
    })

    result = evaluate(
        dataset=dataset,
        metrics=_METRICS,
        llm=_ragas_llm,
        embeddings=_ragas_emb,
    )

    scores: dict[str, float] = {}
    for metric in _METRICS:
        raw = result[metric.name]
        val = float(raw[0]) if isinstance(raw, list) else float(raw)
        scores[metric.name] = round(0.0 if (val != val) else val, 4)

    return scores


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """
    Load agent results, evaluate each scenario with Ragas, and save scores.

    Raises:
        SystemExit: If ``agent_results.json`` does not exist.
    """
    if not os.path.exists(AGENT_RESULTS_FILE):
        logger.error(
            "%s not found. Run agent_workflow.py first to generate agent outputs.",
            AGENT_RESULTS_FILE,
        )
        sys.exit(1)

    with open(AGENT_RESULTS_FILE, "r") as fh:
        agent_results: list[dict] = json.load(fh)

    logger.info(
        "Loaded %d scenario result(s) from %s — evaluating real agent outputs...",
        len(agent_results), AGENT_RESULTS_FILE,
    )

    all_scores: list[dict] = []

    for entry in agent_results:
        sid = entry["scenario_id"]
        logger.info("─" * 55)
        logger.info("📋 Evaluating: %s", sid.upper().replace("_", " "))
        logger.info("   Rounds until halt : %d", entry["rounds"])
        logger.info("   Question          : %s", entry["question"])
        logger.info("   Final answer tail : ...%s", entry["answer"][-80:])
        logger.info("─" * 55)

        scores = _evaluate_scenario(entry)

        for col in METRIC_COLS:
            logger.info("   %-22s %.4f", col + ":", scores.get(col, 0.0))

        all_scores.append({
            "scenario_id":       sid,
            "rounds":            entry["rounds"],
            "halt_reason":       entry.get("halt_reason", ""),
            "faithfulness":      scores.get("faithfulness", 0.0),
            "answer_relevancy":  scores.get("answer_relevancy", 0.0),
            "context_precision": scores.get("context_precision", 0.0),
            "context_recall":    scores.get("context_recall", 0.0),
        })

    with open(RAGAS_SCORES_FILE, "w") as fh:
        json.dump(all_scores, fh, indent=2)
    logger.info("💾 Ragas scores saved → %s", RAGAS_SCORES_FILE)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("📊 RAGAS EVALUATION SUMMARY")
    logger.info("═" * 55)
    for s in all_scores:
        avg = sum(s[col] for col in METRIC_COLS) / len(METRIC_COLS)
        logger.info(
            "  [%s]  avg=%.3f  rounds=%d  halt=%s",
            s["scenario_id"], avg, s["rounds"], s.get("halt_reason", ""),
        )
    logger.info("═" * 55)


if __name__ == "__main__":
    main()