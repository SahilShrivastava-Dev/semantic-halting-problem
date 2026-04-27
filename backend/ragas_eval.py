"""
ragas_eval.py

Evaluates final agent-produced drafts using Ragas v0.4.

Reads agent_results.json and runs four RAG quality metrics against each
scenario's final answer using the configured LLM provider.

Usage
-----
    python ragas_eval.py
    python ragas_eval.py --provider openai --eval-model gpt-4o-mini
"""

import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config import (
    AGENT_RESULTS_FILE,
    DEFAULT_EVAL_MODELS,
    DEFAULT_PROVIDER,
    EMBEDDING_MODEL_NAME,
    METRIC_COLS,
    RAGAS_MAX_WORKERS,
    RAGAS_EVAL_TIMEOUT,
    RAGAS_SCORES_FILE,
)
from providers import get_ragas_llm

try:
    from ragas.run_config import RunConfig as _RagasRunConfig
    _HAS_RUN_CONFIG = True
except ImportError:
    _HAS_RUN_CONFIG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_all(
    provider: str = DEFAULT_PROVIDER,
    eval_model: str | None = None,
) -> list[dict]:
    """
    Load agent results and evaluate each with Ragas metrics.

    Returns:
        List of score dicts saved to RAGAS_SCORES_FILE.
    """
    eval_model = eval_model or DEFAULT_EVAL_MODELS[provider]

    if not os.path.exists(AGENT_RESULTS_FILE):
        logger.error(
            "%s not found. Run agent_workflow.py first.", AGENT_RESULTS_FILE
        )
        sys.exit(1)

    with open(AGENT_RESULTS_FILE, "r", encoding="utf-8") as fh:
        agent_results: list[dict] = json.load(fh)

    logger.info(
        "Loaded %d result(s) from %s — provider=%s eval_model=%s",
        len(agent_results), AGENT_RESULTS_FILE, provider, eval_model,
    )

    ragas_llm = get_ragas_llm(provider, eval_model)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    _n_workers = RAGAS_MAX_WORKERS if provider == "groq" else min(2, RAGAS_MAX_WORKERS + 1)
    _run_config = (
        _RagasRunConfig(max_workers=_n_workers, timeout=RAGAS_EVAL_TIMEOUT)
        if _HAS_RUN_CONFIG else None
    )

    all_scores: list[dict] = []

    for entry in agent_results:
        sid = entry["scenario_id"]
        logger.info("─" * 55)
        logger.info("Evaluating: %s", sid.upper().replace("_", " "))
        logger.info("  Rounds : %d  |  Halt : %s", entry["rounds"], entry.get("halt_reason", ""))

        dataset = Dataset.from_dict({
            "question":     [entry["question"]],
            "answer":       [entry["answer"]],
            "contexts":     [entry["contexts"]],
            "ground_truth": [entry["ground_truth"]],
        })

        eval_kwargs: dict = dict(
            dataset=dataset, metrics=metrics, llm=ragas_llm, embeddings=embeddings,
        )
        if _run_config is not None:
            eval_kwargs["run_config"] = _run_config

        result = evaluate(**eval_kwargs)

        scores: dict[str, float] = {}
        for metric in metrics:
            raw = result[metric.name]
            val = float(raw[0]) if isinstance(raw, list) else float(raw)
            scores[metric.name] = round(0.0 if (val != val) else val, 4)
            logger.info("  %-22s %.4f", metric.name + ":", scores[metric.name])

        all_scores.append({
            "scenario_id":       sid,
            "rounds":            entry["rounds"],
            "halt_reason":       entry.get("halt_reason", ""),
            "faithfulness":      scores.get("faithfulness", 0.0),
            "answer_relevancy":  scores.get("answer_relevancy", 0.0),
            "context_precision": scores.get("context_precision", 0.0),
            "context_recall":    scores.get("context_recall", 0.0),
        })

    with open(RAGAS_SCORES_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_scores, fh, indent=2)
    logger.info("💾 Ragas scores saved → %s", RAGAS_SCORES_FILE)

    logger.info("═" * 55)
    logger.info("RAGAS SUMMARY")
    logger.info("═" * 55)
    for s in all_scores:
        avg = sum(s[col] for col in METRIC_COLS) / len(METRIC_COLS)
        logger.info(
            "  [%s]  avg=%.3f  rounds=%d  halt=%s",
            s["scenario_id"], avg, s["rounds"], s.get("halt_reason", ""),
        )
    logger.info("═" * 55)

    return all_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="SHP Ragas Evaluation")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=["groq", "openai"])
    parser.add_argument("--eval-model", type=str, default=None)
    args = parser.parse_args()
    evaluate_all(provider=args.provider, eval_model=args.eval_model)


if __name__ == "__main__":
    main()
