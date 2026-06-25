"""
judge.py

RAGAS quality judge wrapper — computes the four metrics + Information Score for a
single (question, answer, contexts, ground_truth) tuple, with token usage
attributed to the "judge" role via METER.

This isolates the binding cost of the whole study (LLM-judge calls on Groq free
tier) behind one cached call site. The harness invokes ``score`` at most once per
distinct draft (caching keyed on the draft hash), so identical drafts are never
re-judged.

Reuses ``providers.get_ragas_llm`` and the same four metrics the production
``evaluator_node`` uses, so the IS computed here matches deployment.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

try:
    from ragas.run_config import RunConfig as _RagasRunConfig
    _HAS_RUN_CONFIG = True
except ImportError:
    _HAS_RUN_CONFIG = False

from langchain_huggingface import HuggingFaceEmbeddings

from shp.config import (
    EMBEDDING_MODEL_NAME,
    METRIC_COLS,
    RAGAS_EVAL_TIMEOUT,
    RAGAS_MAX_WORKERS,
    DEFAULT_EVAL_MODELS,
)
from shp.providers import get_ragas_llm
from shp.token_meter import METER

logger = logging.getLogger(__name__)


class RagasJudge:
    """Builds the judge LLM + metrics once; ``score`` evaluates one draft."""

    def __init__(self, provider: str, eval_model: str | None, is_weights: Dict[str, float]):
        eval_model = eval_model or DEFAULT_EVAL_MODELS[provider]
        self.provider = provider
        self.eval_model = eval_model
        self.is_weights = is_weights
        self.ragas_llm = get_ragas_llm(provider, eval_model)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
        # Groq free tier 429s with any parallelism → 1. NVIDIA build also rate-limits,
        # so keep modest concurrency (2) and rely on the SDK's 429 backoff (max_retries
        # set on the model in providers.py). OpenAI → 2.
        n_workers = {"groq": RAGAS_MAX_WORKERS}.get(provider, 2)
        self.run_config = (
            _RagasRunConfig(max_workers=n_workers, timeout=RAGAS_EVAL_TIMEOUT)
            if _HAS_RUN_CONFIG else None
        )
        logger.info("RagasJudge ready: provider=%s model=%s workers=%d",
                    provider, eval_model, n_workers)

    def information_score(self, metrics: Dict[str, float]) -> float:
        return float(sum(self.is_weights.get(m, 0.0) * metrics.get(m, 0.0) for m in METRIC_COLS))

    def score(self, question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict:
        """
        Evaluate one draft. Returns {metrics, information_score, judge_tokens}.
        Token usage during the evaluate() call is captured under the judge role.
        """
        dataset = Dataset.from_dict({
            "question":     [question],
            "answer":       [answer],
            "contexts":     [contexts],
            "ground_truth": [ground_truth],
        })
        eval_kwargs = dict(dataset=dataset, metrics=self.metrics,
                           llm=self.ragas_llm, embeddings=self.embeddings)
        if self.run_config is not None:
            eval_kwargs["run_config"] = self.run_config

        METER.reset("judge")
        with METER.scope("judge"):
            results = evaluate(**eval_kwargs)
        judge_tokens = METER.snapshot().get("judge", {}).get("total_tokens", 0)

        per_metric: Dict[str, float] = {}
        for metric in self.metrics:
            name = metric.name
            try:
                raw = results[name]
                val = float(raw[0]) if isinstance(raw, list) else float(raw)
                if val != val:  # NaN
                    val = 0.0
            except Exception:
                val = 0.0
            per_metric[name] = round(val, 4)

        return {
            "metrics": per_metric,
            "information_score": round(self.information_score(per_metric), 4),
            "judge_tokens": judge_tokens,
        }
