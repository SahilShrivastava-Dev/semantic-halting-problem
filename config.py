"""
config.py

Centralised configuration for the Semantic Halting Problem (SHP) pipeline.

All tuneable constants, file paths, and model identifiers are declared here so
that every other module imports from a single source of truth.  Changing a value
here propagates automatically across the entire codebase.

Environment variables (API keys) are loaded from the project-root .env file via
python-dotenv.  Never hard-code secrets; always use environment variables.
"""

import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────
load_dotenv()  # Loads GROQ_API_KEY, HF_HOME, etc. from .env

# ─────────────────────────────────────────────────────────────
# LLM / Embedding models
# ─────────────────────────────────────────────────────────────
# Writer and Critic agent LLM (Groq free-tier — fast inference, no local GPU).
AGENT_LLM_MODEL: str = "llama-3.1-8b-instant"
AGENT_LLM_TEMPERATURE: float = 0.0
AGENT_LLM_MAX_TOKENS: int = 700

# Judge LLM used by Ragas for in-loop evaluation and the stand-alone eval step.
EVAL_LLM_MODEL: str = "llama-3.1-8b-instant"
EVAL_LLM_TEMPERATURE: float = 0.0
EVAL_LLM_MAX_TOKENS: int = 2048
EVAL_LLM_MAX_RETRIES: int = 3

# Local sentence-embedding model used for:
#   (a) cosine-distance convergence check  (b) Ragas AnswerRelevancy scoring
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

# ─────────────────────────────────────────────────────────────
# Halting thresholds
# ─────────────────────────────────────────────────────────────
# Cosine distance below which consecutive drafts are declared semantically
# identical → semantic convergence halt.
# Calibration note: synonym rewrites ("cement" ↔ "concrete") score ~0.04,
# so 0.06 comfortably catches deadlocks while ignoring meaningful revision.
CONVERGENCE_THRESHOLD: float = 0.06

# Hard safety cap — prevents unbounded API usage in pathological inputs.
MAX_ROUNDS: int = 12

# Minimum number of completed rounds before the IS-Gain halt is evaluated.
# Prevents a false-positive halt on the very first iteration where gain = 0.
MIN_ROUNDS_FOR_GAIN_CHECK: int = 2

# ─────────────────────────────────────────────────────────────
# Retry policy
# ─────────────────────────────────────────────────────────────
MAX_LLM_RETRIES: int = 3   # Exponential-backoff attempts for transient errors

# ─────────────────────────────────────────────────────────────
# File paths  (all relative to the project root)
# ─────────────────────────────────────────────────────────────
SCENARIOS_FILE: str = "test_scenarios.json"
AGENT_RESULTS_FILE: str = "agent_results.json"
RAGAS_SCORES_FILE: str = "ragas_scores.json"
WEIGHTS_FILE: str = "optimized_weights.json"
IS_TEST_RESULTS_CSV: str = os.path.join("doc", "information_score_test_results.csv")

# ─────────────────────────────────────────────────────────────
# Information Score: default equal weights (overridden by optimized_weights.json)
# ─────────────────────────────────────────────────────────────
DEFAULT_IS_WEIGHTS: dict[str, float] = {
    "faithfulness":      0.25,
    "answer_relevancy":  0.25,
    "context_precision": 0.25,
    "context_recall":    0.25,
}

# ─────────────────────────────────────────────────────────────
# Ragas metric columns (must match ragas metric .name attributes)
# ─────────────────────────────────────────────────────────────
METRIC_COLS: list[str] = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# ─────────────────────────────────────────────────────────────
# Regression / optimization
# ─────────────────────────────────────────────────────────────
# Minimum rows in ragas_scores.json before real-data regression is attempted.
MIN_REAL_ROWS_FOR_REGRESSION: int = 2

# Synthetic fallback: number of mock rows generated when real data is scarce.
SYNTHETIC_FALLBACK_ROWS: int = 100

# Ground-truth weights embedded in the synthetic dataset (used for validation).
SYNTHETIC_TRUE_WEIGHTS: list[float] = [0.4, 0.3, 0.1, 0.2]

# R² thresholds for fit-quality messages
R2_STRONG_FIT: float = 0.75
R2_MODERATE_FIT: float = 0.50
