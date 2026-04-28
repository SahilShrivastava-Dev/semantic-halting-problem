"""
config.py

Centralised, provider-agnostic configuration for the SHP pipeline.
All secrets are loaded from a .env file — never hard-coded here.

Supports:
    - Groq  (free-tier, llama-3.1 / llama-3.3 / mixtral)
    - OpenAI (gpt-4o-mini / gpt-4o)

Set PROVIDER=groq or PROVIDER=openai in your .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Provider selection
# ─────────────────────────────────────────────────────────────
DEFAULT_PROVIDER: str = os.getenv("PROVIDER", "groq")

PROVIDER_MODELS: dict[str, list[str]] = {
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
    ],
}

DEFAULT_AGENT_MODELS: dict[str, str] = {
    "groq":   "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
}

DEFAULT_EVAL_MODELS: dict[str, str] = {
    "groq":   "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
}

# ─────────────────────────────────────────────────────────────
# LLM parameters
# ─────────────────────────────────────────────────────────────
AGENT_LLM_TEMPERATURE: float = 0.0
AGENT_LLM_MAX_TOKENS: int = 700

EVAL_LLM_TEMPERATURE: float = 0.0
# 4096 required for Faithfulness: it must extract every claim from the full draft.
# 2048 causes LLMDidNotFinishException on drafts longer than ~400 words.
EVAL_LLM_MAX_TOKENS: int = 4096
EVAL_LLM_MAX_RETRIES: int = 3

# ─────────────────────────────────────────────────────────────
# Embedding model — always local, provider-independent
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

# ─────────────────────────────────────────────────────────────
# Halting thresholds
# ─────────────────────────────────────────────────────────────
# Cosine distance below which consecutive drafts are declared semantically
# identical. ~0.04 = synonym-level; 0.06 catches deadlocks comfortably.
CONVERGENCE_THRESHOLD: float = 0.06

# k=2 patience window: halt only when distance stays below threshold for
# this many consecutive rounds — mirrors ML early-stopping patience.
CONVERGENCE_PATIENCE: int = 2

MAX_ROUNDS: int = 12
MIN_ROUNDS_FOR_GAIN_CHECK: int = 2

# ─────────────────────────────────────────────────────────────
# Ragas evaluation concurrency
# ─────────────────────────────────────────────────────────────
# Groq free tier: ~30 req/min. Running 4 metrics in parallel = instant 429 storm.
# max_workers=1 serialises metric evaluation; slower but reliably avoids rate limits.
# Increase to 2-4 if using OpenAI (higher rate limits).
RAGAS_MAX_WORKERS: int = 1
RAGAS_EVAL_TIMEOUT: int = 300   # seconds per metric evaluation

# Seconds to sleep between rounds when using Groq, to let the rate-limit bucket recover.
GROQ_INTER_ROUND_SLEEP: float = 3.0

# ─────────────────────────────────────────────────────────────
# Retry policy
# ─────────────────────────────────────────────────────────────
MAX_LLM_RETRIES: int = 3

# ─────────────────────────────────────────────────────────────
# File paths (relative to backend/)
# ─────────────────────────────────────────────────────────────
SCENARIOS_FILE: str = "test_scenarios.json"
AGENT_RESULTS_FILE: str = "agent_results.json"
RAGAS_SCORES_FILE: str = "ragas_scores.json"
WEIGHTS_FILE: str = "optimized_weights.json"
IS_TEST_RESULTS_CSV: str = os.path.join("doc", "information_score_test_results.csv")

# ─────────────────────────────────────────────────────────────
# Information Score weights (equal defaults; replaced by optimized_weights.json)
# ─────────────────────────────────────────────────────────────
DEFAULT_IS_WEIGHTS: dict[str, float] = {
    "faithfulness":      0.25,
    "answer_relevancy":  0.25,
    "context_precision": 0.25,
    "context_recall":    0.25,
}

METRIC_COLS: list[str] = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# ─────────────────────────────────────────────────────────────
# IS weight optimisation strategy
# ─────────────────────────────────────────────────────────────
# Select via env var IS_WEIGHT_STRATEGY or the --strategy CLI flag on optimize_score.py.
#
#   "entropy"        (DEFAULT) — Objective, label-free Shannon entropy weighting.
#                    Metrics that vary more across scenarios carry more discriminative
#                    information and receive higher weight. No human labels required.
#                    Ref: MIGRASCOPE (arXiv:2602.21553); Springer MCDM 2025.
#
#   "constrained_ls" — Supervised constrained quadratic optimisation (scipy SLSQP).
#                    Fits weights against a harmonic-mean quality proxy; enforces
#                    w_i ≥ 0 and Σw_i = 1 exactly — no post-hoc clip+normalise.
#                    Falls back to entropy when fewer than MIN_REAL_ROWS rows exist.
#
#   "ahp"            — Expert-driven Analytic Hierarchy Process. Derives weights from
#                    AHP_PAIRWISE_MATRIX via the principal eigenvector; includes
#                    Saaty's Consistency Ratio check (CR ≤ 0.10 accepted).
#                    Ref: Mathematics MDPI 2023 (doi:10.3390/math11030627);
#                         IEEE Xplore 2024 (10.1109/CSEI64419.2024.10649145).
#
#   "equal"          — Uniform 0.25 baseline for ablation / cold-start.
IS_WEIGHT_STRATEGY: str = os.getenv("IS_WEIGHT_STRATEGY", "entropy")

# AHP pairwise comparison matrix
# Rows / Columns: [faithfulness, answer_relevancy, context_precision, context_recall]
# Saaty scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme advantage
#
# Justification (RAG domain knowledge):
#   Faithfulness (hallucination prevention) is most critical — a factually incorrect
#   answer is unusable regardless of other scores.  Answer relevancy comes second.
#   Context recall ensures key information is not omitted.  Context precision has
#   diminishing benefit when recall is already high.
#
# Consistency check (geometric-mean approximation):
#   λ_max ≈ 4.057  |  CI = (λ_max - n)/(n-1) ≈ 0.019  |  RI(4) = 0.90
#   CR = CI / RI ≈ 0.021  <  0.10  ✓  (Saaty's threshold)
AHP_PAIRWISE_MATRIX: list[list[float]] = [
    #  F      AR     CP     CR
    [1.000, 2.000, 5.000, 3.000],   # faithfulness
    [0.500, 1.000, 4.000, 2.000],   # answer_relevancy
    [0.200, 0.250, 1.000, 0.333],   # context_precision
    [0.333, 0.500, 3.000, 1.000],   # context_recall
]

# Minimum real scenario rows before constrained_ls accepts real data
# (falls back to entropy below this threshold rather than synthetic data)
MIN_REAL_ROWS_FOR_REGRESSION: int = 2
R2_STRONG_FIT: float = 0.75
R2_MODERATE_FIT: float = 0.50
