"""
optimize_score.py

Derives optimal Information Score (IS) weights via Linear Regression.

IS = w₁·Faithfulness + w₂·AnswerRelevancy + w₃·ContextPrecision + w₄·ContextRecall

Data priority
-------------
1. ragas_scores.json  — real Ragas scores from actual agent runs (preferred).
2. Synthetic fallback — SYNTHETIC_FALLBACK_ROWS mock rows when real data is
   scarce (useful for cold-start / CI environments).

Output
------
    Saves optimized_weights.json consumed by agent_workflow.py and
    test_information_score.py.

Usage
-----
    python optimize_score.py
"""

import json
import logging
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import (
    METRIC_COLS,
    MIN_REAL_ROWS_FOR_REGRESSION,
    R2_MODERATE_FIT,
    R2_STRONG_FIT,
    RAGAS_SCORES_FILE,
    SYNTHETIC_FALLBACK_ROWS,
    SYNTHETIC_TRUE_WEIGHTS,
    WEIGHTS_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────
def _load_real_data() -> pd.DataFrame | None:
    if not os.path.exists(RAGAS_SCORES_FILE):
        logger.warning("%s not found — using synthetic fallback.", RAGAS_SCORES_FILE)
        return None

    with open(RAGAS_SCORES_FILE, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    df = pd.DataFrame(records)
    if len(df) < MIN_REAL_ROWS_FOR_REGRESSION:
        logger.warning(
            "%s has only %d row(s); need ≥ %d. Using synthetic fallback.",
            RAGAS_SCORES_FILE, len(df), MIN_REAL_ROWS_FOR_REGRESSION,
        )
        return None

    logger.info("Loaded %d real rows from %s.", len(df), RAGAS_SCORES_FILE)
    return df


def _build_synthetic_data(num_samples: int = SYNTHETIC_FALLBACK_ROWS) -> pd.DataFrame:
    logger.info("Generating %d synthetic rows as regression fallback...", num_samples)
    rng = np.random.default_rng(seed=42)

    faithfulness      = rng.uniform(0.3,  1.0,  num_samples)
    answer_relevancy  = rng.uniform(0.4,  1.0,  num_samples)
    context_precision = rng.uniform(0.2,  0.9,  num_samples)
    context_recall    = rng.uniform(0.4,  0.95, num_samples)

    true_w = np.array(SYNTHETIC_TRUE_WEIGHTS, dtype=np.float64)
    noise  = rng.normal(0, 0.05, num_samples)

    human_score = np.clip(
        faithfulness      * true_w[0]
        + answer_relevancy  * true_w[1]
        + context_precision * true_w[2]
        + context_recall    * true_w[3]
        + noise,
        0.0, 1.0,
    )

    return pd.DataFrame({
        "faithfulness":        faithfulness,
        "answer_relevancy":    answer_relevancy,
        "context_precision":   context_precision,
        "context_recall":      context_recall,
        "Human_Quality_Score": human_score,
    })


def _add_proxy_quality_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Human_Quality_Score"] = out[METRIC_COLS].mean(axis=1)
    return out


# ─────────────────────────────────────────────────────────────
# Main optimisation
# ─────────────────────────────────────────────────────────────
def optimize_information_score_weights() -> dict[str, float]:
    """
    Learn IS weights via LinearRegression and persist to optimized_weights.json.

    Returns:
        Normalised weight dict keyed by metric name.
    """
    logger.info("─" * 50)
    logger.info("Information Score Optimizer")
    logger.info("─" * 50)

    df = _load_real_data()
    using_real = df is not None

    if using_real:
        df = _add_proxy_quality_column(df)
        logger.info("Mode: REAL DATA from %s", RAGAS_SCORES_FILE)
    else:
        df = _build_synthetic_data()
        logger.info("Mode: SYNTHETIC FALLBACK")

    X = df[METRIC_COLS].values
    y = df["Human_Quality_Score"].values
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    raw_weights  = np.clip(model.coef_, 0, None)
    weight_sum   = raw_weights.sum() if raw_weights.sum() > 0 else 1.0
    normalised   = raw_weights / weight_sum

    weights_dict: dict[str, float] = {
        metric: round(float(w), 4)
        for metric, w in zip(METRIC_COLS, normalised)
    }
    r2 = float(model.score(X, y))

    logger.info("─" * 50)
    logger.info("Optimisation Results")
    logger.info("─" * 50)
    logger.info("Data source: %s", "Real" if using_real else "Synthetic (fallback)")
    logger.info("IS formula:")
    for metric, w in weights_dict.items():
        logger.info("  (%.4f × %s)", w, metric)
    logger.info("R² Score: %.4f", r2)

    if r2 >= R2_STRONG_FIT:
        logger.info("✅ Strong fit — weights reliably explain quality variance.")
    elif r2 >= R2_MODERATE_FIT:
        logger.warning("⚠️  Moderate fit — collect more real data for better calibration.")
    else:
        logger.error("❌ Weak fit (R²=%.4f) — collect human-annotated data.", r2)

    output = {
        "weights":     weights_dict,
        "r2_score":    round(r2, 4),
        "data_source": "real" if using_real else "synthetic",
    }
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("💾 Weights saved → %s", WEIGHTS_FILE)

    return weights_dict


if __name__ == "__main__":
    optimize_information_score_weights()
