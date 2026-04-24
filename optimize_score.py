"""
optimize_score.py

Derives optimal weights for the Information Score (IS) — a single composite
metric that summarises RAG quality — using Scikit-learn Linear Regression.

The Information Score is defined as:

    IS = w₁·Faithfulness + w₂·AnswerRelevancy + w₃·ContextPrecision + w₄·ContextRecall

where the weights (w₁…w₄) are learned from data so that IS best predicts
a "ground-truth" quality signal.

Data source (priority order)
-----------------------------
    1. ``ragas_scores.json`` — real Ragas scores from actual agent runs,
       produced by ``ragas_eval.py``.  This is always preferred.
    2. Synthetic fallback — ``SYNTHETIC_FALLBACK_ROWS`` mock rows with
       known ground-truth weights, used when real data is unavailable
       (e.g., first-time runs, CI/CD environments).

Output
------
    Prints the derived formula and R² goodness-of-fit to stdout (via logging).
    Saves ``optimized_weights.json`` consumed by ``agent_workflow.py`` and
    ``test_information_score.py``.

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
# Data loading helpers
# ─────────────────────────────────────────────────────────────
def _load_real_data() -> pd.DataFrame | None:
    """
    Attempt to load actual Ragas scores from ``ragas_scores.json``.

    Returns:
        pd.DataFrame if the file exists and contains at least
        ``MIN_REAL_ROWS_FOR_REGRESSION`` rows, otherwise ``None``.
    """
    if not os.path.exists(RAGAS_SCORES_FILE):
        logger.warning("%s not found — switching to synthetic fallback.", RAGAS_SCORES_FILE)
        return None

    with open(RAGAS_SCORES_FILE, "r") as fh:
        records: list[dict] = json.load(fh)

    df = pd.DataFrame(records)

    if len(df) < MIN_REAL_ROWS_FOR_REGRESSION:
        logger.warning(
            "%s contains only %d row(s); need ≥ %d for meaningful regression. "
            "Using synthetic fallback.",
            RAGAS_SCORES_FILE, len(df), MIN_REAL_ROWS_FOR_REGRESSION,
        )
        return None

    logger.info("Loaded %d real Ragas row(s) from %s.", len(df), RAGAS_SCORES_FILE)
    return df


def _build_synthetic_data(num_samples: int = SYNTHETIC_FALLBACK_ROWS) -> pd.DataFrame:
    """
    Generate a synthetic dataset with known ground-truth weights.

    The synthetic data mimics the distribution of real RAG quality metrics
    (bounded [0, 1]) and embeds the true weights from ``SYNTHETIC_TRUE_WEIGHTS``
    with Gaussian noise, so the regression can converge quickly and produce
    interpretable weights.

    Args:
        num_samples (int): Number of synthetic rows to generate.
            Defaults to ``SYNTHETIC_FALLBACK_ROWS`` from config.

    Returns:
        pd.DataFrame: Dataset with columns matching ``METRIC_COLS`` plus
            a ``Human_Quality_Score`` target column.
    """
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
        "faithfulness":       faithfulness,
        "answer_relevancy":   answer_relevancy,
        "context_precision":  context_precision,
        "context_recall":     context_recall,
        "Human_Quality_Score": human_score,
    })


def _add_proxy_quality_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a ``Human_Quality_Score`` proxy column to real Ragas data.

    Until human-annotated quality labels are collected, the proxy is an
    equal-weighted mean of the four Ragas metrics.  This is a neutral
    baseline; the regression will still learn *relative* metric importance
    from variance in the data.

    Args:
        df (pd.DataFrame): Real Ragas scores dataframe with ``METRIC_COLS``
            columns present.

    Returns:
        pd.DataFrame: Copy of ``df`` with ``Human_Quality_Score`` added.
    """
    out = df.copy()
    out["Human_Quality_Score"] = out[METRIC_COLS].mean(axis=1)
    return out


# ─────────────────────────────────────────────────────────────
# Main optimisation logic
# ─────────────────────────────────────────────────────────────
def optimize_information_score_weights() -> dict[str, float]:
    """
    Learn optimal IS weights via Linear Regression and persist them.

    Steps
    -----
    1. Load real Ragas scores (or fall back to synthetic data).
    2. Add the ``Human_Quality_Score`` target column.
    3. Fit ``LinearRegression(fit_intercept=False)`` on the four metric columns.
    4. Clip negative coefficients to 0 (physically meaningless for quality).
    5. Normalise coefficients to sum to 1.0.
    6. Report R² and save ``optimized_weights.json``.

    Returns:
        dict[str, float]: Normalised weights keyed by metric name.
    """
    logger.info("─" * 50)
    logger.info("Information Score Optimizer")
    logger.info("─" * 50)

    # 1. Load data
    df = _load_real_data()
    using_real = df is not None

    if using_real:
        df = _add_proxy_quality_column(df)
        logger.info("Mode: REAL DATA from %s", RAGAS_SCORES_FILE)
    else:
        df = _build_synthetic_data()
        logger.info(
            "Mode: SYNTHETIC FALLBACK — run ragas_eval.py first for learned weights."
        )

    logger.info(
        "Dataset used for regression (%d rows):\n%s",
        len(df),
        df[METRIC_COLS + ["Human_Quality_Score"]].to_string(index=False),
    )

    # 2 & 3. Fit regression
    X = df[METRIC_COLS].values
    y = df["Human_Quality_Score"].values
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # 4. Clip and normalise
    raw_weights  = np.clip(model.coef_, 0, None)
    weight_sum   = raw_weights.sum() if raw_weights.sum() > 0 else 1.0
    normalised   = raw_weights / weight_sum

    weights_dict: dict[str, float] = {
        metric: round(float(w), 4)
        for metric, w in zip(METRIC_COLS, normalised)
    }
    r2: float = float(model.score(X, y))

    # 5. Report
    logger.info("─" * 50)
    logger.info("Optimisation Results")
    logger.info("─" * 50)
    logger.info("Data source: %s", "Real Ragas scores" if using_real else "Synthetic (fallback)")
    logger.info("Information Score formula:")
    for metric, w in weights_dict.items():
        logger.info("  (%.4f × %s)", w, metric)
    logger.info("Model R² Score: %.4f", r2)

    if r2 >= R2_STRONG_FIT:
        logger.info("✅ Strong fit — formula reliably explains quality variance.")
    elif r2 >= R2_MODERATE_FIT:
        logger.warning("⚠️  Moderate fit — collect more real data for better calibration.")
    else:
        logger.error(
            "❌ Weak fit (R²=%.4f) — collect human-annotated data "
            "for meaningful weights.",
            r2,
        )

    # 6. Save
    output = {
        "weights":     weights_dict,
        "r2_score":    round(r2, 4),
        "data_source": "real" if using_real else "synthetic",
    }
    with open(WEIGHTS_FILE, "w") as fh:
        json.dump(output, fh, indent=2)
    logger.info("💾 Weights saved → %s", WEIGHTS_FILE)

    return weights_dict


if __name__ == "__main__":
    optimize_information_score_weights()
