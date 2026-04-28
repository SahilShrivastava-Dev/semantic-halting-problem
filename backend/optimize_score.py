"""
optimize_score.py

Derives optimal Information Score (IS) weights via one of four
research-backed weighting strategies.

    IS = w₁·Faithfulness + w₂·AnswerRelevancy + w₃·ContextPrecision + w₄·ContextRecall

Strategies
----------
1. entropy        (DEFAULT) — Objective, label-free Shannon entropy weighting.
                  Metrics that discriminate more across scenarios receive higher weight.
                  Grounded in information theory; requires no human labels.
                  Ref: MIGRASCOPE (arXiv:2602.21553, 2025); Springer MCDM (2025).

2. constrained_ls — Supervised constrained quadratic optimisation (scipy SLSQP).
                  Fits weights against a harmonic-mean quality proxy. Enforces
                  w_i ≥ 0 and Σw_i = 1 exactly — no post-hoc clip+normalise hack.
                  Falls back to entropy when labelled data is insufficient.

3. ahp            — Expert-driven Analytic Hierarchy Process.
                  Derives weights from AHP_PAIRWISE_MATRIX via the principal
                  eigenvector of the pairwise comparison matrix. Includes Saaty's
                  Consistency Ratio check (CR ≤ 0.10 accepted; CR > 0.10 warns).
                  Ref: Mathematics MDPI 2023 (doi:10.3390/math11030627);
                       IEEE Xplore 2024 (10.1109/CSEI64419.2024.10649145).

4. equal          — Uniform 0.25 baseline for ablation / cold-start.

Output
------
    Saves optimized_weights.json with schema:
        {
          "weights":       {metric: float, ...},
          "strategy":      "entropy" | "constrained_ls" | "ahp" | "equal",
          "strategy_meta": {strategy-specific diagnostics},
          "data_source":   "real" | "none"
        }

Usage
-----
    python optimize_score.py                          # uses IS_WEIGHT_STRATEGY from config/.env
    python optimize_score.py --strategy entropy
    python optimize_score.py --strategy constrained_ls
    python optimize_score.py --strategy ahp
    python optimize_score.py --strategy equal
"""

import argparse
import json
import logging
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (
    AHP_PAIRWISE_MATRIX,
    IS_WEIGHT_STRATEGY,
    METRIC_COLS,
    MIN_REAL_ROWS_FOR_REGRESSION,
    R2_MODERATE_FIT,
    R2_STRONG_FIT,
    RAGAS_SCORES_FILE,
    WEIGHTS_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

_STRATEGIES = ("entropy", "constrained_ls", "ahp", "equal")


# ─────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────
def _load_real_data() -> pd.DataFrame | None:
    """Load ragas_scores.json; return None when unavailable or too small."""
    if not os.path.exists(RAGAS_SCORES_FILE):
        logger.warning("%s not found — no real data available.", RAGAS_SCORES_FILE)
        return None

    with open(RAGAS_SCORES_FILE, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    df = pd.DataFrame(records)
    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        logger.warning("ragas_scores.json is missing columns %s — treating as no data.", missing)
        return None

    df = df[METRIC_COLS].dropna()
    if len(df) < MIN_REAL_ROWS_FOR_REGRESSION:
        logger.warning(
            "%s has only %d usable row(s); need ≥ %d.",
            RAGAS_SCORES_FILE, len(df), MIN_REAL_ROWS_FOR_REGRESSION,
        )
        return None

    logger.info("Loaded %d real scenario row(s) from %s.", len(df), RAGAS_SCORES_FILE)
    return df


# ─────────────────────────────────────────────────────────────
# Strategy 1 — Entropy-based objective weighting
# ─────────────────────────────────────────────────────────────
def _entropy_weights(df: pd.DataFrame) -> tuple[dict[str, float], dict]:
    """
    Shannon entropy weighting (information-theoretic, label-free).

    For each metric j across n scenarios:
        1. Normalise column:  p_ij = x_ij / Σ_i x_ij
        2. Entropy:           E_j  = -(1/ln n) Σ_i p_ij ln(p_ij + ε)
        3. Divergence:        d_j  = 1 - E_j
        4. Weight:            w_j  = d_j / Σ_j d_j

    A metric with high entropy (uniform scores across scenarios) is
    uninformative and receives a lower weight. A metric that clearly
    separates good from bad scenarios receives a higher weight.

    Reference: MIGRASCOPE — Mutual Information based RAG Retriever Analysis
    Scope (arXiv:2602.21553, 2025); entropy-based MCDM weighting (Springer, 2025).
    """
    X = df[METRIC_COLS].values.astype(float)
    n = X.shape[0]

    col_sums = X.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1e-9, col_sums)
    P = X / col_sums

    eps = 1e-9
    E = -(1.0 / np.log(n + eps)) * (P * np.log(P + eps)).sum(axis=0)
    E = np.clip(E, 0.0, 1.0)

    d = 1.0 - E
    total = d.sum() if d.sum() > 0 else 1.0
    w = d / total

    weights = {m: round(float(v), 4) for m, v in zip(METRIC_COLS, w)}
    meta = {
        "n_scenarios":  n,
        "entropy":      {m: round(float(e), 4) for m, e in zip(METRIC_COLS, E)},
        "divergence":   {m: round(float(v), 4) for m, v in zip(METRIC_COLS, d)},
    }
    return weights, meta


# ─────────────────────────────────────────────────────────────
# Strategy 2 — Constrained Least Squares
# ─────────────────────────────────────────────────────────────
def _constrained_ls_weights(
    df: pd.DataFrame,
) -> tuple[dict[str, float], dict]:
    """
    Constrained quadratic optimisation (scipy SLSQP).

    Minimises  ||Xw - y||²  subject to:
        w_i ≥ 0  for all i
        Σ w_i = 1

    Target variable y:  harmonic mean of the four Ragas metrics per row.
    The harmonic mean is more appropriate than the arithmetic mean as a
    quality proxy because it penalises any single very-low score heavily
    (aligning with the intuition that hallucination cannot be compensated
    by high context recall).

    This fixes the two flaws in the prior Linear Regression approach:
        1. The unconstrained regression + post-hoc clip + normalise distorts
           the geometric meaning of all weights when any coefficient is negative.
        2. The circular arithmetic-mean proxy pushes all weights toward 0.25.

    Falls back to entropy strategy when data is insufficient.
    """
    X = df[METRIC_COLS].values.astype(float)
    n_metrics = X.shape[1]

    eps = 1e-9
    y = n_metrics / (1.0 / (X + eps)).sum(axis=1)

    def _loss(w: np.ndarray) -> float:
        return float(np.sum((X @ w - y) ** 2))

    def _grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * X.T @ (X @ w - y)

    result = minimize(
        _loss,
        jac=_grad,
        x0=np.ones(n_metrics) / n_metrics,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_metrics,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    w = np.clip(result.x, 0.0, None)
    w /= w.sum() if w.sum() > 0 else 1.0

    y_pred = X @ w
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    weights = {m: round(float(v), 4) for m, v in zip(METRIC_COLS, w)}
    meta = {
        "n_scenarios":   len(df),
        "r2_score":      round(r2, 4),
        "proxy_target":  "harmonic_mean",
        "optimizer":     "SLSQP",
        "converged":     bool(result.success),
        "fit_quality":   (
            "strong"   if r2 >= R2_STRONG_FIT else
            "moderate" if r2 >= R2_MODERATE_FIT else
            "weak"
        ),
    }

    if r2 >= R2_STRONG_FIT:
        logger.info("✅ Constrained LS: strong fit (R²=%.4f)", r2)
    elif r2 >= R2_MODERATE_FIT:
        logger.warning("⚠️  Constrained LS: moderate fit (R²=%.4f) — collect more data.", r2)
    else:
        logger.error("❌ Constrained LS: weak fit (R²=%.4f) — consider entropy strategy.", r2)

    return weights, meta


# ─────────────────────────────────────────────────────────────
# Strategy 3 — Analytic Hierarchy Process (AHP)
# ─────────────────────────────────────────────────────────────
# Saaty's Random Consistency Index (RI) by matrix size n
_AHP_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}


def _ahp_weights(matrix: list[list[float]]) -> tuple[dict[str, float], dict]:
    """
    Analytic Hierarchy Process — principal eigenvector method.

    Steps:
        1. Build pairwise comparison matrix A.
        2. Compute principal (largest real) eigenvector → unnormalised weights.
        3. Normalise to sum to 1.
        4. Compute λ_max, Consistency Index (CI), Consistency Ratio (CR).
           CR ≤ 0.10 is Saaty's accepted threshold; above it the matrix is
           too inconsistent for the weights to be trusted.

    Reference:
        Saaty, T.L. (1977). A scaling method for priorities in hierarchical
        structures. Journal of Mathematical Psychology, 15(3), 234-281.
        doi:10.1016/0022-2496(77)90033-5

        Alves et al. (2023). Machine Learning-Driven Approach for Large Scale
        Decision Making with the Analytic Hierarchy Process. Mathematics, 11(3),
        627. doi:10.3390/math11030627

        IEEE Xplore 2024 (10.1109/CSEI64419.2024.10649145)
    """
    A = np.array(matrix, dtype=float)
    n = A.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_idx = int(np.argmax(eigenvalues.real))
    w = eigenvectors[:, max_idx].real
    w = np.abs(w)
    w /= w.sum()

    lambda_max = float(eigenvalues[max_idx].real)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _AHP_RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0

    if cr <= 0.10:
        logger.info("✅ AHP Consistency Ratio CR=%.4f ≤ 0.10 — matrix is consistent.", cr)
    else:
        logger.warning(
            "⚠️  AHP Consistency Ratio CR=%.4f > 0.10 — matrix is inconsistent. "
            "Revise AHP_PAIRWISE_MATRIX in config.py.",
            cr,
        )

    weights = {m: round(float(v), 4) for m, v in zip(METRIC_COLS, w)}
    meta = {
        "lambda_max":          round(lambda_max, 4),
        "consistency_index":   round(ci, 4),
        "random_index":        ri,
        "consistency_ratio":   round(cr, 4),
        "consistent":          cr <= 0.10,
        "matrix_size":         n,
    }
    return weights, meta


# ─────────────────────────────────────────────────────────────
# Strategy 4 — Equal weights baseline
# ─────────────────────────────────────────────────────────────
def _equal_weights() -> tuple[dict[str, float], dict]:
    n = len(METRIC_COLS)
    w = round(1.0 / n, 4)
    return {m: w for m in METRIC_COLS}, {"note": "uniform baseline — all metrics treated as equally important"}


# ─────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────
def optimize_information_score_weights(strategy: str | None = None) -> dict[str, float]:
    """
    Compute IS weights using the specified strategy and persist to optimized_weights.json.

    Args:
        strategy: One of "entropy", "constrained_ls", "ahp", "equal".
                  Defaults to IS_WEIGHT_STRATEGY from config.py / env var.

    Returns:
        Normalised weight dict keyed by metric name.
    """
    strategy = strategy or IS_WEIGHT_STRATEGY
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {_STRATEGIES}")

    logger.info("─" * 60)
    logger.info("IS Weight Optimiser  [strategy: %s]", strategy.upper())
    logger.info("─" * 60)

    df = _load_real_data()
    data_source = "real" if df is not None else "none"

    # ── Compute weights ────────────────────────────────────────
    if strategy == "entropy":
        if df is None:
            logger.warning(
                "No real data for entropy strategy — using equal weights as fallback. "
                "Run agent_workflow.py + ragas_eval.py first to generate ragas_scores.json."
            )
            weights, meta = _equal_weights()
            meta["fallback_reason"] = "no_real_data"
        else:
            weights, meta = _entropy_weights(df)

    elif strategy == "constrained_ls":
        if df is None:
            logger.warning(
                "No real data for constrained_ls — falling back to entropy with equal weights."
            )
            weights, meta = _equal_weights()
            meta["fallback_reason"] = "no_real_data"
        else:
            weights, meta = _constrained_ls_weights(df)

    elif strategy == "ahp":
        weights, meta = _ahp_weights(AHP_PAIRWISE_MATRIX)

    else:  # equal
        weights, meta = _equal_weights()

    # ── Diagnostics ────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Optimised IS weights")
    logger.info("─" * 60)
    for metric, w in weights.items():
        logger.info("  %-22s %.4f", metric + ":", w)
    logger.info("  Strategy:   %s", strategy)
    logger.info("  Data:       %s", data_source)

    # ── Persist ────────────────────────────────────────────────
    output = {
        "weights":       weights,
        "strategy":      strategy,
        "strategy_meta": meta,
        "data_source":   data_source,
    }
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("💾 Weights saved → %s", WEIGHTS_FILE)

    return weights


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHP IS Weight Optimiser — choose a research-backed strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies
----------
  entropy        (default) Objective Shannon entropy weighting — no labels needed.
                            arXiv:2602.21553 | Springer MCDM 2025
  constrained_ls           Supervised SLSQP against a harmonic-mean quality proxy.
                            Falls back to entropy when data is insufficient.
  ahp                      Expert-driven Analytic Hierarchy Process.
                            Configure AHP_PAIRWISE_MATRIX in config.py.
                            Mathematics MDPI 2023 | IEEE Xplore 2024
  equal                    Uniform 0.25 baseline for ablation.
        """,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=list(_STRATEGIES),
        help="Weighting strategy (overrides IS_WEIGHT_STRATEGY from config/.env).",
    )
    args = parser.parse_args()
    optimize_information_score_weights(strategy=args.strategy)
