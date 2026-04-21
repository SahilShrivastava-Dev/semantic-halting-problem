"""
optimize_score.py

Uses Linear Regression to derive optimal weights for the 'Information Score' —
a single composite metric that summarises RAG quality.

Data source (priority order):
    1. ragas_scores.json  — real Ragas scores from the actual agent runs.
       This is the ground-truth data produced by ragas_eval.py.
    2. Synthetic fallback — 100 mock rows with known weights, used when
       ragas_scores.json does not exist (e.g., first-time / demo runs).

Output:
    - Prints the derived Information Score formula.
    - Prints the R² score (how well the formula fits the data).
    - Saves learned weights to optimized_weights.json for test_information_score.py.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

RAGAS_SCORES_FILE   = "ragas_scores.json"
WEIGHTS_OUTPUT_FILE = "optimized_weights.json"

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_real_data() -> pd.DataFrame | None:
    """
    Attempts to load actual Ragas scores from ragas_scores.json.

    Returns:
        pd.DataFrame if the file exists and has enough rows, else None.
    """
    if not os.path.exists(RAGAS_SCORES_FILE):
        return None
    with open(RAGAS_SCORES_FILE, "r") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    # Need at least 2 data points for regression
    if len(df) < 2:
        print(f"[Warning] {RAGAS_SCORES_FILE} has only {len(df)} row(s). "
              "Using synthetic fallback for meaningful regression.")
        return None
    print(f"[Info] Loaded {len(df)} real Ragas score row(s) from {RAGAS_SCORES_FILE}")
    return df


def build_synthetic_data(num_samples: int = 100) -> pd.DataFrame:
    """
    Generates a synthetic dataset of RAG interactions with known ground-truth weights.

    Used as a fallback when real Ragas scores are unavailable.

    Args:
        num_samples (int): Number of synthetic rows to generate.

    Returns:
        pd.DataFrame: Dataset with metric columns + Human_Quality_Score.
    """
    print(f"[Info] Generating {num_samples} synthetic rows (fallback mode)...")
    np.random.seed(42)
    faithfulness      = np.random.uniform(0.3, 1.0, num_samples)
    answer_relevancy  = np.random.uniform(0.4, 1.0, num_samples)
    context_precision = np.random.uniform(0.2, 0.9, num_samples)
    context_recall    = np.random.uniform(0.4, 0.95, num_samples)

    # Ground-truth weights for the synthetic set
    true_weights = np.array([0.4, 0.3, 0.1, 0.2])
    noise = np.random.normal(0, 0.05, num_samples)
    human_score = np.clip(
        faithfulness * true_weights[0] +
        answer_relevancy * true_weights[1] +
        context_precision * true_weights[2] +
        context_recall * true_weights[3] + noise,
        0, 1
    )
    return pd.DataFrame({
        "faithfulness":       faithfulness,
        "answer_relevancy":   answer_relevancy,
        "context_precision":  context_precision,
        "context_recall":     context_recall,
        "Human_Quality_Score": human_score
    })


def derive_human_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    When using real Ragas data, there is no 'Human_Quality_Score' column.
    We approximate it as the equal-weighted average of all four metrics,
    which serves as a neutral proxy until human annotations are collected.

    Args:
        df (pd.DataFrame): Real Ragas scores dataframe.

    Returns:
        pd.DataFrame: Same dataframe with an added 'Human_Quality_Score' column.
    """
    df = df.copy()
    df["Human_Quality_Score"] = df[METRIC_COLS].mean(axis=1)
    return df


def optimize_information_score_weights():
    """
    Main function: loads data, trains the Linear Regression model, and
    prints + saves the optimized Information Score formula.

    Steps:
        1. Load real data from ragas_scores.json (or fall back to synthetic).
        2. Fit LinearRegression(X=metrics, y=human_score).
        3. Normalise coefficients so they sum to 1.0.
        4. Print the formula and R² score.
        5. Save weights to optimized_weights.json.
    """
    print("\n--- Information Score Optimizer ---")

    # 1. Load data
    df = load_real_data()
    using_real = df is not None
    if using_real:
        df = derive_human_score_column(df)
        print(f"   Mode: REAL DATA from {RAGAS_SCORES_FILE}")
    else:
        df = build_synthetic_data()
        print("   Mode: SYNTHETIC FALLBACK (run ragas_eval.py first for real weights)")

    print("\nDataset used for regression:")
    print(df[METRIC_COLS + ["Human_Quality_Score"]].to_string(index=False), "\n")

    # 2. Train Linear Regression
    X = df[METRIC_COLS]
    y = df["Human_Quality_Score"]
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # 3. Normalise weights
    raw_weights = model.coef_
    # Clip any negatives to 0 before normalising (can occur with tiny real datasets)
    raw_weights = np.clip(raw_weights, 0, None)
    weight_sum = np.sum(raw_weights) if np.sum(raw_weights) > 0 else 1.0
    normalized = raw_weights / weight_sum

    weights_dict = {metric: round(float(w), 4) for metric, w in zip(METRIC_COLS, normalized)}
    r2 = model.score(X, y)

    # 4. Print results
    print("--- Optimization Results ---")
    print(f"Data source: {'Real Ragas scores' if using_real else 'Synthetic (fallback)'}")
    print("\nInformation Score formula:")
    for metric, w in weights_dict.items():
        print(f"  ({w:.2f} * {metric})")
    print(f"\nModel R² Score: {r2:.4f}")
    if r2 >= 0.75:
        print("✅ Strong fit — formula reliably explains quality variance.")
    elif r2 >= 0.5:
        print("⚠️  Moderate fit — formula is reasonable but more data would help.")
    else:
        print("❌ Weak fit — collect more real human-annotated data for better weights.")

    # 5. Save weights
    output = {"weights": weights_dict, "r2_score": round(r2, 4),
              "data_source": "real" if using_real else "synthetic"}
    with open(WEIGHTS_OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Weights saved to {WEIGHTS_OUTPUT_FILE}")


if __name__ == "__main__":
    optimize_information_score_weights()
