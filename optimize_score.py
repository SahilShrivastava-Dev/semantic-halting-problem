"""
optimize_score.py

This module contains a machine learning script that uses Linear Regression to 
optimize the weights of a custom composite 'Information Score'. It models the 
relationship between base Ragas metrics and holistic human quality scores.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def optimize_information_score_weights():
    """
    Simulates a dataset of RAG interactions and uses Linear Regression 
    to derive the optimal weights for a custom 'Information Score'.
    
    This function generates mock Ragas evaluation scores (Faithfulness, 
    Answer Relevancy, Context Precision, Context Recall) alongside a 
    'Ground Truth Quality' score. It then fits a linear model to find 
    the mathematically optimal coefficients (weights) that maximize 
    alignment with human preference.
    """
    print("--- Information Score Optimizer ---")
    print("Generating mock dataset of 100 RAG interactions...\n")
    
    # 1. Generate a mock dataset representing Ragas evaluations
    # In reality, this data comes from the output of your ragas_eval.py runs.
    np.random.seed(42)
    num_samples = 100
    
    # Randomly generate metrics (0.0 to 1.0)
    faithfulness = np.random.uniform(0.3, 1.0, num_samples)
    answer_relevancy = np.random.uniform(0.4, 1.0, num_samples)
    context_precision = np.random.uniform(0.2, 0.9, num_samples)
    context_recall = np.random.uniform(0.4, 0.95, num_samples)
    
    # Let's say in our hypothetical business logic:
    # Faithfulness is VERY important (w=0.4)
    # Relevancy is moderately important (w=0.3)
    # Precision is least important (w=0.1)
    # Recall is somewhat important (w=0.2)
    # Plus some random human noise/disagreement
    true_weights = np.array([0.4, 0.3, 0.1, 0.2])
    noise = np.random.normal(0, 0.05, num_samples)
    
    # The Ground Truth Quality is what a Human (or GPT-4 Judge) rated the interaction
    human_quality_score = (
        (faithfulness * true_weights[0]) +
        (answer_relevancy * true_weights[1]) +
        (context_precision * true_weights[2]) +
        (context_recall * true_weights[3]) +
        noise
    )
    # Clip between 0 and 1
    human_quality_score = np.clip(human_quality_score, 0, 1)
    
    df = pd.DataFrame({
        'Faithfulness': faithfulness,
        'AnswerRelevancy': answer_relevancy,
        'ContextPrecision': context_precision,
        'ContextRecall': context_recall,
        'Human_Quality_Score': human_quality_score
    })
    
    print("Mock Dataset Sample:")
    print(df.head(), "\n")
    
    # 2. Train the Linear Regression Model to find the weights
    # Features (X) are the Ragas metrics, Target (y) is the Human Score
    X = df[['Faithfulness', 'AnswerRelevancy', 'ContextPrecision', 'ContextRecall']]
    y = df['Human_Quality_Score']
    
    model = LinearRegression(fit_intercept=False) # Force intercept to 0 so weights sum nicely
    model.fit(X, y)
    
    # 3. Extract and Normalize the Weights
    raw_weights = model.coef_
    normalized_weights = raw_weights / np.sum(raw_weights)
    
    print("--- Optimization Results ---")
    print("To maximize alignment with human preference, your Information Score formula should be:\n")
    
    print(f"Information Score = ")
    print(f"  ({normalized_weights[0]:.2f} * Faithfulness) +")
    print(f"  ({normalized_weights[1]:.2f} * AnswerRelevancy) +")
    print(f"  ({normalized_weights[2]:.2f} * ContextPrecision) +")
    print(f"  ({normalized_weights[3]:.2f} * ContextRecall)")
    
    print(f"\nModel R² Score (How well these weights predict human preference): {model.score(X, y):.4f}")

if __name__ == "__main__":
    optimize_information_score_weights()
