"""
test_information_score.py

Validates the Information Score formula across multiple answer quality levels.

Reads optimized_weights.json produced by optimize_score.py and applies the
derived formula to three quality variants for each scenario:
    - good:       A complete, faithful, factual answer.
    - convergent: A semantically identical paraphrase (simulates deadlock output).
    - poor:       An incorrect / hallucinated answer.

Expected ordering if the formula works correctly:
    IS(good) > IS(convergent) > IS(poor)

Saves full results to doc/information_score_test_results.csv.
"""
import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Load optimized weights (or use defaults if not yet generated)
# ─────────────────────────────────────────────────────────────
WEIGHTS_FILE = "optimized_weights.json"

if os.path.exists(WEIGHTS_FILE):
    with open(WEIGHTS_FILE, "r") as f:
        weight_data = json.load(f)
    IS_WEIGHTS = weight_data["weights"]
    data_source = weight_data.get("data_source", "unknown")
    r2 = weight_data.get("r2_score", "N/A")
    print(f"[Info] Loaded weights from {WEIGHTS_FILE} (source={data_source}, R²={r2})")
else:
    print(f"[Warning] {WEIGHTS_FILE} not found. Using default weights.")
    print("          Run optimize_score.py first for learned weights.")
    IS_WEIGHTS = {
        "faithfulness":      0.39,
        "answer_relevancy":  0.30,
        "context_precision": 0.09,
        "context_recall":    0.23,
    }

METRIC_COLS = list(IS_WEIGHTS.keys())


def compute_information_score(scores: dict) -> float:
    """
    Applies the Information Score formula to a dict of Ragas metric scores.

    Args:
        scores (dict): Ragas metric name → float value.

    Returns:
        float: Composite Information Score (0.0 – 1.0).
    """
    return round(sum(IS_WEIGHTS[m] * scores.get(m, 0.0) for m in METRIC_COLS), 4)


# ─────────────────────────────────────────────────────────────
# Validation test cases — 3 quality levels per scenario
# ─────────────────────────────────────────────────────────────
TEST_CASES = {
    "dubai_real_estate": {
        "question": "What structural methods and materials ensure the integrity of the Dubai high-rise foundation?",
        "contexts": [
            "High-rise buildings in Dubai's coastal sandy soil typically require deep pile foundations drilled 30-50 meters into bedrock to prevent differential settlement.",
            "Reinforced concrete pile caps connected by ground beams form the standard foundation transfer system for Dubai skyscrapers, distributing structural loads evenly.",
            "Grade 60 deformed steel rebar embedded in high-strength concrete (C50 or above) is mandated by Dubai Municipality for structural columns and shear walls.",
            "Dubai Building Code Section 4.2 requires corrosion-resistant coatings on steel reinforcement due to the high salinity of groundwater."
        ],
        "ground_truth": "The Dubai high-rise uses deep reinforced concrete pile foundations (30-50m into bedrock) with Grade 60 steel rebar in C50+ concrete and corrosion-resistant coatings per Dubai Building Code Section 4.2.",
        "answers": {
            "good": (
                "The structural integrity of the Dubai high-rise is ensured through deep reinforced concrete pile foundations drilled 30–50 metres into bedrock, preventing differential settlement in the coastal sandy soil. "
                "Pile caps connected by reinforced ground beams distribute vertical loads uniformly across the foundation. Grade 60 deformed steel rebar embedded in C50+ high-strength concrete is used throughout structural columns and shear walls, as mandated by Dubai Municipality. "
                "All steel reinforcement is coated with corrosion-resistant treatments in compliance with Dubai Building Code Section 4.2, protecting against high groundwater salinity. "
                "Post-tensioned flat-plate slabs further reduce floor-to-floor height while maintaining structural performance under high vertical loads."
            ),
            "convergent": (
                "The foundation of the Dubai high-rise relies on deep pile foundations embedded in bedrock, with reinforced concrete pile caps distributing loads. "
                "Grade 60 rebar set in high-grade concrete meets Dubai Municipality requirements, and corrosion-resistant coatings on all steel comply with Section 4.2 of the Dubai Building Code."
            ),
            "poor": (
                "The Dubai high-rise uses a standard wooden post-and-beam foundation system with bamboo reinforcement, which is lightweight and eco-friendly. "
                "The structure uses pre-cast aluminum panels for the load-bearing walls, reducing construction time. No specific building code requirements apply to this type of structure."
            ),
        }
    },
    "medical_discharge": {
        "question": "What treatment was administered for the hypertensive crisis and what is the patient's discharge plan?",
        "contexts": [
            "A hypertensive crisis (BP > 180/120 mmHg) requires immediate IV labetalol or hydralazine for rapid blood pressure reduction.",
            "Target blood pressure reduction in hypertensive emergencies is no more than 25% within the first hour to prevent cerebral hypoperfusion.",
            "Amlodipine 5mg OD combined with ramipril 5mg OD is a standard oral antihypertensive regimen post-crisis.",
            "Patients require 48-72 hour cardiology follow-up and weekly BP monitoring for the first month post-discharge.",
            "Renal function (serum creatinine, eGFR) must be rechecked within one week post-discharge."
        ],
        "ground_truth": "IV labetalol was administered for acute BP reduction (target <25% in hour 1), followed by oral amlodipine 5mg and ramipril 5mg. Discharged in stable condition with 48-72 hour cardiology follow-up, weekly BP monitoring, and renal function recheck within one week.",
        "answers": {
            "good": (
                "The patient presented with a hypertensive crisis at 195/115 mmHg and received IV labetalol titrated to achieve a controlled reduction of no more than 25% within the first hour, preventing cerebral hypoperfusion. "
                "Following haemodynamic stabilisation, an oral antihypertensive regimen of amlodipine 5mg OD and ramipril 5mg OD was commenced. "
                "The patient is discharged in stable condition with instructions for: (1) a 48–72 hour cardiology follow-up appointment, (2) weekly home blood pressure monitoring for the first month, (3) renal function panel (serum creatinine, eGFR) within one week, and (4) lifestyle modifications including sodium restriction to <2g/day and 30 minutes of daily aerobic exercise."
            ),
            "convergent": (
                "The patient was treated with intravenous labetalol to lower blood pressure acutely, with a target reduction under 25% in the first hour. "
                "Oral amlodipine and ramipril were started for ongoing control. The patient is discharged in a stable state with a follow-up scheduled with cardiology within 48-72 hours and renal labs within one week."
            ),
            "poor": (
                "The patient was given aspirin and ibuprofen to manage headache associated with the elevated blood pressure. "
                "A low-sodium diet was recommended. The patient was advised to return to the emergency department if symptoms recur. No specialist follow-up was arranged."
            ),
        }
    }
}


# ─────────────────────────────────────────────────────────────
# Model Setup
# ─────────────────────────────────────────────────────────────
print("[Setup] Initialising models for Ragas evaluation (Groq judge + BAAI embeddings)...")
ragas_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
))
ragas_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]


def evaluate_single(question, answer, contexts, ground_truth) -> dict:
    """
    Runs all four Ragas metrics on a single (question, answer, contexts, ground_truth) row.

    Args:
        question (str):     The question.
        answer (str):       The answer being evaluated.
        contexts (list):    Retrieved context chunks.
        ground_truth (str): Reference answer.

    Returns:
        dict: Metric name → float score.
    """
    ds = Dataset.from_dict({
        "question":     [question],
        "answer":       [answer],
        "contexts":     [contexts],
        "ground_truth": [ground_truth],
    })
    result = evaluate(dataset=ds, metrics=METRICS, llm=ragas_llm, embeddings=ragas_emb)
    return {
        m.name: round(float(result[m.name][0])
                      if isinstance(result[m.name], list)
                      else float(result[m.name]), 4)
        for m in METRICS
    }


# ─────────────────────────────────────────────────────────────
# Main Validation Loop
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'═'*65}")
    print("⚖️  INFORMATION SCORE VALIDATION")
    print(f"   Weights: { {k: v for k, v in IS_WEIGHTS.items()} }")
    print(f"{'═'*65}")

    all_rows = []

    for scenario_id, data in TEST_CASES.items():
        print(f"\n{'━'*65}")
        print(f"📋 Scenario: {scenario_id.upper().replace('_', ' ')}")
        print(f"{'━'*65}")

        for quality, answer in data["answers"].items():
            print(f"\n  ▶ [{quality.upper()}]  {answer[:80]}...")

            scores     = evaluate_single(data["question"], answer, data["contexts"], data["ground_truth"])
            info_score = compute_information_score(scores)

            print(f"    Faithfulness:      {scores['faithfulness']:.4f}")
            print(f"    Answer Relevancy:  {scores['answer_relevancy']:.4f}")
            print(f"    Context Precision: {scores['context_precision']:.4f}")
            print(f"    Context Recall:    {scores['context_recall']:.4f}")
            print(f"    ─────────────────────────────────────────")
            print(f"    ⭐ Information Score: {info_score:.4f}")

            all_rows.append({
                "Scenario":          scenario_id,
                "Quality":           quality,
                **scores,
                "Information Score": info_score,
            })

    # ─── Summary table ────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    pivot = df.pivot_table(
        index="Scenario", columns="Quality", values="Information Score"
    )[["good", "convergent", "poor"]]
    pivot["good>convergent"] = pivot["good"] > pivot["convergent"]
    pivot["convergent>poor"] = pivot["convergent"] > pivot["poor"]

    print(f"\n\n{'═'*65}")
    print("📊 VALIDATION SUMMARY — Information Score by Quality Level")
    print(f"{'═'*65}")
    print(pivot.to_string())

    formula_str = " + ".join([f"({v}×{k})" for k, v in IS_WEIGHTS.items()])
    print(f"\nFormula: IS = {formula_str}")

    all_pass = pivot["good>convergent"].all() and pivot["convergent>poor"].all()
    if all_pass:
        print("\n✅ VALIDATION PASSED — Information Score correctly ranks all quality levels!")
    else:
        fails = pivot[~(pivot["good>convergent"] & pivot["convergent>poor"])]
        print(f"\n⚠️  PARTIAL PASS — Failed scenarios:\n{fails}")
        print("   Consider collecting more real data to improve weight calibration.")

    print(f"{'═'*65}")

    # Save CSV
    os.makedirs("doc", exist_ok=True)
    df.to_csv("doc/information_score_test_results.csv", index=False)
    print(f"\n💾 Full results → doc/information_score_test_results.csv")
